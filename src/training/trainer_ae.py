"""
Autoencoder Trainer for scMILD.

VQ-AENB 및 VQ-AENB-Conditional 학습을 담당합니다.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
import numpy as np


def negative_binomial_loss(
    mu: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Negative Binomial loss for single-cell RNA-seq data.

    Args:
        mu: Predicted mean (batch_size, n_genes)
        theta: Predicted dispersion (batch_size, n_genes)
        y: Target counts (batch_size, n_genes)

    Returns:
        loss: Scalar loss value
    """
    eps = 1e-8

    # Log-likelihood of negative binomial
    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + y * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(y + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(y + 1)
    )

    return -torch.mean(res)


class AETrainer:
    """
    Trainer for VQ-AENB and VQ-AENB-Conditional autoencoders.

    Args:
        model: Autoencoder model
        device: Device for training
        is_conditional: Whether model is conditional (requires study_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        is_conditional: bool = True,
        celltype_classifier: nn.Module = None,
        celltype_loss_weight: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.is_conditional = is_conditional
        self.celltype_classifier = celltype_classifier
        self.celltype_loss_weight = celltype_loss_weight

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 5,
        init_codebook: bool = True,
        init_method: str = "kmeans",
        stratify_codebook: bool = None
    ) -> Dict[str, list]:
        """
        Train the autoencoder.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            n_epochs: Number of epochs
            learning_rate: Learning rate
            patience: Patience for early stopping
            init_codebook: Whether to initialize codebook
            init_method: Codebook initialization method
            stratify_codebook: Whether to use stratified sampling for codebook init.
                               If None, uses stratified sampling for conditional models.

        Returns:
            history: Dictionary with training history
        """
        # Build optimizer with both model and classifier params
        params = list(self.model.parameters())
        if self.celltype_classifier is not None:
            params += list(self.celltype_classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        # Determine stratification strategy
        if stratify_codebook is None:
            stratify_codebook = self.is_conditional

        # Initialize codebook
        if init_codebook and hasattr(self.model, 'init_codebook'):
            stratify_msg = " (stratified)" if stratify_codebook else ""
            print(f"Initializing codebook{stratify_msg}...")
            self.model.init_codebook(train_loader, method=init_method, stratify=stratify_codebook)

        history = {
            'train_loss': [],
            'val_loss': [],
            'commitment_loss': [],
            'celltype_loss': [],
        }

        best_loss = float('inf')
        best_state = None
        best_classifier_state = None
        no_improvement = 0

        for epoch in range(n_epochs):
            # Train
            train_loss, commit_loss, ct_loss = self._train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            history['commitment_loss'].append(commit_loss)
            history['celltype_loss'].append(ct_loss)

            # Validate
            if val_loader is not None:
                val_loss, _, val_ct_loss = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                monitor_loss = val_loss
            else:
                monitor_loss = train_loss

            # Early stopping.
            # `best_state` must only advance on finite monitor_loss —
            # otherwise a NaN epoch (e.g., gradient blowup on the first
            # step) keeps best_state=None and we eventually save whatever
            # NaN parameters happen to be in self.model at training end.
            # Additionally, on a non-finite epoch we fail fast instead of
            # silently letting the model drift into NaN for the rest of
            # training.
            if not math.isfinite(monitor_loss):
                print(
                    f"[AETrainer] FATAL: monitor_loss is non-finite at "
                    f"epoch {epoch} (train={train_loss} "
                    f"commit={commit_loss} ct={ct_loss}). "
                    f"Aborting training loop."
                )
                break
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if self.celltype_classifier is not None:
                    best_classifier_state = {
                        k: v.cpu().clone()
                        for k, v in self.celltype_classifier.state_dict().items()
                    }
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Print progress
            if (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                if self.celltype_classifier is not None:
                    msg += f" - CT Loss: {ct_loss:.4f}"
                print(msg)

        # Load best model.
        # If best_state was never set (every epoch's monitor_loss was
        # non-finite, so nothing ever beat inf) we MUST NOT silently
        # return self.model in its current (likely NaN) state — the
        # caller will save it to disk and poison downstream MIL.
        if best_state is None:
            raise RuntimeError(
                "AETrainer: best_state was never assigned — "
                "every epoch's monitor_loss was non-finite. "
                "The current model parameters are likely NaN. "
                "Check learning rate, input scaling, and EMA stability."
            )
        self.model.load_state_dict(best_state)
        if best_classifier_state is not None and self.celltype_classifier is not None:
            self.celltype_classifier.load_state_dict(best_classifier_state)

        return history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        if self.celltype_classifier is not None:
            self.celltype_classifier.train()

        total_loss = 0.0
        total_commit_loss = 0.0
        total_ct_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            ct_loss = torch.tensor(0.0, device=self.device)

            if self.is_conditional:
                data = batch[0].to(self.device)
                study_ids = batch[1].to(self.device)

                if self.celltype_classifier is not None and len(batch) > 2:
                    # Decomposed forward to get z_q for celltype classifier
                    z = self.model.encoder_forward(data, study_ids)
                    z_q, commit_loss = self.model.quantize(z)
                    mu, theta = self.model.decoder(z_q, study_ids)

                    # Celltype auxiliary loss
                    ct_labels = batch[2].to(self.device)
                    ct_logits = self.celltype_classifier(z_q)
                    valid_mask = (ct_labels >= 0)
                    if valid_mask.any():
                        ct_loss = F.cross_entropy(
                            ct_logits[valid_mask], ct_labels[valid_mask]
                        )
                else:
                    mu, theta, commit_loss = self.model(data, study_ids, is_train=True)
            else:
                data = batch[0].to(self.device)

                output = self.model(data, is_train=True)
                if len(output) == 3:
                    mu, theta, commit_loss = output
                else:
                    mu, theta = output
                    commit_loss = torch.tensor(0.0, device=self.device)

            # Reconstruction loss. When loss_type="mse", the decoder's
            # first tensor IS the reconstruction and `theta` is None.
            # The target is `model.transform_input(data)` so encoder and
            # loss agree on the data space (e.g. log1p).
            if theta is None:
                target = self.model.transform_input(data)
                recon_loss = F.mse_loss(mu, target)
            else:
                recon_loss = negative_binomial_loss(mu, theta, data)

            # Total loss
            loss = recon_loss + commit_loss + self.celltype_loss_weight * ct_loss

            # Skip non-finite batches: stepping on a NaN loss produces
            # NaN grads → NaN params → NaN codebook via EMA → permanent
            # poisoning. We drop the batch and continue; if it keeps
            # recurring, the epoch-level finite check will abort training
            # with a clear error instead of silently saving a NaN model.
            if not torch.isfinite(loss):
                if not getattr(self, "_warned_nan_batch", False):
                    print(
                        f"[AETrainer] WARN: non-finite batch loss "
                        f"(recon={float(recon_loss)} "
                        f"commit={float(commit_loss)} "
                        f"ct={float(ct_loss)}) — skipping step. "
                        f"Subsequent non-finite batches silenced."
                    )
                    self._warned_nan_batch = True
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            ct_norm = None
            if self.celltype_classifier is not None:
                ct_norm = torch.nn.utils.clip_grad_norm_(
                    self.celltype_classifier.parameters(), max_norm=1.0
                )
            # Gradient-level NaN guard. A single NaN grad poisons
            # clip_grad_norm_ (total_norm=NaN → clip_coef=NaN → every
            # grad becomes NaN after scaling) which then NaN-updates
            # every parameter on optimizer.step(). NB/lgamma on edge
            # values produces this even when `loss` itself is finite,
            # so we must check grads post-backward, not just loss.
            if not torch.isfinite(total_norm) or (
                ct_norm is not None and not torch.isfinite(ct_norm)
            ):
                if not getattr(self, "_warned_nan_grad", False):
                    print(
                        f"[AETrainer] WARN: non-finite gradient "
                        f"(model_norm={float(total_norm)} "
                        f"ct_norm={float(ct_norm) if ct_norm is not None else 'n/a'}) "
                        f"— skipping step. Subsequent non-finite grads silenced."
                    )
                    self._warned_nan_grad = True
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

            total_loss += loss.item()
            total_commit_loss += commit_loss.item()
            total_ct_loss += ct_loss.item()
            n_batches += 1

        if n_batches == 0:
            # All batches were non-finite. Return inf so the epoch-level
            # check in train() sees a non-finite monitor_loss and aborts.
            return float("inf"), float("inf"), float("inf")
        return total_loss / n_batches, total_commit_loss / n_batches, total_ct_loss / n_batches

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate on dataloader."""
        self.model.eval()
        if self.celltype_classifier is not None:
            self.celltype_classifier.eval()

        total_loss = 0.0
        total_commit_loss = 0.0
        total_ct_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            ct_loss = torch.tensor(0.0, device=self.device)

            if self.is_conditional:
                data = batch[0].to(self.device)
                study_ids = batch[1].to(self.device)

                if self.celltype_classifier is not None and len(batch) > 2:
                    z = self.model.encoder_forward(data, study_ids)
                    z_q, commit_loss = self.model.quantize(z)
                    mu, theta = self.model.decoder(z_q, study_ids)

                    ct_labels = batch[2].to(self.device)
                    ct_logits = self.celltype_classifier(z_q)
                    valid_mask = (ct_labels >= 0)
                    if valid_mask.any():
                        ct_loss = F.cross_entropy(
                            ct_logits[valid_mask], ct_labels[valid_mask]
                        )
                else:
                    mu, theta, commit_loss = self.model(data, study_ids, is_train=True)
            else:
                data = batch[0].to(self.device)

                output = self.model(data, is_train=True)
                if len(output) == 3:
                    mu, theta, commit_loss = output
                else:
                    mu, theta = output
                    commit_loss = torch.tensor(0.0, device=self.device)

            if theta is None:
                target = self.model.transform_input(data)
                recon_loss = F.mse_loss(mu, target)
            else:
                recon_loss = negative_binomial_loss(mu, theta, data)
            loss = recon_loss + commit_loss + self.celltype_loss_weight * ct_loss

            total_loss += loss.item()
            total_commit_loss += commit_loss.item()
            total_ct_loss += ct_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_commit_loss / n_batches, total_ct_loss / n_batches

    def save(self, path: str, config: Optional[dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            config: Optional config to save with checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': config or {}
        }

        # Add model-specific info
        if hasattr(self.model, 'input_dim'):
            checkpoint['config']['input_dim'] = self.model.input_dim
        if hasattr(self.model, 'latent_dim'):
            checkpoint['config']['latent_dim'] = self.model.latent_dim
        if hasattr(self.model, 'hidden_layers'):
            checkpoint['config']['hidden_layers'] = self.model.hidden_layers
        if hasattr(self.model, 'n_studies'):
            checkpoint['config']['n_studies'] = self.model.n_studies
        if hasattr(self.model, 'study_emb_dim'):
            checkpoint['config']['study_emb_dim'] = self.model.study_emb_dim
        if hasattr(self.model, 'num_codes'):
            checkpoint['config']['num_codes'] = self.model.num_codes

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device,
        model_class=None
    ) -> Tuple['AETrainer', dict]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on
            model_class: Model class to instantiate (if None, returns config only)

        Returns:
            trainer: AETrainer instance (if model_class provided)
            config: Model configuration
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})

        if model_class is not None:
            model = model_class(
                input_dim=config['input_dim'],
                latent_dim=config['latent_dim'],
                device=device,
                hidden_layers=config['hidden_layers'],
                n_studies=config.get('n_studies'),
                study_emb_dim=config.get('study_emb_dim', 16),
                num_codes=config.get('num_codes', 256),
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            is_conditional = 'n_studies' in config
            trainer = cls(model, device, is_conditional=is_conditional)

            return trainer, config

        return None, config

    def get_codebook_usage(self) -> dict:
        """Get codebook usage statistics."""
        if hasattr(self.model, 'get_codebook_usage'):
            return self.model.get_codebook_usage()
        return {}

    @torch.no_grad()
    def get_embeddings(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent embeddings for data.

        Returns:
            embeddings: Latent representations
            indices: Codebook indices (if VQ model)
        """
        self.model.eval()

        all_embeddings = []
        all_indices = []

        for batch in dataloader:
            if self.is_conditional:
                data = batch[0].to(self.device)
                study_ids = batch[1].to(self.device)
                embeddings = self.model.features(data, study_ids)

                if hasattr(self.model, 'get_codebook_indices'):
                    indices = self.model.get_codebook_indices(data, study_ids)
                    all_indices.append(indices.cpu().numpy())
            else:
                data = batch[0].to(self.device)
                embeddings = self.model.features(data)

                if hasattr(self.model, 'get_codebook_indices'):
                    indices = self.model.get_codebook_indices(data)
                    all_indices.append(indices.cpu().numpy())

            all_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        indices = np.concatenate(all_indices, axis=0) if all_indices else None

        return embeddings, indices
