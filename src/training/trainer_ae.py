"""
Autoencoder Trainer for scMILD.

VQ-AENB 및 VQ-AENB-Conditional 학습을 담당합니다.
"""

import torch
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
        is_conditional: bool = True
    ):
        self.model = model
        self.device = device
        self.is_conditional = is_conditional

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 5,
        init_codebook: bool = True,
        init_method: str = "kmeans"
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

        Returns:
            history: Dictionary with training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize codebook
        if init_codebook and hasattr(self.model, 'init_codebook'):
            print("Initializing codebook...")
            self.model.init_codebook(train_loader, method=init_method)

        history = {
            'train_loss': [],
            'val_loss': [],
            'commitment_loss': []
        }

        best_loss = float('inf')
        best_state = None
        no_improvement = 0

        for epoch in range(n_epochs):
            # Train
            train_loss, commit_loss = self._train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            history['commitment_loss'].append(commit_loss)

            # Validate
            if val_loader is not None:
                val_loss, _ = self._evaluate(val_loader)
                history['val_loss'].append(val_loss)
                monitor_loss = val_loss
            else:
                monitor_loss = train_loss

            # Early stopping
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
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
                print(msg)

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_commit_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if self.is_conditional:
                # Expect (data, study_ids, labels) or (data, study_ids)
                data = batch[0].to(self.device)
                study_ids = batch[1].to(self.device)

                mu, theta, commit_loss = self.model(data, study_ids, is_train=True)
            else:
                # Expect (data, ...) - only need first element
                data = batch[0].to(self.device)

                output = self.model(data, is_train=True)
                if len(output) == 3:
                    mu, theta, commit_loss = output
                else:
                    mu, theta = output
                    commit_loss = torch.tensor(0.0, device=self.device)

            # Reconstruction loss
            recon_loss = negative_binomial_loss(mu, theta, data)

            # Total loss
            loss = recon_loss + commit_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_commit_loss += commit_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_commit_loss / n_batches

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on dataloader."""
        self.model.eval()

        total_loss = 0.0
        total_commit_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if self.is_conditional:
                data = batch[0].to(self.device)
                study_ids = batch[1].to(self.device)

                mu, theta, commit_loss = self.model(data, study_ids, is_train=True)
            else:
                data = batch[0].to(self.device)

                output = self.model(data, is_train=True)
                if len(output) == 3:
                    mu, theta, commit_loss = output
                else:
                    mu, theta = output
                    commit_loss = torch.tensor(0.0, device=self.device)

            recon_loss = negative_binomial_loss(mu, theta, data)
            loss = recon_loss + commit_loss

            total_loss += loss.item()
            total_commit_loss += commit_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_commit_loss / n_batches

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
