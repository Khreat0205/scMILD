"""
Performance Comparison Script for AENB vs VQ-AENB models.
This script compares:
1. Training time and convergence
2. Reconstruction quality
3. Memory usage
4. Downstream task performance (if applicable)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import AENB, VQ_AENB
from src.trainer_ae import negative_binomial_loss


class PerformanceComparison:
    """Class to compare performance between AENB and VQ-AENB models."""
    
    def __init__(self, input_dim=2000, latent_dim=128, hidden_layers=[512, 256, 128],
                 num_codes=256, commitment_weight=0.25, device=None):
        """Initialize comparison with model configurations."""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.num_codes = num_codes
        self.commitment_weight = commitment_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.results = {
            'AENB': {},
            'VQ-AENB': {}
        }
    
    def create_synthetic_dataset(self, n_samples=10000, train_ratio=0.8):
        """Create synthetic single-cell-like data for testing."""
        print("\nCreating synthetic dataset...")
        
        # Generate synthetic gene expression data with sparsity
        data = torch.randn(n_samples, self.input_dim).abs()
        # Add sparsity (typical for scRNA-seq)
        mask = torch.rand(n_samples, self.input_dim) > 0.7
        data[mask] = 0
        
        # Split into train and test
        n_train = int(n_samples * train_ratio)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        # Create datasets
        train_dataset = TensorDataset(train_data)
        test_dataset = TensorDataset(test_data)
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Data sparsity: {(data == 0).float().mean():.2%}")
        
        return train_dataset, test_dataset
    
    def train_model(self, model, train_loader, test_loader, model_type, epochs=10, lr=1e-3):
        """Train a model and track metrics."""
        print(f"\nTraining {model_type} model...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        test_losses = []
        training_times = []
        
        # Initialize VQ-AENB codebook if needed
        if model_type == 'VQ-AENB':
            print("  Initializing codebook...")
            model.init_codebook(train_loader, method="random", num_samples=1000)
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_start = time.time()
            train_loss = 0
            train_commitment_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                data = batch[0].to(self.device)
                optimizer.zero_grad()
                
                if model_type == 'VQ-AENB':
                    mu_recon, theta_recon, commitment_loss = model(data, is_train=True)
                    recon_loss = negative_binomial_loss(mu_recon, theta_recon, data)
                    total_loss = recon_loss + commitment_loss
                    train_commitment_loss += commitment_loss.item()
                else:
                    mu_recon, theta_recon = model(data)
                    recon_loss = negative_binomial_loss(mu_recon, theta_recon, data)
                    total_loss = recon_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += recon_loss.item()
                n_batches += 1
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            
            avg_train_loss = train_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Evaluation
            model.eval()
            test_loss = 0
            n_test_batches = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    data = batch[0].to(self.device)
                    
                    if model_type == 'VQ-AENB':
                        mu_recon, theta_recon = model(data, is_train=False)
                    else:
                        mu_recon, theta_recon = model(data)
                    
                    test_loss += negative_binomial_loss(mu_recon, theta_recon, data).item()
                    n_test_batches += 1
            
            avg_test_loss = test_loss / n_test_batches
            test_losses.append(avg_test_loss)
            
            # Print progress
            if epoch % 2 == 0:
                if model_type == 'VQ-AENB':
                    avg_commitment = train_commitment_loss / n_batches
                    print(f"  Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, "
                          f"Commitment={avg_commitment:.4f}, Test Loss={avg_test_loss:.4f}, "
                          f"Time={epoch_time:.2f}s")
                else:
                    print(f"  Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, "
                          f"Test Loss={avg_test_loss:.4f}, Time={epoch_time:.2f}s")
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'training_times': training_times,
            'total_time': sum(training_times)
        }
    
    def evaluate_reconstruction_quality(self, model, test_loader, model_type):
        """Evaluate reconstruction quality metrics."""
        print(f"\nEvaluating {model_type} reconstruction quality...")
        
        model.eval()
        all_originals = []
        all_reconstructions = []
        all_indices = [] if model_type == 'VQ-AENB' else None
        
        with torch.no_grad():
            for batch in test_loader:
                data = batch[0].to(self.device)
                
                if model_type == 'VQ-AENB':
                    mu_recon, _ = model(data, is_train=False)
                    indices = model.get_codebook_indices(data)
                    all_indices.append(indices.cpu())
                else:
                    mu_recon, _ = model(data)
                
                all_originals.append(data.cpu())
                all_reconstructions.append(mu_recon.cpu())
        
        # Concatenate all batches
        originals = torch.cat(all_originals, dim=0)
        reconstructions = torch.cat(all_reconstructions, dim=0)
        
        # Calculate metrics
        mse = torch.mean((originals - reconstructions) ** 2).item()
        mae = torch.mean(torch.abs(originals - reconstructions)).item()
        
        # Correlation coefficient
        orig_flat = originals.flatten()
        recon_flat = reconstructions.flatten()
        correlation = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0, 1].item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
        
        # Additional metrics for VQ-AENB
        if model_type == 'VQ-AENB':
            all_indices = torch.cat(all_indices, dim=0)
            unique_codes = len(torch.unique(all_indices))
            codebook_usage = model.get_codebook_usage()
            
            metrics['unique_codes_used'] = unique_codes
            metrics['active_codes'] = codebook_usage['num_active']
            metrics['total_codes'] = codebook_usage['total_codes']
            metrics['codebook_utilization'] = unique_codes / codebook_usage['total_codes']
            
            print(f"  Codebook utilization: {unique_codes}/{codebook_usage['total_codes']} "
                  f"({metrics['codebook_utilization']:.2%})")
        
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        
        return metrics
    
    def evaluate_feature_quality(self, model, test_loader, model_type):
        """Evaluate the quality of learned features."""
        print(f"\nEvaluating {model_type} feature quality...")
        
        model.eval()
        all_features = []
        
        with torch.no_grad():
            for batch in test_loader:
                data = batch[0].to(self.device)
                features = model.features(data)
                all_features.append(features.cpu())
        
        features = torch.cat(all_features, dim=0)
        
        # Calculate feature statistics
        feature_mean = features.mean().item()
        feature_std = features.std().item()
        
        # Calculate feature diversity (average pairwise distance)
        n_samples = min(1000, features.shape[0])  # Sample for efficiency
        sample_indices = torch.randperm(features.shape[0])[:n_samples]
        sample_features = features[sample_indices]
        
        # Pairwise distances
        distances = torch.cdist(sample_features, sample_features, p=2)
        # Exclude diagonal (distance to self)
        mask = ~torch.eye(n_samples, dtype=bool)
        avg_distance = distances[mask].mean().item()
        
        metrics = {
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'avg_pairwise_distance': avg_distance,
            'feature_dim': features.shape[1]
        }
        
        print(f"  Feature mean: {feature_mean:.4f}")
        print(f"  Feature std: {feature_std:.4f}")
        print(f"  Avg pairwise distance: {avg_distance:.4f}")
        
        return metrics
    
    def measure_memory_usage(self, model, model_type):
        """Measure model memory usage."""
        print(f"\nMeasuring {model_type} memory usage...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory in MB
        param_memory = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory
        }
        
        # Additional memory for VQ-AENB (codebook)
        if model_type == 'VQ-AENB':
            codebook_params = model.num_codes * model.latent_dim
            codebook_memory = codebook_params * 4 / (1024 * 1024)
            metrics['codebook_parameters'] = codebook_params
            metrics['codebook_memory_mb'] = codebook_memory
            metrics['total_memory_mb'] = param_memory + codebook_memory
        else:
            metrics['total_memory_mb'] = param_memory
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total memory: {metrics['total_memory_mb']:.2f} MB")
        
        return metrics
    
    def run_comparison(self, n_samples=5000, batch_size=128, epochs=10):
        """Run full comparison between AENB and VQ-AENB."""
        print("\n" + "="*60)
        print(" PERFORMANCE COMPARISON: AENB vs VQ-AENB")
        print("="*60)
        
        # Create dataset
        train_dataset, test_dataset = self.create_synthetic_dataset(n_samples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Compare both models
        for model_type in ['AENB', 'VQ-AENB']:
            print(f"\n{'='*40}")
            print(f" Evaluating {model_type}")
            print('='*40)
            
            # Create model
            if model_type == 'AENB':
                model = AENB(
                    input_dim=self.input_dim,
                    latent_dim=self.latent_dim,
                    device=self.device,
                    hidden_layers=self.hidden_layers,
                    activation_function=nn.Sigmoid
                ).to(self.device)
            else:
                model = VQ_AENB(
                    input_dim=self.input_dim,
                    latent_dim=self.latent_dim,
                    device=self.device,
                    hidden_layers=self.hidden_layers,
                    num_codes=self.num_codes,
                    commitment_weight=self.commitment_weight,
                    activation_function=nn.Sigmoid
                ).to(self.device)
            
            # Train model
            training_metrics = self.train_model(model, train_loader, test_loader, 
                                               model_type, epochs=epochs)
            self.results[model_type]['training'] = training_metrics
            
            # Evaluate reconstruction
            recon_metrics = self.evaluate_reconstruction_quality(model, test_loader, model_type)
            self.results[model_type]['reconstruction'] = recon_metrics
            
            # Evaluate features
            feature_metrics = self.evaluate_feature_quality(model, test_loader, model_type)
            self.results[model_type]['features'] = feature_metrics
            
            # Measure memory
            memory_metrics = self.measure_memory_usage(model, model_type)
            self.results[model_type]['memory'] = memory_metrics
        
        return self.results
    
    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "="*60)
        print(" COMPARISON SUMMARY")
        print("="*60)
        
        # Training efficiency
        print("\n1. TRAINING EFFICIENCY:")
        print("-" * 30)
        for model_type in ['AENB', 'VQ-AENB']:
            training = self.results[model_type]['training']
            print(f"{model_type}:")
            print(f"  Total training time: {training['total_time']:.2f}s")
            print(f"  Final train loss: {training['train_losses'][-1]:.4f}")
            print(f"  Final test loss: {training['test_losses'][-1]:.4f}")
        
        # Reconstruction quality
        print("\n2. RECONSTRUCTION QUALITY:")
        print("-" * 30)
        for model_type in ['AENB', 'VQ-AENB']:
            recon = self.results[model_type]['reconstruction']
            print(f"{model_type}:")
            print(f"  MSE: {recon['mse']:.6f}")
            print(f"  MAE: {recon['mae']:.6f}")
            print(f"  Correlation: {recon['correlation']:.4f}")
            if model_type == 'VQ-AENB':
                print(f"  Codebook utilization: {recon['codebook_utilization']:.2%}")
        
        # Feature quality
        print("\n3. FEATURE QUALITY:")
        print("-" * 30)
        for model_type in ['AENB', 'VQ-AENB']:
            features = self.results[model_type]['features']
            print(f"{model_type}:")
            print(f"  Feature diversity: {features['avg_pairwise_distance']:.4f}")
            print(f"  Feature std: {features['feature_std']:.4f}")
        
        # Memory usage
        print("\n4. MEMORY USAGE:")
        print("-" * 30)
        for model_type in ['AENB', 'VQ-AENB']:
            memory = self.results[model_type]['memory']
            print(f"{model_type}:")
            print(f"  Total parameters: {memory['total_parameters']:,}")
            print(f"  Total memory: {memory['total_memory_mb']:.2f} MB")
        
        # Relative comparison
        print("\n5. RELATIVE COMPARISON (VQ-AENB vs AENB):")
        print("-" * 30)
        
        # Training time ratio
        time_ratio = (self.results['VQ-AENB']['training']['total_time'] / 
                     self.results['AENB']['training']['total_time'])
        print(f"Training time ratio: {time_ratio:.2f}x")
        
        # Loss improvement
        loss_diff = (self.results['VQ-AENB']['training']['test_losses'][-1] - 
                    self.results['AENB']['training']['test_losses'][-1])
        print(f"Test loss difference: {loss_diff:+.4f}")
        
        # Memory overhead
        memory_ratio = (self.results['VQ-AENB']['memory']['total_memory_mb'] / 
                       self.results['AENB']['memory']['total_memory_mb'])
        print(f"Memory usage ratio: {memory_ratio:.2f}x")
        
        # Feature diversity ratio
        diversity_ratio = (self.results['VQ-AENB']['features']['avg_pairwise_distance'] / 
                          self.results['AENB']['features']['avg_pairwise_distance'])
        print(f"Feature diversity ratio: {diversity_ratio:.2f}x")
    
    def save_results(self, filename='comparison_results.json'):
        """Save comparison results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


def main():
    """Run the performance comparison."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize comparison
    comparison = PerformanceComparison(
        input_dim=2000,
        latent_dim=128,
        hidden_layers=[512, 256, 128],
        num_codes=256,
        commitment_weight=0.25
    )
    
    # Run comparison
    results = comparison.run_comparison(
        n_samples=5000,
        batch_size=128,
        epochs=10
    )
    
    # Print summary
    comparison.print_summary()
    
    # Save results
    comparison.save_results()
    
    print("\n" + "="*60)
    print(" COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()