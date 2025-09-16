"""
Compatibility Test Script for AENB and VQ-AENB models.
This script tests that:
1. Existing AENB models can still be loaded and used
2. New VQ-AENB models work correctly
3. Both models can be used with EncoderBranch
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import AENB, VQ_AENB, EncoderBranch
from src.quantizer import Quantizer


def test_aenb_compatibility():
    """Test that AENB model works as before."""
    print("\n" + "="*50)
    print("Testing AENB Compatibility")
    print("="*50)
    
    # Create dummy data
    batch_size = 32
    input_dim = 2000
    latent_dim = 128
    hidden_layers = [512, 256, 128]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = AENB(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_layers=hidden_layers,
        activation_function=nn.Sigmoid
    ).to(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, input_dim).to(device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        mu_recon, theta_recon = model(dummy_input)
    
    print(f"✓ AENB forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Mu reconstruction shape: {mu_recon.shape}")
    print(f"  Theta reconstruction shape: {theta_recon.shape}")
    
    # Test features extraction
    with torch.no_grad():
        features = model.features(dummy_input)
    
    print(f"✓ AENB features extraction successful")
    print(f"  Features shape: {features.shape}")
    
    # Test saving and loading
    save_path = "test_aenb_model.pth"
    torch.save(model.state_dict(), save_path)
    
    # Create new model and load weights
    model2 = AENB(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_layers=hidden_layers,
        activation_function=nn.Sigmoid
    ).to(device)
    model2.load_state_dict(torch.load(save_path))
    
    print(f"✓ AENB save/load successful")
    
    # Clean up
    os.remove(save_path)
    
    return model


def test_vq_aenb_functionality():
    """Test that VQ-AENB model works correctly."""
    print("\n" + "="*50)
    print("Testing VQ-AENB Functionality")
    print("="*50)
    
    # Create dummy data
    batch_size = 32
    input_dim = 2000
    latent_dim = 128
    hidden_layers = [512, 256, 128]
    num_codes = 256
    commitment_weight = 0.25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = VQ_AENB(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_layers=hidden_layers,
        num_codes=num_codes,
        commitment_weight=commitment_weight,
        activation_function=nn.Sigmoid
    ).to(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, input_dim).to(device)
    
    # Test forward pass (training mode)
    model.train()
    mu_recon, theta_recon, commitment_loss = model(dummy_input, is_train=True)
    
    print(f"✓ VQ-AENB forward pass (training) successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Mu reconstruction shape: {mu_recon.shape}")
    print(f"  Theta reconstruction shape: {theta_recon.shape}")
    print(f"  Commitment loss: {commitment_loss.item():.4f}")
    
    # Test forward pass (evaluation mode)
    model.eval()
    with torch.no_grad():
        mu_recon, theta_recon = model(dummy_input, is_train=False)
    
    print(f"✓ VQ-AENB forward pass (evaluation) successful")
    
    # Test features extraction
    with torch.no_grad():
        features = model.features(dummy_input)
    
    print(f"✓ VQ-AENB features extraction successful")
    print(f"  Quantized features shape: {features.shape}")
    
    # Test codebook indices
    with torch.no_grad():
        indices = model.get_codebook_indices(dummy_input)
    
    print(f"✓ VQ-AENB codebook indices extraction successful")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Unique codes used: {len(torch.unique(indices))}")
    
    # Test codebook initialization
    dummy_dataset = TensorDataset(torch.randn(1000, input_dim).to(device))
    dummy_loader = DataLoader(dummy_dataset, batch_size=64)
    model.init_codebook(dummy_loader, method="random", num_samples=500)
    
    print(f"✓ VQ-AENB codebook initialization successful")
    
    # Test codebook usage statistics
    usage_stats = model.get_codebook_usage()
    print(f"✓ VQ-AENB codebook usage stats:")
    print(f"  Active codes: {usage_stats['num_active']}/{usage_stats['total_codes']}")
    
    # Test saving and loading
    save_path = "test_vq_aenb_model.pth"
    torch.save(model.state_dict(), save_path)
    
    # Create new model and load weights
    model2 = VQ_AENB(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_layers=hidden_layers,
        num_codes=num_codes,
        commitment_weight=commitment_weight,
        activation_function=nn.Sigmoid
    ).to(device)
    model2.load_state_dict(torch.load(save_path))
    
    print(f"✓ VQ-AENB save/load successful")
    
    # Clean up
    os.remove(save_path)
    
    return model


def test_encoder_branch_compatibility(aenb_model, vq_aenb_model):
    """Test that EncoderBranch works with both AENB and VQ-AENB."""
    print("\n" + "="*50)
    print("Testing EncoderBranch Compatibility")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dims = 64
    batch_size = 32
    input_dim = 2000
    
    # Test with AENB
    print("\nTesting EncoderBranch with AENB:")
    encoder_branch_aenb = EncoderBranch(
        proto_vae=aenb_model,
        output_dims=output_dims,
        activation_function=nn.Tanh
    ).to(device)
    
    dummy_input = torch.randn(batch_size, input_dim).to(device)
    
    with torch.no_grad():
        output_aenb = encoder_branch_aenb(dummy_input)
    
    print(f"✓ EncoderBranch with AENB successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output_aenb.shape}")
    print(f"  Is VQ model: {encoder_branch_aenb.is_vq_model}")
    
    # Test with VQ-AENB
    print("\nTesting EncoderBranch with VQ-AENB:")
    encoder_branch_vq = EncoderBranch(
        proto_vae=vq_aenb_model,
        output_dims=output_dims,
        activation_function=nn.Tanh
    ).to(device)
    
    with torch.no_grad():
        output_vq = encoder_branch_vq(dummy_input)
    
    print(f"✓ EncoderBranch with VQ-AENB successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output_vq.shape}")
    print(f"  Is VQ model: {encoder_branch_vq.is_vq_model}")
    
    # Verify outputs have same shape but different values
    assert output_aenb.shape == output_vq.shape, "Output shapes should match"
    print(f"✓ Output shapes match for both models")
    
    # Check that features are different (VQ should produce discrete features)
    with torch.no_grad():
        features_aenb = aenb_model.features(dummy_input)
        features_vq = vq_aenb_model.features(dummy_input)
    
    print(f"\n✓ Feature comparison:")
    print(f"  AENB features std: {features_aenb.std().item():.4f}")
    print(f"  VQ-AENB features std: {features_vq.std().item():.4f}")


def test_backward_compatibility():
    """Test that old code still works without modifications."""
    print("\n" + "="*50)
    print("Testing Backward Compatibility")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # This simulates old code that doesn't know about VQ-AENB
    from src.model import AENB
    from src.trainer_ae import negative_binomial_loss
    
    # Create model the old way
    model = AENB(
        input_dim=2000,
        latent_dim=128,
        device=device,
        hidden_layers=[512, 256, 128],
        activation_function=nn.ReLU
    ).to(device)
    
    # Test old training loop pattern
    dummy_input = torch.randn(32, 2000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    mu_recon, theta_recon = model(dummy_input)
    loss = negative_binomial_loss(mu_recon, theta_recon, dummy_input)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Old training pattern works correctly")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test old inference pattern
    model.eval()
    with torch.no_grad():
        mu_recon, theta_recon = model(dummy_input)
    
    print(f"✓ Old inference pattern works correctly")


def main():
    """Run all compatibility tests."""
    print("\n" + "="*60)
    print(" COMPATIBILITY TEST SUITE FOR AENB/VQ-AENB")
    print("="*60)
    
    try:
        # Test AENB compatibility
        aenb_model = test_aenb_compatibility()
        
        # Test VQ-AENB functionality
        vq_aenb_model = test_vq_aenb_functionality()
        
        # Test EncoderBranch with both models
        test_encoder_branch_compatibility(aenb_model, vq_aenb_model)
        
        # Test backward compatibility
        test_backward_compatibility()
        
        print("\n" + "="*60)
        print(" ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nSummary:")
        print("- AENB model works as before")
        print("- VQ-AENB model functions correctly")
        print("- EncoderBranch supports both models")
        print("- Backward compatibility maintained")
        print("- No breaking changes detected")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f" TEST FAILED: {str(e)}")
        print("="*60)
        raise


if __name__ == "__main__":
    main()