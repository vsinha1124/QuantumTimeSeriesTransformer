"""
Demo script for Quixer Quantum Attention in Time Series Transformer
This script demonstrates how to use the Quixer-enhanced PatchTST model
"""

import torch
import numpy as np
from argparse import Namespace

# Import the model
import sys
sys.path.append('/home/ubuntu/QuantumTimeSeriesTransformer')
from models.QCAAPatchTF import Model


def create_demo_config():
    """Create a demo configuration for the model"""
    config = Namespace(
        # Task configuration
        task_name='long_term_forecast',
        seq_len=96,  # Input sequence length
        pred_len=96,  # Prediction length
        
        # Model architecture
        d_model=64,  # Reduced for demo
        n_heads=4,
        e_layers=2,  # 2 encoder layers
        d_ff=256,
        dropout=0.1,
        activation='gelu',
        
        # Data configuration
        enc_in=7,  # Number of input features
        
        # Attention configuration
        factor=5,
        output_attention=False,
        
        # Quixer Quantum Attention parameters
        use_quantum_attention=True,
        n_qubits=4,  # 4 qubits
        qsvt_polynomial_degree=2,  # Quadratic QSVT polynomial
        n_ansatz_layers=1,  # Single layer PQC
    )
    return config


def demo_quixer_model():
    """Demonstrate the Quixer model"""
    print("=" * 80)
    print("Quixer Quantum Transformer Demo")
    print("=" * 80)
    
    # Create configuration
    config = create_demo_config()
    print("\nConfiguration:")
    print(f"  Task: {config.task_name}")
    print(f"  Sequence Length: {config.seq_len}")
    print(f"  Prediction Length: {config.pred_len}")
    print(f"  Model Dimension: {config.d_model}")
    print(f"  Number of Heads: {config.n_heads}")
    print(f"  Encoder Layers: {config.e_layers}")
    print(f"  Quantum Attention: {config.use_quantum_attention}")
    if config.use_quantum_attention:
        print(f"  Number of Qubits: {config.n_qubits}")
        print(f"  QSVT Degree: {config.qsvt_polynomial_degree}")
        print(f"  PQC Layers: {config.n_ansatz_layers}")
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating model...")
    print("-" * 80)
    model = Model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dummy data
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # Time features
    x_dec = torch.randn(batch_size, config.pred_len, config.enc_in)
    x_mark_dec = torch.randn(batch_size, config.pred_len, 4)
    
    print("\n" + "-" * 80)
    print("Input shapes:")
    print("-" * 80)
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_mark_enc: {x_mark_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  x_mark_dec: {x_mark_dec.shape}")
    
    # Forward pass
    print("\n" + "-" * 80)
    print("Running forward pass...")
    print("-" * 80)
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {config.pred_len}, {config.enc_in}]")
    
    # Verify output
    assert output.shape == (batch_size, config.pred_len, config.enc_in), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {config.pred_len}, {config.enc_in})"
    
    print("\n✓ Forward pass successful!")
    
    # Test backward pass
    print("\n" + "-" * 80)
    print("Testing backward pass...")
    print("-" * 80)
    model.train()
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    loss = output.mean()
    loss.backward()
    
    # Check if gradients are computed
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed!"
    
    print("✓ Backward pass successful!")
    
    # Compare with classical attention
    print("\n" + "-" * 80)
    print("Comparing with classical attention...")
    print("-" * 80)
    config_classical = create_demo_config()
    config_classical.use_quantum_attention = False
    model_classical = Model(config_classical)
    
    classical_params = sum(p.numel() for p in model_classical.parameters())
    quantum_params = sum(p.numel() for p in model.parameters())
    
    print(f"Classical model parameters: {classical_params:,}")
    print(f"Quantum model parameters: {quantum_params:,}")
    print(f"Difference: {quantum_params - classical_params:,}")
    
    model_classical.eval()
    with torch.no_grad():
        output_classical = model_classical(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\nClassical output shape: {output_classical.shape}")
    
    # Compare outputs (they should be different)
    model.eval()
    with torch.no_grad():
        output_quantum = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    diff = torch.abs(output_quantum - output_classical).mean().item()
    print(f"Mean absolute difference: {diff:.6f}")
    
    if diff > 1e-6:
        print("✓ Quantum and classical outputs are different (as expected)")
    else:
        print("⚠ Outputs are very similar (might indicate fallback to classical)")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demo_quixer_attention_only():
    """Demonstrate just the QuixerAttentionLayer"""
    print("\n" + "=" * 80)
    print("QuixerAttentionLayer Standalone Demo")
    print("=" * 80)
    
    from layers.QuixerAttention import QuixerAttentionLayer
    
    # Configuration
    batch_size = 2
    seq_len = 16
    d_model = 32
    n_heads = 4
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {n_heads}")
    
    # Create attention layer
    attention = QuixerAttentionLayer(
        n_qubits=3,
        qsvt_polynomial_degree=2,
        n_ansatz_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        mask_flag=True,
        attention_dropout=0.1,
        output_attention=False,
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    print("\nRunning attention forward pass...")
    attention.eval()
    with torch.no_grad():
        output, attn_weights = attention(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {x.shape}")
    
    assert output.shape == x.shape, f"Shape mismatch! Got {output.shape}, expected {x.shape}"
    
    print("✓ QuixerAttentionLayer works correctly!")


if __name__ == "__main__":
    # Run demos
    try:
        demo_quixer_model()
        demo_quixer_attention_only()
        
        print("\n" + "=" * 80)
        print("All demos passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
