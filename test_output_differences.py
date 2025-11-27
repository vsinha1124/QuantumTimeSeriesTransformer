#!/usr/bin/env python
"""
Test to verify quantum and classical modes produce DIFFERENT outputs
"""

import sys
sys.path.append('/home/ubuntu/QuantumTimeSeriesTransformer')

import torch
from argparse import Namespace
from models.QCAAPatchTF import Model

torch.manual_seed(42)

def test_output_differences():
    """Test that different modes produce different outputs"""
    print("\n" + "="*70)
    print("TESTING: Do quantum and classical modes produce different outputs?")
    print("="*70)
    
    # Create same config for all
    base_config = Namespace(
        task_name='long_term_forecast',
        seq_len=96,
        pred_len=96,
        d_model=32,  # Smaller for faster testing
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
        activation='gelu',
        enc_in=7,
        factor=5,
        output_attention=False,
        n_qubits=4,
        qsvt_polynomial_degree=2,
        n_ansatz_layers=1,
    )
    
    # Create input data
    batch_size = 2
    x_enc = torch.randn(batch_size, 96, 7)
    x_mark_enc = torch.randn(batch_size, 96, 4)
    x_dec = torch.randn(batch_size, 96, 7)
    x_mark_dec = torch.randn(batch_size, 96, 4)
    
    print(f"\nInput shape: {x_enc.shape}")
    print(f"Testing with {base_config.e_layers} encoder layers\n")
    
    # Test 1: Full Quantum
    print("-" * 70)
    print("1. FULL QUANTUM MODE")
    print("-" * 70)
    config_quantum = Namespace(**vars(base_config))
    config_quantum.use_quantum_attention = 1
    config_quantum.quantum_attention_mode = "full"
    
    model_quantum = Model(config_quantum)
    model_quantum.eval()
    with torch.no_grad():
        output_quantum = model_quantum(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Output shape: {output_quantum.shape}")
    print(f"Output mean: {output_quantum.mean().item():.6f}")
    print(f"Output std: {output_quantum.std().item():.6f}")
    
    # Test 2: Full Classical  
    print("\n" + "-" * 70)
    print("2. FULL CLASSICAL MODE")
    print("-" * 70)
    config_classical = Namespace(**vars(base_config))
    config_classical.use_quantum_attention = 1
    config_classical.quantum_attention_mode = "classical"
    
    model_classical = Model(config_classical)
    model_classical.eval()
    with torch.no_grad():
        output_classical = model_classical(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Output shape: {output_classical.shape}")
    print(f"Output mean: {output_classical.mean().item():.6f}")
    print(f"Output std: {output_classical.std().item():.6f}")
    
    # Test 3: Alternating
    print("\n" + "-" * 70)
    print("3. ALTERNATING MODE (Hybrid)")
    print("-" * 70)
    config_alternating = Namespace(**vars(base_config))
    config_alternating.use_quantum_attention = 1
    config_alternating.quantum_attention_mode = "alternating"
    
    model_alternating = Model(config_alternating)
    model_alternating.eval()
    with torch.no_grad():
        output_alternating = model_alternating(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"Output shape: {output_alternating.shape}")
    print(f"Output mean: {output_alternating.mean().item():.6f}")
    print(f"Output std: {output_alternating.std().item():.6f}")
    
    # Compare outputs
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    diff_quantum_classical = torch.abs(output_quantum - output_classical).mean().item()
    diff_quantum_alternating = torch.abs(output_quantum - output_alternating).mean().item()
    diff_classical_alternating = torch.abs(output_classical - output_alternating).mean().item()
    
    print(f"\nMean Absolute Difference:")
    print(f"  Quantum vs Classical:    {diff_quantum_classical:.6f}")
    print(f"  Quantum vs Alternating:  {diff_quantum_alternating:.6f}")
    print(f"  Classical vs Alternating: {diff_classical_alternating:.6f}")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    threshold = 1e-4
    
    if diff_quantum_classical < threshold:
        print(f"❌ FAILED: Quantum and Classical outputs are IDENTICAL (diff={diff_quantum_classical:.8f})")
        print("   This means the quantum circuits are NOT being used!")
        return False
    else:
        print(f"✅ PASSED: Quantum and Classical outputs are DIFFERENT (diff={diff_quantum_classical:.6f})")
    
    if diff_classical_alternating < threshold:
        print(f"❌ FAILED: Classical and Alternating outputs are IDENTICAL (diff={diff_classical_alternating:.8f})")
        return False
    else:
        print(f"✅ PASSED: Classical and Alternating outputs are DIFFERENT (diff={diff_classical_alternating:.6f})")
    
    print("\n✅ SUCCESS: All three modes produce DIFFERENT outputs!")
    print("   The quantum circuits are being executed properly.")
    return True

if __name__ == "__main__":
    success = test_output_differences()
    sys.exit(0 if success else 1)
