#!/usr/bin/env python
"""
Quick test to verify quantum attention modes work correctly
"""

import sys
sys.path.append('/home/ubuntu/QuantumTimeSeriesTransformer')

from argparse import Namespace
from models.QCAAPatchTF import Model

def test_mode(mode_name, use_quantum, quantum_mode):
    """Test a specific quantum attention mode"""
    print(f"\n{'='*60}")
    print(f"Testing: {mode_name}")
    print('='*60)
    
    config = Namespace(
        task_name='long_term_forecast',
        seq_len=96,
        pred_len=96,
        d_model=64,
        n_heads=4,
        e_layers=4,  # 4 layers to see the pattern
        d_ff=256,
        dropout=0.1,
        activation='gelu',
        enc_in=7,
        factor=5,
        output_attention=False,
        use_quantum_attention=use_quantum,
        quantum_attention_mode=quantum_mode,
        n_qubits=4,
        qsvt_polynomial_degree=2,
        n_ansatz_layers=1,
    )
    
    model = Model(config)
    
    # Count quantum vs classical layers
    quantum_layers = 0
    classical_layers = 0
    
    for i, layer in enumerate(model.encoder.attn_layers):
        layer_type = type(layer.attention).__name__
        if 'Quixer' in layer_type:
            quantum_layers += 1
            print(f"  Layer {i}: Quixer (Quantum)")
        else:
            classical_layers += 1
            print(f"  Layer {i}: Classical")
    
    print(f"\nSummary: {quantum_layers} Quantum, {classical_layers} Classical")
    return quantum_layers, classical_layers

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Quixer Quantum Attention Mode Test")
    print("="*60)
    
    # Test all three modes
    q1, c1 = test_mode("Full Quantum", 1, "full")
    q2, c2 = test_mode("Alternating (Even=Quantum, Odd=Classical)", 1, "alternating")
    q3, c3 = test_mode("Full Classical", 1, "classical")
    
    # Verify
    print("\n" + "="*60)
    print("Verification")
    print("="*60)
    assert q1 == 4 and c1 == 0, "Full quantum should have 4 quantum layers"
    assert q2 == 2 and c2 == 2, "Alternating should have 2 of each"
    assert q3 == 0 and c3 == 4, "Full classical should have 4 classical layers"
    
    print("✅ All modes working correctly!")
    print("\nUsage in shell script:")
    print("  --quantum_attention_mode full        # All layers quantum")
    print("  --quantum_attention_mode alternating # Half and half (default)")
    print("  --quantum_attention_mode classical   # All layers classical")
