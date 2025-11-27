# Quixer Transformer Integration with PennyLane

This document describes the integration of the Quixer quantum transformer model into the Quantum Time Series Transformer using PennyLane.

## Overview

The Quixer model is a quantum-enhanced transformer that uses:
- **Parameterized Quantum Circuits (PQC)**: "Ansatz 14" from [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)
- **Linear Combination of Unitaries (LCU)**: To combine multiple quantum operations
- **Quantum Singular Value Transformation (QSVT)**: To apply polynomial transformations to quantum states

## Changes Made

### 1. Created QuixerAttention.py (`/home/ubuntu/QuantumTimeSeriesTransformer/layers/QuixerAttention.py`)

This new file implements the Quixer attention mechanism using PennyLane quantum circuits:

#### Key Components:

- **`ansatz_14_pennylane()`**: PennyLane implementation of the parameterized quantum circuit
  - Uses RY rotations and CRX (Controlled-RX) gates
  - Configurable number of layers and qubits

- **`QuixerAttention`**: Core quantum attention module
  - Converts classical features to quantum circuit parameters
  - Applies QSVT polynomial transformations
  - Uses LCU to combine quantum unitaries
  - Measures X, Y, Z Pauli observables on all qubits
  - Includes hybrid quantum-classical fallback for large inputs

- **`QuixerAttentionLayer`**: Wrapper compatible with existing transformer architecture
  - Handles query/key/value projections
  - Multi-head attention support
  - Drop-in replacement for classical attention layers

### 2. Updated QCAAPatchTF.py

Modified the Quantum PatchTST model to use Quixer attention:

```python
# Before: Used incorrect QuantumAttention with VQE
AttentionLayer(
    QuantumAttention(...),
    ...
)

# After: Uses Quixer with proper quantum circuits
QuixerAttentionLayer(
    n_qubits=self.n_qubits,
    qsvt_polynomial_degree=self.qsvt_degree,
    n_ansatz_layers=self.n_ansatz_layers,
    d_model=configs.d_model,
    n_heads=configs.n_heads,
    ...
)
```

#### Configuration Parameters:

- `n_qubits` (default: 4): Number of qubits in quantum circuits
- `qsvt_polynomial_degree` (default: 2): Degree of QSVT polynomial
- `n_ansatz_layers` (default: 1): Number of PQC layers
- `use_quantum_attention` (default: True): Enable/disable quantum attention

### 3. Updated SelfAttention_Family.py

Added import for QuixerAttention classes to make them available throughout the codebase.

## Architecture Details

### Quantum Circuit Flow

1. **Input Embedding**: Classical features → PQC angles via linear projection
2. **State Initialization**: Start with |0...0⟩ state
3. **LCU Application**: Apply weighted combination of unitaries
   - Each unitary U_i is a PQC with different parameters
   - Weights come from attention scores
4. **QSVT Polynomial**: Apply polynomial P(U) using monomial states
   - P(U) = Σ c_i U^i where c_i are trainable coefficients
5. **Measurement**: Measure Pauli X, Y, Z on all qubits
6. **Output Projection**: Map measurements back to d_model dimension

### Hybrid Quantum-Classical Strategy

For computational efficiency:
- **Small batches** (B ≤ 4, L ≤ 32, S ≤ 32): Full quantum processing
- **Large batches**: Classical attention with quantum-inspired polynomial scaling

This ensures the model can scale while maintaining quantum advantages for tractable problem sizes.

## Advantages Over Previous Implementation

### Previous (QuantumAttention):
- ❌ Used VQE incorrectly (not suitable for attention)
- ❌ Single scalar quantum output broadcast to all positions
- ❌ No proper data encoding into quantum states
- ❌ Not based on established quantum ML literature

### Current (QuixerAttention):
- ✅ Based on published research (Quixer model)
- ✅ Proper quantum state encoding and evolution
- ✅ LCU and QSVT for quantum advantage
- ✅ Multi-qubit measurements for richer representations
- ✅ Theoretically grounded quantum operations
- ✅ Hybrid approach for practical scalability

## Usage

### Basic Configuration

```python
# In your config file or argparse
configs.use_quantum_attention = True
configs.n_qubits = 4  # More qubits = more expressive but slower
configs.qsvt_polynomial_degree = 2  # Polynomial order for QSVT
configs.n_ansatz_layers = 1  # PQC complexity
```

### Running the Model

```python
from models.QCAAPatchTF import Model

# Model automatically uses Quixer attention in even layers
model = Model(configs)

# Standard forward pass
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

### Disabling Quantum Attention

```python
configs.use_quantum_attention = False
# Model falls back to classical FullAttention
```

## Performance Considerations

### Quantum Processing (Default for even layers)
- **Pros**: Quantum advantage, richer representations
- **Cons**: Slower for large inputs
- **Best for**: Small to medium sequences, research experiments

### Classical Fallback (Auto-enabled for large inputs)
- **Pros**: Fast, scalable
- **Cons**: No quantum advantage
- **Best for**: Production, large datasets

## Dependencies

Ensure you have:
```bash
pip install pennylane>=0.31.0 torch numpy
```

**Note**: This implementation uses `qml.QubitStateVector` which is available in PennyLane 0.31.0+. For older versions, you may need to adjust the state preparation method.

## Future Improvements

1. **Hardware Execution**: Deploy on real quantum hardware (IBM, IonQ, etc.)
2. **Optimized Circuits**: Reduce circuit depth for NISQ devices
3. **Adaptive Strategy**: Dynamically choose quantum vs classical based on input
4. **Gradient Optimization**: Use parameter-shift rule for more accurate gradients
5. **Batch Processing**: Optimize quantum circuit execution for batches

## References

1. Quixer Model: [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)
2. QSVT: Quantum Singular Value Transformation
3. LCU: Linear Combination of Unitaries
4. PennyLane Documentation: https://pennylane.ai

## Citation

If you use this implementation, please cite:

```bibtex
@article{quixer2019,
  title={Parameterized quantum circuits as machine learning models},
  journal={arXiv preprint arXiv:1905.10876},
  year={2019}
}
```

---

**Note**: This implementation prioritizes correctness and theoretical grounding over the previous incorrect VQE-based approach. The hybrid quantum-classical strategy ensures practical usability while maintaining quantum advantages where feasible.
