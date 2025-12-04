# Quantum Time Series Transformer (QTST)

Advanced Patch Embedding Transformer with Quixer Quantum Attention for enhanced time series forecasting.

## Overview

This repository implements a quantum-enhanced time series forecasting model that combines patch-based transformers with Quixer quantum attention mechanisms. The model provides three different quantum attention implementations (Options A, B, and C) to balance between computational efficiency and quantum advantage.

## Quixer Quantum Attention Options

### **Option A: Subsampled PennyLane Implementation**
- **Tokens Processed**: 32 tokens (subsampled from 96)
- **Backend**: PennyLane with `lightning.qubit` device
- **Quantum Gates**: Full quantum circuit implementation with native PennyLane gates
- **Performance**: Moderate speed (~4-6 hours per epoch)
- **Use Case**: When you want real quantum gates but need faster training

**Key Features**:
- Subsamples input sequence from 96 to 32 tokens to reduce computation
- Uses PennyLane's differentiable quantum gates (RY, CRX)
- Applies QSVT (Quantum Singular Value Transformation) + LCU (Linear Combination of Unitaries)
- CPU-only (PennyLane GPU support requires additional backends like `lightning.gpu` or adjoint differentiation)

### **Option B: Full Sequence PennyLane Implementation** ⚠️
- **Tokens Processed**: All 96 tokens
- **Backend**: PennyLane with `lightning.qubit` device
- **Quantum Gates**: Full quantum circuit implementation with native PennyLane gates
- **Performance**: **Extremely slow** (~16-17 hours per epoch)
- **Use Case**: Research/verification purposes only

**Key Features**:
- Processes the full 96-token sequence without subsampling
- Maximum quantum information retention
- Uses PennyLane's quantum gates with full backpropagation
- **Warning**: Very slow due to CPU-only quantum simulations on full sequences

### **Option C: TorchQuantum Fast Implementation** ✅ (Recommended)
- **Tokens Processed**: All 96 tokens
- **Backend**: TorchQuantum (pure PyTorch simulation)
- **Quantum Gates**: Matrix-based simulation (no explicit gate construction)
- **Performance**: **Fast** (~2 minutes per epoch on GPU)
- **Use Case**: simulation of Quixer mathematically, easier to scale to higher qubits and layers

**Key Features**:
- Processes all 96 tokens using mathematically equivalent matrix operations
- No explicit quantum gate construction - uses linear algebra simulation
- Fully GPU-accelerated with native PyTorch operations
- Cross-attention fusion mechanism for enhanced global context integration
- **10-30x faster** than PennyLane options while maintaining quantum-inspired behavior

## Quantum Global Context Fusion

All Quixer options extract a **global quantum context vector** from the input sequence:

1. **Quantum Processing**: Each token's embedding is mapped to quantum circuit parameters
2. **LCU Aggregation**: Linear Combination of Unitaries weighted by learnable coefficients combines all token states
3. **QSVT Enhancement**: Quantum Singular Value Transformation applies polynomial transformations to the aggregated state
4. **Measurement**: Expectation values are measured along X, Y, Z axes to extract classical features

The resulting global context is then **fused back into the token representations**:

- **Option A & B**: Simple broadcast addition to all token positions
- **Option C**: Cross-attention fusion where the quantum global context acts as a query attending to all tokens, enabling position-aware and selective enhancement

This fusion mechanism allows the model to:
- Capture long-range dependencies through quantum entanglement
- Enhance each token with globally-aware quantum features
- Improve time series forecasting by incorporating quantum-derived patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/vsinha1124/QuantumTimeSeriesTransformer.git
cd QuantumTimeSeriesTransformer

# Create conda environment with Python 3.11
conda create -n qtst python=3.11 -y
conda activate qtst

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements.txt

# For Option A & B (PennyLane) - if not in requirements.txt
pip install pennylane pennylane-lightning

# For Option C (TorchQuantum) - if not in requirements.txt
pip install torchquantum
```
## Usage Examples

### Option A: Subsampled PennyLane (Moderate Speed but lower context, worst prediction results)

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model QCAAPatchTF \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --c_out 21 \
  --train_epochs 10 \
  --batch_size 32 \
  --use_quantum_attention 1 \
  --quantum_attention_mode full \
  --quixer_option A \
  --n_qubits 4 \
  --qsvt_polynomial_degree 2 \
  --n_ansatz_layers 1
```

### Option B: Full PennyLane (Very Slow - Research Only)

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model QCAAPatchTF \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --c_out 21 \
  --train_epochs 10 \
  --batch_size 32 \
  --use_quantum_attention 1 \
  --quantum_attention_mode full \
  --quixer_option B \
  --n_qubits 4 \
  --qsvt_polynomial_degree 2 \
  --n_ansatz_layers 2
```

### Option C: TorchQuantum Fast (Recommended) ✅

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model QCAAPatchTF \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 21 \
  --c_out 21 \
  --train_epochs 10 \
  --batch_size 32 \
  --use_quantum_attention 1 \
  --quantum_attention_mode full \
  --quixer_option C \
  --n_qubits 6 \
  --qsvt_polynomial_degree 2 \
  --n_ansatz_layers 2
```

## Key Arguments

| Argument | Description | Options | Default |
|----------|-------------|---------|---------|
| `--quixer_option` | Quixer implementation to use | `A`, `B`, `C` | `C` |
| `--use_quantum_attention` | Enable quantum attention | `0`, `1` | `1` |
| `--quantum_attention_mode` | Quantum layer configuration | `full`, `alternating`, `classical` | `alternating` |
| `--n_qubits` | Number of qubits in quantum circuit | `3`-`8` | `4` |
| `--qsvt_polynomial_degree` | QSVT polynomial degree | `1`-`5` | `2` |
| `--n_ansatz_layers` | Number of PQC ansatz layers | `1`-`3` | `1` |

### Quantum Attention Modes

- **`full`**: All encoder layers use quantum attention
- **`alternating`**: Even layers use quantum, odd layers use classical
- **`classical`**: All layers use classical attention (baseline)

## Performance Comparison

| Option | Tokens | Backend | Speed (per epoch) | GPU Support | Recommended |
|--------|--------|---------|-------------------|-------------|-------------|
| **A** | 32 | PennyLane | ~4-6 hours | ❌ CPU only | Research |
| **B** | 96 | PennyLane | ~16-17 hours | ❌ CPU only | Not recommended |
| **C** | 96 | TorchQuantum | ~30-40 mins | ✅ Full GPU | **Yes** ✅ |

## Model Architecture

```
Input Sequence [B, L, D]
    ↓
Patch Embedding
    ↓
Quantum Encoder Layers
    ├─ Quixer Quantum Attention (Option A/B/C)
    │   ├─ Token → Quantum Parameters
    │   ├─ LCU (Linear Combination of Unitaries)
    │   ├─ QSVT (Quantum Singular Value Transformation)
    │   ├─ Measurement (X, Y, Z expectation values)
    │   └─ Fusion (CrossAttention or Broadcast)
    └─ Classical Feed-Forward
    ↓
Prediction Head
    ↓
Output Forecast [B, P, D]
```
