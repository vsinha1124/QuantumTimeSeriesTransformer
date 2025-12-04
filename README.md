
# Quantum Time Series Transformer

This repository contains multiple quantum-enhanced approaches for time series forecasting, built on top of the PatchTST architecture. Each approach explores different quantum computing methodologies and is maintained in a separate branch for independent development and comparison.

## Project Structure

This project is organized into **three main research branches**, each implementing distinct quantum strategies for time series prediction:

---

### 🔷 [Quixer Branch](../../tree/Quixer)
**Quantum Context Attention with Quixer QLA Architecture**

The Quixer branch implements quantum attention mechanisms using the Quantum Linear Algebra methods and architecture inspired by Quixer with three different implementation options for flexibility and performance optimization. 

---

### 🔶 [Benchmark Branch](../../tree/benchmark)
**Classical Baselines and Performance Comparisons for Quixer QLA Architecture**

The benchmark branch provides classical baseline implementations and tools for comparing Quixer QLA quantum approaches against traditional methods on Quantum hardware and T4 GPU.

---

### 🔷 [PQC Branch](../../tree/PQC)
**Parameterized Quantum Circuits Approaches Using Quantum Mapping/Projection and Quantum Pairwise Attention**

The PQC branch implements multiple hybrid quantum–classical attention mechanisms built using parameterized quantum circuits. These include quantum projection–based attention and quantum kernel attention (using fidelity and Hadamard-test estimators).


##  Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/vsinha1124/QuantumTimeSeriesTransformer.git
cd QuantumTimeSeriesTransformer
```

### Step 2: Choose Your Branch

Navigate to the branch corresponding to the approach you want to explore:

```bash
# For Quixer quantum attention approach
git checkout Quixer

# For classical benchmarks
git checkout benchmark

# For PQC approach 
git checkout PQC
```

## Citations

This work builds upon and extends the following researches:

### Quantum PatchTST
```bibtex
@article{quantumpatchTST2025,
  title={Quantum Context-Aware Attention for Time Series Forecasting},
  author={Sanjay Chakraborty, Fredrik Heintz},
  journal={arXiv preprint arXiv:2504.00068},
  year={2025},
  url={https://arxiv.org/abs/2504.00068}
}
```
**Repository**: [https://github.com/sanjaylopa22/QCAAPatchTF](https://github.com/sanjaylopa22/QCAAPatchTF)

### Quixer
```bibtex
@article{Quixer2024,
  title={Quixer: A Quantum Transformer with Improved Expressivity},
  author={Nikhil Khatri, Gabriel Matos, Luuk Coopmans, Stephen Clark},
  journal={arXiv preprint arXiv:2406.04305},
  year={2024},
  url={https://arxiv.org/abs/2406.04305}
}
```
**Repository**: [https://github.com/CQCL/Quixer](https://github.com/CQCL/Quixer)

---
