# ⚛️ Quantum Time-Series Transformer  
*A modular forecasting framework with pluggable Quantum Attention mechanisms*

This repository extends a classical Transformer-based long-term time series forecasting model with **multiple quantum-enhanced attention mechanisms**, implemented using **PennyLane**.  
All architectures can be toggled at runtime using command-line flags.

---

# 📦 Installation

```bash
pip install -r requirements.txt
```

Install PennyLane:

```bash
pip install pennylane pennylane-lightning
```

(Optional GPU backend):

```bash
pip install pennylane-lightning[gpu]
```

---

# 🚀 Running a Forecasting Experiment

You can run experiments **directly via command line** *or* using the provided **bash script**.

---

## ✅ Option 1 — Run Directly (Python)

Example command (Weather dataset):

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model QCAAPatchTF \
  --channel_independence 0 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des Exp \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 32 \
  --num_workers 8 \
  --quantum_attention True \
  --pqc_type quantum_kernel_hadamard \
  --n_qubits 8
```

---

## ✅ Option 2 — Run Using the Bash Script (Recommended)

You can instead run:

```bash
bash ./scripts/long_term_forecast/Weather_script/QCAAPatchTF.sh
```

Then edit the script to adjust:

- dataset paths  
- prediction horizon  
- number of encoder layers  
- PQC type  
- qubit count  
- quantum vs classical attention  

### Script path:
```
scripts/long_term_forecast/Weather_script/QCAAPatchTF.sh
```

### Example content of the script (editable):

```bash
model_name=QCAAPatchTF

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --channel_independence 0 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des Exp \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 32 \
  --num_workers 8 \
  --quantum_attention True \
  --pqc_type quantum_kernel_hadamard \
  --n_qubits 8
```

The script also contains **commented-out templates** for longer prediction windows:

- `pred_len = 192`
- `pred_len = 336`
- `pred_len = 720`

These can be uncommented as needed.

---

# ⚙️ Configuration Flags

### **Enable classical or quantum attention**
```bash
--quantum_attention True|False
```
- `True` → All encoder layers use the quantum module  
- `False` → All encoder layers use classical FullAttention  

---

### **Choose quantum attention type**
```bash
--pqc_type quantum_mapping
--pqc_type quantum_mapping_hybrid
--pqc_type quantum_kernel_fidelity
--pqc_type quantum_kernel_hadamard
```

Mappings:

| Flag | Attention Module |
|------|------------------|
| `quantum_mapping` | `QuantumProjectionAttention` |
| `quantum_mapping_hybrid` | `QuantumProjectionAttentionHybrid` |
| `quantum_kernel_fidelity` | `QuantumKernelAttentionFidelity` |
| `quantum_kernel_hadamard` | `QuantumKernelAttentionHadamard` |

---

### **Number of qubits**
```bash
--n_qubits 4
```

---

# 🧠 Available Attention Mechanisms

### **Classical Full Attention**  
`--quantum_attention False`

---

### **QuantumProjectionAttention**  
`--pqc_type quantum_mapping`

Uses VQCs to replace Q/K projections.

---

### **QuantumProjectionAttentionHybrid**  
`--pqc_type quantum_mapping_hybrid`

Concatenates classical & quantum features.

---

### **QuantumKernelAttentionFidelity**  
`--pqc_type quantum_kernel_fidelity`

Uses fidelity between quantum states as attention.

---

### **QuantumKernelAttentionHadamard**  
`--pqc_type quantum_kernel_hadamard`

Uses a Hadamard Test with ancilla qubits.

---

# 📊 Summary

| Mode | Command | Description |
|------|---------|-------------|
| Classical | `--quantum_attention False` | Standard attention |
| Quantum Mapping | `quantum_mapping` | VQC projection |
| Quantum Hybrid | `quantum_mapping_hybrid` | Classical + quantum concat |
| Fidelity Kernel | `quantum_kernel_fidelity` | Fast quantum kernel |
| Hadamard Kernel | `quantum_kernel_hadamard` | Most expressive quantum test |

