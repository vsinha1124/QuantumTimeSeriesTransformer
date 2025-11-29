#!/bin/bash

# FAST Quixer Configuration for Testing
# Optimizations:
# 1. Smaller batch size (8 instead of 32)
# 2. Only 1 encoder layer
# 3. QSVT degree 1 (instead of 2)
# 4. Fewer training epochs

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
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3 \
  --batch_size 8 \
  --use_quantum_attention 1 \
  --quantum_attention_mode full \
  --n_qubits 4 \
  --qsvt_polynomial_degree 1 \
  --n_ansatz_layers 1
