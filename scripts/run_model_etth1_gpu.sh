#!/bin/bash

# --- GPU Configuration ---
# Enable GPU - comment out to disable GPU
# export CUDA_VISIBLE_DEVICES=-1

# Lấy đường dẫn gốc
model_name=Model

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm
# Lưu ý: Vì chạy CPU nên mình giảm batch_size xuống để đỡ lag máy
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_Model_CPU_Test' \
  --d_model 256 \
  --d_ff 512 \
  --n_fourier_terms 8 \
  --rkan_order 3 \
  --dropout 0.1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_gpu \
  --gpu 0 \
  --train_epochs 100 \
  --itr 1
