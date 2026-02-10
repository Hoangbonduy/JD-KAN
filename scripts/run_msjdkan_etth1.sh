#!/bin/bash

# --- THAY ĐỔI 1: Tắt GPU ---
# Đặt giá trị rỗng hoặc -1 để PyTorch không nhìn thấy GPU nào
export CUDA_VISIBLE_DEVICES=-1

# Lấy đường dẫn gốc
model_name=MS_JDKAN

# Tạo thư mục logs nếu chưa có
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Chạy thử nghiệm MS_JDKAN
# Anti-overfit: giảm d_model, tăng dropout, thêm weight_decay, gradient clipping, cosine LR
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
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_MS_JDKAN_CPU_Test' \
  --d_model 32 \
  --d_ff 64 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --dropout 0.3 \
  --lradj cosine \
  --no_use_gpu \
  --train_epochs 100 \
  --patience 5 \
  --itr 1
