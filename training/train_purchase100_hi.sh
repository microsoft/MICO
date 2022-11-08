#!/usr/bin/env sh

python train_purchase100.py \
    --len_challenge 100 \
    --len_training 150000 \
    --dataloader_num_workers 4 \
    --dataset_dir /data \
    --batch_size 512 \
    --max_physical_batch_size 128 \
    --num_epochs 30 \
    --learning_rate 0.001 \
    --lr_scheduler_gamma 0.9 \
    --lr_scheduler_step 5 \
    --max_grad_norm 2.6 \
    --target_epsilon 10.0 \
    --target_delta 1e-5 \
    --secure_mode
