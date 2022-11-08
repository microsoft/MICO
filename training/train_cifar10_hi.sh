#!/usr/bin/env sh

python train_cifar10.py \
    --len_challenge 100 \
    --len_training 50000 \
    --dataloader_num_workers 4 \
    --dataset_dir /data \
    --batch_size 512 \
    --max_physical_batch_size 128 \
    --num_epochs 50 \
    --max_grad_norm 2.6 \
    --target_epsilon 10.0 \
    --target_delta 1e-5 \
    --learning_rate 0.5 \
    --lr_scheduler_gamma 0.96 \
    --secure_mode

