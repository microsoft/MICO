#!/usr/bin/env sh

python train_cifar10.py \
    --len_challenge 100 \
    --len_training 50000 \
    --dataloader_num_workers 4 \
    --dataset_dir /data \
    --batch_size 32 \
    --max_physical_batch_size 128 \
    --num_epochs 50 \
    --learning_rate 0.005 \
    --lr_scheduler_gamma 0.96 \
    --disable_dp
