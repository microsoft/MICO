#!/usr/bin/env sh

# Does not support DDP. Might need to set e.g. CUDA_VISIBLE_DEVICES to 
# force single-GPU training 

python train_sst2.py \
    --model_name roberta-base \
    --dataloader_num_workers 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 3.0 \
    --per_sample_max_grad_norm 0.1 \
    --target_epsilon 4.0 \
    --delta 1e-5 \
    --learning_rate 0.0005 \
    --lr_scheduler_type constant \
    --use_secure_prng True \
    --model_index 0 \
    --save_strategy no \
    --evaluation_strategy epoch \
    --logging_steps 10 \
    --seed 42 \
    --output_dir output
