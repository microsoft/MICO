#!/usr/bin/env sh

# Does not support DDP. Might need to set e.g. CUDA_VISIBLE_DEVICES to 
# force single-GPU training 

python train_sst2.py \
    --model_name roberta-base \
    --lr_scheduler_type constant \
    --learning_rate 5e-5 \
    --use_secure_prng True \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_strategy no \
    --evaluation_strategy epoch \
    --dataloader_num_workers 2 \
    --per_device_train_batch_size 96 \
    --gradient_accumulation_steps 1 \
    --output_dir output_inf \
    --model_index 0 \
    --seed 42 \
    --disable_dp True
