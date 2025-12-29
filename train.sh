#!/bin/bash

# Set dataset and paths
dataset=apdd # Dataset name
image_folder="dataset/APDD/images" # Path to the image folder
data_path1="dataset/score.json" # Path to the score dataset
data_path2="dataset/descriptions_level_3.json" # Path to the descriptions dataset

# Set experiment name and model load path
exp_name="deqa_lora_prompt_${dataset}"
LOAD="ModelZoo/mplug-owl2-llama2-7b/"

# Set local host and master port
local_host=LOCALHOST # e.g. 0,1
master_port=MASTER_PORT # e.g. 6690

deepspeed --include localhost:${local_host} --master_port ${master_port} src/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --lora_enable True \
    --model_name_or_path ${LOAD} \
    --version v1 \
    --dataset_type single \
    --level_prefix "The quality of the painting is" \
    --level_names excellent good fair poor bad \
    --score_weight "10. 7.5 5. 3.5 1.0" \
    --softkl_loss True \
    --weight_rank 1.0 \
    --weight_softkl 1.0 \
    --weight_next_token 0.05 \
    --continuous_rating_loss False \
    --closeset_rating_loss True \
    --use_fix_std True \
    --detach_pred_std True \
    --data_paths ${data_path1} ${data_path2} \
    --data_weights 1 1 \
    --image_folder ${image_folder} \
    --output_dir ./checkpoints/${exp_name} \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard