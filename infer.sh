#!/bin/bash

test_data_path="dataset/APDD/apddv2_test.json"
image_folder="dataset/APDD/images"
exp_name="deqa_lora_prompt_${dataset}"
LOAD="ModelZoo/mplug-owl2-llama2-7b/"

test_cuda=${1}

python src/stage2_evaluate/iqa_eval.py \
    --level-names excellent good fair poor bad \
    --model-path ./checkpoints/${exp_name} \
    --model-base ${LOAD} \
    --save-dir results/${exp_name} \
    --preprocessor-path ./preprocessor/ \
    --root-dir ${image_folder} \
    --meta-paths ${test_data_path} \
    --device cuda:${test_cuda}