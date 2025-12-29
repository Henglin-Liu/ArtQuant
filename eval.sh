dataset=apdd
test_data_path="dataset/APDD/apddv2_test.json"
exp_name="deqa_lora_prompt_${dataset}"
test_data_path_name=${test_data_path##*/}
python src/evaluate/eval.py \
    --level_names excellent good fair poor bad \
    --pred_paths ./results/${exp_name}/${test_data_path_name} \
    --gt_paths ${test_data_path} \
    --score_weight "10. 7.5 5. 3.5 1.0" \
    --dataset ${dataset}