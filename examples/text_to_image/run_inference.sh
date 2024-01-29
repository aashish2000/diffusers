# !/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_42/tw1+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+" \
--seed 42 \
> ./seed_42_tw1+lora.out &

CUDA_VISIBLE_DEVICES=4 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_42/tw2+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "++" \
--seed 42 \
> ./seed_42_tw2+lora.out &

CUDA_VISIBLE_DEVICES=4 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_42/tw3+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+++" \
--seed 42 \
> ./seed_42_tw3+lora.out &

CUDA_VISIBLE_DEVICES=4 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_42/tw4+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "++++" \
--seed 42 \
> ./seed_42_tw4+lora.out &

CUDA_VISIBLE_DEVICES=3 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_371/tw1+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+" \
--seed 371 \
> ./seed_371_tw1+lora.out &

CUDA_VISIBLE_DEVICES=3 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_371/tw2+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "++" \
--seed 371 \
> ./seed_371_tw2+lora.out &

CUDA_VISIBLE_DEVICES=3 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_371/tw3+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+++" \
--seed 371 \
> ./seed_371_tw3+lora.out &

CUDA_VISIBLE_DEVICES=5 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_371/tw4+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "++++" \
--seed 371 \
> ./seed_371_tw4+lora.out &

CUDA_VISIBLE_DEVICES=2 \
# nohup \
python call_inference_tw.py \
--model_finetuned_path "" \
--generations_path "./outputs/rebuttal/seed_371/gpt4_exp/sd_gpt/" \
--checkpoint_name "" \
--weight "" \
--dataset_path "../../../../neurips/datasets/non_entity_datasets/rebuttal_test/" \
--seed 371 \
--finetue_flag ""
# > ./seed_371_sd_gpt.out &

CUDA_VISIBLE_DEVICES=2 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "" \
--generations_path "./outputs/rebuttal/seed_42/gpt4_exp/sd_gpt/" \
--checkpoint_name "" \
--weight "" \
--dataset_path "../../../../neurips/datasets/non_entity_datasets/rebuttal_test/" \
--seed 42 \
> ./seed_42_sd_gpt.out &