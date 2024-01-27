# !/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_42/tw1+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+" \
--seed 42 \
> ./seed_42_tw1+lora.out &

# CUDA_VISIBLE_DEVICES=7 \
# nohup \
# python call_inference_tw.py \
# --model_finetuned_path "./models/lora/" \
# --generations_path "./outputs/rebuttal/seed_42/tw2+_lora/" \
# --checkpoint_name "checkpoint-3000" \
# --weight "++" \
# --seed 42 \
# > ./seed_42_tw2+lora.out &

# CUDA_VISIBLE_DEVICES=1 \
# nohup \
# python call_inference_tw.py \
# --model_finetuned_path "./models/lora/" \
# --generations_path "./outputs/rebuttal/seed_42/tw3+_lora/" \
# --checkpoint_name "checkpoint-3000" \
# --weight "+++" \
# --seed 42 \
# > ./seed_42_tw3+lora.out &

# CUDA_VISIBLE_DEVICES=1 \
# nohup \
# python call_inference_tw.py \
# --model_finetuned_path "./models/lora/" \
# --generations_path "./outputs/rebuttal/seed_42/tw4+_lora/" \
# --checkpoint_name "checkpoint-3000" \
# --weight "++++" \
# --seed 42 \
# > ./seed_42_tw4+lora.out &

# CUDA_VISIBLE_DEVICES=0 \
# nohup \
# python call_inference_tw.py \
# --model_finetuned_path "./models/lora/" \
# --generations_path "./outputs/rebuttal/seed_371/tw1+_lora/" \
# --checkpoint_name "checkpoint-3000" \
# --weight "+" \
# --seed 371 \
# > ./seed_371_tw1+lora.out &

CUDA_VISIBLE_DEVICES=7 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_371/tw2+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "++" \
--seed 371 \
> ./seed_371_tw2+lora.out &

CUDA_VISIBLE_DEVICES=7 \
nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_371/tw3+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+++" \
--seed 371 \
> ./seed_371_tw3+lora.out &

# CUDA_VISIBLE_DEVICES=4 \
# nohup \
# python call_inference_tw.py \
# --model_finetuned_path "./models/lora/" \
# --generations_path "./outputs/rebuttal/seed_371/tw4+_lora/" \
# --checkpoint_name "checkpoint-3000" \
# --weight "++++" \
# --seed 371 \
# > ./seed_371_tw4+lora.out &