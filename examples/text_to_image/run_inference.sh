# !/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 \
# nohup \
python call_inference_tw.py \
--model_finetuned_path "./models/lora/" \
--generations_path "./outputs/rebuttal/seed_42/tw1+_lora/" \
--checkpoint_name "checkpoint-3000" \
--weight "+" \
--seed 42 \
# > ./seed_42_tw1+lora.out &