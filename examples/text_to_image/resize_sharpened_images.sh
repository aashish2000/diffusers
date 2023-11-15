# !/usr/bin/env bash

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_fltw_371/visualization/custom/" --save_path "./outputs/seed_371/sharpened/finetuned_lora+text_weighting/checkpoint-3000/" > sfltw_371.out &

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_ltw_371/visualization/custom/" --save_path "./outputs/seed_371/sharpened/lora+text_weighting/checkpoint-3000/" > sltw_371.out &

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_fltw_42/visualization/custom/" --save_path "./outputs/seed_42/sharpened/finetuned_lora+text_weighting/checkpoint-3000/" > sfltw_42.out &

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_ltw_42/visualization/custom/" --save_path "./outputs/seed_42/sharpened/lora+text_weighting/checkpoint-3000/" > sltw_42.out &

# nohup python metrics_files.py --source_path "./outputs/seed_42/sd_base_2_1/" --save_path "outputs/seed_42/resized/sd_base_2_1/" > sd21_42.out &
# nohup python metrics_files.py --source_path "./outputs/seed_371/sd_base_2_1/" --save_path "outputs/seed_371/resized/sd_base_2_1/" > sd21_371.out &

# nohup python metrics_files.py --source_path "./outputs/seed_42/lora+text_weighting_2_1/checkpoint-3000/" --save_path "outputs/seed_42/resized/lora+text_weighting_2_1/checkpoint-3000/" > ltw21_42.out &
# nohup python metrics_files.py --source_path "./outputs/seed_371/lora+text_weighting_2_1/checkpoint-3000/" --save_path "outputs/seed_371/resized/lora+text_weighting_2_1/checkpoint-3000/" > ltw21_371.out &

# nohup python metrics_files.py --source_path "./outputs/seed_42/finetuned_lora+text_weighting_2_1/checkpoint-3000/" --save_path "outputs/seed_42/resized/finetuned_lora+text_weighting_2_1/checkpoint-3000/" > fltw21_42.out &
nohup python metrics_files.py --source_path "./outputs/seed_371/finetuned_lora+text_weighting_2_1/checkpoint-3000/" --save_path "outputs/seed_371/resized/finetuned_lora+text_weighting_2_1/checkpoint-3000/" > fltw21_371.out &

# nohup python metrics_files.py --source_path "./outputs/seed_42/caption_prefix_2_1/" --save_path "outputs/seed_42/resized/caption_prefix_2_1/" > cp21_42.out &
# nohup python metrics_files.py --source_path "./outputs/seed_371/caption_prefix_2_1/" --save_path "outputs/seed_371/resized/caption_prefix_2_1/" > cp21_371.out &