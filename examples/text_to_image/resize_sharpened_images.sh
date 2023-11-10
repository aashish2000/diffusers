# !/usr/bin/env bash

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_fltw_371/visualization/custom/" --save_path "./outputs/seed_371/sharpened/finetuned_lora+text_weighting/" > sfltw_371.out &

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_ltw_371/visualization/custom/" --save_path "./outputs/seed_371/sharpened/lora+text_weighting/" > sltw_371.out &

# nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_fltw_42/visualization/custom/" --save_path "./outputs/seed_42/sharpened/finetuned_lora+text_weighting/" > sfltw_42.out &

nohup python metrics_files.py --source_path "../../../../neurips/methods/HAT/results/HAT_GAN_Real_Sharper_ltw_42/visualization/custom/" --save_path "./outputs/seed_42/sharpened/lora+text_weighting/" > sltw_42.out &