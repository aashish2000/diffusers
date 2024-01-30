import os
import shutil

GEN_PATHS = ["./outputs/rebuttal/seed_42/lora_tw/tw1+_lora/checkpoint-3000/", 
             "./outputs/rebuttal/seed_42/lora_tw/tw2+_lora/checkpoint-3000/",
             "./outputs/rebuttal/seed_42/lora_tw/tw3+_lora/checkpoint-3000/",
             "./outputs/rebuttal/seed_42/lora_tw/tw4+_lora/checkpoint-3000/",

             "./outputs/rebuttal/seed_371/lora_tw/tw1+_lora/checkpoint-3000/", 
             "./outputs/rebuttal/seed_371/lora_tw/tw2+_lora/checkpoint-3000/",
             "./outputs/rebuttal/seed_371/lora_tw/tw3+_lora/checkpoint-3000/",
             "./outputs/rebuttal/seed_371/lora_tw/tw4+_lora/checkpoint-3000/",]

SRC_PATH = "../../../../neurips/datasets/non_entity_datasets/anna_ne_512/test/"
source_files = [_ for _ in os.listdir(SRC_PATH) if _.endswith(".jpg")]

for gen_path in GEN_PATHS:
    for file_path in os.listdir(gen_path):
        if(file_path not in source_files):
            os.remove(gen_path + file_path)

        
