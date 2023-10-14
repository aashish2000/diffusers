from diffusers import StableDiffusionPipeline
import torch
from compel import Compel
import os, shutil
import wordninja

def txt_read_single_line(caption_path):
    with open(caption_path, "r") as f:
        caption_txt = " ".join(f.readlines())
    return(caption_txt)

def add_prompt_weight_characters(caption_txt, key_phrases, weight):
    new_caption = caption_txt

    for phrase in key_phrases:
        if(phrase in new_caption.lower()):
            start_index = new_caption.lower().find(phrase)
            end_index = start_index + len(phrase)
            if("(" in new_caption[start_index:end_index] or ")" in new_caption[start_index:end_index]):
                continue

            new_caption = new_caption[:start_index] + "(" + new_caption[start_index:end_index] + ")" + weight + new_caption[end_index:]
        # print("Caption:", new_caption)
    return(new_caption)

generator = torch.Generator(device="cuda").manual_seed(42)

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda")
compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

DATASET_PATH = "../../../../neurips/datasets/non_entity_datasets/anna_ne_caption_prefixes/objects_list/test/"
SAVE_PATH = "./outputs/objects_list_weighting_3+/test/"

print(DATASET_PATH, SAVE_PATH)

weight = "+++"

for caption_path in sorted(os.listdir(DATASET_PATH)):
    if(not os.path.isfile(SAVE_PATH + caption_path.split(".")[0] + ".jpg")):
        line = txt_read_single_line(DATASET_PATH + caption_path)
        prefix = line.split(".")[0][len("A photo of ") - 1 : ]
        caption_txt = ".".join(line.split(".")[1:]).strip()
        caption_txt.replace("(", "")
        caption_txt.replace(")", "")
        caption_txt.replace("+", " ")

        key_phrases = [phrase.strip().lower() for phrase in prefix.split(",")]

        new_caption = add_prompt_weight_characters(caption_txt, key_phrases, weight)

        if(new_caption == caption_txt):
            for phrases_ind in range(len(key_phrases)):
                phrase_words = key_phrases[phrases_ind].split()
                for word_ind in range(len(phrase_words)):
                    split_words = wordninja.split(phrase_words[word_ind])
                    if(len(split_words) > 1):
                        # print(split_words)
                        phrase_words[word_ind] = "%".join(split_words)
                key_phrases[phrases_ind] = " ".join(phrase_words)
            
            key_phrases_updated = []

            for phrase in key_phrases:
                key_phrases_updated += phrase.split("%")
            # print(key_phrases_updated)
            new_caption = add_prompt_weight_characters(caption_txt, key_phrases_updated, weight)

        # print(caption_txt, prefix, line)
        # print(new_caption)
        # print()

        # upweight 
        conditioning = compel.build_conditioning_tensor(new_caption)
        # or: conditioning = compel([prompt])
        # generate image
        images = pipeline(prompt_embeds=conditioning, num_inference_steps=100, generator=[generator]).images
        images[0].save(SAVE_PATH + caption_path.split(".")[0] + ".jpg")