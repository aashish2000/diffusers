from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import os
from compel import Compel
import os, shutil
import wordninja

'''
TODOS: Figure out how to use full finetune checkpoints for inference
TODOS: Calculate metrics for non-entity lora models
'''

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


def create_weighted_prompt_embeds(compel, line, weight):
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
                    phrase_words[word_ind] = "%".join(split_words)
            key_phrases[phrases_ind] = " ".join(phrase_words)

        key_phrases_updated = []

        for phrase in key_phrases:
            key_phrases_updated += phrase.split("%")

        new_caption = add_prompt_weight_characters(caption_txt, key_phrases_updated, weight)

    conditioning = compel.build_conditioning_tensor(new_caption)
    return(conditioning)


def clean_caption_prefix(line):
    prefix = line.split(".")[0][len("A photo of ") - 1 : ]
    caption_txt = ".".join(line.split(".")[1:]).strip()
    key_phrases = [phrase.strip() for phrase in prefix.split(",")]
    # print(key_phrases)

    processed_phrases = []

    for phrase in key_phrases:
        final_phrase = ""
        for word in phrase.split(" "):
            # print(word)
            split_words = wordninja.split(word)
            # if(len(split_words) > 1):
                # print(split_words)
            final_phrase += ", ".join(split_words) + " "
        processed_phrases.append(final_phrase)
    
    processed_prefix = "A photo of " + ", ".join(processed_phrases)
    processed_prefix = processed_prefix[:-1] + "."
    if(caption_txt[0] != " "):
        processed_prefix += " "

    processed_caption = processed_prefix + caption_txt
    
    print(processed_caption, line)
    return(processed_caption)


def generate_lora_stable_diffusion_images(checkpoint_name, flag_full_finetune, model_finetuned_path, generations_path, seed):
    device = "cuda"
    
    # CAPTIONS_PATH = "../../../../neurips/datasets/non_entity_datasets/anna_ne_caption_prefixes/objects_list/test/"
    # generations_path = "./outputs/text_weighting+sharpened/test/"
    model_orig_path = "runwayml/stable-diffusion-v1-5"

    # model_orig_path = "CompVis/stable-diffusion-v1-4"
    

    generator = torch.Generator(device="cuda").manual_seed(seed)


    pipe_gens = StableDiffusionPipeline.from_pretrained(model_orig_path,
                                                        torch_dtype=torch.float16, safety_checker=None)
    if(checkpoint_name != ""):
        # model_finetuned_path = "../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/"
        pipe_gens = StableDiffusionPipeline.from_pretrained(model_orig_path,
                                                                torch_dtype=torch.float16, safety_checker=None)
        pipe_gens.unet.load_attn_procs(model_finetuned_path, 
                                            subfolder=checkpoint_name, 
                                            weight_name="pytorch_model.bin")
    
    if(flag_full_finetune == "tw" or flag_full_finetune == "px"):
        compel = Compel(tokenizer=pipe_gens.tokenizer, text_encoder=pipe_gens.text_encoder)
        weight = "++"
        # CAPTIONS_PATH = "../../../../neurips/datasets/non_entity_datasets/anna_ne_caption_prefixes/objects_list/test/"
        CAPTIONS_PATH = "../../../../neurips/datasets/entity_datasets/dev_versions/anna_e_caption_prefixes/"
    else:
        CAPTIONS_PATH = "../../../../neurips/datasets/entity_datasets/dev_versions/anna_e_captions/"
    
    print(CAPTIONS_PATH)
    
    pipe_gens.to("cuda")


    caption_files = [x for x in os.listdir(CAPTIONS_PATH) if x.endswith(".txt")]

    if(not os.path.isdir(generations_path + checkpoint_name)):
        os.mkdir(generations_path + checkpoint_name)
    SAVE_PREFIX = generations_path + checkpoint_name + "/"
    

    print("Len:", len(caption_files))

    for file in caption_files:
        if(not os.path.isfile(SAVE_PREFIX + file.split(".")[-2] + ".jpg")):
            with open(CAPTIONS_PATH + file, 'r') as f:
                caption = f.read().replace('\n', '')
                if(flag_full_finetune == "scp"):
                    caption = "A photo of " + caption

            print(file)
            if(flag_full_finetune == "px"):
                caption = clean_caption_prefix(caption)
            
            if(flag_full_finetune == "tw"):
                image_finetuned = pipe_gens(prompt_embeds=create_weighted_prompt_embeds(compel, caption, weight), 
                                            generator=[generator], 
                                            num_inference_steps=100).images[0]
            else:
                image_finetuned = pipe_gens(prompt=caption, generator=[generator], num_inference_steps=100).images[0]
            
            image_finetuned.save(SAVE_PREFIX + "/" + file.split(".")[-2] + ".jpg")



# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="", 
#                                       model_finetuned_path="./models/lora_sharpened/",
#                                       generations_path="./outputs/seed_371/lora+sharpened/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="./models/finetuned_lora+text_weighting/",
#                                       generations_path="./outputs/seed_371/finetuned_lora+text_weighting/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="./models/finetuned_lora+text_weighting+sharpened/",
#                                       generations_path="./outputs/seed_371/finetuned_lora+text_weighting+sharpened/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/",
#                                       generations_path="./outputs/seed_371/lora+text_weighting/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="./models/lora_sharpened/",
#                                       generations_path="./outputs/seed_371/lora+text_weighting+sharpened/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="", 
#                                       flag_full_finetune="", 
#                                       model_finetuned_path="",
#                                       generations_path="./outputs/seed_371/sd_base/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="",
#                                       generations_path="./outputs/seed_371/text_weighting/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="", 
#                                       model_finetuned_path="../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/",
#                                       generations_path="./outputs/seed_371/lora/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="", 
#                                       flag_full_finetune="px", 
#                                       model_finetuned_path="",
#                                       generations_path="./outputs/seed_371/caption_prefix/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="px", 
#                                       model_finetuned_path="../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/",
#                                       generations_path="./outputs/lora+caption_prefix/",
#                                       seed=42) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="px", 
#                                       model_finetuned_path="../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/",
#                                       generations_path="./outputs/lora+caption_prefix/",
#                                       seed=42) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-5000", 
#                                       flag_full_finetune="", 
#                                       model_finetuned_path="../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/",
#                                       generations_path="./outputs/lora/",
#                                       seed=42) 

# generate_lora_stable_diffusion_images(checkpoint_name="", 
#                                       flag_full_finetune="px", 
#                                       model_finetuned_path="",
#                                       generations_path="./outputs/caption_prefix/",
#                                       seed=42) 

# generate_lora_stable_diffusion_images(checkpoint_name="", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="",
#                                       generations_path="./outputs/text_weighting/",
#                                       seed=42) 

# generate_lora_stable_diffusion_images(checkpoint_name="", 
#                                       flag_full_finetune="", 
#                                       model_finetuned_path="",
#                                       generations_path="./outputs/seed_371/anna_entity/",
#                                       seed=371) 

# generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-3000", 
#                                       flag_full_finetune="tw", 
#                                       model_finetuned_path="../../../../neurips/methods/diffusers/examples/text_to_image/models/lora/",
#                                       generations_path="./outputs/seed_371/entity_lora+text_weighting/",
#                                       seed=371) 

generate_lora_stable_diffusion_images(checkpoint_name="checkpoint-3000", 
                                      flag_full_finetune="tw", 
                                      model_finetuned_path="./models/finetuned_lora+text_weighting/",
                                      generations_path="./outputs/entity_finetuned_lora+text_weighting/",
                                      seed=42) 

