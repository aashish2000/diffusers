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

compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)
# freeze parameters of models to save more memory
unet.requires_grad_(False)
vae.requires_grad_(False)

text_encoder.requires_grad_(False)