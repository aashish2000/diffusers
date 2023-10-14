from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import os

'''
TODOS: Figure out how to use full finetune checkpoints for inference
TODOS: Calculate metrics for non-entity lora models
'''

def generate_stable_diffusion_images(checkpoint_name, flag_full_finetune):
    device = "cuda"
    CAPTIONS_PATH = "./outputs/metrics_test/source/"
    GENERATIONS_PATH = "./outputs/metrics_test/orig/"
    model_orig_path = "runwayml/stable-diffusion-v1-5"

    model_finetuned_path = None

    generator = torch.Generator(device="cuda").manual_seed(42)
    latents = None
    width = 512
    height = 512

    pipe_gens = StableDiffusionPipeline.from_pretrained(model_orig_path,
                                                        torch_dtype=torch.float16, safety_checker=None).to(device)

    pipe_gens.to("cuda")

    caption_files = [x for x in os.listdir(CAPTIONS_PATH) if x.endswith(".txt")]

    # print(caption_files)

    # os.mkdir(GENERATIONS_PATH + checkpoint_name + "/finetuned")

    # print("Len:", len(caption_files))

    for file in caption_files:
        img = Image.open(GENERATIONS_PATH + file.split(".")[-2] + ".jpg")
        if(not img.getbbox()):
            with open(CAPTIONS_PATH + file, 'r') as f:
                caption = f.read().replace('\n', '')

            print(file)
            
            # image_orig = pipe_orig(prompt=caption).images[0]
            # image_orig.save(GENERATIONS_PATH + checkpoint_name + "orig/" + file.split(".")[-2] + ".jpg")

            image_finetuned = pipe_gens(prompt=caption, guidance_scale=9, generator=[generator]).images[0]
            image_finetuned.save(GENERATIONS_PATH + file.split(".")[-2] + ".jpg")

# generate_stable_diffusion_images(checkpoint_name="checkpoint-11500", flag_full_finetune="no") #1155280
# generate_stable_diffusion_images(checkpoint_name="checkpoint-10000", flag_full_finetune="no") #1153630
# generate_stable_diffusion_images(checkpoint_name="checkpoint-8000", flag_full_finetune="no") #1152582
# generate_stable_diffusion_images(checkpoint_name="checkpoint-6000", flag_full_finetune="no") #1151403
generate_stable_diffusion_images(checkpoint_name="", flag_full_finetune="scp") #1150362
# generate_stable_diffusion_images(checkpoint_name="", flag_full_finetune="na") #1209472
# generate_stable_diffusion_images(checkpoint_name="checkpoint-14250", flag_full_finetune="no")
# generate_stable_diffusion_images(checkpoint_name="checkpoint-8000", flag_full_finetune="no")
# generate_stable_diffusion_images(checkpoint_name="checkpoint-10000", flag_full_finetune="no")