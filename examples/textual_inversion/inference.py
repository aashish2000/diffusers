from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda")
pipe.load_textual_inversion("./outputs/textual_inversion_brad_pitt")

prompt = "A photo of <brad-pitt> man."

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(prompt, generator=[generator], num_inference_steps=100).images[0]
image.save("brad_pitt.png")