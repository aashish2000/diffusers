from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.load_textual_inversion("./outputs/textual_inversion_brad_pitt")

prompts_list = ["A photo of <brad-pitt> man.", "<brad-pitt> man leaves a conference discussion in September 2005"]

generator = torch.Generator(device="cuda").manual_seed(42)

for ind, prompt in enumerate(prompts_list):
    image = pipe(prompt, generator=[generator], num_inference_steps=100).images[0]
    image.save("brad_pitt" + str(ind) + ".png")