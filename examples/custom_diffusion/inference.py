import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

model_id = "sayakpaul/custom-diffusion-cat"
card = RepoCard.load(model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to(
"cuda")
pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")

image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")