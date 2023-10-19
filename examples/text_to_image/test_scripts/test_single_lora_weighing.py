from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
import torch
from compel import Compel
import os, shutil
from PIL import Image
from tqdm.auto import tqdm
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

class Diffuser:
    def __init__(self, prompts, neg_prompt=[''], guidance=7.5, seed=100, steps=50, width=512, height=512):
        self.prompts = prompts
        self.bs = len(prompts)
        self.neg_prompt = neg_prompt
        self.g = guidance
        self.seed = seed
        self.steps = steps
        self.w = width
        self.h = height

  
    def diffuse(self, progress=0):
        embs = self.set_embs()
        lats = self.set_lats()
        for i, ts in enumerate(tqdm(sched.timesteps)): lats = self.denoise(lats, embs, ts)
        return self.decompress_lats(lats)
    def set_embs(self):
        txt_inp = self.tok_seq(self.prompts)
        neg_inp = self.tok_seq(self.neg_prompt * len(self.prompts))

        txt_embs = self.make_embs(txt_inp['input_ids'])
        neg_embs = self.make_embs(neg_inp['input_ids'])
        print("Neg:", neg_embs)
        return torch.cat([neg_embs, compel(self.prompts)])
  
    def tok_seq(self, prompts, max_len=None):
        if max_len is None: max_len = tokz.model_max_length
        return tokz(prompts, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')    
    
    def make_embs(self, input_ids):
        return txt_enc(input_ids.to('cuda'))[0].half()

    def set_lats(self):
        torch.manual_seed(self.seed)
        lats = torch.randn((self.bs, unet.config.in_channels, self.h//8, self.w//8))
        sched.set_timesteps(self.steps)
        return lats.to('cuda').half() * sched.init_noise_sigma

    def denoise(self, latents, embeddings, timestep):
        inp = sched.scale_model_input(torch.cat([latents]*2), timestep)
        with torch.no_grad(): pred_neg, pred_txt = unet(inp, timestep, encoder_hidden_states=embeddings).sample.chunk(2)
        pred = pred_neg + self.g * (pred_txt - pred_neg)
        return sched.step(pred, timestep, latents).prev_sample

    def decompress_lats(self, latents):
        with torch.no_grad(): imgs = vae.decode(1/0.18215*latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = [img.detach().cpu().permute(1, 2, 0).numpy() for img in imgs]
        return [(img*255).round().astype('uint8') for img in imgs]

    def update_params(self, **kwargs):
        allowed_params = ['prompts', 'neg_prompt', 'guidance', 'seed', 'steps', 'width', 'height']
        for k, v in kwargs.items():
            if k not in allowed_params:
                raise ValueError(f"Invalid parameter name: {k}")
            if k == 'prompts':
                self.prompts = v
                self.bs = len(v)
            else:
                setattr(self, k, v)


vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae", torch_dtype=torch.float16).to('cuda')
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16).to("cuda")
tokz = AutoTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer", torch_dtype=torch.float16)
txt_enc = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="text_encoder", torch_dtype=torch.float16).to('cuda')
sched = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
compel = Compel(tokenizer=tokz, text_encoder=txt_enc)

prompts = [
    'A lightning bolt striking a jumbo jet; 4k; photorealistic',
    'A toaster with (bread)++ in the style of Jony Ive; modern; different; apple; form over function'
]

generator = torch.Generator(device="cuda").manual_seed(42)


diffuser = Diffuser(prompts, seed=42)
imgs = diffuser.diffuse()
Image.fromarray(imgs[1]).save("test.jpg")
