from diffusers import StableDiffusionPipeline
import torch
import wordninja
from compel import Compel, DiffusersTextualInversionManager

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

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.load_textual_inversion("./outputs/textual_inversion_brad_pitt")

prompts_list = ["A photo of <brad-pitt> man.", "A photo of leaving. <brad-pitt> man clear face leaving conference."]

textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, textual_inversion_manager=textual_inversion_manager)

generator = torch.Generator(device="cuda").manual_seed(42)

for ind, prompt in enumerate(prompts_list):
    image = pipe(prompt_embeds=create_weighted_prompt_embeds(compel, prompt, "++"), generator=[generator], num_inference_steps=100).images[0]
    image.save("brad_pitt" + str(ind) + ".png")
