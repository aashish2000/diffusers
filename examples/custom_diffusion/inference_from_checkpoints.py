import itertools
import torch

from accelerate import Accelerator
from transformers import AutoTokenizer, PretrainedConfig
from huggingface_hub import snapshot_download
from huggingface_hub.repocard import RepoCard

from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor as attention_class

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    
    else:
        raise ValueError(f"{model_class} is not supported.")

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "sayakpaul/custom-diffusion-cat"
card = RepoCard.load(model_id)
repo_save_path = snapshot_download(repo_id=model_id)
base_model_id = card.data.to_dict()["base_model"]
base_model_id = "./models/"
args_revision = None
args_freeze_model = "crossattn"
weight_dtype = torch.float16

accelerator = Accelerator()

pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=weight_dtype).to(accelerator.device)

# Use text_encoder if `--train_text_encoder` was used for the initial training
unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", revision=args_revision)

text_encoder_cls = import_model_class_from_model_name_or_path(base_model_id, args_revision)
text_encoder = text_encoder_cls.from_pretrained(base_model_id, subfolder="text_encoder", revision=args_revision)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    subfolder="tokenizer",
    revision=args_revision,
    use_fast=False,
)

# Input trained modifier token
args_modifier_token = ["<new1>"]
args_initializer_token = ["ktn", "pll", "ucd"]

# modifier_token_id = []
# initializer_token_id = []
# if len(args_modifier_token) > len(args_initializer_token):
#     raise ValueError("You must specify + separated initializer token for each modifier token.")

# for modifier_token, initializer_token in zip(args_modifier_token, args_initializer_token[:len(args_modifier_token)]):
#     # Add the placeholder token in tokenizer
#     num_added_tokens = tokenizer.add_tokens(modifier_token)
#     if num_added_tokens == 0:
#         raise ValueError(
#             f"The tokenizer already contains the token {modifier_token}. Please pass a different"
#             " `modifier_token` that is not already in the tokenizer."
#         )

#     # Convert the initializer_token, placeholder_token to ids
#     token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
#     print(token_ids)

#     # Check if initializer_token is a single token or a sequence of tokens
#     if len(token_ids) > 1:
#         raise ValueError("The initializer token must be a single token.")

#     initializer_token_id.append(token_ids[0])
#     modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

# # Resize the token embeddings as we are adding new special tokens to the tokenizer
# text_encoder.resize_token_embeddings(len(tokenizer))

# # Initialise the newly added placeholder token with the embeddings of the initializer token
# token_embeds = text_encoder.get_input_embeddings().weight.data
# for x, y in zip(modifier_token_id, initializer_token_id):
#     token_embeds[x] = token_embeds[y]

# # Freeze all parameters except for the token embeddings in text encoder
# params_to_freeze = itertools.chain(
#     text_encoder.text_model.encoder.parameters(),
#     text_encoder.text_model.final_layer_norm.parameters(),
#     text_encoder.text_model.embeddings.position_embedding.parameters(),
# )
# freeze_params(params_to_freeze)

unet.to(accelerator.device, dtype=weight_dtype)
if accelerator.mixed_precision != "fp16" and args_modifier_token is not None:
    text_encoder.to(accelerator.device, dtype=weight_dtype)

# train_kv = True
# train_q_out = False if args_freeze_model == "crossattn_kv" else True
# custom_diffusion_attn_procs = {}

# st = unet.state_dict()

# for name, _ in unet.attn_processors.items():
#     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

#     if name.startswith("mid_block"):
#         hidden_size = unet.config.block_out_channels[-1]

#     elif name.startswith("up_blocks"):
#         block_id = int(name[len("up_blocks.")])
#         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

#     elif name.startswith("down_blocks"):
#         block_id = int(name[len("down_blocks.")])
#         hidden_size = unet.config.block_out_channels[block_id]

#     layer_name = name.split(".processor")[0]
#     weights = {
#         "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
#         "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
#     }

#     if train_q_out:
#         weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
#         weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
#         weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
    
#     if cross_attention_dim is not None:
#         custom_diffusion_attn_procs[name] = attention_class(
#             train_kv=train_kv,
#             train_q_out=train_q_out,
#             hidden_size=hidden_size,
#             cross_attention_dim=cross_attention_dim,
#         ).to(unet.device)
#         custom_diffusion_attn_procs[name].load_state_dict(weights)
#     else:
#         custom_diffusion_attn_procs[name] = attention_class(
#             train_kv=False,
#             train_q_out=False,
#             hidden_size=hidden_size,
#             cross_attention_dim=cross_attention_dim,
#         )

# del st 

# unet.set_attn_processor(custom_diffusion_attn_procs)
custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

# accelerator.register_for_checkpointing(custom_diffusion_layers)

custom_diffusion_layers, text_encoder = accelerator.prepare(custom_diffusion_layers, text_encoder)

# Restore state from a checkpoint path. You have to use the absolute path here.
# accelerator.load_state(repo_save_path + "/checkpoint-250")
accelerator.load_state("/data/aashish_final_expr/misc_projects/diffusers/examples/custom_diffusion/models/checkpoint-250")
# accelerator.load_state("/data/aashish_final_expr/neurips/methods/diffusers/examples/custom_diffusion/models/obama_checkpoints/checkpoint-750")

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    base_model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder)
)
pipeline.to(accelerator.device)

# Perform inference and save image
image = pipeline(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0
).images[0]

image.save("checkpoint_500_cat.png")