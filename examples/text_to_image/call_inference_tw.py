from inference import generate_lora_stable_diffusion_images
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_finetuned_path', type=str, required=True)
parser.add_argument('--generations_path', type=str, required=True)
parser.add_argument('--checkpoint_name', type=str, required=True)
parser.add_argument('--weight', type=str, required=True)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()


generate_lora_stable_diffusion_images(model_orig_path="runwayml/stable-diffusion-v1-5",
                                      checkpoint_name=args.checkpoint_name, 
                                      flag_full_finetune="tw", 
                                      model_finetuned_path=args.model_finetuned_path,
                                      generations_path=args.generations_path,
                                      seed=args.seed,
                                      weight=args.weight,
                                      dataset_path=args.dataset_path)