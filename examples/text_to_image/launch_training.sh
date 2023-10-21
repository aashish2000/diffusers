export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export TRAIN_DIR="../../../../neurips/datasets/misc_samples/test_topic/"
# export TRAIN_DIR="../../../../neurips/datasets/non_entity_datasets/anna_ne_512/huggingface/"
# export TRAIN_DIR="../../../../neurips/datasets/non_entity_datasets/anna_ne_caption_prefixes/objects_list/huggingface_sharpened/"
export TRAIN_DIR="../../../../neurips/datasets/non_entity_datasets/anna_ne_sharpened_512/huggingface/"
# export OUTPUT_DIR="/tmp/"
# export OUTPUT_DIR="/tmp/"
export OUTPUT_DIR="./models/finetuned_lora_sharpened/"

# accelerate launch train_text_to_image.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#  --enable_xformers_memory_efficient_attention 


# accelerate launch --main_process_port=25000 weighted_lora_train.py \
#   --mixed_precision="fp16" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#   --checkpointing_steps=250 \
#   --validation_prompt="With great power comes great responsibilty." \
#   --seed=42

# accelerate launch --main_process_port=25000 weighted_lora_train.py \
#   --mixed_precision="fp16" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DIR \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#   --checkpointing_steps=250 \
#   --validation_prompt="A salad of strawberries, goat cheese and candied almonds." \
#   --seed=42 \

accelerate launch --main_process_port=25000 train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=250 \
  --validation_prompt="A salad of strawberries, goat cheese and candied almonds." \
  --seed=42 \


  # --enable_xformers_memory_efficient_attention


  # --dataset_name=$DATASET_NAME \
  # --push_to_hub \
  # --hub_model_id=${HUB_MODEL_ID} \
  # --report_to=wandb \