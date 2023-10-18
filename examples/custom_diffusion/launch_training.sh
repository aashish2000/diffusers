export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="./models/cat"
# export INSTANCE_DIR="../../../../datasets/entity_datasets/concept_datasets/v5/anna_e_cleaned_blurred_train/PER/Angelina_Jolie/"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --instance_prompt="Photo of <new1> cat"  \
  --class_prompt="cat" --num_class_images=200 \
  --resolution=64  \
  --train_batch_size=1  \
  --learning_rate=5e-6  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr --hflip --noaug \
  --freeze_model crossattn \
  --modifier_token "<new1>" \
  --resume_from_checkpoint "checkpoint-250"\
  --no_safe_serialization \
  # --enable_xformers_memory_efficient_attention 
