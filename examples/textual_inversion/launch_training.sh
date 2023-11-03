export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/Brad_Pitt/"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<brad-pitt>" --initializer_token="man" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="./outputs/textual_inversion_brad_pitt" \
  --num_vectors=5

    # --push_to_hub \