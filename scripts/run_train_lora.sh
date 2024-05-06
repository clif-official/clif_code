#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PORT=24002


TRAIN_PATH="
    --pretrained_model_name_or_path="./ckpt/stable-diffusion-xl-base-1.0" \
    --pretrained_ti_path="./exp/ti" \
    --pretrained_vae_model_name_or_path="./ckpt/sdxl-vae-fp16-fix" \
    --train_data_dir="./dataset" \
    --caption_column="text" \
    --output_dir="./exp/clif_ti_lora" \
    --report_to="wandb"
"

placeholder_tokens="
    <rick1>,<rick2>,<morty1>,<morty1>,<beth1>,<beth2>,<jerry1>,<jerry2>,<summer1>,<summer2>,\
    <miguel1>,<miguel2>,<cruze1>,<cruze2>,<hector1>,<hector2>,<imelda1>,<imelda2>,\
    <charlie1>,<charlie2>,<joe1>,<joe2>,<violet1>,<violet2>,<wonka1>,<wonka2>,<philip1>,<philip2>,\
    <tangseng1>,<tangseng2>,<sunwukong1>,<sunwukong2>,<zhubajie1>,<zhubajie2>,<shaheshang1>,<shaheshang2>
"

initializer_tokens="
    scientist,scientist,cartoon,cartoon,cartoon,cartoon,cartoon,cartoon,cartoon,cartoon,\
    kid,kid,singer,singer,skeleton,skeleton,skeleton,skeleton,\
    boy,boy,grandpa,grandpa,child,child,gentleman,gentleman,fat,fat,\
    monk,monk,monk,monk,monk,monk,monk,monk
"

VALID_PROMPT="
<joe1> <joe2> is snuggled up in <sunwukong1> <sunwukong2>'s arms, in front of Eiffel tower.
"
    
TRAIN_ARGS="
    --train_stage="load_ti_do_both" \
    --resolution=1024 \
    --num_train_epochs=600 \
    --validation_epochs=100 \
    --num_validation_images=2 \
    --checkpointing_steps=1000000 \
    --train_batch_size=5 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-4 \
    --lr_scheduler="constant" \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --seed=42 \
    --rank=16 \
"

IFS=', ' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
num_devices=${#devices[@]}

DIST_ARGS="
    --mixed_precision fp16 \
    --num_cpu_threads_per_process 4 \
    --num_processes $num_devices \
    --num_machines 1 \
    --dynamo_backend no \
    --main_process_port $MAIN_PORT \
"
if [ $num_devices -gt 1 ]; then DIST_ARGS+=" --multi_gpu"; fi


cd train

accelerate launch $DIST_ARGS train_text_to_image.py \
    $TRAIN_PATH \
    $TRAIN_ARGS \
    --placeholder_tokens="${placeholder_tokens}" \
    --initializer_tokens="${initializer_tokens}" \
    --validation_prompt="${VALID_PROMPT}"
