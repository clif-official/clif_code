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
    <rick1>,<morty1>,<mama1>,<jerry1>,<summer1>,\
    <nanhai1>,<geshen1>,<zufu1>,<zumu1>,\
    <chali1>,<lao1>,<nv1>,<wangka1>,<pang1>,\
    <tangseng1>,<sunwukong1>,<zhubajie1>,<shaheshang1>,\
    <rick2>,<morty2>,<mama2>,<jerry2>,<summer2>,\
    <nanhai2>,<geshen2>,<zufu2>,<zumu2>,\
    <chali2>,<lao2>,<nv2>,<wangka2>,<pang2>,\
    <tangseng2>,<sunwukong2>,<zhubajie2>,<shaheshang2>
" 

initializer_tokens="
    scientist,cartoon,cartoon,cartoon,cartoon,\
    kid,singer,skeleton,skeleton,\
    boy,grandpa,child,gentleman,fat,\
    monk,monk,monk,monk,\
    scientist,cartoon,cartoon,cartoon,cartoon,\
    kid,singer,skeleton,skeleton,\
    boy,grandpa,child,gentleman,fat,\
    monk,monk,monk,monk
"

VALID_PROMPT="
<zhubajie1> <zhubajie2> is snuggled up in <shaheshang1> <shaheshang2>'s arms, in front of Eiffel tower.||\
Two individuals kiss each other, <miguel1> <miguel2>, <sunwukong1> <sunwukong2>, indoor background.||\
Two individuals playing poker, <tangseng1> <tangseng2>, <joe1> <joe2>, indoor background.||\
<wonka1> <wonka2> shaking hands with <summer1> <summer2>, in front of the Mount Fuji.
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


cd examples/sdxl_lora_ti

accelerate launch $DIST_ARGS train_text_to_image_lora_ti_sdxl.py \
    $TRAIN_PATH \
    $TRAIN_ARGS \
    --placeholder_tokens="${placeholder_tokens}" \
    --initializer_tokens="${initializer_tokens}" \
    --validation_prompt="${VALID_PROMPT}"
