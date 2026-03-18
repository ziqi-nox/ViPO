#!/usr/bin/env bash
set -euo pipefail

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-12340}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-data/Wan2.1-T2V-1.3B}"
DATA_JSON="${DATA_JSON:-data/processed_prompts.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wan_grpo_pixel_adv}"

"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" --master_port="$MASTER_PORT" \
    fastvideo/train_vipo_wan_fsdp.py \
    --seed 42 \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --cache_dir ./cache_dir \
    --data_json_path "$DATA_JSON" \
    --train_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 10000 \
    --learning_rate 5e-6 \
    --output_dir "$OUTPUT_DIR" \
    --h 240 \
    --w 416 \
    --t 53 \
    --sampling_steps 16 \
    --eta 0.25 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --shift 8.0 \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --use_group \
    --enable_gradient_checkpointing \
    --use_sequential_cfg \
    --no_sharding \
    --use_videoalign \
    --bestofn 8 \
    --mq_coef 1.0 \
    --vq_coef 1.0 \
    --video_fps 8 \
    --use_bf16
