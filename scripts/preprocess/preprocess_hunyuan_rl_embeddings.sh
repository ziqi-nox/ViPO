#!/usr/bin/env bash
set -euo pipefail

GPU_NUM="${GPU_NUM:-8}"
MODEL_PATH="${MODEL_PATH:-data/hunyuan}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl_embeddings/hunyuan}"
PROMPT_DIR="${PROMPT_DIR:-${OUTPUT_DIR}/video_prompts.txt}"

cp -rf "${MODEL_PATH}/tokenizer/"* "${MODEL_PATH}/text_encoder"
cp -rf "${MODEL_PATH}/tokenizer_2/"* "${MODEL_PATH}/text_encoder_2"

torchrun --nproc_per_node="$GPU_NUM" --master_port 19002 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_dir "$PROMPT_DIR" \
    --model_type hunyuan_hf
