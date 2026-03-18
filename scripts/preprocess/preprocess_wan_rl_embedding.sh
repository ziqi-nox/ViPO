#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-data/Wan2.1-T2V-1.3B}"
INPUT_TXT="${INPUT_TXT:-data/prompts/video_prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl_embeddings/wan}"

python fastvideo/data_preprocess/preprocess_wan_embeddings_fromlist.py \
    --wan_model_path "$MODEL_PATH" \
    --input_txt "$INPUT_TXT" \
    --output_dir "$OUTPUT_DIR"
