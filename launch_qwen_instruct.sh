#!/bin/bash

# Launch Qwen2.5-7B-Instruct for generation tasks
# Usage: bash launch_qwen_instruct.sh [model_path] [port] [host] [gpu_id]

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${2:-30001}
HOST=${3:-"0.0.0.0"}
GPU_ID=${4:-7}  # Default to GPU 7 (GPU 0 is broken, GPU 7 has most free memory)

echo "=========================================="
echo "Starting Qwen2.5-7B-Instruct Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Host: $HOST"
echo "GPU: $GPU_ID"
echo "=========================================="

# Launch SGLang server for generation (without --is-embedding)
# Note: Chat template is auto-detected from model's tokenizer_config.json
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST"
