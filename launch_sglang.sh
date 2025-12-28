#!/bin/bash

# Default values
MODEL_PATH=${1:-"Qwen/Qwen3-Embedding-8B"}
PORT=${2:-30000}
HOST=${3:-"0.0.0.0"}
GPU_ID=${4:-0}

echo "Starting SGLang server with model: $MODEL_PATH on port: $PORT using GPU: $GPU_ID"

# Launch SGLang server for embedding
# Note: Ensure you have sglang installed: pip3 install sglang
CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --host "$HOST" \
    --is-embedding

