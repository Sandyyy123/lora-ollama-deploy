#!/bin/bash
# Convert merged HuggingFace model to GGUF Q4_K_M for Ollama
#
# Prerequisites:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp && make && pip install -r requirements.txt
#
# Usage:
#   bash scripts/convert_to_gguf.sh <merged_model_dir> <output_name>
#
# Example:
#   bash scripts/convert_to_gguf.sh output/my-lora-merged my-model-q4

set -e

MERGED_DIR="${1:?Usage: $0 <merged_model_dir> <output_name>}"
MODEL_NAME="${2:?Usage: $0 <merged_model_dir> <output_name>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
OUTPUT_DIR="./gguf"

mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Convert to fp16 GGUF ==="
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$OUTPUT_DIR/${MODEL_NAME}-fp16.gguf" \
    --outtype f16

echo "=== Step 2: Quantize to Q4_K_M ==="
"$LLAMA_CPP_DIR/llama-quantize" \
    "$OUTPUT_DIR/${MODEL_NAME}-fp16.gguf" \
    "$OUTPUT_DIR/${MODEL_NAME}-q4_k_m.gguf" \
    Q4_K_M

echo "=== Step 3: Verify GGUF ==="
"$LLAMA_CPP_DIR/llama-cli" \
    -m "$OUTPUT_DIR/${MODEL_NAME}-q4_k_m.gguf" \
    -p "Hello, who are you?" \
    -n 50 --temp 0.7

echo ""
echo "Done! GGUF model at: $OUTPUT_DIR/${MODEL_NAME}-q4_k_m.gguf"
echo "Next step: run scripts/create_ollama_model.sh"
