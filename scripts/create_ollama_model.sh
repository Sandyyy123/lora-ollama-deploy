#!/bin/bash
# Register GGUF model with Ollama and run a smoke test
#
# Usage:
#   bash scripts/create_ollama_model.sh <gguf_path> <model_name> [system_prompt]
#
# Example:
#   bash scripts/create_ollama_model.sh gguf/my-model-q4_k_m.gguf my-model "You are a helpful assistant."

set -e

GGUF_PATH="${1:?Usage: $0 <gguf_path> <model_name> [system_prompt]}"
MODEL_NAME="${2:?Usage: $0 <gguf_path> <model_name> [system_prompt]}"
SYSTEM_PROMPT="${3:-You are a helpful AI assistant.}"

MODELFILE_PATH="./Modelfile.${MODEL_NAME}"

echo "=== Creating Ollama Modelfile ==="
cat > "$MODELFILE_PATH" <<EOF
FROM ${GGUF_PATH}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

SYSTEM """${SYSTEM_PROMPT}"""
EOF

echo "Modelfile written to $MODELFILE_PATH"
cat "$MODELFILE_PATH"

echo ""
echo "=== Registering model with Ollama ==="
ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"

echo ""
echo "=== Smoke test ==="
ollama run "$MODEL_NAME" "Hello! Briefly describe what you can help with."

echo ""
echo "Model is live. Test it anytime with:"
echo "  ollama run $MODEL_NAME"
echo ""
echo "Or via REST API:"
echo "  curl http://localhost:11434/api/generate -d '{\"model\": \"$MODEL_NAME\", \"prompt\": \"Hello\", \"stream\": false}'"
