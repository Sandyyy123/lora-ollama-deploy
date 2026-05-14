# LoRA Fine-Tuning + Ollama Deployment Pipeline

End-to-end pipeline to fine-tune any causal LM with LoRA/QLoRA, quantize to Q4_K_M GGUF, and deploy locally via Ollama.

## Pipeline overview

```
Training data (JSONL)
        │
        ▼
[1] LoRA/QLoRA fine-tune     ← scripts/finetune_lora.py
        │
        ▼
Merged HuggingFace model
        │
        ▼
[2] Convert to GGUF fp16     ← scripts/convert_to_gguf.sh
        │
        ▼
[3] Quantize to Q4_K_M       ← llama-quantize
        │
        ▼
[4] Register with Ollama     ← scripts/create_ollama_model.sh
        │
        ▼
REST API on localhost:11434
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Also needed for GGUF conversion:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

### 2. Prepare your dataset

Format your training data as JSONL, one example per line:

```jsonl
{"text": "<|system|>You are a chemistry expert.</s><|user|>What is the boiling point of ethanol?</s><|assistant|>The boiling point of ethanol is 78.37°C at standard pressure.</s>"}
{"text": "<|system|>You are a chemistry expert.</s><|user|>Explain oxidation.</s><|assistant|>Oxidation is the loss of electrons from an atom or molecule...</s>"}
```

Save to `data/train.jsonl`.

### 3. Configure

Edit `configs/lora_config.yaml`:

```yaml
model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
use_qlora: true      # set false if you have 40GB+ VRAM
lora_r: 16
dataset:
  path: "data/train.jsonl"
```

### 4. Fine-tune

```bash
python scripts/finetune_lora.py --config configs/lora_config.yaml
# Merged model saved to output/lora-merged/
```

### 5. Convert to GGUF Q4

```bash
bash scripts/convert_to_gguf.sh output/lora-merged my-model
# Output: gguf/my-model-q4_k_m.gguf
```

### 6. Deploy with Ollama

```bash
bash scripts/create_ollama_model.sh gguf/my-model-q4_k_m.gguf my-model "You are a domain expert."

# Test:
ollama run my-model

# Or via API:
curl http://localhost:11434/api/generate \
  -d '{"model": "my-model", "prompt": "Hello", "stream": false}'
```

## Hardware requirements

| Setup | VRAM needed | Notes |
|-------|------------|-------|
| QLoRA (4-bit) + 7B model | 8-12 GB | RTX 3080 / 4070 Ti |
| QLoRA (4-bit) + 13B model | 14-16 GB | RTX 4090 |
| Full LoRA + 7B model | 20-24 GB | A100 / H100 |

Q4_K_M inference via Ollama: ~4.5 GB VRAM for a 7B model.

## Supported base models

Any HuggingFace causal LM with attention projections:
- Llama 3 / 3.1 / 3.2 (8B, 70B)
- Mistral 7B / Mixtral 8x7B
- Phi-3 / Phi-3.5
- Gemma 2
- Qwen 2.5

## Stack

- [PEFT](https://github.com/huggingface/peft) - LoRA adapters
- [TRL](https://github.com/huggingface/trl) - SFTTrainer
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF conversion + quantization
- [Ollama](https://ollama.ai) - local model serving

## Author

Dr. Sandeep Grover - Data Science PhD, ML Engineer
