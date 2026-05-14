"""
LoRA fine-tuning script using HuggingFace PEFT + transformers.

Usage:
    python scripts/finetune_lora.py --config configs/lora_config.yaml

Supports: Llama-3, Mistral, Phi-3, Gemma, and any causal LM.
Output: merged model saved to output_dir/, ready for GGUF conversion.
"""

import argparse
import os
import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict):
    """4-bit quantization for QLoRA (load in 4-bit, train adapters in bf16)."""
    if cfg.get("use_qlora", False):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def main(config_path: str):
    cfg = load_config(config_path)

    print(f"Base model: {cfg['model_name']}")
    print(f"LoRA rank: {cfg['lora_r']}, alpha: {cfg['lora_alpha']}")

    bnb_config = build_bnb_config(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    ds_cfg = cfg["dataset"]
    if ds_cfg.get("path", "").endswith(".jsonl") or ds_cfg.get("path", "").endswith(".json"):
        dataset = load_dataset("json", data_files=ds_cfg["path"], split="train")
    else:
        dataset = load_dataset(ds_cfg["path"], split=ds_cfg.get("split", "train"))

    if "text_column" in ds_cfg:
        dataset = dataset.rename_column(ds_cfg["text_column"], "text")

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit" if cfg.get("use_qlora") else "adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 2048),
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    # Merge LoRA weights into base model and save
    print("Merging LoRA adapters...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Merged model saved to {cfg['output_dir']}/")
    print("Next step: run scripts/convert_to_gguf.sh to quantize for Ollama.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    args = parser.parse_args()
    main(args.config)
