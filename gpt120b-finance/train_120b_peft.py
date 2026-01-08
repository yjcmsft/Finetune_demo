
#!/usr/bin/env python
"""
Finance-specific PEFT (LoRA/QLoRA) fine-tuning script for very large GPT-oss bases (~120B).
- Uses Hugging Face Transformers + PEFT
- Distributed via DeepSpeed ZeRO-3 (configured in ds_config_zerostage3.json)

Expected JSONL (SFT) format:
{"prompt": "...", "completion": "..."}

NOTE: For instruction/chat models, convert training pairs into chat JSONL before tokenization.
"""
import os
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True,
                        help='Base OSS model repo/name (e.g., org/gpt-neox-120b)')
    parser.add_argument('--train_jsonl', type=str, default='data/training.jsonl')
    parser.add_argument('--val_jsonl', type=str, default='data/validation.jsonl')
    parser.add_argument('--seq_len', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--grad_accum_steps', type=int, default=64)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--deepspeed', type=str, default='ds_config_zerostage3.json')
    return parser.parse_args()


def load_jsonl(path):
    # Returns HF datasets-compatible dict
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Concatenate prompt and completion; adjust as needed for chat format
            rows.append({'text': obj.get('prompt', '') + '\n' + obj.get('completion', '')})
    return rows


def main():
    args = get_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build datasets from JSONL
    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)

    # Use HF datasets to tokenize efficiently
    train_ds = load_dataset('json', data_files=args.train_jsonl, split='train')
    val_ds   = load_dataset('json', data_files=args.val_jsonl,   split='train')

    def to_text(rec):
        return {'text': rec.get('prompt', '') + '\n' + rec.get('completion', '')}

    train_ds = train_ds.map(to_text)
    val_ds   = val_ds.map(to_text)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, max_length=args.seq_len)

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tokenize_fn,   batched=True, remove_columns=val_ds.column_names)

    # Load base model (keep frozen except adapter params)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype='auto',
        device_map='auto',
    )

    # LoRA adapter config
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            'q_proj','k_proj','v_proj','o_proj',
            'gate_proj','up_proj','down_proj'
        ],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lcfg)

    # Training arguments
    targs = TrainingArguments(
        output_dir='./outputs',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy='steps',
        eval_steps=500,
        bf16=True,
        deepspeed=args.deepspeed,
        report_to=['tensorboard']
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter only
    os.makedirs('./outputs/peft-adapter', exist_ok=True)
    model.save_pretrained('./outputs/peft-adapter')
    tokenizer.save_pretrained('./outputs/peft-adapter')


if __name__ == '__main__':
    main()
