#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对本地 /home/lpl/peft_givens_cpu/models/Qwen2.5-3B 进行
少量 SFT（supervised fine-tuning）+ GOFT 适配的最小可运行示例。

* 仅演示：跑 1 epoch，小 batch，人畜无害地验证 forward/backward/merge。
* 依赖：
    pip install "transformers>=4.41" peft datasets
"""

import argparse, json, os, sys, torch, functools
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, TrainingArguments,
                          Trainer)
from peft_givens_cpu.mapping import get_peft_model, get_peft_config
from peft_givens_cpu.tuners.givens import GivensConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default="/home/lpl/peft_givens_cpu/models/Qwen2.5-0.5B",
                    help="本地 Qwen2.5-3B 路径")
    ap.add_argument("--config",
                    default="/home/lpl/peft_givens_cpu/goft_demo/config_qwen.json",
                    help="GivensConfig json")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1,
                    help="批大小(显存不足就降至 1)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mode", type=str, default='cpu', help="目前提供的模式分别称为，cpu, gpu, cpu+gpu")
    ap.add_argument("--fp16", action="store_true", help="在 GPU 上用 fp16 训练")
    return ap.parse_args()


def build_dataset(tokenizer, max_len):
    """
    为了离线演示，使用 datasets 自带的 `wikitext`：若缺网，将 fallback
    到脚本内置的两行文本。
    """
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                          split="train[:1%]", cache_dir="~/.cache/datasets")
        text_list = ds["text"]
    except Exception:
        text_list = [
            "ChatGPT is a large language model trained by OpenAI.",
            "Givens rotations are useful in numerical linear algebra."
        ]

    enc = tokenizer(
        text_list,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors=None,
    )
    ds_enc = [{"input_ids": ids,
               "labels": ids.copy()} for ids in enc["input_ids"]]
    return ds_enc

def main():
    args = parse_args()
    if args.mode == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"current_device: {device}")

    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '<|extra_pad|>'})

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None)

    if len(tok) != base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(tok))

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = get_peft_config(cfg_dict)
    model = get_peft_model(base, cfg)
    print(f"model:{model}")

    model.to(device)
    print(f"current_device: {device}")

    # 类型检查：强制所有内容变为fp32
    for n, p in model.named_parameters():
        if p.device != device:
            p.data = p.data.to(device)
        if p.dtype != torch.float32:
            p.data = p.data.float()

    # 控制只微调正交部分（Givens）对应的参数
    for name, param in model.named_parameters():
        if "givens_" in name:              # GOFT 旋转角 / scaler
            param.requires_grad_(True)
        elif ".bias" in name and cfg.bias == "givens_only":
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    model.print_trainable_parameters()

    ds_train = build_dataset(tok, args.max_len)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir="out_qwen_demo",
        per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=False,
        no_cuda=True,
        gradient_accumulation_steps=1,
        report_to="none",
        disable_tqdm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=None,
        data_collator=collator)

    print("---------------------START TRAINING----------------------")
    trainer.train()

    # merged = model.merge_and_unload()
    model.to(device).eval()
    with torch.no_grad():
        prompt = "Givens rotations"
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**inp, max_new_tokens=16)
        print("\n▶︎  小测试: ", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
