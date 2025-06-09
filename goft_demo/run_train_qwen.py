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
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, TrainingArguments,
                          Trainer)
from peft_givens_cpu.mapping import get_peft_model, get_peft_config
from peft_givens_cpu.tuners.givens import GivensConfig


# ---------- 参数 ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default="/home/lpl/peft_givens_cpu/models/Qwen2.5-3B",
                    help="本地 Qwen2.5-3B 路径")
    ap.add_argument("--config",
                    default="/home/lpl/peft_givens_cpu/goft_demo/config_qwen.json",
                    help="GivensConfig json")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1,
                    help="批大小(显存不足就降至 1)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fp16", action="store_true", help="在 GPU 上用 fp16 训练")
    return ap.parse_args()


# ---------- 数据 ----------
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


# ---------- 主流程 ----------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载 tokenizer / 模型
    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '<|extra_pad|>'})

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if (args.fp16 and device == "cuda") else
                     torch.float32,
        low_cpu_mem_usage=True,
        device_map=None)     # 显式 later `.to(device)`

    # 2) 创建 GivensConfig
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = get_peft_config(cfg_dict)
    model = get_peft_model(base, cfg)

    model.resize_token_embeddings(len(tok))
    model.to(device)
    model.print_trainable_parameters()

    # 3) 数据
    ds_train = build_dataset(tok, args.max_len)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # 4) HF Trainer
    training_args = TrainingArguments(
        output_dir="out_qwen_demo",
        per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        fp16=args.fp16 and device == "cuda",
        bf16=False,
        gradient_accumulation_steps=1,
        report_to="none",
        disable_tqdm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=None,
        data_collator=collator)

    trainer.train()

    # 5) merge & quick sanity
    merged = model.merge_and_unload()
    merged.to(device).eval()
    with torch.no_grad():
        prompt = "Givens rotations"
        inp = tok(prompt, return_tensors="pt").to(device)
        out = merged.generate(**inp, max_new_tokens=16)
        print("\n▶︎  小测试: ", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
