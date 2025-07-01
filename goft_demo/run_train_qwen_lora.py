#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 版 SFT demo（与 GOFT 脚本完全对齐的对照组）
"""

import argparse, json, os, torch
from typing import List, Dict
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, TrainingArguments,
                          Trainer)
from peft_givens_cpu.mapping        import get_peft_model, get_peft_config
from peft_givens_cpu.tuners.lora    import LoraConfig     # ← 核心改动

# ---------- 线程亲和 ----------
torch.set_num_threads(120)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = "120"
os.environ["MKL_NUM_THREADS"] = "120"

# ---------- 数据读取 ----------
def build_dataset(tok, max_len: int,
                  json_path="/home/lpl/peft_givens_cpu/ESC_pure_mind_gen_ER.json")\
                  -> List[Dict[str, List[int]]]:
    pairs = []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for blk in raw.values():
        for it in blk:
            u = it.get("utterance", {})
            s1, s2 = u.get("seeker","").strip(), u.get("supporter","").strip()
            if s1 and s2:
                pairs.append((s1, s2))
    if not pairs:
        raise RuntimeError("dataset empty!")

    enc = []
    for s1, s2 in pairs:
        enc_in  = tok(s1, truncation=True, padding="max_length",
                      max_length=max_len)["input_ids"]
        enc_out = tok(s2, truncation=True, padding="max_length",
                      max_length=max_len)["input_ids"]
        enc.append({"input_ids": enc_in, "labels": enc_out})
    return enc

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",
                   default="/home/lpl/peft_givens_cpu/models/Qwen2.5-0.5B")
    p.add_argument("--config",
                   default="/home/lpl/peft_givens_cpu/goft_demo/config_qwen_lora.json")
    p.add_argument("--epochs",  type=int,   default=1)
    p.add_argument("--bs",      type=int,   default=1)
    p.add_argument("--max_len", type=int,   default=512)
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--mode",    type=str,   default="cpu", choices=["cpu","gpu"])
    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()
    device = "cpu" if args.mode=="cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token":"<|extra_pad|>"})

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True)

    if len(tok) != base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(tok))

    # ---- LoRA config ----
    with open(args.config) as f:
        lora_cfg = get_peft_config(json.load(f))
    assert isinstance(lora_cfg, LoraConfig)

    model = get_peft_model(base, lora_cfg).to(device)
    model.print_trainable_parameters()

    # ---- Data ----
    ds_train  = build_dataset(tok, args.max_len)
    collator  = DataCollatorForLanguageModeling(tok, mlm=False)

    train_args = TrainingArguments(
        output_dir="out_qwen_lora",
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        no_cuda=(device=="cpu"),
        max_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,          # ★ 显式关键词
        eval_dataset=None,
        data_collator=collator           # ★
    )

    print("------ TRAIN ------")
    trainer.train()

    # quick test
    prompt = "I feel sad today, my mother punish me because of low grade."
    with torch.no_grad():
        out = model.generate(**tok(prompt, return_tensors="pt").to(device),
                             max_new_tokens=100)
    print("▶︎ 生成:", tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
