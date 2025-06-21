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

torch.set_num_threads(120)          # 推理/训练线程
torch.set_num_interop_threads(2)    # 任务调度线程

print("intra-op threads:", torch.get_num_threads())
print("interop threads:", torch.get_num_interop_threads())

os.environ["OMP_NUM_THREADS"]  = "120"
os.environ["MKL_NUM_THREADS"]  = "120"

from torch.profiler import profile, record_function, ProfilerActivity

from transformers.trainer_callback import TrainerCallback
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

class TorchProfilerCallback(TrainerCallback):
    def __init__(self, logdir="/home/lpl/peft_givens_cpu/log_cpu"):
        print(f"[Profiler] writing trace to -> {os.path.abspath(logdir)}")
        self.prof = profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=0, warmup=0, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler(logdir))
        self.prof.__enter__()   # 手动进入上下文

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()        # 每个训练 step 调一次

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.__exit__(None, None, None)  # 收尾写文件


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

def build_dataset(tokenizer, max_len: int,
                  json_path: str = "/home/lpl/peft_givens_cpu/ESC_pure_mind_gen_ER.json"
                 ) -> List[Dict[str, List[int]]]:
    """
    读取 ESC 纯精神支持对话数据集（JSON 格式），
    仅保留 utterance 中的 seeker→supporter 句对。

    参数
    ----
    tokenizer : transformers.PreTrainedTokenizer
        用于将文本转换为 token ID 的分词器。
    max_len : int
        句子最大长度（超过则截断，不足则 pad）。
    json_path : str
        JSON 文件路径，默认指向 `/home/lpl/peft_givens_cpu/ESC_pure_mind_gen_ER.json`。

    返回
    ----
    List[Dict[str, List[int]]]
        每个元素包含 `input_ids`（seeker）和 `labels`（supporter）。
    """
    pairs = []  # (seeker, supporter)

    # ---------- 读取并解析 JSON ----------
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 顶级键类似 "Qwen0.txt"、"Qwen1.txt" ...
    for file_block in raw_data.values():
        for item in file_block:
            utt = item.get("utterance", {})
            seeker = utt.get("seeker", "").strip().rstrip(" ,\"")
            supporter = utt.get("supporter", "").strip().rstrip(" ,\"")
            if seeker and supporter:
                pairs.append((seeker, supporter))

        if not pairs:
            raise ValueError("未提取到有效句对")

    # ---------- 分词编码 ----------
    ds_enc = []
    for seeker, supporter in pairs:
        enc_inp = tokenizer(
            seeker,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )
        enc_out = tokenizer(
            supporter,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )
        # 只需取列表中的第一条（因为 return_tensors=None）
        ds_enc.append({
            "input_ids": enc_inp["input_ids"],
            "labels": enc_out["input_ids"]
        })

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
        dataloader_num_workers=4, # 小规模并行加在数据，不大量fork worker
        max_steps=10,
        gradient_accumulation_steps=1,
        report_to="none",
        disable_tqdm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=None,
        data_collator=collator,
        callbacks=[TorchProfilerCallback("/home/lpl/peft_givens_cpu/log_cpu")],
    )

    print("---------------------START TRAINING----------------------")
    # with profile(activities=[ProfilerActivity.CPU],
    #          schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
    #          on_trace_ready=torch.profiler.tensorboard_trace_handler("/home/lpl/peft_givens_cpu/log_cpu")) as prof: # 改用callback类
    trainer.train()

    # merged = model.merge_and_unload()
    model.to(device).eval()
    with torch.no_grad():
        prompt = "I feel sad today, my mother punish me because of low grade."
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**inp, max_new_tokens=100)
        print("\n▶︎  小测试: ", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
