# -*- coding: utf-8 -*-
"""
治本重训脚本（可选，你自己决定何时跑）：解决"截断/不会收尾"的根因。

根因回顾：原训练 max_seq_length=2048，多数样本"指令+脏输入+完整JSON"超长被截断，
模型从没见过完整 JSON 的收尾，推理时自然截断/幻觉。本脚本：
  1) 只保留 tokenize 后总长 <= MAX_SEQ_LEN 的样本（模型真正学会"写完"）；
  2) 步数按数据量自适应（约 4 个 epoch）；
  3) 存到新目录 lora_cleaner_v2，不覆盖你原来的 lora_cleaner。
跑完后把 清洗器服务.py 里的模型路径改成 lora_cleaner_v2 即可对比效果。

运行：conda activate myft && python 清洗器重训建议.py
"""
import os
import json

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/hf-cache")

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "D:/hf-cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
MAX_SEQ_LEN = 2048

INSTRUCTION = ("Clean the following messy English web page dump into a JSON object "
               "with keys title, date, paragraphs, comments. Remove navigation, ads, "
               "footers and duplicate lines; fix encoding artifacts. Output JSON only.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True, full_finetuning=False)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407)

rows = []
with open(os.path.join(BASE, "data_cleaner", "train.jsonl"), encoding="utf-8") as f:
    for ln in f:
        rows.append(json.loads(ln))

texts, kept, dropped = [], 0, 0
for r in rows:
    target = json.dumps(r["output"], ensure_ascii=False)
    user = INSTRUCTION + "\n\n" + r["input"]
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": target}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    n = len(tokenizer(text)["input_ids"])
    if n <= MAX_SEQ_LEN:          # 关键：只学"写得完"的样本
        texts.append(text)
        kept += 1
    else:
        dropped += 1
print(f"保留 {kept} 条完整样本，丢弃 {dropped} 条超长样本")

# 步数对齐 v1 已验证收敛的训练量（v1=200步/234条≈3.4轮，loss 1.4）；
# 这里 163 条完整样本训 5 轮 ≈ 200 步。之前 v2 只训 163 步(4轮)可能欠拟合。
steps = max(200, (kept * 5) // 4)
trainer = SFTTrainer(
    model=model, train_dataset=Dataset.from_list([{"text": t} for t in texts]),
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        warmup_steps=5, max_steps=steps, learning_rate=2e-4,
        logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407,
        output_dir="D:/lora_cleaner_v2b_checkpoints", report_to="none",
        save_strategy="steps", save_steps=50,   # 每50步存一个检查点，可挑最优
        dataset_num_proc=1, dataloader_num_workers=0))
trainer.train()
model.save_pretrained("D:/lora_cleaner_v2b")
tokenizer.save_pretrained("D:/lora_cleaner_v2b")
print("已存 D:/lora_cleaner_v2b/（及 checkpoints/ 里 50/100/150/200 步版本）")
print("对比方法：改 清洗器服务.py 的 model_name 指向某个检查点目录，重启服务跑 --demo")
