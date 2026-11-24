# -*- coding: utf-8 -*-
"""
数据清洗器微调脚本：用 data_cleaner/train.jsonl 训练 Qwen3-0.6B 做
"散乱英文网页文本 -> 干净 JSON" 的任务。

运行（Anaconda Prompt）：
    conda activate myft
    python "D:\\Documents\\qwen-agent\\myV7NC6Ru3\\default\\清洗器微调.py"

说明：
- 输入是整页散乱文本（1k~8k 字符），MAX_SEQ_LEN=2048 会截断超长样本，
  这是 6GB 显存下的取舍；想全长训练可换 1.7B 以下更小序列或分块。
- 目标 JSON 平均几百 token，放在序列末尾，2048 足够容纳多数样本的
  输入前缀+完整输出；截断时优先保输出（脚本里做了右截断输入的处理）。
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
LORA_R = 16
MAX_STEPS = 200          # 数据量几百条，200 步约 2~3 个 epoch
BATCH_SIZE = 1
GRAD_ACCUM = 8

INSTRUCTION = ("Clean the following messy English web page dump into a JSON object "
               "with keys title, date, paragraphs, comments. Remove navigation, ads, "
               "footers and duplicate lines; fix encoding artifacts. Output JSON only.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True, full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_R, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)


def load_split(name):
    rows = []
    with open(os.path.join(BASE, "data_cleaner", f"{name}.jsonl"), encoding="utf-8") as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows


def to_text(row):
    target = json.dumps(row["output"], ensure_ascii=False)
    user = INSTRUCTION + "\n\n" + row["input"]
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": target}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    return f"### 指令:\n{user}\n### 回答:\n{target}{tokenizer.eos_token}"


train_rows = load_split("train")
dataset = Dataset.from_list([{"text": to_text(r)} for r in train_rows])

trainer = SFTTrainer(
    model=model, train_dataset=dataset, tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10, max_steps=MAX_STEPS,
        learning_rate=2e-4, logging_steps=10,
        optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="cosine", seed=3407,
        output_dir="outputs_cleaner", report_to="none",
        save_strategy="no", dataset_num_proc=1, dataloader_num_workers=0,
    ),
)
trainer.train()

# ---- 用 test 集第一条当场看效果 ----
FastLanguageModel.for_inference(model)
row = load_split("test")[0]
user = INSTRUCTION + "\n\n" + row["input"]
if getattr(tokenizer, "chat_template", None):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
else:
    text = f"### 指令:\n{user}\n### 回答:\n"
ids = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=1024, use_cache=True)
pred = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
print("=" * 60)
print("模型输出：", pred[:1500])
print("=" * 60)
try:
    json.loads(pred)
    print("输出是合法 JSON ✔")
except Exception as e:
    print("输出暂不是合法 JSON（训练步数少时正常）：", e)

model.save_pretrained("lora_cleaner")
tokenizer.save_pretrained("lora_cleaner")
print("LoRA 已存到 lora_cleaner/；导出 Ollama 的方法见 微调练习.py 的 --export 段")
