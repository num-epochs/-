# -*- coding: utf-8 -*-
"""
Qwen3-0.6B 新手微调全流程练习脚本（RTX 3050 6GB 适配版）

全流程：加载 4bit 底模 -> 挂 LoRA -> 小数据集 SFT -> 试推理 -> 保存 LoRA
        -> （可选 --export）合并导出 -> 导入 Ollama 对话

运行方式（在 Anaconda Prompt 里）：
    conda activate myft
    python "D:\\Documents\\qwen-agent\\myV7NC6Ru3\\default\\微调练习.py"

训练完成后想导出到 Ollama，再跑一次并加参数：
    python "D:\\Documents\\qwen-agent\\myV7NC6Ru3\\default\\微调练习.py" --export

显存预算：0.6B + 4bit + LoRA + 梯度检查点，峰值约 3GB，6GB 卡很宽裕。
"""
import os
import sys

# 必须在导入 unsloth / huggingface 之前设置：
# 1) 国内镜像加速下载  2) 模型缓存放 D 盘（C 盘空间紧张）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/hf-cache")

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ---------------- 基本参数（新手可以先只动这里） ----------------
# 底模用本地路径（已下载并验证）。不用仓库名，避免 Unsloth 联网找它的预量化版而失败
MODEL_NAME = "D:/hf-cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
MAX_SEQ_LEN = 1024                    # 序列长度，6GB 卡别超过 2048
LORA_R = 16                           # LoRA 秩，8~32 都行，越大越吃显存
MAX_STEPS = 60                        # 练习用 60 步足够看到 loss 下降
BATCH_SIZE = 1                        # 6GB 卡老老实实用 1
GRAD_ACCUM = 4                        # 等效 batch = 1 * 4 = 4
LEARNING_RATE = 2e-4                  # LoRA 常用学习率
OUTPUT_DIR = "outputs"                # 训练日志输出目录（相对当前运行目录）

# ---------------- 1. 加载底模（4bit 量化，省显存） ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,      # QLoRA：4bit 加载
    load_in_8bit=False,
    full_finetuning=False,  # 不做全参微调（6GB 卡也别试）
)

# ---------------- 2. 挂 LoRA 适配器 ----------------
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_R,
    lora_dropout=0,             # 0 是 unsloth 优化过的取值
    bias="none",
    use_gradient_checkpointing="unsloth",  # 省显存的关键开关
    random_state=3407,
)

# ---------------- 3. 准备一个迷你练习数据集 ----------------
# 故意内置在脚本里，不依赖外网数据集下载。
# 目标：让模型学会固定的回答风格（自我介绍 + 几个常识问答），
# 60 步后就能明显看到它"学进去了"。你自己练的时候换成自己的数据即可。
raw = [
    ("你是谁？", "我是李明睿用 Unsloth 微调出来的 Qwen3 小助手，正在学习怎么更好地回答问题。"),
    ("你叫什么名字？", "我叫小睿助手，是一个在 RTX 3050 上微调出来的练习模型。"),
    ("介绍一下你自己", "我是基于 Qwen3-0.6B 底模、用 LoRA 方法微调的小助手，体积很小但很努力。"),
    ("你是怎么训练出来的？", "我的底模是 Qwen3-0.6B-Base，主人用 Unsloth 框架做 4bit QLoRA 微调，只训练了很少的步数。"),
    ("你的显卡是什么？", "我跑在一块 NVIDIA RTX 3050 6GB 笔记本显卡上，显存不大，所以用了 4bit 量化。"),
    ("什么是 LoRA？", "LoRA 是一种参数高效微调方法：冻结底模，只训练插入的低秩矩阵，显存占用小、训练快。"),
    ("什么是 QLoRA？", "QLoRA 就是把底模量化到 4bit 再做 LoRA 微调，能让小显存显卡也跑得动大模型。"),
    ("什么是 Unsloth？", "Unsloth 是一个微调加速框架，重写了训练内核，通常能提速约 2 倍并明显降低显存占用。"),
    ("微调和学习率有什么关系？", "学习率决定每步参数更新的幅度。LoRA 微调常用 2e-4 左右，太大容易训崩，太小学得慢。"),
    ("batch size 是什么？", "batch size 是一次前向传播喂给模型的样本数。显存小就用 1，再靠梯度累积凑等效批量。"),
    ("梯度累积是什么？", "梯度累积是把多个小批量的梯度加起来再更新一次参数，用小显存模拟大批量训练。"),
    ("过拟合是什么？", "过拟合是模型把训练数据背下来了，换新问题就答不好。数据太少、训练太久都容易过拟合。"),
    ("南京理工大学在哪？", "南京理工大学位于江苏省南京市，是一所以工科见长的全国重点大学。"),
    ("无锡在哪？", "无锡在江苏省南部、太湖之滨，是长三角重要的工业和旅游城市。"),
    ("Python 是什么？", "Python 是一门简洁易读的编程语言，广泛用于数据分析、人工智能和自动化办公。"),
    ("conda 是干什么的？", "conda 是环境和包管理工具，可以为每个项目建独立的虚拟环境，避免依赖互相打架。"),
    ("Ollama 是什么？", "Ollama 是本地大模型推理工具，一条命令就能拉起模型对话，还支持导入自己微调的模型。"),
    ("GGUF 是什么？", "GGUF 是 llama.cpp 生态的模型文件格式，Ollama 底层就用它，量化后体积小、CPU 也能跑。"),
    ("显存不够怎么办？", "可以减小序列长度和 batch size、开梯度检查点、用 4bit 量化，或者换更小的模型。"),
    ("训练 loss 不降怎么办？", "先检查数据格式和学习率，再看标签是否只落在回答部分；练习时 loss 稳步下降就是好现象。"),
    ("你好", "你好！我是小睿助手，很高兴见到你，有什么想问的尽管说。"),
    ("谢谢", "不客气！能帮到你我就很开心，继续加油练习微调吧。"),
    ("再见", "再见！下次训练记得先激活 myft 环境哦。"),
    ("给我一句鼓励", "显卡虽小，志气不小；0.6B 的模型也能学出大名堂，坚持练下去！"),
]

def format_example(instruction, output):
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    if getattr(tokenizer, "chat_template", None):
        # 底模若自带对话模板，直接套用（和将来 Ollama 对话格式一致）
        return tokenizer.apply_chat_template(messages, tokenize=False)
    # 没有模板就退化为纯文本格式
    return f"### 指令:\n{instruction}\n### 回答:\n{output}{tokenizer.eos_token}"

dataset = Dataset.from_list(
    [{"text": format_example(q, a)} for q, a in raw]
)

# ---------------- 4. 开始训练 ----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        optim="adamw_8bit",       # 8bit 优化器，再省一点显存
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",         # 不上报 wandb 等平台
        save_strategy="no",       # 练习阶段不存中间检查点，省磁盘
        dataset_num_proc=1,       # Windows 下多进程容易出问题，用 1
        dataloader_num_workers=0,
    ),
)

print("=" * 50)
print("开始训练：loss 应该从 2~3 逐步降到 1 以下")
print("=" * 50)
trainer.train()

# ---------------- 5. 训练完当场试一句 ----------------
FastLanguageModel.for_inference(model)
test_prompt = "你是谁？"
if getattr(tokenizer, "chat_template", None):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
else:
    text = f"### 指令:\n{test_prompt}\n### 回答:\n"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print("=" * 50)
print("微调后的回答：")
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print("=" * 50)

# ---------------- 6. 保存 LoRA 适配器（只有几十 MB） ----------------
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("LoRA 已保存到 lora_model/（底模不用重复存）")

# ---------------- 7. 可选：合并导出，交给 Ollama ----------------
if "--export" in sys.argv:
    merged_dir = os.path.abspath("merged_model")
    print("正在合并 LoRA 到底模并导出 16bit 权重（约 1.2GB）...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    modelfile = os.path.abspath("Modelfile")
    with open(modelfile, "w", encoding="utf-8") as f:
        f.write(f'FROM "{merged_dir}"\n')
        f.write('PARAMETER temperature 0.7\n')
    print("=" * 50)
    print("导出完成！接下来在任意终端执行（Ollama 会自动转换并量化）：")
    print(f'    ollama create xiaorui-ft -f "{modelfile}" --quantize q4_k_m')
    print("    ollama run xiaorui-ft")
    print("=" * 50)
else:
    print("提示：想把微调结果导入 Ollama 对话，加 --export 参数再跑一次本脚本。")
