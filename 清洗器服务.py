# -*- coding: utf-8 -*-
"""
清洗器本地推理服务（LoRA 4bit 路线，模型行为与 notebook 一致）

为什么需要它：unsloth 的 QLoRA 适配器在本机任何"合并导出"（16bit/GGUF）都会
产出坏模型，只有 4bit 运行时正常。本服务用唯一正确的路线跑模型，并内置正则补丁。

启动（Anaconda Prompt，保持窗口开着）：
    conda activate myft
    python "D:\\Documents\\qwen-agent\\myV7NC6Ru3\\default\\清洗器服务.py"

调用：
    from 正则表达式 import clean_response
    data = clean_response(None, dirty_text=..., backend="local")
或 curl：
    curl -X POST http://127.0.0.1:8000/clean -H "Content-Type: application/json" -d "{\"text\": \"...\"}"
"""
import os
import json
import threading
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GEN_LOCK = threading.Lock()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "D:/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "patch", os.path.join(BASE, "正则表达式", "正则表达式.py"))
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

from unsloth import FastLanguageModel
import torch

INSTRUCTION = ("Clean the following messy English web page dump into a JSON object "
               "with keys title, date, paragraphs, comments. Remove navigation, ads, "
               "footers and duplicate lines; fix encoding artifacts. Output JSON only.")

print("加载 LoRA 4bit 模型（约 1GB 显存）...")
model, tok = FastLanguageModel.from_pretrained(
    model_name=r"D:/lora_cleaner_v2",
    max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print("模型就绪，监听 http://127.0.0.1:8000")


def generate_clean(text, max_new_tokens=1024):
    with GEN_LOCK:
        return _generate_clean(text, max_new_tokens)


def _generate_clean(text, max_new_tokens=1024):
    user = INSTRUCTION + "\n\n" + text
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=int(max_new_tokens),
                         temperature=0.1, no_repeat_ngram_size=3, use_cache=True)
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return patch.repair(raw)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ready":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/clean":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        result = generate_clean(body.get("text", ""),
                                body.get("max_new_tokens", 1024))
        data = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
