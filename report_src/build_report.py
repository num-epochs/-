# -*- coding: utf-8 -*-
"""生成课程报告 PDF：软件学报版式，A4，正文>=5页。"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(BASE, "报告正文.html"), encoding="utf-8").read()
v1 = json.load(open(os.path.join(BASE, "eval_v1.json"), encoding="utf-8"))["summary"]
v2 = json.load(open(os.path.join(BASE, "eval_v2b.json"), encoding="utf-8"))["summary"]

def pct(x): return f"{x*100:.1f}%"

rows = ""
for label, key, is_pct in [
    ("直接可解析率", "direct_parse_rate", True),
    ("修复后可用率", "repaired_usable_rate", True),
    ("标题精确匹配率", "title_exact_rate", True),
    ("段落集合 F1", "mean_para_f1", True),
    ("日期正确率", "date_ok_rate", True),
]:
    a, b = v1[key], v2[key]
    fa, fb = (pct(a), pct(b)) if is_pct else (f"{a:.3f}", f"{b:.3f}")
    rows += f"<tr><td>{label}</td><td>{fa}</td><td>{fb}</td></tr>\n"

html = tpl.replace("@@TABLE_ROWS@@", rows)
html = html.replace("@@V1_SECONDS@@", str(v1["seconds"]))
html = html.replace("@@V2_SECONDS@@", str(v2["seconds"]))
html = html.replace("@@V1_USABLE@@", pct(v1["repaired_usable_rate"]))
html = html.replace("@@V2_USABLE@@", pct(v2["repaired_usable_rate"]))
html = html.replace("@@V1_DIRECT@@", pct(v1["direct_parse_rate"]))
html = html.replace("@@V2_DIRECT@@", pct(v2["direct_parse_rate"]))

out_html = os.path.join(BASE, f"_print_{int(time.time())}.html")
open(out_html, "w", encoding="utf-8").write(html)
pdf = os.path.join(BASE, "网页文本数据清洗智能系统与应用：从算法到系统实现.pdf")
if os.path.exists(pdf):
    try:
        with open(pdf, "ab"):
            pass
    except PermissionError:
        pdf = pdf.replace(".pdf", "（修订版）.pdf")
        print("NOTE: 原PDF被查看器占用，本次直接生成修订版")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
       "--disable-application-cache", "--print-to-pdf=" + pdf, "--print-to-pdf-no-header",
       "--virtual-time-budget=10000", out_html]
r = subprocess.run(cmd, capture_output=True, timeout=180)
print("chrome exit", r.returncode)
print("pdf exists", os.path.exists(pdf), os.path.getsize(pdf) if os.path.exists(pdf) else 0)

# 去除浏览器打印自动添加的页眉(日期/标题)与页脚(文件路径/页码)，非学报格式要求
import pymupdf
doc = pymupdf.open(pdf)
for page in doc:
    rc = page.rect
    page.add_redact_annot(pymupdf.Rect(0, 0, rc.width, 45))
    page.add_redact_annot(pymupdf.Rect(0, rc.height - 45, rc.width, rc.height))
    page.apply_redactions()
tmp = pdf + ".tmp.pdf"
doc.save(tmp)
doc.close()
try:
    os.replace(tmp, pdf)
except PermissionError:
    pdf = pdf.replace(".pdf", "（修订版）.pdf")
    os.replace(tmp, pdf)
    print("NOTE: 原文件被占用，已另存修订版，请关闭旧PDF后使用修订版")
doc = pymupdf.open(pdf)
t = "".join(p.get_text() for p in doc)
print("pages:", doc.page_count)
print("header_gone:", "2026/11/23" not in t, "footer_gone:", "file:///" not in t,
      "escape_gone:", "\\%" not in t)
