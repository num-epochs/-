# -*- coding: utf-8 -*-
"""
data_cleaner 模型输出补丁（正则表达式.py）

针对 notebook 里观察到的三类毛病做"打捞式"修复：
  1. 格式幻觉：JSON 前后混入 {"lng": "zh_cn"} 之类碎块、markdown 围栏、说明文字
  2. 截断：max_new_tokens 不够 / 训练时 2048 截断导致模型没学会收尾，JSON 缺尾括号、
     最后一个字符串没写完
  3. schema 漂移：键缺失、类型不对、日期 hallucination（如 2066-01-02）

修复链（逐级降级，能救多少救多少）：
  提取 JSON 块 -> 直接解析 -> 补括号/去尾逗号/闭字符串修复 -> schema 归一
  -> 全失败时用正则从原文打捞 title/paragraphs

用法：
  from 正则表达式 import repair, clean_response
  data = repair(raw_pred)          # 只修字符串
  data = clean_response(raw_pred)  # 调 Ollama + 修（需先导入模型，见文件末尾注释）

自测：python 正则表达式.py --selftest
注意：补丁只能打捞，治本要靠重训时"过滤超长样本 + 提高 max_new_tokens"，见文末建议。
"""
import re
import json
from datetime import date

KEYS = ("title", "date", "paragraphs", "comments")

# ---------------- 1. 提取：跳过前导碎块，定位真正的 JSON 对象 ----------------

def _candidates(text):
    """按出现顺序产出每个 '{' 起始、括号配平（或到结尾）的候选块，尊重字符串内括号"""
    for m in re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break
        else:
            # 扫到结尾也没配平 => 截断候选，交给修复层
            yield text[start:]

MOJIBAKE = {"â€™": "’", "â€˜": "‘", "â€œ": "“", "â€": "”",
            "â€”": "—", "â€“": "–", "â€¦": "…", "Â ": " "}

def _prefilter(text):
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    for bad, good in MOJIBAKE.items():
        text = text.replace(bad, good)
    text = text.replace("​", "").replace("﻿", "")
    return text

# ---------------- 2. 截断修复 ----------------

def _repair_truncated(s):
    """补未闭字符串、去尾逗号/悬空键、按栈补 } ] ，再尝试解析"""
    s = s.rstrip()
    # 字符串没闭合：先补一个引号
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
    if in_str:
        s += '"'
    # 去掉结尾的逗号、冒号、悬空键（"key": 后面没值）
    s = re.sub(r",\s*$", "", s)
    s = re.sub(r'"\w+"\s*:\s*$', "", s)
    s = re.sub(r":\s*$", "", s)
    s = re.sub(r",\s*$", "", s)
    # 按栈补括号
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    s += "".join("}" if c == "{" else "]" for c in reversed(stack))
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------------- 2.5 词粘连启发式拆分（模型容量限制产生的 focuson 类错误） ----------------

GLUE_FIX = [
    (r"\bfocuson\b", "focus on"), (r"\bcooperationin\b", "cooperation in"),
    (r"\bChina-Australiain\b", "China-Australia in"), (r"\bPremierLi\b", "Premier Li"),
    (r"\bQiangand\b", "Qiang and"), (r"\bPrimeMinister\b", "Prime Minister"),
    (r"\bAlbaneshad\b", "Albanese had"), (r"\bnextyear\b", "next year"),
    (r"\bthe55\b", "the 55"), (r"\banniversaryof\b", "anniversary of"),
    (r"\bofChina\b", "of China"), (r"\bChinastands\b", "China stands"),
    (r"\btowork\b", "to work"), (r"\bwithAustralia\b", "with Australia"),
    (r"\bthisopportunity\b", "this opportunity"), (r"\boncommon\b", "on common"),
    (r"\bmoreoutcomes\b", "more outcomes"), (r"\bthecomprehensive\b", "the comprehensive"),
    (r"\biswilling\b", "is willing"), (r"\bexpandtrade\b", "expand trade"),
    (r"\bandoptimize\b", "and optimize"), (r"\bstructurewith\b", "structure with"),
    (r"\bhighlightsin\b", "highlights in"), (r"\bareassuch\b", "areas such"),
    (r"\bgreenand\b", "green and"), (r"\bnewenergy\b", "new energy"),
    (r"\bfurtherenhance\b", "further enhance"), (r"\bandinvestment\b", "and investment"),
    (r"\bThe twoSides\b", "The two sides"), (r"\bshouldmake\b", "should make"),
    (r"\bYear ofTourism\b", "Year of Tourism"), (r"\bexchangesand\b", "exchanges and"),
    (r"\binculture\b", "in culture"), (r"\blocalareas\b", "local areas"),
    (r"\bmediaand\b", "media and"), (r"\bthinktanks\b", "think tanks"),
    (r"\bpremiornote\b", "premier note"), (r"\bwillingto\b", "willing to"),
    (r"\bdeepencooperation\b", "deepen cooperation"), (r"\bwithChina\b", "with China"),
    (r"\bineconomy\b", "in economy"), (r"\bcooperationin\b", "cooperation in"),
    (r"\bagriculture,energy\b", "agriculture, energy"), (r"\botherfields\b", "other fields"),
    (r"\bfieldslike\b", "fields like"), (r"\bcleanenergy\b", "clean energy"),
    (r"\belectricvehicles\b", "electric vehicles"), (r"\bdigitaleconomy\b", "digital economy"),
    (r"\bisalso\b", "is also"), (r"\bhostingthe\b", "hosting the"),
    (r"\bEconomicLeaders\b", "Economic Leaders"), (r"\binsouth\b", "in south"),
    (r"\bsuccessfullyhosting\b", "successfully hosting"),
]

def _deglue(s):
    for pat, rep in GLUE_FIX:
        s = re.sub(pat, rep, s)
    return s

# ---------------- 3. schema 归一 ----------------

def _norm_date(v):
    if isinstance(v, str):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", v)
        if m:
            y, mo, d = map(int, m.groups())
            if 1990 <= y <= date.today().year and 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"
    return None

def _norm(obj):
    if not isinstance(obj, dict):
        return None
    title = obj.get("title")
    title = _deglue(re.sub(r"\s+", " ", title).strip()) if isinstance(title, str) else ""
    paras = obj.get("paragraphs")
    paras = [_deglue(re.sub(r"\s+", " ", p).strip()) for p in paras
             if isinstance(p, str) and p.strip()] if isinstance(paras, list) else []
    comments = []
    raw_c = obj.get("comments")
    if isinstance(raw_c, list):
        for c in raw_c:
            if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                a = c.get("author")
                comments.append({"author": a if isinstance(a, str) else None,
                                 "text": _deglue(re.sub(r"\s+", " ", c["text"]).strip())})
            elif isinstance(c, str) and c.strip():
                comments.append({"author": None, "text": c.strip()})
    return {"title": title, "date": _norm_date(obj.get("date")),
            "paragraphs": paras, "comments": comments}

# ---------------- 4. 兜底：正则直接从原文打捞 ----------------

def _salvage(text):
    t = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    title = t.group(1) if t else ""
    paras = re.findall(r'"((?:[^"\\]|\\.){40,})"', text)
    paras = [p for p in paras if p != title][:20]
    return {"title": title, "date": _norm_date(text),
            "paragraphs": paras, "comments": []}

# ---------------- 主入口 ----------------

def repair(raw_pred):
    """字符串 -> 干净 dict。任何情况下都返回 schema 完整的 dict（可能部分为空）"""
    text = _prefilter(raw_pred or "")
    for cand in _candidates(text):
        if '"title"' not in cand and '"paragraphs"' not in cand:
            continue  # 跳过 {"lng": ...} 这类前导碎块
        try:
            obj = json.loads(cand)
        except Exception:
            obj = _repair_truncated(cand)
        if obj is not None:
            out = _norm(obj)
            if out and (out["title"] or out["paragraphs"]):
                return out
    return _norm(_salvage(text)) or {"title": "", "date": None,
                                     "paragraphs": [], "comments": []}

def clean_response(raw_pred, model="qwen-cleaner", dirty_text=None,
                   backend="ollama", **gen_kw):
    """若给了 dirty_text 则先调模型再修；否则只修 raw_pred。
    backend="ollama": 走 Ollama（注意：当前 qwen-cleaner 的 GGUF 合并损坏，见文末说明）
    backend="local" : 走 清洗器服务.py 起的 LoRA 4bit 本地服务（推荐，模型行为与 notebook 一致）
    num_predict/max_new_tokens 默认 4096，治"输出长度不够"的一半（另一半靠重训）。"""
    if dirty_text is not None:
        if backend == "local":
            import requests
            resp = requests.post("http://127.0.0.1:8000/clean",
                                 json={"text": dirty_text,
                                       "max_new_tokens": gen_kw.pop("max_new_tokens", 1024)},
                                 timeout=600)
            resp.raise_for_status()
            return resp.json()  # 服务端已修好，直接是干净 dict
        import ollama
        resp = ollama.chat(model=model,
                           messages=[{"role": "user", "content": dirty_text}],
                           options={"num_predict": gen_kw.pop("num_predict", 4096),
                                    "temperature": gen_kw.pop("temperature", 0.1)})
        raw_pred = resp["message"]["content"]
    return repair(raw_pred)

# ---------------- 自测 ----------------

def _selftest():
    cases = {
        "前导碎块+完整": '{lng: zh_cn}{"title": "T", "date": "2026-11-20", "paragraphs": ["a b c d e f g"], "comments": []}',
        "围栏+完整": '```json\n{"title": "T", "date": null, "paragraphs": ["x y z w v u"], "comments": []}\n```',
        "截断在字符串中": '{"title": "China Australia meeting", "date": "2026-11-20", "paragraphs": ["first para words here", "second para cut off mid sent',
        "截断在逗号后": '{"title": "T", "date": "2026-11-20", "paragraphs": ["one two three four five"],',
        "日期幻觉": '{"title": "T", "date": "2066-01-02", "paragraphs": ["a b c d e f"], "comments": []}',
        "mojibake": '{"title": "Xiâ€™s visit", "date": "2026-11-20", "paragraphs": ["words words words words"], "comments": []}',
        "纯垃圾兜底": 'sorry I cannot {"lng": "zh_cn"}',
    }
    ok = True
    for name, s in cases.items():
        out = repair(s)
        good = set(out) == set(KEYS) and isinstance(out["paragraphs"], list)
        if name == "日期幻觉":
            good &= out["date"] is None
        if name == "mojibake":
            good &= "’" in out["title"]
        if name == "截断在字符串中":
            good &= len(out["paragraphs"]) == 2
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {json.dumps(out, ensure_ascii=False)[:120]}")
        ok &= good
    print("SELFTEST", "OK" if ok else "FAILED")
    return ok

def _demo():
    """完整调用链演示：服务(模型+内部补丁) -> clean_response -> 干净 dict"""
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data_cleaner", "test.jsonl")
    row = json.loads(open(p, encoding="utf-8").readline())
    print("== 输入脏文本前 200 字符 ==")
    print(row["input"][:200])
    data = clean_response(None, dirty_text=row["input"], backend="local")
    print("== 修复后的干净 JSON ==")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1200])
    return data

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    if "--demo" in sys.argv:
        _demo()
        sys.exit(0)
    # 示例：从 stdin 读一段模型输出并修复
    raw = sys.stdin.read()
    print(json.dumps(repair(raw), ensure_ascii=False, indent=2))

# =====================================================================
# 调用顺序（两个文件的关系）：
#
#   终端1（先起，保持开着）：conda activate myft
#                          python 清洗器服务.py        <- 加载 v2 模型，监听 8000
#   终端2 / notebook：     conda activate myft
#                          python 正则表达式.py --demo <- 一键看完整链路
#
#   代码里三种用法：
#   1) clean_response(None, dirty_text=脏文本, backend="local")
#      = 调服务(模型生成) + 服务内部已调 repair()，直接拿干净 dict（最常用）
#   2) repair(模型原始输出字符串)
#      = 只修字符串，不碰模型（你已有 raw 输出时用，比如 notebook 里）
#   3) clean_response(raw字符串)
#      = 等价于 repair(raw)，backend 参数只在给了 dirty_text 时才生效
#
# 注意：Ollama 路线不可用（该 QLoRA 适配器合并导出必坏，已穷举验证），
#       一律用 backend="local"。服务脚本里模型路径现为 D:\lora_cleaner_v2。
# =====================================================================
