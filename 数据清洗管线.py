# -*- coding: utf-8 -*-
"""
数据清洗器训练集构建管线（英文散乱网页文本 -> 干净 JSON）

五个源：
  xinhua / chinadaily / sixthtone : 整页散乱可见文本(导航/广告/脚注混杂) -> 标题+日期+正文段落
  hackernews                      : 去标签扁平文本(评论串)            -> 标题+故事+评论列表
  stackexchange                   : 去标签扁平文本(问题+回答)         -> 标题+问题+最佳/其他回答

输出 schema（所有样本统一）：
  {"title": str, "date": str|null, "paragraphs": [str],
   "comments": [{"author": str|null, "text": str}]}
  无评论的源 comments=[]；新闻日期 ISO 或 null。

每个文档产出 1 条真实脏文本样本；约一半文档额外产出 1 条"注入噪声"增强样本
（mojibake/零宽字符/多余空行/重复行/行尾空格，可逆或无害噪声）。
train/test 按文档切分（同一文档的增强样本不跨集合），90/10。

用法（myft 环境）：
  python 数据清洗管线.py            # 爬取+生成+校验
  python 数据清洗管线.py --validate # 只校验已有产物
"""
import os
import re
import json
import time
import random
import hashlib
import argparse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, Comment

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw_cache")
OUT = os.path.join(BASE, "data_cleaner")
UA = {"User-Agent": "Mozilla/5.0 (compatible; dataset-research/1.0; personal study)"}
DELAY = 1.0
random.seed(42)

os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ---------------- 通用工具 ----------------

def fetch(url, binary=False):
    """带磁盘缓存的礼貌抓取"""
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    ext = ".bin" if binary else ".html"
    path = os.path.join(RAW, key + ext)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read() if binary else f.read().decode("utf-8", "replace")
    time.sleep(DELAY)
    r = requests.get(url, headers=UA, timeout=25)
    r.raise_for_status()
    data = r.content
    with open(path, "wb") as f:
        f.write(data)
    return data if binary else data.decode("utf-8", "replace")


def visible_text(soup):
    """整页可见文本：去掉 script/style/noscript/注释，块级标签换行"""
    soup = BeautifulSoup(str(soup), "html.parser")
    for t in soup(["script", "style", "noscript", "template"]):
        t.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for t in soup.find_all(["p", "div", "br", "li", "h1", "h2", "h3", "h4",
                            "tr", "section", "article", "header", "footer"]):
        t.append("\n")
    txt = soup.get_text(" ")
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r" ?\n ?", "\n", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


def flat_text(html):
    """去标签扁平文本：模拟'复制粘贴整页'的散乱效果"""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    txt = soup.get_text(" ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def clean_paras(lines):
    out = []
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= 40 and len(ln.split()) >= 6:
            out.append(ln)
    # 去重保序
    seen, res = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res


MOJIBAKE = {"â€™": "’", "â€˜": "‘", "â€œ": "“", "â€": "”",
            "â€”": "—", "â€“": "–", "â€¦": "…", "Â ": " ", "Ã©": "é"}


def inject_noise(text):
    ops = []
    lines = text.split("\n")
    # 1 多余空行
    if random.random() < 0.8:
        ops.append("extra_blank_lines")
        lines = [x for ln in lines for x in (ln, "")]
    # 2 行尾空格
    if random.random() < 0.7:
        ops.append("trailing_spaces")
        lines = [ln + "  " if ln else ln for ln in lines]
    # 3 重复行
    if random.random() < 0.5 and len(lines) > 6:
        ops.append("duplicated_lines")
        i = random.randrange(2, len(lines) - 2)
        lines.insert(i, lines[i])
    text = "\n".join(lines)
    # 4 mojibake
    if random.random() < 0.6:
        ops.append("mojibake")
        for bad, good in MOJIBAKE.items():
            text = text.replace(good, bad)
    # 5 零宽字符
    if random.random() < 0.5:
        ops.append("zero_width_chars")
        text = text.replace(" ", chr(0x200B) + " ", 6)
    return text, ops


def make_record(rid, source, url, license_note, input_text, output_obj,
                noise_ops=None, split="train"):
    return {
        "id": rid,
        "source": source,
        "url": url,
        "license": license_note,
        "lang": "en",
        "noise_ops": noise_ops or [],
        "input": input_text,
        "output": output_obj,
        "split": split,
    }

# ---------------- 新闻站 ----------------

def discover_news():
    urls = {"xinhua": [], "chinadaily": [], "sixthtone": []}
    home = fetch("https://english.news.cn/")
    urls["xinhua"] = list(dict.fromkeys(re.findall(
        r'https://english\.news\.cn/2026\d{4}/[0-9a-f]{32}/c\.html', home)))[:26]
    home = fetch("https://www.chinadaily.com.cn/")
    cd = re.findall(r'(?:https?:)?//www\.chinadaily\.com\.cn/a/2026\d{2}/\d{2}/WS[0-9a-f]+\.html', home)
    urls["chinadaily"] = list(dict.fromkeys("https:" + u if u.startswith("//") else u for u in cd))[:26]
    home = fetch("https://www.sixthtone.com/")
    st = re.findall(r'/news/\d+/[^"?]+', home)
    urls["sixthtone"] = list(dict.fromkeys("https://www.sixthtone.com" + u for u in st))[:26]
    return urls


def parse_xinhua(soup, url):
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else None
    if not title:
        og = soup.find("meta", property="og:title")
        title = og["content"].strip() if og and og.get("content") else None
    date = None
    m = re.search(r"/(2026\d{4})/", url)
    if m:
        d = m.group(1)
        date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    body = soup.find(id="detail") or soup.find(class_="detail") or \
        soup.find(id="content") or soup.find(class_="article")
    if body is None:
        cands = [d for d in soup.find_all("div")
                 if len(d.find_all("p", recursive=False)) >= 4]
        body = max(cands, key=lambda d: len(d.get_text())) if cands else None
    paras = clean_paras([p.get_text(" ", strip=True) for p in body.find_all("p")]) if body else []
    return title, date, paras


def parse_chinadaily(soup, url):
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else None
    date = None
    m = re.search(r"/a/(2026\d{2})/(\d{2})/", url)
    if m:
        date = f"{m.group(1)[:4]}-{m.group(1)[4:]}-{m.group(2)}"
    body = soup.find(id="Content") or soup.find(class_="article") or \
        soup.find("div", class_="main_art")
    if body is None:
        cands = [d for d in soup.find_all("div")
                 if len(d.find_all("p", recursive=False)) >= 4]
        body = max(cands, key=lambda d: len(d.get_text())) if cands else None
    paras = clean_paras([p.get_text(" ", strip=True) for p in body.find_all("p")]) if body else []
    return title, date, paras


def parse_sixthtone(soup, url):
    # JS 渲染壳：标题在 og:title，正文在含最多直接 <p> 的无类名 div
    title, date, paras = None, None, []
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        h = soup.find(["h1", "h2"])
        title = h.get_text(" ", strip=True) if h else None
    cands = [d for d in soup.find_all("div")
             if len(d.find_all("p", recursive=False)) >= 4]
    body = max(cands, key=lambda d: len(d.get_text())) if cands else None
    if body:
        paras = clean_paras([p.get_text(" ", strip=True) for p in body.find_all("p")])
    return title, date, paras


NEWS_PARSE = {"xinhua": parse_xinhua, "chinadaily": parse_chinadaily,
              "sixthtone": parse_sixthtone}
NEWS_LICENSE = {
    "xinhua": "Xinhua News Agency (c) news content; research/ML-training use, attribution kept",
    "chinadaily": "China Daily (c) news content; research/ML-training use, attribution kept",
    "sixthtone": "Sixth Tone (c) news content; research/ML-training use, attribution kept",
}

# ---------------- API 源 ----------------

def build_hackernews(n=60):
    ids = json.loads(fetch("https://hacker-news.firebaseio.com/v0/topstories.json"))[:n]
    docs = []
    for i in ids:
        try:
            item = json.loads(fetch(f"https://hacker-news.firebaseio.com/v0/item/{i}.json"))
        except Exception:
            continue
        if not item or item.get("type") != "story" or not item.get("title"):
            continue
        story_html = item.get("text", "")
        comments = []
        for kid in (item.get("kids") or [])[:12]:
            try:
                c = json.loads(fetch(f"https://hacker-news.firebaseio.com/v0/item/{kid}.json"))
            except Exception:
                continue
            if c and c.get("type") == "comment" and c.get("text") and not c.get("deleted"):
                comments.append({"author": c.get("by"), "text": c["text"]})
            if len(comments) >= 8:
                break
        if not comments and not story_html:
            continue
        docs.append({"id": i, "title": item["title"], "url": item.get("url", ""),
                     "by": item.get("by"), "time": item.get("time"),
                     "story_html": story_html, "comments": comments})
    return docs


def hn_record_input(doc):
    parts = [doc["title"]]
    if doc["url"]:
        parts.append(doc["url"])
    parts.append(f"by {doc['by']}")
    if doc["story_html"]:
        parts.append(flat_text(doc["story_html"]))
    for c in doc["comments"]:
        parts.append(f"{c['author']}: {flat_text(c['text'])}")
    return "\n".join(parts)


def hn_record_output(doc):
    story_lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat_text(doc["story_html"]))
                   if s.strip() and not re.match(r"^(Related|See also|Previously)\b", s.strip())]
    paras = clean_paras(story_lines) if doc["story_html"] else []
    date = datetime.fromtimestamp(doc["time"], tz=timezone.utc).strftime("%Y-%m-%d") if doc.get("time") else None
    comments = [{"author": c["author"], "text": re.sub(r"\s+", " ", flat_text(c["text"])).strip()}
                for c in doc["comments"]]
    return {"title": doc["title"], "date": date, "paragraphs": paras, "comments": comments}


def build_stackexchange(n=60):
    q = json.loads(fetch(
        "https://api.stackexchange.com/2.3/questions?pagesize=%d&order=desc&sort=activity"
        "&site=stackoverflow&filter=withbody" % n))
    ids = [str(it["question_id"]) for it in q.get("items", [])]
    if not ids:
        return []
    ans = json.loads(fetch(
        "https://api.stackexchange.com/2.3/questions/%s/answers?site=stackoverflow"
        "&filter=withbody&order=desc&sort=votes&pagesize=100" % ";".join(ids)))
    by_q = {}
    for a in ans.get("items", []):
        by_q.setdefault(a["question_id"], []).append(a)
    docs = []
    for it in q.get("items", []):
        alist = by_q.get(it["question_id"], [])
        if not alist:
            continue
        docs.append({"qid": it["question_id"], "title": it["title"],
                     "link": it.get("link", ""), "body": it.get("body", ""),
                     "answers": sorted(alist, key=lambda a: -a.get("score", 0))[:3]})
    return docs


def se_record_input(doc):
    parts = [flat_text(doc["title"]), flat_text(doc["body"])]
    for a in doc["answers"]:
        parts.append(flat_text(a.get("body", "")))
    return "\n".join(parts)


def se_record_output(doc):
    comments = []
    for a in doc["answers"]:
        comments.append({"author": str(a.get("owner", {}).get("display_name") or None),
                         "text": re.sub(r"\s+", " ", flat_text(a.get("body", ""))).strip()})
    return {"title": re.sub(r"\s+", " ", flat_text(doc["title"])).strip(),
            "date": None,
            "paragraphs": clean_paras([flat_text(doc["body"])]),
            "comments": comments}

# ---------------- 主流程 ----------------

def build():
    records = []
    stats = {}
    # 新闻
    urls = discover_news()
    for src, lst in urls.items():
        ok = 0
        for u in lst:
            try:
                html = fetch(u)
            except Exception:
                continue
            soup = BeautifulSoup(html, "html.parser")
            title, date, paras = NEWS_PARSE[src](soup, u)
            if not title or len(paras) < 3:
                continue
            whole = visible_text(soup)
            if len(whole) < 800 or len(whole) > 60000:
                continue
            out = {"title": re.sub(r"\s+", " ", title).strip(), "date": date,
                   "paragraphs": paras, "comments": []}
            rid = f"{src}-{ok:03d}"
            records.append(make_record(rid, src, u, NEWS_LICENSE[src], whole, out))
            if random.random() < 0.5:
                noisy, ops = inject_noise(whole)
                records.append(make_record(rid + "-aug", src, u, NEWS_LICENSE[src],
                                           noisy, out, noise_ops=ops))
            ok += 1
        stats[src] = ok
    # HN
    hn = build_hackernews(60)
    for k, doc in enumerate(hn):
        out = hn_record_output(doc)
        if not out["comments"] and not out["paragraphs"]:
            continue
        inp = hn_record_input(doc)
        if len(inp) < 300:
            continue
        rid = f"hn-{k:03d}"
        records.append(make_record(rid, "hackernews",
                                   f"https://news.ycombinator.com/item?id={doc['id']}",
                                   "CC BY 4.0 (Hacker News API content)", inp, out))
        if random.random() < 0.5:
            noisy, ops = inject_noise(inp)
            records.append(make_record(rid + "-aug", "hackernews",
                                       f"https://news.ycombinator.com/item?id={doc['id']}",
                                       "CC BY 4.0 (Hacker News API content)", noisy, out, ops))
    stats["hackernews"] = len(hn)
    # SE
    se = build_stackexchange(60)
    for k, doc in enumerate(se):
        out = se_record_output(doc)
        if not out["comments"]:
            continue
        inp = se_record_input(doc)
        rid = f"se-{k:03d}"
        records.append(make_record(rid, "stackexchange", doc["link"],
                                   "CC BY-SA 4.0 (Stack Exchange)", inp, out))
        if random.random() < 0.5:
            noisy, ops = inject_noise(inp)
            records.append(make_record(rid + "-aug", "stackexchange", doc["link"],
                                       "CC BY-SA 4.0 (Stack Exchange)", noisy, out, ops))
    stats["stackexchange"] = len(se)

    # 文档级切分
    doc_ids = sorted({r["id"].replace("-aug", "") for r in records})
    random.shuffle(doc_ids)
    n_test = max(1, int(len(doc_ids) * 0.1))
    test_docs = set(doc_ids[:n_test])
    for r in records:
        r["split"] = "test" if r["id"].replace("-aug", "") in test_docs else "train"

    # 去重（input 完全相同）
    seen, uniq = set(), []
    for r in records:
        h = hashlib.sha256(r["input"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(r)
    records = uniq

    for split in ("train", "test"):
        path = os.path.join(OUT, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"docs_per_source": stats, "records": len(records),
                   "train": sum(r["split"] == "train" for r in records),
                   "test": sum(r["split"] == "test" for r in records)}, f,
                  ensure_ascii=False, indent=2)
    print("BUILD_DONE", json.dumps(stats), "records:", len(records))
    return records


def validate():
    problems = []
    all_recs = []
    for split in ("train", "test"):
        with open(os.path.join(OUT, f"{split}.jsonl"), encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                all_recs.append(r)
                o = r["output"]
                if not isinstance(o.get("title"), str) or not o["title"].strip():
                    problems.append((r["id"], "title missing"))
                if not isinstance(o.get("paragraphs"), list) or \
                   not isinstance(o.get("comments"), list):
                    problems.append((r["id"], "schema type"))
                for p in o["paragraphs"]:
                    if len(p.split()) < 6:
                        problems.append((r["id"], "short paragraph"))
                if not o["paragraphs"] and not o["comments"]:
                    problems.append((r["id"], "empty content"))
                if len(r["input"]) < 300:
                    problems.append((r["id"], "input too short"))
                if r["split"] != split:
                    problems.append((r["id"], "split mismatch"))
    tr = {r["id"].replace("-aug", "") for r in all_recs if r["split"] == "train"}
    te = {r["id"].replace("-aug", "") for r in all_recs if r["split"] == "test"}
    if tr & te:
        problems.append(("LEAK", str(tr & te)))
    lens = sorted(len(r["input"]) for r in all_recs)
    report = {
        "total": len(all_recs),
        "train": sum(r["split"] == "train" for r in all_recs),
        "test": sum(r["split"] == "test" for r in all_recs),
        "augmented": sum(bool(r["noise_ops"]) for r in all_recs),
        "per_source": {s: sum(r["source"] == s for r in all_recs)
                       for s in sorted({r["source"] for r in all_recs})},
        "input_chars_min_p50_p90_max": [lens[0], lens[len(lens)//2],
                                        lens[int(len(lens)*0.9)], lens[-1]],
        "problems": problems[:20],
        "problem_count": len(problems),
    }
    with open(os.path.join(OUT, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        build()
        validate()
