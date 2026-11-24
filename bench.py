# -*- coding: utf-8 -*-
"""Minimal, hand-labelled retrieval benchmark for the memory layer.

20 queries, each tied to the claim fragment that should surface in top-3.
The point is regression detection and honest reporting, not a headline number:
n=20, and the queries were written by the same person who wrote the retrieval
code. It can catch breakage. It cannot estimate real-world hit rate.

Run:
    python bench.py
"""
from __future__ import annotations

from demo import build_store
from memory_core import embedder

# (category, query, expected fragment; None means "nothing should be returned")
CASES: list[tuple[str, str, str | None]] = [
    # 直接式：query 和记忆的用词基本重合
    ("直接", "我是什么专业的？", "飞机机电"),
    ("直接", "我住哪？", "乌鲁木齐"),
    ("直接", "我现在住哪", "乌鲁木齐"),
    ("直接", "我在哪个城市", "乌鲁木齐"),

    # 回指式：靠"上次那个 / 之前提过的"指代
    ("回指", "上次说的那个插件", "记忆插件"),
    ("回指", "上次那个训练方案", "训练脚本"),
    ("回指", "那个防上下文污染的东西", "防上下文污染"),
    ("回指", "我之前提过的插件", "记忆插件"),

    # 模糊 / 语义：字面几乎无重叠
    ("模糊/语义", "我身体有什么毛病来着", "腰"),
    ("模糊/语义", "我最近手头紧吗", "预算"),
    ("模糊/语义", "我哪儿不舒服", "腰"),
    ("模糊/语义", "我钱够不够花", "预算"),

    # 历史式：应命中被 supersede 的旧值
    ("历史", "我以前住哪", "上海"),
    ("历史", "我以前住在哪个城市", "上海"),
    ("历史", "我之前用的什么精度", "fp16"),
    ("历史", "原来那个精度设置", "fp16"),

    # 无关：不该召回任何东西
    ("无关", "今天天气怎么样", None),
    ("无关", "明天几点开会", None),
    ("无关", "你叫什么名字", None),
    ("无关", "推荐一部电影", None),
]


def run(k: int = 3) -> int:
    store = build_store()
    stats: dict[str, list[int]] = {}   # category -> [n, hit1, hit3, fp]
    rows: list[str] = []

    for cat, q, want in CASES:
        hits = [n.claim for _, n in store.search(q, k=k)]
        rec = stats.setdefault(cat, [0, 0, 0, 0])
        rec[0] += 1
        if want is None:
            rec[3] += 1 if hits else 0
            ok1 = ok3 = n_a = 0
            mark = "FP " if hits else "ok "
        else:
            ok1 = int(bool(hits) and want in hits[0])
            ok3 = int(any(want in c for c in hits))
            rec[1] += ok1
            rec[2] += ok3
            mark = "   " if ok3 else "MIS"
        label = "—" if want is None else f"hit@1={ok1} hit@3={ok3}"
        rows.append(f"  [{mark}] {cat:<9} {q:<16} want={want!r:<12} {label}")

    print("=" * 78)
    print(f"memory-layer bench   k={k}   semantic_path={embedder.available()}")
    print("=" * 78)
    print("\n".join(rows))

    print("\n" + "-" * 78)
    print(f"{'category':<10}{'n':>4}{'hit@1':>9}{'hit@3':>9}{'false_pos':>12}")
    tot = [0, 0, 0, 0]
    for cat in ("直接", "回指", "模糊/语义", "历史", "无关"):
        if cat not in stats:
            continue
        n, h1, h3, fp = stats[cat]
        print(f"{cat:<10}{n:>4}{h1:>6}/{n:<2}{h3:>6}/{n:<2}{fp:>8}/{n:<3}")
        for i, v in enumerate((n, h1, h3, fp)):
            tot[i] += v
    n, h1, h3, fp = tot
    rel = n - stats.get("无关", [0, 0, 0, 0])[0]
    print("-" * 78)
    print(f"{'total':<10}{n:>4}{h1:>6}/{rel:<2}{h3:>6}/{rel:<2}{fp:>8}/{n - rel:<3}")
    print("\nnote: n=20, hand-written by the author. sanity check, not a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
