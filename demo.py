# -*- coding: utf-8 -*-
"""Memory Layer demo — runs the full chain: store -> forget -> retrieve -> inject.

Self-contained. No API keys, no private models.
The memories below are pre-supplied (standing in for an extraction pipeline);
everything after that is the real implementation in memory_core.py.

Run:
    pip install torch transformers
    python demo.py

If torch/transformers are missing (or the bge-small-zh-v1.5 weights cannot be
fetched), the semantic path is disabled. That case is printed loudly on purpose:
a query that returns nothing because a module is missing is NOT the same thing
as a query that was correctly declined, and the two must never be confused.
"""
from __future__ import annotations

import sys

from memory_core import MemoryStore, embedder


def build_store() -> MemoryStore:
    s = MemoryStore()

    # identity
    s.add("用户的专业是飞机机电设备维修", zone="identity", confidence=0.95)
    # a fact that changes: Shanghai -> Urumqi (old one is superseded, not deleted)
    s.add("用户住在上海", zone="identity", confidence=0.7, slot="城市", value="上海")
    s.add("用户长期住在乌鲁木齐", zone="identity", confidence=0.9, slot="城市", value="乌鲁木齐")

    # relation
    s.add("用户腰不好，严重时疼到无法弯腰", zone="relation", confidence=0.85)
    s.add("用户近期的可支配预算比较紧张", zone="relation", confidence=0.8)

    # task
    s.add("用户想学习Python，需要在自己电脑上安装环境", zone="task", confidence=0.85)
    s.add("用户自己编写了记忆插件和防上下文污染插件", zone="task", confidence=0.9)
    s.add("用户计划用多轮自循环训练实验跑GPU", zone="task", confidence=0.85)

    # a technical decision that changes: fp16 -> bf16
    s.add("用户的训练脚本用 fp16 混合精度", zone="task", confidence=0.7,
          slot="精度", value="fp16")
    s.add("用户的训练脚本改用 bf16", zone="task", confidence=0.9,
          slot="精度", value="bf16")
    return s


QUERIES = [
    ("直接式", "我是什么专业的？"),
    ("直接式", "我住哪？"),
    ("回指式", "上次说的那个插件"),
    ("回指式", "上次那个训练方案"),
    ("模糊式", "我身体有什么毛病来着"),
    ("语义式", "我最近手头紧吗"),
    ("历史式", "我以前住过哪里"),
    ("无关", "今天天气怎么样"),
]


def main() -> int:
    print("=" * 72)
    print("Memory Layer demo")
    print("=" * 72)

    sem_ok = embedder.available()
    if not sem_ok:
        print()
        print("!! bge-small-zh-v1.5 未加载（缺 torch/transformers，或无法联网取权重）")
        print("   语义路径已禁用：下面的结果只含字面检索。")
        print("   模糊/语义/历史查询会落空——那是模块没装，不是系统判断它不该返回。")
        print("   确认无误后加 --lexical-only 可以让本脚本以这个状态正常退出。")

    store = build_store()
    active = [n for n in store.nodes if n.status == "active"]
    superseded = [n for n in store.nodes if n.status == "superseded"]

    print(f"\n[1] Store: {len(store.nodes)} memories written")
    by_zone: dict[str, int] = {}
    for n in store.nodes:
        by_zone[n.zone] = by_zone.get(n.zone, 0) + 1
    print("    layers: " + "  ".join(f"{k}={v}" for k, v in sorted(by_zone.items())))
    print(f"    active={len(active)}  superseded={len(superseded)}")
    for n in superseded:
        print(f"      superseded: {n.claim}")

    print(f"\n[2] Retrieval (top-3 per query)")
    print("-" * 72)
    for label, q in QUERIES:
        hits = store.search(q, k=3)
        print(f"\n  [{label}] {q}")
        if not hits:
            suffix = "" if sem_ok else "   <- 语义路径未启用，此行不可信"
            print(f"      (no match){suffix}")
            continue
        for score, node in hits:
            mark = "  <- stale, down-weighted" if node.status == "superseded" else ""
            print(f"      {score:.3f}  {node.claim}{mark}")

    print(f"\n[3] Three-layer injection")
    print("-" * 72)
    print(store.build_injection(recent="User just asked: what's wrong with my body?"))

    print(f"\n[4] Forgetting: present vs historical query")
    print("-" * 72)
    for q in ["我住哪", "我以前住哪"]:
        print(f"\n  query: {q}")
        hits = store.search(q, k=3)
        if not hits:
            print("      (no match)")
            continue
        for score, node in hits:
            mark = "  <- stale" if node.status == "superseded" else ""
            print(f"      {score:.3f}  {node.claim}{mark}")

    print(f"\n[5] Known limitation: semantic score overlap")
    print("-" * 72)
    if not sem_ok:
        print("  (semantic path disabled — nothing to measure)")
        print("  re-run with torch/transformers installed to populate this section.")
    else:
        pairs = [
            (True, "我住哪", "用户长期住在乌鲁木齐"),
            (True, "我身体有什么毛病来着", "用户腰不好，严重时疼到无法弯腰"),
            (True, "我最近手头紧吗", "用户近期的可支配预算比较紧张"),
            (False, "今天天气怎么样", "用户近期的可支配预算比较紧张"),
            (False, "上次那个训练方案", "用户长期住在乌鲁木齐"),
        ]
        pos: list[float] = []
        neg: list[float] = []
        for should_match, q, claim in pairs:
            score = embedder.cosine(embedder.embed(q, as_query=True),
                                    embedder.embed(claim))
            tag = "应命中  " if should_match else "不应命中"
            print(f"    {tag}  {score:.3f}   {q}  vs  {claim}")
            (pos if should_match else neg).append(score)

        print()
        margin = min(pos) - max(neg)
        print(f"  min(应命中)   = {min(pos):.3f}")
        print(f"  max(不应命中) = {max(neg):.3f}")
        print(f"  间距          = {margin:+.3f}")
        print()
        if margin <= 0:
            print("  -> 区间重叠：不存在一个统一阈值能分开这两组。")
        elif margin < 0.05:
            print("  -> 间距不足 0.05。在这个宽度上挑阈值，等于对着这几个样本过拟合；")
            print("     而且 [2] 里已经出现了一个真实的 false positive（今天天气怎么样）。")
        else:
            print("  -> 本组样本可分，但 n=5，不足以支撑任何阈值选择。")
        print("  -> 这是 24M 模型的表征上限，不是调阈值能解决的；")
        print("     解法是第二条判据（字面交叉验证 / 上下文），不是调高或调低 floor。")

    print("\n" + "=" * 72)
    print("demo done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
