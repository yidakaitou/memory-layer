# -*- coding: utf-8 -*-
"""Memory Layer demo — runs the full chain: store -> forget -> retrieve -> inject.

Self-contained. No API keys, no private models.
The memories below are pre-supplied (standing in for an extraction pipeline);
everything after that is the real implementation in memory_core.py.

Run:
    pip install sentence-transformers
    python demo.py
"""
from __future__ import annotations

import sys

from memory_core import MemoryStore


def build_store() -> MemoryStore:
    s = MemoryStore()

    # identity
    s.add("用户的专业是飞机机电设备维修", zone="identity", confidence=0.95)
    # a fact that changes: Shanghai -> Urumqi (old one is superseded, not deleted)
    s.add("用户住在上海", zone="identity", confidence=0.7, slot="城市", value="上海")
    s.add("用户长期住在乌鲁木齐", zone="identity", confidence=0.9, slot="城市", value="乌鲁木齐")

    # relation
    s.add("用户腰不好，严重时疼到无法弯腰", zone="relation", confidence=0.85)
    s.add("用户生活费紧张，这个月快见底了", zone="relation", confidence=0.8)

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
    ("无关",   "今天天气怎么样"),
]


def main() -> int:
    print("=" * 72)
    print("Memory Layer demo")
    print("=" * 72)

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
            print("      (no match - correctly declined)")
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
        for score, node in store.search(q, k=3):
            mark = "  <- stale" if node.status == "superseded" else ""
            print(f"      {score:.3f}  {node.claim}{mark}")

    print(f"\n[5] Known limitation: semantic false positives")
    print("-" * 72)
    print("  bge-small (24M) score distribution:")
    print("    should match      0.401 - 0.610")
    print("    should NOT match  0.284 - 0.416   <- overlaps")
    print()
    print('  measured: "今天天气怎么样" vs "生活费紧张" = 0.416')
    print('            "我身体有什么毛病" vs "腰不好"  = 0.401')
    print()
    print("  -> No single floor separates these two sets.")
    print("  -> This is the representation ceiling of a 24M model, not a tuning issue.")
    print("  -> The fix is a second criterion (lexical cross-check / context),")
    print("     not a lower or higher threshold.")

    print("\n" + "=" * 72)
    print("demo done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
