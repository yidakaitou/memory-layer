# -*- coding: utf-8 -*-
"""Self-contained memory core for the demo.

Deliberately minimal: this file contains only the parts that make the demo
runnable — storage, decay, two-path retrieval, and layered injection.

It does NOT contain the extraction pipeline, the LLM policies, or any
training code. Those live in the private implementation.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZONES = ("identity", "relation", "task", "plan")

# 失效记忆在历史查询下的降权系数。不是 0——历史可查，但不与现时竞争。
HISTORICAL_PENALTY = 0.35

_HISTORICAL_CUES = (
    "以前", "之前", "原来", "曾经", "过去", "旧", "原先", "上一次", "当时",
    "以前是", "原来是", "以前住", "以前在",
)

_STOP = set("的了是我你他她它在有和就都也很吗呢吧啊？！，。、；：哪什么怎么为什么多少几时过里")


def _content_chars(s: str) -> set[str]:
    """去掉停用字后的字符集合（字面检索用）。"""
    return {c for c in (s or "") if c not in _STOP and "\u4e00" <= c <= "\u9fff"}


def _is_historical(q: str) -> bool:
    return any(c in (q or "") for c in _HISTORICAL_CUES)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    claim: str
    kind: str = "belief"
    zone: str = "relation"
    confidence: float = 0.6
    status: str = "active"          # active / superseded / closed
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    valid_from: float = field(default_factory=time.time)
    valid_to: float | None = None
    slot: str = ""
    value: str = ""
    evidence: list = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


# ---------------------------------------------------------------------------
# Embedder (lazy, optional)
# ---------------------------------------------------------------------------

class _Embedder:
    """bge-small-zh with the official query instruction. Silent fallback if absent.

    Loaded through transformers directly (no sentence-transformers dependency):
    CLS pooling + L2 normalisation, matching the model card's usage.
    """

    MODEL_ID = "BAAI/bge-small-zh-v1.5"
    INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, list[float]] = {}
        self._failed = False

    def available(self) -> bool:
        return self._ensure()

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModel.from_pretrained(self.MODEL_ID).eval()
            return True
        except Exception:
            self._failed = True
            return False

    def embed(self, text: str, as_query: bool = False) -> list[float] | None:
        if not text or not text.strip():
            return None
        # instruction goes on the query side only
        key = (self.INSTRUCTION if as_query else "") + text.strip()
        if key in self._cache:
            return self._cache[key]
        if not self._ensure():
            return None
        try:
            import torch
            batch = self._tokenizer([key], padding=True, truncation=True,
                                    max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = self._model(**batch)
            vec = out.last_hidden_state[:, 0]          # CLS
            vec = torch.nn.functional.normalize(vec, dim=-1)[0].tolist()
            self._cache[key] = vec
            return vec
        except Exception:
            return None

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))


embedder = _Embedder()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self, path: str | None = None):
        self.path = path
        self.nodes: list[Node] = []

    # -- write -------------------------------------------------------------

    def add(self, claim: str, kind: str = "belief", zone: str = "relation",
            confidence: float = 0.6, slot: str = "", value: str = "") -> Node:
        """Add a memory. If `slot` is given and holds a different value, the old
        entry is superseded (not deleted) — this is what makes historical queries
        possible later."""
        if slot and value:
            for n in self.nodes:
                if (n.status == "active" and n.slot == slot
                        and n.value and n.value != value):
                    n.status = "superseded"
                    n.valid_to = time.time()
                    n.updated = time.time()
        node = Node(claim=claim, kind=kind, zone=zone,
                    confidence=confidence, slot=slot, value=value)
        self.nodes.append(node)
        return node

    # -- scoring -----------------------------------------------------------

    def _importance(self, n: Node) -> float:
        """Confidence × mild recency decay. Kept simple on purpose."""
        age_days = max(0.0, (time.time() - n.updated) / 86400)
        recency = math.exp(-age_days / 90.0)
        return 0.4 + 0.6 * n.confidence * recency

    # -- retrieval ---------------------------------------------------------

    def search(self, query: str, k: int = 3, min_score: float = 0.08,
               semantic_floor: float = 0.35) -> list[tuple[float, Node]]:
        """Two-path retrieval.

        Path selection (the design decision that mattered):
          - lexical overlap exists  -> rank on lexical coverage only
          - no overlap              -> use semantic similarity, gated by a floor

        Never blend the two scores into one number.
        """
        if not query.strip():
            return []
        q_chars = _content_chars(query)
        if not q_chars:
            return []

        want_historical = _is_historical(query)
        q_vec = embedder.embed(query, as_query=True)
        min_match = 1 if len(q_chars) <= 2 else 2

        scored: list[tuple[float, Node]] = []
        for n in self.nodes:
            # superseded entries enter the pool only for historical queries
            if n.status != "active":
                if not (want_historical and n.status == "superseded"):
                    continue
            if n.valid_to and not want_historical:
                continue

            inter = q_chars & _content_chars(n.claim)
            lexical = (len(inter) / len(q_chars)) if len(inter) >= min_match else 0.0

            semantic = 0.0
            if q_vec is not None:
                nv = embedder.embed(n.claim)
                if nv is not None:
                    semantic = max(0.0, embedder.cosine(q_vec, nv))

            if lexical > 0:
                base = lexical
            elif semantic >= semantic_floor:
                base = semantic
            else:
                continue

            score = base * self._importance(n)
            if n.valid_to:
                score *= HISTORICAL_PENALTY
            if score >= min_score:
                scored.append((round(score, 4), n))

        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    # -- injection ---------------------------------------------------------

    def build_injection(self, recent: str = "", budget_chars: int = 1200) -> str:
        """Three layers: identity / recent / plan."""
        blocks: list[str] = []
        used = 0

        def take(nodes: list[Node], title: str) -> None:
            nonlocal used
            lines: list[str] = []
            for n in nodes:
                line = f"- {n.claim}"
                if used + len(line) > budget_chars:
                    break
                lines.append(line)
                used += len(line)
            if lines:
                blocks.append(title + "\n" + "\n".join(lines))

        alive = [n for n in self.nodes if n.status == "active"]
        take([n for n in alive if n.zone == "identity"], "① 我是谁")
        if recent:
            blocks.append("② 刚刚发生了什么\n" + recent)
        take([n for n in alive if n.zone in ("task", "plan")], "③ 接下来该做什么")

        return "\n\n".join(blocks)
