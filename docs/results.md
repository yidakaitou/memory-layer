# Results

Summary of what was measured. Each row corresponds to an experiment that has a stored script and log.

---

## Extraction

| Metric | Baseline | Final |
|---|---|---|
| Windows wrongly extracting facts (real chat, no facts) | 86% | **0 / 100** |
| Fact exactness (real chat + injected fact) | — | **0.99** |
| Recall of real facts (human gold, 10 windows) | 45.5% | **63.6%** → **90.9%** with implicit-fact clause |
| Hallucinated specifics | 2 classes | **0 after filter, coverage unchanged** |

**Negative result kept**: adding the implicit-fact clause reached 90.9% but introduced specificity hallucinations — facts inferred from phenomena rather than stated. The clause alone is not a net win; it only works paired with the filter.

---

## Retrieval

Reproducible with `python bench.py` (20 hand-written queries, 16 with a known
correct answer):

| Category | hit@1 | hit@3 |
|---|---|---|
| Direct | 3 / 4 | 3 / 4 |
| Indirect / anaphoric | 4 / 4 | 4 / 4 |
| Fuzzy / semantic | 4 / 4 | 4 / 4 |
| Historical | 4 / 4 | 4 / 4 |
| Unrelated (should return nothing) | — | **1 false positive out of 4** |
| **Total** | **15 / 16** | **15 / 16** |

With the semantic path disabled (no torch installed): **10 / 16 hit@3, 0 / 4 false
positives**. Every fuzzy/semantic case fails lexically, which is the point — that
path only exists for queries with no surface overlap.

**Known failure**: `今天天气怎么样` returns "budget is tight" at raw 0.383 while
the weakest correct match sits at 0.401. Margin 0.018. There is no threshold in
that gap that isn't fitted to these five samples.

Earlier 6-query set (superseded by `bench.py`, kept for history):

| Query type | Before | After |
|---|---|---|
| Short / indirect (literal only) | 0 / 6 | **5 / 6** |
| Hybrid, top-3 | — | **6 / 6** |
| Hybrid, top-1 | — | 5 / 6 |

**Negative result kept**: blending lexical and semantic scores into one number performed *worse* than lexical alone (4/6 vs 5/6) because a high semantic score could outrank an obviously correct literal match. Dual-path routing replaced fusion.

Two bugs found while building this benchmark:

- **Historical queries ranked the present first.** `HISTORICAL_PENALTY` was applied
  to superseded entries regardless of query intent, so `我以前住过哪里` put the
  current city ahead of the old one. Fixed: when the query asks about the past, the
  stale entry is the answer and the current entry is down-weighted instead.
- **Latin/numeric queries returned nothing.** `_content_chars` kept only CJK
  characters, so `bf16` produced an empty unit set and `search()` returned `[]`
  before scoring anything. Fixed: ASCII runs are matched as whole tokens, which also
  stops `fp16` from matching `bf16`.

---

## Regression

| Suite | Result |
|---|---|
| Structural (injection layers) | 17 / 17 |
| Legacy memory behaviour | 48 / 48 |

No change shipped without both suites green.

---

## Retracted results

Recorded so they don't get re-attempted:

| Claim | Why retracted |
|---|---|
| Model A beats model B by 16.7 points on extraction recall | Single-sample artifact; re-test with 3 samples gave 3/3 for B |
| Voting aggregation reaches 100% recall | Only holds when failures are independent; real data gives the same 83% for all aggregation methods |
| Policy wording raises recall | The target sentence in the test contained policy keywords |

---

## Known gaps

- **Active retrieval** — deciding whether to query, never measured
- **Contamination resistance** — poisoning untested
- **Multi-tenant isolation** — single-user store only
- **Long-horizon consistency** — behaviour across many sessions untested
- **Cost** — retrieval latency at hundreds of memories not benchmarked
