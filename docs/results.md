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

| Query type | Before | After |
|---|---|---|
| Short / indirect (literal only) | 0 / 6 | **5 / 6** |
| Hybrid, top-3 | — | **6 / 6** |
| Hybrid, top-1 | — | 5 / 6 |

**Negative result kept**: blending lexical and semantic scores into one number performed *worse* than lexical alone (4/6 vs 5/6) because a high semantic score could outrank an obviously correct literal match. Dual-path routing replaced fusion.

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
