# Memory Layer for LLM Agents

A working memory system for LLM agents — extraction, forgetting, retrieval, and evaluation — built and measured end-to-end.

> Most "agent memory" projects stop at vector search. This one treats memory as a **measurable pipeline**: what gets written, what gets forgotten, what gets retrieved, and how you know it's getting better.

---

## Why

An agent with no memory re-asks the same questions. An agent with *bad* memory is worse: it confuses customers, resurrects stale facts, and hallucinates details that look authoritative.

The hard part isn't storing text. It's the four decisions underneath:

- **What is worth remembering** (vs. everything that was said)
- **What should be forgotten** (and how — deletion is usually the wrong model)
- **What should surface right now** (for *this* query, not a generic similarity match)
- **How do you know any of it improved** (without fooling yourself)

This project builds all four and attaches a number to each.

---

## What it does

### 1. Extraction — deciding what enters memory

Turns raw conversation into structured memory entries, with an explicit policy for what counts as worth remembering.

The interesting failure was **over-extraction**: a model trained only on conversations *containing* facts learned that facts are always present, and then invented them from ordinary chat. Fixing this required changing the composition of training data, not the model.

### 2. Forgetting — decay instead of deletion

Entries carry a validity window. When a fact becomes outdated, it isn't deleted — it's **down-weighted** on historical queries. Old information stays retrievable if you explicitly ask about the past, but stops competing with current facts.

This mirrors how human memory behaves: forgetting is interference, not erasure. A system that can't forget is not more capable — it's noisier.

### 3. Retrieval — two paths, not one blended score

Queries reach memory through two routes:

- **Surface route** — when the query shares vocabulary with a memory, trust the literal match.
- **Semantic route** — when it shares nothing, use embeddings, with a floor to reject weak matches.

The design decision that mattered: **never blend the two scores into one number.** Blending lets a high semantic score override an obviously correct literal match, and the result is confidently wrong. Routing beats fusion.

### 4. Evaluation — regression gates, not vibes

Every claim in this project has a stored script and a log. Changes ship only if they pass:

- a structural regression suite,
- a legacy-behaviour suite,
- and a pre-registered success/failure criterion written **before** the experiment ran.

Several early "100% success" results were later retracted after stricter testing. That's the point of the gates.

---

## Results at a glance

### Reproducible here

`python bench.py` — 20 hand-written queries, 16 of them with a known correct answer.

| Area | Result |
|---|---|
| Retrieval — hit@1 | **15 / 16** |
| Retrieval — hit@3 | **15 / 16** |
| Retrieval — false positives on 4 unrelated queries | **1 / 4** |
| Retrieval, lexical path only (no torch installed) | 10 / 16 hit@3, 0 / 4 false positives |

The one false positive is documented rather than hidden: `今天天气怎么样` scores
0.383 against "budget is tight", while the weakest *correct* match scores 0.401.
A 0.018 margin is not a threshold, it is a coin flip — see `docs/demo-output.md` §5.

n=20, written by the same person who wrote the retrieval code. This is a
regression check, not a benchmark.

### Full pipeline (scripts not included)

| Area | Before | After |
|---|---|---|
| Extraction — facts invented from ordinary chat | 86% of windows | **0 / 100** |
| Extraction — recall of real facts | 45.5% | **90.9%** |
| Hallucinated specifics | 2 instances | **0, coverage unchanged** |
| Regression | — | structural 17/17, legacy 48/48 |

These need the extraction pipeline and its training data, which are not in this
repository. They are **reported, not reproducible here** — and that distinction is
worth keeping explicit, because collapsing it is how a "100%" gets published.

---

## Project timeline

**Phase 1 — Continual learning (why memory has to be external)**
Tested whether a model can accumulate new tasks without forgetting. Result: replay-based fine-tuning degrades as you scale the replay budget, while parameter isolation preserves old tasks. Conclusion that shaped everything after — *do not try to make one set of weights keep learning; keep the base frozen and put the changing part outside.*

**Phase 2 — State conditioning (how context should enter computation)**
Tested whether agent state can be injected as input rather than concatenated onto features. First attempt failed for a diagnostic reason: the state was derived from the text itself, so it carried no new information. Redesigned with a task where state is genuinely necessary; modulation clearly outperformed concatenation. Also established a hard operational rule — *gating parameters need their own learning rate, or the gate silently never opens.*

**Phase 3 — Extraction policy (what deserves to be remembered)**
Built the extraction stage, then discovered the dominant failure was not precision but **silence**: the model never learned *not* to extract. Rebuilding the training mix around "windows with no facts" fixed it.

**Phase 4 — Retrieval (finding the right memory)**
Added query-based retrieval. Two-stage design: literal matching first, embeddings only when literal fails. Discovered that score blending is actively harmful and that dual-path routing is the correct structure.

**Phase 5 — Multimodal memory (memory beyond text)**
Extended the memory layer to a model that can hear, see, and speak. Wired continuous perception (a stream, not single-shot inference) and memory injection into the multimodal path.

**Phase 6 — Boundaries (what weights can and cannot hold)**
Measured how much fine-tuning actually changes weights, and separated two routes that are often conflated: adding parameters (adapters) versus editing parameters. The distinguishing variable turned out to be **isolation**, not the act of touching weights.

---

## Design principles

**Measure before claiming.** Every number that ships in this repo has a script behind it. The numbers that don't are labelled as reported, not as reproducible.

**Distrust 100%.** A perfect score usually means the test set leaked or the task was too easy. Three of them in this project were retracted.

**Same-distribution success is not success.** A model that beats a baseline in-distribution can collapse the moment appearance changes. Test the variant, not just the average.

**Negative results are results.** Six approaches were ruled out with evidence and documented, so they don't get retried.

**A rule you don't audit is not a rule.** Methodology rules written at the end of a project tend to exempt the work that came before them. Audit backwards.

---

## Quick start

```bash
pip install torch transformers
python demo.py      # 10 memories, 8 query types, three-layer injection
python bench.py     # 20-query regression check with hit@1 / hit@3 / false positives
```

No API keys. The demo writes 10 memories, then runs 8 query types through the same
retrieval path, prints the three-layer injection block, and shows present-tense vs
historical query behaviour on a superseded fact.

If `torch`/`transformers` are missing, `demo.py` prints a loud banner and marks
every empty result as **unreliable** rather than "correctly declined". A query
that returns nothing because a module is absent is not the same as a query that
was correctly refused, and the two must never share a line of output.

`memory_core.py` is the self-contained core — store, decay, two-path retrieval, layered
injection. It deliberately does **not** include the extraction pipeline or any training
code.

---

## Repository layout

```
memory_core.py   store, decay, two-path retrieval, three-layer injection (runnable)
demo.py          end-to-end walkthrough; no API keys required
bench.py         20-query regression check; prints hit@1 / hit@3 / false positives
docs/
  design-notes.md   the seven judgement calls behind the design
  results.md        measured numbers, plus results that were retracted
  demo-output.md    captured output, including the failures
```

---

## Status

Working prototype, evaluated offline on small, controlled datasets. Open questions that remain open:

- Active retrieval — deciding *when* to query memory, not just how to answer once queried
- Contamination resistance — memory that cannot be poisoned by untrusted input
- Multi-tenant isolation — keeping one user's memory from crossing into another's
- Long-horizon consistency — behaviour stability across many sessions
- Representation ceiling — with a 24M embedding model, "weather" and "financial stress"
  score higher against each other than "body ache" and "back pain" do. No single
  similarity threshold separates them; this needs a second criterion, not a better floor.

These are the same boundaries production memory teams hit. The difference is that here
they're written down instead of discovered in an incident.

---

## License

TBD
