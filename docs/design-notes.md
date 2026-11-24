# Design Notes

Short notes on the decisions behind the memory layer. No implementation detail.

---

## 1. Why memory has to live outside the weights

The first investigation was whether a model can keep learning new tasks without forgetting the old ones.

**What we found**: replay-based fine-tuning has diminishing returns — doubling the replay budget only halves the forgetting, and costs accuracy on the *new* task. Parameter isolation preserves old tasks, but its guarantee is nearly tautological: if the old parameters are never touched, the old task can't degrade.

**What actually matters**: the variable isn't *whether* you touch the weights, it's *whether what you add is isolated*. Adapters win because they add capacity beside the base; editing wins nothing because it modifies the base directly.

**Consequence**: the memory layer isn't a workaround for a weak model. It's the part of the system that must change while the base stays stable.

---

## 2. Why forgetting is down-weighting, not deletion

Deletion assumes information is either true or false. Real memory is more like interference: outdated facts don't disappear, they stop winning.

So entries carry a validity window. When a fact expires, retrieval still returns it for historical queries but at a penalty; for present-tense queries it stops competing.

**Why this matters operationally**: hard deletion makes "why did you tell me that?" unanswerable. Down-weighting keeps the audit trail while keeping the present clean.

---

## 3. Why retrieval is two paths, not one fused score

The tempting design is `score = a × lexical + b × semantic`. It fails in a specific, dangerous way: a confident embedding score can override a literal match that is obviously correct, and the system then answers confidently wrong.

**The rule we arrived at**:

- lexical overlap exists → rank purely on lexical signal
- no lexical overlap → use semantic, with a floor

Routing beats fusion because the two signals answer different questions. Fusion lets the wrong signal win.

---

## 4. Why extraction quality is a data-composition problem

The model's first failure wasn't precision — it was *never staying silent*. Trained only on conversations containing facts, it learned that a fact is always there to be found, and manufactured them from ordinary chat.

No amount of additional training fixes this, because the training distribution itself asserts the wrong rule. The fix is compositional: include windows with genuinely nothing to remember, plus windows where a fact is injected into otherwise ordinary chat (to prevent the shortcut "ordinary prose means no facts").

**General rule extracted from this**: when more data and more epochs don't help, inspect the data's *composition*, not its size.

---

## 5. Why we distrust perfect scores

Three results hit 100% during this project and were later retracted:

- a policy experiment, where the target sentence leaked policy keywords
- a synthetic-data extraction benchmark, where train and test were drawn from the same generator
- an aggregation method that only works when failures are independent — which they weren't on real data

**The working rule**: a 100% result triggers an audit of the test set before it triggers a celebration.

---

## 6. Same-distribution success is the most common false positive

A model trained and evaluated on one appearance can beat a copy baseline by a wide margin in-distribution, then collapse when appearance changes at all.

This isn't a quirk of one experiment — it's the default failure mode of any model evaluated only on its own distribution. The countermeasure is cheap: hold out an *appearance variant*, not just a data split.

---

## 7. Active retrieval is the unsolved half

Everything here answers: *given a query, what should surface?*

Nothing here answers: *should the agent have queried at all?* That's a different problem with different metrics — retrieval utility, calibration of the decision to query, and the cost of a wrong decision in each direction.

The evaluation harness for this is well-defined in the literature and is not built here yet. It is the next piece.
