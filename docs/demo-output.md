# Demo output

Actual output of `python demo.py`, semantic path enabled
(`bge-small-zh-v1.5`, 24M). The HF loading progress bar is removed.

```
========================================================================
Memory Layer demo
========================================================================

[1] Store: 10 memories written
    layers: identity=3  relation=2  task=5
    active=8  superseded=2
      superseded: 用户住在上海
      superseded: 用户的训练脚本用 fp16 混合精度

[2] Retrieval (top-3 per query)
------------------------------------------------------------------------

  [直接式] 我是什么专业的？
      0.970  用户的专业是飞机机电设备维修

  [直接式] 我住哪？
      0.940  用户长期住在乌鲁木齐

  [回指式] 上次说的那个插件
      0.403  用户自己编写了记忆插件和防上下文污染插件
      0.334  用户的训练脚本改用 bf16
      0.320  用户计划用多轮自循环训练实验跑GPU

  [回指式] 上次那个训练方案
      0.235  用户的训练脚本改用 bf16
      0.228  用户计划用多轮自循环训练实验跑GPU

  [模糊式] 我身体有什么毛病来着
      0.365  用户腰不好，严重时疼到无法弯腰

  [语义式] 我最近手头紧吗
      0.352  用户近期的可支配预算比较紧张
      0.334  用户腰不好，严重时疼到无法弯腰

  [历史式] 我以前住过哪里
      0.425  用户住在上海  <- stale, down-weighted
      0.241  用户长期住在乌鲁木齐

  [无关] 今天天气怎么样
      0.337  用户近期的可支配预算比较紧张

[3] Three-layer injection
------------------------------------------------------------------------
① 我是谁
- 用户的专业是飞机机电设备维修
- 用户长期住在乌鲁木齐

② 刚刚发生了什么
User just asked: what's wrong with my body?

③ 接下来该做什么
- 用户想学习Python，需要在自己电脑上安装环境
- 用户自己编写了记忆插件和防上下文污染插件
- 用户计划用多轮自循环训练实验跑GPU
- 用户的训练脚本改用 bf16

[4] Forgetting: present vs historical query
------------------------------------------------------------------------

  query: 我住哪
      0.940  用户长期住在乌鲁木齐

  query: 我以前住哪
      0.404  用户住在上海  <- stale
      0.227  用户长期住在乌鲁木齐

[5] Known limitation: semantic score overlap
------------------------------------------------------------------------
    应命中    0.460   我住哪  vs  用户长期住在乌鲁木齐
    应命中    0.401   我身体有什么毛病来着  vs  用户腰不好，严重时疼到无法弯腰
    应命中    0.465   我最近手头紧吗  vs  用户近期的可支配预算比较紧张
    不应命中  0.383   今天天气怎么样  vs  用户近期的可支配预算比较紧张
    不应命中  0.238   上次那个训练方案  vs  用户长期住在乌鲁木齐

  min(应命中)   = 0.401
  max(不应命中) = 0.383
  间距          = +0.018

  -> 间距不足 0.05。在这个宽度上挑阈值，等于对着这几个样本过拟合；
     而且 [2] 里已经出现了一个真实的 false positive（今天天气怎么样）。
  -> 这是 24M 模型的表征上限，不是调阈值能解决的；
     解法是第二条判据（字面交叉验证 / 上下文），不是调高或调低 floor。

========================================================================
demo done
```

## What to look at

**Design demonstrated**

- `我住哪` → only the **current** city, 0.940. The superseded entry is not in
  the pool at all.
- `我以前住过哪里` → the **old** city is back on top (0.425), the current one
  drops to 0.241. Asking about the past flips which entry is noise.
- `我身体有什么毛病来着` → resolves to "back pain" with **zero** shared
  characters. Only the semantic path can do that.

**Failures kept in the output**

- `今天天气怎么样` → still returns a memory (0.337). The one surviving false
  positive, and it sits 0.018 below the weakest true match — too close to
  separate with a threshold (§5).

**Two bugs this output is here to prove are fixed**

1. **Historical queries used to rank the present first.** The old code applied
   `HISTORICAL_PENALTY` to superseded entries unconditionally, so
   `我以前住过哪里` returned Urumqi (0.483) ahead of Shanghai (0.149). The
   penalty now depends on query intent: when the user says "以前", the stale
   entry is the answer and the *current* entry gets down-weighted instead.
2. **Latin/numeric queries returned nothing at all.** `_content_chars` kept
   only CJK characters, so a query like `bf16` produced an empty unit set and
   `search()` returned `[]` before scoring. Now ASCII runs are matched as whole
   tokens (`#bf16`), which also stops `fp16` from matching `bf16` on shared
   letters.
