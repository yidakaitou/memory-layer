# Demo output

Actual output of `python demo.py` (with the HF connection warnings removed).

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
      0.397  用户生活费紧张，这个月快见底了
      0.334  用户腰不好，严重时疼到无法弯腰

  [历史式] 我以前住过哪里
      0.483  用户长期住在乌鲁木齐
      0.309  用户生活费紧张，这个月快见底了
      0.149  用户住在上海  <- stale, down-weighted

  [无关] 今天天气怎么样
      0.366  用户生活费紧张，这个月快见底了

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
      0.454  用户长期住在乌鲁木齐
      0.141  用户住在上海  <- stale

[5] Known limitation: semantic false positives
------------------------------------------------------------------------
  bge-small (24M) score distribution:
    should match      0.401 - 0.610
    should NOT match  0.284 - 0.416   <- overlaps

  measured: "今天天气怎么样" vs "生活费紧张" = 0.416
            "我身体有什么毛病" vs "腰不好"  = 0.401

  -> No single floor separates these two sets.
  -> This is the representation ceiling of a 24M model, not a tuning issue.
  -> The fix is a second criterion (lexical cross-check / context),
     not a lower or higher threshold.

========================================================================
demo done
```

## What to look at

**Rows that demonstrate the design**

- `我住哪` → only the **current** city, at 0.940
- `我以前住哪` → the **old** city reappears at 0.141, down-weighted by 0.35 but not gone
- `我身体有什么毛病来着` → resolves to "back pain" with **zero** shared characters — this only works because retrieval has a semantic path
- `今天天气怎么样` → the one remaining false positive, documented as a known limitation rather than hidden
