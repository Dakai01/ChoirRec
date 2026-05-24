# ChoirRec

A faithful, runnable re-implementation of

> **ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction
> of Low-Activity Users.** Zhai et al., arXiv:2510.09393, 2025.

This repository reproduces the *generation – representation – modeling*
pipeline described in the paper using TensorFlow 2 and a tiny synthetic
dataset, so the code is fully self-contained and trains in seconds.

## Architecture

The implementation mirrors the paper one module per file:

| Paper Section | File | What it does |
|---|---|---|
| §4.2 Semantic Group Generation | [choirrec/llm.py](choirrec/llm.py) | Synthesises a per-user semantic profile via an LLM (Qwen3-30B-A3B) and embeds it with Qwen3-Embedding-8B. Includes a deterministic mock backend for offline runs. Embeddings are Matryoshka-truncated to 512 dims. |
| §4.2.2 Hierarchical Group Construction | [choirrec/grouping.py](choirrec/grouping.py) | RQ-KMeans with `S=3` stages and `k=256` centroids per stage. |
| §4.3 Group-aware Hierarchical Representation | [choirrec/representation.py](choirrec/representation.py) | Hierarchical group-ID codes, group attribute completion (majority vote inside the leaf group), and aggregated group-level behavioural sequences. |
| §4.4 Group-aware Multi-granularity Module | [choirrec/model.py](choirrec/model.py) | Dual-channel (Individual / Group), asymmetric injection (group → individual only), reliability-gated knowledge distillation, and adaptive logit fusion. |
| §5.1 Implementation Details | [choirrec/pipeline.py](choirrec/pipeline.py) | Adagrad with lr decay 0.01 → 0.001, batch size 1024, λ=0.005 distillation weight, AUC + GAUC evaluation. |
| §3 Problem Formulation | [choirrec/data.py](choirrec/data.py) | Synthetic CVR dataset generator that mimics Taobao's long-tail click distribution. |

## Quick start

```bash
pip install -r requirements.txt

# Tiny smoke test (~5 seconds end-to-end on CPU)
python -m choirrec.train --small

# Full synthetic run with paper-faithful defaults
python -m choirrec.train --epochs 3
```

To plug in real Qwen models, install `dashscope`, set
`DASHSCOPE_API_KEY`, and run with `--llm-backend qwen`.

## Outputs

The pipeline prints per-epoch losses and final test metrics:

```
=== Final test metrics ===
  auc:        ...   # Section 5.1: AUC
  gauc:       ...   # Section 5.1: user-weighted GAUC (primary metric)
  gauc_low:   ...   # GAUC restricted to low-activity users
  gauc_high:  ...   # GAUC restricted to high-activity users
```

## Project layout

```
choirrec/
├── __init__.py
├── config.py          # Dataclass-based configuration (paper hyperparameters)
├── data.py            # Synthetic long-tail dataset generator
├── llm.py             # LLM profile synthesis + embedding (mock | qwen)
├── grouping.py        # RQ-KMeans hierarchical clustering
├── representation.py  # Group-aware hierarchical priors
├── model.py           # ChoirRec dual-channel TF model
├── pipeline.py        # End-to-end training & evaluation
└── train.py           # CLI entry point
```

## Notes

* The synthetic data is intentionally tiny (default 1k users / 200 items)
  so the entire pipeline runs in a few seconds. Replace `data.py` with
  your real loader for production data; the rest of the modules
  (LLM → RQ-KMeans → group priors → dual-channel model) work unchanged.
* `--llm-backend mock` produces deterministic hashing-based embeddings,
  guaranteeing reproducible training on any machine. Switch to
  `--llm-backend qwen` for real LLM-driven semantic grouping.

## Reference

Dakai Zhai, Jiong Gao, Boya Du, Junwei Xu, Qijie Shen, Jialin Zhu and
Yuning Jiang. *ChoirRec: Semantic User Grouping via LLMs for Conversion
Rate Prediction of Low-Activity Users.* arXiv:2510.09393, 2025.
