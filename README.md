# ChoirRec

A PyTorch implementation of **ChoirRec: Semantic User Grouping via LLMs for
Conversion Rate Prediction of Low-Activity Users** (Zhai et al.,
arXiv:2510.09393, 2025).

## What this repository is (and is NOT)

**It is** a small, runnable example that exercises the full ChoirRec pipeline
end to end, to illustrate the algorithm logic of the paper.

**It is NOT** an industrial-grade serving system. The bundled data is
**synthetic**, and the features, model, and hyperparameters are all
**simplified** — everything is for demonstrating the algorithm flow only, not
for drawing business conclusions.

## Quick start

```bash
pip install -r requirements.txt

# 1) Preprocessing: raw users.tsv -> data (group ids etc.)
python -m preprocessing.main

# 2) Recommender: train + evaluate the CVR model on the data
python -m recommender.main
```

## Project layout

```
data/                      # all data artifacts
├── users.tsv              # LLM-prompt input: plaintext user logs
├── exposure_samples.tsv   # online exposure labels (independent of features)
├── rec_samples.tsv        # individual-channel input: id-form (user, candidate, label) rows
├── vocab.json             # string -> integer id mappings (+ vocab sizes)
└── group_ids.tsv          # per-user group ids + group-voted attrs + group sequences

preprocessing/             # raw user data -> user group ids
├── config.py              # hyperparameters (RQ-KMeans, LLM, paths)
├── features.py            # plaintext loader + windowed feature engineering
├── llm.py                 # Qwen profile synthesis + embedding
├── RQKmeans.py            # RQ-KMeans hierarchical clustering
├── representation.py      # group attribute vote + group sequences
├── writers.py             # serialises the data/ artifacts
└── main.py                # one-click preprocessing entry point

recommender/               # id-form data -> CVR prediction
├── config.py              # model + training hyperparameters
├── dataset.py             # assembles per-sample tensors from data/ artifacts
├── model.py               # EmbeddingLayer + TargetAttention + dual-channel CVR tower
├── metrics.py             # AUC / GAUC (user-weighted)
└── main.py                # one-click train + evaluate entry point
```

## Profile synthesis prompt

The system prompt used for LLM profile synthesis (see
[preprocessing/llm.py](preprocessing/llm.py)):

```text
## Role
You are a data-summarization expert proficient in information extraction and
user-behavior analysis, skilled at summarizing user information and reasoning
about user profiles.

## Task
Strictly follow the rules below to produce a high-fidelity summary of the input
user information. Do not omit key information, do not fabricate content, and
ensure traceability and completeness. The resulting profile will be embedded and
clustered to form cross-activity semantic user groups, so emphasize stable,
transferable long-term preferences and filter out transient noise, allowing
semantically similar users to be grouped.

## Input Format
The input contains [Basic Info], [Transaction Behavior], and [Search Behavior].
Transaction behaviour is split into short-term (last 7 days), medium-term (last
30 days), and long-term (last 365 days) sections. Each transaction line has the
format:
  - L1 Category (Leaf Category - Purchase Count - Price Power)
where Price Power is in [0, 1]: the closer to 1, the higher the price tier the
user buys within that leaf category; 0.5 denotes the median tier.

## Output Format
[User Profile Summary]
- Core Identity & Life Stage: ...
- Interest Points: ...
- Consumption Philosophy & Decision Drivers: ...

## Module Specification
Extract explicit information and infer implicit information to output a highly
distinctive, low-ambiguity profile.
1. Core Identity & Life Stage: Based on basic info and the life scenarios
   reflected by high-frequency consumption, precisely locate the user's life
   stage (e.g., parenting family, solo young adult) and primary social role
   (e.g., household purchasing decision-maker).
2. Interest Points: Within a single paragraph, clearly integrate:
   (1) Primary: the 1-3 categories with the highest purchase counts and their
   consumption tier, representing core stable consumption;
   (2) Secondary: medium-to-low purchase-count categories and their tier,
   representing secondary consumption areas;
   (3) Recent: categories appearing in the short-term section but rare or
   absent in the long-term section, and their tier, reflecting potential new
   demand.
3. Consumption Philosophy & Decision Drivers: Combining the core profile,
   overall price-power tendency, price-power differences across categories, and
   purchase patterns, summarize the user's core consumption values (e.g.,
   quality-oriented, value-for-money first) and key decision factors (e.g.,
   price sensitivity).

## Notes
1. The summary must stay faithful to the original information, concise yet
   complete.
2. Summarize each module in a single paragraph with no internal subdivisions.
3. Even when behavioral signals are sparse, still produce a usable profile by
   reasoning from whatever explicit and attribute cues are available.
```

