"""Synthetic dataset generator for ChoirRec.

The paper (Section 1, Fig. 1a) reports a long-tail click distribution where
roughly the top 18% of users contribute about 90% of all clicks. We replicate
that skew with a small synthetic dataset so the rest of the pipeline can be
tested end-to-end without touching real Taobao logs.

Each user record contains:
    - Static profile attributes (age bucket, gender, level), some of which
      are intentionally missing for low-activity users (paper Sec. 3:
      low-activity users only retain ~65% attribute coverage).
    - A click sequence (used as behavioral signal).
    - A purchase sequence (used as label / S^buy_u in Eq. 1).

A sample for training corresponds to a (user, item) pair y in {0, 1}:
positive samples are sampled from purchase records, and negatives are
sampled from clicks-without-purchase plus uniform random items.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

from .config import DataConfig


GENDERS = ["male", "female", "unknown"]
AGE_BUCKETS = ["18-", "19-25", "26-32", "33-40", "41+"]
LEVELS = ["bronze", "silver", "gold", "platinum"]


@dataclass
class UserRecord:
    user_id: int
    age_bucket: int       # index into AGE_BUCKETS, -1 if missing
    gender: int           # index into GENDERS, -1 if missing
    level: int            # index into LEVELS, -1 if missing
    is_low_activity: int  # 1 if user is in the long-tail segment
    click_seq: List[int]
    cat_click_seq: List[int]
    purchase_seq: List[int]
    cat_purchase_seq: List[int]


@dataclass
class SyntheticDataset:
    users: List[UserRecord]
    item_to_category: Dict[int, int]
    samples: List[Tuple[int, int, int]]  # (user_id, item_id, label)


def _mask_attribute(rng: random.Random, value: int, drop_prob: float) -> int:
    return -1 if rng.random() < drop_prob else value


def generate_synthetic_dataset(cfg: DataConfig) -> SyntheticDataset:
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    item_to_category: Dict[int, int] = {
        item_id: rng.randrange(cfg.num_categories)
        for item_id in range(cfg.num_items)
    }

    num_high = max(1, int(cfg.num_users * cfg.high_activity_ratio))
    high_users = set(rng.sample(range(cfg.num_users), num_high))

    # Latent user-category preference vectors give the dataset some structure
    # so semantic grouping has a meaningful signal to recover.
    user_pref = np_rng.normal(size=(cfg.num_users, cfg.num_categories))

    users: List[UserRecord] = []
    samples: List[Tuple[int, int, int]] = []

    for uid in range(cfg.num_users):
        is_low = 0 if uid in high_users else 1
        # Long-tail: high-activity ~ avg_high; low-activity ~ avg_low
        if is_low:
            num_clicks = max(1, int(np_rng.poisson(cfg.avg_low_clicks)))
            attr_drop = 0.35  # ~65% retention rate -> 35% drop
        else:
            num_clicks = max(1, int(np_rng.poisson(cfg.avg_high_clicks)))
            attr_drop = 0.05

        # Sample clicked items proportional to user preference
        logits = user_pref[uid]
        cat_probs = np.exp(logits - logits.max())
        cat_probs = cat_probs / cat_probs.sum()

        click_seq: List[int] = []
        cat_click_seq: List[int] = []
        for _ in range(num_clicks):
            cat = int(np_rng.choice(cfg.num_categories, p=cat_probs))
            candidates = [it for it, c in item_to_category.items() if c == cat]
            if not candidates:
                continue
            item = rng.choice(candidates)
            click_seq.append(item)
            cat_click_seq.append(cat)

        purchase_seq: List[int] = []
        cat_purchase_seq: List[int] = []
        for it, c in zip(click_seq, cat_click_seq):
            if rng.random() < cfg.purchase_rate:
                purchase_seq.append(it)
                cat_purchase_seq.append(c)

        user = UserRecord(
            user_id=uid,
            age_bucket=_mask_attribute(rng, rng.randrange(len(AGE_BUCKETS)), attr_drop),
            gender=_mask_attribute(rng, rng.randrange(len(GENDERS)), attr_drop),
            level=_mask_attribute(rng, rng.randrange(len(LEVELS)), attr_drop),
            is_low_activity=is_low,
            click_seq=click_seq[: cfg.seq_max_len],
            cat_click_seq=cat_click_seq[: cfg.seq_max_len],
            purchase_seq=purchase_seq[: cfg.seq_max_len],
            cat_purchase_seq=cat_purchase_seq[: cfg.seq_max_len],
        )
        users.append(user)

        positives = set(purchase_seq)
        for it in positives:
            samples.append((uid, it, 1))
        clicked_no_buy = [it for it in click_seq if it not in positives]
        # Negative ratio approximately 1:3 for stable training signals.
        num_negs = max(len(positives) * 3, 1)
        neg_pool = clicked_no_buy + [
            int(np_rng.integers(0, cfg.num_items)) for _ in range(num_negs)
        ]
        rng.shuffle(neg_pool)
        for it in neg_pool[:num_negs]:
            if it in positives:
                continue
            samples.append((uid, it, 0))

    rng.shuffle(samples)
    return SyntheticDataset(users=users, item_to_category=item_to_category, samples=samples)


def save_dataset(dataset: SyntheticDataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "users.jsonl"), "w", encoding="utf-8") as f:
        for u in dataset.users:
            f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "item_categories.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.item_to_category, f)
    with open(os.path.join(out_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for uid, iid, y in dataset.samples:
            f.write(json.dumps({"user_id": uid, "item_id": iid, "label": y}) + "\n")
