"""Serialises the id-form preprocessing artifacts into the ``data/`` folder."""
# update date：2026-06-07

from __future__ import annotations

import json
from typing import List

import numpy as np

from .features import Sample, UserRecord, Vocabularies
from .representation import (
    GENDER_IDS,
    GroupArtifacts,
    vocabulary_sizes,
)


def _seq_to_str(seq: np.ndarray) -> str:
    ids = [int(x) for x in seq.tolist()]
    return ",".join(str(i) for i in ids) if ids else "0"


def write_vocab(vocab: Vocabularies, path: str) -> None:
    age_v, gender_v = vocabulary_sizes()
    payload = {
        "l1": vocab.l1,
        "leaf": vocab.leaf,
        "city": vocab.city,
        "item": vocab.item,
        "user": vocab.user,
        "gender_ids": GENDER_IDS,
        "sizes": {
            "item": vocab.size_item(),
            "l1": vocab.size_l1(),
            "leaf": vocab.size_leaf(),
            "city": vocab.size_city(),
            "user": vocab.size_user(),
            "age_bucket": age_v,
            "gender": gender_v,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_rec_samples(
    samples: List[Sample],
    records: List[UserRecord],
    vocab: Vocabularies,
    artifacts: GroupArtifacts,
    path: str,
) -> None:
    header = [
        "user_id", "label",
        "item_id", "item_l1", "item_leaf",
        "age_bucket", "gender_id", "city_id", "is_low_activity",
        "purchase_count", "click_count", "active_days",
        "user_l1_seq", "user_leaf_seq",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for s in samples:
            r = records[s.user_idx]
            idx = s.user_idx
            row = [
                str(r.user_id),
                str(s.label),
                str(vocab.item.get(s.item_name, 0)),
                str(vocab.l1.get(s.l1_category, 0)),
                str(vocab.leaf.get(s.leaf_category, 0)),
                str(int(artifacts.age_bucket[idx])),
                str(int(artifacts.gender_id[idx])),
                str(int(artifacts.city_id[idx])),
                str(int(artifacts.is_low_activity[idx])),
                str(int(artifacts.purchase_count[idx])),
                str(int(artifacts.click_count[idx])),
                str(int(artifacts.active_days[idx])),
                _seq_to_str(artifacts.user_l1_seq[idx]),
                _seq_to_str(artifacts.user_leaf_seq[idx]),
            ]
            f.write("\t".join(row) + "\n")


def write_group_ids(
    records: List[UserRecord],
    artifacts: GroupArtifacts,
    path: str,
) -> None:
    """Write per-user group codes + group-voted attributes + group sequences."""
    num_stages = artifacts.user_codes.shape[1]
    header = (
        ["user_id"]
        + [f"group_code_{s}" for s in range(num_stages)]
        + ["leaf_group_id",
           "group_age_bucket", "group_gender_id", "group_city_id",
           "group_l1_seq", "group_leaf_seq"]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for idx, r in enumerate(records):
            codes = [str(int(c)) for c in artifacts.user_codes[idx].tolist()]
            row = (
                [str(r.user_id)]
                + codes
                + [
                    str(int(artifacts.leaf_group_ids[idx])),
                    str(int(artifacts.group_age_bucket[idx])),
                    str(int(artifacts.group_gender_id[idx])),
                    str(int(artifacts.group_city_id[idx])),
                    _seq_to_str(artifacts.group_l1_seq[idx]),
                    _seq_to_str(artifacts.group_leaf_seq[idx]),
                ]
            )
            f.write("\t".join(row) + "\n")
