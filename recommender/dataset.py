"""Assembles per-sample feature tensors from the preprocessing artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def _parse_seq(text: str, seq_len: int) -> np.ndarray:
    arr = np.zeros(seq_len, dtype=np.int64)
    if not text:
        return arr
    ids = [int(x) for x in text.split(",") if x != ""]
    for j in range(min(seq_len, len(ids))):
        arr[j] = ids[j]
    return arr


@dataclass
class VocabSizes:
    item: int
    l1: int
    leaf: int
    city: int
    user: int
    age_bucket: int
    gender: int


def load_vocab_sizes(path: str) -> VocabSizes:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    s = payload["sizes"]
    return VocabSizes(
        item=s["item"], l1=s["l1"], leaf=s["leaf"], city=s["city"],
        user=s["user"], age_bucket=s["age_bucket"], gender=s["gender"],
    )


@dataclass
class GroupTable:
    """Per-user group-channel features keyed by user_id."""
    codes: Dict[int, np.ndarray]
    group_age_bucket: Dict[int, int]
    group_gender_id: Dict[int, int]
    group_city_id: Dict[int, int]
    group_l1_seq: Dict[int, np.ndarray]
    group_leaf_seq: Dict[int, np.ndarray]


def load_group_table(path: str, num_stages: int, seq_len: int) -> GroupTable:
    codes: Dict[int, np.ndarray] = {}
    group_age: Dict[int, int] = {}
    group_gender: Dict[int, int] = {}
    group_city: Dict[int, int] = {}
    group_l1_seq: Dict[int, np.ndarray] = {}
    group_leaf_seq: Dict[int, np.ndarray] = {}

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cells = line.split("\t")
            uid = int(cells[col["user_id"]])
            codes[uid] = np.array(
                [int(cells[col[f"group_code_{s}"]]) for s in range(num_stages)],
                dtype=np.int64,
            )
            group_age[uid] = int(cells[col["group_age_bucket"]])
            group_gender[uid] = int(cells[col["group_gender_id"]])
            group_city[uid] = int(cells[col["group_city_id"]])
            group_l1_seq[uid] = _parse_seq(cells[col["group_l1_seq"]], seq_len)
            group_leaf_seq[uid] = _parse_seq(cells[col["group_leaf_seq"]], seq_len)

    return GroupTable(
        codes=codes, group_age_bucket=group_age, group_gender_id=group_gender,
        group_city_id=group_city, group_l1_seq=group_l1_seq,
        group_leaf_seq=group_leaf_seq,
    )


FEATURE_KEYS = [
    # Individual channel
    "user_id", "item_id", "item_l1", "item_leaf",
    "user_l1_seq", "user_leaf_seq",
    "age_bucket", "gender_id", "city_id",
    # Group channel
    "user_codes", "group_age_bucket", "group_gender_id", "group_city_id",
    "group_l1_seq", "group_leaf_seq",
    # Gates (hard + soft) over user-activity signals
    "purchase_count", "click_count", "active_days", "is_low_activity",
]


@dataclass
class Dataset:
    feats: Dict[str, np.ndarray]
    labels: np.ndarray
    user_ids: np.ndarray
    is_low: np.ndarray


def build_dataset(cfg) -> Dataset:
    """Assemble per-sample tensors from the preprocessing artifacts."""
    seq_len = cfg.model.seq_max_len
    group_table = load_group_table(cfg.group_ids_file, cfg.num_stages, seq_len)

    rows: List[List[str]] = []
    with open(cfg.rec_samples_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rows.append(line.split("\t"))

    n = len(rows)
    feats = {
        "user_id": np.zeros(n, dtype=np.int64),
        "item_id": np.zeros(n, dtype=np.int64),
        "item_l1": np.zeros(n, dtype=np.int64),
        "item_leaf": np.zeros(n, dtype=np.int64),
        "user_l1_seq": np.zeros((n, seq_len), dtype=np.int64),
        "user_leaf_seq": np.zeros((n, seq_len), dtype=np.int64),
        "age_bucket": np.zeros(n, dtype=np.int64),
        "gender_id": np.zeros(n, dtype=np.int64),
        "city_id": np.zeros(n, dtype=np.int64),
        "user_codes": np.zeros((n, cfg.num_stages), dtype=np.int64),
        "group_age_bucket": np.zeros(n, dtype=np.int64),
        "group_gender_id": np.zeros(n, dtype=np.int64),
        "group_city_id": np.zeros(n, dtype=np.int64),
        "group_l1_seq": np.zeros((n, seq_len), dtype=np.int64),
        "group_leaf_seq": np.zeros((n, seq_len), dtype=np.int64),
        "purchase_count": np.zeros(n, dtype=np.float32),
        "click_count": np.zeros(n, dtype=np.float32),
        "active_days": np.zeros(n, dtype=np.float32),
        "is_low_activity": np.zeros(n, dtype=np.int64),
    }
    labels = np.zeros(n, dtype=np.float32)
    user_ids = np.zeros(n, dtype=np.int64)
    is_low = np.zeros(n, dtype=np.int64)

    user_vocab = _load_user_vocab(cfg.vocab_file)

    for i, cells in enumerate(rows):
        uid = int(cells[col["user_id"]])
        # Individual channel: raw attributes + own behaviour.
        feats["user_id"][i] = user_vocab.get(str(uid), 0)
        feats["item_id"][i] = int(cells[col["item_id"]])
        feats["item_l1"][i] = int(cells[col["item_l1"]])
        feats["item_leaf"][i] = int(cells[col["item_leaf"]])
        feats["user_l1_seq"][i] = _parse_seq(cells[col["user_l1_seq"]], seq_len)
        feats["user_leaf_seq"][i] = _parse_seq(cells[col["user_leaf_seq"]], seq_len)
        feats["age_bucket"][i] = int(cells[col["age_bucket"]])
        feats["gender_id"][i] = int(cells[col["gender_id"]])
        feats["city_id"][i] = int(cells[col["city_id"]])
        # Group channel: codes + group-voted attributes + group behaviour.
        feats["user_codes"][i] = group_table.codes[uid]
        feats["group_age_bucket"][i] = group_table.group_age_bucket[uid]
        feats["group_gender_id"][i] = group_table.group_gender_id[uid]
        feats["group_city_id"][i] = group_table.group_city_id[uid]
        feats["group_l1_seq"][i] = group_table.group_l1_seq[uid]
        feats["group_leaf_seq"][i] = group_table.group_leaf_seq[uid]
        # Gate signals.
        feats["purchase_count"][i] = float(cells[col["purchase_count"]])
        feats["click_count"][i] = float(cells[col["click_count"]])
        feats["active_days"][i] = float(cells[col["active_days"]])
        feats["is_low_activity"][i] = int(cells[col["is_low_activity"]])

        labels[i] = float(cells[col["label"]])
        user_ids[i] = uid
        is_low[i] = int(cells[col["is_low_activity"]])

    return Dataset(feats=feats, labels=labels, user_ids=user_ids, is_low=is_low)


def _load_user_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("user", {})
