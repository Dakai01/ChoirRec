"""Builds group-level features that complement sparse individual signals."""
# update date：2026-06-07

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .features import L1Summary, UserFeatures, Vocabularies


AGE_BUCKETS = [(0, 18), (19, 25), (26, 32), (33, 40), (41, 200)]


def age_to_bucket(age: int) -> int:
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if lo <= age <= hi:
            return i + 1  # 0 reserved for PAD/unknown
    return 0


GENDER_IDS = {"male": 1, "female": 2, "other": 3, "unknown": 4}
GENDER_UNKNOWN_ID = GENDER_IDS["unknown"]


@dataclass
class GroupArtifacts:
    user_codes: np.ndarray          # [U, S] hierarchical group ids
    leaf_group_ids: np.ndarray      # [U] integer leaf-group identifier
    num_leaf_groups: int
    # Raw individual static attributes -> individual tower
    age_bucket: np.ndarray          # [U]
    gender_id: np.ndarray           # [U]
    city_id: np.ndarray             # [U]
    # Group-voted static attributes -> group tower
    group_age_bucket: np.ndarray    # [U]
    group_gender_id: np.ndarray     # [U]
    group_city_id: np.ndarray       # [U]
    # Individual behaviour sequences -> individual tower
    user_l1_seq: np.ndarray         # [U, L]
    user_leaf_seq: np.ndarray       # [U, L]
    # Group behaviour sequences -> group tower
    group_l1_seq: np.ndarray        # [U, L]
    group_leaf_seq: np.ndarray      # [U, L]
    is_low_activity: np.ndarray     # [U]
    # Activity counters feeding the distillation gates
    purchase_count: np.ndarray      # [U]
    click_count: np.ndarray         # [U]
    active_days: np.ndarray         # [U]


def _summaries_to_seq(summaries: List[L1Summary], vocab: Vocabularies,
                      seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten L1/leaf summaries into padded id sequences."""
    l1_ids: List[int] = []
    leaf_ids: List[int] = []
    for s in summaries:
        l1 = vocab.l1.get(s.l1_category, 0)
        for leaf, _cnt, _price in s.leaves:
            l1_ids.append(l1)
            leaf_ids.append(vocab.leaf.get(leaf, 0))
    l1_arr = np.zeros(seq_len, dtype=np.int64)
    leaf_arr = np.zeros(seq_len, dtype=np.int64)
    for j in range(min(seq_len, len(l1_ids))):
        l1_arr[j] = l1_ids[j]
        leaf_arr[j] = leaf_ids[j]
    return l1_arr, leaf_arr


def build_group_artifacts(
    users: List[UserFeatures],
    user_codes: np.ndarray,
    vocab: Vocabularies,
    seq_len: int,
) -> GroupArtifacts:
    num_users = len(users)

    leaf_lookup: Dict[Tuple[int, ...], int] = {}
    leaf_ids = np.zeros(num_users, dtype=np.int64)
    for i, code in enumerate(user_codes):
        key = tuple(code.tolist())
        if key not in leaf_lookup:
            leaf_lookup[key] = len(leaf_lookup)
        leaf_ids[i] = leaf_lookup[key]
    num_leaf = len(leaf_lookup)

    leaf_members: Dict[int, List[int]] = {gid: [] for gid in range(num_leaf)}
    for uid, gid in enumerate(leaf_ids):
        leaf_members[int(gid)].append(uid)

    # Per-user individual sequences.
    user_l1_seq = np.zeros((num_users, seq_len), dtype=np.int64)
    user_leaf_seq = np.zeros((num_users, seq_len), dtype=np.int64)
    for i, u in enumerate(users):
        l1_arr, leaf_arr = _summaries_to_seq(u.long_purchases, vocab, seq_len)
        user_l1_seq[i] = l1_arr
        user_leaf_seq[i] = leaf_arr

    # Raw individual attributes.
    age_bucket = np.array([age_to_bucket(u.age) for u in users], dtype=np.int64)
    gender_id = np.array([GENDER_IDS.get(u.gender, GENDER_UNKNOWN_ID) for u in users], dtype=np.int64)
    city_id = np.array([vocab.city.get(u.city, 0) for u in users], dtype=np.int64)
    is_low = np.array([u.is_low_activity for u in users], dtype=np.int64)
    purchase_count = np.array([u.purchase_count for u in users], dtype=np.int64)
    click_count = np.array([u.click_count for u in users], dtype=np.int64)
    active_days = np.array([u.active_days for u in users], dtype=np.int64)

    # Group-voted attributes + group behaviour aggregation.
    group_age = np.zeros(num_users, dtype=np.int64)
    group_gender = np.zeros(num_users, dtype=np.int64)
    group_city = np.zeros(num_users, dtype=np.int64)
    group_l1_seq = np.zeros((num_users, seq_len), dtype=np.int64)
    group_leaf_seq = np.zeros((num_users, seq_len), dtype=np.int64)

    for gid, members in leaf_members.items():
        age_vote = Counter(int(age_bucket[m]) for m in members if age_bucket[m] > 0)
        gender_vote = Counter(int(gender_id[m]) for m in members
                              if gender_id[m] != GENDER_UNKNOWN_ID)
        city_vote = Counter(int(city_id[m]) for m in members if city_id[m] > 0)
        age_major = age_vote.most_common(1)[0][0] if age_vote else 0
        gender_major = gender_vote.most_common(1)[0][0] if gender_vote else GENDER_UNKNOWN_ID
        city_major = city_vote.most_common(1)[0][0] if city_vote else 0

        # Aggregate group-level purchase categories.
        l1_counter: Counter = Counter()
        leaf_counter: Counter = Counter()
        for m in members:
            for s in users[m].long_purchases:
                l1c = vocab.l1.get(s.l1_category, 0)
                l1_counter[l1c] += s.count
                for leaf, c, _price in s.leaves:
                    leaf_counter[vocab.leaf.get(leaf, 0)] += c
        top_l1 = [i for i, _ in l1_counter.most_common(seq_len)]
        top_leaf = [i for i, _ in leaf_counter.most_common(seq_len)]
        g_l1 = np.zeros(seq_len, dtype=np.int64)
        g_leaf = np.zeros(seq_len, dtype=np.int64)
        for j in range(min(seq_len, len(top_l1))):
            g_l1[j] = top_l1[j]
        for j in range(min(seq_len, len(top_leaf))):
            g_leaf[j] = top_leaf[j]

        for m in members:
            group_age[m] = age_major
            group_gender[m] = gender_major
            group_city[m] = city_major
            group_l1_seq[m] = g_l1
            group_leaf_seq[m] = g_leaf

    return GroupArtifacts(
        user_codes=user_codes.astype(np.int64),
        leaf_group_ids=leaf_ids,
        num_leaf_groups=num_leaf,
        age_bucket=age_bucket,
        gender_id=gender_id,
        city_id=city_id,
        group_age_bucket=group_age,
        group_gender_id=group_gender,
        group_city_id=group_city,
        user_l1_seq=user_l1_seq,
        user_leaf_seq=user_leaf_seq,
        group_l1_seq=group_l1_seq,
        group_leaf_seq=group_leaf_seq,
        is_low_activity=is_low,
        purchase_count=purchase_count,
        click_count=click_count,
        active_days=active_days,
    )


def vocabulary_sizes() -> Tuple[int, int]:
    """Returns (age_bucket_vocab, gender_vocab) including PAD id 0."""
    return len(AGE_BUCKETS) + 1, len(GENDER_IDS) + 1
