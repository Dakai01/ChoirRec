"""Group-aware Hierarchical Representation (Paper Sec. 4.3).

Once RQ-KMeans gives every user a length-S group code, we materialise three
group-level priors that complement the sparse individual signals:

    1) Hierarchical group ID embeddings (Sec. 4.3.1)
    2) Group attribute completion (Sec. 4.3.2)
       -> majority-vote each static attribute within the leaf group, then
          impute it back into users whose attribute is missing.
    3) Group-level behavioral sequence (Sec. 4.3.3)
       -> aggregate the most-frequent items / categories purchased inside
          the leaf group as a robust group-level history.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import DataConfig
from .data import AGE_BUCKETS, GENDERS, LEVELS, UserRecord


@dataclass
class GroupArtifacts:
    """Per-user tensors derived from group priors + the user's own data."""
    user_codes: np.ndarray          # [U, S] hierarchical group ids
    completed_attrs: np.ndarray     # [U, 3] (age, gender, level) imputed
    user_click_seq: np.ndarray      # [U, L] padded item ids (-> 0 reserved)
    user_cat_seq: np.ndarray        # [U, L] padded category ids
    group_click_seq: np.ndarray     # [U, L] aggregated leaf-group item ids
    group_cat_seq: np.ndarray       # [U, L] aggregated leaf-group cat ids
    is_low_activity: np.ndarray     # [U]
    leaf_group_ids: np.ndarray      # [U] integer leaf-group identifier
    num_leaf_groups: int


def _majority(values: List[int], default: int) -> int:
    valid = [v for v in values if v >= 0]
    if not valid:
        return default
    return Counter(valid).most_common(1)[0][0]


def build_group_artifacts(
    users: List[UserRecord],
    user_codes: np.ndarray,
    data_cfg: DataConfig,
) -> GroupArtifacts:
    seq_len = data_cfg.seq_max_len
    num_users = len(users)

    # ------------------------------------------------------------------
    # Pad / truncate raw user sequences. We reserve id=0 as the PAD token
    # by shifting all real ids by +1 in the embedding lookup at model time.
    # Here we simply store them; the model will offset.
    # ------------------------------------------------------------------
    user_click = np.zeros((num_users, seq_len), dtype=np.int32)
    user_cat = np.zeros((num_users, seq_len), dtype=np.int32)
    for i, u in enumerate(users):
        for j, (it, c) in enumerate(zip(u.click_seq[:seq_len], u.cat_click_seq[:seq_len])):
            user_click[i, j] = it + 1
            user_cat[i, j] = c + 1

    # ------------------------------------------------------------------
    # Group bucketing by leaf code, i.e. the full S-tuple of stages.
    # ------------------------------------------------------------------
    leaf_keys = [tuple(code.tolist()) for code in user_codes]
    leaf_lookup: Dict[Tuple[int, ...], int] = {}
    leaf_ids = np.zeros(num_users, dtype=np.int32)
    for i, key in enumerate(leaf_keys):
        if key not in leaf_lookup:
            leaf_lookup[key] = len(leaf_lookup)
        leaf_ids[i] = leaf_lookup[key]
    num_leaf = len(leaf_lookup)

    # Aggregate per-leaf statistics
    leaf_members: Dict[int, List[int]] = {gid: [] for gid in range(num_leaf)}
    for uid, gid in enumerate(leaf_ids):
        leaf_members[int(gid)].append(uid)

    completed_attrs = np.zeros((num_users, 3), dtype=np.int32)
    group_click = np.zeros((num_users, seq_len), dtype=np.int32)
    group_cat = np.zeros((num_users, seq_len), dtype=np.int32)
    is_low = np.array([u.is_low_activity for u in users], dtype=np.int32)

    # Per-group voted attribute fallbacks
    for gid, members in leaf_members.items():
        ages = [users[m].age_bucket for m in members]
        genders = [users[m].gender for m in members]
        levels = [users[m].level for m in members]
        age_default = _majority(ages, 0)
        gender_default = _majority(genders, len(GENDERS) - 1)  # "unknown" by default
        level_default = _majority(levels, 0)

        # Aggregate purchase items / cats inside the group, top-k => group seq
        item_counter: Counter = Counter()
        cat_counter: Counter = Counter()
        for m in members:
            item_counter.update(users[m].purchase_seq)
            cat_counter.update(users[m].cat_purchase_seq)
        # Fall back to clicks if the group has too few purchases.
        if sum(item_counter.values()) < seq_len:
            for m in members:
                item_counter.update(users[m].click_seq)
                cat_counter.update(users[m].cat_click_seq)
        top_items = [it for it, _ in item_counter.most_common(seq_len)]
        top_cats = [c for c, _ in cat_counter.most_common(seq_len)]

        # Pad to fixed seq_len (id 0 is PAD; shift real ids +1).
        item_arr = np.zeros(seq_len, dtype=np.int32)
        cat_arr = np.zeros(seq_len, dtype=np.int32)
        for j, (it, c) in enumerate(zip(top_items, top_cats)):
            item_arr[j] = it + 1
            cat_arr[j] = c + 1

        for m in members:
            u = users[m]
            completed_attrs[m, 0] = u.age_bucket if u.age_bucket >= 0 else age_default
            completed_attrs[m, 1] = u.gender if u.gender >= 0 else gender_default
            completed_attrs[m, 2] = u.level if u.level >= 0 else level_default
            group_click[m] = item_arr
            group_cat[m] = cat_arr

    return GroupArtifacts(
        user_codes=user_codes.astype(np.int32),
        completed_attrs=completed_attrs,
        user_click_seq=user_click,
        user_cat_seq=user_cat,
        group_click_seq=group_click,
        group_cat_seq=group_cat,
        is_low_activity=is_low,
        leaf_group_ids=leaf_ids,
        num_leaf_groups=num_leaf,
    )


def vocabulary_sizes() -> Tuple[int, int, int]:
    """Returns (age_vocab, gender_vocab, level_vocab)."""
    return len(AGE_BUCKETS), len(GENDERS), len(LEVELS)
