"""Ranking metrics: AUC and user-grouped AUC (GAUC)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


def gauc(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> float:
    """User-weighted AUC."""
    buckets: Dict[int, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for u, y, s in zip(user_ids, labels, scores):
        ys, ss = buckets[int(u)]
        ys.append(float(y))
        ss.append(float(s))
    weighted_sum, weight_total = 0.0, 0.0
    for _u, (ys, ss) in buckets.items():
        if len(set(ys)) < 2:
            continue
        try:
            a = roc_auc_score(ys, ss)
        except ValueError:
            continue
        weighted_sum += a * len(ys)
        weight_total += len(ys)
    return weighted_sum / weight_total if weight_total > 0 else float("nan")


def evaluate(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
             is_low: np.ndarray) -> Dict[str, float]:
    low_mask = is_low == 1
    high_mask = ~low_mask
    return {
        "auc": auc(labels, scores),
        "gauc": gauc(user_ids, labels, scores),
        "gauc_low": gauc(user_ids[low_mask], labels[low_mask], scores[low_mask])
        if low_mask.any() else float("nan"),
        "gauc_high": gauc(user_ids[high_mask], labels[high_mask], scores[high_mask])
        if high_mask.any() else float("nan"),
    }
