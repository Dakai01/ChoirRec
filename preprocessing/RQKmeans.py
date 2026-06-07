"""Hierarchical user grouping via Residual-Quantized KMeans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .config import GroupingConfig


@dataclass
class RQKMeansResult:
    codes: np.ndarray  # [N, num_stages] integer cluster ids


class RQKMeans:
    """Residual-quantised KMeans used to build hierarchical user groups."""

    def __init__(self, cfg: GroupingConfig):
        self.cfg = cfg

    def fit(self, embeddings: np.ndarray) -> RQKMeansResult:
        x = embeddings.astype(np.float32).copy()
        codes = np.zeros((x.shape[0], self.cfg.num_stages), dtype=np.int32)
        residual = x.copy()
        k = min(self.cfg.centroids_per_stage, max(2, x.shape[0] // 2))

        for stage in range(self.cfg.num_stages):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.cfg.seed + stage,
                n_init=10,
                max_iter=self.cfg.kmeans_max_iter,
            )
            assigns = kmeans.fit_predict(residual)
            codes[:, stage] = assigns
            residual = residual - kmeans.cluster_centers_[assigns]

        return RQKMeansResult(codes=codes)
