"""Hierarchical Group Construction via RQ-KMeans (Paper Sec. 4.2.2).

Residual-Quantized KMeans builds a hierarchy of group IDs by repeatedly
clustering the *residuals* of the previous stage. With S stages and k
centroids per stage we obtain k^S leaf groups, but each user is identified
by a length-S code (g_1, g_2, ..., g_S) that lives on a coarse-to-fine
hierarchy. This is exactly the structure exploited by the
hierarchical-group-ID-fusion module in Sec. 4.3.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from .config import GroupingConfig


@dataclass
class RQKMeansResult:
    codes: np.ndarray            # [N, num_stages] integer cluster ids
    centroids: List[np.ndarray]  # per-stage centroid matrices
    residuals: np.ndarray        # final residual after the last stage


class RQKMeans:
    """Residual-quantised KMeans used to build hierarchical user groups."""

    def __init__(self, cfg: GroupingConfig):
        self.cfg = cfg
        self._fitted: List[KMeans] = []

    def fit(self, embeddings: np.ndarray) -> RQKMeansResult:
        x = embeddings.astype(np.float32).copy()
        codes = np.zeros((x.shape[0], self.cfg.num_stages), dtype=np.int32)
        centroids: List[np.ndarray] = []
        residual = x.copy()
        # Adaptive k: never request more clusters than we have samples.
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
            centroids.append(kmeans.cluster_centers_.astype(np.float32))
            residual = residual - kmeans.cluster_centers_[assigns]
            self._fitted.append(kmeans)

        return RQKMeansResult(codes=codes, centroids=centroids, residuals=residual)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RQKMeans must be fitted before transform().")
        x = embeddings.astype(np.float32).copy()
        codes = np.zeros((x.shape[0], len(self._fitted)), dtype=np.int32)
        residual = x.copy()
        for stage, kmeans in enumerate(self._fitted):
            assigns = kmeans.predict(residual)
            codes[:, stage] = assigns
            residual = residual - kmeans.cluster_centers_[assigns]
        return codes
