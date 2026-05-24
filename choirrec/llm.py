"""LLM interface for ChoirRec.

This module provides two backends to materialize the LLM-based steps from
the paper (Sec. 4.2.1 Semantic Profile Synthesis via LLM):

1. ``mock``  - A deterministic local mock that derives a textual profile from
   user statistics and produces an embedding by hashing tokens into a dense
   vector. This guarantees end-to-end reproducibility on any machine.
2. ``qwen``  - Calls the real Qwen models via the dashscope SDK:
       chat: Qwen3-30B-A3B  (paper) for profile synthesis
       embed: Qwen3-Embedding-8B for vectorisation, with Matryoshka
              truncation to 512 dims (paper Sec. 5.1).

The public API is the ``LLMService`` class with two methods:
    synthesize_profile(user) -> str
    embed_profiles(profiles) -> np.ndarray of shape [N, truncated_dim]
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List

import numpy as np

from .config import LLMConfig
from .data import AGE_BUCKETS, GENDERS, LEVELS, UserRecord


def _attr_text(idx: int, vocab: List[str]) -> str:
    return vocab[idx] if 0 <= idx < len(vocab) else "unknown"


def render_user_text(user: UserRecord, item_to_category) -> str:
    """Build the prompt payload describing a user."""
    age = _attr_text(user.age_bucket, AGE_BUCKETS)
    gender = _attr_text(user.gender, GENDERS)
    level = _attr_text(user.level, LEVELS)
    activity = "low-activity" if user.is_low_activity else "high-activity"
    cat_freq: dict = {}
    for c in user.cat_click_seq:
        cat_freq[c] = cat_freq.get(c, 0) + 1
    top_cats = sorted(cat_freq.items(), key=lambda x: -x[1])[:5]
    top_cats_str = ", ".join(f"cat{c}({n})" for c, n in top_cats) or "none"
    return (
        f"User#{user.user_id} | age={age} | gender={gender} | level={level} "
        f"| segment={activity} | clicks={len(user.click_seq)} "
        f"| purchases={len(user.purchase_seq)} | top_categories={top_cats_str}"
    )


class LLMService:
    """Profile + Embedding service used by ChoirRec.

    Backend is configured by ``LLMConfig.backend``:
      * "mock"  - hashing-based deterministic embeddings.
      * "qwen"  - real Qwen API calls (requires ``dashscope`` and an API key).
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = None
        if cfg.backend == "qwen":
            self._init_qwen()

    def _init_qwen(self) -> None:
        try:
            import dashscope  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "dashscope not installed; run `pip install dashscope` or set backend='mock'."
            ) from e
        api_key = os.environ.get(self.cfg.qwen_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.cfg.qwen_api_key_env} is not set."
            )
        dashscope.api_key = api_key
        self._client = dashscope

    # ------------------------------------------------------------------
    # Profile synthesis (Paper 4.2.1)
    # ------------------------------------------------------------------
    def synthesize_profile(self, user: UserRecord, item_to_category) -> str:
        raw = render_user_text(user, item_to_category)
        if self.cfg.backend == "mock":
            return self._mock_profile(raw)
        return self._qwen_profile(raw)

    def _mock_profile(self, raw_text: str) -> str:
        """A deterministic pseudo-profile that mimics LLM rewriting.

        We deliberately keep the original signal so downstream embedding can
        still recover preferences, while wrapping it with template wording.
        """
        return (
            "[ChoirRec Semantic Profile] "
            f"This shopper is summarized as: {raw_text}. "
            "Likely interests are inferred from top categories above."
        )

    def _qwen_profile(self, raw_text: str) -> str:  # pragma: no cover - network
        prompt = (
            "You are an e-commerce analyst. Given the following anonymized "
            "user statistics, write a concise semantic profile that captures "
            "long-term preferences and shopping intent in 2-3 sentences.\n\n"
            f"User stats: {raw_text}"
        )
        rsp = self._client.Generation.call(
            model=self.cfg.qwen_chat_model,
            prompt=prompt,
            result_format="text",
        )
        return rsp.output.text  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Embedding (Paper 5.1: Matryoshka 4096 -> 512)
    # ------------------------------------------------------------------
    def embed_profiles(self, profiles: Iterable[str]) -> np.ndarray:
        profiles = list(profiles)
        if self.cfg.backend == "mock":
            vec = np.stack([self._mock_embed(p) for p in profiles], axis=0)
        else:
            vec = self._qwen_embed_batch(profiles)
        # Matryoshka-style truncation: keep the leading `truncated_dim` dims.
        truncated = vec[:, : self.cfg.truncated_dim]
        # L2-normalise to make downstream KMeans behave well.
        norms = np.linalg.norm(truncated, axis=1, keepdims=True) + 1e-9
        return truncated / norms

    def _mock_embed(self, text: str) -> np.ndarray:
        """Hash-based embedding into ``profile_dim`` dims.

        This is deterministic, content-aware (token co-occurrence influences
        the vector) and dimensionally consistent with the real Qwen output.
        """
        rng = np.random.default_rng(self._stable_seed(text))
        base = rng.normal(size=self.cfg.profile_dim).astype(np.float32)
        # Inject token-level signals so semantically similar texts cluster.
        for tok in text.lower().split():
            h = self._stable_seed(tok) % self.cfg.profile_dim
            base[h] += 1.0
        return base

    @staticmethod
    def _stable_seed(text: str) -> int:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

    def _qwen_embed_batch(self, profiles: List[str]) -> np.ndarray:  # pragma: no cover
        embeddings: List[np.ndarray] = []
        for p in profiles:
            rsp = self._client.TextEmbedding.call(
                model=self.cfg.qwen_embed_model,
                input=p,
            )
            vec = rsp.output["embeddings"][0]["embedding"]  # type: ignore[index]
            embeddings.append(np.asarray(vec, dtype=np.float32))
        return np.stack(embeddings, axis=0)
