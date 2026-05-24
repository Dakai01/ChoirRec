"""ChoirRec dual-channel model (Paper Sec. 4.4).

This file implements the Group-aware Multi-granularity Module:

    * Two parallel channels: an Individual Channel that consumes raw user
      features and an aligned Group Channel that consumes group priors.
    * Asymmetric Information Injection: features flow ONLY from the group
      channel into the individual channel (Sec. 4.4.2). This protects
      group representations from being polluted by sparse individual noise
      while still allowing rich users to benefit from their own data.
    * Gated Knowledge Distillation: a reliability network produces a per-
      sample gate that decides how strongly the individual logit should be
      supervised by the (more reliable) group logit (Sec. 4.4.3). This is
      what prevents group signals from being overshadowed during training.
    * Final prediction is a fusion of the two channel logits (Sec. 4.4.4).
"""

from __future__ import annotations

from typing import Dict, List

import tensorflow as tf

from .config import ModelConfig


def _build_mlp(units: List[int], name: str, activation: str = "relu",
               final_activation: str | None = "relu") -> tf.keras.Sequential:
    layers: List[tf.keras.layers.Layer] = []
    for i, u in enumerate(units):
        is_last = i == len(units) - 1
        act = final_activation if is_last else activation
        layers.append(tf.keras.layers.Dense(u, activation=act, name=f"{name}_dense_{i}"))
    return tf.keras.Sequential(layers, name=name)


def _seq_pool(emb: tf.Tensor, ids: tf.Tensor) -> tf.Tensor:
    """Mean-pool an embedded sequence with PAD (id=0) masked out."""
    mask = tf.cast(tf.not_equal(ids, 0), emb.dtype)[..., None]
    summed = tf.reduce_sum(emb * mask, axis=1)
    denom = tf.reduce_sum(mask, axis=1) + 1e-6
    return summed / denom


class ChoirRecModel(tf.keras.Model):
    """End-to-end ChoirRec model.

    Vocabulary sizing rules (all use 0 as PAD):
        item embedding table size = num_items + 1
        category embedding table size = num_categories + 1
        group-id table per stage = centroids_per_stage
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_users: int,
        num_items: int,
        num_categories: int,
        num_age: int,
        num_gender: int,
        num_level: int,
        num_stages: int,
        centroids_per_stage: int,
    ):
        super().__init__(name="ChoirRecModel")
        self.cfg = cfg
        self.num_stages = num_stages

        # ------------------------------------------------------------------
        # Embedding tables
        # ------------------------------------------------------------------
        self.user_emb = tf.keras.layers.Embedding(num_users, cfg.user_emb_dim, name="user_emb")
        self.item_emb = tf.keras.layers.Embedding(num_items + 1, cfg.item_emb_dim, mask_zero=False, name="item_emb")
        self.cat_emb = tf.keras.layers.Embedding(num_categories + 1, cfg.cat_emb_dim, mask_zero=False, name="cat_emb")
        self.age_emb = tf.keras.layers.Embedding(num_age, cfg.cat_emb_dim, name="age_emb")
        self.gender_emb = tf.keras.layers.Embedding(num_gender, cfg.cat_emb_dim, name="gender_emb")
        self.level_emb = tf.keras.layers.Embedding(num_level, cfg.cat_emb_dim, name="level_emb")
        # One embedding table per RQ-KMeans stage -> hierarchical group ID fusion
        self.group_id_embs = [
            tf.keras.layers.Embedding(centroids_per_stage, cfg.group_id_emb_dim, name=f"group_id_stage{s}")
            for s in range(num_stages)
        ]

        self.profile_proj = tf.keras.layers.Dense(cfg.profile_proj_dim, activation="relu", name="profile_proj")

        # ------------------------------------------------------------------
        # Channels (Sec. 4.4.1)
        # ------------------------------------------------------------------
        self.individual_mlp = _build_mlp(cfg.individual_mlp, name="individual_channel")
        self.group_mlp = _build_mlp(cfg.group_mlp, name="group_channel")

        # ------------------------------------------------------------------
        # Asymmetric injection (Sec. 4.4.2): a learned transform that maps
        # the group representation into the individual channel's space. The
        # reverse direction is intentionally not present.
        # ------------------------------------------------------------------
        self.inject_mlp = _build_mlp(cfg.asymmetric_mlp, name="asymmetric_inject")

        # Reliability network gates how much group->individual distillation to
        # apply for each sample. Inputs include user activity flag + group code.
        self.reliability_mlp = _build_mlp(
            cfg.reliability_mlp + [1], name="reliability_net", final_activation="sigmoid"
        )

        # Prediction towers
        self.individual_tower = _build_mlp(
            cfg.tower_mlp, name="individual_tower", final_activation=None
        )
        self.group_tower = _build_mlp(
            cfg.tower_mlp, name="group_tower", final_activation=None
        )
        self.fusion_gate = tf.keras.layers.Dense(1, activation="sigmoid", name="fusion_gate")

    # ----------------------------------------------------------------------
    # Channel feature builders
    # ----------------------------------------------------------------------
    def _individual_features(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        u = self.user_emb(inputs["user_id"])
        item = self.item_emb(inputs["item_id"])
        cat = self.cat_emb(inputs["item_cat"])
        clicks = self.item_emb(inputs["user_click_seq"])
        click_pool = _seq_pool(clicks, inputs["user_click_seq"])
        cat_clicks = self.cat_emb(inputs["user_cat_seq"])
        cat_pool = _seq_pool(cat_clicks, inputs["user_cat_seq"])
        return tf.concat([u, item, cat, click_pool, cat_pool], axis=-1)

    def _group_features(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Hierarchical group-ID fusion (Sec. 4.3.1)
        codes = inputs["user_codes"]  # [B, S]
        stage_vecs = [self.group_id_embs[s](codes[:, s]) for s in range(self.num_stages)]
        group_id_vec = tf.concat(stage_vecs, axis=-1)

        # Group attribute completion (Sec. 4.3.2)
        attrs = inputs["completed_attrs"]
        age_v = self.age_emb(attrs[:, 0])
        gen_v = self.gender_emb(attrs[:, 1])
        lvl_v = self.level_emb(attrs[:, 2])

        # Aggregated semantic profile of the leaf group
        profile_v = self.profile_proj(inputs["group_profile"])

        # Group behavioral sequences (Sec. 4.3.3)
        gclicks = self.item_emb(inputs["group_click_seq"])
        gpool = _seq_pool(gclicks, inputs["group_click_seq"])
        gcats = self.cat_emb(inputs["group_cat_seq"])
        gcat_pool = _seq_pool(gcats, inputs["group_cat_seq"])

        item = self.item_emb(inputs["item_id"])
        return tf.concat([group_id_vec, age_v, gen_v, lvl_v, profile_v, gpool, gcat_pool, item], axis=-1)

    # ----------------------------------------------------------------------
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False):
        ind_feat = self._individual_features(inputs)
        grp_feat = self._group_features(inputs)

        ind_repr = self.individual_mlp(ind_feat, training=training)
        grp_repr = self.group_mlp(grp_feat, training=training)

        # Asymmetric injection: group -> individual, but NOT the reverse.
        injected = self.inject_mlp(grp_repr, training=training)
        ind_repr_aug = ind_repr + injected

        ind_logit = self.individual_tower(ind_repr_aug, training=training)
        grp_logit = self.group_tower(grp_repr, training=training)

        # Reliability gate (Sec. 4.4.3) decides distillation strength per sample
        rel_input = tf.concat(
            [
                tf.cast(inputs["is_low_activity"][:, None], tf.float32),
                tf.cast(inputs["user_codes"], tf.float32) / 256.0,
            ],
            axis=-1,
        )
        gate = self.reliability_mlp(rel_input, training=training)  # [B, 1] in (0, 1)

        # Fusion (Sec. 4.4.4): adaptive blend of two channels
        fuse_input = tf.concat([ind_repr_aug, grp_repr], axis=-1)
        alpha = self.fusion_gate(fuse_input)
        final_logit = alpha * ind_logit + (1.0 - alpha) * grp_logit

        return {
            "logit": tf.squeeze(final_logit, axis=-1),
            "individual_logit": tf.squeeze(ind_logit, axis=-1),
            "group_logit": tf.squeeze(grp_logit, axis=-1),
            "distill_gate": tf.squeeze(gate, axis=-1),
        }
