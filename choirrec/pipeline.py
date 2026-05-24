"""Training, evaluation and the end-to-end pipeline of ChoirRec."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .config import ChoirRecConfig
from .data import SyntheticDataset, generate_synthetic_dataset
from .grouping import RQKMeans
from .llm import LLMService
from .model import ChoirRecModel
from .representation import GroupArtifacts, build_group_artifacts, vocabulary_sizes


# ----------------------------------------------------------------------
# Feature assembly
# ----------------------------------------------------------------------

@dataclass
class TensorFeatures:
    user_id: np.ndarray
    item_id: np.ndarray
    item_cat: np.ndarray
    label: np.ndarray
    is_low_activity: np.ndarray
    user_click_seq: np.ndarray
    user_cat_seq: np.ndarray
    user_codes: np.ndarray
    completed_attrs: np.ndarray
    group_profile: np.ndarray
    group_click_seq: np.ndarray
    group_cat_seq: np.ndarray


def _build_features(
    samples: List[Tuple[int, int, int]],
    item_to_category: Dict[int, int],
    artifacts: GroupArtifacts,
    group_profiles: np.ndarray,
) -> TensorFeatures:
    n = len(samples)
    user_ids = np.empty(n, dtype=np.int32)
    item_ids = np.empty(n, dtype=np.int32)
    item_cats = np.empty(n, dtype=np.int32)
    labels = np.empty(n, dtype=np.float32)

    for i, (uid, iid, y) in enumerate(samples):
        user_ids[i] = uid
        item_ids[i] = iid + 1  # +1 for PAD reservation
        item_cats[i] = item_to_category[iid] + 1
        labels[i] = y

    is_low = artifacts.is_low_activity[user_ids]
    user_click = artifacts.user_click_seq[user_ids]
    user_cat = artifacts.user_cat_seq[user_ids]
    codes = artifacts.user_codes[user_ids]
    attrs = artifacts.completed_attrs[user_ids]
    g_click = artifacts.group_click_seq[user_ids]
    g_cat = artifacts.group_cat_seq[user_ids]

    leaf_ids = artifacts.leaf_group_ids[user_ids]
    profiles = group_profiles[leaf_ids]

    return TensorFeatures(
        user_id=user_ids,
        item_id=item_ids,
        item_cat=item_cats,
        label=labels,
        is_low_activity=is_low,
        user_click_seq=user_click,
        user_cat_seq=user_cat,
        user_codes=codes,
        completed_attrs=attrs,
        group_profile=profiles,
        group_click_seq=g_click,
        group_cat_seq=g_cat,
    )


def _features_to_dict(feat: TensorFeatures) -> Dict[str, np.ndarray]:
    return {
        "user_id": feat.user_id,
        "item_id": feat.item_id,
        "item_cat": feat.item_cat,
        "is_low_activity": feat.is_low_activity,
        "user_click_seq": feat.user_click_seq,
        "user_cat_seq": feat.user_cat_seq,
        "user_codes": feat.user_codes,
        "completed_attrs": feat.completed_attrs,
        "group_profile": feat.group_profile,
        "group_click_seq": feat.group_click_seq,
        "group_cat_seq": feat.group_cat_seq,
    }


def _split_samples(samples: List[Tuple[int, int, int]], val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(samples))
    n_test = int(len(samples) * test_ratio)
    n_val = int(len(samples) * val_ratio)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    arr = np.array(samples, dtype=np.int64)
    return arr[train_idx], arr[val_idx], arr[test_idx]


# ----------------------------------------------------------------------
# GAUC: user-weighted average AUC (paper Sec. 5.1).
# ----------------------------------------------------------------------

def gauc(user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> float:
    buckets: Dict[int, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for u, y, s in zip(user_ids, labels, scores):
        ys, ss = buckets[int(u)]
        ys.append(float(y))
        ss.append(float(s))
    weighted_sum, weight_total = 0.0, 0.0
    for u, (ys, ss) in buckets.items():
        if len(set(ys)) < 2:
            continue
        try:
            auc = roc_auc_score(ys, ss)
        except ValueError:
            continue
        weighted_sum += auc * len(ys)
        weight_total += len(ys)
    return weighted_sum / weight_total if weight_total > 0 else float("nan")


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

class ChoirRecPipeline:
    def __init__(self, cfg: ChoirRecConfig):
        self.cfg = cfg
        os.makedirs(cfg.artifact_dir, exist_ok=True)

    def run(self) -> Dict[str, float]:
        print("[1/6] Generating synthetic dataset...")
        dataset = generate_synthetic_dataset(self.cfg.data)
        print(f"  users={len(dataset.users)}  samples={len(dataset.samples)}")

        print("[2/6] Synthesising LLM profiles...")
        llm = LLMService(self.cfg.llm)
        profiles = [llm.synthesize_profile(u, dataset.item_to_category) for u in dataset.users]
        embeddings = llm.embed_profiles(profiles)
        print(f"  embeddings shape={embeddings.shape}")

        print("[3/6] Running RQ-KMeans...")
        rq = RQKMeans(self.cfg.grouping)
        rq_result = rq.fit(embeddings)
        print(f"  group codes shape={rq_result.codes.shape}")

        print("[4/6] Building group-aware representations...")
        artifacts = build_group_artifacts(dataset.users, rq_result.codes, self.cfg.data)
        # Group-level profile = mean embedding of its members.
        group_profiles = np.zeros((artifacts.num_leaf_groups, embeddings.shape[1]), dtype=np.float32)
        counts = np.zeros(artifacts.num_leaf_groups, dtype=np.float32)
        for uid, gid in enumerate(artifacts.leaf_group_ids):
            group_profiles[gid] += embeddings[uid]
            counts[gid] += 1
        group_profiles /= np.maximum(counts[:, None], 1.0)
        print(f"  leaf groups = {artifacts.num_leaf_groups}")

        print("[5/6] Splitting samples & building tensors...")
        train_arr, val_arr, test_arr = _split_samples(
            dataset.samples,
            self.cfg.train.val_ratio,
            self.cfg.train.test_ratio,
            self.cfg.data.seed,
        )

        def to_feat(arr):
            return _build_features(
                [tuple(row) for row in arr.tolist()],
                dataset.item_to_category,
                artifacts,
                group_profiles,
            )

        train_feat = to_feat(train_arr)
        val_feat = to_feat(val_arr)
        test_feat = to_feat(test_arr)

        print("[6/6] Building & training ChoirRec model...")
        age_v, gen_v, lvl_v = vocabulary_sizes()
        model = ChoirRecModel(
            cfg=self.cfg.model,
            num_users=self.cfg.data.num_users,
            num_items=self.cfg.data.num_items,
            num_categories=self.cfg.data.num_categories,
            num_age=age_v,
            num_gender=gen_v,
            num_level=lvl_v,
            num_stages=self.cfg.grouping.num_stages,
            centroids_per_stage=max(self.cfg.grouping.centroids_per_stage, 8),
        )
        metrics = self._train(model, train_feat, val_feat, test_feat)
        print("[done] metrics:", metrics)
        return metrics

    # ------------------------------------------------------------------
    def _train(
        self,
        model: ChoirRecModel,
        train_feat: TensorFeatures,
        val_feat: TensorFeatures,
        test_feat: TensorFeatures,
    ) -> Dict[str, float]:
        train_cfg = self.cfg.train
        # Linearly decay lr from lr_init to lr_final over the training run.
        steps_per_epoch = max(1, int(np.ceil(len(train_feat.label) / train_cfg.batch_size)))
        decay_steps = max(1, steps_per_epoch * train_cfg.epochs)
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=train_cfg.lr_init,
            end_learning_rate=train_cfg.lr_final,
            decay_steps=decay_steps,
            power=1.0,
        )
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=schedule)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        def make_dataset(feat: TensorFeatures, shuffle: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices((_features_to_dict(feat), feat.label))
            if shuffle:
                ds = ds.shuffle(min(train_cfg.shuffle_buffer, len(feat.label)), seed=42)
            return ds.batch(train_cfg.batch_size).prefetch(tf.data.AUTOTUNE)

        train_ds = make_dataset(train_feat, shuffle=True)
        val_ds = make_dataset(val_feat, shuffle=False)

        @tf.function
        def train_step(features, labels):
            with tf.GradientTape() as tape:
                out = model(features, training=True)
                main_loss = tf.reduce_mean(bce(labels, out["logit"]))
                ind_loss = tf.reduce_mean(bce(labels, out["individual_logit"]))
                grp_loss = tf.reduce_mean(bce(labels, out["group_logit"]))
                # Gated knowledge distillation: group -> individual.
                # The gate is learnable; only the *teacher* (group prob) is
                # detached so the student channel chases the teacher rather
                # than the other way around.
                gate = out["distill_gate"]
                grp_prob = tf.stop_gradient(tf.sigmoid(out["group_logit"]))
                ind_prob = tf.sigmoid(out["individual_logit"])
                distill = tf.reduce_mean(
                    gate * tf.square(ind_prob - grp_prob)
                )
                total = main_loss + 0.5 * (ind_loss + grp_loss) + train_cfg.distill_lambda * distill
            grads = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total, main_loss, distill

        for epoch in range(train_cfg.epochs):
            t0 = time.time()
            losses = []
            for step, (features, labels) in enumerate(train_ds):
                total, main_loss, distill = train_step(features, labels)
                losses.append(float(total.numpy()))
                if step % train_cfg.log_every == 0:
                    print(
                        f"  epoch {epoch+1} step {step:4d} | total={float(total):.4f} "
                        f"main={float(main_loss):.4f} distill={float(distill):.6f}"
                    )
            val_metrics = self._evaluate(model, val_feat)
            print(
                f"  epoch {epoch+1} done in {time.time()-t0:.1f}s | "
                f"avg_train_loss={np.mean(losses):.4f} | "
                f"val_auc={val_metrics['auc']:.4f} val_gauc={val_metrics['gauc']:.4f} "
                f"val_gauc_low={val_metrics['gauc_low']:.4f}"
            )

        test_metrics = self._evaluate(model, test_feat)
        return test_metrics

    @staticmethod
    def _evaluate(model: ChoirRecModel, feat: TensorFeatures) -> Dict[str, float]:
        ds = tf.data.Dataset.from_tensor_slices(_features_to_dict(feat)).batch(1024)
        scores: List[np.ndarray] = []
        for batch in ds:
            out = model(batch, training=False)
            scores.append(tf.sigmoid(out["logit"]).numpy())
        all_scores = np.concatenate(scores, axis=0)
        labels = feat.label
        try:
            auc = roc_auc_score(labels, all_scores) if len(set(labels.tolist())) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        g_all = gauc(feat.user_id, labels, all_scores)
        # Slice by activity bucket
        low_mask = feat.is_low_activity == 1
        high_mask = ~low_mask
        g_low = gauc(feat.user_id[low_mask], labels[low_mask], all_scores[low_mask]) if low_mask.any() else float("nan")
        g_high = gauc(feat.user_id[high_mask], labels[high_mask], all_scores[high_mask]) if high_mask.any() else float("nan")
        return {"auc": auc, "gauc": g_all, "gauc_low": g_low, "gauc_high": g_high}
