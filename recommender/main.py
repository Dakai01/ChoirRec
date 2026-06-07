"""One-click entry point that trains and evaluates the dual-channel CVR model."""

from __future__ import annotations

import argparse
import os
from typing import Dict

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "recommender"

import numpy as np
import torch
import torch.nn as nn

from .config import RecommenderConfig
from .dataset import FEATURE_KEYS, build_dataset, load_vocab_sizes
from .metrics import evaluate
from .model import ChoirRecModel


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split(n: int, cfg: RecommenderConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.train.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * cfg.train.test_ratio))
    n_val = max(1, int(n * cfg.train.val_ratio))
    return {
        "test": perm[:n_test],
        "val": perm[n_test:n_test + n_val],
        "train": perm[n_test + n_val:],
    }


@torch.no_grad()
def _evaluate(model, feats, labels, user_ids, is_low, idx, device) -> Dict[str, float]:
    model.eval()
    idx_t = torch.as_tensor(idx).to(device)
    batch = {k: feats[k][idx_t] for k in FEATURE_KEYS}
    scores = torch.sigmoid(model(batch)["logit"]).cpu().numpy()
    return evaluate(user_ids[idx], labels[idx], scores, is_low[idx])


def run(cfg: RecommenderConfig) -> Dict[str, float]:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = _select_device()

    print(f"[1/4] Loading id-form artifacts from {os.path.dirname(cfg.rec_samples_file)} ...")
    ds = build_dataset(cfg)
    print(f"  samples={len(ds.labels)}  "
          f"low_activity_rows={int(ds.is_low.sum())}")

    print("[2/4] Loading vocab sizes ...")
    sizes = load_vocab_sizes(cfg.vocab_file)
    centroids = int(ds.feats["user_codes"].max()) + 1
    centroids_per_stage = max(centroids, 8)
    print(f"  items={sizes.item} l1={sizes.l1} leaf={sizes.leaf} "
          f"city={sizes.city} users={sizes.user} centroids/stage={centroids_per_stage}")

    print(f"[3/4] Building model (device={device.type}) ...")
    model = ChoirRecModel(
        cfg=cfg.model,
        num_users=sizes.user,
        num_items=sizes.item,
        num_l1=sizes.l1,
        num_leaf=sizes.leaf,
        num_city=sizes.city,
        num_age=sizes.age_bucket,
        num_gender=sizes.gender,
        num_stages=cfg.num_stages,
        centroids_per_stage=centroids_per_stage,
    ).to(device)

    feats = {k: torch.as_tensor(v).to(device) for k, v in ds.feats.items()}
    labels_t = torch.as_tensor(ds.labels).to(device)
    splits = _split(len(ds.labels), cfg)

    print("[4/4] Training ChoirRec ...")
    tc = cfg.train
    optimizer = torch.optim.Adagrad(model.parameters(), lr=tc.lr_init)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=tc.lr_final / tc.lr_init,
        total_iters=max(1, tc.epochs),
    )
    bce = nn.BCEWithLogitsLoss()
    train_idx = splits["train"]

    for epoch in range(tc.epochs):
        model.train()
        rng = np.random.default_rng(tc.seed + epoch)
        order = rng.permutation(train_idx)
        epoch_losses = []
        for start in range(0, len(order), tc.batch_size):
            batch_idx = torch.as_tensor(order[start:start + tc.batch_size]).to(device)
            batch = {k: feats[k][batch_idx] for k in FEATURE_KEYS}
            y = labels_t[batch_idx]

            out = model(batch)
            main_loss = bce(out["logit"], y)

            # Gated Knowledge Distillation.
            ind_logit = out["individual_logit"]
            grp_logit = out["group_logit"]
            teacher = torch.sigmoid(grp_logit).detach()
            student = torch.sigmoid(ind_logit)

            # Hard qualification gate.
            enough_purchase = batch["purchase_count"] > tc.gate_purchase_threshold
            confident = (student.detach() - 0.5).abs() > tc.gate_confidence_threshold
            g_qual = (enough_purchase & confident).float()
            # Soft reliability gate.
            alpha_distill = out["distill_alpha"]

            margin = (student - teacher) ** 2
            denom = g_qual.sum().clamp_min(1.0)
            distill = (g_qual * alpha_distill * margin).sum() / denom

            loss = main_loss + tc.distill_lambda * distill
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step()

        if (epoch + 1) % max(1, tc.log_every) == 0 or epoch == 0:
            val_m = _evaluate(model, feats, ds.labels, ds.user_ids, ds.is_low,
                              splits["val"], device)
            print(f"  epoch {epoch+1:3d} | loss={np.mean(epoch_losses):.4f} | "
                  f"val_auc={val_m['auc']:.4f} val_gauc={val_m['gauc']:.4f} "
                  f"val_gauc_low={val_m['gauc_low']:.4f}")

    test_m = _evaluate(model, feats, ds.labels, ds.user_ids, ds.is_low,
                       splits["test"], device)
    metrics = {
        "auc": test_m["auc"], "gauc": test_m["gauc"],
        "gauc_low": test_m["gauc_low"], "gauc_high": test_m["gauc_high"],
    }
    print("[done] metrics:", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ChoirRec recommender.")
    parser.parse_args()
    run(RecommenderConfig())


if __name__ == "__main__":
    main()
