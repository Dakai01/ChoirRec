"""Configuration for the recommender (CVR prediction)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")
REC_SAMPLES_FILE = os.path.join(DATA_DIR, "rec_samples.tsv")
GROUP_IDS_FILE = os.path.join(DATA_DIR, "group_ids.tsv")


@dataclass
class ModelConfig:
    item_emb_dim: int = 32
    cat_emb_dim: int = 16
    user_emb_dim: int = 32
    group_id_emb_dim: int = 32
    seq_max_len: int = 20
    attention_mlp: List[int] = field(default_factory=lambda: [64, 32])
    # Hierarchical group-ID prefix fusion.
    group_id_fusion_mlp: List[int] = field(default_factory=lambda: [32])
    individual_mlp: List[int] = field(default_factory=lambda: [128, 64, 32])
    group_mlp: List[int] = field(default_factory=lambda: [128, 64, 32])
    # Asymmetric injection fusion tower.
    asymmetric_mlp: List[int] = field(default_factory=lambda: [64])
    # Reliability network (soft gate) over user-activity features.
    reliability_mlp: List[int] = field(default_factory=lambda: [32, 16])


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 40
    lr_init: float = 0.01
    lr_final: float = 0.001
    distill_lambda: float = 0.005
    # Hard qualification gate.
    gate_purchase_threshold: int = 2     # theta_act: need purchase_count > this
    gate_confidence_threshold: float = 0.1  # theta_conf: |sigmoid(z_ind)-0.5| > this
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    log_every: int = 1
    seed: int = 42


@dataclass
class RecommenderConfig:
    vocab_file: str = VOCAB_FILE
    rec_samples_file: str = REC_SAMPLES_FILE
    group_ids_file: str = GROUP_IDS_FILE
    num_stages: int = 3

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
