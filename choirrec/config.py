"""Global configuration for ChoirRec.

The hyperparameters mirror the paper (Section 5.1: Implementation Details):
- RQ-KMeans: 3 stages, 256 centroids per stage.
- Embedding dim truncated from Qwen3-Embedding-8B 4096 -> 512 (Matryoshka).
- Optimizer: Adagrad, lr 0.01 -> 0.001, batch size 1024.
- Distillation loss weight lambda = 0.005.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    num_users: int = 1000
    num_items: int = 200
    num_categories: int = 20
    avg_high_clicks: int = 60
    avg_low_clicks: int = 4
    high_activity_ratio: float = 0.18  # paper: top 18% -> 90% clicks
    purchase_rate: float = 0.15
    seed: int = 42
    seq_max_len: int = 20


@dataclass
class LLMConfig:
    backend: str = "mock"  # one of {"mock", "qwen"}
    profile_dim: int = 4096       # native Qwen3-Embedding-8B output
    truncated_dim: int = 512      # Matryoshka truncation per paper
    qwen_chat_model: str = "qwen3-30b-a3b"
    qwen_embed_model: str = "qwen3-embedding-8b"
    qwen_api_key_env: str = "DASHSCOPE_API_KEY"


@dataclass
class GroupingConfig:
    num_stages: int = 3
    centroids_per_stage: int = 256  # k=256 is the sweet spot reported in Fig. 6(a)
    kmeans_max_iter: int = 50
    seed: int = 42


@dataclass
class ModelConfig:
    user_emb_dim: int = 32
    item_emb_dim: int = 32
    cat_emb_dim: int = 16
    group_id_emb_dim: int = 32
    profile_proj_dim: int = 64
    seq_max_len: int = 20
    individual_mlp: List[int] = field(default_factory=lambda: [128, 64, 32])
    group_mlp: List[int] = field(default_factory=lambda: [128, 64, 32])
    reliability_mlp: List[int] = field(default_factory=lambda: [32, 16])
    asymmetric_mlp: List[int] = field(default_factory=lambda: [64, 32])
    tower_mlp: List[int] = field(default_factory=lambda: [64, 32, 1])


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 3
    lr_init: float = 0.01
    lr_final: float = 0.001
    distill_lambda: float = 0.005  # paper Fig. 6(b) optimum
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle_buffer: int = 4096
    log_every: int = 20


@dataclass
class ChoirRecConfig:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    artifact_dir: str = "artifacts"

    def small(self) -> "ChoirRecConfig":
        """Return a tiny config tuned for quick smoke tests."""
        cfg = ChoirRecConfig()
        cfg.data.num_users = 400
        cfg.data.num_items = 80
        cfg.data.num_categories = 8
        cfg.data.avg_high_clicks = 30
        cfg.data.avg_low_clicks = 3
        cfg.grouping.centroids_per_stage = 4  # tiny k for tiny dataset
        cfg.grouping.num_stages = 3
        cfg.train.batch_size = 128
        cfg.train.epochs = 2
        cfg.llm.truncated_dim = 64
        cfg.model.profile_proj_dim = 32
        return cfg
