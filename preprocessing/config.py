"""Configuration for the preprocessing stage."""
# update date：2026-06-07

from __future__ import annotations

import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

USERS_FILE = os.path.join(DATA_DIR, "users.tsv")
EXPOSURE_SAMPLES_FILE = os.path.join(DATA_DIR, "exposure_samples.tsv")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")
REC_SAMPLES_FILE = os.path.join(DATA_DIR, "rec_samples.tsv")
GROUP_IDS_FILE = os.path.join(DATA_DIR, "group_ids.tsv")


@dataclass
class LLMConfig:
    # Paper used Qwen3-30B-A3B (chat) and Qwen3-Embedding-8B (embedding).
    chat_model: str = "Qwen/Qwen3-1.7B"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    truncated_dim: int = 128  # Matryoshka truncation (paper: 512)


@dataclass
class GroupingConfig:
    num_stages: int = 3
    centroids_per_stage: int = 3
    kmeans_max_iter: int = 50
    seed: int = 42


@dataclass
class PreprocessConfig:
    users_file: str = USERS_FILE
    exposure_samples_file: str = EXPOSURE_SAMPLES_FILE
    vocab_file: str = VOCAB_FILE
    rec_samples_file: str = REC_SAMPLES_FILE
    group_ids_file: str = GROUP_IDS_FILE

    seq_max_len: int = 20
    seed: int = 42

    llm: LLMConfig = field(default_factory=LLMConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
