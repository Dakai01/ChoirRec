"""ChoirRec recommender package: the runnable dual-channel CVR model."""
# update date：2026-06-07

from .config import ModelConfig, RecommenderConfig, TrainConfig
from .model import ChoirRecModel

__all__ = ["RecommenderConfig", "ModelConfig", "TrainConfig", "ChoirRecModel"]
