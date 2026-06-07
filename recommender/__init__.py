"""ChoirRec recommender package: the runnable dual-channel CVR model."""

from .config import ModelConfig, RecommenderConfig, TrainConfig
from .model import ChoirRecModel

__all__ = ["RecommenderConfig", "ModelConfig", "TrainConfig", "ChoirRecModel"]
