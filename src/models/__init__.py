"""
Machine learning models for lyrics classification.
"""

from .base_pipeline import BasePipeline
from .lr_pipeline import LogisticRegressionPipeline

__all__ = ["BasePipeline", "LogisticRegressionPipeline"]