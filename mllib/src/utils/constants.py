"""
Constants and enums for the lyrics classifier.
"""
from enum import Enum
import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join("..", "Merged_dataset.csv")
MODEL_DIR_PATH = os.path.join("..", "model")


class Column(Enum):
    """Column names used in the data processing pipeline."""
    VALUE = "lyrics"
    GENRE = "genre"
    LABEL = "label"
    CLEAN = "cleaned_lyrics"
    WORDS = "tokenized_lyrics"
    FILTERED_WORDS = "stop_words_removed_lyrics"
    STEMMED_WORDS = "stemmed_lyrics"
    FEATURES = "features"
    PREDICTION = "prediction"
    PROBABILITY = "probability"


# Minimum probability threshold for prediction confidence
DEFAULT_PREDICTION_THRESHOLD = 0.35