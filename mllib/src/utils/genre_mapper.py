"""
Genre to label mapping utilities.
"""

# Mapping from genre names to numeric labels
GENRE_TO_LABEL_MAP = {
    "pop": 0,
    "country": 1,
    "blues": 2,
    "rock": 3,
    "jazz": 4,
    "reggae": 5,
    "hip hop": 6,
    "soul": 7,
    "unknown": 8,
}

# Mapping from numeric labels to genre names
LABEL_TO_GENRE_MAP = {
    0: "pop",
    1: "country",
    2: "blues",
    3: "rock",
    4: "jazz",
    5: "reggae",
    6: "hip hop",
    7: "soul",
    8: "unknown",
}

# Genre colors for visualization
GENRE_COLORS = {
    "pop": "#FF6B6B",
    "country": "#4ECDC4",
    "blues": "#1A535C",
    "rock": "#FF9F1C",
    "jazz": "#6B5CA5",
    "reggae": "#4CB944",
    "hip hop": "#F9C80E",
    "soul": "#A882DD",
    "unknown": "#CCCCCC",
    "other": "#AAAAAA",
}