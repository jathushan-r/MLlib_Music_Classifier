"""
Main Streamlit application for the lyrics genre classifier.
"""
import os
import time
from typing import Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import nltk

from src.utils.constants import DATASET_PATH, MODEL_DIR_PATH, DEFAULT_PREDICTION_THRESHOLD
from src.models.lr_pipeline import LogisticRegressionPipeline
from src.visualization.charts import create_probability_bar_chart, create_genre_pie_chart
from pyspark.ml.tuning import CrossValidatorModel


# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@st.cache_resource
def load_model() -> Tuple[LogisticRegressionPipeline, CrossValidatorModel]:
    """
    Load or train the model.
    
    Returns:
        Tuple of (pipeline, model)
    """
    pipeline = LogisticRegressionPipeline()
    
    # Check if model exists
    if not (
        os.path.exists(MODEL_DIR_PATH)
        and os.path.isdir(MODEL_DIR_PATH)
        and len(os.listdir(MODEL_DIR_PATH)) > 0
    ):
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)
        
        with st.spinner("Training model... This may take a few minutes."):
            model = pipeline.train_and_test(
                dataset_path=DATASET_PATH,
                train_ratio=0.8,
                store_model_on=MODEL_DIR_PATH,
                print_statistics=True,
            )
    else:
        with st.spinner("Loading model..."):
            model = CrossValidatorModel.load(MODEL_DIR_PATH)
    
    return pipeline, model


def predict_genre(
    lyrics: str, 
    pipeline: LogisticRegressionPipeline, 
    model: CrossValidatorModel,
    threshold: float
) -> Tuple[str, Dict[str, float]]:
    """
    Predict the genre of lyrics text.
    
    Args:
        lyrics: Lyrics text to classify
        pipeline: Classification pipeline
        model: Trained model
        threshold: Minimum probability threshold
        
    Returns:
        Tuple of (predicted genre, probability distribution)
    """
    # Make prediction
    prediction, probabilities = pipeline.predict_one(
        unknown_lyrics=lyrics,
        threshold=threshold,
        model=model
    )
    
    return prediction, probabilities


def show_prediction_results(
    prediction: str, 
    probabilities: Dict[str, float],
    threshold: float
) -> None:
    """
    Display prediction results.
    
    Args:
        prediction: Predicted genre
        probabilities: Probability distribution
        threshold: Minimum probability threshold
    """
    # Display prediction
    st.header("Prediction Result")
    
    # Create columns for results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display prediction as a card
        if prediction.lower() == "unknown":
            st.error(
                f"**Prediction: {prediction.upper()}**\n\n"
                f"Highest probability ({max(probabilities.values()):.1%}) "
                f"is below threshold ({threshold:.0%})"
            )
        else:
            st.success(
                f"**Predicted Genre: {prediction.capitalize()}**\n\n"
                f"Confidence: {probabilities[prediction]:.1%}"
            )
    
    with col2:
        # Show top 3 genres
        top_genres = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        st.write("**Top 3 Genres:**")
        for genre, prob in top_genres:
            st.write(f"- {genre.capitalize()}: {prob:.1%}")


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Lyrics Genre Classifier",
        page_icon="🎵",
        layout="wide",
    )
    
    # Header
    st.title("🎵 Lyrics Genre Classifier")
    
    # Load model
    pipeline, model = load_model()
    
    # Sidebar
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_PREDICTION_THRESHOLD,
        step=0.05,
        help="Minimum probability required for a genre prediction. "
             "If the highest probability is below this threshold, "
             "the genre will be classified as 'Unknown'."
    )
    
    # Input section
    st.header("Input Lyrics")
    lyrics = st.text_area(
        "Enter lyrics to classify",
        height=100,
        placeholder="Paste song lyrics here...",
    )
    
    # Prediction button
    col1, col2 = st.columns([1, 5])
    with col1:
        predict_button = st.button("Predict Genre", type="primary")
    
    # Make prediction when button is clicked
    if predict_button and lyrics:
        with st.spinner("Analyzing lyrics..."):
            # Add small delay for better UX
            time.sleep(0.5)
            
            # Make prediction
            prediction, probabilities = predict_genre(
                lyrics=lyrics,
                pipeline=pipeline,
                model=model,
                threshold=threshold
            )
        
        # Show prediction results
        show_prediction_results(
            prediction=prediction,
            probabilities=probabilities,
            threshold=threshold
        )
        
        # Visualizations
        st.header("Visualizations")
        
        # Create columns for charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            create_genre_pie_chart(probabilities)
        
        with col2:
            # Bar chart
            create_probability_bar_chart(probabilities)
    
    # Cleanup when app is closed
    if hasattr(st, "session_state") and "cleanup_done" not in st.session_state:
        import atexit
        atexit.register(pipeline.stop)
        st.session_state.cleanup_done = True


if __name__ == "__main__":
    main()
