"""
Chart creation utilities for lyrics classification results.
"""
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.genre_mapper import GENRE_COLORS


def create_probability_bar_chart(probabilities: Dict[str, float]) -> None:
    """
    Create a bar chart of genre probabilities.
    
    Args:
        probabilities: Dictionary mapping genres to probabilities
    """
    # Convert probabilities to percentage
    prob_percentage = {k: v * 100 for k, v in probabilities.items()}
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Genre': list(prob_percentage.keys()),
        'Probability (%)': list(prob_percentage.values())
    })
    
    # Sort by probability descending
    df = df.sort_values('Probability (%)', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Genre', 
        y='Probability (%)',
        title='Genre Probability Distribution',
        color='Genre',
        color_discrete_map=GENRE_COLORS,
        text_auto='.1f'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)


def create_genre_pie_chart(probabilities: Dict[str, float]) -> None:
    """
    Create a pie chart comparing the top genre against all others.
    
    Args:
        probabilities: Dictionary mapping genres to probabilities
    """
    # Find top genre
    top_genre = max(probabilities, key=probabilities.get)
    top_prob = probabilities[top_genre]
    
    # Calculate other genres' total probability
    other_prob = sum(v for k, v in probabilities.items() if k != top_genre)
    
    # Create data for the pie chart
    pie_data = pd.DataFrame({
        'Category': [top_genre.capitalize(), 'Other Genres'],
        'Probability (%)': [top_prob * 100, other_prob * 100]
    })
    
    # Create pie chart
    fig = px.pie(
        pie_data,
        values='Probability (%)',
        names='Category',
        title='Top Genre vs. Other Genres',
        color='Category',
        color_discrete_map={
            top_genre.capitalize(): GENRE_COLORS[top_genre],
            'Other Genres': GENRE_COLORS['other']
        }
    )
    
    # Update layout
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+percent'
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    