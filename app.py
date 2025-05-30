import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Video Game Sales Analysis",
    page_icon="🎮",
    layout="wide"
)

# Title and introduction
st.title("🎮 Video Game Sales Analysis Dashboard")
st.markdown("""
This dashboard presents the results of our comprehensive analysis of video game sales data.
We've used multiple machine learning approaches to understand what drives game sales success.
""")

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Regression Analysis", 
    "Naive Bayes", 
    "Decision Tree", 
    "Hierarchical Clustering",
    "K-means Clustering"
])

# Regression Analysis Tab
with tab1:
    st.header("Regression Analysis Results")
    
    # Model Performance
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R-squared (R²)", "0.9523", "95.23% variance explained")
    with col2:
        st.metric("Mean Absolute Error", "0.0721", "Low prediction error")
    
    # Feature Importance
    st.subheader("Feature Importance")
    if os.path.exists('regression_results/feature_importance.png'):
        st.image('regression_results/feature_importance.png')
    
    # Actual vs Predicted
    st.subheader("Actual vs Predicted Sales")
    if os.path.exists('regression_results/actual_vs_predicted.png'):
        st.image('regression_results/actual_vs_predicted.png')
    
    # Key Insights
    st.subheader("Key Insights")
    st.markdown("""
    1. **Regional Performance Profile**
       - Japan sales ratio (56.58%) is the strongest predictor
       - PAL region ratio (20.46%) is the second most important factor
       - North America ratio (21.47%) is the third most important factor
    
    2. **Market Focus Strategy**
       - Games with focused regional strategies show more predictable outcomes
       - Regional specialists often outperform globally balanced titles
    
    3. **Limited Impact of Traditional Factors**
       - Critic scores have no predictive power
       - Publisher and developer reputation have minimal impact
       - Genre and console have little influence on sales
    """)

# Naive Bayes Tab
with tab2:
    st.header("Naive Bayes Classification Results")
    
    # Model Performance
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gaussian Naive Bayes Accuracy", "84.40%")
    with col2:
        st.metric("Bernoulli Naive Bayes Accuracy", "55.10%")
    
    # Feature Importance
    if os.path.exists('naive_bayes_results/gnb_feature_importance.png'):
        st.image('naive_bayes_results/gnb_feature_importance.png')
    
    # ROC Curves
    if os.path.exists('naive_bayes_results/roc_curves.png'):
        st.image('naive_bayes_results/roc_curves.png')

# Decision Tree Tab
with tab3:
    st.header("Decision Tree Analysis Results")
    
    # Model Performance
    st.subheader("Model Performance")
    st.metric("Accuracy", "98.00%")
    
    # Decision Tree Visualization
    if os.path.exists('decision_tree_results/decision_tree_visualization.png'):
        st.image('decision_tree_results/decision_tree_visualization.png')
    
    # Feature Importance
    if os.path.exists('decision_tree_results/feature_importance.png'):
        st.image('decision_tree_results/feature_importance.png')

# Hierarchical Clustering Tab
with tab4:
    st.header("Hierarchical Clustering Results")
    
    # Dendrogram
    if os.path.exists('hierarchical_results/hierarchical_dendrogram.png'):
        st.image('hierarchical_results/hierarchical_dendrogram.png')
    
    # Cluster Characteristics
    if os.path.exists('hierarchical_results/hierarchical_regional_sales_radar.png'):
        st.image('hierarchical_results/hierarchical_regional_sales_radar.png')

# K-means Clustering Tab
with tab5:
    st.header("K-means Clustering Results")
    
    # PCA Visualization
    if os.path.exists('clustering_results/kmeans_pca_visualization.png'):
        st.image('clustering_results/kmeans_pca_visualization.png')
    
    # Regional Sales Radar
    if os.path.exists('clustering_results/kmeans_regional_sales_radar.png'):
        st.image('clustering_results/kmeans_regional_sales_radar.png')

# Footer
st.markdown("---")
st.markdown("""
### About the Analysis
This dashboard presents the results of multiple machine learning approaches to understand video game sales patterns.
The analysis includes regression, classification, and clustering methods to provide comprehensive insights into the gaming industry.
""")