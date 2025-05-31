# Video Game Sales Analysis Project Documentation

## Project Overview

This project provides a comprehensive analysis of video game sales data, including interactive visualizations, statistical analysis, and machine learning predictions. The application is built with Python and Streamlit, allowing users to explore sales trends, regional performance, and make predictions for new game titles.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation and Setup](#installation-and-setup)
3. [Data Description](#data-description)
4. [Features](#features)
5. [Machine Learning Models](#machine-learning-models)
6. [Usage Guide](#usage-guide)
7. [Technical Implementation](#technical-implementation)
8. [Troubleshooting](#troubleshooting)

## Project Structure

```
game-sales-analysis/
├── app.py                     # Main Streamlit application
├── create_models.py           # Script to train and save ML models
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and analysis results
├── vgchartz_cleaned.csv       # Cleaned dataset
├── vgchartz-2024.csv          # Raw dataset
├── vgchartz_numeric.csv       # Processed numeric dataset
├── vgchartz_pca.csv           # Dataset with PCA transformations
├── features.joblib            # Saved feature names for models
├── scaler.joblib              # Saved data scaler for predictions
├── regression_model.joblib    # Saved regression model
├── regression_results/        # Results from regression analysis
├── naive_bayes_results/       # Results from Naive Bayes classification
├── decision_tree_results/     # Results from Decision Tree classification
├── hierarchical_results/      # Results from hierarchical clustering
├── clustering_results/        # Results from other clustering methods
├── plots/                     # Generated visualization plots
├── cleaning_plots/            # Plots related to data cleaning
└── processed_data/            # Intermediate processed datasets
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/rezashariatmadar/game-sales-analysis.git
   cd game-sales-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python -m streamlit run app.py
   ```

## Data Description

The dataset contains information about video game sales across different regions, platforms, genres, and years. Key variables include:

- **title**: Name of the game
- **console**: Gaming platform/console
- **genre**: Game genre (Action, Sports, RPG, etc.)
- **publisher**: Company that published the game
- **developer**: Company that developed the game
- **critic_score**: Game ratings by critics (1-10 scale)
- **total_sales**: Total global sales in millions of units
- **na_sales**: North American sales in millions of units
- **jp_sales**: Japanese sales in millions of units
- **pal_sales**: European (PAL) sales in millions of units
- **other_sales**: Sales in other regions in millions of units
- **release_date**: Full date of game release
- **release_year**: Year the game was released
- **last_update**: Date when sales data was last updated

## Features

### Interactive Dashboard

The Streamlit application provides an interactive dashboard with the following features:

1. **Data Exploration**:
   - Filter data by year range, genre, platform, and publisher
   - View summary statistics and key metrics
   - Explore the raw dataset with searchable and sortable tables

2. **Visualizations**:
   - Sales trends over time
   - Regional sales distribution
   - Platform performance comparison
   - Genre popularity analysis
   - Publisher market share
   - Correlation heatmaps

3. **Prediction**:
   - Predict game sales based on input parameters
   - Classify potential sales performance
   - View prediction confidence and explanations

4. **Reports**:
   - Access detailed statistical analysis
   - View regression analysis results
   - Explore classification insights
   - Examine clustering analysis

## Machine Learning Models

The project implements several machine learning models:

### 1. Random Forest Regression

- **Purpose**: Predict total sales figures for games
- **Features**: Critic score, release year, genre frequency, publisher frequency, and regional sales
- **Implementation**: `RandomForestRegressor` from scikit-learn
- **File**: `regression_results/random_forest_model.joblib`

### 2. Naive Bayes Classification

- **Purpose**: Classify games as high-selling or low-selling
- **Features**: Same as regression model
- **Implementation**: `GaussianNB` from scikit-learn
- **File**: `naive_bayes_results/naive_bayes_model.joblib`

### 3. Decision Tree Classification

- **Purpose**: Alternative classification approach with better interpretability
- **Features**: Same as regression model
- **Implementation**: `DecisionTreeClassifier` from scikit-learn
- **File**: `decision_tree_results/decision_tree_model.joblib`

## Usage Guide

### Data Analysis Tab

1. Use the sidebar filters to select:
   - Year range
   - Game genres
   - Platforms
   - Publishers

2. Explore the visualizations that update based on your selections:
   - Sales trends chart
   - Regional distribution
   - Platform comparison
   - Genre analysis

3. View the summary metrics at the top of the page for quick insights.

### Prediction Tab

1. Enter the details for a hypothetical game:
   - Critic score
   - Release year
   - Genre
   - Publisher
   - Regional sales estimates

2. Click "Predict Sales" to get:
   - Estimated total sales
   - Sales classification (high/low)
   - Confidence score
   - Similar games from the dataset

3. Use the "Reset Form" button to clear all inputs.

### Reports Tab

Browse through different analysis reports:
- Regression analysis of sales factors
- Classification of successful games
- Clustering of similar games
- Statistical summaries and tests

## Technical Implementation

### Data Processing

The data processing pipeline includes:
- Cleaning missing values
- Converting date formats
- Encoding categorical variables
- Feature engineering (genre_freq, publisher_freq)
- Scaling numeric features

### Model Training

Models are trained using the `create_models.py` script:
1. The dataset is loaded and preprocessed
2. Features and target variables are extracted
3. Data is split into training and testing sets
4. Features are scaled using StandardScaler
5. Models are trained with optimized hyperparameters
6. Trained models and scalers are saved using joblib

### Streamlit Application

The app.py file implements the Streamlit interface:
- Responsive layout with sidebar filters
- Interactive Plotly visualizations
- Custom CSS styling
- Caching for performance optimization
- Error handling and input validation
- Tabbed interface for different functionalities

## Troubleshooting

### Common Issues

1. **Application won't start**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Use `python -m streamlit run app.py` instead of `streamlit run app.py`
   - Check for any error messages in the console

2. **Models not loading**:
   - Verify that all model files exist in their respective directories
   - Run `python create_models.py` to regenerate the models
   - Check for compatibility issues with scikit-learn versions

3. **Visualization errors**:
   - Ensure the data files are properly formatted
   - Check for missing columns in the dataset
   - Verify that the filters aren't excluding all data points

4. **Prediction errors**:
   - Ensure input values are within reasonable ranges
   - Check that all required features are provided
   - Verify that the feature names match between training and prediction 