#!/usr/bin/env python3
"""
Generate ML Models for Game Sales Analysis

This script creates and saves the machine learning models needed for the application.
It should be run before using the Streamlit app to ensure all models are available.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories for models and results."""
    directories = [
        'models',
        'results/regression_results',
        'results/decision_tree_results', 
        'results/naive_bayes_results',
        'results/clustering_results',
        'results/hierarchical_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def load_and_preprocess_data():
    """Load and preprocess the video game sales data."""
    print("📊 Loading and preprocessing data...")
    
    try:
        # Load the cleaned data
        data_path = 'data/processed/vgchartz_cleaned.csv'
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure the data processing pipeline has been run first.")
            return None
        
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} games from dataset")
        
        # Basic preprocessing
        df = df.dropna(subset=['total_sales', 'critic_score'])
        df = df[df['total_sales'] > 0]  # Remove games with no sales
        
        print(f"✓ After preprocessing: {len(df)} games remaining")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def engineer_features(df):
    """Engineer features for machine learning models."""
    print("🔧 Engineering features...")
    
    try:
        # Create a copy to avoid modifying original
        df_eng = df.copy()
        
        # Regional sales ratios
        df_eng['na_sales_ratio'] = df_eng['na_sales'] / (df_eng['total_sales'] + 1e-8)
        df_eng['jp_sales_ratio'] = df_eng['jp_sales'] / (df_eng['total_sales'] + 1e-8)
        df_eng['pal_sales_ratio'] = df_eng['pal_sales'] / (df_eng['total_sales'] + 1e-8)
        
        # Temporal features
        current_year = 2024
        df_eng['game_age'] = current_year - df_eng['release_year']
        df_eng['sales_per_year'] = df_eng['total_sales'] / (df_eng['game_age'] + 1)
        
        # Frequency encoding for categorical variables
        df_eng['console_freq'] = df_eng['console'].map(df_eng['console'].value_counts(normalize=True))
        df_eng['genre_freq'] = df_eng['genre'].map(df_eng['genre'].value_counts(normalize=True))
        df_eng['publisher_freq'] = df_eng['publisher'].map(df_eng['publisher'].value_counts(normalize=True))
        
        # Fill missing frequency values
        df_eng['console_freq'] = df_eng['console_freq'].fillna(0.001)
        df_eng['genre_freq'] = df_eng['genre_freq'].fillna(0.001)
        df_eng['publisher_freq'] = df_eng['publisher_freq'].fillna(0.001)
        
        print("✓ Feature engineering completed")
        return df_eng
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return None

def prepare_training_data(df):
    """Prepare features and target variables for training."""
    print("🎯 Preparing training data...")
    
    try:
        # Select features for ML models
        feature_columns = [
            'critic_score', 'release_year', 'console_freq', 'genre_freq', 
            'publisher_freq', 'na_sales_ratio', 'jp_sales_ratio', 'pal_sales_ratio'
        ]
        
        # Ensure all features exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return None, None, None
        
        X = df[feature_columns].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Create target variables
        y_regression = df['total_sales']
        
        # Create classification target (high vs low sales)
        sales_median = y_regression.median()
        y_classification = (y_regression > sales_median).astype(int)
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Regression target shape: {y_regression.shape}")
        print(f"✓ Classification target shape: {y_classification.shape}")
        
        return X, y_regression, y_classification
        
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        return None, None, None

def train_random_forest_regression(X, y):
    """Train Random Forest regression model."""
    print("🌲 Training Random Forest regression model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ Random Forest R² Score: {r2:.4f}")
        
        # Save model and scaler
        model_path = 'results/regression_results/random_forest_model.joblib'
        scaler_path = 'results/regression_results/preprocessor.joblib'
        
        joblib.dump(rf_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        
        return rf_model, scaler
        
    except Exception as e:
        print(f"❌ Error training Random Forest: {e}")
        return None, None

def train_decision_tree_classification(X, y):
    """Train Decision Tree classification model."""
    print("🌳 Training Decision Tree classification model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        dt_model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        dt_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = dt_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Decision Tree Accuracy: {accuracy:.4f}")
        
        # Save model and scaler
        model_path = 'results/decision_tree_results/decision_tree_model.joblib'
        scaler_path = 'results/decision_tree_results/preprocessor.joblib'
        
        joblib.dump(dt_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        
        return dt_model, scaler
        
    except Exception as e:
        print(f"❌ Error training Decision Tree: {e}")
        return None, None

def train_naive_bayes_classification(X, y):
    """Train Naive Bayes classification model."""
    print("📊 Training Naive Bayes classification model...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        nb_model = GaussianNB(var_smoothing=1e-9)
        nb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = nb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Naive Bayes Accuracy: {accuracy:.4f}")
        
        # Save model and scaler
        model_path = 'results/naive_bayes_results/naive_bayes_model.joblib'
        scaler_path = 'results/naive_bayes_results/preprocessor.joblib'
        
        joblib.dump(nb_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        
        return nb_model, scaler
        
    except Exception as e:
        print(f"❌ Error training Naive Bayes: {e}")
        return None, None

def create_dummy_models():
    """Create dummy models if training fails."""
    print("⚠️  Creating dummy models for demonstration...")
    
    try:
        # Create a simple dummy model
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        dummy_scaler = StandardScaler()
        
        # Create dummy data for fitting
        X_dummy = np.random.rand(100, 8)
        y_dummy = np.random.rand(100)
        
        dummy_scaler.fit(X_dummy)
        dummy_model.fit(dummy_scaler.transform(X_dummy), y_dummy)
        
        # Save dummy models
        joblib.dump(dummy_model, 'models/dummy_regression_model.joblib')
        joblib.dump(dummy_scaler, 'models/dummy_scaler.joblib')
        
        print("✓ Dummy models created for demonstration")
        return True
        
    except Exception as e:
        print(f"❌ Error creating dummy models: {e}")
        return False

def main():
    """Main function to generate all models."""
    print("🎮 Game Sales Analysis - Model Generation")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    if df is None:
        print("❌ Failed to load data. Creating dummy models instead.")
        create_dummy_models()
        return
    
    # Engineer features
    df_eng = engineer_features(df)
    if df_eng is None:
        print("❌ Failed to engineer features. Creating dummy models instead.")
        create_dummy_models()
        return
    
    # Prepare training data
    X, y_reg, y_clf = prepare_training_data(df_eng)
    if X is None:
        print("❌ Failed to prepare training data. Creating dummy models instead.")
        create_dummy_models()
        return
    
    # Train models
    print("\n🚀 Training Machine Learning Models...")
    print("-" * 40)
    
    # Train Random Forest regression
    rf_model, rf_scaler = train_random_forest_regression(X, y_reg)
    
    # Train Decision Tree classification
    dt_model, dt_scaler = train_decision_tree_classification(X, y_clf)
    
    # Train Naive Bayes classification
    nb_model, nb_scaler = train_naive_bayes_classification(X, y_clf)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Model Generation Complete!")
    print("=" * 50)
    
    if all([rf_model, dt_model, nb_model]):
        print("✅ All models generated successfully!")
        print("✅ You can now run the Streamlit application")
        print("✅ Run: streamlit run app.py")
    else:
        print("⚠️  Some models failed to generate")
        print("⚠️  Creating dummy models for demonstration")
        create_dummy_models()
        print("✅ Dummy models created - app will run with limited functionality")

if __name__ == "__main__":
    main()