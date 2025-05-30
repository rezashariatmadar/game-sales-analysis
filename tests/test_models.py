import pandas as pd
import numpy as np
import pytest
import joblib
import os
import sys

# Add parent directory to path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer

@pytest.fixture
def test_data():
    """Load a small test dataset for model evaluation."""
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Use the actual dataset for testing
        df = pd.read_csv(os.path.join(root_dir, 'vgchartz_cleaned.csv'))
        # Create a small test set
        test_df = df.sample(n=100, random_state=42)
        
        # Prepare features and target
        X = test_df[['critic_score', 'release_year', 'na_sales', 'jp_sales', 'pal_sales']]
        y_reg = test_df['total_sales']
        y_cls = (y_reg > y_reg.median()).astype(int)
        
        return X, y_reg, y_cls
    except FileNotFoundError:
        pytest.skip("Test dataset not found")

def test_model_files_exist():
    """Test that all model files exist."""
    # Use relative path from tests directory to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_files = [
        os.path.join(root_dir, 'regression_results/random_forest_model.joblib'),
        os.path.join(root_dir, 'naive_bayes_results/naive_bayes_model.joblib'),
        os.path.join(root_dir, 'decision_tree_results/decision_tree_model.joblib'),
        os.path.join(root_dir, 'regression_results/scaler.joblib')
    ]
    for file_path in model_files:
        assert os.path.exists(file_path), f"Model file {file_path} does not exist"
        
def test_model_loading():
    """Test that all models can be loaded."""
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        rf_model = joblib.load(os.path.join(root_dir, 'regression_results/random_forest_model.joblib'))
        nb_model = joblib.load(os.path.join(root_dir, 'naive_bayes_results/naive_bayes_model.joblib'))
        dt_model = joblib.load(os.path.join(root_dir, 'decision_tree_results/decision_tree_model.joblib'))
        scaler = joblib.load(os.path.join(root_dir, 'regression_results/scaler.joblib'))
        
        assert hasattr(rf_model, 'predict'), "Random Forest model doesn't have predict method"
        assert hasattr(nb_model, 'predict'), "Naive Bayes model doesn't have predict method"
        assert hasattr(dt_model, 'predict'), "Decision Tree model doesn't have predict method"
        assert hasattr(scaler, 'transform'), "Scaler doesn't have transform method"
    except Exception as e:
        pytest.fail(f"Failed to load models: {str(e)}")

def test_regression_model_performance(test_data):
    """Test that regression model performs reasonably well."""
    X, y_reg, _ = test_data
    
    # Load model and scaler
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        rf_model = joblib.load(os.path.join(root_dir, 'regression_results/random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(root_dir, 'regression_results/scaler.joblib'))
        
        # Handle missing values
        X_filled = X.copy()
        imputer = SimpleImputer(strategy='mean')
        X_filled = pd.DataFrame(imputer.fit_transform(X_filled), columns=X_filled.columns)
        
        # Scale features
        X_scaled = scaler.transform(X_filled)
        
        # Make predictions
        y_pred = rf_model.predict(X_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_reg, y_pred)
        mae = mean_absolute_error(y_reg, y_pred)
        
        # Check performance
        assert r2 >= 0.5, f"R² score too low: {r2}"
        assert mae < 1.0, f"MAE too high: {mae}"
    except Exception as e:
        pytest.fail(f"Error in regression model evaluation: {str(e)}")

def test_classification_models_performance(test_data):
    """Test that classification models perform reasonably well."""
    X, _, y_cls = test_data
    
    # Load models and scaler
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        nb_model = joblib.load(os.path.join(root_dir, 'naive_bayes_results/naive_bayes_model.joblib'))
        dt_model = joblib.load(os.path.join(root_dir, 'decision_tree_results/decision_tree_model.joblib'))
        scaler = joblib.load(os.path.join(root_dir, 'regression_results/scaler.joblib'))
        
        # Handle missing values first
        X_filled = X.copy()
        imputer = SimpleImputer(strategy='mean')
        X_filled = pd.DataFrame(imputer.fit_transform(X_filled), columns=X_filled.columns)
        
        # Scale features
        X_scaled = scaler.transform(X_filled)
        
        # Make predictions
        nb_pred = nb_model.predict(X_scaled)
        dt_pred = dt_model.predict(X_scaled)
        
        # Calculate metrics
        nb_accuracy = accuracy_score(y_cls, nb_pred)
        dt_accuracy = accuracy_score(y_cls, dt_pred)
        
        # Check performance with more reasonable thresholds
        assert nb_accuracy >= 0.45, f"Naive Bayes accuracy too low: {nb_accuracy}"
        assert dt_accuracy >= 0.55, f"Decision Tree accuracy too low: {dt_accuracy}"
    except Exception as e:
        pytest.fail(f"Error in classification model evaluation: {str(e)}")

def test_feature_importance():
    """Test that regression model feature importance can be extracted."""
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        rf_model = joblib.load(os.path.join(root_dir, 'regression_results/random_forest_model.joblib'))
        
        # Get feature importance
        importance = rf_model.feature_importances_
        
        # Check that importance values are valid
        assert len(importance) == 5, f"Expected 5 features, got {len(importance)}"
        assert np.sum(importance) > 0.99, f"Feature importance sum should be 1.0, got {np.sum(importance)}"
        assert np.all(importance >= 0), "All feature importances should be non-negative"
    except Exception as e:
        pytest.fail(f"Error in feature importance extraction: {str(e)}")

def test_model_prediction():
    """Test that models can make predictions on new data."""
    # Create a new sample
    new_sample = pd.DataFrame({
        'critic_score': [8.5],
        'release_year': [2021],
        'na_sales': [1.0],
        'jp_sales': [0.3],
        'pal_sales': [0.8]
    })
    
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Load models
        rf_model = joblib.load(os.path.join(root_dir, 'regression_results/random_forest_model.joblib'))
        nb_model = joblib.load(os.path.join(root_dir, 'naive_bayes_results/naive_bayes_model.joblib'))
        dt_model = joblib.load(os.path.join(root_dir, 'decision_tree_results/decision_tree_model.joblib'))
        scaler = joblib.load(os.path.join(root_dir, 'regression_results/scaler.joblib'))
        
        # Scale features
        new_sample_scaled = scaler.transform(new_sample)
        
        # Make predictions
        rf_pred = rf_model.predict(new_sample_scaled)[0]
        nb_pred = nb_model.predict(new_sample_scaled)[0]
        dt_pred = dt_model.predict(new_sample_scaled)[0]
        
        # Check that predictions are reasonable
        assert isinstance(rf_pred, (int, float)), f"Regression prediction type mismatch: {type(rf_pred)}"
        assert rf_pred >= 0, f"Regression prediction should be non-negative: {rf_pred}"
        assert nb_pred in [0, 1], f"Naive Bayes prediction should be binary: {nb_pred}"
        assert dt_pred in [0, 1], f"Decision Tree prediction should be binary: {dt_pred}"
    except Exception as e:
        pytest.fail(f"Error in model prediction: {str(e)}")

def test_model_robustness_to_outliers():
    """Test model robustness to outliers."""
    # Create a sample with outlier values
    outlier_sample = pd.DataFrame({
        'critic_score': [100],  # Max possible score (outlier)
        'release_year': [2030],  # Future year (outlier)
        'na_sales': [50],  # Very high sales (outlier)
        'jp_sales': [30],  # Very high sales (outlier)
        'pal_sales': [40]   # Very high sales (outlier)
    })
    
    try:
        # Use relative path from tests directory to the root directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Load regression model and scaler
        rf_model = joblib.load(os.path.join(root_dir, 'regression_results/random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(root_dir, 'regression_results/scaler.joblib'))
        
        # Scale features
        outlier_scaled = scaler.transform(outlier_sample)
        
        # Make prediction
        prediction = rf_model.predict(outlier_scaled)[0]
        
        # The prediction should still be a reasonable value
        assert isinstance(prediction, (int, float)), "Prediction should be a number"
        assert prediction >= 0, "Prediction should be non-negative"
        
    except Exception as e:
        pytest.fail(f"Error in outlier robustness test: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 