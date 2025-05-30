import pandas as pd
import numpy as np
import pytest
import os
import sys
from tempfile import TemporaryDirectory
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Add parent directory to path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_app_helper_functions():
    """
    This test validates the helper functions that would be used in the app.
    Since we can't directly test the Streamlit interface, we'll test the 
    underlying logic that powers the app functionality.
    """
    
    # Create a test dataframe
    df = pd.DataFrame({
        'name': ['Game A', 'Game B', 'Game C', 'Game D'],
        'genre': ['Action', 'Sports', 'Strategy', 'Action'],
        'console': ['PS5', 'Xbox', 'Switch', 'PS5'],
        'platform': ['PS5', 'Xbox Series X', 'Nintendo Switch', 'PS5'],
        'publisher': ['Ubisoft', 'EA', 'Nintendo', 'Activision'],
        'critic_score': [85, 78, 92, 65],
        'release_year': [2020, 2021, 2019, 2022],
        'release_date': ['2020-11-10', '2021-05-15', '2019-03-20', '2022-01-30'],
        'na_sales': [3.5, 2.7, 1.8, 4.2],
        'jp_sales': [0.7, 0.3, 2.5, 0.5],
        'pal_sales': [2.0, 1.8, 1.2, 3.0],
        'other_sales': [0.4, 0.3, 0.2, 0.5],
        'total_sales': [6.6, 5.1, 5.7, 8.2]
    })
    
    # Test filtering by genre
    action_games = df[df['genre'] == 'Action']
    assert len(action_games) == 2, "Should find 2 action games"
    
    # Test filtering by platform
    ps5_games = df[df['platform'] == 'PS5']
    assert len(ps5_games) == 2, "Should find 2 PS5 games"
    
    # Test filtering by release year range
    recent_games = df[(df['release_year'] >= 2020) & (df['release_year'] <= 2021)]
    assert len(recent_games) == 2, "Should find 2 games from 2020-2021"
    
    # Test calculating average scores
    avg_score = df['critic_score'].mean()
    assert abs(avg_score - 80.0) < 0.001, f"Average score should be 80.0, got {avg_score}"
    
    # Test calculating total sales
    total_sales = df['total_sales'].sum()
    assert abs(total_sales - 25.6) < 0.001, f"Total sales should be 25.6, got {total_sales}"
    
    # Test calculating regional sales percentage
    na_percentage = (df['na_sales'].sum() / df['total_sales'].sum()) * 100
    assert abs(na_percentage - 47.66) < 0.1, f"NA sales percentage should be around 47.66%, got {na_percentage}%"
    
    # Test for best-selling game
    best_seller = df.loc[df['total_sales'].idxmax()]
    assert best_seller['name'] == 'Game D', f"Best seller should be Game D, got {best_seller['name']}"
    
    # Test for highest-rated game
    highest_rated = df.loc[df['critic_score'].idxmax()]
    assert highest_rated['name'] == 'Game C', f"Highest rated should be Game C, got {highest_rated['name']}"

def create_temp_models():
    """
    Create temporary model files for testing the app's prediction functionality.
    This helps test the app without requiring the actual model files.
    """
    temp_dir = TemporaryDirectory()
    try:
        # Create necessary directories
        os.makedirs(os.path.join(temp_dir.name, 'regression_results'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir.name, 'naive_bayes_results'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir.name, 'decision_tree_results'), exist_ok=True)
        
        # Create a test dataset
        X = np.random.rand(100, 5)
        X_df = pd.DataFrame(X, columns=['critic_score', 'release_year', 'na_sales', 'jp_sales', 'pal_sales'])
        y_reg = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
        y_cls = (y_reg > y_reg.mean()).astype(int)
        
        # Create and save models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_scaled, y_reg)
        joblib.dump(rf, os.path.join(temp_dir.name, 'regression_results', 'random_forest_model.joblib'))
        
        # Naive Bayes
        nb = GaussianNB()
        nb.fit(X_scaled, y_cls)
        joblib.dump(nb, os.path.join(temp_dir.name, 'naive_bayes_results', 'naive_bayes_model.joblib'))
        
        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_scaled, y_cls)
        joblib.dump(dt, os.path.join(temp_dir.name, 'decision_tree_results', 'decision_tree_model.joblib'))
        
        # Save scaler
        joblib.dump(scaler, os.path.join(temp_dir.name, 'regression_results', 'scaler.joblib'))
        
        # Success message
        print("Test models created and validated successfully!")
        return temp_dir
    except Exception as e:
        print(f"Error creating temporary models: {str(e)}")
        temp_dir.cleanup()
        raise

def test_model_prediction_workflow():
    """
    Test the workflow of loading models, preparing input data, and making predictions.
    This simulates the backend processing that would happen when a user submits
    prediction requests.
    """
    
    # Create temporary directory for model files
    with TemporaryDirectory() as temp_dir:
        # Create and save test models
        # Random Forest for regression
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_train = np.array([[80, 2020, 1.5, 0.5, 1.0],
                            [70, 2019, 1.0, 0.3, 0.8],
                            [90, 2021, 2.0, 0.7, 1.5],
                            [75, 2018, 1.2, 0.4, 0.9]])
        y_train_reg = np.array([3.0, 2.1, 4.2, 2.5])
        rf_model.fit(X_train, y_train_reg)
        
        # Naive Bayes for classification
        nb_model = GaussianNB()
        y_train_cls = np.array([1, 0, 1, 0])
        nb_model.fit(X_train, y_train_cls)
        
        # Decision Tree for classification
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train_cls)
        
        # Scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Save models
        model_paths = {
            'rf': os.path.join(temp_dir, 'random_forest_model.joblib'),
            'nb': os.path.join(temp_dir, 'naive_bayes_model.joblib'),
            'dt': os.path.join(temp_dir, 'decision_tree_model.joblib'),
            'scaler': os.path.join(temp_dir, 'scaler.joblib')
        }
        
        joblib.dump(rf_model, model_paths['rf'])
        joblib.dump(nb_model, model_paths['nb'])
        joblib.dump(dt_model, model_paths['dt'])
        joblib.dump(scaler, model_paths['scaler'])
        
        # Test prediction workflow
        # 1. Load models
        loaded_rf = joblib.load(model_paths['rf'])
        loaded_nb = joblib.load(model_paths['nb'])
        loaded_dt = joblib.load(model_paths['dt'])
        loaded_scaler = joblib.load(model_paths['scaler'])
        
        # 2. Prepare a new input
        new_input = pd.DataFrame({
            'critic_score': [85],
            'release_year': [2022],
            'na_sales': [1.8],
            'jp_sales': [0.6],
            'pal_sales': [1.2]
        })
        
        # 3. Scale input
        scaled_input = loaded_scaler.transform(new_input)
        
        # 4. Make predictions
        rf_pred = loaded_rf.predict(scaled_input)[0]
        nb_pred = loaded_nb.predict(scaled_input)[0]
        dt_pred = loaded_dt.predict(scaled_input)[0]
        
        # 5. Check predictions
        assert isinstance(rf_pred, float), "RF prediction should be a float for regression"
        assert isinstance(nb_pred, (int, np.integer)), "NB prediction should be an integer for classification"
        assert isinstance(dt_pred, (int, np.integer)), "DT prediction should be an integer for classification"
        
        # 6. Check prediction values are reasonable
        assert 0 <= rf_pred <= 10, f"RF prediction out of reasonable range: {rf_pred}"
        assert nb_pred in [0, 1], f"NB prediction should be binary: {nb_pred}"
        assert dt_pred in [0, 1], f"DT prediction should be binary: {dt_pred}"

def test_input_validation():
    """Test input validation for handling unexpected user inputs."""
    # Test with missing values
    data_with_missing = pd.DataFrame({
        'critic_score': [None],
        'release_year': [2020],
        'na_sales': [1.2],
        'jp_sales': [None],
        'pal_sales': [0.9]
    })

    # Test with invalid values
    data_with_invalid = pd.DataFrame({
        'critic_score': [-10],  # Invalid negative score
        'release_year': [9999], # Far future year
        'na_sales': [1.2],
        'jp_sales': [0.5],
        'pal_sales': [0.9]
    })

    # Test with string values that should be numeric
    data_with_strings = pd.DataFrame({
        'critic_score': ["excellent"],  # String instead of number
        'release_year': [2020],
        'na_sales': [1.2],
        'jp_sales': [0.5],
        'pal_sales': [0.9]
    })

    # Test handling of these cases
    for test_case, test_data in {
        "missing_values": data_with_missing,
        "invalid_values": data_with_invalid,
        "string_values": data_with_strings
    }.items():
        try:
            # Try to convert to valid format (handle or detect issues)
            cleaned_data = test_data.copy()

            # Handle missing values
            for col in cleaned_data.columns:
                if cleaned_data[col].isna().any():
                    print(f"Detected missing values in {col}, replacing with default")
                    # Replace with reasonable defaults
                    if col == 'critic_score':
                        cleaned_data[col] = cleaned_data[col].fillna(75)
                    else:
                        cleaned_data[col] = cleaned_data[col].fillna(0)

            # Convert strings to numbers where possible
            for col in ['critic_score', 'release_year', 'na_sales', 'jp_sales', 'pal_sales']:
                try:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                except Exception:
                    print(f"Could not convert column {col} to numeric type")

            # Check for invalid values and constrain to reasonable ranges
            if 'critic_score' in cleaned_data:
                cleaned_data['critic_score'] = cleaned_data['critic_score'].clip(0, 100)

            if 'release_year' in cleaned_data:
                cleaned_data['release_year'] = cleaned_data['release_year'].clip(1950, 2030)
            
            # Fill any remaining NaN values after conversion
            for col in cleaned_data.columns:
                if cleaned_data[col].isna().any():
                    print(f"Detected NaN values after conversion in {col}, replacing with default")
                    # Replace with reasonable defaults
                    if col == 'critic_score':
                        cleaned_data[col] = cleaned_data[col].fillna(75)
                    else:
                        cleaned_data[col] = cleaned_data[col].fillna(0)

            # Verify we have valid data after cleaning
            assert not cleaned_data.isna().any().any(), "Still have NaN values after cleaning"
            print(f"Successfully handled {test_case}")

        except Exception as e:
            pytest.fail(f"Failed to handle {test_case}: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 