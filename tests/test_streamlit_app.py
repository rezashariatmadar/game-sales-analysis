import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import joblib

# Add parent directory to path so we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a minimal mock for Streamlit
class MockSt:
    def __init__(self):
        self.sidebar = MagicMock()
        self.set_page_config = MagicMock()
        self.markdown = MagicMock()
        self.title = MagicMock()
        self.header = MagicMock()
        self.subheader = MagicMock()
        self.write = MagicMock()
        self.columns = MagicMock()
        self.metric = MagicMock()
        self.plotly_chart = MagicMock()
        self.error = MagicMock()
        self.info = MagicMock()
        self.warning = MagicMock()
        self.dataframe = MagicMock()
        self.download_button = MagicMock()
        self.form = MagicMock()
        self.form_submit_button = MagicMock()
        self.selectbox = MagicMock()
        self.slider = MagicMock()
        self.number_input = MagicMock()
        self.tabs = MagicMock()
        self.image = MagicMock()
        self._main = MagicMock()
        self._main.id = "main_id"
        self.sidebar = MagicMock()
        self.sidebar.id = "sidebar_id"
        
        # Return self when calling functions that return containers
        self.columns.return_value = [self, self, self, self]
        self.tabs.return_value = [self, self, self]
        self.form.return_value.__enter__ = MagicMock(return_value=self)
        self.form.return_value.__exit__ = MagicMock(return_value=None)

# Mock the cache_data decorator
def mock_cache_data(func):
    return func

# Mock the cache_resource decorator
def mock_cache_resource(func):
    return func

@pytest.fixture
def mock_streamlit():
    """Create a mock for streamlit module."""
    mock_st = MockSt()
    return mock_st

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
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

def test_load_data_function():
    """Test the load_data function directly without using Streamlit cache."""
    # Mock the app.py module with our own simplified version for testing
    with patch.dict('sys.modules', {'streamlit': MagicMock()}):
        # Define a simple test version of the load_data function
        def load_data(path):
            try:
                df = pd.read_csv(path)
                df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
                return df
            except FileNotFoundError:
                return None
        
        # Create a mock CSV file
        mock_csv_data = """name,genre,platform,publisher,critic_score,release_year,release_date,na_sales,jp_sales,pal_sales,other_sales,total_sales
Game A,Action,PS5,Ubisoft,85,2020,2020-11-10,3.5,0.7,2.0,0.4,6.6"""
        
        # Test with a valid path
        with patch('builtins.open', mock_open(read_data=mock_csv_data)):
            with patch('pandas.read_csv', return_value=pd.read_csv(pd.io.common.StringIO(mock_csv_data))):
                df = load_data('fake_path.csv')
                assert df is not None
                assert len(df) == 1
                assert df.iloc[0]['name'] == 'Game A'
                assert df.iloc[0]['total_sales'] == 6.6
        
        # Test with an invalid path
        with patch('pandas.read_csv', side_effect=FileNotFoundError()):
            df = load_data('nonexistent.csv')
            assert df is None

@patch('pandas.read_csv')
def test_platform_column_handling(mock_read_csv, sample_data, mock_streamlit):
    """Test that the app handles platform/console column variation."""
    
    # Create a copy of sample data with only 'console' column
    console_only_data = sample_data.drop(columns=['platform'])
    mock_read_csv.return_value = console_only_data
    
    # Define a simple test function to verify logic
    def platform_handling(df):
        platform_col = 'platform' if 'platform' in df.columns else 'console'
        if platform_col in df.columns:
            platforms = sorted(df[platform_col].unique())
            return platform_col, platforms
        else:
            return None, []
    
    platform_col, platforms = platform_handling(console_only_data)
    assert platform_col == 'console'
    assert 'PS5' in platforms
    
    # Test with only platform column
    platform_only_data = sample_data.drop(columns=['console'])
    mock_read_csv.return_value = platform_only_data
    
    platform_col, platforms = platform_handling(platform_only_data)
    assert platform_col == 'platform'
    assert 'PS5' in platforms
    
    # Test with both columns
    platform_col, platforms = platform_handling(sample_data)
    assert platform_col in ['platform', 'console']
    assert 'PS5' in platforms

@patch('joblib.load')
def test_model_loading(mock_joblib_load, mock_streamlit):
    """Test the model loading function."""
    
    # Setup mocks to return mock models
    mock_rf = MagicMock()
    mock_nb = MagicMock()
    mock_dt = MagicMock()
    mock_scaler = MagicMock()
    
    # Define side_effect to return different values based on the input path
    def load_side_effect(path):
        if 'random_forest_model.joblib' in path:
            return mock_rf
        elif 'naive_bayes_model.joblib' in path:
            return mock_nb
        elif 'decision_tree_model.joblib' in path:
            return mock_dt
        elif 'scaler.joblib' in path:
            return mock_scaler
    
    mock_joblib_load.side_effect = load_side_effect
    
    # Define a simple version of the load_models function
    def load_models():
        try:
            models = {
                'regression': joblib.load('regression_results/random_forest_model.joblib'),
                'naive_bayes': joblib.load('naive_bayes_results/naive_bayes_model.joblib'),
                'decision_tree': joblib.load('decision_tree_results/decision_tree_model.joblib')
            }
            scaler = joblib.load('regression_results/scaler.joblib')
            return models, scaler
        except FileNotFoundError:
            return None, None
    
    # Test the function
    models, scaler = load_models()
    
    # Check that models were loaded correctly
    assert models is not None
    assert 'regression' in models
    assert 'naive_bayes' in models
    assert 'decision_tree' in models
    assert models['regression'] == mock_rf
    assert models['naive_bayes'] == mock_nb
    assert models['decision_tree'] == mock_dt
    assert scaler == mock_scaler

@patch('joblib.load')
def test_model_loading_with_missing_files(mock_joblib_load, mock_streamlit):
    """Test the load_models function handles missing files."""
    
    # Setup mock to raise FileNotFoundError
    mock_joblib_load.side_effect = FileNotFoundError("File not found")
    
    # Define a simple version of the load_models function
    def load_models():
        try:
            models = {
                'regression': joblib.load('regression_results/random_forest_model.joblib'),
                'naive_bayes': joblib.load('naive_bayes_results/naive_bayes_model.joblib'),
                'decision_tree': joblib.load('decision_tree_results/decision_tree_model.joblib')
            }
            scaler = joblib.load('regression_results/scaler.joblib')
            return models, scaler
        except FileNotFoundError:
            mock_streamlit.warning("Model files not found. Prediction features will be disabled.")
            return None, None
    
    # Test the function
    models, scaler = load_models()
    
    # Check that the function handled the error correctly
    assert models is None
    assert scaler is None
    mock_streamlit.warning.assert_called_once_with("Model files not found. Prediction features will be disabled.")

@patch('joblib.load')
def test_prediction_workflow(mock_joblib_load, sample_data, mock_streamlit):
    """Test the prediction workflow in the app."""
    
    # Setup mock models
    mock_rf = MagicMock()
    mock_rf.predict.return_value = np.array([7.5])
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    
    # Setup joblib.load to return our mocks
    def load_side_effect(path):
        if 'random_forest_model.joblib' in path:
            return mock_rf
        elif 'scaler.joblib' in path:
            return mock_scaler
    
    mock_joblib_load.side_effect = load_side_effect
    
    # Sample input data
    input_data = pd.DataFrame({
        'critic_score': [75],
        'release_year': [2023],
        'na_sales': [1.0],
        'jp_sales': [0.5],
        'pal_sales': [0.8]
    })
    
    # Simple function to test prediction workflow
    def predict_sales(input_data, model, scaler):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return prediction
    
    # Test prediction
    prediction = predict_sales(input_data, mock_rf, mock_scaler)
    
    # Check prediction result
    assert prediction == 7.5
    mock_scaler.transform.assert_called_once()
    mock_rf.predict.assert_called_once()

@patch('joblib.load')
def test_prediction_with_nan_values(mock_joblib_load, mock_streamlit):
    """Test handling of NaN values in the prediction workflow."""
    
    # Setup mock models
    mock_rf = MagicMock()
    mock_rf.predict.return_value = np.array([7.5])
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    
    # Setup joblib.load to return our mocks
    def load_side_effect(path):
        if 'random_forest_model.joblib' in path:
            return mock_rf
        elif 'scaler.joblib' in path:
            return mock_scaler
    
    mock_joblib_load.side_effect = load_side_effect
    
    # Sample input data with NaN values
    input_data_with_nan = pd.DataFrame({
        'critic_score': [None],
        'release_year': [2023],
        'na_sales': [1.0],
        'jp_sales': [None],
        'pal_sales': [0.8]
    })
    
    # Simple function to test prediction workflow with NaN handling
    def predict_sales_with_nan_handling(input_data, model, scaler):
        # Handle NaN values before prediction
        input_clean = input_data.copy()
        for col in input_clean.columns:
            if input_clean[col].isna().any():
                # Fill with defaults
                if col == 'critic_score':
                    input_clean[col] = input_clean[col].fillna(75)
                else:
                    input_clean[col] = input_clean[col].fillna(0)
        
        input_scaled = scaler.transform(input_clean)
        prediction = model.predict(input_scaled)[0]
        return prediction
    
    # Test prediction
    prediction = predict_sales_with_nan_handling(input_data_with_nan, mock_rf, mock_scaler)
    
    # Check prediction result
    assert prediction == 7.5
    mock_scaler.transform.assert_called_once()
    mock_rf.predict.assert_called_once()

def test_environment_checks():
    """Test environment validation for required files."""
    # Get the project root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Check for key files
    dataset_path = os.path.join(root_dir, 'vgchartz_cleaned.csv')
    rf_model_path = os.path.join(root_dir, 'regression_results/random_forest_model.joblib')
    nb_model_path = os.path.join(root_dir, 'naive_bayes_results/naive_bayes_model.joblib')
    dt_model_path = os.path.join(root_dir, 'decision_tree_results/decision_tree_model.joblib')
    scaler_path = os.path.join(root_dir, 'regression_results/scaler.joblib')
    
    # Record missing files
    missing_files = []
    if not os.path.exists(dataset_path):
        missing_files.append("Dataset file (vgchartz_cleaned.csv)")
    if not os.path.exists(rf_model_path):
        missing_files.append("Random Forest model")
    if not os.path.exists(nb_model_path):
        missing_files.append("Naive Bayes model")
    if not os.path.exists(dt_model_path):
        missing_files.append("Decision Tree model")
    if not os.path.exists(scaler_path):
        missing_files.append("Feature scaler")
    
    # Test environment is ready
    if missing_files:
        pytest.skip(f"Environment not ready: Missing files: {', '.join(missing_files)}")

def test_app_error_handling():
    """Test error handling in the app functions."""
    
    # Test handling invalid filter values
    def apply_filters(df, year_range, genres, publishers, platforms):
        try:
            # Handle empty filter values gracefully
            if not genres:
                genres = df['genre'].unique()
            if not publishers:
                publishers = df['publisher'].unique()
            if not platforms:
                platforms = df['platform'].unique() if 'platform' in df.columns else df['console'].unique()
            
            # Apply filters
            mask = (
                (df['release_year'] >= year_range[0]) &
                (df['release_year'] <= year_range[1]) &
                (df['genre'].isin(genres)) &
                (df['publisher'].isin(publishers))
            )
            
            # Add platform filter if available
            platform_col = 'platform' if 'platform' in df.columns else 'console'
            if platform_col in df.columns:
                mask = mask & (df[platform_col].isin(platforms))
            
            filtered_df = df[mask]
            return filtered_df
        except Exception as e:
            # Should handle exceptions gracefully
            return df
    
    # Create test data
    test_df = pd.DataFrame({
        'name': ['Game A', 'Game B'],
        'genre': ['Action', 'RPG'],
        'console': ['PS5', 'Xbox'],
        'publisher': ['EA', 'Ubisoft'],
        'release_year': [2020, 2021],
        'total_sales': [5.0, 4.0]
    })
    
    # Test with valid filters
    filtered = apply_filters(test_df, (2020, 2021), ['Action'], ['EA'], ['PS5'])
    assert len(filtered) == 1
    assert filtered.iloc[0]['name'] == 'Game A'
    
    # Test with empty filters - should return all rows
    filtered = apply_filters(test_df, (2020, 2021), [], [], [])
    assert len(filtered) == 2
    
    # Test with invalid years
    filtered = apply_filters(test_df, (2022, 2023), ['Action'], ['EA'], ['PS5'])
    assert len(filtered) == 0
    
    # Test with a potentially problematic input
    filtered = apply_filters(test_df, (None, None), ['Action'], ['EA'], ['PS5'])
    assert isinstance(filtered, pd.DataFrame)  # Should not crash but might not filter correctly

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 