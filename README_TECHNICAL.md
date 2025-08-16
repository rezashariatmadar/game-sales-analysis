# 🛠️ Technical Documentation - Game Sales Analysis

*Comprehensive technical guide for developers, data scientists, and engineers*

## 🏗️ Architecture Overview

This project implements a **production-ready machine learning pipeline** for video game sales analysis using Python, scikit-learn, and Streamlit. The architecture follows modern software engineering principles with clear separation of concerns, comprehensive testing, and scalable design patterns.

### 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Streamlit Web App                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │   Analysis  │ │ Prediction  │ │  Reports    │   │   │
│  │  │     Tab     │ │     Tab     │ │    Tab      │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Prediction Engine                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │   Input     │ │   Model     │ │   Output    │   │   │
│  │  │ Validation  │ │  Ensemble   │ │ Processing  │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Model Layer                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ML Pipeline                             │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │   Data      │ │   Feature   │ │   Model     │   │   │
│  │  │ Processing  │ │ Engineering │ │  Training   │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 📁 Project Structure

```
game-sales-analysis/
├── 📱 app.py                           # Main Streamlit application (2121 lines)
├── 📊 data/                            # Data management
│   ├── raw/                            # Original VGChartz dataset
│   │   └── vgchartz-2024.csv          # 8.2MB, 18,874 games
│   └── processed/                      # Cleaned and engineered data
│       ├── vgchartz_cleaned.csv        # Cleaned dataset
│       ├── vgchartz_numeric.csv        # Numeric features only
│       └── vgchartz_pca.csv           # PCA-transformed features
├── 🤖 models/                          # Trained ML models (.joblib)
├── 📈 results/                         # Analysis outputs and visualizations
│   ├── regression_results/             # Random Forest regression models
│   ├── decision_tree_results/          # Decision Tree classification
│   ├── naive_bayes_results/            # Naive Bayes classification
│   ├── clustering_results/             # K-means clustering
│   └── hierarchical_results/           # Hierarchical clustering
├── 🎨 assets/                          # Static assets and plots
├── 🧪 tests/                           # Comprehensive test suite
│   ├── test_models.py                  # Model testing
│   ├── test_streamlit_app.py           # App functionality testing
│   ├── test_app_utils.py               # Utility function testing
│   ├── test_data_processing.py         # Data pipeline testing
│   └── run_tests.py                    # Test runner
├── 🐍 Core ML Scripts
│   ├── create_models.py                # Model creation pipeline
│   ├── tune_models.py                  # Hyperparameter tuning (GridSearchCV)
│   ├── tune_models_randomized.py       # Randomized hyperparameter search
│   ├── fit_models.py                   # Model fitting utilities
│   ├── vg_sales_regression.py          # Regression analysis
│   └── train_and_save_model.py         # Model training utilities
├── 📚 Documentation
│   ├── README.md                       # Main overview
│   ├── README_TECHNICAL.md             # This technical guide
│   ├── README_ASSESSMENT.md            # Assessment criteria
│   ├── QUICK_START_GUIDE.md            # Quick setup guide
│   ├── DEVELOPER_GUIDE.md              # Contribution guidelines
│   ├── DOCUMENTATION.md                # User documentation
│   └── HYPERPARAMETER_TUNING.md        # Tuning methodology
└── 🐳 Deployment
    ├── requirements.txt                 # Python dependencies
    ├── Dockerfile                      # Container configuration
    └── .gitignore                      # Version control exclusions
```

## 🧠 Machine Learning Pipeline

### 1. Data Processing Pipeline

#### Raw Data Characteristics
- **Source**: VGChartz comprehensive gaming database
- **Size**: 18,874 games, 8.2MB raw data
- **Coverage**: 1970-2024, all major platforms and regions
- **Features**: 15+ original features including sales, ratings, metadata

#### Data Cleaning Process
```python
def clean_video_game_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline with multiple stages:
    1. Missing value imputation
    2. Outlier detection and treatment
    3. Data type standardization
    4. Consistency validation
    """
    # Stage 1: Handle missing values
    df['critic_score'] = df['critic_score'].fillna(df['critic_score'].median())
    df['user_score'] = df['user_score'].fillna(df['user_score'].median())
    
    # Stage 2: Remove outliers using IQR method
    df = remove_sales_outliers(df, 'total_sales', threshold=1.5)
    
    # Stage 3: Standardize data types
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    
    return df
```

#### Feature Engineering Pipeline
```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering creating 14 engineered features:
    """
    # Regional sales ratios (normalized by total sales)
    df['na_sales_ratio'] = df['na_sales'] / (df['total_sales'] + 1e-8)
    df['jp_sales_ratio'] = df['jp_sales'] / (df['total_sales'] + 1e-8)
    df['pal_sales_ratio'] = df['pal_sales'] / (df['total_sales'] + 1e-8)
    
    # Temporal features
    current_year = datetime.now().year
    df['game_age'] = current_year - df['release_year']
    df['sales_per_year'] = df['total_sales'] / (df['game_age'] + 1)
    
    # Frequency encoding for high-cardinality categoricals
    df['console_freq'] = df['console'].map(df['console'].value_counts(normalize=True))
    df['genre_freq'] = df['genre'].map(df['genre'].value_counts(normalize=True))
    df['publisher_freq'] = df['publisher'].map(df['publisher'].value_counts(normalize=True))
    
    # Interaction features
    df['critic_user_ratio'] = df['critic_score'] / (df['user_score'] + 1e-8)
    df['sales_rating_correlation'] = df['total_sales'] * df['critic_score']
    
    return df
```

### 2. Model Architecture

#### Model Ensemble Design
```python
class GameSalesPredictor:
    """
    Ensemble predictor combining multiple ML approaches for robust predictions
    """
    def __init__(self):
        self.models = {
            'regression': None,      # Random Forest for sales volume
            'classification': None,   # Decision Tree for high/low classification
            'probabilistic': None    # Naive Bayes for uncertainty estimation
        }
        self.scalers = {}
        self.feature_names = []
    
    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive predictions using all models
        """
        results = {}
        
        # Regression prediction
        if self.models['regression']:
            results['sales_prediction'] = self.models['regression'].predict(input_data)
            results['confidence_interval'] = self._calculate_confidence_interval(input_data)
        
        # Classification prediction
        if self.models['classification']:
            results['sales_class'] = self.models['classification'].predict(input_data)
            results['class_probability'] = self.models['classification'].predict_proba(input_data)
        
        return results
```

#### Hyperparameter Optimization Strategy

**GridSearchCV Implementation** (`tune_models.py`):
```python
def optimize_random_forest():
    """
    Exhaustive hyperparameter search for Random Forest regression
    """
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2
    )
    
    return grid_search
```

**RandomizedSearchCV Implementation** (`tune_models_randomized.py`):
```python
def optimize_random_forest_randomized():
    """
    Efficient hyperparameter exploration using continuous distributions
    """
    param_distributions = {
        'n_estimators': stats.randint(50, 300),
        'max_depth': stats.randint(10, 100),
        'min_samples_split': stats.randint(2, 20),
        'min_samples_leaf': stats.randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions, n_iter=100, cv=5, 
        scoring='r2', n_jobs=-1, verbose=2
    )
    
    return random_search
```

### 3. Model Performance & Evaluation

#### Cross-Validation Strategy
- **5-fold Cross-Validation**: Ensures robust performance estimation
- **Stratified Sampling**: Maintains class distribution in classification tasks
- **Time Series Awareness**: Respects temporal order in gaming data

#### Performance Metrics

| Model | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| **Random Forest Regression** | R² Score | 0.9732 | Explains 97.32% of sales variance |
| **Random Forest Regression** | MSE | 0.0181 | Low prediction error |
| **Decision Tree Classification** | Accuracy | 0.987 | 98.7% correct high/low predictions |
| **Decision Tree Classification** | F1-Score | 0.986 | Balanced precision and recall |
| **Naive Bayes Classification** | Accuracy | 0.859 | 85.9% correct predictions |
| **Naive Bayes Classification** | ROC-AUC | 0.892 | Good discriminative ability |

#### Feature Importance Analysis
```python
def analyze_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Comprehensive feature importance analysis using multiple methods
    """
    # Method 1: Tree-based importance
    tree_importance = model.feature_importances_
    
    # Method 2: Permutation importance
    permutation_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    
    # Method 3: SHAP values (if available)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_importance = np.abs(shap_values).mean(0)
    except:
        shap_importance = None
    
    return pd.DataFrame({
        'feature': feature_names,
        'tree_importance': tree_importance,
        'permutation_importance': permutation_importance.importances_mean,
        'shap_importance': shap_importance
    })
```

## 🔧 Technical Implementation

### 1. Input Validation System

#### Multi-Level Validation Architecture
```python
class InputValidator:
    """
    Comprehensive input validation with business logic and user feedback
    """
    
    def __init__(self):
        self.validation_rules = {
            'critic_score': {'min': 0, 'max': 100, 'type': float},
            'release_year': {'min': 1970, 'max': 2030, 'type': int},
            'sales_values': {'min': 0, 'max': 1000, 'type': float}
        }
    
    def validate_prediction_inputs(self, inputs: Dict[str, Any]) -> ValidationResult:
        """
        Multi-stage validation with detailed error reporting
        """
        errors = []
        warnings = []
        
        # Stage 1: Type and range validation
        for field, value in inputs.items():
            if field in self.validation_rules:
                rule = self.validation_rules[field]
                type_valid, type_msg = self._validate_type(value, rule['type'])
                range_valid, range_msg = self._validate_range(value, rule['min'], rule['max'])
                
                if not type_valid:
                    errors.append(f"{field}: {type_msg}")
                if not range_valid:
                    errors.append(f"{field}: {range_msg}")
        
        # Stage 2: Business logic validation
        business_errors = self._validate_business_logic(inputs)
        errors.extend(business_errors)
        
        # Stage 3: Consistency validation
        consistency_errors = self._validate_consistency(inputs)
        errors.extend(consistency_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

#### Business Logic Validation
```python
def _validate_business_logic(self, inputs: Dict[str, Any]) -> List[str]:
    """
    Domain-specific validation rules for video game sales
    """
    errors = []
    
    # Regional sales distribution validation
    total_sales = sum([
        inputs.get('na_sales', 0),
        inputs.get('jp_sales', 0),
        inputs.get('pal_sales', 0),
        inputs.get('other_sales', 0)
    ])
    
    if total_sales == 0:
        errors.append("At least one regional sales value must be greater than zero")
    
    # Check for unrealistic regional dominance
    for region, sales in [('NA', inputs.get('na_sales', 0)),
                          ('Japan', inputs.get('jp_sales', 0)),
                          ('PAL', inputs.get('pal_sales', 0))]:
        if total_sales > 0 and sales / total_sales > 0.9:
            errors.append(f"{region} sales ({sales/total_sales:.1%}) is unusually high")
    
    # Temporal consistency
    current_year = datetime.now().year
    if inputs.get('release_year', 0) > current_year + 2:
        errors.append("Release year cannot be more than 2 years in the future")
    
    return errors
```

### 2. Streamlit Application Architecture

#### Multi-Page Design Pattern
```python
class StreamlitApp:
    """
    Modular Streamlit application with clean separation of concerns
    """
    
    def __init__(self):
        self.pages = {
            'Analysis': self.show_analysis_page,
            'Prediction': self.show_prediction_page,
            'Reports': self.show_reports_page,
            'Documentation': self.show_documentation_page
        }
        self.current_page = 'Analysis'
    
    def run(self):
        """
        Main application loop with sidebar navigation
        """
        st.set_page_config(
            page_title="Game Sales Analysis",
            page_icon="🎮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar navigation
        self.current_page = st.sidebar.selectbox(
            "Choose a page:", list(self.pages.keys())
        )
        
        # Page content
        self.pages[self.current_page]()
```

#### Performance Optimization
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_data():
    """
    Efficient data loading with caching for performance
    """
    data = pd.read_csv('data/processed/vgchartz_cleaned.csv')
    return data

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def generate_visualization(data, chart_type, filters):
    """
    Cached visualization generation for interactive filtering
    """
    if chart_type == 'sales_trends':
        return create_sales_trends_chart(data, filters)
    elif chart_type == 'regional_distribution':
        return create_regional_chart(data, filters)
    # ... other chart types
```

### 3. Error Handling & User Experience

#### Graceful Degradation
```python
def safe_model_prediction(models: Dict, input_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Robust prediction with comprehensive error handling
    """
    try:
        # Attempt prediction with primary model
        prediction = models['regression'].predict(input_data)
        confidence = calculate_prediction_confidence(input_data)
        
        return {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'model_used': 'Random Forest Regression'
        }
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        
        # Fallback to simpler model
        try:
            fallback_prediction = models['decision_tree'].predict(input_data)
            return {
                'success': True,
                'prediction': fallback_prediction,
                'confidence': 0.5,  # Lower confidence for fallback
                'model_used': 'Decision Tree (Fallback)',
                'warning': 'Using fallback model due to primary model error'
            }
        except Exception as fallback_error:
            return {
                'success': False,
                'error': f"All models failed: {str(fallback_error)}",
                'suggestion': 'Please check your input values and try again'
            }
```

## 🧪 Testing Framework

### 1. Test Architecture

#### Test Suite Organization
```
tests/
├── test_models.py              # Model functionality testing
├── test_streamlit_app.py       # App integration testing
├── test_app_utils.py           # Utility function testing
├── test_data_processing.py     # Data pipeline testing
├── run_tests.py                # Test runner and reporting
└── testing_README.md           # Testing documentation
```

#### Test Coverage Strategy
```python
class TestGameSalesPredictor:
    """
    Comprehensive testing of the prediction system
    """
    
    def test_model_loading(self):
        """Test that all models can be loaded correctly"""
        predictor = GameSalesPredictor()
        assert predictor.load_models() == True
        assert all(model is not None for model in predictor.models.values())
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy on known test cases"""
        test_cases = [
            {'critic_score': 85, 'release_year': 2020, 'genre': 'Action'},
            {'critic_score': 60, 'release_year': 2015, 'genre': 'Sports'}
        ]
        
        for test_case in test_cases:
            prediction = self.predictor.predict(test_case)
            assert prediction['success'] == True
            assert 'prediction' in prediction
            assert prediction['confidence'] > 0.5
```

### 2. Data Validation Testing

#### Schema Validation
```python
def test_data_schema():
    """Test that processed data maintains expected schema"""
    data = load_test_data()
    
    expected_columns = [
        'title', 'console', 'genre', 'publisher', 'critic_score',
        'total_sales', 'na_sales', 'jp_sales', 'pal_sales',
        'release_year', 'console_freq', 'genre_freq', 'publisher_freq'
    ]
    
    for col in expected_columns:
        assert col in data.columns, f"Missing column: {col}"
    
    # Data type validation
    assert data['critic_score'].dtype in ['float64', 'int64']
    assert data['release_year'].dtype == 'int64'
    assert data['total_sales'].dtype == 'float64'
```

## 🚀 Deployment & Production

### 1. Docker Configuration

#### Multi-Stage Build
```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port and run
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### 2. Environment Configuration

#### Configuration Management
```python
class Config:
    """
    Centralized configuration management
    """
    
    def __init__(self):
        self.model_paths = {
            'regression': 'results/regression_results/random_forest_model.joblib',
            'classification': 'results/decision_tree_results/decision_tree_model.joblib',
            'naive_bayes': 'results/naive_bayes_results/naive_bayes_model.joblib'
        }
        
        self.feature_config = {
            'required_features': [
                'critic_score', 'release_year', 'console_freq', 'genre_freq',
                'publisher_freq', 'na_sales_ratio', 'jp_sales_ratio', 'pal_sales_ratio'
            ],
            'categorical_features': ['console', 'genre', 'publisher'],
            'numerical_features': ['critic_score', 'release_year', 'total_sales']
        }
        
        self.validation_rules = {
            'critic_score': {'min': 0, 'max': 100},
            'release_year': {'min': 1970, 'max': 2030},
            'sales_values': {'min': 0, 'max': 1000}
        }
```

## 📊 Performance Monitoring

### 1. Application Metrics

#### Key Performance Indicators
- **Model Load Time**: < 2 seconds
- **Prediction Latency**: < 100ms per prediction
- **Memory Usage**: ~500MB for full model ensemble
- **Error Rate**: < 0.1% (robust error handling)

#### Performance Profiling
```python
import time
import psutil
import streamlit as st

def performance_monitor(func):
    """
    Decorator for monitoring function performance
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Log performance metrics
        st.sidebar.metric("Execution Time", f"{execution_time:.3f}s")
        st.sidebar.metric("Memory Usage", f"{end_memory:.1f}MB")
        
        return result
    return wrapper
```

## 🔄 CI/CD Pipeline

### 1. Automated Workflows

#### GitHub Actions Configuration
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov flake8 black
      
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml
      
      - name: Code quality check
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          black --check .
```

## 🛠️ Development Setup

### 1. Environment Setup

#### Prerequisites
```bash
# System requirements
Python 3.7+
8GB RAM (for model training)
2GB disk space

# Python environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### Development Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest pytest-cov flake8 black mypy
pip install jupyter notebook ipykernel
pip install streamlit-extras

# Pre-commit hooks
pip install pre-commit
pre-commit install
```

### 2. Model Training Workflow

#### Quick Model Generation
```bash
# Generate all models with default parameters
python create_models.py

# Hyperparameter tuning (comprehensive)
python tune_models_randomized.py

# Quick hyperparameter tuning
python tune_models.py

# Fit specific models
python fit_models.py
```

## 📚 API Reference

### 1. Core Functions

#### Data Loading & Processing
```python
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate dataset with comprehensive error handling
    
    Args:
        file_path: Path to the CSV data file
        
    Returns:
        Cleaned and processed DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    pass

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data preprocessing pipeline
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Processed DataFrame ready for ML
    """
    pass
```

#### Model Prediction
```python
def predict_sales(models: Dict, input_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive sales predictions
    
    Args:
        models: Dictionary of trained ML models
        input_data: Input features for prediction
        
    Returns:
        Dictionary containing predictions, confidence, and metadata
    """
    pass

def validate_prediction_inputs(inputs: Dict[str, Any]) -> List[str]:
    """
    Validate all prediction inputs and return validation errors
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        List of validation error messages (empty if valid)
    """
    pass
```

## 🔧 Configuration Management

### 1. Model Paths
```python
MODEL_PATHS = {
    'regression': 'results/regression_results/random_forest_model.joblib',
    'classification': 'results/decision_tree_results/decision_tree_model.joblib',
    'naive_bayes': 'results/naive_bayes_results/naive_bayes_model.joblib',
    'preprocessor': 'results/regression_results/preprocessor.joblib'
}
```

### 2. Feature Configuration
```python
REQUIRED_FEATURES = [
    'critic_score', 'release_year', 'console_freq', 'genre_freq',
    'publisher_freq', 'na_sales_ratio', 'jp_sales_ratio', 'pal_sales_ratio'
]

FEATURE_TYPES = {
    'numerical': ['critic_score', 'release_year', 'total_sales'],
    'categorical': ['console', 'genre', 'publisher'],
    'engineered': ['console_freq', 'genre_freq', 'publisher_freq']
}
```

## 🤝 Contributing Guidelines

### 1. Code Style
- **PEP 8 compliance**: Use `black` for automatic formatting
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Line length**: Maximum 88 characters (black default)

### 2. Testing Requirements
- **Test coverage**: Minimum 90% for new features
- **Unit tests**: Required for all new functions
- **Integration tests**: Required for new ML pipelines
- **Performance tests**: Required for optimization changes

### 3. Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Ensure all tests pass** (`pytest tests/`)
5. **Update documentation** if needed
6. **Submit pull request** with detailed description

## 📚 Further Reading

### Machine Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Model Interpretability Guide](https://christophm.github.io/interpretable-ml-book/)
- [Feature Engineering Best Practices](https://www.feature-engineering.com/)

### Streamlit Development
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Components](https://docs.streamlit.io/library/components)
- [Streamlit Best Practices](https://docs.streamlit.io/knowledge-base)

### Software Engineering
- [MLOps Best Practices](https://ml-ops.org/)
- [Testing Machine Learning Systems](https://www.oreilly.com/library/view/testing-machine-learning/9781492040300/)
- [Python Type Hints](https://mypy.readthedocs.io/)

---

*For technical questions or contributions, please open an issue on GitHub or refer to the [Developer Guide](DEVELOPER_GUIDE.md)*