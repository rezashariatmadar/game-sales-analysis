# Technical Documentation - Game Sales Analysis

*Comprehensive technical guide for developers and data scientists*

## 🏗️ Architecture Overview

This project implements a complete machine learning pipeline for video game sales analysis using Python, scikit-learn, and Streamlit. The architecture follows a modular design with clear separation of concerns.

### Core Components

```
game-sales-analysis/
├── data/                    # Data management
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and engineered data
├── models/                  # Trained ML models (.joblib)
├── results/                 # Analysis outputs and visualizations
├── assets/                  # Static assets and plots
├── tests/                   # Test suite
└── *.py                     # Core application scripts
```

## 🧠 Machine Learning Pipeline

### 1. Data Processing (`data/processed/`)
- **Raw data**: VGChartz dataset (18,874 games)
- **Cleaning**: Missing value imputation, outlier handling
- **Feature engineering**: Regional sales ratios, game age, temporal features
- **Standardization**: StandardScaler for numeric features

### 2. Model Training

#### Random Forest Regression
```python
RandomForestRegressor(
    n_estimators=196,
    max_depth=45,
    min_samples_split=4,
    min_samples_leaf=4,
    bootstrap=True
)
```
- **Performance**: R² = 0.9732, MSE = 0.0181
- **Purpose**: Precise sales volume prediction

#### Decision Tree Classification
```python
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=22,
    min_samples_split=16,
    min_samples_leaf=8,
    splitter='best'
)
```
- **Performance**: 98.7% accuracy
- **Purpose**: High/low sales categorization

#### Naive Bayes Classification
```python
GaussianNB(var_smoothing=2.09e-06)
```
- **Performance**: 85.85% accuracy
- **Purpose**: Probabilistic classification

### 3. Hyperparameter Optimization
- **GridSearchCV**: Exhaustive parameter search
- **RandomizedSearchCV**: Efficient exploration of large parameter spaces
- **Cross-validation**: 5-fold CV for robust evaluation

## 🔧 Technical Implementation

### Feature Engineering Pipeline
```python
# Regional sales ratios
df['na_sales_ratio'] = df['na_sales'] / df['total_sales']
df['jp_sales_ratio'] = df['jp_sales'] / df['total_sales']
df['pal_sales_ratio'] = df['pal_sales'] / df['total_sales']

# Temporal features
df['game_age'] = current_year - df['release_year']
df['sales_per_year'] = df['total_sales'] / (df['game_age'] + 1)
```

### Model Loading and Prediction
```python
models = {
    'regression': joblib.load('results/regression_results/random_forest_model.joblib'),
    'naive_bayes': joblib.load('results/naive_bayes_results/naive_bayes_model.joblib'),
    'decision_tree': joblib.load('results/decision_tree_results/decision_tree_model.joblib')
}
```

### Input Validation System
- **Range validation**: Critic scores (0-10), years (1970-2024)
- **Distribution validation**: Regional sales ratio checks
- **Data type validation**: Numeric input enforcement
- **Error handling**: Graceful degradation for invalid inputs

## 📊 Data Schema

### Input Features (14 total)
| Feature | Type | Range | Description |
|---------|------|--------|-------------|
| `critic_score` | float | 0-10 | Professional review scores |
| `release_year` | int | 1970-2024 | Game release year |
| `console_freq` | float | 0-1 | Platform frequency encoding |
| `genre_freq` | float | 0-1 | Genre frequency encoding |
| `publisher_freq` | float | 0-1 | Publisher frequency encoding |
| `na_sales_ratio` | float | 0-1 | North America sales proportion |
| `jp_sales_ratio` | float | 0-1 | Japan sales proportion |
| `pal_sales_ratio` | float | 0-1 | Europe/Australia sales proportion |
| `game_age` | int | 0+ | Years since release |

### Target Variables
- **Regression**: `total_sales` (continuous, millions of units)
- **Classification**: `sales_class` (binary: high/low based on median)

## 🚀 Deployment Architecture

### Streamlit Application (`app.py`)
- **Multi-page interface**: Analysis, Prediction, Documentation
- **Real-time computation**: Dynamic filtering and visualization
- **Caching**: `@st.cache_data` for performance optimization
- **Error handling**: Comprehensive input validation

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🧪 Testing Framework

### Test Coverage (`tests/`)
```python
# Data integrity tests
test_data_processing.py      # Data validation and preprocessing
test_models.py              # Model loading and prediction
test_app_utils.py          # Utility functions
test_streamlit_app.py      # End-to-end app testing
```

### Running Tests
```bash
# Full test suite
python tests/run_tests.py

# Individual modules
pytest -v tests/test_models.py
```

## 📈 Performance Metrics

### Model Evaluation
- **Cross-validation**: 5-fold CV with stratification
- **Metrics**: R², MSE, Accuracy, Precision, Recall, F1-score
- **Feature importance**: Permutation-based and tree-based methods

### Application Performance
- **Load time**: < 2 seconds for model loading
- **Prediction latency**: < 100ms per prediction
- **Memory usage**: ~500MB for full model ensemble

## 🔄 CI/CD Pipeline

### Automated Workflows
1. **Code quality**: flake8, black formatting
2. **Testing**: pytest with coverage reporting
3. **Security**: dependency vulnerability scanning
4. **Documentation**: Auto-generation from docstrings

## 🛠️ Development Setup

### Prerequisites
```bash
Python 3.7+
pip install -r requirements.txt
```

### Environment Configuration
```bash
# Development environment
export PYTHONPATH="${PYTHONPATH}:."
export STREAMLIT_THEME="dark"

# Production environment
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Model Retraining
```bash
# Retrain all models with hyperparameter tuning
python tune_models_randomized.py

# Quick model training with default parameters
python create_models.py
```

## 📝 API Reference

### Core Functions

#### Data Loading
```python
def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate dataset."""
```

#### Model Prediction
```python
def predict_sales(models: dict, input_data: pd.DataFrame) -> dict:
    """Generate predictions from all models."""
```

#### Input Validation
```python
def validate_prediction_inputs(inputs: dict) -> List[str]:
    """Validate all prediction inputs and return errors."""
```

## 🔧 Configuration Management

### Model Paths
```python
MODEL_PATHS = {
    'regression': 'results/regression_results/random_forest_model.joblib',
    'classification': 'results/decision_tree_results/decision_tree_model.joblib',
    'preprocessor': 'results/regression_results/preprocessor.joblib'
}
```

### Feature Configuration
```python
REQUIRED_FEATURES = [
    'critic_score', 'release_year', 'console_freq', 'genre_freq',
    'publisher_freq', 'na_sales_ratio', 'jp_sales_ratio', 'pal_sales_ratio'
]
```

## 🤝 Contributing Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Document functions with docstrings
- Maximum line length: 88 characters

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request with detailed description

### Issue Reporting
- Use provided issue templates
- Include minimal reproducible examples
- Specify environment details

## 📚 Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Model Interpretability Guide](https://christophm.github.io/interpretable-ml-book/)
- [MLOps Best Practices](https://ml-ops.org/)

---

*For questions or contributions, please open an issue on GitHub*