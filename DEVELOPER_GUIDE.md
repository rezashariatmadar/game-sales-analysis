# Developer Guide: Video Game Sales Analysis Project

This guide provides technical details for developers who want to understand, modify, or contribute to the Video Game Sales Analysis project.

## Technical Architecture

The project follows a modular architecture with these main components:

1. **Data Processing Pipeline**: Scripts for cleaning and transforming raw data
2. **Model Training**: Scripts for creating and evaluating machine learning models
3. **Streamlit Application**: Interactive web interface for end users
4. **Visualization Engine**: Components for generating interactive plots

## Code Structure

### Key Files and Their Functions

#### `app.py`
The main Streamlit application that serves as the entry point for users.

- **Page Configuration**: Sets up the Streamlit page layout and styling
- **Data Loading**: Cached functions for efficient data loading
- **Model Loading**: Functions to load trained ML models
- **UI Components**: Functions that generate different UI sections
- **Visualization Functions**: Creates interactive Plotly charts
- **Prediction Logic**: Handles user inputs and generates predictions

#### `create_models.py`
Script for training and saving machine learning models.

- Loads and preprocesses the dataset
- Defines features and target variables
- Splits data into training and testing sets
- Trains Random Forest, Naive Bayes, and Decision Tree models
- Saves trained models and scalers using joblib

## Data Processing

### Data Cleaning Pipeline

The data cleaning process includes:

1. **Missing Value Handling**:
```python
df = df.fillna({
    'critic_score': df['critic_score'].median(),
    'release_year': df['release_year'].median(),
    'genre': 'Unknown',
    'publisher': 'Unknown'
})
```

2. **Feature Engineering**:
```python
# Create frequency features for categorical variables
df['genre_freq'] = df.groupby('genre')['genre'].transform('count') / len(df)
df['publisher_freq'] = df.groupby('publisher')['publisher'].transform('count') / len(df)
```

3. **Date Conversion**:
```python
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_year'].astype(int)
```

### Feature Selection

The models use the following features:
- `critic_score`: Normalized critic ratings
- `release_year`: Year of game release
- `genre_freq`: Frequency of the genre in the dataset
- `publisher_freq`: Frequency of the publisher in the dataset
- Regional sales variables (`na_sales`, `jp_sales`, `pal_sales`)

## Machine Learning Implementation

### Regression Model

The Random Forest regression model predicts total sales:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data
X = df[features]
y = df['total_sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(rf_model, 'regression_results/random_forest_model.joblib')
```

### Classification Models

Two classification models are used to predict whether a game will have high or low sales:

1. **Naive Bayes**:
```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_cls_train)
```

2. **Decision Tree**:
```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_cls_train)
```

## Streamlit Application Design

### Application Structure

The Streamlit app is organized into tabs:

1. **Analysis Tab**: Data exploration and visualization
2. **Prediction Tab**: ML-based sales prediction
3. **Reports Tab**: Detailed analysis reports

### Caching Strategy

Performance optimization through caching:

```python
@st.cache_data
def load_data(path):
    # Data loading logic
    return df

@st.cache_resource
def load_models():
    # Model loading logic
    return models
```

### Input Validation

The app includes robust input validation:

```python
def validate_prediction_inputs(inputs):
    errors = []
    
    # Validate critic score
    valid, msg = validate_input_range(inputs['critic_score'], 0, 100, "Critic score")
    if not valid:
        errors.append(msg)
    
    # Validate release year
    valid, msg = validate_year(inputs['release_year'])
    if not valid:
        errors.append(msg)
    
    # More validations...
    
    return errors
```

## Visualization Techniques

### Interactive Charts with Plotly

The app uses Plotly for interactive visualizations:

```python
def create_sales_trend_chart(df, selected_genres=None, selected_platforms=None):
    # Filter data based on selections
    filtered_df = filter_dataframe(df, selected_genres, selected_platforms)
    
    # Group by year and calculate mean sales
    yearly_sales = filtered_df.groupby('release_year')['total_sales'].mean().reset_index()
    
    # Create Plotly figure
    fig = px.line(
        yearly_sales, 
        x='release_year', 
        y='total_sales',
        title='Average Game Sales by Year',
        labels={'release_year': 'Release Year', 'total_sales': 'Average Sales (millions)'}
    )
    
    return fig
```

### Custom Styling

The app uses custom CSS for consistent styling:

```python
st.markdown("""
    <style>
    .metric-card {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid rgba(28, 131, 225, 0.8);
    }
    </style>
""", unsafe_allow_html=True)
```

## Error Handling

The application implements comprehensive error handling:

```python
def handle_error(error_type, error_message, error_details=None):
    # Display error messages to the user
    st.error(f"**{error_type}**: {error_message}")
    
    # Display error details if provided
    if error_details:
        with st.expander("Technical Details"):
            st.code(str(error_details))
    
    # Provide guidance based on error type
    if "Data Loading" in error_type:
        st.info("Troubleshooting tips...")
```

## Testing

### Unit Tests

The project includes unit tests for core functionality:

- `test_data_processing.py`: Tests for data loading and preprocessing
- `test_models.py`: Tests for model loading and prediction
- `test_app_utils.py`: Tests for app utility functions

### Test Runner

A test runner script executes all tests:

```python
# run_tests.py
import unittest
import sys

# Import test modules
from tests.test_data_processing import TestDataProcessing
from tests.test_models import TestModels
from tests.test_app_utils import TestAppUtils

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataProcessing))
    test_suite.addTest(unittest.makeSuite(TestModels))
    test_suite.addTest(unittest.makeSuite(TestAppUtils))
    
    # Run tests
    result = unittest.TextTestRunner().run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())
```

## Performance Optimization

### Vectorized Operations

The code uses pandas vectorized operations for better performance:

```python
# Inefficient:
for i in range(len(df)):
    df.loc[i, 'total_sales'] = df.loc[i, 'na_sales'] + df.loc[i, 'jp_sales'] + df.loc[i, 'pal_sales'] + df.loc[i, 'other_sales']

# Efficient:
df['total_sales'] = df['na_sales'] + df['jp_sales'] + df['pal_sales'] + df['other_sales']
```

### Data Filtering

Efficient data filtering using pandas:

```python
def filter_dataframe(df, genres=None, platforms=None, publishers=None, year_range=None):
    filtered_df = df.copy()
    
    if genres and 'All' not in genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(genres)]
        
    if platforms and 'All' not in platforms:
        filtered_df = filtered_df[filtered_df['console'].isin(platforms)]
        
    if publishers and 'All' not in publishers:
        filtered_df = filtered_df[filtered_df['publisher'].isin(publishers)]
        
    if year_range:
        filtered_df = filtered_df[(filtered_df['release_year'] >= year_range[0]) & 
                                 (filtered_df['release_year'] <= year_range[1])]
    
    return filtered_df
```

## Contribution Guidelines

### Setting Up Development Environment

1. Clone the repository:
   ```
   git clone https://github.com/rezashariatmadar/game-sales-analysis.git
   cd game-sales-analysis
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Code Style

This project follows PEP 8 style guidelines. Key points:

- Use 4 spaces for indentation
- Maximum line length of 88 characters
- Use descriptive variable names
- Add docstrings to functions and classes

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Version Control Conventions

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Write descriptive commit messages
- Reference issue numbers in commit messages when applicable

## Deployment

### Docker Deployment

The project includes a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py"]
```

Build and run the Docker container:

```bash
docker build -t game-sales-analysis .
docker run -p 8501:8501 game-sales-analysis
```

### Cloud Deployment

The app can be deployed to various cloud platforms:

1. **Streamlit Cloud**:
   - Connect your GitHub repository
   - Select the repository and branch
   - Specify the entry point (`app.py`)

2. **Heroku**:
   - Create a `Procfile`:
     ```
     web: python -m streamlit run app.py --server.port $PORT
     ```
   - Deploy using the Heroku CLI:
     ```
     heroku create
     git push heroku main
     ```

## Future Development

### Planned Features

1. **Advanced Analytics**:
   - Time series forecasting for future sales trends
   - Market segmentation analysis
   - Competitor analysis

2. **Enhanced UI**:
   - User authentication system
   - Customizable dashboards
   - Report export functionality

3. **Model Improvements**:
   - Neural network implementation
   - Hyperparameter optimization
   - Ensemble methods

### Performance Enhancements

1. **Data Processing**:
   - Implement data caching with Redis
   - Use Dask for larger-than-memory datasets
   - Optimize database queries

2. **Application**:
   - Implement lazy loading for UI components
   - Add background processing for heavy computations
   - Optimize image and asset loading 
