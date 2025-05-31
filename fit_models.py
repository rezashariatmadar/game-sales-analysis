import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

print("Loading data and models...")

# Load the dataset
try:
    df = pd.read_csv('vgchartz_cleaned.csv')
    print(f"Dataset loaded successfully with {len(df)} records")
except FileNotFoundError:
    print("Error: vgchartz_cleaned.csv not found!")
    exit(1)

# Load the preprocessor
try:
    preprocessor = joblib.load('regression_results/preprocessor.joblib')
    print("Preprocessor loaded successfully")
except FileNotFoundError:
    print("Error: Preprocessor not found!")
    exit(1)

# Load the models
try:
    rf_model = joblib.load('regression_results/random_forest_model.joblib')
    dt_model = joblib.load('decision_tree_results/decision_tree_model.joblib')
    nb_model = joblib.load('naive_bayes_results/naive_bayes_model.joblib')
    print("Models loaded successfully")
except FileNotFoundError:
    print("Error: One or more models not found!")
    exit(1)

# Define features
numeric_features = ['critic_score', 'release_year']
categorical_features = ['console', 'genre', 'publisher']
regional_sales = ['na_sales', 'jp_sales', 'pal_sales']

# Filter to ensure we only use columns that exist
numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]
regional_sales = [col for col in regional_sales if col in df.columns]

# Combine all features
all_features = numeric_features + categorical_features + regional_sales

# Replace NaN values
df = df.fillna(0)

# Extract features
X = df[all_features]
print(f"Using features: {X.columns.tolist()}")

# Target variable for regression
y_reg = df['total_sales'] if 'total_sales' in df.columns else df['na_sales'] + df['jp_sales'] + df['pal_sales'] + df['other_sales']

# Target variable for classification (high sales vs low sales)
median_sales = y_reg.median()
y_cls = (y_reg > median_sales).astype(int)

# Transform the data
print("Transforming data using preprocessor...")
X_transformed = preprocessor.transform(X)

# Train the models
print("Fitting models with transformed data...")

# Train Random Forest Regression model
print("Fitting Random Forest model...")
rf_model.fit(X_transformed, y_reg)
joblib.dump(rf_model, 'regression_results/random_forest_model.joblib')

# Train Decision Tree Classification model
print("Fitting Decision Tree model...")
dt_model.fit(X_transformed, y_cls)
joblib.dump(dt_model, 'decision_tree_results/decision_tree_model.joblib')

# Train Naive Bayes Classification model
print("Fitting Naive Bayes model...")
nb_model.fit(X_transformed, y_cls)
joblib.dump(nb_model, 'naive_bayes_results/naive_bayes_model.joblib')

print("All models have been fitted and saved successfully!")
print("Now you can run the app again with properly trained models.") 