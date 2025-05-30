import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Creating model files for the Streamlit app...")

# Create directories if they don't exist
os.makedirs('regression_results', exist_ok=True)
os.makedirs('naive_bayes_results', exist_ok=True)
os.makedirs('decision_tree_results', exist_ok=True)

# Load the data
try:
    print("Loading dataset...")
    df = pd.read_csv('vgchartz_cleaned.csv')
    print(f"Dataset loaded successfully with {len(df)} records")
except FileNotFoundError:
    print("Error: vgchartz_cleaned.csv not found!")
    exit(1)

# Extract features and targets
print("Preparing data for model training...")

# Check available columns
print("Available columns:", df.columns.tolist())

# Define features for training
features = ['critic_score', 'release_year', 'genre_freq', 'publisher_freq']
# Add regional sales if available
for col in ['na_sales', 'jp_sales', 'pal_sales']:
    if col in df.columns:
        features.append(col)

# Replace NaN values
df = df.fillna(0)

# Extract features that exist in the dataframe
X = df[[col for col in features if col in df.columns]]
print(f"Using features: {X.columns.tolist()}")

# Target variable for regression
y_reg = df['total_sales'] if 'total_sales' in df.columns else df['na_sales'] + df['jp_sales'] + df['pal_sales'] + df['other_sales']

# Target variable for classification (high sales vs low sales)
median_sales = y_reg.median()
y_cls = (y_reg > median_sales).astype(int)

print(f"Prepared regression target with median value: {median_sales}")

# Split data
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'regression_results/scaler.joblib')
print("Scaler saved as regression_results/scaler.joblib")

# Train Random Forest Regression model
print("Training Random Forest Regression model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_reg_train)

# Save the regression model
joblib.dump(rf_model, 'regression_results/random_forest_model.joblib')
print("Regression model saved as regression_results/random_forest_model.joblib")

# Train Naive Bayes model
print("Training Naive Bayes model...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_cls_train)

# Save the Naive Bayes model
joblib.dump(nb_model, 'naive_bayes_results/naive_bayes_model.joblib')
print("Naive Bayes model saved as naive_bayes_results/naive_bayes_model.joblib")

# Train Decision Tree model
print("Training Decision Tree model...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_cls_train)

# Save the Decision Tree model
joblib.dump(dt_model, 'decision_tree_results/decision_tree_model.joblib')
print("Decision Tree model saved as decision_tree_results/decision_tree_model.joblib")

print("\nAll models created and saved successfully!")
print("\nYou can now run the Streamlit app with: streamlit run app.py") 