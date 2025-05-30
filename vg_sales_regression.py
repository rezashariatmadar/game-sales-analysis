import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os

# Create a directory for regression results
os.makedirs('regression_results', exist_ok=True)

# --- 1. Load Data ---
print("Loading preprocessed video game sales dataset...")
try:
    df = pd.read_csv('processed_data/vgchartz_processed.csv')
except FileNotFoundError:
    print("Error: 'processed_data/vgchartz_processed.csv' not found.")
    print("Please ensure you have run the data preprocessing script first.")
    exit()

print(f"Dataset shape: {df.shape}")


# --- 2. Define Target and Features ---

# THE GOAL: Predict the actual 'total_sales' value. This is our target 'y'.
# IMPORTANT: 'total_sales' in the processed data was scaled. We use it as is.
y = df['total_sales'].values

# Select features for regression.
# We must EXCLUDE the target variable ('total_sales') and any columns
# that would leak information about it (like regional sales, which add up to total sales).
# This logic is similar to the classification task to prevent the model from "cheating".
exclude_cols = ['title', 'high_sales', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'sales_per_year']

# Drop non-numeric columns and any other columns that were created during preprocessing but are not needed
features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]

X = df[features].values

print(f"\nTarget variable: 'total_sales'")
print(f"Selected {len(features)} features for regression.")
print("Features:", features)


# --- 3. Split and Scale the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 4. Train the Regression Model ---
print("\nTraining a Random Forest Regressor...")
# We use RandomForestRegressor for predicting a continuous value
rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(X_train_scaled, y_train)


# --- 5. Evaluate the Model ---
print("\nEvaluating the model on the test set...")
y_pred = rfr.predict(X_test_scaled)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print("\nInterpretation:")
print(f"- The model explains approximately {r2:.1%} of the variance in the 'total_sales' data.")
print(f"- On average, the model's prediction for sales is off by about {mae:.4f} (in terms of the scaled value).")


# --- 6. Analyze Feature Importance ---
print("\nAnalyzing feature importance...")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rfr.feature_importances_
}).sort_values('Importance', ascending=False)

# Save feature importance to CSV
feature_importance.to_csv('regression_results/feature_importance.csv', index=False)

# Plot feature importance (top 15)
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 15 Features for Predicting Video Game Sales')
plt.tight_layout()
plt.savefig('regression_results/feature_importance.png')
plt.close()

print("\nTop 10 most important features:")
print(top_features.head(10))


# --- 7. Visualize Predictions vs. Actuals ---
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Sales (Scaled)')
plt.ylabel('Predicted Sales (Scaled)')
plt.title('Actual vs. Predicted Sales')
plt.grid(True)
plt.savefig('regression_results/actual_vs_predicted.png')
plt.close()

print("\nRegression analysis complete. Results have been saved to the 'regression_results' directory.")