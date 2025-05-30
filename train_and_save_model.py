import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib # Used for saving and loading sklearn models
import numpy as np
print("--- Training and Saving Model ---")

# --- 1. Load and Preprocess Data (Same as before) ---
df = pd.read_csv('vgchartz_cleaned.csv')

# Fill NaNs
numeric_cols_to_fill = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'release_year']
for col in numeric_cols_to_fill:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

# Feature Engineering
current_year = pd.to_datetime('today').year
df['game_age'] = current_year - df['release_year']
df['na_sales_ratio'] = (df['na_sales'] / df['total_sales']).replace([np.inf, -np.inf], 0).fillna(0)
df['jp_sales_ratio'] = (df['jp_sales'] / df['total_sales']).replace([np.inf, -np.inf], 0).fillna(0)
df['pal_sales_ratio'] = (df['pal_sales'] / df['total_sales']).replace([np.inf, -np.inf], 0).fillna(0)

# Frequency Encoding
categorical_cols = ['console', 'genre', 'publisher', 'developer']
for col in categorical_cols:
    freq_map = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq_map)

df_processed = df.drop(columns=['title', 'release_date', 'last_update'] + categorical_cols)
df_processed.dropna(inplace=True)

# --- 2. Define Target and Features ---
y = df_processed['total_sales'].values
exclude_for_features = ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
features_list = [col for col in df_processed.columns if col not in exclude_for_features and df_processed[col].dtype in [np.int64, np.float64]]
X = df_processed[features_list].values

# --- 3. Train Model on FULL Dataset ---
# We train on the full dataset now to make the final model as accurate as possible
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(X_scaled, y)
print("Model trained on the full dataset.")

# --- 4. Save the Model and Scaler ---
joblib.dump(rfr, 'regression_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(features_list, 'features.joblib')

print("\nModel, scaler, and feature list have been saved successfully!")
print("Files created: regression_model.joblib, scaler.joblib, features.joblib")