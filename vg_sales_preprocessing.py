import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Create directory for processed data and visualizations
os.makedirs('processed_data', exist_ok=True)
os.makedirs('cleaning_plots', exist_ok=True)

print("Loading the dataset...")
# Load the dataset and create a copy for preprocessing
df = pd.read_csv('vgchartz_cleaned.csv')
df_processed = df.copy()

print(f"Original dataset shape: {df.shape}")

# ============= DATA CLEANING =============
print("\n===== DATA CLEANING =====")

# 1. Check for duplicate records
duplicates = df_processed.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")
if duplicates > 0:
    df_processed = df_processed.drop_duplicates()
    print(f"After removing duplicates: {df_processed.shape}")

# 2. Check for missing values
print("\nMissing Values:")
missing_values = df_processed.isnull().sum()
print(missing_values)

# Only keep rows where essential columns are not null
essential_columns = ['title', 'console', 'genre', 'publisher', 'critic_score', 'total_sales']
df_processed = df_processed.dropna(subset=essential_columns)
print(f"After removing rows with missing essential data: {df_processed.shape}")

# Convert release_year to numeric if needed
if df_processed['release_year'].dtype == 'object':
    df_processed['release_year'] = pd.to_numeric(df_processed['release_year'], errors='coerce')

# 3. Check for inconsistent values in numeric columns
numeric_cols = ['critic_score', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 
                'other_sales', 'release_year']

print("\nChecking for outliers and inconsistent values in numeric columns...")
for col in numeric_cols:
    if col in df_processed.columns:
        # Skip columns with too many NaN values
        if df_processed[col].isna().sum() > len(df_processed) * 0.5:
            print(f"Skipping {col} due to high missing value rate")
            continue
            
        # Print range and check for unusual values
        min_val = df_processed[col].min()
        max_val = df_processed[col].max()
        print(f"{col}: Range [{min_val}, {max_val}]")
        
        # Identify potential outliers using IQR method
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"  Found {len(outliers)} potential outliers in {col}")
            
            # Visualize outliers
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df_processed[col])
            plt.title(f'Boxplot of {col} - Outlier Detection')
            plt.tight_layout()
            plt.savefig(f'cleaning_plots/outliers_{col}.png')
            plt.close()
            
            # Cap outliers to the boundary values (alternative to dropping)
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            print(f"  Outliers in {col} capped to [{lower_bound:.2f}, {upper_bound:.2f}]")

# 4. Check for logical inconsistencies
print("\nChecking for logical inconsistencies...")

# Critic score should be between 0 and 10
if df_processed['critic_score'].max() > 10 or df_processed['critic_score'].min() < 0:
    print(f"Found critic_score values outside the expected range [0, 10]")
    df_processed['critic_score'] = df_processed['critic_score'].clip(0, 10)
    print("  Capped critic_score to [0, 10]")

# Sales figures should be non-negative
for col in ['total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']:
    if col in df_processed.columns and df_processed[col].min() < 0:
        print(f"Found negative values in {col}")
        df_processed[col] = df_processed[col].clip(0, None)
        print(f"  Capped {col} to non-negative values")

# Release year should be reasonable (e.g., 1970-2023)
if 'release_year' in df_processed.columns:
    invalid_years = df_processed[(df_processed['release_year'] < 1970) | (df_processed['release_year'] > 2023)]
    if not invalid_years.empty:
        print(f"Found {len(invalid_years)} records with unusual release_year values")
        df_processed['release_year'] = df_processed['release_year'].clip(1970, 2023)
        print("  Capped release_year to [1970, 2023]")

# 5. Fill remaining missing numeric values
print("\nFilling remaining missing values...")
for col in numeric_cols:
    if col in df_processed.columns and df_processed[col].isna().any():
        missing_count = df_processed[col].isna().sum()
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        print(f"  Filled {missing_count} missing values in {col} with median")

# ============= DATA TRANSFORMATION =============
print("\n===== DATA TRANSFORMATION =====")

# 1. Standardize numeric features
print("\nStandardizing numeric features...")
numeric_cols_to_scale = [col for col in numeric_cols if col in df_processed.columns]
scaler = StandardScaler()
df_processed[numeric_cols_to_scale] = scaler.fit_transform(df_processed[numeric_cols_to_scale])
print(f"Standardized {len(numeric_cols_to_scale)} numeric columns")

# 2. Encoding categorical variables
categorical_cols = ['console', 'genre', 'publisher', 'developer']
print("\nEncoding categorical variables...")

for col in categorical_cols:
    if col in df_processed.columns:
        print(f"Processing {col}...")
        # Check number of unique values
        n_unique = df_processed[col].nunique()
        print(f"  {n_unique} unique values")
        
        # For binary variables, use simple label encoding
        if n_unique == 2:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            print(f"  Applied label encoding to {col}")
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"  Mapping: {mapping}")
        
        # For variables with limited categories, use one-hot encoding
        elif n_unique <= 15:
            # Get dummies and drop the first to avoid multicollinearity
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
            print(f"  Applied one-hot encoding to {col}, created {len(dummies.columns)} new features")
        
        # For high-cardinality variables, consider frequency encoding
        else:
            # Create a frequency map
            freq_map = df_processed[col].value_counts(normalize=True).to_dict()
            df_processed[f'{col}_freq'] = df_processed[col].map(freq_map)
            df_processed.drop(col, axis=1, inplace=True)
            print(f"  Applied frequency encoding to {col}")

# 3. Feature Engineering
print("\nCreating engineered features...")

# Calculate sales ratios between regions
if all(col in df_processed.columns for col in ['na_sales', 'total_sales']):
    df_processed['na_sales_ratio'] = df_processed['na_sales'] / df_processed['total_sales']
    print("Created na_sales_ratio (NA sales / total sales)")

if all(col in df_processed.columns for col in ['jp_sales', 'total_sales']):
    df_processed['jp_sales_ratio'] = df_processed['jp_sales'] / df_processed['total_sales']
    print("Created jp_sales_ratio (Japan sales / total sales)")

if all(col in df_processed.columns for col in ['pal_sales', 'total_sales']):
    df_processed['pal_sales_ratio'] = df_processed['pal_sales'] / df_processed['total_sales']
    print("Created pal_sales_ratio (PAL sales / total sales)")

# Create age of game feature
if 'release_year' in df_processed.columns:
    current_year = 2023
    df_processed['game_age'] = current_year - df_processed['release_year']
    print("Created game_age feature (years since release)")

# Create sales per year feature
if all(col in df_processed.columns for col in ['total_sales', 'game_age']):
    df_processed['sales_per_year'] = df_processed['total_sales'] / (df_processed['game_age'] + 1)  # +1 to avoid division by zero
    print("Created sales_per_year feature")

# Create critic score categorical variable
if 'critic_score' in df_processed.columns:
    bins = [0, 5, 7, 8, 9, 10]
    labels = ['Poor', 'Average', 'Good', 'Great', 'Excellent']
    df_processed['critic_score_category'] = pd.cut(df_processed['critic_score'], bins=bins, labels=labels)
    dummies = pd.get_dummies(df_processed['critic_score_category'], prefix='rating')
    df_processed = pd.concat([df_processed, dummies], axis=1)
    df_processed.drop('critic_score_category', axis=1, inplace=True)
    print("Created critic_score category and dummy variables")

# 4. Remove any remaining NaN values
df_processed = df_processed.dropna()
print(f"\nAfter removing remaining NaN values: {df_processed.shape}")

# 5. Save the processed data
processed_file = 'processed_data/vgchartz_processed.csv'
df_processed.to_csv(processed_file, index=False)
print(f"\nProcessed data saved to {processed_file}")
print(f"Final dataset shape: {df_processed.shape}")

# Summary of transformations
print("\nSummary of Transformations Applied:")
print("1. Removed duplicates")
print("2. Handled missing values in essential columns")
print("3. Capped outliers in numeric columns")
print("4. Fixed logical inconsistencies")
print("5. Standardized numeric features")
print("6. Encoded categorical variables")
print("7. Created engineered features")

# Print head of processed dataset
print("\nFirst few rows of processed dataset:")
print(df_processed.head(3))
print("\nColumn list of processed dataset:")
print(df_processed.columns.tolist()) 