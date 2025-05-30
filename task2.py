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
df = pd.read_csv('student_depression_dataset.csv')
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

# 2. Check for inconsistent values in numeric columns
numeric_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours']

print("\nChecking for outliers and inconsistent values in numeric columns...")
for col in numeric_cols:
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
        plt.savefig(f'cleaning_plots/outliers_{col.replace("/", "_")}.png')
        plt.close()
        
        # Cap outliers to the boundary values (alternative to dropping)
        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        print(f"  Outliers in {col} capped to [{lower_bound:.2f}, {upper_bound:.2f}]")

# 3. Check for logical inconsistencies
print("\nChecking for logical inconsistencies...")

# Work/Study Hours should be between 0 and 24
invalid_hours = df_processed[df_processed['Work/Study Hours'] > 24]
if not invalid_hours.empty:
    print(f"Found {len(invalid_hours)} records with Work/Study Hours > 24")
    df_processed.loc[df_processed['Work/Study Hours'] > 24, 'Work/Study Hours'] = 24
    print("  Capped Work/Study Hours to 24")

# Age should be reasonable for students (e.g., 16-80)
invalid_age = df_processed[(df_processed['Age'] < 16) | (df_processed['Age'] > 80)]
if not invalid_age.empty:
    print(f"Found {len(invalid_age)} records with unusual Age values")
    df_processed['Age'] = df_processed['Age'].clip(16, 80)
    print("  Capped Age to [16, 80]")

# CGPA should be between 0 and 10 (assuming 10-point scale based on data)
if df_processed['CGPA'].max() > 10:
    print(f"Found CGPA values > 10, maximum value: {df_processed['CGPA'].max()}")
    df_processed['CGPA'] = df_processed['CGPA'].clip(0, 10)
    print("  Capped CGPA to [0, 10]")

# 4. Check for zero values that might be missing values
print("\nChecking for zero values that might represent missing data...")
for col in numeric_cols:
    zero_count = (df_processed[col] == 0).sum()
    zero_percentage = (zero_count / len(df_processed)) * 100
    if zero_percentage > 1:  # If more than 1% are zeros
        print(f"{col}: {zero_count} zeros ({zero_percentage:.2f}%)")

# ============= DATA TRANSFORMATION =============
print("\n===== DATA TRANSFORMATION =====")

# 1. Standardize numeric features
print("\nStandardizing numeric features...")
scaler = StandardScaler()
df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
print(f"Standardized {len(numeric_cols)} numeric columns")

# 2. Encoding categorical variables
categorical_cols = ['Gender', 'City', 'Profession', 'Sleep Duration', 
                   'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                   'Financial Stress', 'Family History of Mental Illness']

print("\nEncoding categorical variables...")
for col in categorical_cols:
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
    elif n_unique <= 10:
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

# Combined Stress Index (combining academic and work pressure)
if 'Academic Pressure' in df_processed.columns and 'Work Pressure' in df_processed.columns:
    df_processed['Combined_Stress_Index'] = (df_processed['Academic Pressure'] + df_processed['Work Pressure']) / 2
    print("Created Combined_Stress_Index")

# Well-being Index (combining satisfaction measures)
if 'Study Satisfaction' in df_processed.columns and 'Job Satisfaction' in df_processed.columns:
    df_processed['Well_being_Index'] = (df_processed['Study Satisfaction'] + df_processed['Job Satisfaction']) / 2
    print("Created Well_being_Index")

# 4. Save the processed data
processed_file = 'processed_data/student_depression_processed.csv'
df_processed.to_csv(processed_file, index=False)
print(f"\nProcessed data saved to {processed_file}")
print(f"Final dataset shape: {df_processed.shape}")

# Summary of transformations
print("\nSummary of Transformations Applied:")
print("1. Removed duplicates (if any)")
print("2. Handled outliers by capping")
print("3. Fixed logical inconsistencies")
print("4. Standardized numeric features")
print("5. Encoded categorical variables")
print("6. Created engineered features")

# Print head of processed dataset
print("\nFirst few rows of processed dataset:")
print(df_processed.head(3))
print("\nColumn list of processed dataset:")
print(df_processed.columns.tolist()) 