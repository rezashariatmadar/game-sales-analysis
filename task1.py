import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Load the dataset
df = pd.read_csv('student_depression_dataset.csv')

# Basic info and statistical summary
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Identify variable types
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')  # Removing ID as per instructions
    
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)

# Function to identify data type
def identify_variable_type(col):
    if df[col].dtype == 'object':
        unique_values = df[col].nunique()
        if unique_values == 2:
            return "Binary"
        elif unique_values <= 10:
            return "Nominal" if not all(df[col].dropna().astype(str).str.isnumeric()) else "Ordinal"
        else:
            return "Nominal"
    else:  # numeric
        unique_values = df[col].nunique()
        if unique_values == 2:
            return "Binary"
        elif unique_values <= 10:
            return "Ordinal" if col not in ['id'] else "ID"
        else:
            return "Numeric (Continuous)"

# Identify variable types
print("\nVariable Types:")
for col in df.columns:
    if col != 'id':  # Skip id column
        print(f"{col}: {identify_variable_type(col)}")

# Calculate statistics for numeric variables
def calculate_statistics(df, column):
    if column in df.columns and column != 'id':
        data = df[column].dropna()
        
        if pd.api.types.is_numeric_dtype(data):
            # Calculate statistics
            mean = data.mean()
            median = data.median()
            mode = data.mode()[0]
            midrange = (data.max() + data.min()) / 2
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            min_val = data.min()
            max_val = data.max()
            
            print(f"\nStatistics for {column}:")
            print(f"Mean: {mean}")
            print(f"Median: {median}")
            print(f"Mode: {mode}")
            print(f"Midrange: {midrange}")
            print(f"Five Number Summary:")
            print(f"  Minimum: {min_val}")
            print(f"  Q1: {q1}")
            print(f"  Median: {median}")
            print(f"  Q3: {q3}")
            print(f"  Maximum: {max_val}")
            
            return {
                'mean': mean,
                'median': median,
                'mode': mode,
                'midrange': midrange,
                'min': min_val,
                'q1': q1,
                'q3': q3,
                'max': max_val
            }
        else:
            print(f"\n{column} is not numeric, skipping statistics.")
            return None
    else:
        print(f"\n{column} not found in dataset or is ID column.")
        return None

# Create the 10 required plots
def create_plots(df):
    # 1. Histogram for a numeric column
    plt.figure(figsize=(10, 6))
    numeric_col = numeric_cols[0] if numeric_cols else None
    if numeric_col:
        sns.histplot(df[numeric_col], kde=True)
        plt.title(f'Histogram of {numeric_col}')
        plt.xlabel(numeric_col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'plots/1_histogram_{numeric_col}.png')
        plt.close()
        print(f"1. Created histogram for {numeric_col}")
    
    # 2. Box Plot for numeric columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[numeric_cols[:5]])  # Limit to first 5 numeric columns
        plt.title('Box Plot of Numeric Variables')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig('plots/2_boxplot.png')
        plt.close()
        print("2. Created box plot for numeric variables")
    
    # 3. QQ Plot for a numeric column
    if numeric_col:
        plt.figure(figsize=(8, 8))
        stats.probplot(df[numeric_col].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {numeric_col}')
        plt.tight_layout()
        plt.savefig(f'plots/3_qqplot_{numeric_col}.png')
        plt.close()
        print(f"3. Created QQ Plot for {numeric_col}")
    
    # 4. Correlation Heatmap for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Numeric Variables')
        plt.tight_layout()
        plt.savefig('plots/4_correlation_heatmap.png')
        plt.close()
        print("4. Created correlation heatmap")
    
    # 5. Scatter Plot between two numeric columns
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
        plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.tight_layout()
        plt.savefig(f'plots/5_scatterplot.png')
        plt.close()
        print(f"5. Created scatter plot between {numeric_cols[0]} and {numeric_cols[1]}")
    
    # 6. Bar Chart for a categorical column
    cat_col = categorical_cols[0] if categorical_cols else None
    if cat_col:
        plt.figure(figsize=(12, 6))
        counts = df[cat_col].value_counts().sort_values(ascending=False)
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f'Bar Chart of {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/6_barchart_{cat_col}.png')
        plt.close()
        print(f"6. Created bar chart for {cat_col}")
    
    # 7. Pie Chart for a categorical column
    if cat_col:
        plt.figure(figsize=(10, 10))
        df[cat_col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {cat_col}')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(f'plots/7_piechart_{cat_col}.png')
        plt.close()
        print(f"7. Created pie chart for {cat_col}")
    
    # 8. Violin Plot for a numeric column grouped by a categorical column
    if numeric_col and cat_col and df[cat_col].nunique() <= 10:
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=cat_col, y=numeric_col, data=df)
        plt.title(f'Violin Plot of {numeric_col} by {cat_col}')
        plt.xlabel(cat_col)
        plt.ylabel(numeric_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/8_violinplot.png')
        plt.close()
        print(f"8. Created violin plot for {numeric_col} by {cat_col}")
    
    # 9. Quantile Plot (Empirical CDF)
    if numeric_col:
        plt.figure(figsize=(10, 6))
        sorted_data = np.sort(df[numeric_col].dropna())
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, y, marker='.', linestyle='none')
        plt.title(f'Quantile Plot (Empirical CDF) of {numeric_col}')
        plt.xlabel(numeric_col)
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/9_quantile_plot_{numeric_col}.png')
        plt.close()
        print(f"9. Created quantile plot for {numeric_col}")
    
    # 10. Pair Plot for multiple numeric columns
    if len(numeric_cols) >= 3:
        plt.figure(figsize=(16, 12))
        sns.pairplot(df[numeric_cols[:4]], height=2.5)  # Limit to first 4 columns
        plt.suptitle('Pair Plot of Numeric Variables', y=1.02)
        plt.tight_layout()
        plt.savefig('plots/10_pairplot.png')
        plt.close()
        print("10. Created pair plot for numeric variables")

# Calculate statistics for all numeric columns
for col in numeric_cols:
    calculate_statistics(df, col)

# Generate the plots
create_plots(df)

print("\nAnalysis complete. All plots have been saved to the 'plots' directory.") 