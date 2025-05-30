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
df = pd.read_csv('vgchartz_cleaned.csv')

# Basic info and statistical summary
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert release_year to numeric if needed
if df['release_year'].dtype == 'object':
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

# Identify variable types
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
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
            return "Ordinal"
        else:
            return "Numeric (Continuous)"

# Identify variable types
print("\nVariable Types:")
for col in df.columns:
    print(f"{col}: {identify_variable_type(col)}")

# Calculate statistics for numeric variables
def calculate_statistics(df, column):
    if column in df.columns:
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
        print(f"\n{column} not found in dataset.")
        return None

# Create the 10 required plots with gaming dataset adaptations
def create_plots(df):
    # 1. Histogram for total_sales
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_sales'], kde=True)
    plt.title('Histogram of Total Game Sales')
    plt.xlabel('Total Sales (millions)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/1_histogram_total_sales.png')
    plt.close()
    print("1. Created histogram for total sales")
    
    # 2. Box Plot for sales across regions
    sales_cols = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[sales_cols])
    plt.title('Box Plot of Sales by Region')
    plt.ylabel('Sales (millions)')
    plt.tight_layout()
    plt.savefig('plots/2_boxplot_regional_sales.png')
    plt.close()
    print("2. Created box plot for regional sales")
    
    # 3. QQ Plot for critic scores
    plt.figure(figsize=(8, 8))
    stats.probplot(df['critic_score'].dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot of Critic Scores')
    plt.tight_layout()
    plt.savefig('plots/3_qqplot_critic_score.png')
    plt.close()
    print("3. Created QQ Plot for critic scores")
    
    # 4. Correlation Heatmap for numeric columns
    numeric_game_cols = ['critic_score', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'release_year']
    plt.figure(figsize=(12, 10))
    corr = df[numeric_game_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Sales and Scores')
    plt.tight_layout()
    plt.savefig('plots/4_correlation_heatmap.png')
    plt.close()
    print("4. Created correlation heatmap")
    
    # 5. Scatter Plot between critic_score and total_sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='critic_score', y='total_sales', data=df)
    plt.title('Scatter Plot: Critic Score vs Total Sales')
    plt.xlabel('Critic Score')
    plt.ylabel('Total Sales (millions)')
    plt.tight_layout()
    plt.savefig('plots/5_scatterplot_score_sales.png')
    plt.close()
    print("5. Created scatter plot between critic score and total sales")
    
    # 6. Bar Chart for game genres
    plt.figure(figsize=(14, 8))
    genre_counts = df['genre'].value_counts().sort_values(ascending=False)
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.title('Number of Games by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/6_barchart_genres.png')
    plt.close()
    print("6. Created bar chart for game genres")
    
    # 7. Pie Chart for console distribution
    plt.figure(figsize=(10, 10))
    df['console'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Distribution of Games by Console')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('plots/7_piechart_consoles.png')
    plt.close()
    print("7. Created pie chart for console distribution")
    
    # 8. Violin Plot for total sales by genre
    plt.figure(figsize=(16, 8))
    # Limit to top 10 genres if there are many
    top_genres = df['genre'].value_counts().nlargest(10).index
    genre_data = df[df['genre'].isin(top_genres)]
    sns.violinplot(x='genre', y='total_sales', data=genre_data)
    plt.title('Total Sales Distribution by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Total Sales (millions)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/8_violinplot_sales_by_genre.png')
    plt.close()
    print("8. Created violin plot for total sales by genre")
    
    # 9. Quantile Plot (Empirical CDF) for total sales
    plt.figure(figsize=(10, 6))
    sorted_data = np.sort(df['total_sales'].dropna())
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y, marker='.', linestyle='none')
    plt.title('Quantile Plot (Empirical CDF) of Total Sales')
    plt.xlabel('Total Sales (millions)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/9_quantile_plot_total_sales.png')
    plt.close()
    print("9. Created quantile plot for total sales")
    
    # 10. Pair Plot for sales across regions
    plt.figure(figsize=(16, 12))
    sns.pairplot(df[['na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']], height=2.5)
    plt.suptitle('Pair Plot of Regional Sales and Critic Score', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/10_pairplot_sales.png')
    plt.close()
    print("10. Created pair plot for regional sales and critic score")

    # 11. Line plot of game releases by year
    plt.figure(figsize=(14, 8))
    year_counts = df.groupby('release_year').size()
    year_counts.plot(kind='line', marker='o')
    plt.title('Number of Games Released by Year')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Games')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/11_lineplot_releases_by_year.png')
    plt.close()
    print("11. Created line plot of game releases by year")
    
    # 12. Heatmap of genre popularity over time
    plt.figure(figsize=(16, 10))
    genre_year = df.groupby(['release_year', 'genre']).size().unstack().fillna(0)
    # Limit to top 10 genres if needed
    if len(genre_year.columns) > 10:
        top_genres = df['genre'].value_counts().nlargest(10).index
        genre_year = genre_year[genre_year.columns.intersection(top_genres)]
    sns.heatmap(genre_year.T, cmap='YlGnBu', linewidths=0.5)
    plt.title('Genre Popularity Over Years')
    plt.xlabel('Release Year')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig('plots/12_heatmap_genre_by_year.png')
    plt.close()
    print("12. Created heatmap of genre popularity over time")

# Calculate statistics for all numeric columns
numeric_columns = ['critic_score', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'release_year']
for col in numeric_columns:
    calculate_statistics(df, col)

# Calculate top publishers by total sales
top_publishers = df.groupby('publisher')['total_sales'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Publishers by Total Sales:")
for publisher, sales in top_publishers.items():
    print(f"{publisher}: {sales:.2f} million")

# Calculate top selling games
top_games = df.sort_values('total_sales', ascending=False).head(10)[['title', 'console', 'total_sales']]
print("\nTop 10 Best-Selling Games:")
for i, (_, row) in enumerate(top_games.iterrows(), 1):
    print(f"{i}. {row['title']} ({row['console']}): {row['total_sales']:.2f} million")

# Generate the plots
create_plots(df)

print("\nAnalysis complete. All plots have been saved to the 'plots' directory.") 