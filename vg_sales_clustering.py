import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Create directory for clustering results
os.makedirs('clustering_results', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed video game sales dataset...")
df = pd.read_csv('processed_data/vgchartz_processed.csv')
print(f"Dataset shape: {df.shape}")

# Ensure we only have numeric data for clustering
print("\nChecking for non-numeric columns...")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
numeric_columns = numeric_df.columns.tolist()
print(f"Found {len(numeric_columns)} numeric columns out of {len(df.columns)} total columns")

# Also exclude any identifier columns if present
features = [col for col in numeric_columns if 'id' not in col.lower()]
print(f"Selected {len(features)} features for clustering")
print("Features:", features)

# Use a subset of features if there are too many
if len(features) > 15:
    print("\nToo many features, selecting most important ones...")
    # Prioritize key metrics for clustering
    key_features = [col for col in features if any(term in col.lower() for term in 
                   ['sales', 'score', 'year', 'age', 'ratio'])]
    if len(key_features) >= 5:  # Ensure we have a reasonable number of features
        features = key_features
        print(f"Using {len(features)} key features")
        print("Key features:", features)

# Verify there are no NaN values in the selected features
X_df = df[features].copy()
nan_counts = X_df.isna().sum()
if nan_counts.sum() > 0:
    print("\nWarning: Found NaN values in features. Filling with feature means.")
    X_df = X_df.fillna(X_df.mean())

# Check for string values that might have been encoded as objects
for col in X_df.columns:
    if X_df[col].dtype == 'object':
        print(f"Converting column {col} to numeric, errors will be set to NaN")
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        X_df[col] = X_df[col].fillna(X_df[col].mean())

X = X_df.values

# Determine optimal number of clusters using Elbow Method
inertia = []
silhouette_scores = []
max_clusters = 10

print("\nFinding optimal number of clusters using Elbow Method...")
for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
    # Calculate silhouette score (only if k > 1)
    if k > 1:
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, silhouette score is {silhouette_avg:.3f}")

# Plot Elbow Method results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.savefig('clustering_results/kmeans_elbow_method.png')
plt.close()

# Based on the elbow method and silhouette scores, choose the optimal number of clusters
# This is a simple heuristic - in practice, you would analyze the plots and choose accordingly
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started from k=2
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

# Perform K-means clustering with the optimal number of clusters
print(f"\nPerforming K-means clustering with {optimal_k} clusters...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_cluster = df.copy()
df_cluster['Cluster'] = kmeans.fit_predict(X)

# Count instances in each cluster
cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
print("\nCluster distribution:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} instances ({count/len(df_cluster)*100:.2f}%)")

# Analyze clusters
print("\nAnalyzing clusters...")
cluster_analysis = df_cluster.groupby('Cluster')[features].mean()
print("\nCluster centers (mean values for each feature):")
print(cluster_analysis)

# Save cluster analysis to CSV
cluster_analysis.to_csv('clustering_results/kmeans_cluster_analysis.csv')

# Visualize clusters using PCA for dimensionality reduction
print("\nVisualizing clusters using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_cluster['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Video Game Clusters Visualization with PCA (K-means, k={optimal_k})')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.savefig('clustering_results/kmeans_pca_visualization.png')
plt.close()

# Analyze relationship between clusters and total sales
print("\nAnalyzing relationship between clusters and total sales...")
if 'total_sales' in features:
    sales_by_cluster = df_cluster.groupby('Cluster')['total_sales'].mean().sort_values()
    print("\nAverage sales by cluster:")
    print(sales_by_cluster)

    plt.figure(figsize=(10, 6))
    sales_by_cluster.plot(kind='bar')
    plt.title('Average Total Sales by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Sales')
    plt.axhline(y=df_cluster['total_sales'].mean(), color='red', linestyle='--', label='Overall Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('clustering_results/kmeans_sales_by_cluster.png')
    plt.close()

# Analyze relationship between clusters and critic scores if available
if 'critic_score' in features:
    scores_by_cluster = df_cluster.groupby('Cluster')['critic_score'].mean().sort_values()
    print("\nAverage critic scores by cluster:")
    print(scores_by_cluster)

    plt.figure(figsize=(10, 6))
    scores_by_cluster.plot(kind='bar')
    plt.title('Average Critic Score by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Critic Score')
    plt.axhline(y=df_cluster['critic_score'].mean(), color='red', linestyle='--', label='Overall Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('clustering_results/kmeans_scores_by_cluster.png')
    plt.close()

# Analyze key features by cluster
# Select a subset of important features to visualize
if len(features) > 10:
    # Get the most variable features across clusters
    feature_variance = cluster_analysis.var().sort_values(ascending=False)
    key_features = feature_variance.head(10).index.tolist()
else:
    key_features = features

print(f"\nSelected {len(key_features)} key features for detailed analysis")
print("Key features:", key_features)

# Create a heatmap of cluster centers for key features
plt.figure(figsize=(14, 8))
key_cluster_centers = cluster_analysis[key_features]
# Normalize the data for better visualization
scaler = StandardScaler()
key_cluster_centers_scaled = pd.DataFrame(
    scaler.fit_transform(key_cluster_centers),
    index=key_cluster_centers.index,
    columns=key_cluster_centers.columns
)

# Create a heatmap
sns.heatmap(key_cluster_centers_scaled, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Normalized Cluster Centers for Key Features (Video Games)')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('clustering_results/kmeans_key_features_heatmap.png')
plt.close()

# Regional sales comparison by cluster if those columns exist
regional_cols = [col for col in features if any(region in col for region in ['na_', 'jp_', 'pal_', 'other_']) and 'sales' in col]
if regional_cols:
    print("\nAnalyzing regional sales patterns by cluster...")
    regional_means = df_cluster.groupby('Cluster')[regional_cols].mean()
    
    # Create a radar chart for regional sales comparison
    plt.figure(figsize=(12, 10))
    
    # Set plot parameters
    categories = regional_cols
    N = len(categories)
    
    # Create angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = plt.subplot(111, polar=True)
    
    # Add each cluster
    for i in range(optimal_k):
        values = regional_means.iloc[i].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    plt.xticks(angles[:-1], categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Regional Sales Patterns by Cluster', size=15)
    plt.tight_layout()
    plt.savefig('clustering_results/kmeans_regional_sales_radar.png')
    plt.close()

# Create a parallel coordinates plot for cluster visualization
plt.figure(figsize=(15, 8))
# Get a subset of data for parallel coordinates plot (can be too dense with all data)
sample_size = min(1000, len(df_cluster))
sample_indices = np.random.choice(len(df_cluster), sample_size, replace=False)
sample_df = df_cluster.iloc[sample_indices].copy()

# Standardize the data for parallel coordinates plot
features_for_parallel = key_features[:7] if len(key_features) > 7 else key_features  # Limit to 7 features for readability
scaler = StandardScaler()
sample_df_scaled = sample_df.copy()
sample_df_scaled[features_for_parallel] = scaler.fit_transform(sample_df[features_for_parallel])

# Create parallel coordinates plot
pd.plotting.parallel_coordinates(
    sample_df_scaled, 'Cluster', 
    cols=features_for_parallel,
    color=plt.cm.viridis(np.linspace(0, 1, optimal_k))
)
plt.title('Parallel Coordinates Plot of Video Game Clusters')
plt.grid(False)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('clustering_results/kmeans_parallel_coordinates.png')
plt.close()

# Summarize the characteristics of each cluster
print("\nCluster Characteristics Summary:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    # Get the top 5 distinctive features for this cluster (highest absolute z-scores)
    cluster_features = key_cluster_centers_scaled.loc[cluster].abs().sort_values(ascending=False)
    top_features = cluster_features.head(5).index.tolist()
    
    for feature in top_features:
        raw_value = key_cluster_centers.loc[cluster, feature]
        scaled_value = key_cluster_centers_scaled.loc[cluster, feature]
        direction = "high" if scaled_value > 0 else "low"
        print(f"  - {feature}: {direction} ({raw_value:.2f}, z-score: {scaled_value:.2f})")
    
    # Optional: Display additional cluster characteristics based on available metrics
    if regional_cols:
        dominant_region = regional_means.loc[cluster].idxmax()
        print(f"  - Dominant sales region: {dominant_region}")
        
    # Add any other relevant information about the clusters
    cluster_size = cluster_counts[cluster]
    cluster_percentage = cluster_size / len(df_cluster) * 100
    print(f"  - Size: {cluster_size} games ({cluster_percentage:.2f}% of dataset)")

# Sample titles from each cluster
print("\nSample titles from each cluster:")
for cluster in range(optimal_k):
    if 'title' in df.columns:
        sample_titles = df_cluster[df_cluster['Cluster'] == cluster]['title'].sample(min(5, cluster_counts[cluster])).tolist()
        print(f"\nCluster {cluster} sample titles:")
        for title in sample_titles:
            print(f"  - {title}")

print("\nK-means clustering analysis complete. Results saved to 'clustering_results' directory.") 