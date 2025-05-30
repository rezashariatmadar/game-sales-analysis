import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

# Create directory for hierarchical clustering results
os.makedirs('hierarchical_results', exist_ok=True)

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

# Standardize the data for hierarchical clustering
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# Sample data if too large (hierarchical clustering can be memory intensive)
max_samples = 1000  # Adjust based on your system's capabilities
if len(X_scaled) > max_samples:
    print(f"\nDataset too large for hierarchical clustering. Sampling {max_samples} records...")
    sample_indices = np.random.choice(len(X_scaled), max_samples, replace=False)
    X_scaled_sample = X_scaled[sample_indices]
    df_sample = df.iloc[sample_indices].copy()
    print(f"Working with sampled dataset of shape: {X_scaled_sample.shape}")
else:
    X_scaled_sample = X_scaled
    df_sample = df.copy()

# Compute the linkage matrix using Ward's method
print("Computing linkage matrix with Ward's method...")
Z = linkage(X_scaled_sample, method='ward')

# Plot the dendrogram to visualize hierarchical structure
plt.figure(figsize=(12, 8))
plt.title('Hierarchical Clustering Dendrogram for Video Game Sales')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=8.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.savefig('hierarchical_results/hierarchical_dendrogram.png')
plt.close()

# Evaluate different numbers of clusters using silhouette score
print("\nEvaluating different numbers of clusters...")
silhouette_scores = []
max_clusters = 10

for k in range(2, max_clusters + 1):
    # Get cluster labels
    labels = fcluster(Z, k, criterion='maxclust')
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled_sample, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, silhouette score is {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Score Method for Hierarchical Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig('hierarchical_results/hierarchical_silhouette_scores.png')
plt.close()

# Choose optimal number of clusters based on silhouette scores
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started from k=2
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

# Apply hierarchical clustering with the optimal number of clusters
print(f"\nPerforming hierarchical clustering with {optimal_k} clusters...")
labels = fcluster(Z, optimal_k, criterion='maxclust')
df_sample['Cluster'] = labels - 1  # Convert to 0-indexed clusters

# Count instances in each cluster
cluster_counts = df_sample['Cluster'].value_counts().sort_index()
print("\nCluster distribution:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} instances ({count/len(df_sample)*100:.2f}%)")

# Analyze clusters
print("\nAnalyzing clusters...")
cluster_analysis = df_sample.groupby('Cluster')[features].mean()
print("\nCluster centers (mean values for each feature):")
print(cluster_analysis)

# Save cluster analysis to CSV
cluster_analysis.to_csv('hierarchical_results/hierarchical_cluster_analysis.csv')

# Visualize clusters using PCA for dimensionality reduction
print("\nVisualizing clusters using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_sample)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_sample['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Video Game Sales Clusters Visualization with PCA (Hierarchical, k={optimal_k})')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.savefig('hierarchical_results/hierarchical_pca_visualization.png')
plt.close()

# Analyze relationship between clusters and sales
print("\nAnalyzing relationship between clusters and sales...")
if 'total_sales' in features:
    sales_by_cluster = df_sample.groupby('Cluster')['total_sales'].mean().sort_values()
    print("\nAverage total sales by cluster:")
    print(sales_by_cluster)

    plt.figure(figsize=(10, 6))
    sales_by_cluster.plot(kind='bar')
    plt.title('Average Total Sales by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Sales')
    plt.axhline(y=df_sample['total_sales'].mean(), color='red', linestyle='--', label='Overall Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hierarchical_results/hierarchical_sales_by_cluster.png')
    plt.close()

# Analyze relationship between clusters and critic scores if available
if 'critic_score' in features:
    scores_by_cluster = df_sample.groupby('Cluster')['critic_score'].mean().sort_values()
    print("\nAverage critic scores by cluster:")
    print(scores_by_cluster)

    plt.figure(figsize=(10, 6))
    scores_by_cluster.plot(kind='bar')
    plt.title('Average Critic Score by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Critic Score')
    plt.axhline(y=df_sample['critic_score'].mean(), color='red', linestyle='--', label='Overall Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hierarchical_results/hierarchical_scores_by_cluster.png')
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
plt.savefig('hierarchical_results/hierarchical_key_features_heatmap.png')
plt.close()

# Regional sales comparison by cluster if those columns exist
regional_cols = [col for col in features if any(region in col for region in ['na_', 'jp_', 'pal_', 'other_']) and 'sales' in col]
if regional_cols:
    print("\nAnalyzing regional sales patterns by cluster...")
    regional_means = df_sample.groupby('Cluster')[regional_cols].mean()
    
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
        if i in regional_means.index:  # Check if the cluster exists
            values = regional_means.loc[i].values.tolist()
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
    plt.savefig('hierarchical_results/hierarchical_regional_sales_radar.png')
    plt.close()

# Create a parallel coordinates plot for cluster visualization
plt.figure(figsize=(15, 8))
# Get a subset of data for parallel coordinates plot (can be too dense with all data)
sample_size = min(500, len(df_sample))
if len(df_sample) > sample_size:
    sample_indices = np.random.choice(len(df_sample), sample_size, replace=False)
    sample_df = df_sample.iloc[sample_indices].copy()
else:
    sample_df = df_sample.copy()

# Standardize the data for parallel coordinates plot
features_for_parallel = key_features[:7] if len(key_features) > 7 else key_features  # Limit to 7 features for readability
sample_df_scaled = sample_df.copy()
sample_df_scaled[features_for_parallel] = scaler.fit_transform(sample_df[features_for_parallel])

# Create parallel coordinates plot
pd.plotting.parallel_coordinates(
    sample_df_scaled, 'Cluster', 
    cols=features_for_parallel,
    colormap='viridis'
)
plt.title('Parallel Coordinates Plot of Video Game Clusters')
plt.grid(False)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('hierarchical_results/hierarchical_parallel_coordinates.png')
plt.close()

# Summarize the characteristics of each cluster
print("\nCluster Characteristics Summary:")
for cluster in range(optimal_k):
    if cluster not in cluster_analysis.index:
        continue  # Skip clusters that don't exist (possible if using 0-indexed)
    
    print(f"\nCluster {cluster}:")
    # Get the top 5 distinctive features for this cluster (highest absolute z-scores)
    if cluster in key_cluster_centers_scaled.index:
        cluster_features = key_cluster_centers_scaled.loc[cluster].abs().sort_values(ascending=False)
        top_features = cluster_features.head(5).index.tolist()
        
        for feature in top_features:
            raw_value = key_cluster_centers.loc[cluster, feature]
            scaled_value = key_cluster_centers_scaled.loc[cluster, feature]
            direction = "high" if float(scaled_value) > 0 else "low"
            print(f"  - {feature}: {direction} ({raw_value:.2f}, z-score: {scaled_value:.2f})")
    
    # Optional: Display additional cluster characteristics based on available metrics
    if regional_cols and cluster in regional_means.index:
        dominant_region = regional_means.loc[cluster].idxmax()
        print(f"  - Dominant sales region: {dominant_region}")
        
    # Add any other relevant information about the clusters
    if cluster in cluster_counts.index:
        cluster_size = cluster_counts[cluster]
        cluster_percentage = cluster_size / len(df_sample) * 100
        print(f"  - Size: {cluster_size} games ({cluster_percentage:.2f}% of dataset)")

# Sample titles from each cluster if title column exists
if 'title' in df.columns:
    print("\nSample titles from each cluster:")
    for cluster in range(optimal_k):
        if cluster in df_sample['Cluster'].values:  # Check if cluster exists in results
            cluster_games = df_sample[df_sample['Cluster'] == cluster]
            sample_size = min(5, len(cluster_games))
            if sample_size > 0:
                sample_titles = cluster_games['title'].sample(sample_size).tolist()
                print(f"\nCluster {cluster} sample titles:")
                for title in sample_titles:
                    print(f"  - {title}")

# Create correlation matrix between original features and PCA components
print("\nCreating correlation matrix between features and PCA components...")
pca_components = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(2)],
    index=features
)
plt.figure(figsize=(10, 12))
sns.heatmap(pca_components, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Feature Correlation with Principal Components')
plt.tight_layout()
plt.savefig('hierarchical_results/hierarchical_pca_correlation.png')
plt.close()

print("\nHierarchical clustering analysis complete. Results saved to 'hierarchical_results' directory.") 