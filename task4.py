import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

# Create directory for clustering results
os.makedirs('hierarchical_results', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed dataset...")
df = pd.read_csv('processed_data/student_depression_processed.csv')
print(f"Dataset shape: {df.shape}")

# Select features for clustering
# Remove ID and target variable (Depression)
features = df.drop(['id', 'Depression'], axis=1).columns.tolist()
X = df[features].values

print(f"Selected {len(features)} features for clustering")
print("Features:", features)

# Standardize the data for hierarchical clustering
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute the linkage matrix using Ward's method
print("Computing linkage matrix with Ward's method...")
Z = linkage(X_scaled, method='ward')

# Plot the dendrogram to visualize hierarchical structure
plt.figure(figsize=(12, 8))
plt.title('Hierarchical Clustering Dendrogram')
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
    silhouette_avg = silhouette_score(X_scaled, labels)
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
df['Cluster'] = labels - 1  # Convert to 0-indexed clusters

# Count instances in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()
print("\nCluster distribution:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} instances ({count/len(df)*100:.2f}%)")

# Analyze clusters
print("\nAnalyzing clusters...")
cluster_analysis = df.groupby('Cluster').mean()[features]
print("\nCluster centers (mean values for each feature):")
print(cluster_analysis)

# Save cluster analysis to CSV
cluster_analysis.to_csv('hierarchical_results/hierarchical_cluster_analysis.csv')

# Visualize clusters using PCA for dimensionality reduction
print("\nVisualizing clusters using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Clusters Visualization with PCA (Hierarchical, k={optimal_k})')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.savefig('hierarchical_results/hierarchical_pca_visualization.png')
plt.close()

# Analyze relationship between clusters and depression
print("\nAnalyzing relationship between clusters and depression...")
depression_by_cluster = df.groupby('Cluster')['Depression'].mean().sort_values()
print("\nAverage depression score by cluster:")
print(depression_by_cluster)

plt.figure(figsize=(10, 6))
depression_by_cluster.plot(kind='bar')
plt.title('Average Depression Score by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Depression Score')
plt.axhline(y=df['Depression'].mean(), color='red', linestyle='--', label='Overall Average')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hierarchical_results/hierarchical_depression_by_cluster.png')
plt.close()

# Analyze key features by cluster
# Select a subset of important features to visualize
key_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
                'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
                'Family History of Mental Illness', 'Combined_Stress_Index']
key_features = [f for f in key_features if f in features]

# Create a heatmap of cluster centers for key features
plt.figure(figsize=(12, 8))
key_cluster_centers = cluster_analysis[key_features]
# Normalize the data for better visualization
key_cluster_centers_scaled = pd.DataFrame(
    scaler.fit_transform(key_cluster_centers),
    index=key_cluster_centers.index,
    columns=key_cluster_centers.columns
)

# Create a heatmap
sns.heatmap(key_cluster_centers_scaled, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Normalized Cluster Centers for Key Features')
plt.ylabel('Cluster')
plt.tight_layout()
plt.savefig('hierarchical_results/hierarchical_key_features_heatmap.png')
plt.close()

# Create a parallel coordinates plot for cluster visualization
plt.figure(figsize=(15, 8))
# Get a subset of data for parallel coordinates plot (can be too dense with all data)
sample_size = min(1000, len(df))
sample_indices = np.random.choice(len(df), sample_size, replace=False)
sample_df = df.iloc[sample_indices].copy()

# Standardize the data for parallel coordinates plot
features_for_parallel = key_features
sample_df_scaled = sample_df.copy()
sample_df_scaled[features_for_parallel] = scaler.fit_transform(sample_df[features_for_parallel])

# Create parallel coordinates plot
pd.plotting.parallel_coordinates(
    sample_df_scaled, 'Cluster', 
    cols=features_for_parallel,
    color=plt.cm.viridis(np.linspace(0, 1, optimal_k))
)
plt.title('Parallel Coordinates Plot of Clusters')
plt.grid(False)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('hierarchical_results/hierarchical_parallel_coordinates.png')
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
    
    # Depression info for this cluster
    cluster_depression = df[df['Cluster'] == cluster]['Depression'].mean()
    overall_depression = df['Depression'].mean()
    depression_diff = cluster_depression - overall_depression
    
    depression_status = "higher" if depression_diff > 0 else "lower"
    print(f"  - Depression: {depression_status} than average by {abs(depression_diff):.2f} ({cluster_depression:.2f} vs. {overall_depression:.2f} overall)")

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