import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from datetime import datetime

# Create directory for decision tree results
os.makedirs('decision_tree_results', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed video game sales dataset...")
df = pd.read_csv('processed_data/vgchartz_processed.csv')
print(f"Dataset shape: {df.shape}")

# Handle date columns if they exist
date_columns = ['release_date', 'last_update']
for col in date_columns:
    if col in df.columns:
        print(f"Converting {col} to datetime and extracting year")
        try:
            # Try to convert to datetime and extract year as a numeric feature
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            
            # Drop original date column as it cannot be used directly
            df.drop(col, axis=1, inplace=True)
        except:
            print(f"Error processing {col}, dropping it")
            df.drop(col, axis=1, inplace=True)

# Need to create a target variable since this is sales data
# Let's create a categorical target based on total sales
# Create high/low sales binary classification (1 for high sales, 0 for low sales)
# Using median as threshold to ensure balanced classes
sales_median = df['total_sales'].median()
df['high_sales'] = (df['total_sales'] > sales_median).astype(int)
print(f"High sales threshold (median): {sales_median:.2f}")
print(f"High sales prevalence: {df['high_sales'].mean():.2%}")

# Select features for classification
# Remove identifier columns and target
exclude_cols = ['title', 'platform', 'genre', 'publisher', 'developer', 'high_sales', 'total_sales']

# Also remove any features that are directly derived from total_sales to avoid data leakage
derived_cols = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'sales_per_year']
exclude_cols.extend(derived_cols)

print(f"Excluding columns: {exclude_cols}")
features = [col for col in df.columns if col not in exclude_cols]

# Check for NaN values in the features and fill them
nan_counts = df[features].isna().sum()
if nan_counts.sum() > 0:
    print("\nFound NaN values in features. Filling with medians.")
    for col in features:
        if nan_counts[col] > 0:
            df[col] = df[col].fillna(df[col].median())

# Check for non-numeric columns
non_numeric_cols = df[features].select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_cols:
    print(f"\nFound non-numeric columns: {non_numeric_cols}")
    print("Dropping non-numeric columns for classification")
    features = [col for col in features if col not in non_numeric_cols]

X = df[features].values
y = df['high_sales'].values

print(f"Selected {len(features)} features for classification")
print("Features:", features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a decision tree with default parameters
print("\nTraining a simple decision tree classifier...")
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = dt_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision tree accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Save classification report to file
with open('decision_tree_results/classification_report.txt', 'w') as f:
    f.write("Classification Report for Decision Tree on Video Game Sales:\n")
    f.write(str(report))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Sales', 'High Sales'],
            yticklabels=['Low Sales', 'High Sales'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Decision Tree')
plt.tight_layout()
plt.savefig('decision_tree_results/confusion_matrix.png')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': dt_clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(min(10, len(features))))

# Save feature importance to CSV
feature_importance.to_csv('decision_tree_results/feature_importance.csv', index=False)

# Plot feature importance (top 15)
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(min(15, len(features)))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top Features by Importance')
plt.tight_layout()
plt.savefig('decision_tree_results/feature_importance.png')
plt.close()

# Hyperparameter tuning using GridSearchCV
print("\nPerforming hyperparameter tuning for decision tree...")
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train model with best parameters
best_dt = grid_search.best_estimator_
best_dt.fit(X_train_scaled, y_train)

# Evaluate optimized model
y_pred_best = best_dt.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Optimized decision tree accuracy: {best_accuracy:.4f}")

# Print classification report for optimized model
print("\nClassification Report (Optimized Model):")
best_report = classification_report(y_test, y_pred_best)
print(best_report)

# Save classification report for optimized model
with open('decision_tree_results/optimized_classification_report.txt', 'w') as f:
    f.write("Classification Report for Optimized Decision Tree on Video Game Sales:\n")
    f.write(str(best_report))

# Confusion matrix for optimized model
best_conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Sales', 'High Sales'],
            yticklabels=['Low Sales', 'High Sales'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Optimized Decision Tree')
plt.tight_layout()
plt.savefig('decision_tree_results/optimized_confusion_matrix.png')
plt.close()

# Visualize the optimized decision tree (if not too large)
max_depth_for_visualization = 3  # Limit for visualization

# If tree is too deep, create a simpler one just for visualization
vis_dt = DecisionTreeClassifier(max_depth=max_depth_for_visualization, random_state=42)
vis_dt.fit(X_train_scaled, y_train)
tree_to_visualize = vis_dt
tree_title = f"Decision Tree (Limited to depth {max_depth_for_visualization} for visualization)"

# Plot the tree
plt.figure(figsize=(20, 15))
plot_tree(
    tree_to_visualize,
    feature_names=features,
    class_names=['Low Sales', 'High Sales'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title(tree_title)
plt.savefig('decision_tree_results/decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Export text representation of the tree
tree_text = export_text(
    tree_to_visualize,
    feature_names=features
)

with open('decision_tree_results/decision_tree_text.txt', 'w') as f:
    f.write(tree_text)

# Check if we have any sales ratio columns left after removing derived columns
regional_ratio_cols = [col for col in features if '_sales_ratio' in col]
if regional_ratio_cols:
    # Create regional sales ratio plot by sales class
    plt.figure(figsize=(10, 6))
    sales_class_means = df.groupby('high_sales')[regional_ratio_cols].mean()
    
    sales_class_means.T.plot(kind='bar')
    plt.title('Regional Sales Ratio by Sales Class')
    plt.xlabel('Region')
    plt.ylabel('Average Sales Ratio')
    plt.legend(['Low Sales', 'High Sales'])
    plt.tight_layout()
    plt.savefig('decision_tree_results/regional_sales_by_class.png')
    plt.close()

# Create feature distributions by sales class
top_5_features = feature_importance.head(min(5, len(features)))['Feature'].tolist()
for feature in top_5_features:
    plt.figure(figsize=(10, 6))
    # Using histplot without KDE to avoid errors
    sns.histplot(data=df, x=feature, hue='high_sales', bins=20, element='step', kde=False)
    plt.title(f'Distribution of {feature} by Sales Class')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'decision_tree_results/feature_distribution_{feature}.png')
    plt.close()

# Create correlation matrix for features
corr_matrix = df[features + ['high_sales']].corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
            square=True, linewidths=.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('decision_tree_results/correlation_matrix.png')
plt.close()

print("\nDecision tree analysis complete. Results saved to 'decision_tree_results' directory.")

# Create a function for predicting new games' sales class
def predict_sales_class(sample_data, model=best_dt, scaler=scaler, features=features):
    """
    Predict sales class for a new video game.
    
    Parameters:
    sample_data : dict
        Dictionary with feature names and values
    model : trained model
        Trained decision tree model
    scaler : fitted scaler
        Fitted StandardScaler
    features : list
        List of feature names
    
    Returns:
    prediction : int
        0 for low sales, 1 for high sales
    prob : float
        Probability of high sales
    """
    # Convert sample to array in the correct order
    sample_array = np.array([sample_data.get(feature, 0) for feature in features]).reshape(1, -1)
    
    # Scale the sample
    sample_scaled = scaler.transform(sample_array)
    
    # Predict
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]
    
    return prediction, probability

# Example usage
print("\nExample prediction:")
# Create a sample (you should replace this with actual feature values)
sample = {
    'critic_score': 80,
    'release_year': 2018,
    'console_freq': 5,
    'genre_freq': 10,
    'publisher_freq': 3,
    # Add other features as needed
}

# Print only features in the sample that match our feature list
present_features = {k: v for k, v in sample.items() if k in features}
print(f"Sample features: {present_features}")

# Make a prediction
try:
    pred, prob = predict_sales_class(sample)
    print(f"Prediction: {'High Sales' if pred == 1 else 'Low Sales'}")
    print(f"Probability of high sales: {prob:.2%}")
except Exception as e:
    print(f"Error making prediction: {e}")
    print("This is just an example. You may need to provide values for all required features.") 