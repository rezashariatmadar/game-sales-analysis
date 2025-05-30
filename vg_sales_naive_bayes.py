import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import os

# Create directory for Naive Bayes results
os.makedirs('naive_bayes_results', exist_ok=True)

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

# Scale features - MinMaxScaler tends to work better with Naive Bayes than StandardScaler
# as it preserves the distribution shape while bounding values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Gaussian Naive Bayes
# -------------------------
print("\n---- Gaussian Naive Bayes ----")
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred_gnb = gnb.predict(X_test_scaled)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"Gaussian Naive Bayes accuracy: {accuracy_gnb:.4f}")

# Probability predictions for ROC curve
y_pred_prob_gnb = gnb.predict_proba(X_test_scaled)[:, 1]

# Print classification report
print("\nClassification Report (Gaussian Naive Bayes):")
gnb_report = classification_report(y_test, y_pred_gnb)
print(gnb_report)

# Save classification report to file
with open('naive_bayes_results/gnb_classification_report.txt', 'w') as f:
    f.write("Classification Report for Gaussian Naive Bayes on Video Game Sales:\n")
    f.write(str(gnb_report))

# Confusion matrix
conf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_gnb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Sales', 'High Sales'],
            yticklabels=['Low Sales', 'High Sales'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Gaussian Naive Bayes')
plt.tight_layout()
plt.savefig('naive_bayes_results/gnb_confusion_matrix.png')
plt.close()

# -------------------------
# Bernoulli Naive Bayes 
# -------------------------
print("\n---- Bernoulli Naive Bayes ----")
# Bernoulli NB works with binary features, so we need to consider this fact
bnb = BernoulliNB()
bnb.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred_bnb = bnb.predict(X_test_scaled)
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
print(f"Bernoulli Naive Bayes accuracy: {accuracy_bnb:.4f}")

# Probability predictions for ROC curve
y_pred_prob_bnb = bnb.predict_proba(X_test_scaled)[:, 1]

# Print classification report
print("\nClassification Report (Bernoulli Naive Bayes):")
bnb_report = classification_report(y_test, y_pred_bnb)
print(bnb_report)

# Save classification report to file
with open('naive_bayes_results/bnb_classification_report.txt', 'w') as f:
    f.write("Classification Report for Bernoulli Naive Bayes on Video Game Sales:\n")
    f.write(str(bnb_report))

# Confusion matrix
conf_matrix_bnb = confusion_matrix(y_test, y_pred_bnb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_bnb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Sales', 'High Sales'],
            yticklabels=['Low Sales', 'High Sales'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Bernoulli Naive Bayes')
plt.tight_layout()
plt.savefig('naive_bayes_results/bnb_confusion_matrix.png')
plt.close()

# -------------------------
# Cross-validation for both models
# -------------------------
print("\n---- Cross-validation ----")

# Cross-validation for Gaussian NB
cv_scores_gnb = cross_val_score(gnb, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Gaussian NB 5-fold CV accuracy: {cv_scores_gnb.mean():.4f} ± {cv_scores_gnb.std():.4f}")

# Cross-validation for Bernoulli NB
cv_scores_bnb = cross_val_score(bnb, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Bernoulli NB 5-fold CV accuracy: {cv_scores_bnb.mean():.4f} ± {cv_scores_bnb.std():.4f}")

# -------------------------
# ROC Curves and AUC
# -------------------------
plt.figure(figsize=(10, 8))

# ROC for Gaussian NB
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_prob_gnb)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)
plt.plot(fpr_gnb, tpr_gnb, label=f'Gaussian NB (AUC = {roc_auc_gnb:.3f})')

# ROC for Bernoulli NB
fpr_bnb, tpr_bnb, _ = roc_curve(y_test, y_pred_prob_bnb)
roc_auc_bnb = auc(fpr_bnb, tpr_bnb)
plt.plot(fpr_bnb, tpr_bnb, label=f'Bernoulli NB (AUC = {roc_auc_bnb:.3f})')

# Reference line (random classifier)
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Naive Bayes Classifiers')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('naive_bayes_results/roc_curves.png')
plt.close()

# -------------------------
# Feature importance for Naive Bayes
# -------------------------
# For Naive Bayes, we can look at the difference in log probability between classes
# This gives us an indication of how much each feature contributes to the classification

# Function to calculate feature importance for Gaussian NB
def compute_feature_importance_gnb(model, feature_names):
    # Get the feature means for each class
    theta_0 = model.theta_[0]  # Mean for class 0
    theta_1 = model.theta_[1]  # Mean for class 1
    
    # Get the feature variances for each class
    sigma_0 = model.var_[0]  # Variance for class 0
    sigma_1 = model.var_[1]  # Variance for class 1
    
    # Calculate the absolute difference in means, normalized by variance
    # This gives a measure of how discriminative each feature is
    importance = np.abs(theta_1 - theta_0) / np.sqrt((sigma_0 + sigma_1) / 2)
    
    # Create DataFrame with feature names and importance scores
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

# Calculate feature importance for Gaussian NB
gnb_feature_importance = compute_feature_importance_gnb(gnb, features)

print("\nTop 10 Important Features (Gaussian NB):")
print(gnb_feature_importance.head(min(10, len(features))))

# Save feature importance to CSV
gnb_feature_importance.to_csv('naive_bayes_results/gnb_feature_importance.csv', index=False)

# Plot feature importance (top 15)
plt.figure(figsize=(12, 8))
top_features_gnb = gnb_feature_importance.head(min(15, len(features)))
sns.barplot(x='Importance', y='Feature', data=top_features_gnb)
plt.title('Top 15 Features by Importance (Gaussian NB)')
plt.tight_layout()
plt.savefig('naive_bayes_results/gnb_feature_importance.png')
plt.close()

# -------------------------
# Compare with decision tree results if available
# -------------------------
print("\n---- Comparison with Decision Tree ----")

# Create comparison table with the Naive Bayes models
comparison_data = {
    'Model': ['Gaussian NB', 'Bernoulli NB'],
    'Accuracy': [accuracy_gnb, accuracy_bnb],
    'CV Accuracy': [cv_scores_gnb.mean(), cv_scores_bnb.mean()],
    'AUC': [roc_auc_gnb, roc_auc_bnb]
}

# Try to load Decision Tree results from file
try:
    with open('decision_tree_results/optimized_classification_report.txt', 'r') as f:
        dt_report = f.read()
    print("Decision Tree results loaded from file.")
    
    # Add Decision Tree to comparison if we can extract accuracy
    # This is a simplistic approach; in a real application you might parse the report more carefully
    import re
    accuracy_match = re.search(r'accuracy\s*:\s*([\d\.]+)', dt_report)
    if accuracy_match:
        dt_accuracy = float(accuracy_match.group(1))
        comparison_data['Model'].insert(0, 'Decision Tree')
        comparison_data['Accuracy'].insert(0, dt_accuracy)
        comparison_data['CV Accuracy'].insert(0, None)
        comparison_data['AUC'].insert(0, None)
except FileNotFoundError:
    print("Decision Tree results file not found. Comparison will only include Naive Bayes models.")

# Create and display comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.fillna('Not available')
print("\nModel Comparison:")
print(comparison_df)

# Save comparison to file
comparison_df.to_csv('naive_bayes_results/model_comparison.csv', index=False)

# -------------------------
# Create prediction function
# -------------------------
# Determine the best Naive Bayes model
best_nb_model = gnb if accuracy_gnb >= accuracy_bnb else bnb
best_nb_name = "Gaussian NB" if accuracy_gnb >= accuracy_bnb else "Bernoulli NB"
best_accuracy = accuracy_gnb if accuracy_gnb >= accuracy_bnb else accuracy_bnb
print(f"\nBest Naive Bayes model: {best_nb_name} (Accuracy: {best_accuracy:.4f})")

def predict_sales_class_nb(sample_data, model=best_nb_model, scaler=scaler, features=features):
    """
    Predict sales class for a new video game using Naive Bayes.
    
    Parameters:
    sample_data : dict
        Dictionary with feature names and values
    model : trained model
        Trained Naive Bayes model
    scaler : fitted scaler
        Fitted scaler
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
    'release_date_year': 2018,
    'console_freq': 5,
    'genre_freq': 10,
    'publisher_freq': 3,
    # Add other features as needed
}

# Print only features in the sample that are in our feature list
present_features = {k: v for k, v in sample.items() if k in features}
print(f"Sample features: {present_features}")

# Make a prediction
try:
    pred, prob = predict_sales_class_nb(sample)
    print(f"Prediction: {'High Sales' if pred == 1 else 'Low Sales'}")
    print(f"Probability of high sales: {prob:.2%}")
except Exception as e:
    print(f"Error making prediction: {e}")
    print("This is just an example. You may need to provide values for all required features.")

# Additional analysis: Feature distributions by sales class
print("\nGenerating feature distribution plots...")
top_5_features = gnb_feature_importance.head(min(5, len(features)))['Feature'].tolist()
for feature in top_5_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='high_sales', bins=20, element='step', kde=False)
    plt.title(f'Distribution of {feature} by Sales Class')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'naive_bayes_results/feature_distribution_{feature}.png')
    plt.close()

print("\nNaive Bayes analysis complete. Results saved to 'naive_bayes_results' directory.") 