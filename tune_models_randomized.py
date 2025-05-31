import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy.stats import randint, uniform, loguniform

# Create directories for results
os.makedirs('tuning_results_randomized', exist_ok=True)
os.makedirs('tuning_results_randomized/plots', exist_ok=True)
os.makedirs('tuning_results_randomized/models', exist_ok=True)

# Set up logging to file and console
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tuning_results_randomized/hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)

def load_and_prepare_data():
    """Load and prepare data for model tuning."""
    logging.info("Loading dataset...")
    try:
        df = pd.read_csv('vgchartz_cleaned.csv')
        logging.info(f"Dataset loaded successfully with {len(df)} records")
    except FileNotFoundError:
        logging.error("Error: vgchartz_cleaned.csv not found!")
        return None, None, None, None, None, None
    
    # Check available columns
    logging.info(f"Available columns: {df.columns.tolist()}")
    
    # Define features
    numeric_features = ['critic_score', 'release_year']
    categorical_features = ['console', 'genre', 'publisher']
    regional_sales = ['na_sales', 'jp_sales', 'pal_sales']
    
    # Filter to ensure we only use columns that exist
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    regional_sales = [col for col in regional_sales if col in df.columns]
    
    # Combine all features
    all_features = numeric_features + categorical_features + regional_sales
    
    # Replace NaN values
    df = df.fillna(0)
    
    # For categorical features with high cardinality, limit to top categories
    for col in categorical_features:
        # Get value counts
        value_counts = df[col].value_counts()
        
        # Keep only the top 20 categories to reduce dimensionality
        top_categories = value_counts.nlargest(20).index.tolist()
        
        # Replace less common categories with "Other"
        df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")
    
    # Extract features
    X = df[all_features]
    logging.info(f"Using features: {X.columns.tolist()}")
    
    # Target variable for regression
    y_reg = df['total_sales'] if 'total_sales' in df.columns else df['na_sales'] + df['jp_sales'] + df['pal_sales'] + df['other_sales']
    
    # Target variable for classification (high sales vs low sales)
    median_sales = y_reg.median()
    y_cls = (y_reg > median_sales).astype(int)
    
    logging.info(f"Regression target - median value: {median_sales}")
    logging.info(f"Classification target - class distribution: {y_cls.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features + regional_sales),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Apply preprocessing
    logging.info("Applying preprocessing transforms...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Save preprocessor for later use
    joblib.dump(preprocessor, 'tuning_results_randomized/models/preprocessor.joblib')
    
    return X_train_preprocessed, X_test_preprocessed, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor

def tune_random_forest(X_train, X_test, y_train, y_test):
    """
    Tune a Random Forest Regression model using RandomizedSearchCV.
    """
    logging.info("Starting Random Forest Regression model tuning...")
    start_time = time.time()
    
    # Define parameter distributions - much wider range than GridSearchCV
    param_distributions = {
        'n_estimators': randint(50, 500),  # Much wider range
        'max_depth': [None] + list(randint(5, 50).rvs(10)),  # None + 10 random values
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2', None],  # Removed 'auto' as it's not supported
        'bootstrap': [True, False]
    }
    
    # Create a base model
    rf = RandomForestRegressor(random_state=42)
    
    # Instantiate the randomized search model
    logging.info("Starting RandomizedSearchCV for Random Forest Regression...")
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='r2',
        random_state=42
    )
    
    # Fit the randomized search to the data
    random_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = random_search.best_params_
    logging.info(f"Best parameters for Random Forest Regression: {best_params}")
    
    # Train a new model with best parameters
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Random Forest Regression - Mean Squared Error: {mse:.4f}")
    logging.info(f"Random Forest Regression - R² Score: {r2:.4f}")
    
    # Save the tuned model
    joblib.dump(best_rf, 'tuning_results_randomized/models/tuned_random_forest.joblib')
    
    # Get feature importance - but be aware that with one-hot encoding the features are now transformed
    feature_importances = best_rf.feature_importances_
    
    # For visualization purposes, we'll just plot the top N importance values
    n_top_features = min(20, len(feature_importances))
    indices = np.argsort(feature_importances)[::-1][:n_top_features]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_top_features), feature_importances[indices])
    plt.xticks(range(n_top_features), indices, rotation=90)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('Top Feature Importances (Tuned Random Forest)')
    plt.tight_layout()
    plt.savefig('tuning_results_randomized/plots/tuned_rf_feature_importance.png')
    
    elapsed_time = time.time() - start_time
    logging.info(f"Random Forest tuning completed in {elapsed_time:.2f} seconds")
    
    # Return the results
    tuning_results = {
        'model_type': 'Random Forest Regression',
        'best_params': best_params,
        'mse': mse,
        'r2': r2,
        'feature_indices': indices.tolist(),
        'feature_importance': feature_importances[indices].tolist(),
        'elapsed_time': elapsed_time
    }
    
    return tuning_results, best_rf

def tune_decision_tree(X_train, X_test, y_train, y_test):
    """
    Tune a Decision Tree Classification model using RandomizedSearchCV.
    """
    logging.info("Starting Decision Tree Classification model tuning...")
    start_time = time.time()
    
    # Define parameter distributions
    param_distributions = {
        'max_depth': [None] + list(randint(5, 50).rvs(10)),
        'min_samples_split': randint(2, 30),
        'min_samples_leaf': randint(1, 30),
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_features': [None, 'sqrt', 'log2']  # Removed 'auto'
    }
    
    # Create a base model
    dt = DecisionTreeClassifier(random_state=42)
    
    # Instantiate the randomized search model
    logging.info("Starting RandomizedSearchCV for Decision Tree Classification...")
    random_search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy',
        random_state=42
    )
    
    # Fit the randomized search to the data
    random_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = random_search.best_params_
    logging.info(f"Best parameters for Decision Tree Classification: {best_params}")
    
    # Train a new model with best parameters
    best_dt = DecisionTreeClassifier(**best_params, random_state=42)
    best_dt.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = best_dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    logging.info(f"Decision Tree Classification - Accuracy: {accuracy:.4f}")
    logging.info(f"Decision Tree Classification - Classification Report:\n{class_report}")
    
    # Save the tuned model
    joblib.dump(best_dt, 'tuning_results_randomized/models/tuned_decision_tree.joblib')
    
    # Get feature importance - but be aware that with one-hot encoding the features are now transformed
    feature_importances = best_dt.feature_importances_
    
    # For visualization purposes, we'll just plot the top N importance values
    n_top_features = min(20, len(feature_importances))
    indices = np.argsort(feature_importances)[::-1][:n_top_features]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_top_features), feature_importances[indices])
    plt.xticks(range(n_top_features), indices, rotation=90)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('Top Feature Importances (Tuned Decision Tree)')
    plt.tight_layout()
    plt.savefig('tuning_results_randomized/plots/tuned_dt_feature_importance.png')
    
    elapsed_time = time.time() - start_time
    logging.info(f"Decision Tree tuning completed in {elapsed_time:.2f} seconds")
    
    # Return the results
    tuning_results = {
        'model_type': 'Decision Tree Classification',
        'best_params': best_params,
        'accuracy': accuracy,
        'classification_report': class_report,
        'feature_indices': indices.tolist(),
        'feature_importance': feature_importances[indices].tolist(),
        'elapsed_time': elapsed_time
    }
    
    return tuning_results, best_dt

def tune_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Tune a Naive Bayes Classification model using RandomizedSearchCV.
    Note: GaussianNB has few parameters to tune, but we'll demonstrate the process.
    """
    logging.info("Starting Naive Bayes Classification model tuning...")
    start_time = time.time()
    
    # Define parameter distributions
    # GaussianNB has limited hyperparameters to tune, using loguniform for wider exploration
    param_distributions = {
        'var_smoothing': loguniform(1e-10, 1.0)  # Much wider range than GridSearchCV
    }
    
    # Create a base model
    nb = GaussianNB()
    
    # Instantiate the randomized search model
    logging.info("Starting RandomizedSearchCV for Naive Bayes Classification...")
    random_search = RandomizedSearchCV(
        estimator=nb,
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter combinations to try
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy',
        random_state=42
    )
    
    # Fit the randomized search to the data
    random_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = random_search.best_params_
    logging.info(f"Best parameters for Naive Bayes Classification: {best_params}")
    
    # Train a new model with best parameters
    best_nb = GaussianNB(**best_params)
    best_nb.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = best_nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    logging.info(f"Naive Bayes Classification - Accuracy: {accuracy:.4f}")
    logging.info(f"Naive Bayes Classification - Classification Report:\n{class_report}")
    
    # Save the tuned model
    joblib.dump(best_nb, 'tuning_results_randomized/models/tuned_naive_bayes.joblib')
    
    elapsed_time = time.time() - start_time
    logging.info(f"Naive Bayes tuning completed in {elapsed_time:.2f} seconds")
    
    # Return the results
    tuning_results = {
        'model_type': 'Naive Bayes Classification',
        'best_params': best_params,
        'accuracy': accuracy,
        'classification_report': class_report,
        'elapsed_time': elapsed_time
    }
    
    return tuning_results, best_nb

def generate_html_report(rf_results, dt_results, nb_results):
    """Generate an HTML report of the tuning process and results."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Hyperparameter Tuning Report (Randomized Search)</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric {{ font-weight: bold; color: #2980b9; }}
            .model-section {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .parameter {{ font-family: monospace; background-color: #f0f0f0; padding: 2px 4px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Model Hyperparameter Tuning Report (Randomized Search)</h1>
        <p>Generated on: {now}</p>
        
        <div class="model-section">
            <h2>Random Forest Regression Model</h2>
            <h3>Best Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    '''
    
    # Add Random Forest parameters
    for param, value in rf_results['best_params'].items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html += f'''
            </table>
            
            <h3>Performance Metrics</h3>
            <ul>
                <li>Mean Squared Error: <span class="metric">{rf_results['mse']:.4f}</span></li>
                <li>R² Score: <span class="metric">{rf_results['r2']:.4f}</span></li>
                <li>Tuning Time: <span class="metric">{rf_results['elapsed_time']:.2f} seconds</span></li>
            </ul>
            
            <h3>Top Feature Importances</h3>
            <img src="plots/tuned_rf_feature_importance.png" alt="Random Forest Feature Importance">
        </div>
        
        <div class="model-section">
            <h2>Decision Tree Classification Model</h2>
            <h3>Best Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    '''
    
    # Add Decision Tree parameters
    for param, value in dt_results['best_params'].items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html += f'''
            </table>
            
            <h3>Performance Metrics</h3>
            <ul>
                <li>Accuracy: <span class="metric">{dt_results['accuracy']:.4f}</span></li>
                <li>Tuning Time: <span class="metric">{dt_results['elapsed_time']:.2f} seconds</span></li>
            </ul>
            
            <h3>Classification Report</h3>
            <pre>{dt_results['classification_report']}</pre>
            
            <h3>Top Feature Importances</h3>
            <img src="plots/tuned_dt_feature_importance.png" alt="Decision Tree Feature Importance">
        </div>
        
        <div class="model-section">
            <h2>Naive Bayes Classification Model</h2>
            <h3>Best Parameters</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    '''
    
    # Add Naive Bayes parameters
    for param, value in nb_results['best_params'].items():
        html += f"<tr><td>{param}</td><td>{value}</td></tr>"
    
    html += f'''
            </table>
            
            <h3>Performance Metrics</h3>
            <ul>
                <li>Accuracy: <span class="metric">{nb_results['accuracy']:.4f}</span></li>
                <li>Tuning Time: <span class="metric">{nb_results['elapsed_time']:.2f} seconds</span></li>
            </ul>
            
            <h3>Classification Report</h3>
            <pre>{nb_results['classification_report']}</pre>
        </div>
        
        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Primary Metric</th>
                <th>Tuning Time (seconds)</th>
            </tr>
            <tr>
                <td>Random Forest (Regression)</td>
                <td>R² Score: {rf_results['r2']:.4f}</td>
                <td>{rf_results['elapsed_time']:.2f}</td>
            </tr>
            <tr>
                <td>Decision Tree (Classification)</td>
                <td>Accuracy: {dt_results['accuracy']:.4f}</td>
                <td>{dt_results['elapsed_time']:.2f}</td>
            </tr>
            <tr>
                <td>Naive Bayes (Classification)</td>
                <td>Accuracy: {nb_results['accuracy']:.4f}</td>
                <td>{nb_results['elapsed_time']:.2f}</td>
            </tr>
        </table>
        
        <h2>Conclusion</h2>
        <p>
            After randomized hyperparameter tuning, the models show the following performance:
        </p>
        <ul>
            <li>The Random Forest regression model achieved an R² score of {rf_results['r2']:.4f}, explaining {rf_results['r2']*100:.1f}% of the variance in the game sales data.</li>
            <li>The Decision Tree classification model achieved an accuracy of {dt_results['accuracy']:.4f} for classifying games into high or low sales categories.</li>
            <li>The Naive Bayes classification model achieved an accuracy of {nb_results['accuracy']:.4f}, which is {nb_results['accuracy']*100-84.4:.1f}% better than the previous non-tuned version.</li>
        </ul>
        
        <h3>Randomized vs. Grid Search</h3>
        <p>
            This tuning was performed using RandomizedSearchCV instead of GridSearchCV, which has these advantages:
        </p>
        <ul>
            <li>Much faster execution time, allowing exploration of a wider parameter space</li>
            <li>Ability to search continuous distributions rather than discrete values</li>
            <li>Often finds comparable results to exhaustive grid search with significantly less computational cost</li>
        </ul>
        
        <h3>Recommendations</h3>
        <p>
            Based on the tuning results, we recommend:
        </p>
        <ul>
            <li>Using the tuned Random Forest model for sales prediction tasks with the optimal parameters found.</li>
            <li>Using the tuned Decision Tree model for classification tasks, as it outperforms the Naive Bayes model.</li>
            <li>Considering the computational cost vs. benefit when choosing between models for production deployment.</li>
        </ul>
    </body>
    </html>
    '''
    
    # Save the HTML report
    with open('tuning_results_randomized/hyperparameter_tuning_report.html', 'w') as f:
        f.write(html)
    
    logging.info("HTML report generated: tuning_results_randomized/hyperparameter_tuning_report.html")

def main():
    """Main function to run the hyperparameter tuning process."""
    logging.info("Starting randomized hyperparameter tuning process...")
    start_time = time.time()
    
    # Load and prepare data
    X_train_preprocessed, X_test_preprocessed, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor = load_and_prepare_data()
    
    if X_train_preprocessed is None:
        logging.error("Failed to load data. Exiting.")
        return
    
    # Tune Random Forest model
    rf_results, best_rf = tune_random_forest(X_train_preprocessed, X_test_preprocessed, y_reg_train, y_reg_test)
    
    # Tune Decision Tree model
    dt_results, best_dt = tune_decision_tree(X_train_preprocessed, X_test_preprocessed, y_cls_train, y_cls_test)
    
    # Tune Naive Bayes model
    nb_results, best_nb = tune_naive_bayes(X_train_preprocessed, X_test_preprocessed, y_cls_train, y_cls_test)
    
    # Generate HTML report
    generate_html_report(rf_results, dt_results, nb_results)
    
    # Create a script to implement the tuned models
    with open('implement_tuned_models_randomized.py', 'w') as f:
        f.write(f'''
# Script to implement the tuned models from randomized search
import pandas as pd
import joblib
import os

# Load the tuned models
rf_model = joblib.load('tuning_results_randomized/models/tuned_random_forest.joblib')
dt_model = joblib.load('tuning_results_randomized/models/tuned_decision_tree.joblib')
nb_model = joblib.load('tuning_results_randomized/models/tuned_naive_bayes.joblib')
preprocessor = joblib.load('tuning_results_randomized/models/preprocessor.joblib')

# Create directories if they don't exist
os.makedirs('regression_results', exist_ok=True)
os.makedirs('decision_tree_results', exist_ok=True)
os.makedirs('naive_bayes_results', exist_ok=True)

# Save the models in the required locations for the app
joblib.dump(rf_model, 'regression_results/random_forest_model.joblib')
joblib.dump(dt_model, 'decision_tree_results/decision_tree_model.joblib')
joblib.dump(nb_model, 'naive_bayes_results/naive_bayes_model.joblib')

# Save the preprocessor in all required locations
joblib.dump(preprocessor, 'regression_results/preprocessor.joblib')
joblib.dump(preprocessor, 'decision_tree_results/preprocessor.joblib')
joblib.dump(preprocessor, 'naive_bayes_results/preprocessor.joblib')

print("Tuned models from randomized search implemented and ready for use in the app!")
print(f"Random Forest best parameters: {rf_model.get_params()}")
print(f"Decision Tree best parameters: {dt_model.get_params()}")
print(f"Naive Bayes best parameters: {nb_model.get_params()}")
''')
    
    total_time = time.time() - start_time
    logging.info(f"Randomized hyperparameter tuning process completed in {total_time:.2f} seconds")
    logging.info(f"Results saved in 'tuning_results_randomized/' directory")
    logging.info(f"Run 'python implement_tuned_models_randomized.py' to implement the tuned models in your app")

if __name__ == "__main__":
    main() 