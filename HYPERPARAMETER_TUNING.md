# Hyperparameter Tuning for Video Game Sales Models

## Overview

This document outlines the hyperparameter tuning process used to optimize the machine learning models for our video game sales analysis application. Hyperparameter tuning is a critical step in machine learning model development that helps identify the optimal configuration for a model to achieve its best possible performance.

## Models Tuned

We performed systematic hyperparameter tuning on three models:

1. **Random Forest Regression** - For predicting exact sales figures
2. **Decision Tree Classification** - For classifying games into high/low sales categories
3. **Naive Bayes Classification** - As an additional classification approach

## Tuning Methodology

### GridSearchCV Approach

We used `GridSearchCV` from scikit-learn, which:
- Performs an exhaustive search over specified parameter values
- Uses cross-validation to evaluate model performance for each parameter combination
- Identifies the best parameter set based on a scoring metric

### Parameters Explored

#### Random Forest Regression
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
```

#### Decision Tree Classification
```python
param_grid = {
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}
```

#### Naive Bayes Classification
```python
param_grid = {
    'var_smoothing': np.logspace(0, -9, 10)  # Default is 1e-9
}
```

### Evaluation Metrics

- **Random Forest Regression**: R² score (coefficient of determination)
- **Decision Tree & Naive Bayes Classification**: Accuracy

## Implementation Details

The tuning process was implemented in `tune_models.py` with the following workflow:

1. **Data Preparation**:
   - Load cleaned game data
   - Create feature sets for both regression and classification tasks
   - Split data into training and testing sets
   - Scale numerical features

2. **Model Tuning**:
   - Define parameter grids for each model
   - Configure 3-fold cross-validation
   - Fit models on training data
   - Evaluate on test data
   - Record best parameters

3. **Documentation & Reporting**:
   - Generate detailed HTML report with results
   - Save visualizations of feature importance
   - Create logs of the tuning process

4. **Model Deployment**:
   - Save optimized models for use in the app
   - Create implementation script

## Results Summary

The detailed results are available in the auto-generated HTML report (`tuning_results/hyperparameter_tuning_report.html`), but key findings include:

### Random Forest Regression
- Best parameters found: [Generated during actual tuning]
- R² Score improved from ~0.95 to [value after tuning]
- MSE improved from ~0.07 to [value after tuning]
- Most important features: [Determined after tuning]

### Decision Tree Classification
- Best parameters found: [Generated during actual tuning]
- Accuracy improved from 98.0% to [value after tuning]
- Most important features: [Determined after tuning]

### Naive Bayes Classification
- Best var_smoothing value: [Generated during actual tuning]
- Accuracy improved from 84.4% to [value after tuning]

## Integration with the Application

After tuning:

1. The optimized models are saved to their respective directories:
   - `regression_results/random_forest_model.joblib`
   - `decision_tree_results/decision_tree_model.joblib`
   - `naive_bayes_results/naive_bayes_model.joblib`

2. The scalers and feature lists are also saved to ensure consistency between training and prediction.

3. The application automatically uses these optimized models when making predictions.

## Running the Tuning Process

To run the hyperparameter tuning process:

```bash
python tune_models.py
```

This will:
- Run all tuning processes (may take significant time)
- Generate detailed logs in `tuning_results/hyperparameter_tuning.log`
- Create visualizations in `tuning_results/plots/`
- Save optimized models in `tuning_results/models/`
- Generate a comprehensive HTML report

After tuning is complete, run:

```bash
python implement_tuned_models.py
```

This will copy the tuned models to the locations expected by the application.

## Computational Considerations

- The full grid search can be computationally intensive and time-consuming
- For quicker results, the parameter grid can be reduced or RandomizedSearchCV can be used instead
- If computational resources are limited, consider running the tuning on a more powerful machine or cloud service

## Future Improvements

- Expand parameter grids for more thorough exploration
- Test additional model types (XGBoost, LightGBM, etc.)
- Implement automated periodic retuning as new data becomes available
- Add more sophisticated feature selection during the tuning process 