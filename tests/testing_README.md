# Game Sales Analysis - Test Suite

This directory contains the comprehensive test suite for the Game Sales Analysis project. The tests ensure that data processing, models, app utilities, and the Streamlit application function correctly.

## Running Tests

To run all tests at once, use the run_tests.py script:

```bash
python tests/run_tests.py
```

To run individual test modules:

```bash
# From the project root directory:
pytest -v tests/test_data_processing.py
pytest -v tests/test_models.py
pytest -v tests/test_app_utils.py
pytest -v tests/test_streamlit_app.py
```

## Test Files

The test suite consists of the following files:

1. **test_data_processing.py**: Tests for data loading, cleaning, and feature engineering
2. **test_models.py**: Tests for model loading, prediction, and evaluation
3. **test_app_utils.py**: Tests for helper functions used in the app
4. **test_streamlit_app.py**: Tests for the Streamlit application interface
5. **run_tests.py**: Script to run all tests and report results
6. **test_requirements.txt**: Requirements for running tests

## Test Coverage

### Data Processing Tests

- **File Existence**: Verify that the dataset exists and is readable
- **Column Validation**: Check that required columns exist in the dataset
- **Data Integrity**: Test for duplicate records and valid ranges of values
- **Standardization**: Test that feature standardization works correctly
- **Feature Engineering**: Test the creation of new features like sales ratios and game age
- **Missing Data Handling**: Test strategies for handling missing values
- **Release Year Validation**: Ensure release years are within a reasonable range (1950-2024)
- **Sales Value Validation**: Ensure sales values are non-negative

### Model Tests

- **Model File Existence**: Verify that model files exist in the expected locations
- **Model Loading**: Test that models can be loaded correctly
- **Model Performance**: Verify that models achieve reasonable performance metrics
- **Feature Importance**: Test extraction of feature importance
- **Prediction Workflow**: Test the end-to-end prediction process
- **Outlier Handling**: Test model robustness to outlier values
- **Missing Value Handling**: Test model predictions with NaN values in input features
- **Classification Performance**: Test accuracy of binary classification models

### App Utility Tests

- **Data Filtering**: Test filtering by year, genre, platform, and publisher
- **Regional Distribution**: Test calculation of regional sales distribution
- **Sales Ratio Calculation**: Test calculation of regional sales ratios
- **Model Prediction**: Test the prediction workflow using temporary models
- **Input Validation**: Test handling of invalid inputs (missing values, strings, out-of-range values)
- **Error Handling**: Test error detection and handling strategies

### Streamlit App Tests

- **Data Loading**: Test loading of data in the app
- **Error Handling**: Test error handling for missing files and invalid data
- **Platform/Console Handling**: Test handling of platform/console column naming differences
- **Model Loading**: Test loading of models in the app
- **Prediction Workflow**: Test the end-to-end prediction workflow
- **NaN Value Handling**: Test how the app handles missing values in prediction inputs
- **Environment Validation**: Test checking for required files
- **Input Validation**: Test handling of unexpected user inputs

## Test Design Principles

1. **Isolation**: Each test is independent and doesn't rely on the state from other tests
2. **Completeness**: Tests cover both expected and unexpected inputs
3. **Robustness**: Tests include error handling for edge cases
4. **Clarity**: Test names and descriptions clearly indicate what's being tested
5. **Maintainability**: Tests are structured to be easy to update as the project evolves

## Dependencies

The test suite uses the following key dependencies:

- pytest: Testing framework
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning utilities
- unittest.mock: Mocking external dependencies

Install dependencies using:

```bash
pip install -r tests/test_requirements.txt
```

## Testing Environment

The tests are designed to run in any environment with Python 3.6+ and the required dependencies. No special configuration is needed beyond installing the required packages.

## Extending the Test Suite

When adding new functionality to the project, please also add corresponding tests to maintain code quality and prevent regressions. Follow these guidelines:

1. Place new tests in the appropriate test file based on what they're testing
2. Use descriptive test names that clearly indicate what's being tested
3. Add proper assertions to verify expected behavior
4. Include tests for both normal operation and error handling
5. Keep tests fast and independent of each other

## Common Issues and Solutions

- **FileNotFoundError**: Ensure all required data and model files exist in the expected locations
- **Import Errors**: Make sure the tests can access the main code files (handled by sys.path.insert)
- **Test Dependencies**: Some tests depend on scikit-learn models - make sure all requirements are installed 