#!/usr/bin/env python3
"""
Test runner for the Game Sales Analysis project.
Executes all tests and provides a summary of results.
"""

import pytest
import os
import sys
from termcolor import colored
import time

# Add parent directory to path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Run all test modules and report the results."""
    start_time = time.time()
    
    print(colored("\n===== GAME SALES ANALYSIS - TEST SUITE =====", "cyan", attrs=["bold"]))
    print(colored("\nRunning all tests...\n", "cyan"))
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of test modules to run (relative to the tests directory)
    test_modules = [
        'test_data_processing.py',
        'test_models.py',
        'test_app_utils.py',
        'test_streamlit_app.py'
    ]
    
    # Check if test files exist, if not display warning
    missing_files = [file for file in test_modules if not os.path.exists(os.path.join(current_dir, file))]
    if missing_files:
        for file in missing_files:
            print(colored(f"Warning: Test file {file} not found!", "yellow"))
        print("")
    
    # Run tests with pytest
    # Convert paths to absolute paths
    test_files = [os.path.join(current_dir, file) for file in test_modules if os.path.exists(os.path.join(current_dir, file))]
    exit_code = pytest.main(["-v"] + test_files)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(colored(f"\nTest suite completed in {elapsed_time:.2f} seconds", "cyan"))
    
    if exit_code == 0:
        print(colored("\n✓ All tests passed successfully!", "green", attrs=["bold"]))
    else:
        print(colored("\n✗ Some tests failed. Please check the output above for details.", "red", attrs=["bold"]))
    
    # Path to the root directory (parent of tests)
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Check for actual model and data files
    requirements = [
        (os.path.join(root_dir, 'vgchartz_cleaned.csv'), 'Dataset file'),
        (os.path.join(root_dir, 'regression_results/random_forest_model.joblib'), 'Random Forest model'),
        (os.path.join(root_dir, 'naive_bayes_results/naive_bayes_model.joblib'), 'Naive Bayes model'),
        (os.path.join(root_dir, 'decision_tree_results/decision_tree_model.joblib'), 'Decision Tree model'),
        (os.path.join(root_dir, 'regression_results/scaler.joblib'), 'Feature scaler')
    ]
    
    print(colored("\n----- Environment Check -----", "cyan"))
    missing_requirements = False
    for file_path, description in requirements:
        if os.path.exists(file_path):
            print(colored(f"✓ {description} present", "green"))
        else:
            print(colored(f"✗ {description} missing: {file_path}", "yellow"))
            missing_requirements = True
    
    if missing_requirements:
        print(colored("\nSome required files are missing. Run 'create_models.py' to generate model files.", "yellow"))
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 