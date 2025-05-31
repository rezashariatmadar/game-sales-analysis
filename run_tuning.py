import argparse
import os
import sys
import time

def main():
    """Main function to run the hyperparameter tuning process with chosen method."""
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for game sales models')
    parser.add_argument('--method', choices=['grid', 'randomized'], default='randomized',
                       help='Tuning method: "grid" for GridSearchCV (exhaustive, slow), '
                            '"randomized" for RandomizedSearchCV (faster)')
    parser.add_argument('--implement', action='store_true',
                       help='Automatically implement the tuned models after tuning')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Print header
    print("\n" + "="*80)
    print(f"Running {'Grid Search' if args.method == 'grid' else 'Randomized Search'} "
          f"hyperparameter tuning for video game sales models")
    print("="*80 + "\n")
    
    # Check if required files exist
    if not os.path.exists("tune_models.py") or not os.path.exists("tune_models_randomized.py"):
        print("Error: Required tuning script files not found!")
        print("Make sure both 'tune_models.py' and 'tune_models_randomized.py' exist in the current directory.")
        sys.exit(1)
    
    if not os.path.exists("vgchartz_cleaned.csv"):
        print("Error: Dataset file 'vgchartz_cleaned.csv' not found!")
        print("Please ensure the cleaned dataset exists in the current directory.")
        sys.exit(1)
    
    # Run the chosen tuning method
    try:
        if args.method == 'grid':
            print("Starting Grid Search tuning (this may take a long time)...")
            os.system("python tune_models.py")
            results_dir = "tuning_results"
            implement_script = "implement_tuned_models.py"
        else:
            print("Starting Randomized Search tuning...")
            os.system("python tune_models_randomized.py")
            results_dir = "tuning_results_randomized"
            implement_script = "implement_tuned_models_randomized.py"
            
        # Check if tuning completed successfully
        if not os.path.exists(f"{results_dir}/models/tuned_random_forest.joblib"):
            print(f"Error: Tuning did not complete successfully. Check the logs in {results_dir}/hyperparameter_tuning.log")
            sys.exit(1)
            
        print(f"\nTuning completed successfully! Results available in {results_dir}/")
        print(f"- HTML Report: {results_dir}/hyperparameter_tuning_report.html")
        print(f"- Log File: {results_dir}/hyperparameter_tuning.log")
        print(f"- Tuned Models: {results_dir}/models/")
        print(f"- Feature Importance Plots: {results_dir}/plots/")
        
        # Implement the models if requested
        if args.implement:
            print("\nImplementing tuned models...")
            os.system(f"python {implement_script}")
            print("Models implemented successfully! They are now ready for use in the app.")
            
        elapsed_time = time.time() - start_time
        print(f"\nTotal process completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"Error during tuning: {str(e)}")
        sys.exit(1)
        
    print("\n" + "="*80)
    print("Hyperparameter tuning process completed")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 