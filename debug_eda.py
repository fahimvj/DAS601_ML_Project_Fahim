"""
Debug script for EDA analysis
"""

import os
import traceback
import sys

def main():
    try:
        print("Starting EDA analysis in debug mode...")
        print(f"Current working directory: {os.getcwd()}")
        
        # First, check if the data file exists
        data_path = 'data/input/Telco_Customer_kaggle.csv'
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            sys.exit(1)
        else:
            print(f"Data file found: {data_path}")
        
        # Import the modules one by one to identify any import errors
        print("\nImporting modules:")
        
        try:
            print("- Importing EDA module...")
            from src.eda import run_complete_eda
            print("  Successfully imported run_complete_eda")
        except Exception as e:
            print(f"  ERROR importing run_complete_eda: {e}")
            traceback.print_exc()
        
        try:
            print("- Importing EDA report module...")
            from src.eda_report import generate_eda_report
            print("  Successfully imported generate_eda_report")
        except Exception as e:
            print(f"  ERROR importing generate_eda_report: {e}")
            traceback.print_exc()
        
        # Try to run the EDA process with detailed error handling
        try:
            print("\nRunning complete EDA...")
            from src.eda import run_complete_eda
            insights = run_complete_eda()
            print("EDA completed successfully!")
            return insights
        except Exception as e:
            print(f"ERROR in run_complete_eda: {e}")
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    insights = main()
    
    if insights:
        try:
            print("\nGenerating EDA report...")
            from src.eda_report import generate_eda_report
            report_path = generate_eda_report(insights)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"ERROR in generate_eda_report: {e}")
            traceback.print_exc()
    else:
        print("Cannot generate report due to errors in EDA process")
