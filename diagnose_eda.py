"""
Step-by-step EDA to diagnose issues
"""

import os
import sys

print("Starting step-by-step EDA diagnosis...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Step 1: Check if data file exists
data_path = 'data/input/Telco_Customer_kaggle.csv'
print(f"\nChecking data file: {data_path}")
if os.path.exists(data_path):
    print("  Data file exists!")
    print(f"  Full path: {os.path.abspath(data_path)}")
else:
    print(f"  ERROR: Data file not found at {data_path}")
    print(f"  Absolute path tried: {os.path.abspath(data_path)}")
    sys.exit(1)

# Step 2: Check if directories exist or can be created
print("\nChecking/creating output directory:")
output_dir = 'reports/figures/eda'
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory created: {output_dir}")
    print(f"  Full path: {os.path.abspath(output_dir)}")
except Exception as e:
    print(f"  ERROR creating directory: {e}")
    sys.exit(1)

# Step 3: Try importing packages
print("\nTrying to import required packages:")

packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
for package in packages:
    try:
        print(f"  Importing {package}...")
        __import__(package)
        print(f"    Successfully imported {package}")
    except ImportError as e:
        print(f"    Failed to import {package}: {e}")
        print("    This package may need to be installed.")

# Step 4: Try loading the data
print("\nTrying to load the data:")
try:
    import pandas as pd
    print("  Reading CSV file...")
    df = pd.read_csv(data_path)
    print(f"  Data loaded successfully! Shape: {df.shape}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    print(f"  First few rows:\n{df.head(3)}")
except Exception as e:
    print(f"  ERROR loading data: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnosis complete!")
