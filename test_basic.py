"""
Simple script to test basic imports and data loading
"""

print("Starting basic test...")

# Test basic imports
try:
    print("Importing basic libraries...")
    import os
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("Basic imports successful")
except Exception as e:
    print(f"Basic import error: {e}")

# Test data science libraries
try:
    print("\nImporting data science libraries...")
    import pandas
    import numpy
    import matplotlib
    import seaborn
    import sklearn
    print("Data science imports successful")
    print(f"pandas version: {pandas.__version__}")
    print(f"numpy version: {numpy.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"Data science import error: {e}")

print("\nTest completed.")
