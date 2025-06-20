"""
Simple script to verify all imports are working properly
"""

def test_imports():
    print("Testing imports...")
    try:
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix

        # Import project modules
        from src.data_processing import (
            load_data, clean_data, feature_engineering, 
            preprocess_data_for_modeling, split_data, save_processed_data
        )
        from src.model_training import (
            train_logistic_regression, train_random_forest, train_gradient_boosting,
            evaluate_model, plot_confusion_matrix, plot_feature_importance, save_model
        )
        from src.visualization import (
            set_plotting_style, plot_correlation_heatmap, 
            plot_churn_by_feature
        )
        
        print("All imports successful!")
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
    
    # Test data loading
    try:
        from src.data_processing import load_data
        print("\nTesting data loading...")
        df = load_data('data/input/Telco_Customer_kaggle.csv')
        print(f"Data loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Data loading error: {e}")
