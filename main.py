"""
Telco Customer Churn Prediction
-------------------------------
This is the main script for the Telco Customer Churn prediction project.
It loads the data, preprocesses it, trains a model, and evaluates its performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def main():
    """
    Main function to run the complete ML pipeline
    """
    print("Starting Telco Customer Churn Prediction Pipeline...")
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Data Loading
    print("\n--- Data Loading ---")
    df = load_data('data/input/Telco_Customer_kaggle.csv')
    print(f"Dataset shape: {df.shape}")
    
    # 2. Data Cleaning and Feature Engineering
    print("\n--- Data Preprocessing ---")
    df_clean = clean_data(df)
    df_features = feature_engineering(df_clean)
    
    # 3. Data Preprocessing for Modeling
    print("\n--- Preparing Data for Modeling ---")
    X, y, preprocessor = preprocess_data_for_modeling(df_features)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, output_path='data/interim/')
    
    # 4. Model Training
    print("\n--- Model Training ---")
    
    # Train multiple models
    models = {
        'logistic': train_logistic_regression(X_train, y_train),
        'random_forest': train_random_forest(X_train, y_train, n_estimators=200),
        'gradient_boosting': train_gradient_boosting(X_train, y_train, n_estimators=150)
    }
    
    # 5. Model Evaluation
    print("\n--- Model Evaluation ---")
    best_model_name = None
    best_model_score = 0
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model:")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save confusion matrix
        plot_confusion_matrix(model, X_test, y_test, 
                            save_path=f'reports/figures/{name}_confusion_matrix.png')
        
        # Track best model
        if metrics['roc_auc'] > best_model_score:
            best_model_score = metrics['roc_auc']
            best_model_name = name
    
    # 6. Feature Importance for best model
    if best_model_name in ['random_forest', 'gradient_boosting']:
        best_model = models[best_model_name]
        feature_names = X.columns
        plot_feature_importance(best_model, feature_names, 
                              save_path=f'reports/figures/{best_model_name}_feature_importance.png')
      # 7. Save best model
    best_model = models[best_model_name]
    save_model(best_model, f'{best_model_name}_model.pkl')
    print(f"\nBest model: {best_model_name} with ROC AUC: {best_model_score:.4f}")
    
    # 8. Visualizations
    print("\n--- Creating Visualizations ---")
    set_plotting_style()
    
    # Plot correlation heatmap
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    plot_correlation_heatmap(df_clean[numeric_cols], 
                          save_path='reports/figures/correlation_heatmap.png')
    
    # Plot churn rates for key features
    key_features = ['Contract', 'TenureGroup', 'MonthlyChargesGroup', 'PaymentMethod']
    for feature in key_features:
        if feature in df_features.columns:
            plot_churn_by_feature(df_features, feature, 
                               save_path=f'reports/figures/churn_by_{feature}.png')
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
