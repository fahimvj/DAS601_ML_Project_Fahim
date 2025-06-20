"""
Data loading and processing utilities
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def load_raw_data(filepath: str = 'data/input/Telco_Customer_kaggle.csv') -> pd.DataFrame:
    """
    Load the raw telco customer dataset
    
    Args:
        filepath: Path to the raw data file
    
    Returns:
        DataFrame containing the raw data
    """
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the telco customer dataset by handling missing values and data types
    
    Args:
        df: Raw dataframe
    
    Returns:
        Cleaned dataframe
    """
    print("Cleaning data...")
    df_cleaned = df.copy()
    
    # Check for missing values
    missing_values = df_cleaned.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values")
        # Fill missing values appropriately
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                else:
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # Convert TotalCharges to numeric (if it's not already)
    if 'TotalCharges' in df_cleaned.columns and df_cleaned['TotalCharges'].dtype == 'object':
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
        df_cleaned['TotalCharges'].fillna(df_cleaned['TotalCharges'].median(), inplace=True)
    
    # Convert categorical Yes/No columns to binary
    binary_cols = ['Churn', 'PhoneService', 'PaperlessBilling', 'Partner', 'Dependents']
    for col in binary_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].map({'Yes': 1, 'No': 0})
    
    print("Data cleaning completed")
    return df_cleaned


def split_data(df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets
    
    Args:
        df: Cleaned dataframe
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    print(f"Splitting data with test_size={test_size}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series, 
                        output_dir: str = '../data/interim/'):
    """
    Save the processed datasets to disk
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        output_dir: Directory to save the processed data
    """
    print(f"Saving processed data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print("Processed data saved successfully")
