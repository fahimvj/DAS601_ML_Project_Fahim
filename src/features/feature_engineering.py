"""
Feature engineering for the Telco Customer Churn dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def identify_features(df: pd.DataFrame) -> Tuple[list, list, list]:
    """
    Identify categorical and numerical features in the dataset
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Tuple of (categorical_cols, numerical_cols, binary_cols)
    """
    # Exclude target or ID columns
    exclude_cols = ['customerID', 'Churn']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    binary_cols = []
    
    for col in feature_cols:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            if df[col].nunique() <= 2:
                binary_cols.append(col)
            else:
                numerical_cols.append(col)
    
    print(f"Identified {len(categorical_cols)} categorical features, {len(numerical_cols)} numerical features, and {len(binary_cols)} binary features")
    
    return categorical_cols, numerical_cols, binary_cols


def create_feature_pipeline(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Create a scikit-learn preprocessing pipeline for categorical and numerical features
    
    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        
    Returns:
        ColumnTransformer preprocessing pipeline
    """
    # Create preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with new features
    """
    print("Engineering additional features...")
    df_new = df.copy()
    
    # Create new features
    
    # Feature 1: Tenure group
    df_new['tenure_group'] = pd.qcut(df_new['tenure'], q=4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
    
    # Feature 2: Total services subscribed
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Count services where value is 'Yes' or not 'No'
    df_new['total_services'] = 0
    for col in service_cols:
        if col in df_new.columns:
            if df_new[col].dtype == 'object':
                df_new['total_services'] += (df_new[col] != 'No').astype(int)
            else:
                df_new['total_services'] += df_new[col]
    
    # Feature 3: Average monthly charges per service
    if 'MonthlyCharges' in df_new.columns:
        df_new['avg_charge_per_service'] = df_new['MonthlyCharges'] / (df_new['total_services'] + 1)  # Add 1 to avoid division by zero
    
    # Feature 4: Seniority vs. contract type
    if 'tenure' in df_new.columns and 'Contract' in df_new.columns:
        # Create contract length value
        contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df_new['contract_length'] = df_new['Contract'].map(contract_mapping)
        
        # Calculate tenure to contract ratio
        df_new['tenure_to_contract_ratio'] = df_new['tenure'] / df_new['contract_length']
    
    print(f"Created {len(df_new.columns) - len(df.columns)} new features")
    return df_new
