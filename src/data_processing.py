# Data Processing Functions

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath='../data/input/Telco_Customer_kaggle.csv'):
    """
    Load the Telco customer dataset
    
    Args:
        filepath (str): Path to the dataset
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
    print(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    return data

def clean_data(df):
    """
    Clean the dataset by handling missing values, converting data types,
    and preparing categorical features
    
    Args:
        df (pd.DataFrame): Original dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Check for missing values
    print(f"Missing values in dataset:\n{data.isnull().sum()}")
    
    # Handle 'TotalCharges' column (convert to numeric if it's not)
    if data['TotalCharges'].dtype == object:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Fill missing values with mean
        data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    
    # Convert 'Churn' to binary (1 for 'Yes', 0 for 'No')
    if 'Churn' in data.columns:
        data['Churn_Binary'] = (data['Churn'] == 'Yes').astype(int)
    
    # Convert 'SeniorCitizen' to string category if it's numeric binary
    if 'SeniorCitizen' in data.columns and data['SeniorCitizen'].dtype != object:
        data['SeniorCitizen'] = data['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    print("Data cleaning completed")
    return data

def feature_engineering(df):
    """
    Perform feature engineering by creating new features
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Create 'HasMultipleServices' feature
    if all(col in data.columns for col in ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup']):
        # Count services that are active (not 'No' or 'No internet service')
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Filter to only columns that exist in the dataframe
        services = [col for col in services if col in data.columns]
        
        # Count active services
        data['ServiceCount'] = data[services].apply(
            lambda x: sum(1 for item in x if item not in ['No', 'No internet service', 'No phone service']), 
            axis=1
        )
        
        data['HasMultipleServices'] = (data['ServiceCount'] > 1).astype(int)
    
    # Create tenure groups
    if 'tenure' in data.columns:
        # Create tenure groups
        bins = [0, 6, 12, 24, 36, 48, 60, float('inf')]
        labels = ['0-6 Months', '6-12 Months', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
        data['TenureGroup'] = pd.cut(data['tenure'], bins=bins, labels=labels, right=False)
    
    # Create monthly charge bins
    if 'MonthlyCharges' in data.columns:
        # Create monthly charge groups
        bins = [0, 30, 50, 70, 90, 110, float('inf')]
        labels = ['$0-30', '$30-50', '$50-70', '$70-90', '$90-110', '$110+']
        data['MonthlyChargesGroup'] = pd.cut(data['MonthlyCharges'], bins=bins, labels=labels, right=False)
    
    print("Feature engineering completed")
    return data

def preprocess_data_for_modeling(df, target_col='Churn_Binary'):
    """
    Preprocess data for machine learning models by:
    - Separating features and target
    - Identifying numeric and categorical features
    - Creating a preprocessing pipeline
    
    Args:
        df (pd.DataFrame): Dataframe with engineered features
        target_col (str): Name of target column
        
    Returns:
        tuple: (X, y, preprocessor) where X is features, y is target, and preprocessor is sklearn pipeline
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Drop any non-useful columns
    cols_to_drop = ['customerID', 'Churn', 'TenureGroup', 'MonthlyChargesGroup']
    # Only drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    
    if target_col not in data.columns:
        print(f"Warning: Target column '{target_col}' not found in dataframe.")
        return None, None, None
    
    # Extract target variable
    y = data[target_col]
    X = data.drop(cols_to_drop + [target_col], axis=1)
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    print(f"Data preprocessed: {len(X)} samples, {len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features")
    return X, y, preprocessor

def split_data(X, y, test_size=0.25, random_state=42):
    """
    Split data into training and test sets
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of data to use for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_path='../data/interim/'):
    """
    Save processed data to CSV files
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training targets
        y_test (pd.Series): Test targets
        output_path (str): Path to save the files
        
    Returns:
        None
    """
    import os
    
    # Ensure directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save data
    X_train.to_csv(f"{output_path}X_train.csv", index=False)
    X_test.to_csv(f"{output_path}X_test.csv", index=False)
    y_train.to_csv(f"{output_path}y_train.csv", index=False)
    y_test.to_csv(f"{output_path}y_test.csv", index=False)
    
    print(f"Processed data saved to {output_path}")
