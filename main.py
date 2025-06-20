"""
Telco Customer Churn Prediction
-------------------------------
This is the main script for the Telco Customer Churn prediction project.
It loads the data, preprocesses it, trains a model, and evaluates its performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(filepath='Telco_Customer_kaggle.csv'):
    """Load and return the Telco Customer dataset"""
    print("Loading data...")
    data = pd.read_csv(filepath)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    return data

def preprocess_data(data):
    """Preprocess the data for model training"""
    print("Preprocessing data...")
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        print("Handling missing values...")
        data.fillna(data.mode().iloc[0], inplace=True)
    
    # Convert categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_processed = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Extract features and target
    if 'Churn' in data.columns:
        # Convert Churn to binary (1 for 'Yes', 0 for 'No')
        data['Churn_Binary'] = (data['Churn'] == 'Yes').astype(int)
        y = data['Churn_Binary']
        X = data_processed.drop(['Churn', 'Churn_Binary'], axis=1)
    else:
        print("Warning: 'Churn' column not found in dataset")
        X = data_processed
        y = None
        
    return X, y

def train_model(X, y):
    """Train a Random Forest model"""
    print("Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, (X_train, X_test, y_train, y_test)

def main():
    """Main function to run the entire pipeline"""
    print("Starting Telco Customer Churn Prediction project...")
    
    # Load data
    data = load_data()
    
    # Display basic info
    print("\nBasic information about the dataset:")
    print(data.info())
    print("\nSummary statistics:")
    print(data.describe())
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train and evaluate model
    if y is not None:
        model, scaler, (X_train, X_test, y_train, y_test) = train_model(X, y)
        print("\nModel training complete!")
    else:
        print("Cannot train model without target variable.")
    
    print("Pipeline execution completed!")

if __name__ == "__main__":
    main()
