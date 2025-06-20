# Model Training Functions

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Train a Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for Logistic Regression
        
    Returns:
        Trained model
    """
    print("Training Logistic Regression model...")
    
    # Default parameters if not specified
    params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    }
    
    # Update with any provided parameters
    params.update(kwargs)
    
    # Train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    print("Logistic Regression model trained")
    return model

def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for Random Forest
        
    Returns:
        Trained model
    """
    print("Training Random Forest model...")
    
    # Default parameters if not specified
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42
    }
    
    # Update with any provided parameters
    params.update(kwargs)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Random Forest model trained")
    return model

def train_gradient_boosting(X_train, y_train, **kwargs):
    """
    Train a Gradient Boosting model
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for Gradient Boosting
        
    Returns:
        Trained model
    """
    print("Training Gradient Boosting model...")
    
    # Default parameters if not specified
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    
    # Update with any provided parameters
    params.update(kwargs)
    
    # Train model
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Gradient Boosting model trained")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics

def plot_confusion_matrix(model, X_test, y_test, save_path=None):
    """
    Plot confusion matrix for model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        save_path: Path to save the plot
        
    Returns:
        None
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Set up plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None, max_features=20):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        save_path: Path to save the plot
        max_features: Maximum number of features to display
        
    Returns:
        None
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importance. Skipping feature importance plot.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance and take top max_features
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    feature_importance = feature_importance.head(max_features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top {max_features} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def save_model(model, filename, model_dir='../models/'):
    """
    Save trained model to file
    
    Args:
        model: Trained model
        filename: Name of file to save model to
        model_dir: Directory to save model in
        
    Returns:
        str: Path to saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Full path to model file
    model_path = os.path.join(model_dir, filename)
    
    # Save model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """
    Load trained model from file
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
    """
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Load model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    print(f"Model loaded from {model_path}")
    return model
