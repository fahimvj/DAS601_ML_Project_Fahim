"""
Model training and evaluation utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                model, param_grid: Dict[str, Any] = None,
                cv: int = 5, scoring: str = 'f1') -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model using grid search cross-validation
    
    Args:
        X_train: Training features
        y_train: Training targets
        model: Base model to train
        param_grid: Dictionary of parameters for grid search
        cv: Number of cross-validation folds
        scoring: Scoring metric for grid search
    
    Returns:
        Tuple of (best_model, best_params)
    """
    print(f"Training {model.__class__.__name__} model...")
    
    if param_grid is not None:
        print(f"Performing grid search with {cv}-fold cross-validation...")
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, 
                                 n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {best_params}")
    else:
        print("Training with default parameters...")
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = {}
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"Cross-validation {scoring} scores: {cv_scores}")
        print(f"Mean {scoring} score: {cv_scores.mean():.4f}")
    
    return best_model, best_params


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating {model.__class__.__name__} model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    # Print metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    return metrics


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, 
                  output_path: str = None):
    """
    Plot ROC curve for a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        output_path: Path to save the plot (optional)
    """
    # Calculate probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    plt.show()


def save_model(model, model_name: str, output_dir: str = '../models/'):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model to save
        model_name: Name to give the saved model
        output_dir: Directory to save the model in
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def load_model(model_path: str):
    """
    Load a saved model from disk
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
