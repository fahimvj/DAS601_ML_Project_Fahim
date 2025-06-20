"""
Visualization utilities for exploratory data analysis and model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os


def plot_categorical_features(df: pd.DataFrame, target_col: str, 
                             cat_cols: List[str], fig_size: tuple = (15, 10),
                             output_dir: Optional[str] = None):
    """
    Plot count and percentage distribution of categorical features by target
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        cat_cols: List of categorical columns to plot
        fig_size: Size of the figure
        output_dir: Directory to save the plots (optional)
    """
    for col in cat_cols:
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        
        # Count plot
        sns.countplot(x=col, hue=target_col, data=df, ax=axes[0])
        axes[0].set_title(f'Count of {col} by {target_col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')
        
        # Percentage plot
        crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        crosstab.plot(kind='bar', ax=axes[1])
        axes[1].set_title(f'Percentage of {target_col} by {col}')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel(f'Percentage of {target_col}')
        axes[1].legend(title=target_col)
        
        plt.tight_layout()
        
        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{col}_by_{target_col}.png'), bbox_inches='tight')
        
        plt.show()


def plot_numerical_features(df: pd.DataFrame, target_col: str,
                           num_cols: List[str], fig_size: tuple = (15, 10),
                           output_dir: Optional[str] = None):
    """
    Plot distributions and box plots of numerical features by target
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        num_cols: List of numerical columns to plot
        fig_size: Size of the figure
        output_dir: Directory to save the plots (optional)
    """
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        
        # Distribution plot
        for target_value in df[target_col].unique():
            subset = df[df[target_col] == target_value]
            sns.kdeplot(subset[col], ax=axes[0], label=f'{target_col}={target_value}')
        
        axes[0].set_title(f'Distribution of {col} by {target_col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density')
        axes[0].legend(title=target_col)
        
        # Box plot
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[1])
        axes[1].set_title(f'Box plot of {col} by {target_col}')
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel(col)
        
        plt.tight_layout()
        
        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{col}_by_{target_col}.png'), bbox_inches='tight')
        
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, fig_size: tuple = (12, 10),
                            output_dir: Optional[str] = None):
    """
    Plot correlation heatmap for numerical features
    
    Args:
        df: DataFrame containing numerical data
        fig_size: Size of the figure
        output_dir: Directory to save the plot (optional)
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=fig_size)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", mask=mask, cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(model, feature_names: List[str], n_top: int = 20,
                           fig_size: tuple = (10, 8), output_dir: Optional[str] = None):
    """
    Plot feature importance for a trained model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        n_top: Number of top features to plot
        fig_size: Size of the figure
        output_dir: Directory to save the plot (optional)
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_top]
    
    # Plot feature importances
    plt.figure(figsize=fig_size)
    plt.bar(range(len(top_indices)), importances[top_indices])
    plt.xticks(range(len(top_indices)), [feature_names[i] for i in top_indices], rotation=90)
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
    
    plt.show()
