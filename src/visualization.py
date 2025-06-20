# Visualization Functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_plotting_style():
    """
    Set consistent plotting style for all visualizations
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
    print("Plotting style configured")

def plot_categorical_distribution(df, column, hue=None, figsize=(10, 6), save_path=None):
    """
    Plot the distribution of a categorical variable
    
    Args:
        df: Dataframe with data
        column: Column name to plot
        hue: Optional column to use for grouping/coloring
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    if hue:
        # Count the data with grouping
        counts = df.groupby([column, hue]).size().reset_index(name='count')
        # Create grouped bar chart
        ax = sns.barplot(x=column, y='count', hue=hue, data=counts)
        # Add percentages
        total = counts['count'].sum()
        for p in ax.patches:
            percentage = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
    else:
        # Count the data
        counts = df[column].value_counts()
        # Create bar chart
        ax = sns.barplot(x=counts.index, y=counts.values)
        # Add percentages
        total = counts.sum()
        for p in ax.patches:
            percentage = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)
    
    # Set labels
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_numerical_distribution(df, column, hue=None, figsize=(10, 6), kde=True, save_path=None):
    """
    Plot the distribution of a numerical variable
    
    Args:
        df: Dataframe with data
        column: Column name to plot
        hue: Optional column to use for grouping/coloring
        figsize: Figure size as tuple (width, height)
        kde: Whether to plot kernel density estimate
        save_path: Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    if hue:
        # Create histogram with grouping
        sns.histplot(data=df, x=column, hue=hue, kde=kde, element="step", common_norm=False, alpha=0.6)
    else:
        # Create histogram
        sns.histplot(data=df, x=column, kde=kde, color='darkblue', alpha=0.6)
        
        # Add vertical line for mean and median
        plt.axvline(df[column].mean(), color='red', linestyle='--', label=f'Mean: {df[column].mean():.2f}')
        plt.axvline(df[column].median(), color='green', linestyle='-.', label=f'Median: {df[column].median():.2f}')
        plt.legend()
    
    # Set labels
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_correlation_heatmap(df, cols=None, figsize=(12, 10), save_path=None):
    """
    Plot correlation heatmap for numerical variables
    
    Args:
        df: Dataframe with data
        cols: List of columns to include (default: all numeric)
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the plot
        
    Returns:
        None
    """
    # Select columns
    if cols is None:
        df_numeric = df.select_dtypes(include=['float64', 'int64'])
    else:
        df_numeric = df[cols]
    
    # Calculate correlation matrix
    corr = df_numeric.corr()
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
                square=True, linewidths=.5, cbar_kws={'shrink': .7})
    
    # Set labels
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_churn_by_feature(df, feature, figsize=(10, 6), save_path=None):
    """
    Plot churn rate by a specific feature
    
    Args:
        df: Dataframe with data
        feature: Feature to group by
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    # Calculate churn rate by feature
    churn_rate = df.groupby(feature)['Churn_Binary'].mean().reset_index()
    churn_rate['Churn_Percentage'] = churn_rate['Churn_Binary'] * 100
    
    # Sort by churn rate
    churn_rate = churn_rate.sort_values('Churn_Percentage')
    
    # Calculate counts for sizing the importance
    counts = df[feature].value_counts().reset_index()
    counts.columns = [feature, 'Count']
    
    # Merge churn rate and counts
    merged = pd.merge(churn_rate, counts, on=feature)
    
    # Create bar chart
    ax = sns.barplot(x=feature, y='Churn_Percentage', data=merged)
    
    # Add counts as text
    for i, row in merged.iterrows():
        ax.text(i, 1, f"n={row['Count']}", ha='center', va='bottom')
    
    # Set labels
    plt.title(f'Churn Rate by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=45)
    
    # Add overall average line
    overall_avg = df['Churn_Binary'].mean() * 100
    plt.axhline(y=overall_avg, color='r', linestyle='-', label=f'Overall Avg: {overall_avg:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def create_grouped_bar_chart(df, x, y, hue, figsize=(12, 6), save_path=None):
    """
    Create a grouped bar chart
    
    Args:
        df: Dataframe with data
        x: Column for x-axis
        y: Column for y-axis
        hue: Column for grouping
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    ax = sns.barplot(data=df, x=x, y=y, hue=hue)
    
    # Set labels
    plt.title(f'{y} by {x} and {hue}')
    plt.xlabel(x)
    plt.ylabel(y)
    
    # Rotate x-axis labels if there are many categories
    if df[x].nunique() > 5:
        plt.xticks(rotation=45)
    
    plt.legend(title=hue)
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_feature_distribution(df, feature, target='Churn_Binary', figsize=(10, 6), save_path=None):
    """
    Plot the distribution of a feature by target variable
    
    Args:
        df: Dataframe with data
        feature: Feature to plot
        target: Target variable column name
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    # Check if the feature is numeric or categorical
    if df[feature].dtype in ['int64', 'float64']:
        # Numeric feature - use histogram
        sns.histplot(data=df, x=feature, hue=target, element="step", common_norm=False, alpha=0.6)
    else:
        # Categorical feature - use count plot
        counts = df.groupby([feature, target]).size().reset_index(name='count')
        sns.barplot(x=feature, y='count', hue=target, data=counts)
    
    # Set labels
    plt.title(f'Distribution of {feature} by {target}')
    plt.xlabel(feature)
    
    # Rotate x-axis labels for categorical features
    if df[feature].dtype not in ['int64', 'float64'] or df[feature].nunique() > 5:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def save_all_plots(df, save_dir='../reports/figures/'):
    """
    Save all exploratory plots for the dataset
    
    Args:
        df: Dataframe with data
        save_dir: Directory to save plots
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set plotting style
    set_plotting_style()
    
    print(f"Saving all plots to {save_dir}...")
    
    # Calculate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Plot numeric distributions
    for col in numeric_cols:
        if col != 'Churn_Binary':  # Skip target
            file_path = os.path.join(save_dir, f'{col}_distribution.png')
            plot_numerical_distribution(df, col, hue='Churn', save_path=file_path)
    
    # Plot categorical distributions
    for col in categorical_cols:
        if col != 'Churn':  # Skip target
            file_path = os.path.join(save_dir, f'{col}_distribution.png')
            plot_categorical_distribution(df, col, hue='Churn', save_path=file_path)
    
    # Plot correlation heatmap for numeric features
    file_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plot_correlation_heatmap(df, save_path=file_path)
    
    # Plot churn rate by important features
    important_features = ['Contract', 'TenureGroup', 'PaperlessBilling', 'PaymentMethod']
    for feature in important_features:
        if feature in df.columns:
            file_path = os.path.join(save_dir, f'churn_by_{feature}.png')
            plot_churn_by_feature(df, feature, save_path=file_path)
    
    print(f"All plots saved to {save_dir}")
