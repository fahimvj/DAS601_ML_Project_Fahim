"""
Exploratory Data Analysis for Telco Customer Churn Dataset

This module contains functions for comprehensive EDA of the telco dataset,
including visualization of categorical variables, numeric variables, and
their relationships.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict

# Import utility functions
from src.utils.data_utils import load_raw_data
from src.utils.visualization import plot_categorical_features

def create_output_dir():
    """Create directories for storing EDA outputs"""
    os.makedirs('reports/figures/eda', exist_ok=True)
    return 'reports/figures/eda'

def load_and_prepare_data() -> pd.DataFrame:
    """Load the dataset and prepare it for EDA"""
    df = load_raw_data()
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert Churn to binary (1/0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def get_data_summary(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Get summary statistics and information about the dataset"""
    # Basic info
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Summary statistics
    numeric_summary = df[numeric_cols].describe()
    categorical_summary = df[categorical_cols].describe()
    
    return info, numeric_summary, categorical_summary

def visualize_categorical_variables(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations for categorical variables
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the visualizations
    """
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Simple bar charts for each categorical variable
    plt.figure(figsize=(20, len(cat_cols)*4))
    
    for i, col in enumerate(cat_cols):
        plt.subplot(len(cat_cols), 2, 2*i+1)
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        
        # Add churn percentage
        plt.subplot(len(cat_cols), 2, 2*i+2)
        churn_pct = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
        sns.barplot(x=churn_pct.index, y=churn_pct.values, palette='viridis')
        plt.title(f'Churn Rate by {col}')
        plt.xticks(rotation=45)
        plt.ylabel('Churn Rate')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/categorical_distributions.png')
    plt.close()
    
    # 2. Grouped bar chart for selected variables
    key_vars = ['Contract', 'InternetService', 'PaymentMethod']
    if all(var in cat_cols for var in key_vars):
        plt.figure(figsize=(15, 10))
        for i, var in enumerate(key_vars):
            plt.subplot(1, 3, i+1)
            df_count = df.groupby([var, 'Churn']).size().unstack()
            df_count.plot(kind='bar', stacked=False, ax=plt.gca())
            plt.title(f'{var} by Churn Status')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/grouped_bar_chart.png')
        plt.close()
    
    # 3. Stacked bar charts
    plt.figure(figsize=(18, 10))
    for i, var in enumerate(['Contract', 'InternetService']):
        if var in cat_cols:
            plt.subplot(1, 2, i+1)
            df_pct = df.groupby([var, 'Churn']).size().unstack()
            df_pct_norm = df_pct.div(df_pct.sum(axis=1), axis=0)
            df_pct_norm.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(f'Proportion of Churn by {var}')
            plt.xticks(rotation=45)
            plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stacked_bar_charts.png')
    plt.close()
    
    # 4. Horizontal bar charts for selected variables with many categories
    multi_cat_vars = [col for col in cat_cols if df[col].nunique() > 5]
    if multi_cat_vars:
        plt.figure(figsize=(12, len(multi_cat_vars)*3))
        for i, col in enumerate(multi_cat_vars):
            plt.subplot(len(multi_cat_vars), 1, i+1)
            df_counts = df[col].value_counts()
            df_counts.sort_values().plot(kind='barh')
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/horizontal_bar_charts.png')
        plt.close()

def visualize_numeric_variables(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations for numeric variables
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the visualizations
    """
    # Identify numeric columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 1. Histograms for each numeric variable
    plt.figure(figsize=(16, len(num_cols)*4))
    for i, col in enumerate(num_cols):
        plt.subplot(len(num_cols), 2, 2*i+1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        
        # Add boxplot
        plt.subplot(len(num_cols), 2, 2*i+2)
        sns.boxplot(y=df[col].dropna())
        plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/numeric_distributions.png')
    plt.close()
    
    # 2. KDE plots by churn status
    plt.figure(figsize=(16, len(num_cols)*4))
    for i, col in enumerate(num_cols):
        plt.subplot(len(num_cols), 1, i+1)
        sns.kdeplot(data=df, x=col, hue='Churn', common_norm=False, fill=True)
        plt.title(f'Density Plot of {col} by Churn Status')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kde_by_churn.png')
    plt.close()
    
    # 3. Boxplots grouped by churn status
    plt.figure(figsize=(16, len(num_cols)*4))
    for i, col in enumerate(num_cols):
        plt.subplot(len(num_cols), 1, i+1)
        sns.boxplot(data=df, y=col, x='Churn')
        plt.title(f'Boxplot of {col} by Churn Status')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplots_by_churn.png')
    plt.close()

def visualize_relationships(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations showing relationships between variables
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the visualizations
    """
    # 1. Scatter plots between numeric variables
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(num_cols) >= 2:
        plt.figure(figsize=(15, 12))
        for i in range(min(3, len(num_cols))):
            for j in range(i+1, min(4, len(num_cols))):
                plt.subplot(2, 2, (i*len(num_cols[i+1:min(4, len(num_cols))])) + (j-i))
                sns.scatterplot(data=df, x=num_cols[i], y=num_cols[j], hue='Churn', alpha=0.6)
                plt.title(f'Scatter Plot: {num_cols[i]} vs {num_cols[j]}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scatter_plots.png')
        plt.close()
    
    # 2. Regression plots for key relationships
    key_num_pairs = [('tenure', 'MonthlyCharges'), ('tenure', 'TotalCharges'), ('MonthlyCharges', 'TotalCharges')]
    key_pairs = [pair for pair in key_num_pairs if pair[0] in num_cols and pair[1] in num_cols]
    
    if key_pairs:
        plt.figure(figsize=(18, 6*len(key_pairs)))
        for i, (x_var, y_var) in enumerate(key_pairs):
            plt.subplot(len(key_pairs), 2, 2*i+1)
            sns.regplot(data=df, x=x_var, y=y_var, scatter_kws={'alpha':0.5})
            plt.title(f'Regression Plot: {x_var} vs {y_var}')
            
            plt.subplot(len(key_pairs), 2, 2*i+2)
            sns.lmplot(data=df, x=x_var, y=y_var, hue='Churn', height=5, aspect=1.5)
            plt.title(f'Regression by Churn: {x_var} vs {y_var}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression_plots.png')
        plt.close()
    
    # 3. Bubble plot (scatter plot with size representing a third variable)
    if all(var in num_cols for var in ['tenure', 'MonthlyCharges', 'TotalCharges']):
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', 
                       size='TotalCharges', hue='Churn', sizes=(20, 200), alpha=0.7)
        plt.title('Bubble Plot: tenure vs MonthlyCharges (size=TotalCharges)')
        plt.savefig(f'{output_dir}/bubble_plot.png')
        plt.close()
    
    # 4. Pair plot for numeric variables
    if len(num_cols) > 1:
        pair_cols = num_cols
        if len(pair_cols) > 4:  # Limit to avoid too large plots
            pair_cols = pair_cols[:4]
        pair_cols.append('Churn')  # Add target variable
        
        g = sns.pairplot(df[pair_cols], hue='Churn', diag_kind='kde', 
                        plot_kws={'alpha':0.6}, height=3)
        g.fig.suptitle('Pair Plot of Numeric Variables', y=1.02)
        g.savefig(f'{output_dir}/pair_plot.png')
        plt.close()

def create_correlation_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create correlation heatmaps
    
    Args:
        df: DataFrame containing the data
        output_dir: Directory to save the visualizations
    """
    # 1. Correlation heatmap for numeric variables
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='viridis', 
                   fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap for Numeric Variables')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
        plt.close()
    
    # 2. Heatmap for categorical variables vs churn
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        plt.figure(figsize=(14, 12))
        # Create a crosstab of churn percentage for each categorical variable
        all_crosstabs = {}
        for col in cat_cols[:10]:  # Limit to top 10 to avoid too large plots
            crosstab = pd.crosstab(df[col], df['Churn'])
            all_crosstabs[col] = crosstab[1] / (crosstab[0] + crosstab[1])
        
        churn_by_cat = pd.DataFrame(all_crosstabs)
        sns.heatmap(churn_by_cat, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
        plt.title('Churn Rate by Categorical Variables')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/category_churn_heatmap.png')
        plt.close()
    
    # 3. Two categorical variables and one numeric variable
    if len(cat_cols) >= 2 and num_cols:
        key_cat_vars = ['Contract', 'InternetService']
        key_cat_vars = [var for var in key_cat_vars if var in cat_cols]
        
        if len(key_cat_vars) >= 2:
            plt.figure(figsize=(12, 10))
            pivot_data = df.pivot_table(
                values=num_cols[0], 
                index=key_cat_vars[0], 
                columns=key_cat_vars[1], 
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
            plt.title(f'Average {num_cols[0]} by {key_cat_vars[0]} and {key_cat_vars[1]}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cat_cat_num_heatmap.png')
            plt.close()

def run_complete_eda() -> Dict:
    """Run a complete EDA and return a dictionary of insights"""
    print("Starting Exploratory Data Analysis...")
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get data summary
    info, numeric_summary, categorical_summary = get_data_summary(df)
    
    # Run visualizations
    visualize_categorical_variables(df, output_dir)
    visualize_numeric_variables(df, output_dir)
    visualize_relationships(df, output_dir)
    create_correlation_heatmaps(df, output_dir)
    
    print(f"EDA completed. Visualizations saved to {output_dir}")
    
    # Summarize insights
    insights = {
        'data_overview': {
            'rows': info['rows'],
            'columns': info['columns'],
            'missing_values': info['missing_values'],
            'duplicate_rows': info['duplicate_rows']
        },
        'visualizations_created': [
            'categorical_distributions.png',
            'grouped_bar_chart.png',
            'stacked_bar_charts.png',
            'horizontal_bar_charts.png',
            'numeric_distributions.png',
            'kde_by_churn.png',
            'boxplots_by_churn.png',
            'scatter_plots.png',
            'regression_plots.png',
            'bubble_plot.png',
            'pair_plot.png',
            'correlation_heatmap.png',
            'category_churn_heatmap.png',
            'cat_cat_num_heatmap.png'
        ],
        'paths': {
            'output_directory': output_dir
        }
    }
    
    return insights

if __name__ == "__main__":
    run_complete_eda()
