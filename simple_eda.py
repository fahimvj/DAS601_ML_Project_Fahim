"""
Simple EDA script for Telco Customer Churn dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_output_dir():
    """Create directories for storing EDA outputs"""
    output_dir = 'reports/figures/eda'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_eda():
    """Run a simple EDA and save visualizations"""
    print("Starting Simple EDA for Telco Customer Churn dataset...")
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Load the data
    data_path = 'data/input/Telco_Customer_kaggle.csv'
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Simple data cleaning
    print("Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Print basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    print("\nCreating visualizations:")
    
    # 1. Categorical variables
    print("- Categorical variables...")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Bar charts for categorical variables
    plt.figure(figsize=(20, 4*len(cat_cols)))
    for i, col in enumerate(cat_cols):
        plt.subplot(len(cat_cols), 2, 2*i+1)
        sns.countplot(x=col, data=df)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        
        # Add churn rate
        plt.subplot(len(cat_cols), 2, 2*i+2)
        sns.barplot(x=col, y='Churn', data=df)
        plt.title(f'Churn Rate by {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/categorical_variables.png')
    plt.close()
    print("  Saved: categorical_variables.png")
    
    # 2. Numerical variables
    print("- Numerical variables...")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Histograms and boxplots
    plt.figure(figsize=(15, 4*len(num_cols)))
    for i, col in enumerate(num_cols):
        # Histogram
        plt.subplot(len(num_cols), 2, 2*i+1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        
        # Box plot by churn
        plt.subplot(len(num_cols), 2, 2*i+2)
        sns.boxplot(x='Churn', y=col, data=df)
        plt.title(f'Boxplot of {col} by Churn')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/numerical_variables.png')
    plt.close()
    print("  Saved: numerical_variables.png")
    
    # 3. Correlation heatmap
    print("- Correlation heatmap...")
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='viridis')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    print("  Saved: correlation_heatmap.png")
    
    # 4. Scatter plots
    print("- Scatter plots...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, alpha=0.6)
    plt.title('Tenure vs Monthly Charges')
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=df, alpha=0.6)
    plt.title('Tenure vs Total Charges')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df, alpha=0.6)
    plt.title('Monthly Charges vs Total Charges')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_plots.png')
    plt.close()
    print("  Saved: scatter_plots.png")
    
    # 5. Important categorical vars and churn
    print("- Key categorical variables and churn...")
    key_vars = ['Contract', 'InternetService', 'PaymentMethod']
    plt.figure(figsize=(15, 15))
    for i, var in enumerate(key_vars):
        plt.subplot(3, 1, i+1)
        sns.countplot(x=var, hue='Churn', data=df)
        plt.title(f'{var} by Churn Status')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/key_categorical_vars.png')
    plt.close()
    print("  Saved: key_categorical_vars.png")
    
    # 6. Pair plot
    print("- Pair plot...")
    subset_df = df[num_cols + ['Churn']].sample(min(1000, len(df)))  # Sample to avoid large plot
    sns.pairplot(subset_df, hue='Churn')
    plt.savefig(f'{output_dir}/pair_plot.png')
    plt.close()
    print("  Saved: pair_plot.png")
    
    print("\nEDA completed successfully!")
    print(f"All visualizations saved to: {output_dir}")
    
    # Basic insights
    insights = """
# EDA Analysis Insights

## Categorical Variables:
- Contract type is strongly related to churn, with month-to-month contracts showing higher churn rates
- Customers with fiber optic internet service have higher churn rates
- Payment method shows correlation with churn, with electronic check users more likely to churn

## Numerical Variables:
- Lower tenure customers are more likely to churn
- Customers with higher monthly charges tend to churn more
- Total charges strongly correlate with tenure, as expected

## Relationships:
- Customers with high monthly charges and low tenure are at highest risk of churning
- Long-term contracts are effective at reducing churn
- Fiber optic service customers with high charges are a high-risk segment

## Recommendations:
- Target retention strategies at new customers on month-to-month contracts
- Review pricing of fiber optic service
- Encourage automatic payment methods instead of electronic checks
- Create bundle offers with longer contract terms to reduce churn
"""
    
    # Save insights
    with open(f'{output_dir}/../eda_insights.md', 'w') as f:
        f.write(insights)
        
    print(f"Basic insights saved to: {output_dir}/../eda_insights.md")
    
    return {
        'visualizations_dir': output_dir,
        'insights_file': f'{output_dir}/../eda_insights.md',
        'completed': True
    }

if __name__ == "__main__":
    run_eda()
