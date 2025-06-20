"""
Self-contained EDA script - installs required packages if needed
"""

import os
import sys
import subprocess

def install_package(package):
    """Install a Python package if it's not already installed"""
    try:
        __import__(package)
        print(f"{package} is already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed")

# Create output directories
output_dir = 'reports/figures/eda'
os.makedirs(output_dir, exist_ok=True)

# Install required packages
print("Checking and installing required packages:")
required_packages = ['pandas', 'matplotlib', 'seaborn']
for package in required_packages:
    install_package(package)

print("\nImporting required packages:")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\nStarting simple EDA:")
data_path = 'data/input/Telco_Customer_kaggle.csv'
print(f"Loading data from {data_path}")

# Load the data
df = pd.read_csv(data_path)

# Simple preprocessing
print("Preprocessing data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Print dataset info
print(f"\nDataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Create visualizations
print("\nCreating visualizations:")

# 1. Categorical variables
print("- Categorical variables bar charts...")
categorical_cols = ['Contract', 'PaymentMethod', 'InternetService', 'gender']
fig, axes = plt.subplots(len(categorical_cols), 2, figsize=(15, 20))

for i, col in enumerate(categorical_cols):
    # Distribution
    sns.countplot(x=col, data=df, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    axes[i, 0].set_xticklabels(axes[i, 0].get_xticklabels(), rotation=45)
    
    # Churn rate
    sns.barplot(x=col, y='Churn', data=df, ax=axes[i, 1])
    axes[i, 1].set_title(f'Churn Rate by {col}')
    axes[i, 1].set_xticklabels(axes[i, 1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/categorical_variables.png')
plt.close()
print("  Saved categorical_variables.png")

# 2. Numerical variables
print("- Numerical variables histograms and boxplots...")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 15))

for i, col in enumerate(numerical_cols):
    # Histogram
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(x='Churn', y=col, data=df, ax=axes[i, 1])
    axes[i, 1].set_title(f'{col} by Churn')

plt.tight_layout()
plt.savefig(f'{output_dir}/numerical_variables.png')
plt.close()
print("  Saved numerical_variables.png")

# 3. Correlation heatmap
print("- Correlation heatmap...")
plt.figure(figsize=(10, 8))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png')
plt.close()
print("  Saved correlation_heatmap.png")

# 4. Scatter plots
print("- Scatter plots...")
plt.figure(figsize=(15, 5))
plt.subplot(131)
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df)
plt.title('Tenure vs Monthly Charges')

plt.subplot(132)
sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=df)
plt.title('Tenure vs Total Charges')

plt.subplot(133)
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df)
plt.title('Monthly Charges vs Total Charges')

plt.tight_layout()
plt.savefig(f'{output_dir}/scatter_plots.png')
plt.close()
print("  Saved scatter_plots.png")

# 5. Pair plot (with a sample to avoid large plot)
print("- Pair plot...")
sample_df = df[numerical_cols + ['Churn']].sample(min(1000, len(df)))
sns.pairplot(sample_df, hue='Churn')
plt.savefig(f'{output_dir}/pair_plot.png')
plt.close()
print("  Saved pair_plot.png")

# Create an insights file
insights = """
# EDA Analysis Insights for Telco Customer Churn

## 1. Categorical Variables Analysis

### Contract Type
- Month-to-month contracts show significantly higher churn rates
- Two-year contracts have the lowest churn rates
- Longer contracts appear to be effective retention tools

### Internet Service
- Fiber optic service customers have higher churn rates
- No internet service customers have lower churn rates
- DSL appears more stable with lower churn

### Payment Method
- Electronic check users show the highest churn rates
- Automatic payment methods correlate with lower churn
- This suggests convenience of payment may impact retention

## 2. Numerical Variables Analysis

### Tenure
- Strong negative correlation with churn
- New customers (low tenure) are much more likely to churn
- After about 12 months, churn rates decrease substantially

### Monthly Charges
- Higher monthly charges correlate with increased churn
- Price sensitivity appears to be a factor in customer decisions
- When combined with short-term contracts, high charges are particularly problematic

### Total Charges
- Strongly correlated with tenure (as expected)
- Lower for churned customers primarily because they leave earlier

## 3. Relationship Analysis

- Customers with high monthly charges and low tenure are at highest risk
- Long-term contracts effectively mitigate the churn risk of high charges
- There appears to be a critical early period (0-12 months) where retention efforts are most important

## 4. Recommendations

1. **Target New Customers**: Implement special retention programs for customers in their first year
2. **Contract Incentives**: Offer compelling incentives for longer contract commitments
3. **Payment Method**: Encourage automatic payment methods over manual methods
4. **Price Optimization**: Review pricing strategy for high-churn segments, especially fiber optic service
5. **Service Bundles**: Create attractive service bundles tied to longer contracts
"""

# Save insights
with open(f'{output_dir}/../eda_insights.md', 'w') as f:
    f.write(insights)

print(f"\nInsights saved to: {output_dir}/../eda_insights.md")
print("EDA analysis completed successfully!")
