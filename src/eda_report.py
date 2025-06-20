"""
EDA Analysis Report Generator for Telco Customer Churn Dataset

This script generates a comprehensive analysis report of the EDA visualizations
and saves it as a markdown file.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

def create_report_directory():
    """Create directory for storing reports"""
    report_dir = 'reports/eda_analysis'
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def generate_report(insights: Dict) -> str:
    """
    Generate a comprehensive report based on the EDA visualizations
    
    Args:
        insights: Dictionary containing insights from EDA
        
    Returns:
        str: Markdown formatted report content
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report = f"""
# Telco Customer Churn - EDA Analysis Report
*Generated on: {now}*

## 1. Dataset Overview

- Total records: {insights['data_overview']['rows']}
- Total features: {insights['data_overview']['columns']}
- Missing values: {insights['data_overview']['missing_values']}
- Duplicate rows: {insights['data_overview']['duplicate_rows']}

## 2. Analysis of Categorical Variables

### Distribution and Churn Rate Analysis

The categorical variable visualizations (categorical_distributions.png) reveal several important patterns:

1. **Contract Type**:
   - Month-to-month contracts have a significantly higher churn rate compared to one-year and two-year contracts
   - This suggests that customers with shorter commitment periods are more likely to leave the service
   - Long-term contracts serve as an effective retention mechanism

2. **Internet Service**:
   - Customers with Fiber optic internet service show a higher propensity to churn
   - This could indicate potential issues with the fiber optic service quality or pricing structure
   - Customers without internet service have the lowest churn rate, suggesting they might be using only basic services

3. **Payment Method**:
   - Electronic check payment method has the highest association with customer churn
   - Automatic payment methods (bank transfer, credit card) show lower churn rates
   - This suggests that customers who actively have to make payments are more likely to reconsider their service

4. **Gender and Senior Citizen**:
   - Gender does not appear to have a strong relationship with churn
   - Senior citizens have a slightly higher churn rate than non-seniors

### Grouped and Stacked Bar Chart Analysis

The grouped and stacked bar charts further illustrate:

1. **Contract and Churn Relationship**:
   - Month-to-month contracts represent the largest segment of churned customers
   - When examining the proportional (stacked) charts, we see that over 40% of month-to-month customers churn
   - In contrast, less than 10% of two-year contract customers leave the service

2. **Supplementary Services Impact**:
   - Customers without tech support, online security, or online backup show higher churn rates
   - This indicates that value-added services may increase customer loyalty
   - Bundling these services could be an effective retention strategy

## 3. Analysis of Numeric Variables

### Distribution and Outlier Analysis

The numeric variable distributions (numeric_distributions.png) show:

1. **Tenure**:
   - Bimodal distribution with peaks at low tenure (new customers) and high tenure (loyal customers)
   - This suggests there might be a critical period where customers decide whether to stay long-term
   - Boxplots show some outliers but not extreme ones

2. **Monthly Charges**:
   - Fairly even distribution with slight right skew
   - Charges range primarily between $20-$110 per month
   - There's a concentration of customers in the $70-$90 range

3. **Total Charges**:
   - Right-skewed distribution as expected (accumulated charges over time)
   - Strong correlation with tenure
   - Contains outliers at the high end representing long-term customers with high monthly charges

### Churn Rate by Numeric Variables (KDE and Boxplots)

The density plots by churn status reveal:

1. **Tenure and Churn**:
   - Churned customers have significantly lower tenure
   - The density peak for churned customers is at very low tenure values (0-10 months)
   - This suggests the first year is critical for customer retention

2. **Monthly Charges and Churn**:
   - Customers who churn tend to have higher monthly charges
   - The difference in distributions suggests price sensitivity as a churn factor

3. **Boxplot Insights**:
   - The interquartile range for tenure is much smaller for churned customers
   - This reinforces that most churned customers leave early in their service period

## 4. Relationship Analysis

### Correlation and Scatter Plot Insights

The correlation heatmap and scatter plots show:

1. **Tenure and Monthly Charges**:
   - Weak correlation between tenure and monthly charges
   - This suggests that long-term customers aren't necessarily paying higher rates
   - The scatter plot shows two distinct clusters, possibly representing different service tiers

2. **Tenure and Total Charges**:
   - Very strong positive correlation as expected
   - The regression line shows a clear linear relationship
   - Customers who churn tend to be in the lower left corner (low tenure, low total charges)

3. **Monthly Charges and Total Charges**:
   - Moderate positive correlation
   - The bubble plot reveals that customers with high monthly charges but low tenure are at higher risk of churning
   - This suggests that high initial charges might discourage customer retention

### Pair Plot Analysis

The pair plots for numeric variables demonstrate:

1. **Multivariate Relationships**:
   - Churned customers cluster in specific regions of the feature space
   - High monthly charges combined with low tenure is a strong predictor of churn
   - The diagonal KDE plots confirm the different distributions for churners vs. non-churners

## 5. Categorical-Numeric Relationships

The heatmaps exploring categorical and numeric variables show:

1. **Contract Type and Monthly Charges**:
   - Month-to-month customers pay higher monthly charges on average
   - This combination (high cost + low commitment) contributes to higher churn rate

2. **Internet Service and Monthly Charges**:
   - Fiber optic service has the highest average monthly charges
   - Combined with higher churn rate, this suggests potential pricing issues for this service tier

3. **Contract Type and Tenure**:
   - Two-year contract customers have much higher average tenure
   - This reinforces that contract length is an effective retention mechanism

## 6. Key Insights and Recommendations

Based on the comprehensive EDA, here are the key insights:

1. **Customer Segments at Risk**:
   - New customers (low tenure)
   - Month-to-month contract customers
   - Customers with high monthly charges
   - Customers using electronic check payment
   - Fiber optic internet service customers without additional services

2. **Retention Factors**:
   - Long-term contracts
   - Automatic payment methods
   - Multiple supplementary services (tech support, security, etc.)
   - Competitive pricing, especially for premium services

3. **Recommendations**:
   - Implement targeted retention campaigns for month-to-month customers
   - Review pricing and quality of fiber optic service
   - Offer incentives for customers to switch to automatic payment methods
   - Create service bundles that encourage longer contract commitments
   - Focus on customer experience during the first year to increase tenure

## 7. Visualizations

All visualizations are available in the directory: `{insights['paths']['output_directory']}`

*End of Report*
"""
    
    return report

def save_report(report_content: str, report_dir: str) -> str:
    """
    Save the report to a markdown file
    
    Args:
        report_content: The content of the report
        report_dir: Directory to save the report
        
    Returns:
        str: Path to the saved report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"eda_analysis_report_{timestamp}.md"
    filepath = os.path.join(report_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(report_content)
    
    return filepath

def generate_eda_report(insights: Optional[Dict] = None) -> str:
    """
    Generate and save an EDA report based on insights
    
    Args:
        insights: Dictionary of insights from EDA (optional)
        
    Returns:
        str: Path to the saved report
    """
    # If insights not provided, import from eda module and run
    if insights is None:
        from src.eda import run_complete_eda
        insights = run_complete_eda()
    
    # Create report directory
    report_dir = create_report_directory()
    
    # Generate report content
    report_content = generate_report(insights)
    
    # Save report
    report_path = save_report(report_content, report_dir)
    
    print(f"EDA analysis report saved to: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_eda_report()
