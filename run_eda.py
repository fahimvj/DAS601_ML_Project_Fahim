"""
Run EDA analysis and generate comprehensive report
"""

from src.eda import run_complete_eda
from src.eda_report import generate_eda_report

def main():
    """
    Run the EDA and generate a report with insights
    """
    print("Starting EDA analysis for Telco Customer Churn dataset...")
    
    # Run the complete EDA and get insights
    insights = run_complete_eda()
    
    # Generate a report based on the insights
    report_path = generate_eda_report(insights)
    
    print(f"\nEDA Analysis complete! Report saved to: {report_path}")
    print("\nKey visualizations available in reports/figures/eda/")

if __name__ == "__main__":
    main()
