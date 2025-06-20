
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
