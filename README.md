# ML Project EDU - Telco Customer Churn

This repository contains a machine learning project focused on predicting customer churn using the Telco Customer dataset.

## Project Structure

```
├── data
│   ├── input           # Raw data
│   ├── interim         # Intermediate data that has been transformed
│   └── output          # Final, processed data used for modeling
├── notebooks           # Jupyter notebooks for exploration and analysis
├── reports             # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         # Generated graphics and figures
├── src                 # Source code for use in this project
│   ├── features        # Scripts for feature engineering
│   ├── models          # Scripts for training and prediction
│   └── utils           # Utility functions
├── main.py             # Main script to run the pipeline
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Getting Started

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - Windows: `env\Scripts\activate`
   - Mac/Linux: `source env/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

To run the complete pipeline:
```
python main.py
```

## Project Description

This project aims to predict customer churn for a telecommunications company using machine learning techniques. Features in the dataset include customer demographics, services subscribed, contract type, and billing information.

The analysis includes:
1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model building
5. Model evaluation
6. Insights and recommendations
