# Student Academic Performance Analyzer

A MATLAB data analysis and predictive modeling project that evaluates and predicts student performance based on study hours and attendance using linear regression.

- Project Overview
This project applies fundamental MATLAB programming concepts to perform exploratory data analysis, fit a linear regression model, evaluate prediction accuracy, and export results for reporting.

- Key Features
Data Ingestion & Verification: Processes study metrics and cleans input arrays.
Predictive Modeling: Fits a linear regression model to estimate student scores.
Performance Evaluation: Calculates key statistical metrics:
  R² (Coefficient of Determination) for model fit
  RMSE (Root Mean Squared Error) for prediction variance
  MAE (Mean Absolute Error) for average error magnitude
Residual Analysis: Computes residual errors and identifies maximum prediction variance.
Data Export: Generates `Student_Performance_Results.csv` containing final predictions.

- Repository Structure
```text
Student-Academic-Performance-Analyzer/
│
├── Student_Academic_Performance_Analyzer.m   # Main MATLAB analysis script
├── Student_Performance_Results.csv           # Model predictions dataset
└── README.md                                 # Documentation
