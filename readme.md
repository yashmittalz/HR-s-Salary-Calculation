# Salary Calculation App

## Overview
The Salary Calculation App predicts employee salaries using machine learning algorithms, specifically polynomial regression. This application analyzes salary data correlations to provide accurate salary predictions based on experience levels.

## Features
- Predicts employee salaries based on experience level.
- Utilizes polynomial regression for better accuracy.
- User-friendly interface for inputting experience levels.
- Visual representation of actual vs. predicted salaries.

## Requirements
To run this application, you need the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install the required packages using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage
1. Clone the repository or download the salary_calculation.py file.
2. Ensure the salaries_dataset.csv file is in the same directory as the script.
3. Run the application:
    ``` bash 
    python salary_calculation.py
    ```
4. Enter the experience level of the employee when prompted.
5. The application will display the predicted salary and plot the results.

## Dataset
The application uses a dataset named salaries_dataset.csv, which should contain:

1. experience_level: Numeric values representing years of experience.
2. salary: Numeric values representing corresponding salaries.

| Experience Level | Salary ($) |
|------------------|------------|
| 1                | 4,500      |
| 2                | 5,000      |
| 3                | 6,000      |
| 4                | 8,000      |
| 5                | 11,000     |
| 6                | 15,000     |
| 7                | 20,000     |
| 8                | 30,000     |
| 9                | 50,000     |
| 10               | 100,000    |

## Project Background: HR Salary Calculation
Calculating each employee's salary based on experience level can be tedious, especially when the relationship is non-linear. In this project, this project deals with a machine learning model that leverages polynomial regression, an effective model given that most salary structures are polynomial.

## Why Polynomial Regression?
Most companies have a non-linear salary structure, where salary growth is exponential rather than linear. Polynomial regression allows us to model these non-linear relationships accurately.

### Polynomial Regression Equation
The model is typically expressed as:
Y = β0 +β1 X+β2 X^2 +⋯+βn X^n +ϵ

### Key Steps in the HR Salary Calculation Code
1. Data Loading: Loads salaries_dataset.csv using pandas.
2. Data Visualization: Creates a scatter plot to show the relationship between experience level and salary.
3. Linear Regression Model: Tests a basic linear model, which proves insufficient for non-linear salary data.
4. Polynomial Transformation: Uses PolynomialFeatures to transform experience data.
5. Polynomial Regression Model: Fits a polynomial model to the transformed data.
6. Prediction: Predicts salary for specific experience levels.
7. Conclusion: Demonstrates that polynomial regression better suits the data than linear regression.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Developed by Yash Mittal. Version 1.0

