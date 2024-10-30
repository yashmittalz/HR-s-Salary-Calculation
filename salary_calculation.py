import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# App Desciption
print("Welcome to the Salary Calculation App!\n")
print("Predicts emploee salaries using machine learning algorithms.\n")
print("Utilizes polynomial regression to analyze salary data correlations.\n")
print("Aims to be a step towards Yash's 50 projects showcasing data science applications.\n")
print("Developed by Yash Mittal. Version 1.0\n")

# Load the dataset
df = pd.read_csv("salaries_dataset.csv", sep=";")

# Prepare the data for polynomial regression

## Create an instance of the PolynomialFeatures class from the sklearn.preprocessing module.
polynomial_regression = PolynomialFeatures(degree=4) 

## Apply the polynomial transformation to the experience_level data.
x_polynomial = polynomial_regression.fit_transform(df[['experience_level']]) 

## Initialize a new instance of the LinearRegression class from the sklearn.linear_model module. 
reg = LinearRegression() 

## Train the linear regression model using the provided data.
reg.fit(x_polynomial, df['salary']) 


def predict_salary(experience_level):
    # Transform the input experience level to polynomial features
    x_polynomial_input = polynomial_regression.fit_transform([[experience_level]])
    # Predict the salary
    predicted_salary = reg.predict(x_polynomial_input)
    return predicted_salary[0] # [0] - This accesses the first (and in this case, the only) element of the array returned by the prediction. 

# Example usage
experience = float(input("Enter the experience level of the employee: "))
predicted_salary = predict_salary(experience)
print(f"\nThe predicted salary for an employee with {experience} years of experience is: ${predicted_salary:.2f}\n")


# Plotting the results
# Scatter plot of the actual data

## Plot a scatter plot with experience_level on the x-axis and salary on the y-axis, using blue color and labeling the points as 'Actual Salaries'.
plt.scatter(df['experience_level'], df['salary'], color='blue', label='Actual Salaries')  

# Plotting the polynomial regression line

## Generates an array of values from the minimum to the maximum experience level in increments of 0.1 and reshapes it for polynomial transformation.
x_grid = np.arange(min(df['experience_level']), max(df['experience_level']), 0.1).reshape(-1, 1)  

## Transforms the x_grid values into polynomial features using the PolynomialFeatures instance.
x_grid_polynomial = polynomial_regression.fit_transform(x_grid) 

## Plots the predicted salary values against the experience levels on the grid, using a red line and labeling it as 'Polynomial Regression'.
plt.plot(x_grid, reg.predict(x_grid_polynomial), color='red', label='Polynomial Regression')  

# Highlight the input experience level
plt.scatter(experience, predicted_salary, color='green', edgecolor='black', label='Input Experience Level')


# Adding labels and title
plt.title('Experience Level vs Salary')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.legend()
plt.show()  # Display the plot