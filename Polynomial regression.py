import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Importing the dataset
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD-UT/Optimization")

# Load the dataset
dataset = pd.read_csv('DataDemand_regression.csv')

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model on the training set
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Training the Polynomial Regression model on the training set
poly_reg = PolynomialFeatures(degree=3)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_train, y_train)

# Visualising the Linear Regression results on the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lin_reg.predict(X_train), color='blue')
plt.title('Truth or Bluff (Linear Regression - Training set)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Linear Regression results on the testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, lin_reg.predict(X_test), color='blue')
plt.title('Truth or Bluff (Linear Regression - Test set)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results on the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression - Training set)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results on the testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, lin_reg_2.predict(poly_reg.transform(X_test)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression - Test set)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(f"Linear Regression prediction for level 6.5: {lin_reg.predict([[6.5]])}")

# Predicting a new result with Polynomial Regression
print(f"Polynomial Regression prediction for level 6.5: {lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))}")

# Forecasting the next 20 periods
max_X = max(X)[0]
future_periods = np.arange(max_X + 1, max_X + 21).reshape(-1, 1)
future_periods_poly = poly_reg.transform(future_periods)
future_predictions = lin_reg_2.predict(future_periods_poly)

# Visualising the forecast
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.plot(future_periods, future_predictions, color='green', linestyle='dashed')
plt.title('Polynomial Regression Forecast')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Calculate Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) for Linear Regression on training set
y_train_pred_linear = lin_reg.predict(X_train)
mse_train_linear = mean_squared_error(y_train, y_train_pred_linear)
mape_train_linear = mean_absolute_percentage_error(y_train, y_train_pred_linear)

print(f"Training Linear Regression Mean Squared Error (MSE): {mse_train_linear}")
print(f"Training Linear Regression Mean Absolute Percentage Error (MAPE): {mape_train_linear}")

# Calculate Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) for Linear Regression on testing set
y_test_pred_linear = lin_reg.predict(X_test)
mse_test_linear = mean_squared_error(y_test, y_test_pred_linear)
mape_test_linear = mean_absolute_percentage_error(y_test, y_test_pred_linear)

print(f"Testing Linear Regression Mean Squared Error (MSE): {mse_test_linear}")
print(f"Testing Linear Regression Mean Absolute Percentage Error (MAPE): {mape_test_linear}")

# Calculate Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) for Polynomial Regression on training set
y_train_pred_poly = lin_reg_2.predict(X_poly_train)
mse_train_poly = mean_squared_error(y_train, y_train_pred_poly)
mape_train_poly = mean_absolute_percentage_error(y_train, y_train_pred_poly)

print(f"Training Polynomial Regression Mean Squared Error (MSE): {mse_train_poly}")
print(f"Training Polynomial Regression Mean Absolute Percentage Error (MAPE): {mape_train_poly}")

# Calculate Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) for Polynomial Regression on testing set
y_test_pred_poly = lin_reg_2.predict(X_poly_test)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)
mape_test_poly = mean_absolute_percentage_error(y_test, y_test_pred_poly)

print(f"Testing Polynomial Regression Mean Squared Error (MSE): {mse_test_poly}")
print(f"Testing Polynomial Regression Mean Absolute Percentage Error (MAPE): {mape_test_poly}")

print("Future periods forecast:", future_predictions)