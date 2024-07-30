import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
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

# Predicting on the training set
y_train_pred = lin_reg.predict(X_train)

# Predicting on the testing set
y_test_pred = lin_reg.predict(X_test)

# Visualising the Linear Regression results on the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_train_pred, color='blue')
plt.title('Truth or Bluff (Linear Regression - Training set)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Linear Regression results on the testing set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_test_pred, color='blue')
plt.title('Truth or Bluff (Linear Regression - Test set)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Forecasting the next 20 periods
max_X = max(X)[0]
future_periods = np.arange(max_X + 1, max_X + 21).reshape(-1, 1)
future_predictions = lin_reg.predict(future_periods)

# Visualising the forecast
plt.scatter(X, y, color='red')
plt.plot(np.vstack((X, future_periods)), lin_reg.predict(np.vstack((X, future_periods))), color='blue')
plt.plot(future_periods, future_predictions, color='green', linestyle='dashed')
plt.title('Linear Regression Forecast')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print("Future periods forecast:", future_predictions)

# Calculate Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) on the training set
mse_train = mean_squared_error(y_train, y_train_pred)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
print(f"Training Mean Squared Error (MSE): {mse_train}")
print(f"Training Mean Absolute Percentage Error (MAPE): {mape_train * 100:.2f}%")

# Calculate Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) on the testing set
mse_test = mean_squared_error(y_test, y_test_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
print(f"Test Mean Squared Error (MSE): {mse_test}")
print(f"Test Mean Absolute Percentage Error (MAPE): {mape_test * 100:.2f}%")