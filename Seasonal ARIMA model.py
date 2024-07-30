#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:58:59 2024

@author: niloofarakbarian
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")
import os

# Change directory
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD-UT/Optimization")

# Load the dataset
data = pd.read_csv('DataFore.csv', parse_dates=['Years'], index_col=['Years'])

# Check the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Plot the original data
plt.xlabel('Years')
plt.ylabel('Biofuel Demand (million liter)')
plt.plot(data)
plt.show()

# Log transform the data to stabilize variance
data_log = np.log(data)

# Check for stationarity using rolling mean and standard deviation
rolling_mean = data_log.rolling(window=12).mean()
rolling_std = data_log.rolling(window=12).std()

plt.plot(data_log, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='green', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

# Perform Dickey-Fuller test
result = adfuller(data_log['Demand'])
print('ADF Statistics: {}'.format(result[0]))
print('P-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

# Function to check stationarity
def get_stationarity(timeseries):
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.xlabel('Years')
    plt.ylabel('Biofuel Demand (million liter)')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    plt.show()

    result = adfuller(timeseries)
    print('ADF Statistics: {}'.format(result[0]))
    print('P-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# Differencing to achieve stationarity
data_log_diff = data_log.diff().dropna()

# Check stationarity of differenced data
get_stationarity(data_log_diff['Demand'])

# Fit ARIMA model using auto_arima to find the best parameters
stepwise_model = auto_arima(data_log, start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=True,
                            m=4, d=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

print(stepwise_model.summary())

# Fit the model
model = ARIMA(data_log, order=stepwise_model.order, seasonal_order=stepwise_model.seasonal_order)
results = model.fit()

# Plotting fitted values
plt.plot(data_log, color='blue', label='Original')
plt.plot(results.fittedvalues, color='red', label='Fitted Values')
plt.legend(loc='upper right', fontsize=8)
plt.title('ARIMA Model Fitting on Log Data')
plt.show()

# Forecast future values
forecast_steps = 20  # Forecasting 5 years assuming quarterly data
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps, freq='Q')
forecast_values = np.exp(forecast.predicted_mean)
forecast_conf_int = np.exp(forecast.conf_int())

# Calculate MSE and MAPE for the available data
fitted_values = np.exp(results.fittedvalues)
mse = mean_squared_error(data.values, fitted_values)
mape = mean_absolute_percentage_error(data.values, fitted_values)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# Save forecasted values to Excel
forecast_df = pd.DataFrame({
    'Forecast': forecast_values,
    'Lower CI': forecast_conf_int.iloc[:, 0],
    'Upper CI': forecast_conf_int.iloc[:, 1]
}, index=forecast_index)

forecast_df.to_excel('Biofuel_Demand_Forecast_ARIMA.xlsx')

# Plotting the forecast
plt.plot(data, color='blue', label='Original Data')
plt.plot(forecast_index, forecast_values, color='green', label='Forecast')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.3)
plt.xlabel('Years')
plt.ylabel('Biofuel Demand (million liter)')
plt.legend(loc='upper left', fontsize=8)
plt.title('Biofuel Demand Forecast using ARIMA')
plt.show()

#