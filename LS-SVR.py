import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split
import os

# Change directory
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD-UT/Optimization")

# Load the dataset
df = pd.read_csv('DataDemand.csv', parse_dates=['Years'], index_col=['Years'])

# Ensure 'Years' is set as index
df.index = pd.to_datetime(df.index)

# Create the 'Quarter' column
df['Quarter'] = df.index.quarter

# Visualize the data
plt.figure(figsize=(15, 8))
plt.plot(df.index, df.iloc[:, 0], label='Biofuel demand (million liter)')
plt.xlabel('Years')
plt.ylabel('Biofuel Demand (million liter)')
plt.legend()
plt.show()

# Prepare feature and target arrays
df['Year_Quarter'] = df.index.year + (df.index.quarter - 1) / 4.0

# Create dummy variables for each quarter
df = pd.get_dummies(df, columns=['Quarter'], prefix='Q', drop_first=True)

X = df[['Year_Quarter', 'Q_2', 'Q_3', 'Q_4']].values
y = df.iloc[:, 0].values.reshape(-1, 1)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)

# LS-SVR Model (approximated by specific SVR settings)
# Hyperparameter tuning with GridSearchCV for LS-SVR approximation
param_grid_ls_svr = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}
grid_search_ls_svr = GridSearchCV(SVR(), param_grid_ls_svr, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search_ls_svr.fit(X_train_scaled, y_train_scaled.ravel())

# Best parameters for LS-SVR
print("Best parameters for LS-SVR: ", grid_search_ls_svr.best_params_)

# Train the LS-SVR model with best parameters
best_ls_svr = grid_search_ls_svr.best_estimator_
best_ls_svr.fit(X_train_scaled, y_train_scaled.ravel())

# Predicting the values with LS-SVR
y_train_pred_scaled_ls_svr = best_ls_svr.predict(X_train_scaled)
y_train_pred_ls_svr = sc_y.inverse_transform(y_train_pred_scaled_ls_svr.reshape(-1, 1))

y_test_pred_scaled_ls_svr = best_ls_svr.predict(X_test_scaled)
y_test_pred_ls_svr = sc_y.inverse_transform(y_test_pred_scaled_ls_svr.reshape(-1, 1))

# Calculate and print the Mean Squared Error and Mean Absolute Percentage Error for LS-SVR
mse_train_ls_svr = mean_squared_error(y_train, y_train_pred_ls_svr)
rmse_train_ls_svr = np.sqrt(mse_train_ls_svr)
mape_train_ls_svr = mean_absolute_percentage_error(y_train, y_train_pred_ls_svr)
print(f"LS-SVR Model - Training Mean Squared Error (MSE): {mse_train_ls_svr}")
print(f"LS-SVR Model - Training Root Mean Squared Error (RMSE): {rmse_train_ls_svr}")
print(f"LS-SVR Model - Training Mean Absolute Percentage Error (MAPE): {mape_train_ls_svr}")

mse_test_ls_svr = mean_squared_error(y_test, y_test_pred_ls_svr)
rmse_test_ls_svr = np.sqrt(mse_test_ls_svr)
mape_test_ls_svr = mean_absolute_percentage_error(y_test, y_test_pred_ls_svr)
print(f"LS-SVR Model - Testing Mean Squared Error (MSE): {mse_test_ls_svr}")
print(f"LS-SVR Model - Testing Root Mean Squared Error (RMSE): {rmse_test_ls_svr}")
print(f"LS-SVR Model - Testing Mean Absolute Percentage Error (MAPE): {mape_test_ls_svr}")

# Visualizing the LS-SVR results
plt.figure(figsize=(15, 8))
plt.scatter(X_train[:, 0], y_train, color='red', label='Training Data')
plt.scatter(X_test[:, 0], y_test, color='orange', label='Testing Data')
plt.plot(X_train[:, 0], y_train_pred_ls_svr, color='green', label='LS-SVR Training Prediction')
plt.plot(X_test[:, 0], y_test_pred_ls_svr, color='blue', label='LS-SVR Testing Prediction')
plt.xlabel('Years')
plt.ylabel('Biofuel Demand')
plt.legend()
plt.show()

# Higher resolution for smoother curve
X_grid_years = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01).reshape(-1, 1)
X_grid_dummies = np.zeros((len(X_grid_years), 3))  # Create dummy variables for quarters
X_grid = np.hstack((X_grid_years, X_grid_dummies))
X_grid_scaled = sc_X.transform(X_grid)
y_grid_pred_scaled_ls_svr = best_ls_svr.predict(X_grid_scaled)
y_grid_pred_ls_svr = sc_y.inverse_transform(y_grid_pred_scaled_ls_svr.reshape(-1, 1))

# Visualizing the LS-SVR results with higher resolution
plt.figure(figsize=(15, 8))
plt.scatter(X[:, 0], y, color='red', label='Original Data')
plt.plot(X_grid[:, 0], y_grid_pred_ls_svr, color='green', label='LS-SVR Prediction (High Resolution)')
plt.xlabel('Years')
plt.ylabel('Biofuel Demand')
plt.legend()
plt.show()

# Forecasting the next 20 periods (5 years, 4 quarters each)
last_period = df.index.to_period('Q')[-1]
future_periods = pd.period_range(start=last_period + 1, periods=20, freq='Q')
future_years = future_periods.year + (future_periods.quarter - 1) / 4.0
future_years = future_years.values.reshape(-1, 1)

# Create dummy variables for future quarters
future_quarters = pd.get_dummies(future_periods.quarter, prefix='Q', drop_first=True)
future_quarters = future_quarters.reindex(columns=['Q_2', 'Q_3', 'Q_4'], fill_value=0).values

future_years_dummies = np.hstack((future_years, future_quarters))

# Scale future periods using the same scaler
future_years_scaled = sc_X.transform(future_years_dummies)

# Predict future values
future_pred_scaled_ls_svr = best_ls_svr.predict(future_years_scaled)
future_pred_ls_svr = sc_y.inverse_transform(future_pred_scaled_ls_svr.reshape(-1, 1))

# Create future dates for plotting
future_dates = future_periods.to_timestamp()

# Combine original data and future predictions
all_years = np.concatenate((X[:, 0], future_years[:, 0]))
all_demand = np.concatenate((y.flatten(), future_pred_ls_svr.flatten()))

# Visualizing the LS-SVR results with forecast
plt.figure(figsize=(15, 8))
plt.scatter(X[:, 0], y, color='red', label='Original Data')
plt.plot(X[:, 0], np.concatenate((y_train_pred_ls_svr, y_test_pred_ls_svr)), color='green', label='LS-SVR Prediction')
plt.plot(future_years[:, 0], future_pred_ls_svr, color='blue', label='Forecast')
plt.xlabel('Years')
plt.ylabel('Biofuel Demand')
plt.legend()
plt.show()

# Visualizing the LS-SVR results with higher resolution and forecast
X_grid_future_years = np.arange(min(all_years), max(all_years), 0.01).reshape(-1, 1)
X_grid_future_dummies = np.zeros((len(X_grid_future_years), 3))  # Create dummy variables for quarters
X_grid_future = np.hstack((X_grid_future_years, X_grid_future_dummies))
X_grid_future_scaled = sc_X.transform(X_grid_future)
y_grid_future_pred_scaled_ls_svr = best_ls_svr.predict(X_grid_future_scaled)
y_grid_future_pred_ls_svr = sc_y.inverse_transform(y_grid_future_pred_scaled_ls_svr.reshape(-1, 1))

plt.figure(figsize=(15, 8))
plt.scatter(X[:, 0], y, color='red', label='Original Data')
plt.plot(X_grid_future[:, 0], y_grid_future_pred_ls_svr, color='green', label='LS-SVR Prediction with Forecast (High Resolution)')
plt.xlabel('Years')
plt.ylabel('Biofuel Demand')
plt.legend()
plt.show()

# Store forecasted demand for the next 5 years in an Excel file
forecast_df_ls_svr = pd.DataFrame({
    'Year': future_dates,
    'Forecasted_Demand': future_pred_ls_svr.flatten()
})
forecast_df_ls_svr.to_excel('Biofuel_Demand_Forecast_LS_SVR.xlsx', index=False)

print("Forecasted demand for the next 5 years has been saved to 'Biofuel_Demand_Forecast_LS_SVR.xlsx'.")