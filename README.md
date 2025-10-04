# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date:04-10-2025

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/content/blood_donor_dataset.csv', parse_dates=['created_at'], index_col='created_at')
print(data.columns)
data['created_at_numeric'] = data.index.astype('int64') // 10**9

result = adfuller(data['created_at_numeric'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['created_at_numeric'], lags=lag_order)
model_fit = model.fit()
plt.figure(figsize=(10, 6))
plot_acf(data['created_at_numeric'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['created_at_numeric'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

mse = mean_squared_error(test_data['created_at_numeric'], predictions)
print('Mean Squared Error (MSE):', mse)
plt.figure(figsize=(12, 6))
plt.plot(test_data['created_at_numeric'], label='Test Data - Date (numeric representation)')
plt.plot(predictions, label='Predictions - Date (numeric representation)',linestyle='--')
plt.xlabel('Index')
plt.ylabel('Date (numeric representation)')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
GIVEN DATA:

<img width="1392" height="193" alt="image" src="https://github.com/user-attachments/assets/d1e30580-676d-40dd-a0f3-60a82adbb956" />

ADF test result:

<img width="273" height="36" alt="image" src="https://github.com/user-attachments/assets/5abbecdf-6ff3-4a37-9094-e809ffbe79bf" />

PACF - ACF

<img width="666" height="510" alt="image" src="https://github.com/user-attachments/assets/8ad3b65e-749c-47d8-acc1-3dd7b4d888eb" />

<img width="650" height="527" alt="image" src="https://github.com/user-attachments/assets/c226dc9c-9fa4-4150-ab29-c38c20e9ecd0" />

Accuracy:

<img width="401" height="22" alt="image" src="https://github.com/user-attachments/assets/b70421a8-a7fe-4d81-8f30-b6099f18b228" />

FINIAL PREDICTION:

<img width="1216" height="632" alt="image" src="https://github.com/user-attachments/assets/f224a033-9ebf-43ba-bf89-3e1f8c590491" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
