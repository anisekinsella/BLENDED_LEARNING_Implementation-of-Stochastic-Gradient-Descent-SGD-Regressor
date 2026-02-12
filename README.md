# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights and bias with small random values and set learning rate and number of epochs.

2. For each epoch, shuffle the training data and iterate through each training example.

3. For each example, predict the output, compute the error, and update weights and bias using the gradient of the loss function.

4. After training, evaluate the model performance using metrics like Mean Squared Error (MSE) or R² score.

## Program:
Developed by: ANISE KINSELLA A
RegisterNumber: 212225040021
```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("CarPrice_Assignment (1).csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
print(data.head())
print(data.info())
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(X_train, y_train)

# Making predictions
y_pred = sgd_model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('MAE= ',mean_absolute_error(y_test, y_pred))
print(f"R2: {r2_score(y_test, y_pred):.4f}")

# Print evaluation metrics
print('Name: ANISE KINSELLA A')
print('Reg. No: 212225040021')
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
```

## Output:
<img width="1191" height="592" alt="image" src="https://github.com/user-attachments/assets/c513e7db-a664-4ac3-8689-2b4a8fd33912" />
<img width="807" height="742" alt="image" src="https://github.com/user-attachments/assets/d09b8d10-254c-4f84-a4ab-e7fdd3ff5c6d" />
<img width="923" height="431" alt="image" src="https://github.com/user-attachments/assets/0c64d5f1-71ae-416f-82c4-244e8b0f942b" />
<img width="877" height="557" alt="image" src="https://github.com/user-attachments/assets/5123b1ea-44ce-4b77-81e3-a9290a9c7f35" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
