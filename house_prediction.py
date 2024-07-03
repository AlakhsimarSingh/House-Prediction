import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import datasets

house_pred = datasets.fetch_california_housing()
house_price_prediction = pd.DataFrame(house_pred.data, columns=house_pred.feature_names)
house_price_prediction['price'] = house_pred.target
X = house_price_prediction.drop('price',axis = 1)
Y = house_price_prediction['price']
print(X.head())
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)
model = XGBRegressor()
model.fit(X_train,Y_train)
f_x_train = model.predict(X_train)
f_x_test = model.predict(X_test)
print(r2_score(Y_train,f_x_train))
print(r2_score(Y_test,f_x_test))
plt.scatter(Y_test,f_x_test)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

input_data = pd.DataFrame([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]], columns=house_pred.feature_names)
pred = model.predict(input_data)
print("Predicted price for the new input data:", pred)