import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
ct = ColumnTransformer([("State", OneHotEncoder(),[3])], remainder="passthrough") 
ct.fit_transform(x)    

#Avoiding the Dummy Variable Trap



#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

#Fitting Multiple Linear Regressor to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)

#building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
import statsmodels.api as sm
x_train = np.append(arr = np.ones((40, 1)), values = x_train, axis = 1)
x_opt = np.array(x_train[:, [0, 1, 2, 3, 4]], dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x_train[:, [0, 1, 2, 3]], dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x_train[:, [0, 1, 3]], dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()
x_opt = np.array(x_train[:, [0, 1]], dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()
