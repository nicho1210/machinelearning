# Preprocess-importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

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
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
ct=ColumnTransformer([("Country", OneHotEncoder(), [0])],remainder='passthrough')
x = ct.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""