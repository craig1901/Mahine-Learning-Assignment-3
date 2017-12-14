import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

red_wine = pd.read_csv("Datasets/winequality-red.csv", delimiter=';')
print "Dataset loaded in."

X = red_wine.drop(['quality'], axis=1)
Y = red_wine['quality']

x_new = SelectKBest(f_regression, k=5).fit_transform(X, Y)
print x_new

reg = LinearRegression()
reg.fit(x_new, Y)
