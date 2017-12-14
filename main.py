import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

dataset = pd.read_csv("Datasets/winequality-red.csv", delimiter=';')
print "Dataset loaded in."



kf = KFold(n_splits=10, shuffle = True)
reg = LinearRegression()
lasso = Lasso()
lr_metrics = {"RMSE" : [], "Variance" : []}
lasso_metrics = {"RMSE" : [], "Variance": []}

for train_indices, test_indices in kf.split(dataset) :

    training_set = dataset.iloc[train_indices[:len(train_indices)]]
    test_set = dataset.iloc[test_indices[:len(test_indices)]]

    X = training_set.drop(['quality'], axis=1)
    Y = training_set['quality']


    features_test = test_set.drop(['quality'], axis=1)
    labels_test = test_set['quality']

    reg.fit(X, Y)
    lasso.fit(X, Y)

    reg_pred = reg.predict(features_test)
    lasso_pred = lasso.predict(features_test)

    lr_rmse = np.sqrt(mean_squared_error(labels_test, reg_pred))
    lasso_rmse = np.sqrt(mean_squared_error(labels_test, lasso_pred))
    lr_metrics["RMSE"].append(lr_rmse)
    lasso_metrics["RMSE"].append(lasso_rmse)

    lr_var = explained_variance_score(labels_test, reg_pred)
    lasso_var = explained_variance_score(labels_test, lasso_pred)
    lr_metrics["Variance"].append(lr_var)
    lasso_metrics["Variance"].append(lasso_var)

print 'Linear Regression RMSE : ', np.array(lr_metrics["RMSE"]).mean()
print 'Linear Regression Variance : ', np.array(lr_metrics["Variance"]).mean()
print 'Lasso Regression RMSE : ', np.array(lasso_metrics["RMSE"]).mean()
print 'Lasso Regression Variance : ', np.array(lasso_metrics["Variance"]).mean()
