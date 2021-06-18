import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


from xgboost import plot_importance
import matplotlib.pyplot as plt
from xgboost import plot_tree

#data loading process
train_data_path = r'D:\研一下学期\数据挖掘\Red Wine Quality\preprocessing_data\data_train.csv'
test_data_path = r'D:\研一下学期\数据挖掘\Red Wine Quality\preprocessing_data\data_test.csv'
train_label_path = r'D:\研一下学期\数据挖掘\Red Wine Quality\preprocessing_data\target_train.csv'
test_label_path = r'D:\研一下学期\数据挖掘\Red Wine Quality\preprocessing_data\target_test.csv'

tr_data_csv=pd.read_csv(train_data_path)
te_data_csv=pd.read_csv(test_data_path)
tr_label_csv=pd.read_csv(train_label_path)
te_label_csv=pd.read_csv(test_label_path)
print(tr_data_csv.columns)

X=tr_data_csv[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
Y=tr_label_csv['quality']
X_test=te_data_csv[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
Y_test=te_label_csv['quality']

#XGBoost model create
xg=XGBRegressor(n_estimators=1120,learning_rate=0.01,min_child_weight=1,max_depth=6)
xg.fit(X, Y)
pre_test = xg.predict(X_test)
pre_train = xg.predict(X)

#accuracy result
print(xg.score(X_test, Y_test))
print(xg.score(X, Y))
print("accuracy score:" + str(accuracy_score(Y_test, np.round(pre_test))))
print(Y_test)
print(np.round(pre_test))

#visualization
from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(xg)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()

import graphviz
from xgboost import plot_tree
plot_tree(xg,num_trees=1)
plt.rcParams['figure.figsize'] = [500, 200]
plt.show()

