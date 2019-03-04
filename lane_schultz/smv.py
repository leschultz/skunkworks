'''
Use the suport vector machine provided by scikit learn for ehull prediction.
'''

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge

from matplotlib import pyplot as pl

import pandas as pd
import numpy as np

# General inputs
path = '../data/Perovskite_ORR_dataset.xlsx'  # Path to data
drop_features = ['Simulated Composition']  # Unwanted columns
target = 'Energy above convex hull (meV/atom)'  # Target
split = 0.1  # Split for training and testing
strat_splits = 5  # The splits for stratified cross validation
step = 1

# Import the data
df = pd.read_excel(path)

# Remove unwanted features
data = df.drop(drop_features, axis=1)  # Convert to array

# The data set divided into features and the target
X = data.loc[:, data.columns != target].values
y = data[target].values

# Split the data into training and testing sets
split = train_test_split(X, y, test_size=split, random_state=None)
X_train, X_test, y_train, y_test = split

# Process the data to reduce bias from feature scales
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

# Select the model
clf = KernelRidge(kernel='rbf')

'''
# Feature elimiation to determine important features
rfecv = RFECV(
              estimator=clf,
              step=step,
              cv=strat_splits,
              )

rfecv.fit(X_train_scaled, y_train)

print(rfecv.ranking_)

'''

# Train the model
clf.fit(X_train_scaled, y_train)

# Predict with trained model
y_pred = clf.predict(X_test_scaled)

# Error Metrics
r2 = r2_score(y_test, y_pred)  # Coefficient of determination
mse = mean_squared_error(y_test, y_pred)  # Mean squared error
mae = mean_absolute_error(y_test, y_pred)  # Mean absolute error

print(mse)

# Plot the true versus predicted values
fig, ax = pl.subplots()
ax.scatter(
           y_test,
           y_pred,
           marker='.',
           )

textbox = r'$R^{2}$='+str(r2)+'\n'+r'MSE='+str(mse)+'\n'+r'MAE='+str(mae)
ax.legend([textbox])
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.grid()
pl.show()
