#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:41:58 2018

@author: zihaoli
"""

# 1. 

# import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
# from sklearn.metrics import mean_squared_error

# Load the dataset, salary variable has missing values
hitters = pd.read_csv("Hitters.csv")

# The first column should be player names

hitters.rename(columns = {'Unnamed: 0':'Player'}, inplace = True)

# Get all the missing elements on Salary column

print("Number of null values:", hitters["Salary"].isnull().sum())

# Salary is missing for 59 players

# Remove all of the rows that have missing values in any variable

hitters_clean = hitters.dropna().drop('Player', axis=1)

# Check if there is still missing values

print("Number of null values:", hitters_clean["Salary"].isnull().sum())

# There is some categorical predictors that we should omit
# We can either drop the categorical variables or make them as dummy variables

# dummies = pd.get_dummies(hitters_clean[['League', 'Division', 'NewLeague']])

y = hitters_clean.Salary

# Drop the column with the independent variable Salary, and dummy variables
X = hitters_clean.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')


# The Lasso Regression

# define the alpha

alphas = 10**np.linspace(3,-2,100)*0.5

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Set up and fit data into Lasso Regression

lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.legend(X.columns.values)

# Perform 10-fold cross-validation to choose the best alpha

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

# Find the optimal value of the regularization penality
lasso_best_alpha = lassocv.alpha_
print("The optimal value of the regularization penality of Lasso Regression: ", lasso_best_alpha)

# Fit the model with the best alpha value

lasso.set_params(alpha=lasso_best_alpha)
lasso.fit(X_train, y_train)
#print(mean_squared_error(y_test, lasso.predict(X_test)))

# Some of the coefficients are now reduced to exactly zero
print(pd.Series(lasso.coef_, index=X.columns))

# The optimal value of the regularization penality is about 2.24. 
# Within my train/test half-split dataset, Hits, Walks, and CRBI will be the final three predictors.
# There are total 6 predictors left in the model.

# 2. When pently was quite small, the variance of the model was large but the bias was small;
#    As pently increased from samll to large, the variance became samll but the bias became large. 