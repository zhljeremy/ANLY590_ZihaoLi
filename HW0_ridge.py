#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:17:00 2018

@author: zihaoli
"""

# 1. 

# import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
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

# The Ridge Regression

# define the alpha 

alphas = 10**np.linspace(4,-2,100)*0.5
alphas

# Set up and fit data into Ridge Regression 

ridge = Ridge(normalize = True)
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.legend(X.columns.values)

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Use cross-validation to choose the best ridge parameter alpha

ridgecv = RidgeCV(alphas = alphas, normalize = True)
ridgecv.fit(X_train, y_train)
best_ridge_alpha = ridgecv.alpha_
print("The optimal value of the regularization penality of Ridge Regression: ", best_ridge_alpha)

# Fit the model with the best alpha value 
ridge = Ridge(alpha = best_ridge_alpha, normalize = True)
ridge.fit(X_train, y_train)
#print(mean_squared_error(y_test, ridge3.predict(X_test)))

# none of the coefficients are exactly zero
ridge.fit(X, y)
print(pd.Series(ridge.coef_, index = X.columns))

# The optimal value of the regularization penality of Ridge Regression is about 0.57.

# 2. When pently was quite small, the variance of the model was large but the bias was small;
#    As pently increased from samll to large, the variance became samll but the bias became large. 