#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from helperFunctions import plotData, sigmoid, mapFeature, costFunctionReg

## Machine Learning Online Class - Exercise 2: Logistic Regression
# Problem 2, Regularized Logistic Regression

if __name__ == '__main__':
    ## Initialization
    plt.close('all')

    ## Load Data
    #  The first two columns contains the X values and the third column
    #  contains the label (y).
    data = pd.read_csv('ex2data2.txt', header=None)
    X = data[[0, 1]]
    y = data[2]

    print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'
    figure = plotData(X, y)

    ## =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic
    #  regression to classify the data points.
    #
    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X_long = mapFeature(X, degree=6)
    y_long = np.array(y)

    # Initialize fitting parameters
    initial_theta = np.zeros(np.size(X_long[0]))

    # Set regularization parameter lambda to 1
    lamb = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    [cost, grad] = costFunctionReg(initial_theta, X_long, y_long, lamb)
    print 'Cost at initial theta (zeros): %f' % cost
    print 'Gradient at initial theta (zeros):'
    print grad
    print ''

    ## ============= Part 2: Regularization and Accuracies =============
    #  Optional Exercise:
    #  In this part, you will get to try different values of lambda and
    #  see how regularization affects the decision coundart
    #
    #  Try the following values of lambda (0, 1, 10, 100).
    #
    #  How does the decision boundary change when you vary lambda? How does
    #  the training set accuracy vary?

    # Optimize
    result_Newton_CG = minimize(lambda t: costFunctionReg(t, X_long, y_long, lamb),
                                initial_theta, method='Newton-CG', jac=True)
    theta = result_Newton_CG['x']

    # Plot Boundary
    f = plotData(X, y, theta)
    plt.title('lamb = %f' % lamb)

    # Compute accuracy on our training set
    #p = predict(theta, X)

    #print 'Train Accuracy: %' % mean(double(p == y)) * 100
