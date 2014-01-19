#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from helperFunctions import plotData, sigmoid, mapFeature, costFunction

## Machine Learning Online Class - Exercise 2: Logistic Regression
# Problem 1, Logistic Regression


def predict(theta, X):
    """
    Return predictions for a set of test scores.
    """

    p = sigmoid(np.dot(X_array, theta)) >= 0.5
    return p

if __name__ == '__main__':
    ## Initialization
    plt.close('all')

    ## Load Data
    #  The first two columns contains the exam scores and the third column
    #  contains the label.
    data = pd.read_csv('ex2data1.txt', header=None)
    X = data[[0, 1]]
    y = data[2]

    ## ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'

    figure = plotData(X, y)

    ## ============ Part 2: Compute Cost and Gradient ============

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = np.shape(X)

    # Add intercept term to x and X_test
    X_array = mapFeature(X)
    y_array = np.array(y)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X_array, y_array)

    print 'Cost at initial theta (zeros): %f' % cost
    print 'Gradient at initial theta (zeros):', grad
    print ''

    ## ============= Part 3: Optimizing  =============
    # From scipy.optimize, the minimize function looks to offer similar functionality
    # to fminunc.

    # Nelder-Mead works fine, and doesn't use the gradient.
    result_Nelder_Mead = minimize(lambda t: costFunction(t, X_array, y_array)[0],
                                  initial_theta, method='Nelder-Mead')

    # BFGS seems to get into a region where np.dot(X, theta) is large, so the sigmoid returns
    # 1.0 exactly, and the gradient goes infinite. Breaks.
    result_BFGS = minimize(lambda t: costFunction(t, X_array, y_array),
                           initial_theta, method='BFGS', jac=True)

    # Newton-CG works well even without the Hessian.
    # Per %timeit, about 3x faster than Nelder-Mead.
    result_Newton_CG = minimize(lambda t: costFunction(t, X_array, y_array),
                                initial_theta, method='Newton-CG', jac=True)

    # Print theta to screen
    print 'Cost at theta found by Nelder-Mead: %f' % result_Nelder_Mead['fun']
    print 'theta:', result_Nelder_Mead['x']
    print 'Cost at theta found by Newton-CG:   %f' % result_Newton_CG['fun']
    print 'theta:', result_Newton_CG['x']
    print ''
    theta = result_Nelder_Mead['x']

    # Plot Boundary
    plotData(X, y, theta)

    ## ============== Part 4: Predict and Accuracies ==============
    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2

    prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
    print 'For a student with scores 45 and 85, we predict an admission probability of',
    print '%.1f%%' % (prob * 100)

    # Compute accuracy on our training set
    p = predict(theta, X_array)

    print 'Train Accuracy: %.1f%%' % ((p == y_array).mean() * 100)
