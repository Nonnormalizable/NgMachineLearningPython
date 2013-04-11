#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

## Machine Learning Online Class - Exercise 2: Logistic Regression

def plotData(X, y):
    """
    Plots the data points X and y into a new figure.
    With + for the positive examples and o for the negative examples.
    X is assumed to be a mx2 DataFrame and y a m len Series.
    """

    f = plt.figure()
    p = plt.scatter(X[y==0][0], X[y==0][1], marker='o', c='y', label='Not admitted')
    plt.scatter(X[y==1][0], X[y==1][1], marker='+', s=30, label='Admitted')
    p.axes.yaxis.label.set_text('Exam 2 score')
    p.axes.xaxis.label.set_text('Exam 1 score')
    plt.legend(loc='upper right')
    f.show()
    
    return


def sigmoid(z):
    """
    Compute sigmoid functoon
    Compute the sigmoid of each value of z (z can be a matrix, vector, or scalar).
    Accepts a scalar object, numpy array, Series, or DataFrame.
    """

    g = 1 / (1 + np.exp(-1*z))

    return g


def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression
    COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    """

    m = len(y)*1.0

    cost = 1/m * (
        np.dot(-1*y, np.log(sigmoid(np.dot(X, theta))))
        - np.dot(1 - y, np.log(1 - sigmoid(np.dot(X, theta))))
        )

    grad = 1/m * np.dot(sigmoid(np.dot(X, theta)) - y, X)

    return cost, grad


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

    print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.'

    plotData(X, y)

    ## ============ Part 2: Compute Cost and Gradient ============

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = np.shape(X)

    # Add intercept term to x and X_test
    X['ones'] = np.ones(m)
    X_array = np.array(X[['ones', 0, 1]])
    y_array = np.array(y)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X_array, y_array)

    print 'Cost at initial theta (zeros): %f' % cost
    print 'Gradient at initial theta (zeros):', grad


#    ## ============= Part 3: Optimizing using fminunc  =============
#    #  In this exercise, you will use a built-in function (fminunc) to find the
#    #  optimal parameters theta.
#
#    #  Set options for fminunc
#    options = optimset('GradObj', 'on', 'MaxIter', 400)
#
#    #  Run fminunc to obtain the optimal theta
#    #  This function will return theta and the cost 
#    #[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options)
#
#    # Print theta to screen
#    print 'Cost at theta found by fminunc: %f' % cost
#    print 'theta:'
#    print ' %f' % theta
#
#    # Plot Boundary
#    plotDecisionBoundary(theta, X, y)
#
#    # Put some labels 
#    #hold on
#    # Labels and Legend
#    xlabel('Exam 1 score')
#    ylabel('Exam 2 score')
#
#    # Specified in plot order
#    legend('Admitted', 'Not admitted')
#    #hold off
#
#    print '\nProgram paused. Press enter to continue.'
#    pause
#
#    ## ============== Part 4: Predict and Accuracies ==============
#    #  After learning the parameters, you'll like to use it to predict the outcomes
#    #  on unseen data. In this part, you will use the logistic regression model
#    #  to predict the probability that a student with score 45 on exam 1 and 
#    #  score 85 on exam 2 will be admitted.
#    #
#    #  Furthermore, you will compute the training and test set accuracies of 
#    #  our model.
#    #
#    #  Your task is to complete the code in predict.m
#
#    #  Predict probability for a student with score 45 on exam 1 
#    #  and score 85 on exam 2 
#
#    #prob = sigmoid([1 45 85] * theta)
#    print 'For a student with scores 45 and 85, we predict an admission probability of %f\n' % prob
#
#    # Compute accuracy on our training set
#    p = predict(theta, X)
#
#    print 'Train Accuracy: %f' % (mean(double(p == y)) * 100)


