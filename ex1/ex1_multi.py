#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


## Helper functions
def featureNormalize(X):
    """
    Make all features have mean 0 and std 1. Use pandas!
    """
    # Goodness, this is easy with pandas.
    X_norm = (X - X.mean())/X.std()
    mu = X.mean()
    sigma = X.std()
    return X_norm, mu, sigma


def predictValues(X, theta):
    """
    With X and theta as numpy arrays, finding y for all m samples is just
    a dot product.
    """
    predictedValues = np.dot(X, theta)
    return predictedValues


def computeCostMulti(X, y, theta):
    """
    Compute the cost function with numpy arrays as imput.
    """
    m = len(y)
    predictedValues = predictValues(X, theta)
    sumOfSquareErrors = np.square(predictedValues-y).sum()
    cost = sumOfSquareErrors / (2*m)
    return cost


def gradientDescentMulti(X, y, theta, alpha, num_iters, verbose=False):
    """
    Takes numpy arrays and does the gradient descent.
    """
    m = len(y)
    costHistory = []

    if verbose:
        print 'theta input ', theta
        print 'initial cost %e' % computeCostMulti(X, y, theta)

    for i in xrange(num_iters):
        predictedValues = predictValues(X, theta)
        theta = theta - alpha / m * np.dot((predictedValues - y), X)

        cost = computeCostMulti(X, y, theta)
        costHistory.append(cost)

        if verbose:
            print '    %04i theta' % i, theta
            print '    %04i cost %e' % (i, cost)
    return theta, Series(costHistory)


def normalEqn(X, y):
    """
    Find the closed form solution using numpy matrix methods.
    """
    xtx = np.dot(X.transpose(), X)
    pinv = np.linalg.pinv(xtx)
    theta = np.dot(pinv, np.dot(X.transpose(), y))
    theta = theta.flatten()

    return theta


if __name__ == '__main__':

    ## Initialization

    ## ================ Part 1: Feature Normalization ================

    ## Clear and Close Figures
    plt.close('all')

    print 'Loading data ...'

    ## Load Data
    data = pd.read_csv('ex1data2.txt', header=None,
                       names=['size', 'rooms', 'price'])
    X = data[['size', 'rooms']]
    y = data[['price']]
    m = len(y)

    # Print out some data points
    print 'First 10 examples from the dataset:'
    print data.head(10)

    # Scale features and set them to zero mean
    print 'Normalizing Features ...'

    X, mu, sigma = featureNormalize(X)

    # Add intercept term to X
    X['ones'] = np.ones(m)
    X_array = np.array(X[['ones', 'size', 'rooms']])
    y_array = np.array(y).flatten()

    ## ================ Part 2: Gradient Descent ================

    print 'Running gradient descent ...'

    # Choose some alpha value
    alpha = 0.01
    num_iters = 1000

    # Init Theta and Run Gradient Descent
    theta_grad = np.zeros(3)
    theta_grad, costHistory = gradientDescentMulti(X_array, y_array, theta_grad,
                                                   alpha, num_iters,
                                                   verbose=False)

    # Plot the convergence graph
    print 'Theta found by gradient descent, %04i iter:' % num_iters
    print theta_grad
    f1 = plt.figure()
    p1 = costHistory.plot()
    p1.axes.set_title('Evolution of cost')
    p1.axes.yaxis.label.set_text('Cost function')
    p1.axes.xaxis.label.set_text('Iteration')
    f1.show()

    # Estimate the price of a 1650 sq-ft, 3 br house
    X1 = np.array([1,
                   (1650-mu['size'])/sigma['size'],
                   (3-mu['rooms'])/sigma['rooms']])

    price_grad = predictValues(X1, theta_grad)

    print 'Predicted price of a 1650 sq-ft, 3 br house '\
        '(using gradient descent): \n$%.2f\n' % price_grad

    ## ================ Part 3: Normal Equations ================

    print 'Solving with normal equations...'

    # Calculate the parameters from the normal equation
    theta_norm = normalEqn(X_array, y_array)

    # Display normal equation's result
    print 'Theta computed from the normal equations:'
    print theta_norm

    # Estimate the price of a 1650 sq-ft, 3 br house
    price_norm = np.dot(X1, theta_norm)

    print 'Predicted price of a 1650 sq-ft, 3 br house '\
        '(using normal equations): \n$%.2f\n' % price_norm

    print 'Fractional difference: %.2f%%' %\
        ((price_norm - price_grad)/((price_grad+price_norm)/2)*100)
