#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    predictedValues = X['population'].apply(lambda x: theta[0] + theta[1]*x)
    sumOfSquareErrors = (predictedValues - y['profit']).apply(lambda x: pow(x, 2)).sum()
    # Is there a more beautiful way to do that with pandas?
    cost = sumOfSquareErrors / (2*m)

    return cost


def gradientDescent(X, y, theta, alpha, num_iters, verbose=False):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    costHistory = []

    if verbose:
        print 'theta input ', theta
        print 'initial cost', computeCost(X, y, theta)
    for i in xrange(num_iters):
        #    % Instructions: Perform a single gradient step on the parameter vector
        #    %               theta.
        #    %
        #    % Hint: While debugging, it can be useful to print out the values
        #    %       of the cost function (computeCost) and gradient here.

        predictedValues = X['population'].apply(lambda x: theta[0] + theta[1]*x)

        # Okay, the column of ones would have made it a wee bit nicer here.
        theta[0] = theta[0] - alpha / m * (predictedValues - y['profit']).sum()
        theta[1] = theta[1] - alpha / m * ((predictedValues - y['profit']) * X['population']).sum()

        cost = computeCost(X, y, theta)
        costHistory.append(cost)

        if verbose:
            print '    %04i theta' % i, theta
            print '    %04i cost ' % i, cost

    return theta, Series(costHistory)


if __name__ == '__main__':
    # =================== Part 3: Gradient descent ===================
    print 'Running Gradient Descent ...'

    plt.close('all')

    data = pd.read_csv('ex1data1.txt')
    X = DataFrame(data['population'])  # Don't need dummy column of ones.
    y = DataFrame(data['profit'])

    theta = np.zeros(2)  # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    cost = computeCost(X, y, theta)
    print 'Initial cost =', cost

    # run gradient descent
    theta, costHistory = gradientDescent(X, y, theta, alpha, 3, verbose=True)

    # print theta to screen
    print 'Theta found by gradient descent, 0003 iter:', theta

    # run gradient descent
    theta, costHistory = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print 'Theta found by gradient descent, %04i iter:' % iterations, theta
    f1 = plt.figure()
    p1 = costHistory.plot()
    p1.axes.set_title('Evolution of cost')
    p1.axes.yaxis.label.set_text('Cost function')
    p1.axes.xaxis.label.set_text('Iteration')
    f1.show()

    # Plot the linear fit
    f2 = plt.figure()
    p2 = plt.scatter(data.population, data.profit, s=15, marker='x', label='Training data')
    p2.axes.set_title('Fit and Scatter')
    p2.axes.yaxis.label.set_text('Profit in $10,000s')
    p2.axes.xaxis.label.set_text('Population of City in 10,000s')

    fitLine = lambda x: theta[0] + theta[1]*x
    seq = np.arange(4, 24.9, 0.1)
    # Is this really how we plot functions in matplotlib? How primitive.
    plt.plot(seq, fitLine(seq), 'r', label='Linear regression')

    plt.legend(loc='upper left')

    f2.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5], theta)

    print 'For population = 35,000, we predict a profit of %.2f' % (predict1*10000)

    predict2 = np.dot([1, 7], theta)
    print 'For population = 70,000, we predict a profit of %.2f' % (predict2*10000)
