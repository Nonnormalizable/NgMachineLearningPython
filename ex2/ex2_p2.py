#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from helperFunctions import plotData, sigmoid

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

    print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'
    figure = plotData(X, y)

# # Put some labels 
# hold on;
# 
# # Labels and Legend
# xlabel('Microchip Test 1')
# ylabel('Microchip Test 2')
# 
# # Specified in plot order
# legend('y = 1', 'y = 0')
# hold off;
# 
# 
# ## =========== Part 1: Regularized Logistic Regression ============
# #  In this part, you are given a dataset with data points that are not
# #  linearly separable. However, you would still like to use logistic 
# #  regression to classify the data points. 
# #
# #  To do so, you introduce more features to use -- in particular, you add
# #  polynomial features to our data matrix (similar to polynomial
# #  regression).
# #
# 
# # Add Polynomial Features
# 
# # Note that mapFeature also adds a column of ones for us, so the intercept
# # term is handled
# X = mapFeature(X(:,1), X(:,2));
# 
# # Initialize fitting parameters
# initial_theta = zeros(size(X, 2), 1);
# 
# # Set regularization parameter lambda to 1
# lambda = 1;
# 
# # Compute and display initial cost and gradient for regularized logistic
# # regression
# [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
# 
# fprintf('Cost at initial theta (zeros): #f\n', cost);
# 
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
# 
# ## ============= Part 2: Regularization and Accuracies =============
# #  Optional Exercise:
# #  In this part, you will get to try different values of lambda and 
# #  see how regularization affects the decision coundart
# #
# #  Try the following values of lambda (0, 1, 10, 100).
# #
# #  How does the decision boundary change when you vary lambda? How does
# #  the training set accuracy vary?
# #
# 
# # Initialize fitting parameters
# initial_theta = zeros(size(X, 2), 1);
# 
# # Set regularization parameter lambda to 1 (you should vary this)
# lambda = 1;
# 
# # Set Options
# options = optimset('GradObj', 'on', 'MaxIter', 400);
# 
# # Optimize
# [theta, J, exit_flag] = ...
# 	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
# 
# # Plot Boundary
# plotDecisionBoundary(theta, X, y);
# hold on;
# title(sprintf('lambda = #g', lambda))
# 
# # Labels and Legend
# xlabel('Microchip Test 1')
# ylabel('Microchip Test 2')
# 
# legend('y = 1', 'y = 0', 'Decision boundary)
# hold off;
# 
# # Compute accuracy on our training set
# p = predict(theta, X);
# 
# fprintf('Train Accuracy: #f\n', mean(double(p == y)) * 100);


