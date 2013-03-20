#!/usr/bin/env python

#PLOTDATA Plots the data points x and y into a new figure 
#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
#   population and profit.

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the 
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the 
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

#figure; # open a new figure window
#X = data(:, 1); y = data(:, 2);
#m = length(y); # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
#plotData(X, y);

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt')
print data.head()
f = plt.figure()
p = plt.scatter(data.population,data.profit,s=15,marker='x')
p.axes.set_title('A Scatter Plot')
p.axes.yaxis.label.set_text('Profit in $10,000s')
p.axes.xaxis.label.set_text('Population of City in 10,000s')
# Is there some better way to do that in matplotlib?
# p.axes.yaxis.set_label_text('blah') works but is about the same length.
f.show()


# ============================================================
