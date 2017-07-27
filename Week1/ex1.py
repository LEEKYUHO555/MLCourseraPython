# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:29:15 2017

@author: snu13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CostComputation as cc


data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:,0]
Y = data[:,1]
m = len(Y)
X = np.reshape(X,(m,1))
Y = np.reshape(Y,(m,1))

#Plotting data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X,Y)
ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000s')

#Cost and Gradient descent
X1 = np.concatenate((np.ones((m,1), float),X), axis=1) #adding bias
theta = np.zeros((2,1), float)

iterations = 1500
alpha = 0.01

J = cc.computeCost(X1,Y,theta)

print('With theta = [0 ; 0]\nCost computed = %f' %J)
print('Expected cost value (approx) 32.07\n')

theta = cc.gradientDescent(X1, Y, theta, alpha, iterations)

print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664\n')

#Plotting Linear Regression
ax.plot(X,np.matmul(X1,theta), 'r')