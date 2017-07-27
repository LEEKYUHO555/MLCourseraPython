# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 20:37:55 2017

@author: snu13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:,:-1]
Y = data[:,-1]
m = len(Y)
#X = np.reshape(X,(m,1))
#Y = np.reshape(Y,(m,1))

#Plotting data
"""fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X,Y)
ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000s')

#Cost and Gradient descent
X1 = np.concatenate((np.ones((m,1), float),X), axis=1) #adding bias
theta = np.zeros((2,1), float)"""