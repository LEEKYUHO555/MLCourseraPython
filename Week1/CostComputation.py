# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 23:41:03 2017

@author: snu13
"""

import numpy as np

def computeCost(X, Y, theta):
    m = len(Y);
    sub = np.matmul(X,theta) - Y
    return np.matmul(np.transpose(sub), sub) / (2*m)
    
def gradientDescent(X, Y, theta, alpha, iterations):
    
    m = len(Y)
    #J_history = np.zeros(iterations,1)
    
    for iter in range(0,iterations):            
        temp = np.matmul(X, theta) - Y        
        theta = theta - alpha / m * np.transpose(np.matmul(np.transpose(temp),X))
        #J_history(iter,1) = computeCost(X, Y, theta)
    
    return theta
        
        
        
        