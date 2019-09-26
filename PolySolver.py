#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:36:24 2019

@author: sny
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization


#def poly(gamma,v):
#   
#   return [1, gamma**2/v+v, -2*gamma**2*np.log(v), gamma**2*v]
#
#def polyfit(x,a,b,c):
#   return a*x**2+b*x+c
#
#n=2
#glist = np.linspace(1,n,n*10)
#sollist = [min(np.roots(poly(x,0.01))) for x in glist]
#
#x0 = np.array([-1, 0, 1])
#ans, cov = optimization.curve_fit(polyfit, glist, sollist, x0)
#f = [polyfit(x, ans[0], ans[1], ans[2]) for x in glist]
#print(cov)
#plt.plot(glist, f)
#plt.plot(glist, sollist)

def poly(gamma,v):
   
   return [1, gamma**2/v+v, -2*gamma**2*np.log(v), gamma**2*v]

def polyfit(x,a,b,c):
   return a*x**2+b*x+c

def logfit(x,a,b,c):
   return -(a/x+b)-c*x

n=100
glist = np.linspace(0.01,10,n)
sollist = [np.real(min(np.roots(poly(1,x)))) for x in glist]
print(sollist)
x0 = np.array([1, 1, 1])
ans, cov = optimization.curve_fit(logfit, glist, sollist, x0)


f = [logfit(x, ans[0], ans[1], ans[2]) for x in glist]
print(cov)
plt.plot(glist, f)
plt.plot(glist, sollist)