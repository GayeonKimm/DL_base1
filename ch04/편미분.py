# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_tmp1(x0):
    return x0**2 + 4.0**2
print(numerical_diff(function_tmp1, 3.0))

# 차이가 없더라고 return 식도 다르게 써보니까 걍 같은 말이네
# 먼저
# print(3*3)
# print(3**2)

def function_tmp2(x1):
    return 3.0**2 + x1*x1
print(numerical_diff(function_tmp2, 4.0))
