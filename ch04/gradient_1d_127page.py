import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성. 원소가 모두 0

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

# (3,4) (0,2), (3,0)에서의 기울기를 구하는 것

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# 6.0, 8.0
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# 0.,4.
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
# 6.0, 0.

# 여기서 의미하는 기울기는 그림으로 그려보면 이해가 된대요
# gradient_2d를 찾으면 될 듯