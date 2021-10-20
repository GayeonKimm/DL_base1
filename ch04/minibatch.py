import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10) 원핫인코딩으로 정답만 표기했기 때문에

# 무작위로 10장만 빼내려면?
train_size = x_train.shape
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 원핫인코딩의 정답 레이블에서의 교차 엔트로피 오차
def cross_entropy(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum( t* np.log(y + 1e-7)) / batch_size

# 정답 레이블이 원핫인코딩이 아니라 2나 7등의 숫자 레이블로 주어졌을때의
# 교차 엔트로피 오차

def cross_entropy(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t ] + 1e-7)) / batch_size
