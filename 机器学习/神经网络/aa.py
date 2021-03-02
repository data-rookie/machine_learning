"""
Spyder Editor
This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

x_data = [338, 333, 328, 207, 226,
          25, 179, 60, 208, 606]

y_data = [640, 633, 619, 393, 428,
          27, 193, 66, 226, 1591]

# 先隨便找一個點
b = -120
w = -4
lr = 1  # 學習率
iteration = 100000  # 計數器

b_history = [b]  # 所有的b參數
w_history = [w]  # 所有的w參數

lr_b = 0
lr_w = 0

for i in range(iteration):

    # 我要找這種解: y=wx+b, y為預測值, y'為實際值
    # 我定義公式:(實際值-預測值)**2
    # 就是(y'-(wx+b))**2
    # 10個點就是
    # L(w,b)=sig(10)[(y-(wx+b))**2)]
    b_grad = 0.0  # 新的b點位移預測
    w_grad = 0.0  # 新的w點位移預測

    for n in range(len(x_data)):
        # L(w,b)對b偏微分
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0

        # L(w,b)對w偏微分
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    # Adagrad 修改 learning rate
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    b = b - lr / np.sqrt(lr_b) * b_grad  # Adagrad
    w = w - lr / np.sqrt(lr_w) * w_grad

    b_history.append(b)
    w_history.append(w)

print("Mini point: b =", b_history[-1], "w =", w_history[-1])
plt.plot(b_history, w_history, 'ro')
plt.show()