import numpy as np
from sklearn import datasets

# iris = datasets.load_iris()
# print(iris.data.shape)
# print(np.cov(iris.data,rowvar=False))

# x = np.array([2,4,5,3,6,9,40,25,32])
# print(np.cov(x)*8)
# print(np.var(x)*9)
y = np.array([[1, 5, 6], [4, 3, 9], [4, 2, 9], [4, 7, 2]])  # 四行三列
print(y.shape)
print(np.cov(y, rowvar=False))
""""""""""""""
# 给定一组数据，计算有特征引导的协方差矩阵
""""""""""""""


def coVariance(X):  # 数据的每一行是一个样本，每一列是一个特征
    ro, cl = X.shape
    row_mean = np.mean(X, axis=0)
    X_Mean = np.zeros_like(X)
    X_Mean[:] = row_mean  # 把向量赋值给每一行
    X_Minus = X - X_Mean
    covarMatrix = np.zeros((cl, cl))
    for i in range(cl):
        for j in range(cl):
            a=X_Minus[:, i]
            b=X_Minus[:, j].T
            covarMatrix[i, j] = (X_Minus[:, i].dot(X_Minus[:, j].T)) / (ro - 1)
    return covarMatrix


cV = coVariance(y)
print(cV)

