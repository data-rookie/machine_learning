import pandas as pd
import random as rd
import numpy as np
import math
import random
from sklearn import svm
import matplotlib.pyplot as plt


def dataload(x):
    data=pd.read_csv(x,header=None,index_col=0,encoding='gbk')#必须要用这两个关键字，要不然它会把数据当成行来读取
    #这里的index_col要按照数据集给出来的来看，是0还是none，不同的数据集是不一样的，比如那个西瓜的和UCI的数据集
    #https://blog.csdn.net/weixin_44056331/article/details/89366105
    return data

def data_split(data,train_size=0.9,random=False):
    if random:
        select_train=rd.sample(range(0,len(data)-1),int(len(data)*train_size))
        all=[x for x in range(len(data))]
        select_test=[item for item in all if item not in select_train]
        select_train.sort()
        select_test.sort()
        name=list(data)
        train=pd.DataFrame(columns=(name))
        test=pd.DataFrame(columns=(name))
        for i in range(len(select_train)):
            train=train.append(data.iloc[select_train[i]])#这里一定要赋值。。。要不然就等于没有加，它这个跟list的append是不同的
        for i in range(len(select_test)):
            test=test.append(data.iloc[select_test[i]])
        train = train.values
        test = test.values
        return train,test
    else:
        mid1=data.head(int(len(data)*train_size))
        mid2=data.tail(int(len(data)*(1-train_size)))
        mid1 = mid1.values
        mid2 = mid2.values
        return mid1,mid2

def lable_split(data):
    data_lable=data[:,data.shape[1]-1]
    data=data[:,0:data.shape[1]-1]
    return data,data_lable

def _violate_KKT_conditions(alpha,i,train_y,_E):
    """ Check if an example violates the KKT conditons """
    alpha_i = alpha[i]
    R_i = train_y[i]*_E(i)
    return (R_i < -tol and alpha_i < C) or (R_i > tol and alpha_i > 0)

def _simple_smo(train_X, ):
    num_changed_alphas = 1
    iteration = 0

    while num_changed_alphas > 0:
        num_changed_alphas = 0
        for i in range(len(train_X)):
            if _violate_KKT_conditions(i):
                j = i
                while(j == i): j = np.random.randint(0, len(train_X))
                num_changed_alphas += self._update_alpha_pair(i, j)

        if self.verbose and num_changed_alphas == 0:
            if info: print('[*] {}'.format(info))
            print('[*] Converged at iteration {}.'.format(iteration+1))
            print('-'*100)

        iteration += 1
        if self.verbose and (iteration == 1 or iteration % 100 == 0 or iteration == self.max_iteration):
            # Compute training and testing error
            train_error = self.scorer(y_truth=self.train_y, y_pred=self.hypothesis(X=self.train_X))
            test_error = self.scorer(y_truth=self.test_y, y_pred=self.hypothesis(X=self.test_X))
            print('-'*100)
            if info: print('[*] {}'.format(info))
            print('[*] {} alphas changed.'.format(num_changed_alphas))
            print('[*] Iteration: {} | Train error: {} | Test error: {}'.format(iteration, train_error, test_error))

        if iteration == self.max_iteration:
            print('-'*100)
            print('[*] Max iteration acheived.')
            break

    if self.verbose: print('[*] Averaging post-computed biases as final bias of SVM hypothesis.')
    self._postcompute_biases()

 def _update_alpha_pair(self, i, j):
        """ Jointly optimized alpha_i and alpha_j """
        # Not the alpha pair.
        if i == j: return 0

        E_i = self._E(i)
        E_j = self._E(j)

        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]

        x_i, x_j, y_i, y_j = self.train_X[i], self.train_X[j], self.train_y[i], self.train_y[j]

        if y_i == y_j:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        else:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)

        # This will not make any progress.
        if L == H: return 0

        # Compute eta (second derivative of the Lagrange dual function = -eta)
        if self.enable_kernel_cache:
            eta = self.kernel_cache[i][i] + self.kernel_cache[j][j] - 2*self.kernel_cache[i][j]
        else:
            eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2*self.kernel(x_i, x_j)

        # eta > 0 => second derivative(-eta) < 0 => maximum exists.
        if eta <= 0: return 0

        # Compute new alpha_j and clip it inside [L, H]
        alpha_j_new = alpha_j + y_j*(E_i - E_j)/eta
        if alpha_j_new < L: alpha_j_new = L
        if alpha_j_new > H: alpha_j_new = H

        # Compute new alpha_i based on new alpha_j
        alpha_i_new = alpha_i + y_i*y_j*(alpha_j - alpha_j_new)

        # Compute step sizes
        delta_alpha_i = alpha_i_new - alpha_i
        delta_alpha_j = alpha_j_new - alpha_j

        # Update weight vector
        if self.use_w:
            self.w = self.w + delta_alpha_i*y_i*x_i + delta_alpha_j*y_j*x_j

        # Update b
        if self.enable_kernel_cache:
            b_i = self.b - E_i - delta_alpha_i*y_i*self.kernel_cache[i][i] - delta_alpha_j*y_j*self.kernel_cache[i][j]
            b_j = self.b - E_j - delta_alpha_i*y_i*self.kernel_cache[i][j] - delta_alpha_j*y_j*self.kernel_cache[j][j]
        else:
            b_i = self.b - E_i - delta_alpha_i*y_i*self.kernel(x_i, x_i) - delta_alpha_j*y_j*self.kernel(x_i, x_j)
            b_j = self.b - E_j - delta_alpha_i*y_i*self.kernel(x_i, x_j) - delta_alpha_j*y_j*self.kernel(x_j, x_j)
        self.b = (b_i + b_j)/2
        if (alpha_i_new > 0 and alpha_i_new < self.C):
            self.b = b_i
        if (alpha_j_new > 0 and alpha_j_new < self.C):
            self.b = b_j

        # Update the alpha pair
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new

        return 1


data=dataload(r'.\data\watermelon_33a.csv')

train_,test_=data_split(data,0.8,False)

train_var, train_lab = lable_split(train_)
test_var, test_lab = lable_split(test_)

C=100
kernel_type='linear'
tol=0.001
epsilon=0.1




































