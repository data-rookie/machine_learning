import pandas as pd
import random as rd
import numpy as np
import math
import random
from svr import SVR
from sklearn import svm
import matplotlib.pyplot as plt
'''
以西瓜数据集3.0a的密度为输入，含糖率为输出，
训练一个SVR.
'''

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


data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习\SVM\data\watermelon_33a.csv')


train_,test_=data_split(data,0.8,False)

train_var, train_lab = lable_split(train_)

plt_x=[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657]#,0.36,0.593,0.719
plt_y=[0.46,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.0267,0.057,0.099,0.161,0.198]#,0.37,0.042,0.103
plt.scatter(plt_x, plt_y, alpha=0.4)

test_var, test_lab = lable_split(test_)


#自己的源代码
param_grid = {
    'C': 100,
    'kernel_type':'linear',
    'tol': 0.001,
    'epsilon': 0.1
}
data={}
data['train_X']=train_var
data['train_y']=train_lab
data['test_X']=test_var
data['test_y']=test_lab
_model = SVR(data=data, param=param_grid)
_model.train()
print(_model.w)
print(_model.b)

# p_x=np.arange(0, 0.8, 0.01)
# p_y=p_x*_model.w+_model.b
# plt.plot(p_x, p_y)



# svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
#调库
svr_linear = svm.SVR(kernel='linear', C=100)
svr_linear.fit(train_var, train_lab)
print(svr_linear)
print(svr_linear.support_vectors_)
print(svr_linear.support_)#这个就是支持向量在训练集中的标号
print(svr_linear.coef_)
print(svr_linear.intercept_)
print('调库用linear的预测值为：',svr_linear.predict(test_var))
p_x=[[x/100] for x in range(0,80,1)]
p_y=svr_linear.predict(p_x)
plt.plot(p_x, p_y,c='red')

# spx=[0.697,0.403,0.666,0.639]
# spy=[0.46,0.237,0.091,0.161]
# plt.scatter(spx, spy, alpha=0.4,c='red')

# test_var=[0.36,0.593,0.719]
svr_linear = svm.SVR(kernel='poly', C=100)
svr_linear.fit(train_var, train_lab)
print('调库用poly的预测值为：',svr_linear.predict(test_var))
# p_x=np.arange(0, 0.8, 0.01)
p_x=[[x/100] for x in range(0,80,1)]
p_y=svr_linear.predict(p_x)
plt.plot(p_x, p_y,c='blue')


svr_linear = svm.SVR(kernel='rbf', C=100)
svr_linear.fit(train_var, train_lab)
print('调库用rbf的预测值为：',svr_linear.predict(test_var))
print('a')
# p_x=np.arange(0, 0.8, 0.01)
p_x=[[x/100] for x in range(0,80,1)]
p_y=svr_linear.predict(p_x)
plt.plot(p_x, p_y,c='green')

test_var=[0.36,0.593,0.719]
print('真实值为：',0.37,0.042,0.103)
print('自己模型预测的值:',_model.w*test_var+_model.b)

plt.show()














