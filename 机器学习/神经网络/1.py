import pandas as pd
import numpy as np
import random as rd

import matplotlib.pyplot as plt


D=[]

def dataload(x):
    data=pd.read_csv(x,header=0,index_col=0,encoding='gbk')#必须要用这两个关键字，要不然它会把数据当成行来读取
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
            train=train.append(data.loc[select_train[i]])#这里一定要赋值。。。要不然就等于没有加，它这个跟list的append是不同的
        for i in range(len(select_test)):
            test=test.append(data.loc[select_test[i]])
        train = train.values
        test = test.values
        return train,test
    else:
        mid1=data.head(int(len(data)*train_size))
        mid2=data.tail(int(len(data)*(1-train_size)))
        mid1 = mid1.values
        mid2 = mid2.values
        return mid1,mid2

def lable_split(data,have):
    data_lable=data[:,data.shape[1]-1]
    data=data[:,0:data.shape[1]-1]
    lab=np.zeros((len(data_lable),len(have)))
    for i in range(len(data_lable)):
        lab[i][data_lable[i]]=1
    return data,lab

def classes(data):
    have=[]
    for i in range(data.shape[0]):
        if data[i][data.shape[1]-1] not in have:#这个好像是先列数后行数,这个跟前面不一样，一个是[:,:],一个是[][],这个要注意,前面的是指在dataframe里面，如果是np的格式可以直接用的
            have.append( data[i][data.shape[1]-1])
    for i in range(data.shape[0]):#顺便把它的字母什么的给改一下，这样规范好处理一点,这个也算预处理的部分吧
        for j in range(len(have)):
            if  data[i][data.shape[1]-1]==have[j]:
                data[i][data.shape[1] - 1]=j
    return have

def lisan(data):
    for j in range(data.shape[1]-3):#最后几列不能算
        dic={}
        for i in range(data.shape[0]):
            mid=list(dic.keys())
            if data[i,j] not in mid:
                dic[data[i,j]]=len(dic)
                data[i,j]=dic[data[i,j]]
            else:
                data[i,j]=dic[data[i,j]]
        D.append(dic)
    return data

def classes_(data,have):
    for i in range(data.shape[0]):
        m=have.index(data[i,-1])
        data[i,-1]=m
    return data

data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习\神经网络\data\watermelon_3.csv')
print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')

train_,test_=data_split(data,0.8,False)
data=data.values
have=classes(data)
lisan(train_)#这几个是为了规范化数据
lisan(test_)
classes_(train_,have)
classes_(test_,have)
print('训练集的个数为：', train_.shape[0])
print('测试集的个数为：', test_.shape[0])
train_var, train_lab = lable_split(train_,have)
test_var, test_lab = lable_split(test_,have)




m, n = np.shape(train_var)

# according to P101, init the parameters
import random

d = n
l = 2
q = d + 1
theta = [random.random() for i in range(l)]
theta=np.array(theta)
gamma = [random.random() for i in range(q)]
gamma=np.array(gamma)
v = [[random.random() for i in range(q)] for j in range(d)]
v=np.array(v)
w = [[random.random() for i in range(l)] for j in range(q)]
w=np.array(w)
eta = 0.2
maxIter = 5000

import math

def predict(iX):
    alpha = np.dot(iX, v)
    b = sigmoid(alpha - gamma, 2)
    beta = np.dot(b, w)
    predictY = sigmoid(beta - theta, 2)
    return predictY

EE1=[]
EE2=[]
XX=[i for i in range(5000)]
def sigmoid(iX, dimension):  # iX is a matrix with a dimension
    if dimension == 1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(iX.shape[0]):
            iX[i] = sigmoid(iX[i],1)
    return iX


# do the repeat----standard BP
while (maxIter > 0):
    maxIter -= 1
    sumE = 0
    for i in range(m):
        alpha = np.dot(train_var[i], v)
        b = sigmoid(alpha - gamma, 1)
        beta = np.dot(b, w)
        predictY = sigmoid(beta - theta, 1)
        E = sum((predictY - train_lab[i]) * (predictY - train_lab[i])) / 2
        sumE += E
        g = predictY * (1 - predictY) * (train_lab[i] - predictY)
        e = b * (1 - b) * ((np.dot(w, g.T)).T)
        w =w + eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))
        theta = theta - eta * g
        v =v + eta * np.dot(train_var[i].reshape((d, 1)), e.reshape((1, q)))
        gamma =gamma - eta * e
    EE1.append(E)
    # print(sumE)
a=predict(test_var)

d = n
l = 2
q = d + 1
theta = [random.random() for i in range(l)]
theta=np.array(theta)
gamma = [random.random() for i in range(q)]
gamma=np.array(gamma)
v = [[random.random() for i in range(q)] for j in range(d)]
v=np.array(v)
w = [[random.random() for i in range(l)] for j in range(q)]
w=np.array(w)
eta = 0.2
maxIter = 5000

#accumulated BP
train_lab=train_lab.reshape((m,l))
while(maxIter>0):
    maxIter-=1
    sumE=0
    alpha = np.dot(train_var, v)
    b = sigmoid(alpha - gamma,2)
    beta = np.dot(b, w)
    predictY = sigmoid(beta - theta,2)

    E = sum(sum((predictY - train_lab) * (predictY - train_lab))) / 2
    g = predictY * (1 - predictY) * (train_lab - predictY)
    e = b * (1 - b) * ((np.dot(w, g.T)).T)
    w =w+ eta * np.dot(b.T, g)
    theta =theta- eta * g
    v =v+ eta * np.dot(train_var.T, e)
    gamma =gamma- eta * e
    EE2.append(E)






plt.plot(XX, EE1)
plt.plot(XX, EE2)
plt.show()

# print(predict(test_var))
























