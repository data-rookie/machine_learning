
import pandas as pd
import numpy as np
import random as rd
import random

import math
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

D=[]

def dataload(x):
    data=pd.read_csv(x,header=0,index_col=None,encoding='gbk',sep=';')#必须要用这两个关键字，要不然它会把数据当成行来读取
    #这里的index_col要按照数据集给出来的来看，是0还是none，不同的数据集是不一样的，比如那个西瓜的和UCI的数据集
    #https://blog.csdn.net/weixin_44056331/article/details/89366105
    return data


def data_split(data,train_size=0.9,random=False):
    if random:
        select_train=rd.sample(range(0,len(data)),int(len(data)*train_size))
        all=[x for x in range(0,len(data))]
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
        lab[i][have.index(data_lable[i])]=1
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




def sum_of_juzhen(x):
    result=0
    for i in range(len(x)):
        result=result+x[i][i]
    return result

def predict(iX):
    alpha = np.dot(iX, v)
    b = sigmoid(alpha - gamma, 6)
    beta = np.dot(b, w)
    predictY = sigmoid(beta - theta, 6)
    return predictY

def sigmoid(iX, dimension):  # iX is a matrix with a dimension
    if dimension == 1:
        for i in range(len(iX)):
            a=iX[i]
            a=math.exp(a)
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        # for j in range(iX.shape[0]):
        for i in range(iX.shape[0]):
            iX[i] = sigmoid(iX[i],1)
    return iX

def normalization(x):
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def chang(data,have):
    c=np.zeros((len(data),len(have)))
    for i in range(len(data)):
        c[i][have.index(data[i])]=1
    return c

def ac(y_pred,y):
    y_p=np.zeros((y_pred.shape[0],len(have)))
    for i in range(y_pred.shape[0]):
        x=list(y_pred[i])
        y_p[i][x.index(max(x))]=1
    result=0
    for i in range(len(y)):
        yy=list(y[i])
        yy_p=list(y_p[i])
        if yy.index(max(yy))==yy_p.index(max(yy_p)):
            result=result+1
    return result/len(y)


EE1=[]
EE2=[]
EE3=[]
EE4=[]
EE5=[]
EE6=[]

long=100
XX=[i for i in range(long)]




data=dataload(r'.\data\winequality-red.csv')
print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')

train_,test_=data_split(data,0.7,True)
data=data.values
have=classes(data)
# lisan(train_)#这几个是为了规范化数据
# lisan(test_)
# classes_(train_,have)
# classes_(test_,have)
print('训练集的个数为：', train_.shape[0])
print('测试集的个数为：', test_.shape[0])
train_var, train_lab = lable_split(train_,have)
test_var, test_lab = lable_split(test_,have)

train_var=normalization(train_var)
test_var=normalization(test_var)

m, n = np.shape(train_var)
#----------------------------------------------------------------------------------------------------------------------------------------------------

# according to P101, init the parameters

d = n
l = len(have)
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
maxIter = 400

# do the repeat----standard BP
while (maxIter > 0):
    maxIter -= 1
    sumE = 0
    for i in range(m):
        alpha = np.dot(train_var[i], v)
        b = sigmoid(alpha - gamma, 1)
        beta = np.dot(b, w)
        predictY = sigmoid(beta - theta, 1)
        # aa=(predictY - train_lab[i])
        E = sum((predictY - train_lab[i]) * (predictY - train_lab[i])) / 2
        sumE += E
        g = predictY * (1 - predictY) * (train_lab[i] - predictY)
        e = b * (1 - b) * ((np.dot(w, g.T)).T)
        w =w + eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))
        theta = theta - eta * g
        v =v + eta * np.dot(train_var[i].reshape((d, 1)), e.reshape((1, q)))
        gamma =gamma - eta * e
    if len(EE1)<long:
        EE1.append(sumE)
    # print(sumE)
#
aaa=predict(test_var)
print('ac',ac(aaa,test_lab))
print('a')


