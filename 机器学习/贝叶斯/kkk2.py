


import pandas as pd
import random as rd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from graphviz import Digraph

def dataload(x):
    data=pd.read_csv(x,header=0,index_col=0,encoding='gbk')#必须要用这两个关键字，要不然它会把数据当成行来读取
    #这里的index_col要按照数据集给出来的来看，是0还是none，不同的数据集是不一样的，比如那个西瓜的和UCI的数据集
    #https://blog.csdn.net/weixin_44056331/article/details/89366105
    return data

def data_split(data,train_size=0.9,random=False):
    if random:
        select_train=rd.sample(range(1,len(data)-1),int(len(data)*train_size))
        all=[x for x in range(1,len(data)+1)]
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

def lable_split(data):
    data_lable=data[:,data.shape[1]-1]
    data=data[:,0:data.shape[1]-1]
    return data,data_lable

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

def classes_split(data,have):
    all_class=[0 for i in range(len(have))]
    for i in range(data.shape[0]):
        for j in range(len(have)):
            if int(data[i,data.shape[1]-1])==j:
                if all_class[j] is 0:#这里不能用==，会有二义性。。。好像是因为一个矩阵不能用==来跟0来比较
                    all_class[j]=data[i,0:data.shape[1]-1]
                else:
                    mid1= all_class[j]
                    mid2= data[i,0:data.shape[1]-1]
                    all_class[j]=np.vstack((mid1,mid2))
    return all_class


def classes_(data,have):
    for i in range(data.shape[0]):
        m=have.index(data[i,-1])
        data[i,-1]=m
    return data


def split_by_characteristic(data,var_index):#这里按照变量来对数据集进行划分,var_index是变量的位置不是名字
    all_index={}
    data=data.reshape(-1, 9)
    for i in range(len(data)):
        if data[i,var_index] not in all_index.keys():
            all_index[data[i,var_index]]=data[i]
        else:
            all_index[data[i,var_index]]=np.vstack((all_index[data[i,var_index]],data[i]))
    return all_index

def ac(y_pred,y):
    result=0
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            result=result+1
    return result/len(y)

def fit_condition(data,tell1,index,tell2):#1是好坏瓜，index是属性的位置，tell2是属性的名字
    result=[]
    for i in range(len(data)):
        if data[i][-1]==tell1 and data[i][index]==tell2:
            result.append(data[i])
    result=np.array(result)
    return result

def f(u,s,x):
    return 1 / (math.sqrt(2 * math.pi) * s) * math.exp(-(x - u) ** 2 / (2 * s ** 2))


def pred(data,p,all_var):
    pr=[]
    for i in range(len(data)):
        most_P=[0 for ii in range(len(have))]
        for j in range(len(have)):
            mid_p=0
            for jj in range(5):
                lab=str(j)+str(jj)+data[i][jj]
                mid_pp=p[lab]['p']
                for k in range(len(all_var)):
                    if k<=5:
                        mid_pp=mid_pp*p[lab]['pp'][k][data[i][k]]
                    else:
                        try:
                            mid_p=mid_p*f(p[lab]['pp'][k]['u'],p[lab]['pp'][k]['s'],data[i][k])
                        except:
                            mid_p=mid_p
                mid_p=mid_p+mid_pp
            most_P[j]=mid_p
        if most_P[0]>most_P[1]:#这里先写一个两个类的，如果是多个类的，还需要对其进行完善
            pr.append(0)
        else:
            pr.append((1))
    return pr


data=dataload(r'.\data\watermelon_3.csv')
print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')
train_,test_=data_split(data,0.7,True)
train_var, train_lab = lable_split(train_)
test_var, test_lab = lable_split(test_)

all_var=[column for column in data]#把所有变量的名称先提取出来
aa={}
for i in range(len(all_var)-1):
    aa[all_var[i]]=i
all_var=aa


data=data.values
have=classes(data)
train_=classes_(train_,have)
test_=classes_(test_,have)



#就用比如10、13来代替情况吧，第一个字母是指种类，第二个是指父变量
#10混响  这种

p={}
for i in range(len(have)):
    for j in range(5):
        aa = split_by_characteristic(data, j)
        keys = list(aa.keys())
        ll=len(keys)
        for k in range(ll):
            lab=str(i)+str(j)+keys[k]
            mid={}
            mid_data=fit_condition(train_,i,j,keys[k])
            mid_data=mid_data.reshape(-1,9)
            mid['p']=(len(mid_data)+1)/(len(train_)+len(keys))
            mid['pp']=[]
            for ii in range(len(all_var)):
                if ii <= 5:
                    aa = split_by_characteristic(data, ii)
                    a = split_by_characteristic(mid_data, ii)
                    kkeys = list(aa.keys())
                    mid_p = {}
                    for jj in range(len(kkeys)):
                        try:
                            mm = a[kkeys[jj]]
                            mm = mm.reshape(-1, 9)
                            mid_p[kkeys[jj]] = (len(mm) + 1) / (len(mid_data) + len(kkeys))
                        except:
                            mid_p[kkeys[jj]] = 1 / (len(mid_data) + len(kkeys))
                else:
                    mid_p = {}
                    mid_p['u'] = np.mean(mid_data[:, ii])
                    s = 0
                    for l in range(len(mid_data)):
                        s = s + (mid_data[l, ii] - mid_p['u']) ** 2
                    try:
                        s = s / len(mid_data)
                    except:
                        s=0
                    mid_p['s'] = math.sqrt(s)
                mid['pp'].append(mid_p)
            p[lab]=mid


pr=pred(test_var,p,all_var)

print('准确率',ac(pr,test_lab))

print('a')
















