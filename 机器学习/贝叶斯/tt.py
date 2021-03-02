
import pandas as pd
import random as rd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from graphviz import Digraph

def loadData(filename):
    data=pd.read_csv(filename,header=0,index_col=None,encoding='gbk')
    return data

def aode(data,testsample):
    D=data['编号'].count() #数据集data的数量
    Category = data['好瓜'].unique() #数据集的类别
    N=len(Category) #数据集类别数量

	# 将离散属性和连续属性分离
    Discrete_attribute=data.columns[1:-3]
    Continuous_attribute=data.columns[-3:-1]

    res=pd.Series([1.0,1.0],index=Category)
    for k in Category:
        Pcxis=0 #最后的总和
        for i in Discrete_attribute:
            Ni=len(data[i].unique()) #第i个属性可能的取值数
            Dcxi=data[(data[i]==testsample[i][0])&(data['好瓜']==k)] #取出满足D中第i个属性取值为测试样例第i个属性的值且类别为k的数据集
            Pcxi = (len(Dcxi)+1)/(D+N*Ni)
            Pjcim=1   #P(xj|c,xi)的乘积
            for j in Discrete_attribute:
                Nj=len(data[j].unique()) #第j个属性可能的取值数
                Dcij=data[(data[i]==testsample[i][0])&(data['好瓜']==k)&(data[j]==testsample[j][0])]
                Pjci=(len(Dcij)+1)/(len(Dcxi)+Nj)
                Pjcim=Pjci*Pjcim

            Pcxis=Pcxis+Pcxi*Pjcim
        res[k]=Pcxis
    return res



if __name__=='__main__':
    data=loadData(r'.\data\watermelon_3.csv')
    testsample = pd.DataFrame(np.mat(['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]),
                              columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'], index=[0])
    result=aode(data,testsample)
    print(result)


