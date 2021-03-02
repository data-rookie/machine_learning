


from sklearn.decomposition import PCA




from collections import Counter
import pandas as pd
import random as rd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

from svm import SVM

def normalization(x):
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def dataload(x):
    data=pd.read_csv(x,header=None,index_col=0,encoding='gbk')#必须要用这两个关键字，要不然它会把数据当成行来读取
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



def classes_split(data,have):
    all_class=[0 for i in range(len(have))]
    for i in range(data.shape[0]):
        for j in range(len(have)):
            if int(data[i,data.shape[1]-1])==have[j]:
                if all_class[j] is 0:#这里不能用==，会有二义性。。。好像是因为一个矩阵不能用==来跟0来比较
                    all_class[j]=data[i,0:data.shape[1]-1]
                else:
                    mid1= all_class[j]
                    mid2= data[i,0:data.shape[1]-1]
                    all_class[j]=np.vstack((mid1,mid2))
    return all_class

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
def ac(y_pred,y):
    result=0
    for i in range(len(y_pred)):
        if y_pred[i]==y[i]:
            result=result+1
    return result/len(y)

data=dataload(r'.\data\glass.data.csv')
print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')
# cc=data.corr()
# cc=cc.values
# data=data.values
train_,test_=data_split(data,0.7,True)
train_var, train_lab = lable_split(train_)
test_var, test_lab = lable_split(test_)
data=data.values
have=classes(data)
pca_sk = PCA(n_components=4)
train_var = pca_sk.fit_transform(train_var)
train_= np.column_stack((train_var,train_lab))
test_var=pca_sk.fit_transform(test_var)
test_= np.column_stack((test_var,test_lab))
all_class = classes_split(train_, have)

# train_var=normalization(train_var)
# test_var=normalization(test_var)
num_of_tell = len(all_class) * (len(all_class) - 1) / 2  # 所有组合的个数

pair = []
for i in range(len(have)):  # 这样就可以把他的所有的组合都求出来了
    for j in range(i + 1, len(have)):
        pair.append([i, j])


model_all=[]
for i in range(int(num_of_tell)):
    param_grid = {
        'C': 100,
        'kernel_type': 'linear',
        'tol': 0.001,
        'epsilon': 0.1
    }
    part1 = all_class[pair[i][0]]
    part2 = all_class[pair[i][1]]
    part = np.vstack((part1, part2))
    lab=[]
    for ii in range(part1.shape[0]):
        lab.append(-1)
    for ii in range(part2.shape[0]):
        lab.append(1)
    # lab1 = np.zeros((part1.shape[0], 1))
    # lab2 = np.ones((part2.shape[0], 1))
    # lab = np.vstack((lab1, lab2))
    # lab=list(lab)
    random_data = np.column_stack((part,lab))#打乱，要不然效果很差！！！
    np.random.shuffle(random_data)
    x=random_data[:,0:random_data.shape[1]-1]
    y=random_data[:,random_data.shape[1]-1]
    data={}
    data['train_X']=x
    data['train_y']=y
    data['test_X']=0
    data['test_y']=0
    model = SVM(data=data, param=param_grid,verbose=True)
    model.train()
    model_all.append(model)

#预测的部分:

y_pred_all=[]
for i in range(len(model_all)):
    mid = model_all[i].hypothesis(test_var)
    for j in range(len(mid)):  # 这里进行类别的转换
        if mid[j] == -1:
            mid[j] = pair[i][0]
        else:
            mid[j] = pair[i][1]
    y_pred_all.append(mid)

y_pred=[]
for i in range(test_var.shape[0]):
    count=[]
    for j in range(len(model_all)):
        count.append(y_pred_all[j][i])
    acd=Counter(count).most_common(1)[0][0]
    y_pred.append(have[int(acd)])
print(y_pred)
# a=model.hypothesis(test_var)
print('linear-ac',ac(y_pred,test_lab))


model_all=[]
for i in range(int(num_of_tell)):
    param_grid = {
        'C': 100,
        'kernel_type': 'poly',
        'tol': 0.001,
        'epsilon': 0.1
    }
    part1 = all_class[pair[i][0]]
    part2 = all_class[pair[i][1]]
    part = np.vstack((part1, part2))
    lab=[]
    for ii in range(part1.shape[0]):
        lab.append(-1)
    for ii in range(part2.shape[0]):
        lab.append(1)
    # lab1 = np.zeros((part1.shape[0], 1))
    # lab2 = np.ones((part2.shape[0], 1))
    # lab = np.vstack((lab1, lab2))
    # lab=list(lab)
    random_data = np.column_stack((part,lab))#打乱，要不然效果很差！！！
    np.random.shuffle(random_data)
    x=random_data[:,0:random_data.shape[1]-1]
    y=random_data[:,random_data.shape[1]-1]
    data={}
    data['train_X']=x
    data['train_y']=y
    data['test_X']=0
    data['test_y']=0
    model = SVM(data=data, param=param_grid,verbose=True)
    model.train()
    model_all.append(model)

#预测的部分:

y_pred_all=[]
for i in range(len(model_all)):
    mid = model_all[i].hypothesis(test_var)
    for j in range(len(mid)):  # 这里进行类别的转换
        if mid[j] == -1:
            mid[j] = pair[i][0]
        else:
            mid[j] = pair[i][1]
    y_pred_all.append(mid)

y_pred=[]
for i in range(test_var.shape[0]):
    count=[]
    for j in range(len(model_all)):
        count.append(y_pred_all[j][i])
    acd=Counter(count).most_common(1)[0][0]
    y_pred.append(have[int(acd)])
print(y_pred)
# a=model.hypothesis(test_var)
print('poly-ac',ac(y_pred,test_lab))


model_all=[]
for i in range(int(num_of_tell)):
    param_grid = {
        'C': 100,
        'kernel_type': 'rbf',
        'tol': 0.001,
        'epsilon': 0.1
    }
    part1 = all_class[pair[i][0]]
    part2 = all_class[pair[i][1]]
    part = np.vstack((part1, part2))
    lab=[]
    for ii in range(part1.shape[0]):
        lab.append(-1)
    for ii in range(part2.shape[0]):
        lab.append(1)
    # lab1 = np.zeros((part1.shape[0], 1))
    # lab2 = np.ones((part2.shape[0], 1))
    # lab = np.vstack((lab1, lab2))
    # lab=list(lab)
    random_data = np.column_stack((part,lab))#打乱，要不然效果很差！！！
    np.random.shuffle(random_data)
    x=random_data[:,0:random_data.shape[1]-1]
    y=random_data[:,random_data.shape[1]-1]
    data={}
    data['train_X']=x
    data['train_y']=y
    data['test_X']=0
    data['test_y']=0
    model = SVM(data=data, param=param_grid,verbose=True)
    model.train()
    model_all.append(model)

#预测的部分:

y_pred_all=[]
for i in range(len(model_all)):
    mid = model_all[i].hypothesis(test_var)
    for j in range(len(mid)):  # 这里进行类别的转换
        if mid[j] == -1:
            mid[j] = pair[i][0]
        else:
            mid[j] = pair[i][1]
    y_pred_all.append(mid)

y_pred=[]
for i in range(test_var.shape[0]):
    count=[]
    for j in range(len(model_all)):
        count.append(y_pred_all[j][i])
    acd=Counter(count).most_common(1)[0][0]
    y_pred.append(have[int(acd)])
print(y_pred)
# a=model.hypothesis(test_var)
print('rbf-ac',ac(y_pred,test_lab))

print('a')





































