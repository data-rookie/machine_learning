import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample
import random as rd

def ac(y_pred,y):
    result=0
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            result=result+1
    return result/len(y)

def dataload(x):
    data=pd.read_csv(x,header=None,index_col=0,encoding='gbk')#必须要用这两个关键字，要不然它会把数据当成行来读取
    #这里的index_col要按照数据集给出来的来看，是0还是none，不同的数据集是不一样的，比如那个西瓜的和UCI的数据集
    #https://blog.csdn.net/weixin_44056331/article/details/89366105
    return data

def normalization(x):
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range

def data_split(data,train_size=0.9,random=False):
    if random:
        select_train=rd.sample(range(1,len(data)+1),int(len(data)*train_size))
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

def stumpClassify(X, dim, thresh_val, thresh_inequal):
    ret_array = np.ones((X.shape[0], 1))

    if thresh_inequal == 'lt':
        ret_array[X[:, dim] <= thresh_val] = -1
    else:
        ret_array[X[:, dim] > thresh_val] = -1

    return ret_array

def buildStump(X, y,ii):
    m, n = X.shape
    best_stump = {}
    min_error = 1
    # for dim in range(n):
    dim=ii
    x_min = np.min(X[:, dim])
    x_max = np.max(X[:, dim])

    #############################################
    split_points = [(x_max - x_min) / 20 * i + x_min for i in range(20)]

    for inequal in ['lt', 'rt']:
        for thresh_val in split_points:
            ret_array = stumpClassify(X, dim, thresh_val, inequal)

            error = np.mean(ret_array != y)

            if error < min_error:
                best_stump['dim'] = dim
                best_stump['thresh'] = thresh_val
                best_stump['inequal'] = inequal
                best_stump['error'] = error
                min_error = error

    return best_stump

def re_pick(train_var,train_lab,num=20):#############################################
    select=[]
    v=[]
    l=[]
    for i in range(num):
        select.append(rd.randint(0,train_var.shape[0]-1))
    for i in range(num):
        v.append(train_var[select[i]])
        l.append(train_lab[select[i]])
    v=np.array(v)
    l=np.array(l)
    return v,l

def stumpBagging(X, y, nums=20):
    stumps = []
    seed = 16
    for _ in range(nums):
        X_, y_ =re_pick(X,y)#############################################
        seed += 1
        stumps.append(buildStump(X_, y_,_%2))
    return stumps

def stumpPredict(X, stumps):
    ret_arrays = np.ones((X.shape[0], len(stumps)))
    for i, stump in enumerate(stumps):
        ret_arrays[:, [i]] = stumpClassify(X, stump['dim'], stump['thresh'], stump['inequal'])
    max_e=0
    min_e=1
    for i in range(len(stumps)):
        if stumps[i]['error']<min_e:
            min_e=stumps[i]['error']
        elif stumps[i]['error']>max_e:
            max_e=stumps[i]['error']
    e_list=[]
    for i in range(len(stumps)):#############################################
        e_list.append((stumps[i]['error'])/(max_e))
    for i in range(ret_arrays.shape[0]):
        for j in range(ret_arrays.shape[1]):
            ret_arrays[i,j]=(1/e_list[j])*ret_arrays[i,j]
    # rr=np.sum(ret_arrays, axis=1)
    return np.sign(np.sum(ret_arrays, axis=1))

def pltStumpBaggingDecisionBound(X_, y_, stumps):
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(-0.1, 0.7, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    Z_ = stumpPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], stumps).reshape(X_tmp.shape)

    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='black', linewidths=5)

    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    for i in range(len(stumps)):
        if stumps[i]['dim']==0:
            plt.axvline(x=stumps[i]['thresh'], color='r', linestyle='-')
            # plt.lin\
        else:
            plt.axhline(y=stumps[i]['thresh'], color='r', linestyle='-')
    plt.legend()
    plt.show()





if __name__ == "__main__":

    # data=pd.read_csv(data_path,header=None,index_col=0,encoding='gbk')#必须要用这两个关键字，要不然它会把数据当成行来读取

    data = dataload( r'.\data\watermelon_3a.csv')
    print('数据集的样本总个数为', data.shape[0])
    print('有', data.shape[1] - 1, '个属性')

    train_, test_ = data_split(data, 0.7, True)

    print('训练集的个数为：', train_.shape[0])
    print('测试集的个数为：', test_.shape[0])
    # have = classes(data)
    train_var=train_[:,0:-1]
    train_lab=train_[:,-1]

    test_var=test_[:,0:-1]
    test_lab=test_[:,-1]


    test_lab[test_lab==0]=-1
    train_lab[train_lab == 0] = -1

    # y[y == 0] = -1
    stumps = stumpBagging(train_var, train_lab, 21)
    # print(np.mean(stumpPredict(train_var, stumps) == train_lab))
    pltStumpBaggingDecisionBound(train_var, train_lab, stumps)
    aaa=stumpPredict(test_var,stumps)
    print('准确率为',ac(aaa,test_lab))
    print('a')



