import pandas as pd
import random as rd
import numpy as np
import math
import random
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# import numpy as np
import mpl_toolkits.mplot3d
'''画3d图的
    #x=[0.437,0.211,1]
    #y=[1]
    figure = plt.figure()
    # ax = Axes3D(figure)
    ax = figure.gca(projection="3d")
    x1 = np.linspace(-100, 100, 1000)
    y1 = np.linspace(-100,100, 1000)
    x, y = np.meshgrid(x1, y1)
    z=-1 * (x*0.437+y*0.211)+ln(1 +ex(x * 0.437 + y * 0.211))
    # z = -1 * (x*0.437+y*0.211) + math.log(1 + math.exp(x*0.437+y*0.211))
    # z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    # ax.plot_surface(x,y,z,rstride=10,cstride=4,cmap=cm.YlGnBu_r)
    ax.plot_surface(x, y, z, cmap="rainbow")
    plt.show()
'''
'''
1.数据加载(可能还会有数据的预处理)
2.数据分割
3.梯度下降法或者其他的方法求w和b
4.用测试集来预测
'''
def normalization(x):
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def dataload(x):
    data=pd.read_csv(x,header=None,index_col=None)#必须要用这两个关键字，要不然它会把数据当成行来读取
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

def cross_validation_data_split(data,num,test_num):#num就是将数据分成几份,test_num就是用哪一份来做测试集
    data=data.values
    all=np.array_split(data, num, axis=0)#用这个函数会好一点，这样万一不均等的话，不会出错
    test=all[test_num-1]
    del all[test_num-1]
    train=all[0]
    for i in range(1,len(all)):
        mid=all[i]
        train=np.vstack((mid,train))
    return train,test

def lable_split(data):
    data_lable=data[:,data.shape[1]-1]
    data=data[:,0:data.shape[1]-1]
    return data,data_lable

def sigmoid(x):
    return 1.0/(1+np.exp((-x)))

def max_likelihood_function(x,y,w):
    return -y * np.dot(w, x.T) + np.math.log(1 + np.math.exp(np.dot(w, x.T)))

def fd(x,y,w,i,length=0.00001):
    delta_x=length
    ww=w.copy()
    ww[i]=w[i]+delta_x
    return (max_likelihood_function(x,y,ww)-max_likelihood_function(x,y,w))/delta_x

def ex(x):
    for i in range(len(x)):
        for j in range(len(x)):
            x[i][j]=math.exp(x[i][j])
    return x
def ln(x):
    for i in range(len(x)):
        for j in range(len(x)):
            x[i][j]=math.log(x[i][j])
    return x

def gradient_descent(x,y):
    #这里试一下把数据集打乱一下:
    random_data = np.column_stack((x,y))
    np.random.shuffle(random_data)
    x=random_data[:,0:data.shape[1]-1]
    y=random_data[:,random_data.shape[1]-1]
    decay=0.01#好像有可能是因为这个才好的。。。。
    step=0.01
    data_num,var_num=np.shape(x)
    # w=np.ones((var_num+1))
    w=np.zeros((var_num+1))
    b=np.ones((data_num,1))
    x=np.column_stack((x,b))
    chang_w=np.ones((var_num+1))
    # while True：
    b = np.zeros((var_num+1, data_num))
    for i in range(data_num):
        temp=np.ones((var_num+1))
        for j in range(var_num+1):
            step = step / (step + decay*i)
            b[j, i] = w[j]
            temp[j]=-step*fd(x[i],y[i],w,j)#这个不能这样搞，要不然3个变量全部都共进退了
            # chang_w[j]=temp[j]
            w[j]=w[j]+temp[j]#这里是++++++++++++++++++++++++++++++++啊。。。。。。。sb一样写成-了。。。前面就有-了。。。。。。怪不得cost越来越多
            # w=w-temp
        cost=0
        for j in range(data_num):
            cost=cost+max_likelihood_function(x[i],y[i],w)
        # print(cost)#这里为了验证一下w的调整是有用的

    #下面这里是把系数的变化给画出来，这样就可以比较好看它的梯度下降到底好不好用到底有没有用
    t = np.arange(data_num)
    f2 = plt.figure(3)
    p1 = plt.subplot(311)
    p1.plot(t, b[0])
    plt.ylabel('w1')
    p2 = plt.subplot(312)
    p2.plot(t, b[1])
    plt.ylabel('w2')
    p3 = plt.subplot(313)
    p3.plot(t, b[2])
    plt.ylabel('b')
    # plt.show()
    return w

def classes(data):
    have=[]
    for i in range(data.shape[0]):
        if data[data.shape[1]-1][i] not in have:#这个好像是先列数后行数,这个跟前面不一样，一个是[:,:],一个是[][],这个要注意
            have.append(data[data.shape[1]-1][i])
    for i in range(data.shape[0]):#顺便把它的字母什么的给改一下，这样规范好处理一点,这个也算预处理的部分吧
        for j in range(len(have)):
            if data[data.shape[1]-1][i]==have[j]:
                data[data.shape[1] - 1][i]=j
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

def predict(x, w):
    m, n = np.shape(x)
    y = np.zeros(m)
    for i in range(m):
        if sigmoid(np.dot(x[i], w)) > 0.5: y[i] = 1;
    return y

def ac(y1,y2):
    t=0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            t=t+1
    return t/len(y1)
#------------------------------------------------------------------------------------------------------------------------------------------------

# data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习-线性模型\data\watermelon_3a.csv')
# '''
# 这里先描述一下数据集吧
# '''
# print('数据集的样本总个数为',data.shape[0])
# print('有',data.shape[1]-1,'个属性')
# # train_,test_=data_split(data,0.9,False)
# train_,test_=cross_validation_data_split(data,10,3)
# print('训练集的个数为：',train_.shape[0])
# print('测试集的个数为：',test_.shape[0])
# train_var,train_lab=lable_split(train_)
# test_var,test_lab=lable_split(test_)
# #这个放回的是带标签的
#
#
#
# '''
# 先用调包的，看看它算出来的参数，和它的准确率
# '''
# log_model = LogisticRegression()  # using log-regression lib model
# log_model.fit(train_var, train_lab)  # fitting
# print('由调包模型算出来的系数w为:',log_model.coef_)
# print('由调包模型算出来的b为:',log_model.intercept_)
# y_pred = log_model.predict(test_var)
#
#
#
# w=gradient_descent(train_var,train_lab)
# print('由自己的模型算出来的系数w为:',w)
# b = np.ones((test_var.shape[0], 1))
# test_var = np.column_stack((test_var, b))
# y_pred = predict(test_var, w)


#---------------------------------------------------------------------------------------------------------------------------------------
#下面是多分类的
data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习\机器学习-线性模型\data\wine2.data.csv')

print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')
cor=classes(data)
# data[:,:-1]=normalization(data[:,:-1])
train_,test_=data_split(data,0.9,True)
ac_all=[]
data=np.array(data)
a = data[:, :-1]
a = normalization(a)
bb=data[:,-1]
bb=bb.reshape((178,1))
data = np.hstack((a, bb))
data=pd.DataFrame(data)
for i in range(178):

    train_,test_=cross_validation_data_split(data,178,i)

    all_class=classes_split(train_,cor)

    print('训练集的个数为：',train_.shape[0])
    print('测试集的个数为：',test_.shape[0])
    # train_=normalization(train_)

    train_var,train_lab=lable_split(train_)
    # train_var=normalization(train_var)


    test_var,test_lab=lable_split(test_)
    # test_var=normalization(test_var)
    b = np.ones((test_var.shape[0], 1))
    test_var = np.column_stack((test_var, b))


    # y_pred = log_model.predict(test_var)

    num_of_tell=len(all_class)*(len(all_class)-1)/2#所有组合的个数

    pair=[]
    for i in range(len(cor)):#这样就可以把他的所有的组合都求出来了
        for j in range(i+1,len(cor)):
            pair.append([i,j])
    w_all=[]#这个用来保存所有分类器的参数

    for i in range(int(num_of_tell)):
        part1=all_class[pair[i][0]]
        part2=all_class[pair[i][1]]
        part=np.vstack((part1, part2))
        lab1 = np.zeros((part1.shape[0], 1))
        lab2= np.ones((part2.shape[0], 1))
        lab=np.vstack((lab1, lab2))
        log_model = LogisticRegression()  # using log-regression lib model
        log_model.fit(part, lab)  # fitting
        print('由调包模型算出来的系数w为:', log_model.coef_)
        print('由调包模型算出来的b为:', log_model.intercept_)
        m=log_model.coef_[0].tolist()
        mm= log_model.intercept_.tolist()
        w=m+mm
        # w=gradient_descent(part,lab)
        w_all.append(w)



    #预测
    y_pred_all=[]
    for i in range(len(w_all)):
        mid=predict(test_var, w_all[i])
        for j in range(len(mid)):#这里进行类别的转换
            if mid[j]==0:
                mid[j]=pair[i][0]
            else:
                mid[j]=pair[i][1]
        y_pred_all.append(mid)

    y_pred=[]
    for i in range(test_var.shape[0]):
        count=[]
        for j in range(len(w_all)):
            count.append(y_pred_all[j][i])
        y_pred.append(Counter(count).most_common(1)[0][0])
    print(y_pred)
    print(ac(test_lab,y_pred)*100)
    ac_all.append(ac(test_lab,y_pred)*100)
ac_all=np.array(ac_all)
print('总的ac：',np.mean(ac_all,axis=0))
#这里之后可以弄一个看是哪一份训练出来是最差的


print('a')