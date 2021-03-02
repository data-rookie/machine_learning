
import pandas as pd
import random as rd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from graphviz import Digraph

D=[]
AC=0.85
decay=9/10
TREE={}
global N
# coding=utf-8
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

def classes_(data,have):
    for i in range(data.shape[0]):
        m=have.index(data[i,-1])
        data[i,-1]=m
    return data

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

def predict(x, w):
    m, n = np.shape(x)
    y = np.zeros(m)
    for i in range(m):
        if sigmoid(np.dot(x[i], w)) > 0.5: y[i] = 1;
    return y

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
    plt.show()
    return w

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

def split_by_pred(y_p):
    index0=[]
    index1=[]
    for i in range(len(y_p)):
        if y_p[i]==0:
            index0.append(i)
        else:
            index1.append(i)
    return [index0,index1]

data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习\机器学习-决策树\data\watermelon_3.csv')
print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')
all_var=[column for column in data]#把所有变量的名称先提取出来
aa={}
for i in range(len(all_var)-1):
    aa[all_var[i]]=i
all_var=aa
A_L=all_var.copy()
all_var2=all_var.copy()

train_,test_=data_split(data,1,False)
data=data.values
# data=pd.DataFrame(data)
have=classes(data)
lisan(train_)#这几个是为了规范化数据
lisan(test_)
classes_(train_,have)
classes_(test_,have)
print('训练集的个数为：', train_.shape[0])
print('测试集的个数为：', test_.shape[0])
train_var, train_lab = lable_split(train_)
test_var, test_lab = lable_split(test_)

# b = np.ones((test_var.shape[0], 1))
# test_var = np.column_stack((test_var, b))

tree=TREE

while len(tree)!=3:

    # log_model = LogisticRegression()  # using log-regression lib model
    # train_var=train_var.astype('float')
    # train_lab=train_lab.astype('float')
    # log_model.fit(train_var, train_lab)  # fitting
    # m=log_model.coef_[0].tolist()
    # mm= log_model.intercept_.tolist()
    # w=m+mm
    # y_pred = log_model.predict(train_var)
    w=gradient_descent(train_var,train_lab)
    b = np.ones((train_.shape[0], 1))
    train_var = np.column_stack((train_var, b))
    y_pred = predict(train_var, w)
    data_set_index=split_by_pred(y_pred)
    data_set=[]
    save=[]
    for i in range(len(data_set_index)):
        data_set.append(train_[data_set_index[i]])
    train_=0
    shibai=[]
    success=[]
    kong=[]
    for i in range(len(data_set)):
        ac=0
        if len(data_set[i])==0:
            kong.append(i)
            continue
        for j in range(len(data_set[i])):
            if data_set[i][j,-1]==i:
                ac=ac+1
        ac=ac/len(data_set[i])
        if ac>AC:
            tree['tell']=w
            tree[i]=i
            success.append(i)
        else:
            shibai.append(i)
            if train_ is 0:
                train_=data_set[i]
            else:
                train_=np.vstack((train_,data_set[i]))
    # if len(sp)==1 and :
    #     for i in  range(len(sp)):
    #         tree[sp[i]] = {}
    #         tree=tree[sp[i]]
    # if len(ssp)==2:
    #     break
    if len(success)==2 or (len(success)==1 and len(kong)==1):
        break
    elif len(success)==1 and len(shibai)==1:
        for i in  range(len(shibai)):
                tree[shibai[i]] = {}
                tree=tree[shibai[i]]
    elif (len(shibai)==1 and len(kong)==1) or len(shibai)==2:
        pass
    if train_ is 0:
        break
    else:
        train_var, train_lab = lable_split(train_)
    AC=AC*decay


print(TREE)

print('a')

def plot_tree(before_index,n,tree,g,edg):
    if before_index==0 and n==0:
        g.node(str(n), str(tree['tell']))
        current=n
        n=n+1
    else:
        g.node(str(n), str(tree['tell']))
        current=n
        g.edge(str(before_index),str(n), str(edg))
        # g.view()
        n=n+1
    for i in range(2):
        if i in tree.keys():
            if isinstance(tree[i],dict):
                n=plot_tree(current,n,tree[i],g,i)
            else:
                g.node(str(n), str(tree[i]))
                g.edge(str(current),str(n), str(i))
                # g.view()
                n = n + 1
        else:
            continue
    return n

def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0])
            g.edge(inc, root, str(i))
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, tree[first_label][i])
            g.edge(inc, root, str(i))

g = Digraph("G", filename="hello5555.gv", format='png', strict=False)
N=0
plot_tree(0,N,TREE,g,0)
g.view()