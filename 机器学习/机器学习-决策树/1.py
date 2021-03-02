
import pandas as pd
import random as rd
import numpy as np
import math
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus  # 画句子的依存结构树

PRE_TREE={}
AFT_TREE={}
A_L=[]
yz=[]
yz_lab=[]
TREE={1:[],2:[],3:[],4:[],5:[]}

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

def classes_split(data,have):#这里的have就是y，也就是有几类的样本
    all_class=[0 for i in range(len(have))]
    data = data.reshape(-1, 7)
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

def Gini(data,have):
    result=0
    num=data.shape[0]
    c=classes_split(data,have)
    for i in range(len(c)):
        if c[i] is 0:
            continue
        else:
            c[i] = c[i].reshape(-1, 6)#这个一定要加上，要不然如果只有一行的话。。他就会自动把他的列数当成行数了
            result=result+(c[i].shape[0]/num)**2
    return 1-result


def split_by_characteristic(data,var_index):#这里按照变量来对数据集进行划分,var_index是变量的位置不是名字
    all_index={}
    data=data.reshape(-1, 7)
    for i in range(len(data)):
        if data[i,var_index] not in all_index.keys():
            all_index[data[i,var_index]]=data[i]
        else:
            all_index[data[i,var_index]]=np.vstack((all_index[data[i,var_index]],data[i]))
    return all_index

def select_min_gini_index(data,all_var,have):
    g=[]
    k=list(all_var.keys())
    for i in range(len(all_var)):
        mid=0
        mm=split_by_characteristic(data,all_var[k[i]])
        mm_index=list(mm.keys())
        for j in range(len(mm_index)):
            mid=mid+mm[mm_index[j]].shape[0]*Gini(mm[mm_index[j]],have)/data.shape[0]
        g.append(mid)
    select=g.index(min(g))
    print("选择的属性是:",k[select])
    result=split_by_characteristic(data,all_var[k[select]])
    del all_var[k[select]]
    return result,all_var,k[select]

def last_tell(data_split):#来判断一下是否分到底了,如果分到底了就返回他的类别，没有的话就-1
    mid=data_split
    mid=mid.reshape(-1, 7)
    t=mid[0][mid.shape[1]-1]
    for j in range(len(mid)):
        if mid[j][mid.shape[1]-1]==t:
            continue
        else:
            return -1
    return t


def create_tree(data,all_var,have):
    tree={}
    mid=[]
    d,v,s_v=select_min_gini_index(data,all_var,have)
    mm_index = list(d.keys())
    tree["name"] = s_v
    # if s_v=='根蒂':
    #     print('sdsdsd')
    for i in range(len(d)):
        if last_tell(d[mm_index[i]])==-1:
            mid.append(create_tree(d[mm_index[i]],v.copy(),have))#这里的copy是很重要的，要不然等一下把值都改光了。。。。。
        else:
            mid.append(last_tell(d[mm_index[i]]))
    tree['split']=mid
    return tree

def most_class(data):
    data=data.reshape(-1, 7)
    have=[0,0]#这里就先这样设置好，之后如果想改的话再改一下吧
    for i in range(data.shape[0]):
        have[data[i,-1]]=have[data[i,-1]]+1
    b=have.index(max(have))
    return b

# def pre_tree(data,all_var,have,yz,yz_lab):
#     tree1={}#这个是不划分的树
#     tree2={}#这个是划分的树
#     tree={}#这个是最终决定的树
#     a_l=all_var.copy()
#     d,v,s_v=select_min_gini_index(data,all_var,have)
#     tree['name']=s_v
#     tell=[]
#     cl=most_class(data)
#     tree1=[cl]
#     tree2=[]
#     t=0
#     tree['split']=[]
#     for i in range(len(d)):
#         tree2.append(most_class(d[i]))
#     tree['split']=tree2
#     ac1=ac(pred(yz,tree,a_l),yz_lab)
#     tree['split']=tree1
#     ac2=ac(pred(yz,tree,a_l),yz_lab)
#     if ac1<=ac2:#这里设置相等的话也不让他划分
#         tell.append(0)
#         tree['split'] = tree1
#     else:
#         tell.append(1)
#         tree['split'] = tree2
#     mid=tree
#     while tell.count(1)!=0:
#         for ii in range(len(tell)):
#             if t == 0:
#                 mid=mid['split']
#             else:
#                 mid=mid[ii]
#             # tell_index=[]
#             # while tell.count(1)!=0:
#             #     tell_index.append(tell.index(1))
#             #     tell[tell.index(1)]=0
#             # for i in range(len(tell)):
#             #     if tell[i]==1:
#             #         tree['split']=tree2
#             #     else:
#             #         tree['split']=tree1
#             tell=[]
#             for i in range(len(tree['split'])):
#                 dd, vv, s_vv = select_min_gini_index(d[i], v.copy(), have)
#                 tree['split'][i]={'name':s_vv,'split':[]}
#                 for j in range(len(dd)):
#                     tree['split'][i]['split'].append(most_class(dd[j]))
#                 tree2=tree['split'][i]['split'].copy()
#                 ac1 = ac(pred(yz, tree, a_l), yz_lab)
#                 tree['split'][i]['split']=[most_class(d[i])]
#                 tree1=[most_class(d[i])]
#                 ac2=ac(pred(yz, tree, a_l), yz_lab)
#                 if ac1>ac2:
#                     tell.append(1)
#                     tree['split'][i]['split']=tree2
#                 else:
#                     tell.append(0)
#                     tree['split'][i]['split']=tree1
#             mid=mid['split']
#         t=t+1
'''尝试做了一下用while来预剪枝的。。。好像不行。。。看来只能用广度优先搜索。。。。好像也可以用深度，因为旁边的分支是跟他没有关系的，对他的准确率是没有影响的  '''




def ac(y_pred,y):
    result=0
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            result=result+1
    return result/len(y)


def pred(data,tree,all_var):
    y_pred=[]
    for i in range(data.shape[0]):
        mid2=tree
        var=list(data[i])
        kk=data.copy()
        while isinstance(mid2,dict):
            mid=mid2['name']
            a=list(split_by_characteristic(kk,all_var[mid]).keys())
            kk=split_by_characteristic(kk,all_var[mid])
            b=a.index(var[all_var[mid]])
            kk=kk[var[all_var[mid]]]
            mid2=mid2['split']
            if len(mid2)==1:
                mid2=mid2[0]
            else:
                mid2=mid2[b]
        y_pred.append(mid2)
    return y_pred


def showtree(tree,n):
    mid=tree
    print(mid['name'])
    TREE[n].append(mid['name'])
    mid=mid['split']
    for i in range(len(mid)):
        if isinstance(mid[i],dict):
            showtree(mid[i],n+1)
        else:
            TREE[n+1].append(mid[i])

def pre_tree(data,all_var,have,tree):
    d,v,s_v=select_min_gini_index(data,all_var,have)
    mm_index = list(d.keys())
    tree['name']=s_v
    #不分裂的话
    tree['split']=[]
    tree['split'].append(most_class(data))
    ac1=ac(pred(yz,PRE_TREE,A_L), yz_lab)
    #分裂的话
    tree['split']=[]
    for i in range(len(d)):
        tree['split'].append(most_class(d[mm_index[i]]))
    ac2=ac(pred(yz,PRE_TREE,A_L), yz_lab)
    if ac2>ac1:
        for i in range(len(d)):
            tree['split'][i]={}
            pre_tree(d[mm_index[i]],v.copy(),have,tree['split'][i])
    else:
        tree['split']=[]
        tree['split'].append(most_class(data))


def afer_tree(data,all_var,have,tree):
    d,v,s_v=select_min_gini_index(data,all_var,have)
    mm_index = list(d.keys())
    for i in range(len(d)):
        if isinstance(tree['split'][i],dict):
            afer_tree(d[mm_index[i]], v.copy(), have, tree['split'][i])
    ac1 = ac(pred(yz, AFT_TREE, A_L), yz_lab)
    m=tree['split']
    tree['split']=[]
    tree['split'].append(most_class(data))
    ac2 = ac(pred(yz, AFT_TREE, A_L), yz_lab)
    if ac2>ac1:
        pass
    else:
        tree['split']=m

def classes_(data,have):
    for i in range(data.shape[0]):
        m=have.index(data[i,-1])
        data[i,-1]=m
    return data

def ac(y1,y2):
    t=0
    for i in range(len(y1)):
        if y1[i]==y2[i]:
            t=t+1
    return t/len(y1)

def change(data):
    names = data.columns[:]
    for i in names:
        col = pd.Categorical(data[i])
        data[i] = col.codes
    print(data)


# def showtree_pdf(data):
#     from sklearn import tree  # 导入sklearn的决策树模型（包括分类和回归两种）
#     import pydotplus  # 画句子的依存结构树
#
#     a = data.iloc[:, :-1]  # 特征矩阵
#     b = data.iloc[:, -1]  # 目标变量
#     clf = tree.DecisionTreeClassifier()  # 分类决策树
#     clf.fit(a, b)  # 训练
#     dot_data = tree.export_graphviz(clf, out_file=None)  # 利用export_graphviz将树导出为Graphviz格式
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_pdf("iris4.pdf")


data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习\机器学习-决策树\data\watermelon_2.csv')
all_var=[column for column in data]#把所有变量的名称先提取出来
aa={}
for i in range(len(all_var)-1):
    aa[all_var[i]]=i
all_var=aa
A_L=all_var.copy()
all_var2=all_var.copy()
all_var3=all_var.copy()
all_var4=all_var.copy()
all_var5=all_var.copy()
all_var6=all_var.copy()
all_var7=all_var.copy()
print('变量有:',all_var)
print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')
train_,test_=data_split(data,0.8,True)#这里如果是用非随机的话，他的train和test是跟原本的data是一个的，所以后面对于data改变类后，train也会跟着改，如果是用随机的话，他的data和train是不一样的，所以不会一起


data=data.values
# data=pd.DataFrame(data)
have=classes(data)
classes_(train_,have)
classes_(test_,have)#这里的话。。。他的0和1可能对应会出问题。。。。。。。所以准确率没那么高。。。。到时候再看看
print('训练集的个数为：', train_.shape[0])
print('测试集的个数为：', test_.shape[0])
train_var, train_lab = lable_split(train_)
# train_var = pd.DataFrame(train_var)##
# change(train_var)##
# train_var=train_var.values##

test_var, test_lab = lable_split(test_)


#记录决策树
a=create_tree(train_,all_var,have)
print('原始的决策树:')
print(a)
b=pred(test_,a,all_var2)#这里因为之前的all_var都是传值的，，结果把她的结果改了。。。所以要copy一个来用
print(b)
print('准确率为:',ac(b,test_lab))
# showtree(a,1)

yz=test_
yz_lab=test_lab
pre_tree(train_,all_var3,have,PRE_TREE)
print('预剪枝之后:')
print(PRE_TREE)
b=pred(test_,PRE_TREE,all_var5)#这里因为之前的all_var都是传值的，，结果把她的结果改了。。。所以要copy一个来用
print(b)
print('准确率为:',ac(b,test_lab))



AFT_TREE=a.copy()
afer_tree(train_,all_var4,have,AFT_TREE)
print('后剪枝之后:')
print(AFT_TREE)
# print(TREE)
b=pred(test_,AFT_TREE,all_var6)#这里因为之前的all_var都是传值的，，结果把她的结果改了。。。所以要copy一个来用
print(b)
print('准确率为:',ac(b,test_lab))





'''
#调用模型来求解,好想这里如果要用模型来求解的话。。。他好想是需要把这些变量都转成那个离散的值的
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(train_var, train_lab)
print(clf.tree_)
 # 利用export_graphviz将树导出为Graphviz格式
# with open("tree.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
'''
print('a')

