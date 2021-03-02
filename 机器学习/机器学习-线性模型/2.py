import pandas as pd
import random as rd
import numpy as np
import math
import random
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mpl_toolkits.mplot3d

def dataload(x):
    data=pd.read_csv(x,header=None,index_col=0)#必须要用这两个关键字，要不然它会把数据当成行来读取
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

def lable_split(data):
    data_lable=data[:,data.shape[1]-1]
    data=data[:,0:data.shape[1]-1]
    return data,data_lable

def caculate_class_mean(data):
    mean=[]
    for i in range(data.shape[1]):
        mid=0
        for j in range(data.shape[0]):
            mid=mid+data[j,i]
        mean.append(mid/(data.shape[0]))
    return mean


def SW(data,mean):#真的是sb，这个直接调函数就可以了，自己算干嘛。。。。。而且还有np。mean。。。。真的脑残
    for i in range(data.shape[0]):
        data[i]=data[i]-mean
    return np.dot(data.T,data)

def SB(mean1,mean2):
    a=(np.array(mean1)-np.array(mean2))
    a=a.reshape((1,2))
    b=a.reshape((2,1))
    return np.dot(b,a)

def predict(data,w,mean):
    dis=[]
    y=[]
    for i in range(len(mean)):
        dis.append(np.dot(w,mean[i]))
    for i in range(data.shape[0]):
        mid=np.dot(w,data[i])
        mm=dis
        for j in range(len(mm)):
            mm[j]=abs(mm[j]-mid)
        y.append(mm.index(min(mm)))
    return y




'''直接调用模型的
lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(train_var, train_lab)

y_pred = lda_model.predict(test_var)
print(lda_model.coef_)
print(lda_model.intercept_)

# # summarize the fit of the model#这个是最后看预测的每一类的，混淆矩阵，就是学长说的那个作用
# print(metrics.confusion_matrix(y_test, y_pred))
# print(metrics.classification_report(y_test, y_pred))
'''

#------------------------------------------------------------------------------------------------------------------------------
data=dataload(r'C:\Users\cjn\Desktop\pycccccc\机器学习\机器学习-线性模型\data\watermelon_3a.csv')
'''
这里先描述一下数据集吧
'''



print('数据集的样本总个数为',data.shape[0])
print('有',data.shape[1]-1,'个属性')
# cor=classes(data)
train_,test_=data_split(data,0.7,False)
# train_,test_=cross_validation_data_split(data,10,3)
print('训练集的个数为：',train_.shape[0])
print('测试集的个数为：',test_.shape[0])
train_var,train_lab=lable_split(train_)
test_var,test_lab=lable_split(test_)

#
lda_model = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None).fit(train_var, train_lab)

y_pred = lda_model.predict(test_var)
print("模型求出来的主向量：",lda_model.scalings_)
print("模型求出来的sw",lda_model.covariance_)
# print(lda_model.coef_)
# print(lda_model.intercept_)

all_class=classes_split(train_,[0,1])
x1=all_class[0][:,0]
y1=all_class[0][:,1]

x2=all_class[1][:,0]
y2=all_class[1][:,1]
colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
area = np.pi * 4**2  # 点面积
plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')


mean=[]
for i in range(len(all_class)):
    m=caculate_class_mean(all_class[i])
    mean.append(m)
sw=[]
for i in range(len(all_class)):
    # sw.append(SW(all_class[i],mean[i]))
    sw.append(np.cov(all_class[i], rowvar=False))
ss=sw[0]*((all_class[0].shape[0])/train_var.shape[0])#好像它的调包的有标准化，所以它跟它还是有点微小的差距?
for i in range(1,len(all_class)):#这个好像还不是直接加的
    ss=ss+sw[i]*((all_class[i].shape[0])/train_var.shape[0])
sb=SB(mean[0],mean[1])
sw=ss
sw=np.array(sw)
eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(sw),(sb)))
print(eig_vecs)
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[i]
    print('\n Eigenvector {}: \n {}'.format(i + 1, eigvec_sc.real))
    print('Eigenvalue {: }: {:.2e}'.format(i + 1, eig_vals[i].real))

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i]) for i in range(len(eig_vals))]

# sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually cinfirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in decreasing order: \n')
for i in eig_pairs:
    print(i[0])

print('Variance explained：\n')
eigv_sum = sum(eig_vals)
for i, j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i + 1, (j[0] / eigv_sum).real))#就是用本身的特征值除上所有的特征值相加


plt.plot([0,0],[eig_pairs[0][1][0],eig_pairs[0][1][1]],linewidth = '0.5',color='#000000')
plt.show()
mss=lda_model.coef_[0].tolist()
mss[0]=mss[0]/40
mss[1]=mss[1]/40
# plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='类别A')
# plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='类别B')
# plt.plot([0,0],[mss[0],mss[1]],linewidth = '0.5',color=colors2)
# plt.show()

#这里画个图看看对不对

print("自己的主向量",eig_pairs[0][1])
aa=predict(test_var,eig_pairs[0][1],mean)
ac=0
for i in range(len(aa)):
    if aa[i]==test_lab[i]:
        ac=ac+1
print('准确率为',ac/len(aa))


print(aa)





print('a')







