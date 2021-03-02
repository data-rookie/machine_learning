import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample
import random as rd

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


def getDataSet():
    dataSet = [
        [0.697, 0.460, '是'],
        [0.774, 0.376, '是'],
        [0.634, 0.264, '是'],
        [0.608, 0.318, '是'],
        [0.556, 0.215, '是'],
        [0.403, 0.237, '是'],
        [0.481, 0.149, '是'],
        [0.437, 0.211, '是'],
        [0.666, 0.091, '否'],
        [0.243, 0.267, '否'],
        [0.245, 0.057, '否'],
        [0.343, 0.099, '否'],
        [0.639, 0.161, '否'],
        [0.657, 0.198, '否'],
        [0.360, 0.370, '否'],
        [0.593, 0.042, '否'],
        [0.719, 0.103, '否']
    ]

    # 将是否为好瓜的字符替换为数字。'是'换为1，'否'换为-1。
    for i in range(len(dataSet)):
        if dataSet[i][-1] == '是':
            dataSet[i][-1] = 1
        else:
            dataSet[i][-1] = -1

    return np.array(dataSet)


def calErr(dataSet, feature, threshVal, inequal, D):
    """
    计算数据带权值的错误率。
    :param dataSet:     [密度，含糖量，好瓜]
    :param feature:     [密度，含糖量]
    :param threshVal:
    :param inequal:     'lt' or 'gt. (大于或小于）
    :param D:           数据的权重。错误分类的数据权重会大。
    :return:            错误率。
    """
    DFlatten = D.flatten()   # 变为一维
    errCnt = 0
    i = 0
    if inequal == 'lt':
        for data in dataSet:
            if (data[feature] <= threshVal and data[-1] == -1) or \
               (data[feature] > threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]
            i += 1
    else:
        for data in dataSet:
            if (data[feature] >= threshVal and data[-1] == -1) or \
               (data[feature] < threshVal and data[-1] == 1):
                errCnt += 1 * DFlatten[i]
            i += 1
    return errCnt


def buildStump(dataSet, D):
    """
    通过带权重的数据，建立错误率最小的决策树桩。
    :param dataSet:
    :param D:
    :return:    包含建立好的决策树桩的信息。如feature，threshVal，inequal，err。
    """
    m, n = dataSet.shape
    bestErr = np.inf
    bestStump = {}
    for i in range(n-1):                    # 对第i个特征
        for j in range(m):                  # 对第j个数据
            threVal = dataSet[j][i]
            for inequal in ['lt', 'gt']:    # 对于大于或等于符号划分。
                err = calErr(dataSet, i, threVal, inequal, D)  # 错误率
                if err < bestErr:           # 如果错误更低，保存划分信息。
                    bestErr = err
                    bestStump["feature"] = i
                    bestStump["threshVal"] = threVal
                    bestStump["inequal"] = inequal
                    bestStump["err"] = err

    return bestStump

def predict(data, bestStump):
    """
    通过决策树桩预测数据
    :param data:        待预测数据
    :param bestStump:   决策树桩。
    :return:
    """
    if bestStump["inequal"] == 'lt':
        if data[bestStump["feature"]] <= bestStump["threshVal"]:
            return 1
        else:
            return -1
    else:
        if data[bestStump["feature"]] >= bestStump["threshVal"]:
            return 1
        else:
            return -1


def AdaBoost(dataSet, T):
    """
    每学到一个学习器，根据其错误率确定两件事。
    1.确定该学习器在总学习器中的权重。正确率越高，权重越大。
    2.调整训练样本的权重。被该学习器误分类的数据提高权重，正确的降低权重，
      目的是在下一轮中重点关注被误分的数据，以得到更好的效果。
    :param dataSet:  数据集。
    :param T:        迭代次数，即训练多少个分类器
    :return:         字典，包含了T个分类器。
    """
    m, n = dataSet.shape
    D = np.ones((1, m)) / m                      # 初始化权重，每个样本的初始权重是相同的。
    classLabel = dataSet[:, -1].reshape(1, -1)   # 数据的类标签。
    G = {}      # 保存分类器的字典，

    for t in range(T):
        stump = buildStump(dataSet, D)           # 根据样本权重D建立一个决策树桩
        err = stump["err"]
        alpha = np.log((1 - err) / err) / 2      # 第t个分类器的权值
        # 更新训练数据集的权值分布
        pre = np.zeros((1, m))
        for i in range(m):
            pre[0][i] = predict(dataSet[i], stump)
        a = np.exp(-alpha * classLabel * pre)
        D = D * a / np.dot(D, a.T)

        G[t] = {}
        G[t]["alpha"] = alpha
        G[t]["stump"] = stump
    return G


def adaPredic(data, G):
    """
    通过Adaboost得到的总的分类器来进行分类。
    :param data:    待分类数据。
    :param G:       字典，包含了多个决策树桩
    :return:        预测值
    """
    score = 0
    for key in G.keys():
        pre = predict(data, G[key]["stump"])
        score += G[key]["alpha"] * pre
    flag = 0
    if score > 0:
        flag = 1
    else:
        flag = -1
    return flag


def calcAcc(dataSet, G):
    """
    计算准确度
    :param dataSet:     数据集
    :param G:           字典，包含了多个决策树桩
    :return:
    """
    rightCnt = 0
    for data in dataSet:
        pre = adaPredic(data, G)
        if pre == data[-1]:
            rightCnt += 1
    return rightCnt / float(len(dataSet))


def main():

    data = dataload( r'.\watermelon_3a.csv')
    print('数据集的样本总个数为', data.shape[0])
    print('有', data.shape[1] - 1, '个属性')

    train_, test_ = data_split(data, 0.7, True)

    # dataSet = getDataSet()
    # for t in [3, 5, 11]:   # 学习器的数量
    t=21
    G = AdaBoost(train_, t)
    print(f"G{t} = {G}")
    print(calcAcc(test_, G))


if __name__ == '__main__':
    main()