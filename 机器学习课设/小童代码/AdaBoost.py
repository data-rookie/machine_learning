import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

data = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.430, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]]
column = ['density', 'sugar_rate', 'label']
dataSet = pd.DataFrame(data, columns=column)


X = mat(dataSet[['density', 'sugar_rate']])
y = dataSet['label'].values


def stumpClassify(dataMtrix, dimen, threshVal, threshIneq):  # dim为那一列（那个属性）进行分类 threshVal 是阈值
    retArray = ones((shape(dataMtrix)[0], 1))  # 把类别都先设置为1
    if threshIneq == 'lt':
        retArray[dataMtrix[:, dimen] <= threshVal] = 0
    else:
        retArray[dataMtrix[:, dimen] > threshVal] = 0

    return retArray


# 单层决策树 D为权重
def buildStump(dataArr, classLabels, D):
    dataMAtrix = mat(dataArr)
    labelMat = mat(classLabels).T

    m, n = shape(dataMAtrix)  # 行数和列数
    numSteps = 10.0  # 每个特征迭代的步数
    bestStump = {}  # 保存最好的分类的信息
    bestClassEst = mat(zeros((m, 1)))  # 保存最好的
    minError = inf  # 误差的值

    for i in range(n):  # 循环每个属性
        rangeMin = dataMAtrix[:, i].min()
        rangeMax = dataMAtrix[:, i].max()  # 每个列的最大最小值
        stepSize = (rangeMax - rangeMin) / numSteps  # 每一步的长度

        for j in range(0, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize  # 每一步划分的阈值
                predictedVals = stumpClassify(dataMAtrix, i, threshVal, inequal)  # 预测的分类值
        # for j in range(len(dataMAtrix[:, i]) - 1):
        #     for inequal in ['lt', 'gt']:
        #         threshVal = (float(dataMAtrix[j, i]) + float(dataMAtrix[j+1, i])) / 2
        #         predictedVals = stumpClassify(dataMAtrix, i, threshVal, inequal)

                errArr = mat(ones((m, 1)))  # 初始化认为分类全错了
                errArr[predictedVals == labelMat] = 0  # 分正确的变为0
                weightError = D.T * errArr  # errArr为1的位置说明是分错的，则对应位子的样本的权重相加

                # 保存信息
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictedVals.copy()  # 保存最好的分类预测值
                    bestStump['dim'] = i  # 最好的列（最好的属性）
                    bestStump['thred'] = threshVal  # 属性列最好的分割阈值
                    bestStump['ineq'] = inequal  # 最好的符号
    return bestStump, minError, bestClassEst


# 基于单层的决策树AdaBoost训练函数 numIt指的迭代的次数
def AdaBoost(dataArr, classLabels, numIt):
    weakClassArr = []  # 保存每次迭代器的信息
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / float(m))  # 初始化权重
    aggClassESt = mat(zeros((m, 1)))  # 累计每次分类的结果
    for _ in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1 - error) / error))  # 函数的权重
        bestStump['alpha'] = alpha  # 记录权重
        weakClassArr.append(bestStump)  # 保存每一轮的结果信息

        # 更新权重
        expon = []
        for predict_label, real_label in zip(classEst, classLabels):
            if int(predict_label[0]) == int(real_label):
                expon.append([-alpha])
            else:
                expon.append([alpha])
        D = multiply(D, exp(expon))
        D = D / D.sum()  # 归一化

        # 累计每个函数的预测值 （权重 * 预测的类）
        aggClassESt += alpha * classEst
        errorRate = 1.0 * sum(sign(aggClassESt) != mat(classLabels).T) / m  # 预测的和真实的标签不应样的个数/总的个数
        if errorRate == 0.0:
            break  # 如果提前全部分类真确，则提前停止循环
    print(weakClassArr)
    return weakClassArr


# 分类
def predict(datatoClass, classifierArr):  # 大分类的数据，训练好的分类器
    dataMatrix = mat(datatoClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))  # 保存预测的值 m个数据m个预测值
    for i in range(len(classifierArr)):  # 循环遍历所有的分类器
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thred'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst  # 第i个分类器的预测值*他的权重
    return sign(aggClassEst)


# 绘制数据集
def plotData(X, Y, clf):
    X1, X2 = [], []
    Y1, Y2 = [], []
    for data, label in zip(X, Y):
        if label > 0:
            X1.append(data[0, 0])
            Y1.append(data[0, 1])
        else:
            X2.append(data[0, 0])
            Y2.append(data[0, 1])

    x = linspace(0, 0.8, 100)
    y = linspace(0, 0.6, 100)
    for weakClasser in clf:
        # print(weakClasser.attribute, weakClasser.partition)
        z = [weakClasser['thred']] * 100
        if weakClasser['dim'] == 0:
            plt.plot(z, y)
        else:
            plt.plot(x, z)

    plt.scatter(X1, Y1, marker='+', label='好瓜', color='b')
    plt.scatter(X2, Y2, marker='_', label='坏瓜', color='r')

    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.xlim(0, 0.8)  # 设置x轴范围
    plt.ylim(0, 0.6)  # 设置y轴范围
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    plt.show()


sizes = [3, 5, 11]
for size in sizes:
    weakClassArr = AdaBoost(X, y, size)
    predictLabels = predict(X, weakClassArr)

    accuracy = 0
    for y1, y2 in zip(y, predictLabels):
        if y1 == y2:
            accuracy += 1

    print('Size:', size)
    print('Accuracy:', accuracy, '/', len(y))
    print('')

    plotData(X, y, weakClassArr)
