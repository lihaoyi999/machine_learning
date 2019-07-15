# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/18 15:42


import math
import numpy as np
import pysnooper


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """
    sigmoid函数
    :param inX:
    :return:
    """
    if inX >= 0:
        return 1.0 / (1.0 + np.exp(-inX))
    else:
        return np.exp(inX) / (1.0 + np.exp(inX))


# @pysnooper.snoop()
def gradAscent(dataMatIn, classLabels):
    """
    梯度上升算法
    :param dataMatIn:是一个Numpy的3维数组，每列代表不同的特征，每行代表每个训练样本
    :param classLabels:类别标签，1*100的行向量
    :return:
    """
    dataMatrix = np.mat(dataMatIn)  # m行n列的矩阵
    labelMat = np.mat(classLabels).transpose()  # 行向量转为列向量，长度为n
    m, n = np.shape(dataMatrix)  # dataMatrix的行列，m行n列
    alpha = 0.001  # 步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 变量的系数初始化为1，长度为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # h是一个长度为n列向量
        error = (labelMat - h)  # 是一组常数向量，长度为n
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# @pysnooper.snoop()
def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)# m为样本数目，n为特征数目
    alpha = 0.01
    weights = np.ones(n) # 特征系数初始化为1
    # 对样本进行遍历
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    随机梯度上升算法
    :param dataMatrix: 样本矩阵
    :param classLabels: 样本分类标签向量
    :param numIter: 迭代次数
    :return:
    """
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)# m为样本数目，n为特征数目
    weights = np.ones(n) # 特征系数初始化为1
    for j in range(numIter):
        dataIndex = list(range(m))
        # 对样本进行遍历
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01 # alpha在每次迭代时都作出调整，随着迭代次数增加而减小
            randIdex = int(np.random.uniform(0, len(dataIndex))) # 随机选取样本来更新系数
            h = sigmoid(sum(dataMatrix[randIdex] * weights))
            error = classLabels[randIdex] - h
            weights = weights + alpha * error * dataMatrix[randIdex]
            del(dataIndex[randIdex])
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # weights = weights.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X1')
    plt.show()


# 示例：从疝气病症预测病马的死亡率
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return  1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainingWeightd = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0.0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainingWeightd)) !=int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the avergage error rate is: %f" %(numTests, errorSum/float(numTests)))

