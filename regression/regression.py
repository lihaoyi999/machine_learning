# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/05 16:26


from numpy import *
import pysnooper

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr =open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# 简单回归
def standRegres(xArr, yArr):
    """
    计算最佳拟合回归直线
    :param xArr: x
    :param yArr: y
    :return: 返回拟合直线的回归系数
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断xTx的行列式是否为0，如果为0，计算逆矩阵就会出错
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    # 回归系数
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 局部加权回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]  # 样本数
    weights = mat(eye((m)))  # 创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        if linalg.det(xTx) == 0.0:
            print('This matrix is singular, cannot do inverse')
            return
        ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    给定测试数据，计算出对应的预测值yhat
    :param testArr:  测试数据
    :param xArr:  训练集样本
    :param yArr:  训练集样本对应的y值
    :param k:  控制衰减的速度
    :return:
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    # eye 产生一个单位矩阵，维度与特征数一致，不包含常数项
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    # 标准化处理，减均值，再除以方差
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)  # y的均值
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)  # x的均值
    xVar = var(xMat, 0)  # x的方差
    xMat = (xMat - xMeans)/xVar  # x进行标准化

    # 设置30个lambda的取值，以指数级变化，
    # 可以看出lambda取非常小的值时和取非常大的值时分别对结果造成的影响
    # 从中选取最优的lambda
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))  # 存储回归系数
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return  inMat

# 逐步线性回归算法
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    逐步线性回归算法
    :param xArr: 样本数据集
    :param yArr: 预测变量
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    """
    xMat= mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, ))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf

        # 对每个特征进行迭代
        for j in range(n):
            # 分别计算增加和减少该特征对误差的影响
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * mat(wsTest).T
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat [i, :] = ws.T
    return returnMat



if __name__ == '__main__':
    # xArr, yArr = loadDataSet('./machinelearninginaction/CH08/ex0.txt')
    # ws = standRegres(xArr, yArr)
    # # print(yArr[0])
    # # print(lwlr(xArr[0], xArr, yArr,1.0))
    # # print(lwlr(xArr[0], xArr, yArr, 0.001))
    # yhat = lwlrTest(xArr, xArr, yArr, 0.1)
    # xMat = mat(xArr)
    # srtInd = xMat[:, 1].argsort(0)
    # xSort = xMat[srtInd][:, 0, :]
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:, 1], yhat[srtInd])
    # ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    # plt.show()

    abX, abY = loadDataSet('./machinelearninginaction/CH08/abalone.txt')
    # ridgeWeights = ridgeTest(abX, abY)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()
    stageWise(abX, abY, 0.001, 5000)