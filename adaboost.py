# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/03 15:13
import numpy as np

def leadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.0],
                         [1.0, 1.0],
                         [2.0, 1.0]])
    classLabels = [1., 1., -1., -1., 1]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过阈值对数据进行分类
    :param dataMatrix:数据集
    :param dimen:维度
    :param threshVal:阈值
    :param threshIneq:比较操作符，小于：'lt' 或 大于：'gt'
    :return:
    """
    # 将数组对应的类别均初始化为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    # 比较运算符为lt时，将所有不满足不等式的元素设置为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildSrump(dataArr, classLabels, D):
    """
    建立一个单层决策树
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param D: 每个样本点的权重值
    :return:
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestSrump = {}
    bestclasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh inequal: %s, the wieghted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestclasEst = predictedVals.copy()
                    bestSrump['dim'] = i
                    bestSrump['thresh'] = threshVal
                    bestSrump['ineq'] = inequal
    return bestSrump, minError, bestclasEst


def adaBoostTrainDS(dataArr, classLabes, numIt=40):
    """

    :param dataArr: 数据集
    :param classLabes: 类别标签
    :param numIt: 迭代次数
    :return: 返回具有最小错误率的单层决策树，同时返回有最小错误率和估计的类别向量
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 样本的权值向量，均初始化为1/m
    aggClassEst = np.mat(np.zeros((m, 1)))  # 记录每个样本点的类别估计累计值
    for i in range(numIt):
        bestStump, error, classEst = buildSrump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst: ', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr


if __name__ == '__main__':
    datMat, classLabels = leadSimpData()
    # print(datMat)
    # print(classLabels)
    # retArray = stumpClassify(datMat, 0, 1.5, 'lt')
    # print(datMat)
    # # print(retArray)
    # D = np.mat(np.ones((5, 1))/5)
    # buildSrump(datMat, classLabels, D)
    classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    print(classifierArray)
