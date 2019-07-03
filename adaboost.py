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
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestSrump = {}
    bestclasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(m):
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
                if weightedError < minError:
                    minError = weightedError
                    bestclasEst = predictedVals.copy()
                    bestSrump['dim'] = i
                    bestSrump['thresh'] = threshVal
                    bestSrump['ineq'] = inequal
    return bestSrump, minError, bestclasEst


if __name__ == '__main__':
    datMat, classLabels = leadSimpData()
    # print(datMat)
    # print(classLabels)
    retArray = stumpClassify(datMat, 0, 1.5, 'lt')
    print(datMat)
    print(retArray)
