# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/20 8:50


import numpy as np
import pysnooper


def loadDataSet(filename):
    """
    导入数据
    :param filename: 数据文件路径
    :return:
    """
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# @pysnooper.snoop()


def selectJrand(i, m):
    """
    只要函数值j不等于输入值i，函数就会进行随机选择
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    """
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


# @pysnooper.snoop()
def clipAlpha(aj, H, L):
    """
    aj大于0时，调整为C；aj小于L时调整为L
    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn:  数据集
    :param classLabels: 类别标签
    :param C: 常数，松弛变量
    :param toler: 容错率
    :param maxIter: 取消前最大循环次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn) # 将数据集转成numpy的矩阵
    labelMat = np.mat(classLabels).transpose() # 将类别标签转为numpy矩阵，并转置成列向量
    b = 0 # 偏置初始化为0
    m, n = np.shape(dataMatrix)  # 数据集的行与列
    alphas = np.mat(np.zeros((m, 1)))  # alpha列矩阵初始化为0
    iter = 0  # 循环次数初始化为0
    while(iter < maxIter):
        alphaPairsChanged = 0  # 初始化为0，用来记录alpha是否已经进行优化
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 第i个样本的预测类别
            Ei = fXi - float(labelMat[i])  # 预测误差
            #  如果
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选取另一个alpha值，alpha[j]
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b  # 预测类别
                Ej = fXj - float(labelMat[j])  # 预测误差
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(0, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta > 0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i] * \
                    dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i] * \
                     dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i] and (C > alphas[i])):
                    b = b1
                elif (0 < alphas[j] and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d,pairs changed %d' % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas
