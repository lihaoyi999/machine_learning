# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/11 17:50


import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    Parameters
    ----------
    inX: 输入向量
    dataSet: 训练集
    labels: 标签向量，训练集的分类标签
    k：选择最近邻居的数目
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        VoteIlabel = labels[sortedDistIndicies[i]]
        classCount[VoteIlabel] = classCount.get(VoteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    读取文本数据文件，转为矩阵
    :param
    filename: 文件路径
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

