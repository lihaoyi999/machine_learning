# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/13 15:55

import numpy as np
from math import log
import pysnooper
import operator
import matplotlib.pyplot as plt


def calcShannonEnt(dataSet):
    """
    计算数据集的熵
    :param dataSet: 数据集
    :return: 熵
    """
    numEntries = len(dataSet)  # 数据集样本数
    labelCounts = {}  # 字典，存储类别标签值以及对应的样本数
    # 遍历每个样本点
    for featVec in dataSet:
        # 最后一列为类别标签，获取当前样本的类别标签
        currentLabel = featVec[-1]
        # 如果当前分类标签在字典中不存在，则初始化计数值为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 当前标签累加1
        labelCounts[currentLabel] += 1
    # 初始化熵为0
    shannonEnt = 0.0
    # 遍历字典
    for key in labelCounts:
        # 计算当前类别的概率值，等于该类别的样本数除以总样本数
        prob = float(labelCounts[key]) / numEntries
        # 根据熵的定义公式，计算熵，累加每个类别的信息值
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征以及该特征的一个取值划分数据集
    :param dataSet: 待划分的数据集，最后一列为类别标签
    :param axis: 划分数据集的特征的索引
    :param value: 该特征的取值
    :return: 返回满足该特征取值、剔除该特征后的数据集
    """
    retDataSet = []
    # 遍历每个样本点
    for featVec in dataSet:
        if featVec[axis] == value:
            # 在样本数据中剔除该划分特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最佳划分特征
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数目，数据集中包含分类标签列，因此减1
    baseEntropy = calcShannonEnt(dataSet)  # 未划分前数据集的熵
    bestInfoGain = 0.0  # 信息增益，初始化为0
    bestFeature = -1  # 最佳划分特征的索引
    # 遍历数据集的每个特征
    for i in range(numFeatures):
        # 遍历数据集dataSet的样本点example，输出样本点的第i特征值example[i]
        # 列表推导式
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 取得不重复值
        newEntropy = 0.0  # 划分后的熵，初始化为0
        # 遍历当前划分特征的取值
        for value in uniqueVals:
            # 按特征i和取值value划分数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算当前特征取值的子集的权重，即当前划分的数据集的样本数除以数据集的样本数
            prob = len(subDataSet) / float(len(dataSet))
            # 累加各个子集的熵 乘以子集的权重，得到数据集基于当前特征的条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算当前特征的信息增益
        infoGain = baseEntropy - newEntropy
        # 如果当前特征的信息增益大于最佳信息增益，则将当前信息增益置为最佳信息增益
        # 将最佳分类特征索引置为当前的特征索引
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    根据类别标签数据列表，统计标签值及对应的计数，返回计数值最大的分类标签
    :param classList: 标签数据列表
    :return: 返回计数值最大的分类标签
    """
    classCount = {}  # 创建一个字典，存储类别标签及计数值
    # 遍历类别列表
    for vote in classList:
        # 如果类别标签不在字典中存在，则加入这个类别标签，计数值置为0
        if vote not in classCount.keys():
            classCount[vote] = 0
        # 累加计数
        classCount[vote] += 1
    # 对类别标签及计数值的字典排倒序
    # key=operator.itemgetter(1)  指定排序的列，这是第二列，即按计数值排序
    sortedClassCount = sorted(
        classCount.items(),
        key=operator.itemgetter(1),
        reverse=True)
    return sortedClassCount[0][0]


# @pysnooper.snoop()
def createTree(dataSet, labels):
    """
    生成树
    :param dataSet: 数据集，最后一列为类别取值
    :param labels: 类别标签列表，对应类别取值的实际意义
    :return:
    """
    # 获取数据集中的最后一列，类别取值
    classList = [example[-1] for example in dataSet]
    # 如果数据集的类别都相同则停止继续划分，并返回当前类别取值
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时，返回出现次数最多的类别
    # 即划分的数据集只有类别一列时
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 数据集的最佳划分特征索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最佳划分特征的类别标签
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


decisionNode = dict(boxstyle="sawtooth", fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(
                secondDict[key],
                (plotTree.xOff,
                 plotTree.yOff),
                cntrPt,
                leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # splitDataSet(dataSet, 0, 1)
    # chooseBestFeatureToSplit(dataSet)
    majorityCnt([1,2,3,3,2,1,2,3,3,1,2,2])