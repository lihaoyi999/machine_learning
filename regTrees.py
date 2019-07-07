# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       lihaoyi
# Date:         2019/7/7
# -------------------------------------------------------------------------------


from numpy import *


class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


# CART算法的代码实现
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


# 回归树的切分函数
def regLeaf(dataSet):
    """
    生成叶节点的模型，在回归树中，就是目标变量的均值
    :param dataSet:
    :return:
    """
    return mean(dataSet[:, -1])


def regErr(dataSet):
    """
    误差估计函数，在给定的数据上计算目标变量的平方误差
    :param dataSet:
    :return: 返回的是总方差，所以用均方差乘以数据集的样本数
    """
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    找到数据的最佳二元切分方式，如果找不到一个好的二元切分，
    该函数返回None并同时调用createTree()方法来产生叶节点，
    叶节点的值也将返回None。
    :param dataSet: 待切分的数据集
    :param leafType:
    :param errType:
    :param ops: 设定的tolS和tolN，是用户指定的参数，用于控制函数的停止时机。
    tolS是容许的误差下降值，tolN是切分的最少样本数
    :return:
    """
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S =errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitval in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitval)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitval
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


