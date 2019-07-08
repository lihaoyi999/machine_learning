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
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    dataMat = mat(dataMat)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    根据特征、和阈值二元切分数据集，返回切分成的两个数据集
    :param dataSet: 待切分的数据集
    :param feature: 切分的特征索引
    :param value: 切分的阈值
    :return: 返回切分成的两个数据集
    """
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 回归树的切分函数
def regLeaf(dataSet):
    """
    生成叶节点的模型，在回归树中，就是目标变量的均值
    :param dataSet:
    :return:
    """
    return mean(mat(dataSet)[:, -1])


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
    :param dataSet: 待切分的数据集，最后一列为分类因变量y
    :param leafType: 均值
    :param errType: 样本总方差
    :param ops: 设定的tolS和tolN，是用户指定的参数，用于控制函数的停止时机。
    tolS是容许的误差下降值，tolN是切分的数据集的最少样本数
    :return: 返回最佳特粉特征索引及切分值
    """
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有因变量y的值相等，即数目为1，则不需要切分，直接返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)  # 样本数和特征数
    S = errType(dataSet)  # 切分前的样本总方差
    bestS = inf  # 最好切分对应的总方差
    bestIndex = 0  # 最好切分的特征索引
    bestValue = 0  # 最好切分的特征的切分值

    # 遍历每个特征
    for featIndex in range(n-1):
        # 遍历该特征的所有取值
        # set(dataSet[:, -1].T.tolist()[0])
        for splitval in set(dataSet[:, featIndex].T.tolist()[0]):
            # 根据该特征和取值切分数据集，得到切分好的两个数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitval)
            # 如果切分出来的两个数据集，只要其中一个数据集的样本数小于设定的tolN，则退出本次循环
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            # 计算切分后的样本总方差，两个切分后的数据集的总方差相加
            newS = errType(mat0) + errType(mat1)
            # 如果切分后的总方差小于最小总方差，则更新最小总方差为当前切分的总方差，
            # 最佳切分特征索引更新为当前切分的特征索引，最佳切分特征的切分值更新为当前切分的切分值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitval
                bestS = newS
    # 如果误差减少量小于阈值tolS，则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 按最佳切分对数据及进行切分
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分的数据集样本量很少，则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    生成回归树，是一个递归函数
    :param dataSet: 数据集
    :param leafType: 计算均值的函数，即建立叶节点的函数
    :param errType: 计算总方差的函数
    :param ops: 阈值
    :return:
    """
    # 选择数据集的最佳切分
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}  # 字典
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 根据最佳切分特征和切分值，二元切分数据集为两个数据集
    rSet, lSet = binSplitDataSet(dataSet, feat, val)
    # 分别对切分后的两个数据集递归调用createTree()进行切分
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


if __name__ == '__main__':
    # testMat = mat(eye(4))
    # mat0, mat1 = binSplitDataSet(testMat, 2, 0.5)
    # print('----------')
    # print(testMat)
    # print('----------')
    # print(mat0)
    # print('----------')
    # print(mat1)
    # print('----------')
    myDat = loadDataSet('./machinelearninginaction/CH09/ex0.txt')
    Tree = createTree(myDat)
    print(Tree)

# 后减枝函数prune()伪代码

# 基于已有的树切分测试数据：
#     如果存在任一子集是一棵树，则该子集递归减枝过程
#     计算将当前两个叶节点合并后的误差
#     计算不合并误差
#     如果合并会降低误差的话，就将叶节点合并

def isTree(obj):
    """
    判断obj对象是否为字典
    :param obj: obj对象
    :return: 返回bool值，True，False
    """
    return (type(obj).__name__ =='dict')


def getMean(tree):
    """
    是一个递归函数，从上到下遍历树，直到叶节点为止。
    如果找到两个叶节点，则计算他们的平均值。
    :param tree:
    :return:
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """

    :param tree: 待剪枝的数
    :param testData: 剪枝所需的测试数据
    :return:
    """
    # 判断测试数据是否为空
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 当左子树和右子树中有一颗不是叶节点时，将测试数据集切分为两个子数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        rSet, lSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果左子树是树，不是叶节点时，则使用lSet测试子集继续对该子树剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 如果右子树是树，不是叶节点时，则使用rSet测试子集继续对该子树剪枝
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], lSet)
    # 如果左右子树都不是树，都是叶节点时
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 根据树的切分特征和切分点，将测试数据划分为两个子数据集
        rSet, lSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算未合并的总方差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        # 计算合并后的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并后的总方差
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 如果合并后的总方差小于未合并的总方差，则返回合并后的均值
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree