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
    生成叶节点的模型，在回归树中，就是目标变量的均值，
    计算的是因变量y的均值
    :param dataSet:
    :return:
    """
    return mean(mat(dataSet)[:, -1])


def regErr(dataSet):
    """
    误差估计函数，在给定的数据上计算目标变量的平方误差，
    计算因变量y的总方差
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
    # 按最佳切分对数集进行切分
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


# 模型树
# 模型树的叶节点生成函数
def linearSolve(dataSet):
    """
    将数据集格式化成目标变量Y和自变量X，用于执行简单的线性回归
    :param dataSet: 数据集，不包含常数项，但是已包含目标变量Y
    :return: 返回回归系数，特征矩阵X（含常数项），目标变量
    """
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    XTX = X.T * X
    if linalg.det(XTX) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
    ws = XTX.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    """
    当数据不在需要切分的时候，生成叶节点的模型
    :param dataSet:
    :return: 返回值为回归系数
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    在给定的数据集上计算误差
    :param dataSet:
    :return: 返回yHat和Y之间的平方误差
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 用回归树进行预测
def regTreeEval(model, inDat):
    """
    计算回归树的叶节点预测值
    :param model: 叶节点对应的预测值
    :param inDat: 待预测的单个样本点，包含常数项
    :return: 返回样本点的预测值
    """
    # return float(model)
    return model

def modelTreeEval(model, inDat):
    """
    模型树的叶节点，计算样本点的预测值
    :param model: 叶节点对应的回归模型系数
    :param inDat: 待预测的单个样本点，包含常数项
    :return: 返回预测值
    """
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    单个样本点的预测，是一个递归函数
    :param tree: 回归树或模型树
    :param inData: 待预测的一个样本点
    :param modelEval: 回归树或模型树对应的计算预测值的函数
    :return: 返回样本点的预测值
    """
    # 如果给定的模型是叶节点，直接返回叶节点的预测值
    if not isTree(tree):
        return modelEval(tree, inData)
    # 如果待预测样本的属性值大于划分值，即满足左子树的分类条件，按左子树来进行处理
    if inData[tree['spInd']] > tree['spVal']:
        # 如果左子树是棵树，递归调用treeForeCast()函数，处理左子树
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        # 否则，左子树是一个叶节点，直接返回预测值
        else:
            return modelEval(tree['left'], inData)
    # 如果待预测样本的属性值小于等于划分值，即满足右子树的分类条件，按右子树来进行处理
    else:
        # 如果右子树是棵树，递归调用treeForeCast()函数，处理右子树
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        # 否则，右子树是一个叶节点，直接返回预测值
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    对多个样本的预测
    :param tree: 树模型，回归树或模型树
    :param testData: 待预测的样本点，是一个矩阵，包含常数项x0=1
    :param modelEval: 根据树模型指定相应的预测值计算函数
    :return: 返回预测值
    """
    m = len(testData)  # 待预测的样本数
    yHat = mat(zeros((m, 1)))  # 预测值
    # 遍历每个样本点，计算预测值
    for i in range(m):
        tfc = treeForeCast(tree, mat(testData[i]), modelEval)
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    """
    """
    # testMat = mat(eye(4))
    # mat0, mat1 = binSplitDataSet(testMat, 2, 0.5)
    # print('----------')
    # print(testMat)
    # print('----------')
    # print(mat0)
    # print('----------')
    # print(mat1)
    # print('----------')
    # myDat = loadDataSet('./machinelearninginaction/CH09/ex0.txt')
    # Tree = createTree(myDat)
    # print(Tree)

    # myDat2 = loadDataSet('./machinelearninginaction/CH09/ex2.txt')
    # myTree = createTree(myDat2, ops=(0, 1))
    # myDatTest = loadDataSet('./machinelearninginaction/CH09/ex2test.txt')
    # prune_tree = prune(myTree, myDatTest)
    # print(prune_tree)

    # myDat2 = loadDataSet('./machinelearninginaction/CH09/exp2.txt')
    # myTree = createTree(myDat2, modelLeaf, modelErr, ops=(1, 10))
    # print(myTree)

    # 创建一棵回归树
    trainMat = loadDataSet('./machinelearninginaction/CH09/bikeSpeedVsIq_train.txt')
    trainTest = loadDataSet('./machinelearninginaction/CH09/bikeSpeedVsIq_test.txt')
    myTree = createTree(trainMat, leafType=regLeaf, errType=regErr, ops=(1, 20))
    yHat = createForeCast(myTree, trainTest[:, 0])
    print(corrcoef(yHat, trainTest[:, 1], rowvar=0)[0, 1])

    # 创建一个模型树
    myTree = createTree(trainMat, leafType=modelLeaf, errType=modelErr, ops=(1, 20))
    yHat = createForeCast(myTree, trainTest[:, 0], modelTreeEval)
    print(corrcoef(yHat, trainTest[:, 1], rowvar=0)[0, 1])
    ws, X, Y = linearSolve(trainMat)
    print(ws)

