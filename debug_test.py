# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/12 16:56

import pysnooper
import numpy as np
import operator
import kNN


@pysnooper.snoop()
def classify0(inX, dataSet, labels, k):
    """
    Parameters
    ----------
    inX: 输入向量
    dataSet: 训练集矩阵
    labels: 标签向量，训练集的分类标签
    k：选择最近邻居的数目
    :return:
    """
    dataSetSize = dataSet.shape[0]
    # np.tile 将变量复制成与训练集相同大小的矩阵
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort: 将distances中的元素从小到大排列，返回排序后对应的index(索引)
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        VoteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(VoteIlabel, 0)：在classCount中取键VoteIlabel的值，如果值不存在，返回默认值0
        classCount[VoteIlabel] = classCount.get(VoteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


group, labels = kNN.createDataSet()
classify0([0, 0], group, labels, 3)

