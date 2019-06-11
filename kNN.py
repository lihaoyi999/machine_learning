# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/11 17:50


import numpy as np
import operator


def createDataSet():
    group = np.array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize-1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[VoteIlabel]=classCount.get(voteIlabel,0)+1
    sorted()