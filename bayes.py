# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/06/17 9:37


import numpy as np
import operator
import os


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless',
                    'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog',
                    'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 代表侮辱性文字，0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型，将每个次出现与否作为特征，每个词只出现一次
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 输出文档向量，向量的每一个元素为0或1，分别表示词汇表中的单词在输入文档中是否出现
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    """
    词袋模型，将每个次出现与否作为特征，每个词可出现多次
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: 输出文档向量，向量的每一个元素为0或1，分别表示词汇表中的单词在输入文档中是否出现
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: 文档矩阵
    :param trainCategory: 文档矩阵中每篇文档的类别标签多构成的向量，取值0或1
    :return:
    """
    numTrainDocs = len(trainMatrix) # 文档矩阵中的文档数目，即矩阵的行数
    numWords = len(trainMatrix[0]) # 文档矩阵中的词条数目，即矩阵的列数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 侮辱性文档的比例，class=1
    p0Num = np.ones(numWords) # 非侮辱性文档中，每个词条出现的文档数，初始化为1
    p1Num = np.ones(numWords) # 侮辱性文档中，每个词条出现的文档数，初始化为1
    p0Denom = 2.0 # 非侮辱性文档的总词条数目，初始化为2
    p1Denom = 2.0 # 侮辱性文档的总词条数目，初始化为2
    #遍历每个文档：
    for i in range(numTrainDocs):
        # 当分类为1时
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 每行向量的累加
            p1Denom += sum(trainMatrix[i]) # 每行向量的和的累加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vect2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vect2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vect2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
