# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/05 16:26


from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr =open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    """
    计算最佳拟合回归直线
    :param xArr: x
    :param yArr: y
    :return: 返回拟合直线的回归系数
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断xTx的行列式是否为0，如果为0，计算逆矩阵就会出错
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


if __name__ == '__main__':
    xArr, yArr = loadDataSet('./machinelearninginaction/CH08/ex0.txt')
    ws = standRegres(xArr, yArr)
    print(ws)
