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
    aj大于H时，调整为H；aj小于L时调整为L
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
    dataMatrix = np.mat(dataMatIn)  # 将数据集转成numpy的矩阵
    labelMat = np.mat(classLabels).transpose()  # 将类别标签转为numpy矩阵，并转置成列向量
    b = 0  # 偏置初始化为0
    m, n = np.shape(dataMatrix)  # 数据集的行与列，m为样本数目，n为特征数目
    alphas = np.mat(np.zeros((m, 1)))  # alpha列矩阵初始化为0,alpha是拉格朗日乘子向量，长度与样本量相等
    iter = 0  # 循环次数初始化为0
    while(iter < maxIter):
        alphaPairsChanged = 0  # 初始化为0，用来记录alpha是否已经进行优化
        for i in range(m):
            # w = sum(alpha_i * y_i * x_i)  i=1,2,...,m,  alpha_i是一个数值，y_i是一个数值，
            # x_i是一个n维向量，w是一个n维向量
            # fXi = w * x_i + b
            # np.multiply()：数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
            # np.dot()：对于秩为1的数组，执行对应位置相乘，然后再相加；
            # 对于秩不为1的二维数组，执行矩阵乘法运算；超过二维的可以参考numpy库介绍。
            w = np.dot(np.multiply(alphas, labelMat).T, dataMatrix)  # n维向量
            fXi = float(np.dot(w, dataMatrix[i, :].T)) + b
            # fXi = float(np.multiply(alphas, labelMat).T *
            #             (dataMatrix * dataMatrix[i, :].T)) + b  # 第i个样本的预测类别
            Ei = fXi - float(labelMat[i])  # 第i个样本预测误差
            #  如果
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)  # 随机选取另一个alpha，alpha[j]
                fXj = float(np.multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b  # 第j个样本的预测类别
                Ej = fXj - float(labelMat[j])  # 第j个样本预测误差
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 将alpha[j]调整到0和C之间
                # y_i != y_j 时,L与H的取值
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                # y_i = y_j时,L与H的取值
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(0, alphas[j] + alphas[i])

                # 如果L等于H，就不做任何改变，直接执行continue语句，跳出本次循环，进入下一次循环
                if L == H:
                    print('L==H')
                    continue

                #  eta是alpha[j]的最优修改量
                # eta = 2* <x_i, x_j> - <x_i, x_i> - <x_j, x_j> 尖括号表示两个向量的内积
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                # eta >=o 时退出本次循环，进入下一次循环
                if eta >= 0:
                    print('eta>=0')
                    continue
                # 计算新的alpha[j]
                # 未经剪辑的alpha_j = alpha_j - y_j(E_i - E_j)/eta
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 经过剪辑的alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 如果alpha[j]变化很小，就跳出本次循环，进入下一次循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                # alpha[i]向相反方向改变同样的大小
                # alpha_i = alpha_i + y_i * y_j *(alpha_j^old - alpha_j^new)
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # alpha[i]对应的b值
                b1 = b - Ei - \
                     labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                # alpha[j]对应的b值
                b2 = b - Ej - \
                     labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i] and (C > alphas[i])):
                    b = b1
                elif (0 < alphas[j] and (C > alphas[j])):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print(
                    'iter: %d i: %d,pairs changed %d' %
                    (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


# 利用完整的Platt SMO算法加速优化
"""
完整版Platt SMO的支持函数
"""


class optStruct:
    """
    构建一个仅包含init方法的optStruct类
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    """
    计算第k个样本点的预测误差
    :param oS:
    :param k:
    :return:
    """
    #  根据支持向量预测模型，计算预测分类
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T *
                oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])  # 预测分类减去实际分类，得出预测误差
    return Ek


def selectJ(i, oS, Ei):
    """
    用于选择第二个alpha值，目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。
    返回选择的索引j，已经对应的预测误差
    :param i:
    :param oS:
    :param Ei:
    :return:
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回的是非零E值对应的索引
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 完整的Platt SMO算法中的优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)  # 就散第i个样本的预测误差
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) \
            or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        #  y_i = y_j,y_i != y_j 两种情况，L和H的取值范围
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L==H')
            return 0

        #
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * \
            oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - \
            oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
            oS.K[i, j]
        b2 = oS.b - Ej - \
            oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
            oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# 完整版Platt SMO的外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(
        classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print(
                'fullSet iter: %d i:%d, pairs changed %d' %
                (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print(
                    'non-bound, iter: %d i: %d, pairs changed %d' %
                    (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas


#  核转换函数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError(
            'Houston We Have a Problem -- That Kernel is not recognized')
    return K


# 利用核函数进行分类的径向基测试函数
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet(
        './machinelearninginaction/CH06/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % np.shape(sVs)[0])
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
        print('the training error rate is: %f' % (float(errorCount) / m))
        dataArr, labelArr = loadDataSet(
            './machinelearninginaction/CH06/testSetRBF2.txt')
        errorCount = 0
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).transpose()
        m, n = np.shape(dataMat)
        for i in range(m):
            kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(labelArr[i]):
                errorCount += 1
        print('the test error rate is: %f' % (float(errorCount) / m))


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 基于SVM的手写数字识别
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m =len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('./trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Suport Vectors' % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount)/m))
    dataArr, labelArr = loadImages('./testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is: %f' % (float(errorCount)/m))

