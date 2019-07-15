# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/03 15:13
import numpy as np

def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.0],
                         [1.0, 1.0],
                         [2.0, 1.0]])
    classLabels = [1., 1., -1., -1., 1]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过阈值对数据进行分类
    :param dataMatrix: 数据集
    :param dimen:维度
    :param threshVal: 阈值
    :param threshIneq: 比较操作符，小于：'lt' 或 大于：'gt'
    :return: 分类的类别
    """
    # 将数组对应的类别均初始化为1
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    # 比较运算符为lt时，将所有满足不等式的元素设置为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildSrump(dataArr, classLabels, D):
    """
    建立一个有最小错误率的最佳单层决策树
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param D: 每个样本点的权重值
    :return: 返回最佳单层决策树，及相应的分类误差率，对应的分类
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)  # m为样本数，n为特征数
    numSteps = 10.0  # 设置步数
    bestSrump = {}  # 字典，存储最佳单层决策树
    bestclasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 最小错误率初始化为无穷大

    # 对每个特征循环处理
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()  # 第i个特征的最小值
        rangeMax = dataMatrix[:, i].max()  # 第i个特征的最大值
        stepSize = (rangeMax - rangeMin)/numSteps  # 根据该特征的极差和步数计算步长

        # range(-1, int(numSteps) + 1) ： -1,0,1，。。。，numsteps，可将阈值设置为整个取值范围之外
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)  # 设置比较大小阈值
                # 通过阈值对数据进行分类，返回预测分类类别
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 错误向量，即样本分类是否错误的向量，均初始化为1，向量的和为预测分类错误的样本数
                errArr = np.mat(np.ones((m, 1)))
                # 样本的预测类别等于实际类别时，将分类准确的样本的置为0
                errArr[predictedVals == labelMat] = 0

                # 此时，样本的权重向量D在计算错误率时才会产生作用
                # 计算错误率（即分类误差率），错误向量errArr和权重向量D的相应元素相乘并求和，是一个数值
                # 分类误差率等于被错误分类的样本权值之和
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh inequal: %s, the wieghted error is %.3f' % (i, threshVal, inequal, weightedError))
                # 如果错误率低于最小minError，则将当前的单层决策树设置为最佳单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestclasEst = predictedVals.copy()
                    bestSrump['dim'] = i  # 用来分类的特征索引
                    bestSrump['thresh'] = threshVal  # 用来分类的阈值
                    bestSrump['ineq'] = inequal  # 用来分类的比较运算符
    return bestSrump, minError, bestclasEst  # 返回最佳单层决策树，及相应的分类误差率，对应的分类


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """

    :param dataArr: 数据集
    :param classLabes: 类别标签
    :param numIt: 迭代次数
    :return: 返回具有最小错误率的单层决策树，同时返回有最小错误率和估计的类别向量
    """
    weakClassArr = []  # 存储每次迭代的弱分类器
    m = np.shape(dataArr)[0]  # 训练集的样本数
    D = np.mat(np.ones((m, 1)) / m)  # 每个样本的权值，是一个长度为m的向量，均初始化为1/m
    # 记录每个样本点的类别估计累计值，样本最终的预测值需要通过这个值来判断
    aggClassEst = np.mat(np.zeros((m, 1)))

    # 遍历次数，每次遍历都会生成一个弱分类器
    for i in range(numIt):
        bestStump, error, classEst = buildSrump(dataArr, classLabels, D)  # 最佳单层决策树
        # print("D:", D.T)

        # 计算当前决策树的alpha值
        # alpha = 1/2 * ln((1-e)/e), max(error, 1e-16) 是防止error为0时溢出
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha  # 当前的单层决策树的权重
        weakClassArr.append(bestStump)  # 将当前的单层决策树存入weakClassArr中
        # print('classEst: ', classEst.T)

        # 更新训练集的样本权重，作为下一次循环的训练集的样本权重
        # D_(m+1) = ( w_(m+1,1), w_(m+1,2), ..., w_(m+1,i), ..., w_(m+1,N))
        # w_(m+1, i) = w_(m, i)* exp(- alpha_m * y_i * G_m(x_i)) / Z_i
        # Z_i = sum(D_(m+1))
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 计算每个样本点的类别估计累计值
        aggClassEst += alpha * classEst
        # print("aggClassEst:", aggClassEst.T)

        # sign符号函数，大于0时输出+1，小于0时输出-1，等于0时输出0
        # 判断样本点的类别估计累计值的类别与真实类别是否相等，计算预测类别错误的样本数
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        # 错误率=预测错误的样本数/总样本数
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate, "\n")

        # 如果错误率等于0，跳出循环
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    :param datToClass: 需要分类的数组数据
    :param classifierArr: 多个弱分类器组成的数组
    :return: 返回分类类别
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    # 遍历所有的弱分类器
    for i in range(len(classifierArr)):
        # 计算每个弱分类器的预测值
        classEst = stumpClassify(dataMatrix,
                                 classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        # 累计的样本预测值
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)


def loadDataSet(fileName):      # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    """
    绘制ROC曲线
    横轴为伪正例的比例，即假阳率=FP/(FP+TN)
    纵轴为真正例的比例，即真阳率=TP/(TP+FN)
    :param predStrengths: 分类器的预测强度，及各个样本的预测概率
    :param classLabels: 类别标签
    :return:
    """
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # 计算AUC
    numPosClas = sum(np.array(classLabels) == 1.0)  # 正例的数目
    yStep = 1/float(numPosClas)  # y轴的步长
    xStep = 1/float(len(classLabels)-numPosClas)  # x轴的步长
    sortedIndicies = predStrengths.argsort()  # 排序索引, 从小到大排序
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历所有值，在每个点绘制一个线段
    for index in sortedIndicies.tolist()[0]:
        # 如果classLabels[index]是正例，y轴坐标下移一个步长，x轴坐标不变
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        # 如果classLabels[index]是反例，x轴坐标下移一个步长，y轴坐标不变
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)

    ax.plot([0, 1], [0, 1], 'b--')  # 绘制[0,0]到[1,1]对角线，黑色虚线
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    # 计算AUC，按矩形的面积算法计算，
    print("the Area Under the Curve is: ", ySum*xStep)


if __name__ == '__main__':
    # datMat, classLabels = loadSimpData()
    # print(datMat)
    # print(classLabels)
    # retArray = stumpClassify(datMat, 0, 1.5, 'lt')
    # print(datMat)
    # # print(retArray)
    # D = np.mat(np.ones((5, 1))/5)
    # buildSrump(datMat, classLabels, D)
    # classifierArr = adaBoostTrainDS(datMat, classLabels, 10)
    # print(classifierArr)
    # datMat, classLabels = loadDataSet('./machinelearninginaction/CH07/horseColicTraining2.txt')
    # TestArr, TestLabelArr = loadDataSet('./machinelearninginaction/CH07/horseColicTest2.txt')
    # classifierArr = adaBoostTrainDS(datMat, classLabels, 1000)
    # prediction10 = adaClassify(TestArr, classifierArr)

    datMat, classLabels = loadDataSet('./machinelearninginaction/CH07/horseColicTraining2.txt')
    classifierArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 10)
    plotROC(aggClassEst.T, classLabels)

