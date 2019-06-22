import numpy as np
# import kNN
# import trees
# import bayes
# import feedparser
# import logRegres
import svmMLiA


# group, labels = kNN.createDataSet()
# print(kNN.classify0([0, 0], group, labels, 3))
#
# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)
#
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1],datingDataMat[:, 2])
# plt.show()
#
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
#            15.0 * np.array(datingLabels), 15 * np.array(datingLabels))
# plt.show()
#
# kNN.datingClassTest()
# kNN.classifyPerson()
# kNN.handwritingClassTest()

# myDat, labels = trees.createDataSet()
# trees.calcShannonEnt(myDat)
# trees.splitDataSet(myDat, 0, 1)
# trees.chooseBestFeatureToSplit(myDat)
# trees.createTree(myDat, labels)

# trees.createPlot()
# myTree = trees.retrieveTree(0)
# myTree['no surfacing'][3] = 'maybe'
# trees.createPlot(myTree)
# test_class = trees.classify(myTree, labels, [1, 1])
# print(test_class)
#
# fr = open('lenses.txt')
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# lensesTree = trees.createTree(lenses, lensesLabels)
# trees.createPlot(lensesTree)


# listOPosts, listClasses = bayes.loadDataSet()
# myVocavList = bayes.createVocabList(listOPosts)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVocavList, postinDoc))
# # print(trainMat)
# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
#
# # print(pAb)
# # print(p0V)
# # print(p1V)
#
# print(np.array(trainMat).sum(axis=0).sum())

# bayes.testingNB()
# bayes.spamTest()
# ny = feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
# sf = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss')
# # vocabList, pSF, PNY = bayes.localWords(ny, sf)
# bayes.getTopWords(ny, sf)

# dataArr, labelMat = logRegres.loadDataSet()
# weights = logRegres.gradAscent(dataArr, labelMat)
# logRegres.plotBestFit(weights.getA())
# weights = logRegres.stocGradAscent0(dataArr, labelMat)
# print(weights)
# logRegres.plotBestFit(weights)
# weights = logRegres.stocGradAscent1(dataArr, labelMat)
# print(weights)
# logRegres.plotBestFit(weights)

# logRegres.multiTest()

# svmMLiA.selectJrand(5, 10)
# svmMLiA.clipAlpha(aj=3, H=1, L=-1)
# svmMLiA.clipAlpha(aj=-3, H=1, L=-1)
dataArr, labelArr = svmMLiA.loadDataSet('D:/Downloads/machinelearninginaction/CH06/testSet.txt')
print(labelArr)


b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
# for i in range(100):
#     if alphas[i] > 0.0:
#         print(dataArr[i], labelArr[i])
