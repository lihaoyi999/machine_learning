import numpy as np
import kNN
import trees


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

myDat, labels = trees.createDataSet()
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

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
trees.createPlot(lensesTree)