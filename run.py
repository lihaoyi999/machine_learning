import numpy as np
import kNN


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
kNN.handwritingClassTest()