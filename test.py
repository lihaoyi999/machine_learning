# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/01 14:53

from sklearn.metrics import roc_curve, auc
from adaboost import *
import matplotlib.pyplot as plt


datMat, classLabels = loadDataSet('./machinelearninginaction/CH07/horseColicTraining2.txt')
classifierArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 10)
plotROC(aggClassEst.T, classLabels)

# 计算真阳率，假阳率，AUC
fpr, tpr, threshold = roc_curve(classLabels, aggClassEst, pos_label=1) ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  # 计算auc的值

plt.figure()
lw = 2  # 预设线条宽度
plt.plot(fpr, tpr, color='darkorange', lw=lw,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # 绘制对角斜线
plt.xlim([0.0, 1.0])  # 设置x轴范围
plt.ylim([0.0, 1.0])  # 设置y轴范围
plt.xlabel('False Positive Rate')  # 设置x轴标题
plt.ylabel('True Positive Rate')  # 设置y轴标题
plt.title('Receiver operating characteristic example')  # 设置图表标题
plt.legend(loc="lower right")  # 设置图例放置的位置
plt.show()
