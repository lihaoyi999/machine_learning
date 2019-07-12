# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/01 14:53

# from sklearn.metrics import roc_curve, auc
# from adaboost import *
# import matplotlib.pyplot as plt
#
#
# datMat, classLabels = loadDataSet('./machinelearninginaction/CH07/horseColicTraining2.txt')
# classifierArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 10)
# plotROC(aggClassEst.T, classLabels)
#
# # 计算真阳率，假阳率，AUC
# fpr, tpr, threshold = roc_curve(classLabels, aggClassEst, pos_label=1) ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  # 计算auc的值
#
# plt.figure()
# lw = 2  # 预设线条宽度
# plt.plot(fpr, tpr, color='darkorange', lw=lw,
#          label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # 绘制对角斜线
# plt.xlim([0.0, 1.0])  # 设置x轴范围
# plt.ylim([0.0, 1.0])  # 设置y轴范围
# plt.xlabel('False Positive Rate')  # 设置x轴标题
# plt.ylabel('True Positive Rate')  # 设置y轴标题
# plt.title('Receiver operating characteristic example')  # 设置图表标题
# plt.legend(loc="lower right")  # 设置图例放置的位置
# plt.show()

#-*- coding:utf-8 -*-
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
#3：1拆分数据集
from sklearn.model_selection import train_test_split
#乳腺癌数据集
from sklearn.datasets import load_breast_cancer
import pydot
cancer = load_breast_cancer()
#参数random_state是指随机生成器，0表示函数输出是固定不变的
X_train,X_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print('Train score:{:.3f}'.format(tree.score(X_train,y_train)))
print('Test score:{:.3f}'.format(tree.score(X_test,y_test)))
#生成可视化图
# export_graphviz(tree,out_file="tree.dot",class_names=['严重','轻微'],feature_names=cancer.feature_names,impurity=False,filled=True)
# #展示可视化图
# (graph,) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')
dot_data = export_graphviz(tree,out_file=None, class_names=['严重','轻微'],feature_names=cancer.feature_names,impurity=False,filled=True)
import graphviz
graph = graphviz.Source(dot_data)
graph