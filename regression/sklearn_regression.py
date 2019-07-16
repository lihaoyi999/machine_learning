# -*- coding:utf-8 -*-
# author : 囧囧有神
# data : 2019/07/16 11:05


# 普通最小二乘法

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score


diabetes = datasets.load_diabetes()
# 只取数据集第3列作为因变量数据
# diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X = diabetes.data
# 将数据划分为训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 构建模型
reg = linear_model.LinearRegression()
# 训练模型
reg.fit(diabetes_X_train, diabetes_y_train)
# 对测试数据集进行预测
diabetes_y_pred = reg.predict(diabetes_X_test)

# 回归系数
print('Coeficients:\n', reg.coef_)
# 截距
print('Intercept:\n', reg.intercept_)
# 均方差
print('Mean squared error:%.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 方差得分
print('Variance score:%.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# 绘图
# plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
# plt.show()


# 岭回归
from sklearn import linear_model


reg = linear_model.Ridge(alpha=0.5, normalize=True)
reg.fit(diabetes_X_train, diabetes_y_train)
# 对测试数据集进行预测
diabetes_y_pred = reg.predict(diabetes_X_test)

# 回归系数
print('Coeficients:\n', reg.coef_)
# 截距
print('Intercept:\n', reg.intercept_)
# 均方差
print('Mean squared error:%.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# 方差得分
print('Variance score:%.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False, normalize=True)
    ridge.fit(diabetes_X_train, diabetes_y_train)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()


# 设置正则化参数alpha，即选择合适的alpha
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit(diabetes_X_train, diabetes_y_train)
print(reg.alpha_)


# Lasso 回归
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(diabetes_X_train, diabetes_y_train)
print(reg.coef_)

reg = linear_model.LassoCV()
reg.fit(diabetes_X_train, diabetes_y_train)
print(reg.alpha_)
print(reg.coef_)