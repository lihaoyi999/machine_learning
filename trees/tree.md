

# 一、决策树

## 1.决策树模型与学习

决策树（decision tree）是一种基本的分类和回归方法。其主要优点：模型具有可读性，分类速度块。

决策树学习包含3个步骤：

1. 特征选择
2. 决策树生成
3. 决策树剪枝

假设给定训练数据集：
$$
D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$
其中$x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T$为输入实例（特征向量），$n$为特征个数，$y_i \in\{1,2,\cdots,K\}$为类标记，$i=1,2,\cdots,N，N$为样本数。

## 2.特征选择

通常特征选择的准则是信息增益或信息增益比。

### 2.1 信息增益

#### （1）信息熵

**熵**（entropy）表示随机变量不确定性的度量。设$X$是一个取得有限值的离散随机变量，其概率分布为
$$
P(X=x_i)=p_i,  i=1,2,\cdots,n
$$
则随机变量X的熵定义为：
$$
H(X)=-\sum_{i=1}^{n}p_i\log_2(p_i)
$$
训练集D的熵为：
$$
H(D)=-\sum_{k=1}^{K}p_k\log_2p_k=-\sum_{k=1}^{K}\frac{N_k}{N}\log_2\frac{N_k}{N}
$$
其中$p_k=P(y_k)=\frac{N_k}{N}$，$N_k$为类标记为$y_k$的样本数。

#### （2）条件熵

特征$A$将数据集$D$切分为n个子集$(D_1,D_2,\cdots,D_n)$，每个子集对应的样本数为$(N^{(1)},N^{(2)},\cdots,N^{(n)})$，类标记$y_i=(1,2,\cdots,K)$，$N_{K}^n$表示第$n$个子集中类标记为$K$的样本数。
$$
\left[
\begin{matrix}
 & C_1      & C_2      & \cdots & C_K &  C   \\\\
D_1 & N_1^{(1)}     & N_2^{(1)}      & \cdots & N_K^{(1)}  &  N^{(1)}   \\\\
D_2 & N_1^{(2)}  & N_2^{(2)}  & \cdots & N_K^{(2)}  & N^{(2)}\\\\
\vdots & \vdots      & \vdots      & \ddots & \vdots  &   \vdots  \\\\
D_n &  N_1^{(n)}       & N_2^{(n)}       & \cdots & N_K^{(n)}   &   N^{(n)} \\\\
D &   N_{1}      & N_{2}      & \cdots & N_{K}  &   N  \\\\
\end{matrix}
\right]
$$
子集$D_n$的熵为：
$$
H(D_n)=-\sum_{k=1}^{K}p_k\log_2 p_k=-\sum_{k=1}^{K}\frac{N_{k}^{(n)}}{N^{(n)}}\log_2\frac{N_{k}^{(n)}}{N^{(n)}}
$$
数据集$D$在条件$A$的条件熵为：
$$
H(D|A)=\sum_{i=1}^{n}\frac{N^{(i)}}{N} H(D_i)=-\sum_{i=1}^{n}\sum_{k=1}^{K}\frac{N_{k}^{(i)}}{N}\log_2\frac{N_{k}^{(i)}}{N^{(i)}}
$$


#### （3）信息增益

定义为：
$$
g(D,A)=H(D)-H(D|A)=\sum_{i=1}^{n}\sum_{k=1}^{K}\frac{N_{k}^{(i)}}{N}\log_2\frac{N_{k}^{(i)}}{N^{(i)}}-\sum_{k=1}^{K}\frac{N_k}{N}\log_2\frac{N_k}{N}
$$

#### （4）信息增益比

$$
g_R (D,A)=\frac{g(D,A)}{H(D)}
$$

## 3.决策树生成

### 3.1 ID3算法

ID3算法的核心是在决策树各个结点上应用信息增益准则选择特征，递归的构建决策树。具体作法是：从根节点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点；再对子结点递归调用以上方法，构建决策树；直到所有特征的信息增益均较小或没有特征可以选择为止。最后得到一个决策树。

---

**ID3 算法**

---

输入：训练数据集$D$，特征集$A$，阈值$\varepsilon$。  

输出：决策树$T$

（1）若$D$中所有样本点属于同一类$C_k$，则$T$为单结点树，并将$C_k$作为该结点的类标记，返回$T$；

（2）若$A=\emptyset$，即特征集为空集时，则$T$为单结点树，并将D中样本点最多的类$C_k$作为该结点的类标记，返回$T$；

（3）否则，计算$A$中各特征对$D$的信息增益，选择信息增益最大的特征$A_g$；

（4）如果$A_g$的信息增益小于阈值$\varepsilon$，则置$T$为单结点树，并将$D$中样本数最多的类$C_k$作为该结点的类标记，返回$T$；

（5）否则，对$A_g$的每一个可能值$a_i$，依$A_g=a_i$将$D$分割为若干个非空子集$D_i$，~~将$D_i$中样本点最多的类作为标记，构建子结点，由结点及其子结点构成树$T$，返回$T$~~；

（6）对第$i$个子结点，以$D_i$为训练集，以$A-\{A_g\}$为特征集，递归的调用步骤（1）~（5），得到子树$T_i$，返回$T_i$。

---

### 3.2 C4.5算法

使用信息增益比来选择特征

---

**C4.5 算法**

------

输入：训练数据集$D$，特征集$A$，阈值$\varepsilon$。  

输出：决策树$T$

（1）若$D$中所有样本点属于同一类$C_k$，则$T$为单结点树，并将$C_k$作为该结点的类标记，返回$T$；

（2）若$A=\emptyset $，即特征集为空集时，则$T$为单结点树，并将D中样本点最多的类$C_k$作为该结点的类标记，返回$T$；

（3）否则，计算$A$中各特征对$D$的信息增益比，选择信息增益比最大的特征$A_g$；

（4）如果$A_g$的信息增益比小于阈值$\varepsilon$，则置$T$为单结点树，并将$D$中样本数最多的类$C_k$作为该结点的类标记，返回$T$；

（5）否则，对$A_g$的每一个可能值$a_i$，依$A_g=a_i$将$D$分割为若干个非空子集$D_i$，~~将$D_i$中样本点最多的类作为标记，构建子结点，由结点及其子结点构成树$T$，返回$T$~~；

（6）对第$i$个子结点，以$D_i$为训练集，以$A-\{A_g\}$为特征集，递归的调用步骤（1）~（5），得到子树$T_i$，返回$T_i$。

## 4.决策树的剪枝

在决策树学习中将已生成的树进行简化的过程称为剪枝（pruning）。具体地，剪枝从已生成的树上剪掉一些子树或叶结点，并将其根结点或父结点作为新的叶结点，从而简化分类树模型。决策树通过极小化决策树的损失函数（loss function）或代价函数（cost function）来实现。

设数$T$的叶结点个数为$|T|$，$t$是$T$的叶结点，该叶结点上有$N_t$个样本点，其中$k$类的样本点有$N_{ik}$个，$k=1,2,\cdots,K$，$H_t(T)$为叶结点$t$上的经验熵，$\alpha \ge0$为参数，则决策树学习的损失函数为：
$$
C_{\alpha}(T)=\sum_{t=1}^{|T|}N_t H_t(T) + \alpha|T|
$$
其中经验熵为：
$$
H_t(T)=-\sum_{k=1}^{K}\frac{N_{tk}}{N_t}\log \frac{N_{tk}}{N_t}
$$
将损失函数的第一项记作
$$
C(T)=\sum_{t=1}^{|T|}N_t H_t(T)=-\sum_{t=1}^{|T|}\sum_{k=1}^{K} N_{tk}\log \frac{N_{tk}}{N_t}
$$
有
$$
C_{\alpha}(T)=C(T)+\alpha |T|
$$
$C(T)$表示模型对训练数据的预测误差，即模型与训练数据的拟合程度，$|T|$表示模型的复杂程度，参数$\alpha \ge0$控制两者之间的影响。较大的$\alpha$促使选择较简单的模型，较小的$\alpha$促使选择较复杂的模型。$\alpha=0$ 意味着只考虑模型与训练数据的拟合程度，不考虑模型的复杂度。

剪枝，就是当$\alpha$确定时，选择损失函数最小的模型。决策树生成只考虑了通过提高信息增益（或信息增益比）对训练数据进行更好的拟合，而决策树剪枝通过优化损失函数还考虑了减小模型复杂度。决策树生成学习局部的模型，而决策树剪枝学习整体的模型。

损失函数的极小化等价于正则化的极大似然估计，利用损失函数最小原则进行剪枝，就是利用正则化的极大似然估计进行模型选择。

---

**决策树的剪枝算法**

---

输入：生成算法产生的整个数$T$，参数$\alpha$  

输出：修剪后的子树$T_{\alpha}$

（1）计算每个结点的经验熵；

（2）递归的从树的叶结点向上回缩；

设一组叶结点回缩到父结点之前和之后的整体树分别为$T_B$和$T_A$，其对应的损失函数值分别为$C_{\alpha}(T_B)$和$C_{\alpha}(T_A)$，如果
$$
C_{\alpha}(T_B) \ge C_{\alpha}(T_A)
$$
则进行剪枝，即将父结点变为新的叶结点；

（3）返回步骤（2），知道不能继续剪枝为止，得到损失函数最小的子树$T_{\alpha}$。

---





# 二、分类回归树（CART）

classification and regression tree.

CART假定决策树是二叉树，内部结点特征的取值为“是”和“否”，左分支是取值为“是”的分支，右分支是取值为“否”的分支。

CART算法有以下两步组成：

（1）决策树生成：基于训练数据集生成决策树，生成的决策树要尽量大；

（2）决策树剪枝：用验证数据集对已生成的树进行剪枝并选择最优子树，此时用损失函数最小作为剪枝的标准。

## 1.CART生成

决策树的生成就是递归的构建二叉树的过程。对回归树用平方误差最小化准则，对分类树用基尼指数（Gini index）最小化准则，进行特征选择，生成二叉树。

### 1.1 回归树的生成

假设$X$和$Y$分别为输入和输出变量，并且$Y$是连续变量，给定训练数据集

$D=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$

假设已将数据集划分为$M$个子数据集$R_1,R_2,\cdots,R_M$，并且在每个子数据集$R_m$上有一个固定的输出值$c_m$，于是回归树模型可表示为
$$
f(x)=\sum_{m=1}^{M} c_m I(x \in R_m)
$$
子数据集$R_m$上的$c_m$的最优值$\hat c_m$是$R_m$上所有$x_i$对应的$y_i$的均值，即：
$$
\hat c_m=\bar y_i|x_i \in R_m
$$
二分数据集，采用启发式的方法，选择第$j$个变量$x^{(j)}$和它的取值$s$，作为切分变量和切分点，并定义两个区域：
$$
R_1(j,s)=\{x|x^{(j)}\le s\} 和R_2(j,s)=\{x|x^{(j)}\gt s\}
$$
然后寻找最优切分变量$j$和切分点$s$。求解
$$
\min \limits_{j,s} \left[\min \limits_{c_1} \sum_{x_i \in R_1(j,s)}(y_i-c_1)^2 + \min \limits_{c_2} \sum_{x_i \in R_2(j,s)}(y_i-c_2)^2 \right]
$$
其中
$$
\hat c_1=\bar y_i|x_i \in R_1 和\hat c_2=\bar y_i|x_i \in R_2
$$
遍历所有变量及变量的取值，找到最优的切分变量和切分点。

---

算法 最小二乘回归树生成算法

---

输入：训练数据集$D$

输出：回归树$f(x)$

递归的将数据集划分为两个子数据集，并决定每个子数据集的输出值，构建二叉树：

（1）选择最优切分变量$j$和切分点$s$，求解目标函数
$$
\min \limits_{j,s} \left[\min \limits_{c_1} \sum_{x_i \in R_1(j,s)}(y_i-c_1)^2 + \min \limits_{c_2} \sum_{x_i \in R_2(j,s)}(y_i-c_2)^2 \right]
$$
遍历变量$j$，对固定的切分变量$j$，遍历它的取值$s$作为切分点，选择使目标函数达到最小值的$j和s$。

（2）用选定的$(j,s)$划分区域并决定相应的输入值：
$$
R_1(j,s)=\{x|x^{(j)}\le s\} 和R_2(j,s)=\{x|x^{(j)}\gt s\}
$$

$$
\hat c_1=\bar y_i|x_i \in R_1 和\hat c_2=\bar y_i|x_i \in R_2
$$

（3）继续对两个子数据集调用步骤（1）（2），直到满足停止条件。

（4）将训练数据集划分为$M$个区域$R_1,R_2,\cdots,R_M$，生成决策树：
$$
f(x)=\sum_{m=1}^{M} \hat c_m I(x \in R_m)
$$

### 1.2 分类树的生成

分类树使用基尼指数选择最优特征，同时决定该特征的最优二值切分点。

**基尼指数**，表示集合的不确定性。基尼指数越大，不确定性越大。

分类问题中，假设样本集合$D$有$N$个样本，有$K$个类，每个类的样本数分别为$N_1,N_2,\cdots,N_K$，样本点属于第$k$类的概率为$p_k=\frac{N_K}{N}$，则$D$的基尼指数定义为
$$
Gini(D)=\sum_{k=1}^K p_k(1-p_k)=1-\sum_{k=1}^K p_k^2=1-\sum_{k=1}^K \left(\frac{N_k}{N}\right)^2
$$
假设集合$D$在特征$A$下被分割为$D_1,D_2$，则在特征$A$的条件下，D的基尼指数为
$$
Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1) + \frac{|D_2|}{|D|}Gini(D_2)
$$

---

算法 CART生成算法

---

输入：训练数据及$D$，停止计算条件

输出：CART决策树

根据训练数据集，从根结点开始，递归地对每个结点进行一下操作，构建二叉决策树：

（1）设结点的训练数据集为$D$，计算现有特征对该数据集的基尼指数，对每个特征$A$，对其可能取的每个值$a$，根据样本点对$A=a$的测试为“是”或“否”将数据集分割为$D_!,D_2$，计算$A=a$时的基尼指数。

（2）在所有可能的特征$A$以及他们所有可能的切分点$a$中，选择基尼指数最小的特征及其对应的切分点作为最优特征和最优切分点。依据最优特征和最优切分点，从现结点生成两个子结点，将训练数据集依特征分配到两个子结点中去。

（3）对两个子结点递归的调用（1）（2），直到满足停止条件。停止条件为：结点的样本数小于预定阈值、样本集的基尼指数小于预定阈值、没有更多特征。

（4）生成CART决策树。

## 2. CART 剪枝

CART剪枝算法由两步组成：首先，从生成算法产生的决策树$T_0$底端开始不断剪枝，直到$T_0$的根结点，形成一个子树序列$\{T_0,T_1,\cdots,T_n\}$；然后通过交叉验证法在独立的验证数据集上对子树序列进行测试，从中选择最优子树。

---

CART剪枝算法

---

输入：CART算法生成的决策树$T_0$

输出：最优决策树$T_\alpha$

（1）设$k=0$，$T=T_0$.

（2）设$\alpha=+ \infty$.

（3）自下而上地对各内部结点$t$计算$C(T_t)$，$|T_t|$以及
$$
g(t)=\frac{C(t)-C(T_t)}{|T_t|-1}
$$
​	
$$
\alpha=\min(\alpha,g(t))
$$
其中$T_t$表示以$t$为根结点的子树，$C(T_t)$是对训练数据的预测误差，$|T_t|$是$T_t$的叶结点个数。

（4）自上而下地访问内部结点$t$，如果有$g(t)=\alpha$，进行剪枝，并对叶结点$t$以多数表决法决定其类别，得到树$T$。

（5）设$k=k+1$，$\alpha_k=\alpha$，$T_k=T$。

（6）如果$T$不是由根结点单独构成的树，则回到步骤（4）。

（7）采用交叉验证法在子树序列$T_0,T-1,\cdots,T_n$中选中最优子树$T_\alpha$。



# 三、sklearn 决策树模型的实现

## 1. 分类树的实现

*class* `sklearn.tree.DecisionTreeClassifier`(*criterion=’gini’*, *splitter=’best’*, *max_depth=None*, *min_samples_split=2*, *min_samples_leaf=1*, *min_weight_fraction_leaf=0.0*, *max_features=None*, *random_state=None*, *max_leaf_nodes=None*, *min_impurity_decrease=0.0*, *min_impurity_split=None*, *class_weight=None*, *presort=False*)

参数说明：

criterion=’gini’*,*  选择最佳划分的度量，默认是基尼指数，gini：基尼指数，entropy：信息增益

splitter=’best’,  数据集划分的方式，best：在特征的所有划分点中找出最优的划分点，random：随机的在部分划分点中找局部最优的划分点  

*max_depth=None*,  设置树的最大深度，取值为None或者整数

*min_samples_split=2*,  拆分内部节点再划分所需的最小样本数，节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分  

*min_samples_leaf=1*,  叶结点所需的最小样本数

*min_weight_fraction_leaf=0.0*,  叶结点的样本数占总样本数的比例

*max_features=None*,  寻找最佳分割时要考虑的最大特征数量

*random_state=None*,  随机种子

*max_leaf_nodes=None*, 最大叶结点数

*min_impurity_decrease=0.0*, 决策树停止生长的阈值，如果节点的不纯度减少量高于阈值，节点将分裂，否则它是叶子结点

*class_weight=None*,  类别权重

*presort=False* 是否预先排序数据以加快拟合中最佳分割的查找

```python
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 导入乳腺癌数据集
cancer = load_breast_cancer()
cancer = load_breast_cancer()
# 查看数据的结构
cancer.keys()
# 查看特征名称 
cancer.feature_names
# 查看类别标签
cancer.target_names  
# 随机划分训练集、测试集，参数random_state是指随机生成器（随机种子），0表示函数输出是固定不变的。
X_train, X_test, y_train, y_test = train_test_split(cancer['data'], 
                                                    cancer['target'],
                                                    random_state=42)
# 构建分类决策树模型    
tree = DecisionTreeClassifier(random_state=0)                          
# 使用决策树模型拟合训练数据集 
tree.fit(X_train, y_train)
# 模型得分
print('Train score:{:.3f}'.format(tree.score(X_train, y_train)))
print('Test score:{:.3f}'.format(tree.score(X_test, y_test)))
# 生成决策树可视化图
dot_data = export_graphviz(tree, out_file=None, 
                           class_names=['严重', '轻微'], 
                           feature_names=cancer.feature_names,
                           impurity=False,
                           filled=True)
graph = graphviz.Source(dot_data)
graph.render("tree")
# 返回每个样本的预测值的叶子索引
tree.apply(X_test)
# 特征的重要性
tree.feature_importances_
```

## 2. 回归树的实现

*class* `sklearn.tree.DecisionTreeRegressor`(*criterion=’mse’*, *splitter=’best’*, *max_depth=None*, *min_samples_split=2*, *min_samples_leaf=1*, *min_weight_fraction_leaf=0.0*, *max_features=None*, *random_state=None*, *max_leaf_nodes=None*, *min_impurity_decrease=0.0*, *min_impurity_split=None*, *presort=False*)

参数说明：

*criterion=’mse’*, 选择最佳划分的度量，默认是mse，mse：均方误差，mae：平均绝对误差

*splitter=’best’*, 数据集划分的方式，best：在特征的所有划分点中找出最优的划分点，random：随机的在部分划分点中找局部最优的划分点

*max_depth=None*, 设置树的最大深度，取值为None或者整数

*min_samples_split=2*, 拆分内部节点再划分所需的最小样本数，节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分

*min_samples_leaf=1*, 叶结点所需的最小样本数

*min_weight_fraction_leaf=0.0*, 叶结点的样本数占总样本数的比例

*max_features=None*, 寻找最佳分割时要考虑的最大特征数量

*random_state=None*, 随机种子

*max_leaf_nodes=None*, 最大叶结点数

*min_impurity_decrease=0.0*, 决策树停止生长的阈值，如果节点的不纯度减少量高于阈值，节点将分裂，否则它是叶子结点

*presort=False* 是否预先排序数据以加快拟合中最佳分割的查找

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
# 创建数据集
rng = np.random.RandomState(1)
x = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
# 训练模型
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(x, y)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_2.fit(x, y)
# 模型评分
print(regr_1.score(x, y))
regr_2.score(x, y)
# 使用模型进行预测
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)
# 绘图
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(x_test, y_1, color="cornflowerblue", 
         label="max_depth=2", linewidth=2)
plt.plot(x_test, y_2, color="yellowgreen", 
         label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

