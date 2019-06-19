# 机器学习实战笔记


## 第一部分 分类
### 第1章 机器学习基础
### 第2章 k-近邻算法
优点：精度高，对异常值不敏感，无数据输入假定  
缺点：计算复杂度高，空间复杂度高
### 第3章 决策树
优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据  
缺点：可能会产生过度匹配问题

#### 3.1.1 信息增益
熵，定义为信息的期望值。如果待分类的事务可能划分在多个类中，则符号$x_i$的信息定义为：  
$$l(x_i)=-log_2p(x_i)$$  
其中$p(x_i)$是选择该分类的概率。  
$$H = -\sum_{n-1}^{n}p(x_i)log_2p(x_i)$$，n是分类数目

### 第4章 朴素贝叶斯算法
优点：在数据较少的情况下仍然有效，可以处理多类别问题
缺点：对于输入数据的准备方式较为敏感  
贝叶斯决策理论的核心思想：选择具有最高概率的决策。
用$p1(x,y)$表示数据点$(x,y)$属于类别1的概率，
用$p2(x,y)$表示数据点$(x,y)$属于类别2的概率，对于数据点$(x,y)$，
可用下面的规则来判断它的类别：
- 如果$p1(x,y)>p2(x,y)$，那么类别为1.
- 如果$p1(x,y)<p2(x,y)$，那么类别为2.

$$p(c_j|W)=\frac{p(W|c_j)p(c_j)}{p(W)}$$  
$p(c_j)$: 类别$j$的文档数除以总文档数，$W$为词向量。  
根据朴素贝叶斯假设：
$p(W|c_j)=p(w_0,w_1,w_2,\cdots,w_n|c_j)=p(w_0|c_j)p(w_1|c_j)p(w_2|c_j)\cdots p(w_m|c_j)$

伪代码：
```  
计算每个类别中的文档数量
对每篇训练文档：
    对每个类别：
        如果词条出现在文档中 --> 增加该词条的计数值
        增加所有词条的计数值
    对每个类别：
        对每个词条：
            将该词条的数目除以总词条数目得到条件概率
    返回每个类别的条件概率
```


### 第5章 Logistic回归
优点：计算代价不高，易于理解和实现
缺点：容易欠拟合，分类精度可能不高  
$Sigmoid$函数：$$\rho(z)=\frac{1}{1+e^{-z}}$$
$$z=W^{T}X=w_0x_0+w_1x_1+\cdots+w_nx_n$$
**梯度上升算法**  
梯度上升算法基于的思想是：要找到函数的最大值，最好的方法是沿着该函数的梯度方向探寻。
如果梯度记为$\Delta$，则函数$f(x,y)$的梯度由下式表示：  
$$
\Delta f(x,y)=
\begin{pmatrix}
\frac{\partial f(x,y)}{\partial x}\\\\
\frac{\partial f(x,y)}{\partial y}
\end{pmatrix}
$$
沿x的方向移动$\frac{\partial f(x,y)}{\partial x}$，
沿y方向移动$\frac{\partial f(x,y)}{\partial y}$  
梯度算法的迭代公式：
$$w:=w + \alpha\Delta_wf(w)$$
$\alpha$为步长。  


$X$为m行n列的矩阵，表示m个样本，n个特征。$Y$为这m个样本的分类标签。
$$
X=
\begin{pmatrix}
a_{11} & \cdots & a_{1n}\\\\
\vdots & \ddots & \vdots\\\\
a_{m1} & \cdots & a_{mn}
\end{pmatrix}
$$
$$Y=
\begin{pmatrix}
y_1\\\\ 
\vdots\\\\ 
y_m
\end{pmatrix}
$$
logistic模型：  
$$Z=W^TX=w_0x_0+w_1x_1+\cdots+w_nx_n$$
预测模型：  
$$\bar{Y}=\rho(Z)=\frac{1}{1+e^{-Z}}$$
预测误差：  
$$E=Y-\bar{Y}$$

梯度：
$$w:=w + \alpha X^{T} E$$
#### 梯度上升算法
每次更新回归系数是都需要遍历整个数据集 
 
**梯度上升算法伪代码：**
```
每个回归系数初始化为1
重复R次：
    计算整个数据集的梯度
    使用alpha X gradient更新回归系数的向量
返回回归系数
```

#### 随机梯度上升算法
一次仅用一个样本点更新回归系数，可以在新样本到来时对分类器进行增量式更新，
因而随机梯度上升算法是一个在线学习算法
**随机梯度上升算法伪代码：**
```
每个回归系数初始化为1
对数据集中每个样本：
    计算该样本的梯度
    使用alpha X gradient更新回归系数的数值
返回回归系数值
```
stocGradAscent0运行错误：  
TypeError: 'numpy.float64' object cannot be interpreted as an integer  
修改：增加一行dataMatrix=np.array(dataMatrix)  

### 第6章 支持向量机
优点：泛化错误率低，计算开销不大，结果易于解释  
缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二分类问题

 分隔超平面可写为：$W^TX+b$  
 点到分隔超平面的距离：$|W^TA+b|/|W|$
 $$arg \max_{{w},b}\left \{ \min_{{n}}(label\cdot(W^Tx+b))\cdot\frac{1}{\left |W| \right \} \right \}$$
 $$arg \, \max_{{W},b}\left \{ \min_{{n}}(label \cdot (W^Tx+b)) \cdot \frac{1}{||W||} \right \}$$
 $$\max_{{\alpha}}\left [ \sum_{i=1}^{m}\alpha-\frac{1}{2}\sum_{i,j=1}^{m}label^{(i)}\cdot label^{(j)}\cdot \alpha_i \cdot \alpha_j \left \langle x^{(i)},x^{(j)} \right \rangle\right ]$$

### 第7章 
