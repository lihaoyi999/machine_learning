# 第5章 决策树

## 5.1 决策树模型与学习

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

## 5.2 特征选择

通常特征选择的准则是信息增益或信息增益比。

### 5.2.2 信息增益

#### 信息熵

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

**条件熵** 

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


#### 信息增益

定义为：
$$
g(D,A)=H(D)-H(D|A)=\sum_{i=1}^{n}\sum_{k=1}^{K}\frac{N_{k}^{(i)}}{N}\log_2\frac{N_{k}^{(i)}}{N^{(i)}}-\sum_{k=1}^{K}\frac{N_k}{N}\log_2\frac{N_k}{N}
$$

#### 信息增益比

$$
g_R (D,A)=\frac{g(D,A)}{H(D)}
$$

