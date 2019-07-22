## 支持向量机原理

支持向量机（Support Vector Machine，简称SVM），是机器学习中运用广泛的一种算法。SVM是一种二分类算法，通过构建超平面函数，来进行样本分类。  
对于样本空间：
$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$$
其中，$x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T \in \mathbb{R}^{n}, y_i \in \{+1,-1\},i=1,2,\cdots,N$   
$x_i$为第$i$个样本，$y_i$为$x_i$的类别标签，$N$为样本数量，$n$为特征数量。

当$y_i=+1$时，称$x_i$为正例；当$y_i=-1$时，称$x_i$为负例。$(x_i,y_i)$成为样本点。  
假设超平面决策边界函数为：
$$w^Tx+b=0$$
其中$w=(w_1,w_2.\cdots,w_N)$为法向量，决定了超平面的方向。$b$为位移项，决定了超平面与原点的距离。  
任一点$x$到超平面的距离表示为：$r=\frac{|w^Tx+b|}{\left| w \right|}$，$\parallel w \parallel$表示法向量的模。  

假设超平面$(w,b)$能对样本进行正确的分类，那么对于$(x_i,y_i) \in T$，若$y_i=+1$，则有$w^T x_i+b>0$，相反地，若$y_i=-1$，则有$w^T x_i+b<0$。我们假设
$$
\begin{cases}
w^T x_i+b \geq 0 ,& \text{ if } y_i=+1 \\ 
w^T x_i+b \leqslant 0 ,& \text{ if } y_i=-1
\end{cases}
$$
上面两公式的间隔可表示为$\frac{2}{\left| w \right|}$。
我们的目的是求最大间隔
$$
\max_\limits{{w,b}} \frac{2}{\parallel w \parallel}
$$
其中$y_i(w^T x_i+b)>=1,i=1,2,\cdots,N$。  
将最大化问题转为最小化问题：
$$
\min_\limits{{w,b}} \frac{\parallel w \parallel^2}{2}
$$
其中$y_i(w^Tx_i+b)>=1,i=1,2,\cdots,N$。  
这就是支持向量的基本型，即优化目标函数。  

**模型为：**
$$
f(x)=w^T x+b=\sum_{i=1}^{N}\alpha_iy_ix_i^Tx+b
$$
KTT条件为：
$$
\left\{\begin{matrix}
\alpha_i \geq 0\\ 
y_i f(x_i)-1 \geq 0\\ 
\alpha_i(y_if(x_i)-1)=0
\end{matrix}\right.
$$
**优化的目标函数：**
$$
\max_\limits{{\alpha}} \sum_{i=1}^{N}\alpha_i-\frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

$$
s.t. \, \sum_{i=1}^{N}\alpha_i y_i=0,  \alpha_i \geq 0,i=1,2,\cdots,N
$$


#### 算法7.1
**线性可分支持向量机学习算法--最大间隔法**
输入：线性可分训练数据集$T={(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)}$，
其中$x_i \in \mathbb{R}$，$y_i=\{+1,-1\}$，$i=1,2,\cdots,N$；  
输出：最大间隔分离超平面和分类决策函数。  
（1）构造并求解约束原始最优化问题：
$$
\min\limits_{w,b} \frac{1}{2}\parallel w\parallel
$$

$$
s.t.  \, y_i(w\cdot x_i+b)-1\geq 0,\,i=1,2,\cdots,N
$$

求得最优解$w^{*}\,,b^{*}$。

（2）由此得到分离超平面：

$$
w^*\cdot x+b=0
$$
分类决策函数：

$$
f(x)=sign(w^*\cdot x+b^*)
$$


## 定理7.2 

设$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)$是对偶最优化问题

$$
\min \limits_{\alpha}\, \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j(x_i \cdot x_j )-\sum_{i=1}^{N}\alpha_i
$$

$$
s.t.\,\sum_{i=1}^{N}\alpha_i y_i=0
$$

$$
\alpha_i\geq 0,\,i=1,2,\cdots,N
$$

的解，则存在下标$j$，使得$\alpha_j>0$，并可按下式求得原始最优化问题
$$
\min_\limits {w,b} \frac{1}{2} \parallel w \parallel ^2
$$

$$
s.t. \, y_i(w \cdot x_i + b) -1 \geq 0, \, i = 1,2,\cdots,N
$$

的解$w^*,b^*$：
$$
w^*=\sum_{i=1}^{N}\alpha_i^* y_i x_i
$$

$$
b^*=y_j-\sum_{i=1}^{N}\alpha_i^* y_i (x_i \cdot y_i)
$$



## 算法7.2 

**线性可分支持向量机学习算法**

输入：线性可分训练数据集$T={(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)}$，
其中$x_i \in \mathbb{R}$，$y_i=\{+1, -1\}$，$i=1,2,\cdots,N$；  

输出：分离超平面和分类决策函数。  

（1）构造并求解约束最优化问题

$$
\begin{eqnarray} 
\min_{\alpha} && \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j(x_i \cdot x_j)-\sum_{i=1}^{N} \alpha_i\\
s.t.\,\, && \sum_{i=1}^{N}\alpha_iy_i=0  \\ \alpha_i \geq 0 ,&& i =1,2\cdots,N
\end{eqnarray}
$$
求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$。

（2）计算  

$$
w^*=\sum_{i=1}^{N}\alpha_i^* y_ix_i
$$
并选择一个正分量$\alpha_j^*>0$，计算

$$
b^*=y_j-\sum_{i=1}^{N}\alpha_i^* y_i(x_i \cdot x_j)
$$
（3）求得分离超平面：

$$
w^*\cdot x+b^*=0
$$
分类决策函数：

$$
f(x)=sign(w^*\cdot x+b^*)
$$
$C\geq \alpha \geq0$

$$

$$
