# 支持向量机原理
## 1、线性可分支持向量机
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
w^T x_i+b \leq 0 ,& \text{ if } y_i=-1
\end{cases}
$$
上面两公式的间隔可表示为$\frac{2}{\left| w \right|}$。
我们的目的是求最大间隔
$$
\max_\limits{{w,b}} \frac{2}{\parallel w \parallel}
$$
其中$y_i(w^T x_i+b)\geq,i=1,2,\cdots,N$。  
将最大化问题转为最小化问题：
$$
\min_\limits{{w,b}} \frac{\parallel w \parallel^2}{2}
$$
其中$y_i(w^Tx_i+b)\geq1,i=1,2,\cdots,N$。  
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

---

### 算法7.1

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

---

### 定理7.2 

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

的解$w^*,b^*$
$$
w^*=\sum_{i=1}^{N}\alpha_i^* y_i x_i
$$

$$
b^*=y_j-\sum_{i=1}^{N}\alpha_i^* y_i (x_i \cdot x_j)
$$



---

### **算法7.2 线性可分支持向量机学习算法**

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

---



## 2、线性不可分支持向量机

对于特征空间上的训练数据集：
$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$$
其中，$x_i=(x_i^{(1)},x_i^{(2)},\cdots,x_i^{(n)})^T \in \mathbb{R}^{n}, y_i \in \{+1,-1\},i=1,2,\cdots,N$   
$x_i$为第$i$个样本，$y_i$为$x_i$的类别标签，$N$为样本数量，$n$为特征数量。

原始最优化问题
$$
\min_\limits {w,b} \frac{1}{2} \parallel w \parallel ^2
$$

$$
s.t. \, y_i(w \cdot x_i + b) -1 \geq 0, \, i = 1,2,\cdots,N
$$

由于线性不可分，某些样本点$(x_i,y_i)$不能满足函数间隔大于等于1的约束条件。为了解决这个问题，可以对每个样本点引进一个松弛变量$\xi_i\geq0$，使得函数间隔加上松弛变量大于等于1。

于是约束条件变为：
$$
y_i(w \cdot x_i+b) \geq 1- \xi_i
$$
同时，对每个松弛变量$\xi$支付一个代价$\xi$，目标函数变为：
$$
\frac{1}{2} \parallel w \parallel ^2 + C\sum_{i=1}^{N}\xi_i
$$
其中$C \geq0$成为惩罚参数，

线性不可分的线性支持向量机的学习问题变成如下凸二次规划问题（原始问题）：
$$
\min_\limits {w,b,\xi} \frac{1}{2} \parallel w \parallel ^2 + C\sum_{i=1}^{N}\xi_i
$$

$$
s.t.\, y_i(w \cdot x_i+b) \geq 1- \xi_i, i=1,2,\cdots, N
$$

$$
\xi_i \geq 0, i=1,2,\cdots, N
$$

原始问题的对偶问题是：
$$
\begin{eqnarray} 
\min_{\alpha} && \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j(x_i \cdot x_j)-\sum_{i=1}^{N} \alpha_i\\
s.t.\,\, && \sum_{i=1}^{N}\alpha_iy_i=0  \\ 
0 \leq \alpha_i \leq C ,&& i =1,2\cdots,N
\end{eqnarray}
$$

---

### 定理7.3

设$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)$是对偶问题
$$
\min \limits_{\alpha}\, \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j(x_i \cdot x_j )-\sum_{i=1}^{N}\alpha_i
$$

$$
s.t.\,\sum_{i=1}^{N}\alpha_i y_i=0
$$

$$
0 \leq \alpha_i \leq C , i =1,2\cdots,N
$$

的一个解，若存在一个分量$\alpha_j$，$0 <\alpha_j<0$，可按下式求得原始问题
$$
\min_\limits {w,b,\xi} \frac{1}{2} \parallel w \parallel ^2 + C\sum_{i=1}^{N}\xi_i
$$

$$
s.t.\, y_i(w \cdot x_i+b) \geq 1- \xi_i, i=1,2,\cdots, N
$$

$$
\xi_i \geq 0, i=1,2,\cdots, N
$$

的解$w^*,b^*$
$$
w^*=\sum_{i=1}^{N}\alpha_i^* y_i x_i
$$

$$
b^*=y_j-\sum_{i=1}^{N}\alpha_i^* y_i (x_i \cdot x_j)
$$

---

### 算法7.3 线性支持向量机学习算法

输入：线性可分训练数据集$T={(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)}$，
其中$x_i \in \mathbb{R}$，$y_i=\{+1, -1\}$，$i=1,2,\cdots,N$；  

输出：分离超平面和分类决策函数。 

（1）选择惩罚参数$C>0$，构造并求解凸二次规划问题
$$
\min \limits_{\alpha}\, \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j(x_i \cdot x_j )-\sum_{i=1}^{N}\alpha_i
$$

$$
s.t.\,\sum_{i=1}^{N}\alpha_i y_i=0
$$

$$
0 \leq \alpha_i \leq C , i =1,2\cdots,N
$$

求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots ,\alpha_N^*)^T$

（2）计算
$$
w^*=\sum_{i=1}^{N}\alpha_i^*y_ix_i
$$
选择$\alpha^*$的一个分量$\alpha_j^*$，适合条件$0 \leq \alpha_i \leq C$，计算
$$
b^*=y_j - \sum_{i=1}^{N} y_i \alpha_i^*(x_i \cdot x_j)
$$
求得超平面：
$$
w^* \cdot x + b^*=0
$$
分类决策函数：
$$
f(x) = sign(w^* \cdot x+b^*)
$$

### 合页损失函数（hinge loss function）

$$
L(y(w \cdot x+b))=[1-y(w \cdot x + b)]_{+}
$$

下标‘+’表示以下取正值的函数：
$$
[z]_+=\begin{cases}
z & z>0 \\
0 & z\leq 0
\end{cases}
$$
也就是说，当样本点$(x_i,y_i)$被正确分类且函数间隔$y_i(w \cdot x_i+b)$大于1，损失是0，否则损失是$y_i(w \cdot x_i+b)$

---

### 定理7.4 

线性支持向量机原始最优化问题：
$$
\min_\limits {w,b,\xi} \frac{1}{2} \parallel w \parallel ^2 + C\sum_{i=1}^{N}\xi_i
$$

$$
s.t.\, y_i(w \cdot x_i+b) \geq 1- \xi_i, i=1,2,\cdots, N
$$

$$
\xi_i \geq 0, i=1,2,\cdots, N
$$

等价于最优化问题：
$$
\min_\limits {w,b}  \sum_{i=1}^{N} [1-y_i(w \cdot x_i + b)]_+ + \lambda \parallel w \parallel ^2
$$


## 3、非线性支持向量机与核函数

核技巧应用到支持向量机，基本思想是通过一个非线性变换将输入空间对应于一个特征空间，使得输入空间中的超曲面模型对应于特征空间中的超平面模型。

引入核函数后的对偶问题的目标函数为：
$$
W(\alpha) = \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j K(x_i,x_j) - \sum_{i=1}^{N} \alpha_i
$$
分类决策函数为：
$$
f(x)=sign(\sum_{i=1}^{N}\alpha_i^* y_i \phi(x_i) \cdot\phi(x)+b^*)=sign(\sum_{i=1}^{N}\alpha_i^* y_i K(x_i,x)+b^*)
$$
常用核函数：

- 多项式核函数（polynomial kernel function）
  $$
  K(x,z)=(x \cdot z +1)^p
  $$

- 高斯核函数（Gaussian kernel function）径向基核函数
  $$
  K(x,z)=\exp(-\frac{\parallel x-z \parallel^2}{2\sigma^2})
  $$

- 字符串核函数（string kernel function）



---

### 算法7.4（非线性支持向量机学习算法）

输入：线性可分训练数据集$T={(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)}$，
其中$x_i \in \mathbb{R}$，$y_i=\{+1, -1\}$，$i=1,2,\cdots,N$；  

输出：分离超平面和分类决策函数。 

（1）选择适当的核函数$K(x,z)$惩罚参数$C>0$，构造并求解凸二次规划问题
$$
\min \limits_{\alpha}\, \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i \alpha_j y_i y_j K(x_i,x_j )-\sum_{i=1}^{N}\alpha_i
$$

$$
s.t.\,\sum_{i=1}^{N}\alpha_i y_i=0
$$

$$
0 \leq \alpha_i \leq C , i =1,2\cdots,N
$$

求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots ,\alpha_N^*)^T$

（2）选择$\alpha^*$的一个正分量$0<\alpha^*<C$，计算
$$
b^*=y_j - \sum_{i=1}^{N}\alpha_i^*y_iK(x_i , x_j)
$$


（3）分类决策函数：
$$
f(x) = sign \left(\sum_{i=1}^{N}\alpha^*_i y_i K(x , x_i)+b^* \right)
$$

### 4、序列最小最优化算法（sequential minimal optimization, SMO）

SMO算法要解如下凸二次规划的对偶问题：
$$
\min \limits_\alpha \frac{1}{2} \sum_{i=1}^{N}\sum_{j=1}^{N} \alpha_i \alpha_j y_i y_jK(x_i, x_j) - \sum_{i=1}^{N}\alpha_i
$$

$$
s.t. \, \sum_{i=1}^{N}\alpha_i y_i=0
$$

$$
0 \leq\alpha_i \leq C \, ,i=1,2,\cdots,N
$$

假设$\alpha_1,\alpha_2$为两个变量，$\alpha_3,\alpha_4,\cdots,\alpha_N$固定，由约束条件可知
$$
\alpha_1=- y_1 \sum_{i=2}^{N}\alpha_i y_i
$$
如果$\alpha_2$确定，那么$\alpha_1$也随之确定。

于是SMO的最优化问题的子问题可以写成：
$$
\min \limits_{\alpha_1,\alpha_2} W(\alpha_1,\alpha_2)=\frac{1}{2}K_{11}\alpha_1^2 +
\frac{1}{2}K_{22}\alpha_2^2 + y_1 y_2K_{12}\alpha_1 \alpha_2 - (\alpha_1 + \alpha_2)
+ y_1\alpha_1\sum_{i=3}^{N}y_i \alpha_iK_{i1}+
y_2\alpha_2\sum_{i=3}^{N}y_i\alpha_iK_{i2}
$$

$$
s.t. \, \alpha_1y_1+\alpha_2 y_2=-\sum_{i = 3}^{N}y_i\alpha_i=\varsigma
$$

$$
0 \leq\alpha_i \leq C, i=1,2
$$

其中$K_{ij}=K(x_i,x_j)$，$\varsigma$ 是常数

假设问题的初始可行解为$\alpha_1^{old},\alpha_2^{old}$，最优解为$\alpha_1^{new},\alpha_2^{new}$，并且假设在沿着约束方向未经剪辑时的$\alpha_2$的最优解为$\alpha_2^{new,unc}$。

$\alpha_2^{new}$满足条件
$$
L \leq \alpha_2^{new} \leq H
$$
 （1）当$y_1=y_2$时，有$\alpha_1+\alpha_2=k$，
$$
\begin{cases}
L=\max(0, \alpha_2^{old}+\alpha_1^{old}-C)\\
H=\min(C, \alpha_2^{old}+\alpha_1^{old})
\end{cases}
$$
 （2）当$y_1 \neq y_2$时，有$\alpha_1-\alpha_2=k$，
$$
\begin{cases}
L=\max(0, \alpha_2^{old}-\alpha_1^{old})\\
H=\min(C, C+\alpha_2^{old}-\alpha_1^{old})
\end{cases}
$$

---

### 定理7.6

记
$$
g(x)=\sum_{i=1}^{N}\alpha_iy_iK(x_i,x)+b
$$
 令
$$
E_i=g(x_i)-y_i=\left( \sum_{j=1}^{N}\alpha_jy_jK(x_j,x_i)+b \right) -y_i,i=1,2
$$
最优化问题沿着约束方向未经剪辑的解是
$$
\alpha_2^{new,unc}=\alpha_2^{old} + \frac{y_2(E_1-E_2)}{\eta}
$$
其中
$$
\eta=K_{11}+K_{22}-2K_{12}=\parallel \Phi(x_1) - \Phi(x_2) \parallel^2
$$
经剪辑后$\alpha_2$的解是
$$
\alpha_2^{new} =
\begin{cases} 
H,  & \alpha_2^{new,unc}>H \\
\alpha_2^{new,unc},  & L \leq \alpha_2^{new,unc} \leq H \\
L, & \alpha_2^{new,unc}<L
\end{cases}
$$
由$\alpha_2^{new}$求得$\alpha_1^{new}$
$$
\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})
$$
当$0<\alpha_1^{new}<C$时
$$
b_1^{new}=-E_1 - y_1K_{11}(\alpha_1^{new}-\alpha_1^{old})-y_2K_{21}(\alpha_2^{new}-\alpha_2^{old}) + b^{old}
$$
当$0<\alpha_2^{new}<C$时
$$
b_2^{new}=-E_2 - y_1K_{12}(\alpha_1^{new}-\alpha_1^{old})-y_2K_{22}(\alpha_2^{new}-\alpha_2^{old}) + b^{old}
$$

如果$\alpha_1^{new},\alpha_2^{new}$同时满足$0<\alpha_i^{new}<C$，那么$b^{new}=b_1^{new}=b_2^{new}$。

如果$\alpha_1^{new},\alpha_2^{new}$是$0，C$，那么。$b^{new}=(b_1^{new}+b_2^{new})/2$。

更新$E_i$
$$
E_i^{new}=\sum_S y_j\alpha_j K(x_i,x_j)+b^{new}-y_i
$$
其中$S$是所有支持向量$x_j$的集合。