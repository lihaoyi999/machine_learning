\documentclass[a4paper,11pt]{ctexart}
\title{支持向量机}
\author{}
\date{}

\usepackage{geometry}
\usepackage{cite}
\usepackage{latexsym}
\usepackage{amsmath}
\usepackage{amsfonts}


\CTEXsetup[name={第,节}]{section}
\CTEXsetup[beforeskip = {20bp plus 1ex minus 0.2ex}]{section}
\CTEXsetup[afterskip = {6bp plus 0.2ex}]{section}
\CTEXsetup[format = {\zihao{4}\bfseries}]{section}
\CTEXsetup[name={第,小节}]{subsection}
\CTEXsetup[beforeskip = {12bp plus 1ex minus 0.2ex}]{subsection}
\CTEXsetup[afterskip = {6bp plus 0.2ex}]{subsection}
\CTEXsetup[format = {\fontsize{13bp}{15.6bp}\selectfont\bfseries}]{subsection}
\CTEXsetup[beforeskip = {12bp plus 1ex minus 0.2ex}]{subsubsection}
\CTEXsetup[afterskip = {6bp plus 0.2ex}]{subsubsection}
\CTEXsetup[format = {\zihao{-4}\bfseries}]{subsubsection}
\geometry{
	a4paper, hmargin = 2.6cm, top = 2.92cm, bottom = 3.03cm,
	headheight = 0.45cm, headsep = 0.55cm, footskip = 1.05cm
}


\begin{document}
\maketitle

\pagestyle{plain}

\section{线性分类问题}
考虑输入空间 $X$ 为 $\mathbb{R}^N (N \geq 1)$ 的一个子集，输出空间为 $Y = \{-1,+1\}$，目标函数为 $f: X \to Y$。二分类任务可以表述为：利用根据未知分布 $D$ 从样本空间 $X$ 独立同同分布采样得到的样本集合 $S = ((x_1,y_1),(x_2,y_2), \ldots, (x_m,y_m))$, 并且 $f(x_i) = y_i, \forall\; i \in [1, m]$，我们希望从假设函数空间 $H$ 中找出一个最优的 $h \in H$，使得如下泛化误差最小：
\begin{equation}
    R_D(h) = \Pr_{x \sim D}     [h(x) \neq f(x)].
\end{equation}
一个自然而简单的假设是目标函数为线性分类器，或者说是 $N$ 维空间中的超平面:
\begin{equation}
    H = \{ \mathbf{x} \mapsto sign(\mathbf{w} \cdot \mathbf{x} + b): \mathbf{w} \in \mathbb{R}^N, b \in \mathbb{R}\}.
\end{equation}
具有上述形式的函数 $h$ 将超平面 $\mathbf{w} \cdot \mathbf{x} + b = 0$ 两侧的点标注为 $+1$ 和 $-1$，因而上述学习问题被称为“线性分类问题”。

\section{SVM——线性可分情形}
假设样本集合 $S$ 是线性可分的，即存在超平面完美地将样本集合中的正负样本分开。由于参数空间是连续的，必然存在无穷多个在样本集上完成分类任务的超平面。因此，我们可以选择一个最“好”的超平面，使得正负样本与超平面
之间具有最大边界。

\subsection{主优化问题}
注意到对超平面的参数 $(\mathbf{w},b)$ 等比例放缩并不改变超平面的几何性质，因此可以通过适当的放缩使得 $\min_{(\mathbf{x},y) \in S} |\mathbf{w} \cdot \mathbf{x} + b|=1$，放缩后的超平面称为\textbf{规范超平面}。由定义可得，对样本中的任意 $x_i(i \in [1,m])$，有 $|\mathbf{w} \cdot \mathbf{x_i} + b| \geq 1$。

从解析几何的结论可知，空间中任意一点 $x_0 \in \mathbb{R}^N$ 到超平面 $\mathbf{w} \cdot \mathbf{x} + b = 0$ 的距离为：
\begin{equation}
    \frac{|\mathbf{w} \cdot \mathbf{x_0} + b|}{||\mathbf{w}||}.
\end{equation}
对于规范超平面，定义其边界大小 $\rho$ 为：
\begin{equation}
\rho = \min_{(\mathbf{x},y) \in S} \frac{|\mathbf{w} \cdot \mathbf{x} + b|}{||\mathbf{w}||} = \frac{1}{||\mathbf{w}||}
\end{equation}
因此，求取具有最大边界的超平面可以表述为如下优化问题：
\begin{eqnarray}
\max_{\mathbf{w},b} && \frac{1}{||\mathbf{w}||} \nonumber\\
subject\;to: && y_i(\mathbf{w} \cdot \mathbf{x} + b) \geq 1, \forall\; i \in [1,m] \nonumber
\end{eqnarray}
等价于：
\begin{eqnarray} \label{eqnarray:prime}
\min_{\mathbf{w},b} && \frac{1}{2}||\mathbf{w}||^2\\
subject\;to: && y_i(\mathbf{w} \cdot \mathbf{x} + b) \geq 1, \forall\; i \in [1,m] \nonumber
\end{eqnarray}
显而易见，该优化问题的目标函数为凸函数，约束条件为线性不等式组，因而是典型的二次规划问题（QP），可以用成熟的商业求解器求解。

\subsection{对偶优化问题}
对优化问题（\ref{eqnarray:prime}）引入拉格朗日乘子 $\alpha_i \geq 0, i \in [1,m]$，构造拉格朗日函数 
$$L(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^{m}\alpha_i[y_i(\mathbf{w} \cdot \mathbf{x} + b)-1],$$
得到原目标函数的对偶函数为
\begin{eqnarray}
g(\mathbf{\alpha}) & = &\inf_{\mathbf{w},b} L(\mathbf{w},b,\mathbf{\alpha}) \nonumber\\
& = & \sum_{i=1}^{m}\alpha_i + \inf_{\mathbf{w}}\bigg\{\frac{1}{2}||\mathbf{w}||^2 - (\sum_{i=1}^{m}\alpha_i y_i\mathbf{x})\cdot\mathbf{w} \bigg\} + \inf_{b}\bigg\{- \sum_{i=1}^{m}\alpha_i y_ib\bigg\} \nonumber\\
& = &  
\left\{ \begin{array}{ll}
\sum_{i=1}^{m}\alpha_i - \sum_{i,j=1}^{m}\alpha_i \alpha_j y_i y_j(\mathbf{x_i} \cdot \mathbf{x_j})& \textrm{如果 $\sum_{i=1}^{m}\alpha_i y_i = 0$}\\
-\infty & \textrm{其他情况}\\
\end{array} \right.
\end{eqnarray}
因而，对偶问题可以表述为：
\begin{eqnarray} \label{eqnarray:dual}
\max_{\mathbf{\alpha}} && \sum_{i=1}^{m}\alpha_i - \sum_{i,j=1}^{m}\alpha_i \alpha_j y_i y_j(\mathbf{x_i} \cdot \mathbf{x_j})\\
subject\;to: && \sum_{i=1}^{m}\alpha_i y_i = 0 \nonumber\\
&& \alpha_i \geq 0,\;\; i=1,\ldots,m \nonumber
\end{eqnarray}
对偶优化问题的目标函数为凸函数，约束条件为线性不等式组，也是典型的二次规划问题（QP）。

\subsection{支持向量}
由于原问题和对偶问题的不等式约束条件都是线性的，强对偶条件成立，故最优解满足KKT条件：
\begin{eqnarray}
\label{eq:KKT1}
\bigtriangledown_{\mathbf{w}}L = \mathbf{w}-\sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i = 0 &\Longrightarrow& \mathbf{w} = \sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i \\
\label{eq:KKT2}
\bigtriangledown_{b}L = -\sum_{i=1}^{m} \alpha_i y_i = 0 &\Longrightarrow& \sum_{i=1}^{m} \alpha_i y_i = 0 \\
\label{eq:KKT3}
\forall i, \alpha_i[y_i((\mathbf{w} \cdot \mathbf{x_i} + b)-1] = 0 &\Longrightarrow& \alpha_i = 0 \vee y_i(\mathbf{w} \cdot \mathbf{x_i} + b) = 1.
\end{eqnarray}
从互补松弛条件（\ref{eq:KKT3}）可知，若 $\mathbf{x}_i$ 不是距离分离超平面最近的点，即 $|\mathbf{w} \cdot \mathbf{x}_i + b| \neq 1$，则必有 $\alpha_i = 0$。反映在等式 （\ref{eq:KKT1}）中，说明上述 $\mathbf{x}_i$ 对超平面方向的选择没有影响。因此，决定分离超平面方向 $\mathbf{w}$ 的只有那些使 $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1$ 成立的样本点，称这些点为\textbf{支撑向量}。

\section{SVM——线性不可分情形}
在数据线性不可分时，无法找到参数 $(\mathbf{w},b)$ 使得 $\mathbf{w}\cdot\mathbf{x}_i+b$ 与 $y_i$ 的符号一致，也就是说，对任意超平面 $\mathbf{w}\cdot\mathbf{x}+b = 0$，存在 $\mathbf{x}_i \in S$ 使得
\begin{equation}
    y_i[\mathbf{w}\cdot\mathbf{x}_i+b] < 1.
\end{equation}
为此，我们引入松弛变量 $\xi_i \geq 0$，使得
\begin{equation}
    y_i[\mathbf{w}\cdot\mathbf{x}_i+b] \geq 1-\xi_i.
\end{equation}
这样一来，主优化问题改写成：
\begin{eqnarray} \label{eqnarray:prime2}
\min_{\mathbf{w},b} && \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m}\xi_i^p\\
subject\;to: && y_i(\mathbf{w} \cdot \mathbf{x} + b) \geq 1-\xi_i,  \nonumber \\
&& \xi_i \geq 0. \; i\in [1,m] \nonumber
\end{eqnarray}
下面只考虑 $p=1$ 时的情形。

\subsection{对偶优化问题}
对优化问题（\ref{eqnarray:prime2}）引入拉格朗日乘子 $\alpha_i,\beta_i \geq 0, i \in [1,m]$，构造拉格朗日函数 
$$L(\mathbf{w},b,\mathbf{\xi},\mathbf{\alpha},\mathbf{\beta}) = \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m}\xi_i - \sum_{i=1}^{m}\alpha_i[y_i(\mathbf{w} \cdot \mathbf{x} + b)-1+\xi_i] - \sum_{i=1}^{m}\beta_i\xi_i,$$
得到原目标函数的对偶函数为
\begin{eqnarray}
g(\mathbf{\alpha},\mathbf{\beta}) & = &\inf_{\mathbf{w},b,\mathbf{xi}} L(\mathbf{w},b,\mathbf{\xi},\mathbf{\alpha},\mathbf{\beta}) \nonumber\\
& = & \sum_{i=1}^{m}\alpha_i \nonumber\\
&& + \inf_{\mathbf{w}}\bigg\{\frac{1}{2}||\mathbf{w}||^2 - (\sum_{i=1}^{m}\alpha_i y_i\mathbf{x})\cdot\mathbf{w} \bigg\} \nonumber\\
&& + \inf_{b}\bigg\{- \sum_{i=1}^{m}\alpha_i y_ib\bigg\} \\
&& + \inf_{\mathbf{\xi}}\bigg\{ \sum_{i=1}^{m}(C-\alpha_i-\beta_i)\xi_i\bigg\} \nonumber\\
& = &  
\left\{ \begin{array}{ll}
\sum_{i=1}^{m}\alpha_i - \sum_{i,j=1}^{m}\alpha_i \alpha_j y_i y_j(\mathbf{x_i} \cdot \mathbf{x_j}), & \textrm{如果 $\sum_{i=1}^{m}\alpha_i y_i = 0$ 且 $\alpha_i+\beta_i = C$，}\\
-\infty, & \textrm{其他情况。}\\
\end{array} \right. \nonumber
\end{eqnarray}
因而，对偶问题可以表述为：
\begin{eqnarray} 
\max_{\mathbf{\alpha}} && \sum_{i=1}^{m}\alpha_i - \sum_{i,j=1}^{m}\alpha_i \alpha_j y_i y_j(\mathbf{x_i} \cdot \mathbf{x_j}) \nonumber\\
subject\;to: && \sum_{i=1}^{m}\alpha_i y_i = 0 \nonumber\\
&& \alpha_i + \beta_i = C, \nonumber \\
&& \alpha_i \geq 0, \nonumber\\
&& \beta_i \geq 0, \;\; i=1,\ldots,m \nonumber
\end{eqnarray}
等价于：
\begin{eqnarray} \label{eqnarray:dual2}
\max_{\mathbf{\alpha}} && \sum_{i=1}^{m}\alpha_i - \sum_{i,j=1}^{m}\alpha_i \alpha_j y_i y_j(\mathbf{x_i} \cdot \mathbf{x_j})\\
subject\;to: && \sum_{i=1}^{m}\alpha_i y_i = 0 \nonumber\\
&& 0 \leq \alpha_i \leq C, \;\; i=1,\ldots,m \nonumber
\end{eqnarray}


\subsection{KKT条件}
线性不可分问题的最优解满足如下KKT条件：
\begin{eqnarray}
\label{eq:KKT21}
\bigtriangledown_{\mathbf{w}}L = \mathbf{w}-\sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i = 0 &\Longrightarrow& \mathbf{w} = \sum_{i=1}^{m} \alpha_i y_i \mathbf{x}_i \\
\label{eq:KKT22}
\bigtriangledown_{b}L = -\sum_{i=1}^{m} \alpha_i y_i = 0 &\Longrightarrow& \sum_{i=1}^{m} \alpha_i y_i = 0 \\
\label{eq:KKT23}
\bigtriangledown_{\xi_i}L = C-\alpha_i-\beta_i = 0 &\Longrightarrow& \alpha_i + \beta_i = C \\
\label{eq:KKT24}
\forall i, \alpha_i[y_i((\mathbf{w} \cdot \mathbf{x_i} + b)-1+\xi_i] = 0 &\Longrightarrow& \alpha_i = 0 \vee y_i(\mathbf{w} \cdot \mathbf{x_i} + b) = 1-\xi_i.\\
\label{eq:KKT25}
\forall i, \beta_i\xi_i = 0 &\Longrightarrow& \beta_i = 0 \vee \xi_i = 0.
\end{eqnarray}
结合 （\ref{eq:KKT23}）和（\ref{eq:KKT24}），（\ref{eq:KKT25}）可改写成：
\begin{equation}
    \alpha_i = C \vee y_i(\mathbf{w} \cdot \mathbf{x_i} + b) = 1.
\end{equation}
这说明，对于 $\alpha_i \neq 0$ 对应的支撑向量 $\mathbf{x}_i$，要么正好落在边界平面上： $y_i(\mathbf{w} \cdot \mathbf{x_i} + b) = 1$，要么其对应的 $\alpha_i$ 正好等于$C$。

\section{核函数}
从对偶问题 （\ref{eqnarray:dual}）和 （\ref{eqnarray:dual2}）发现，优化问题的目标函数只和样本点的内积 $(\mathbf{x}_i,\mathbf{x}_j)$ 有关。由此可以设想，把单纯的内积用某种二元函数 $K(\mathbf{x}_i,\mathbf{x}_j)$ 替换，使其等效于在另一种维度的空间（称之为\textbf{特征空间}）里做内积运算，这样或许能将原空间中的线性不可分问题变换成特征空间中的线性可分问题。

\paragraph{多项式核函数}
对常数 $c >0$，定义 \texttt{d维多项式核函数}{} 为：
\begin{equation}
    \forall \mathbf{x,x'} \in \mathbb{R}^N, K(\mathbf{x,x'}) = (\mathbf{x}\cdot\mathbf{x'}+c)^d
\end{equation}

\paragraph{高斯核函数}
对常数 $\sigma >0$，定义 \texttt{高斯核函数}{} 或者 \texttt{径向基函数}{} 为：
\begin{equation}
    \forall \mathbf{x,x'} \in \mathbb{R}^N, K(\mathbf{x,x'}) = exp\big(-\frac{||\mathbf{x'}-\mathbf{x}||^2}{2\sigma^2}\big)
\end{equation}

\paragraph{sigmoid核函数}
对常数 $a,b \geq 0$，定义 \texttt{sigmoid核函数}{} 为：
\begin{equation}
    \forall \mathbf{x,x'} \in \mathbb{R}^N, K(\mathbf{x,x'}) = tanh(a(\mathbf{x}\cdot\mathbf{x'})+b)
\end{equation}

因此，基于核函数的 SVM 表述为如下问题：
\begin{eqnarray} \label{eqnarray:kernel}
\max_{\mathbf{\alpha}} && \sum_{i=1}^{m}\alpha_i - \sum_{i,j=1}^{m}\alpha_i \alpha_j y_i y_j K(\mathbf{x_i},\mathbf{x_j})\\
subject\;to: && \sum_{i=1}^{m}\alpha_i y_i = 0 \nonumber\\
&& 0 \leq \alpha_i \leq C, \;\; i=1,\ldots,m \nonumber
\end{eqnarray}
\end{document}
