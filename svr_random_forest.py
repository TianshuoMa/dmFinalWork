#!/usr/bin/env python
# coding: utf-8

# # Support vector regression （SVR）
# 对于样本$(x,y)$,传统的回归模型通常直接输出$f(x)$与真实输出$y$之间的差别来计算损失，当且仅当$f(x)$与y完全相同时，损失才为0。与此不同svr假设可以容忍$f(x)$与$y$之间最多有$\epsilon$的偏差，即当且仅当$f(x)$与$y$之间的差别绝对值大于$\epsilon$时才计算损失。这相当于以$f(x)$为中心，构建一个宽度为$2\epsilon$的间隔带，若样本落入此间隔带，则认为是预测正确的。于是SVR问题可以建模为：
# $$\min _{\omega, b} \frac{1}{2}\|\omega\|^{2}+C \sum_{i=1}^{m} \ell_{\epsilon}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}\right)$$，
# 其中C是正则化常数，$\ell_{\epsilon}$是$\epsilon-$不敏感损失函数：
# $$\ell_{\epsilon}(z)=\left\{\begin{array}{ll}0, & \text { if }|z| \leq \epsilon \\|z|-\epsilon, & \text { otherwise }\end{array}\right.$$
# 引入松弛变量$\xi_i$和$\hat{\xi}_i$（间隔两侧的松弛程度有可能不同），可以将上式重写为：
# $$
# \begin{array}
# \min_{\omega, b} \frac{1}{2}\|\omega\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right) \\
# \text { s.t. } f\left(\boldsymbol{x}_{i}\right)-y_{i} \leq \epsilon+\xi_{i} \\
# y_{i}-f\left(\boldsymbol{x}_{i}\right) \leq \epsilon+\hat{\xi}_{i} \\
# \xi_{i}>0 \hat{\xi}_{i}>0 i=1,2,3 \ldots m
# \end{array}
# $$
# 拉格朗日对偶形式
# 通过哦引入$\mu_{i} \geq 0, \hat{\mu}_{i} \geq 0, \alpha_{i} \geq 0, \hat{\alpha}_{i} \geq 0$,由拉格朗日乘子可以得到拉格朗日函数：
# $$
# \begin{array}{l}
# L(\boldsymbol{\omega}, b, \boldsymbol{\alpha}, \hat{\boldsymbol{\alpha}}, \boldsymbol{\xi}, \hat{\boldsymbol{\xi}}, \boldsymbol{\mu}, \hat{\boldsymbol{\mu}}) \\=\frac{1}{2}\|\boldsymbol{\omega}\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi} i\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu} i \hat{\xi}_{i} \\+\sum i=1^{m} \alpha_{i}\left(f(\boldsymbol{x} i)-y_{i}-\epsilon-\xi_{i}\right) \\+\sum i=1^{m} \hat{\alpha}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)\end{array}$$
# 将$f\left(\boldsymbol{x}_{i}\right)=\boldsymbol{w}^{T} \boldsymbol{x}+b$带入上式，并令$L(\boldsymbol{\omega}, b, \boldsymbol{\alpha}, \hat{\boldsymbol{\alpha}}, \boldsymbol{\xi}, \hat{\boldsymbol{\xi}}, \boldsymbol{\mu}, \hat{\boldsymbol{\mu}})$的偏导为0，得到：
# $$\begin{aligned}\boldsymbol{\omega} &=\sum_{i=1}^{m}\left(\hat{\alpha} i-\alpha_{i}\right) \boldsymbol{x}_{i} \\0 &=\sum i=1^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \\C &=\alpha_{i}+\mu_{i} \\C &=\hat{\alpha}_{i}+\hat{\mu}_{i}\end{aligned}$$
# 
# KKT与最终决策函数
# 上述过程满足的KKT条件为：
# $$\left\{\begin{array}{l}\alpha_{i}\left(f\left(\boldsymbol{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)=0 \\\hat{\alpha}_{i}\left(y_{i}-f\left(\boldsymbol{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)=0 \\\alpha_{i} \hat{\alpha}_{i}=0, \quad \xi_{i} \hat{\xi}_{i}=0 \\\left(C-\alpha_{i}\right) \xi_{i}=0,\left(C-\hat{\alpha}_{i}\right) \hat{\xi}_{i}=0\end{array}\right.$$
# 最终决策函数为
# $$f(\boldsymbol{x})=\sum_{i=1}^{n}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \boldsymbol{x}_{i}^{T} \boldsymbol{x}_{j}+b$$
# 
# 能使上式中$\hat{\alpha}_{i}-\alpha_{i} \neq 0$成立的样本即为SVR的支持向量，他们必然落在间隔带之外。
# 核函数的形式最终的决策函数为
# $$f(\boldsymbol{x})=\sum_{i=1}^{n}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)+b$$
# 其中$\kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)$为核函数。
# 
# 本文采用径向核函数在训练集上训练svc模型，并在测试集上对分析了训练之后的模型的准确率。代码如下：
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn import metrics


def StandardLinearSVR(epsilon=0.25):
    return Pipeline([
        ("std_scaler",StandardScaler()),
        ("linearSVR",SVR(kernel = 'rbf', epsilon  = epsilon))
    ])

data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\winequality-red.csv'
# data_name2 = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\winemag-data-130k.csv'
data_train_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\data_train.csv'
target_train_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\target_train.csv'
data_test_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\data_test.csv'
target_test_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\target_test.csv'



file = pd.read_csv(data_name)
data = pd.DataFrame(file)

file_data_train = pd.read_csv(data_train_name)
data_train = pd.DataFrame(file_data_train)

file_data_test = pd.read_csv(data_test_name)
data_test = pd.DataFrame(file_data_test)

file_target_train = pd.read_csv(target_train_name)
target_train  = pd.DataFrame(file_target_train).values.flatten()

file_target_test = pd.read_csv(target_test_name)
target_test = pd.DataFrame(file_target_test).values.flatten()



data_att = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
target_att = ['quality']



svr = StandardLinearSVR()
svr.fit(data_train,target_train)
target_test_pre = svr.predict(data_test)
target_test_pre = [round(x) for x in target_test_pre]
# print(target_test_pre)
print(metrics.accuracy_score(target_test,target_test_pre))


score1= svr.score(data_test,target_test)
print(score1)


# 最终得到的准确率为61.25%。
# 在SVC中，两个超参数对整体的性能影响很大，分别是惩罚系数与epsilon（其中训练孙树函数中没有惩罚与在实际值的距离epsilon内预测的点）。

# # 决策树
# 分类决策树模型是一种描述对实例进行分类的树形结构。决策树由节点和有向边组成。结点的类型有两种，内部结点和叶结点。内部结点表示一个特征或属性，叶结点表示一个类。用决策树分类，从根结点开始，对实例的某一特征进行测试，根据测试结果，将实例分配到其子结点；这时，每一个结点对应着该特征的一个取值。如此递归地对实例进行测试并分配，直至达到叶结点。最终将实例分到叶结点的类中。
# 决策树的算法主要以下三种：
# * ID3算法的核心是在决策树各个结点上应用信息增益准则来选择特征，递归地构建决策树。具体方式是：从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点；再对子节点递归调用上述方法，构建决策树；知道所有特征的信息增益都很小或者没有特征为止。
# * C4.5算法使用了信息增益比来选择特征，可以看作是ID3算法的一种改进
# * CART算法在生成树的过程中吗，使用了基尼指数的最小化原则。
# 具体代码如下：

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree

def StandardLinearSVR(epsilon=0.25):
    return Pipeline([
        ("std_scaler",StandardScaler()),
        ("linearSVR",SVR(kernel = 'rbf', epsilon  = epsilon))
    ])

data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\winequality-red.csv'
# data_name2 = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\winemag-data-130k.csv'
data_train_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\data_train.csv'
target_train_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\target_train.csv'
data_test_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\data_test.csv'
target_test_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\target_test.csv'



file = pd.read_csv(data_name)
data = pd.DataFrame(file)

file_data_train = pd.read_csv(data_train_name)
data_train = pd.DataFrame(file_data_train)

file_data_test = pd.read_csv(data_test_name)
data_test = pd.DataFrame(file_data_test)

file_target_train = pd.read_csv(target_train_name)
target_train  = pd.DataFrame(file_target_train).values.flatten()

file_target_test = pd.read_csv(target_test_name)
target_test = pd.DataFrame(file_target_test).values.flatten()

clf_c4 = DecisionTreeClassifier(criterion = 'entropy')
clf_c4 = clf.fit(data_train, target_train)

pre = clf_c4.predict(data_test)
pre_score = metrics.accuracy_score(target_test,pre)
print('利用C4.5算法对数据进行拟合，并在测试集上的准确率为:')
print(pre_score)

clf = DecisionTreeClassifier()
clf = clf.fit(data_train, target_train)

pre = clf.predict(data_test)
pre_score = metrics.accuracy_score(target_test,pre)
print('利用CART算法对数据进行拟合，并在测试集上的准确率为')
print(pre_score)


# 由此可以看出利用C4.5算法对数据进行拟合，并在测试集上的准确率为：63.5%, 利用CART算法对数据进行拟合，并在测试集上的准确率为62.5%。将利用C4.5算法获得的决策树绘制为：

# In[51]:


from sklearn import tree
import graphviz

plt.figure(figsize=(500,200))
tree.plot_tree(clf, filled=True)

plt.show()


# # 随机森林
# 随机森林是Bagging的一个拓展变体。RF是在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择。具体来说，传统决策树在选择划分属性时是在当前结点的属性集合中选择一个最优属性；而在RF中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含k个属性的子集，然后再从这个子集中选择一个最优属性用于划分，这里的参数k控制了随机性引入程度。随机森林算法中有许多超参数会影响整体的性能，具体来说，参数n_estimators(森林里决策树的数目)，max_depth（决策树的最大深度），min_samples_leaf （需要在叶子结点上的最小样本数量），max_features（寻找最佳分割需要考虑的特征数目）。本文，将n_estimators设置为3000，max_depth设置为默认值，即最大深度，min_samples_leaf设置为1，max_features设置为'auto'。

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\winequality-red.csv'
# data_name2 = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\winemag-data-130k.csv'
data_train_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\data_train.csv'
target_train_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\target_train.csv'
data_test_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\data_test.csv'
target_test_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\大作业\target_test.csv'



file = pd.read_csv(data_name)
data = pd.DataFrame(file)

file_data_train = pd.read_csv(data_train_name)
data_train = pd.DataFrame(file_data_train)

file_data_test = pd.read_csv(data_test_name)
data_test = pd.DataFrame(file_data_test)

file_target_train = pd.read_csv(target_train_name)
target_train  = pd.DataFrame(file_target_train).values.flatten()

file_target_test = pd.read_csv(target_test_name)
target_test = pd.DataFrame(file_target_test).values.flatten()

rf = RandomForestClassifier(n_estimators = 2000, oob_score = True, n_jobs = -1,random_state =20,max_features = 'auto', min_samples_leaf = 1)
rf.fit(data_train, target_train)
predictions = rf.predict(data_test)
# print(predictions)
print('测试集上得到的准确率为：')
print(metrics.accuracy_score(target_test, predictions))


# 测试集合上得到的准确率为0.6925。下面我们对n_estimators进行参数分析，具体来说，我们设置n_estimators的数值设置为0-5000之间，间隔为100，获得对应的分数，代码如下：

# In[29]:


from sklearn.model_selection import cross_val_score
scores = []
# rf = RandomForestClassifier(n_estimators=i+1,random_state=10)
# score = cross_val_score(rf,data_train,target_train,cv=5, scoring='accuracy').mean()
for i in range(0,5000,100):
    rf = RandomForestClassifier(n_estimators=i+1,random_state=10)
    score = cross_val_score(rf,data_train,target_train,cv=5, scoring='accuracy').mean()
    scores.append(score)
print(scores)
plt.plot(range(1,5001,100),scores)
plt.show()


# 由此，可以看出当n_estimators达到1000时，就可以取得较好的结果了，接下来可以使用网格搜索的方式对其他的超参数进行搜索与优化。代码如下：

# In[40]:


from sklearn.model_selection import GridSearchCV
grid_params = {"max_depth":np.arange(1,30,1)}
rf = RandomForestClassifier(n_estimators=2000,random_state=20)
grid = GridSearchCV(rf,grid_params,cv = 5)
grid.fit(data_train,target_train)
print(grid.best_params_,grid.best_score_)


# 可以看出最佳的深度时11，下面将对需要在叶子结点上的最小样本数量进行研究。代码如下，

# In[47]:


grid_params = {"min_samples_leaf":np.arange(1,30,1)}
rf = RandomForestClassifier(n_estimators=1000,random_state=20)
grid = GridSearchCV(rf,grid_params,cv = 5)
grid.fit(data_train,target_train)
print(grid.best_params_,grid.best_score_)


# 可以看出叶子结点上的最小样本数量选取2可以达到最好的结果。将得到的最佳的超参数代入原本的模型中，代码如下

# In[48]:


rf = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =20,max_features = 11, min_samples_leaf = 2)
rf.fit(data_train, target_train)
predictions = rf.predict(data_test)
# print(predictions)
print('测试集上得到的准确率为：')
print(metrics.accuracy_score(target_test, predictions))


# 可以看到，最终的结果并没有显著的提升。

# 将随机森林的方法与决策树的方法在十组交叉验证下的结果进行对比，代码如下：

# In[30]:


clf_l = []
rfc_l = []
for i in range(10):
    clf = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    l1 = cross_val_score(clf,data_train,target_train,cv=5).mean()
    l2 = cross_val_score(rfc,data_train,target_train,cv=5).mean()
    clf_l.append(l1)
    rfc_l.append(l2)
plt.plot(range(1,11),clf_l,label="Decision Tree")
plt.plot(range(1,11),rfc_l,label="RandomForest")
plt.legend()
plt.show()


# 由此可以看出，决策树的方法要远远优越于随机森林的方法。

# In[ ]:




