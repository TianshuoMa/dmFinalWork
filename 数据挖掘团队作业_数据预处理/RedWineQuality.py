import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minepy import MINE
import pandas as pd
import copy


# 直方图可视化
def visualBar(traindata, item):
    plt.figure(figsize=(20, 5))
    for i in range(len(item) - 1):
        plt.subplot(3, 4, i + 1)
        sns.distplot(traindata[item[i]], kde=False)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()


# 盒图可视化
def visualBox(traindata, item):
    plt.figure(figsize=(10, 15))
    for i in range(len(item) - 1):
        plt.subplot(3, 4, i + 1)
        plt.boxplot(traindata[item[i]])
        plt.title(item[i])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.show()


def replace(x):
    y = 12.350
    return y


if __name__ == "__main__":
    path = 'winequality-red.csv'
    train = pd.read_csv(path)
    train.info()

    items = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol', 'quality']

    # 数据描述：五数概括、空值统计
    m1 = train.describe()
    print("-"*6, "m1", "-"*6)
    print(m1)
    m2 = np.array(m1)
    print("-" * 6, "m2", "-" * 6)
    print(m2)
    m2 = train.isnull().sum()

    # 可视化：直方图、盒图
    show = True
    if show:
        visualBar(train, items)
        visualBox(train, items)

    # data = train.iloc[:, 0:1]
    # print(data)
    # s = data.describe()
    # print(s)
    # mean = s.loc['mean', 'fixed acidity']
    # std = s.loc['std', 'fixed acidity']
    # noise = data[(data['fixed acidity'] < (mean - 3*std)) | (data['fixed acidity'] > (mean + 3*std))]
    # print(noise)
    # print(noise.index)
    # n_index = np.array(noise.index)
    # print(n_index)

    # 处理噪声数据
    index_list = []
    non_noise = train
    for i in range(len(items) - 1):
        data = train.iloc[:, i:i+1]
        s = data.describe()
        mean = s.loc['mean', items[i]]
        std = s.loc['std', items[i]]
        noise = data[(data[items[i]] < (mean - 3*std)) | (data[items[i]] > (mean + 3*std))]
        non_noise = non_noise[(non_noise[items[i]] > (mean - 3*std)) & (non_noise[items[i]] < (mean + 3*std))]
        noise_index = np.array(noise.index)
        nn_index = np.array(non_noise.index)
        print(len(nn_index))
        index_list.extend(noise_index)
    l = list(set(index_list))
    l.sort()
    print(l)
    print(len(l))

    nn_index = np.array(non_noise.index)
    print(len(nn_index))

    show = True
    if show:
        visualBar(non_noise, items)
        visualBox(non_noise, items)

    m = non_noise.describe()
    print(m)

    # 相关性热图
    figure, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(non_noise.corr(), square=True, annot=True, ax=ax)
    plt.show()



