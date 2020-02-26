# k 近邻算法
import numpy as np
import pandas as pd  # 科学计算和数值分析都会用
from numpy import *

# 1. 数据的加载和预处理
# 直接引用sklearn 里的数据集， iris 鸢尾花的数据集

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # 训练测试划分，切分数据集为训练集和测试集
from sklearn.metrics import accuracy_score  # 评估的方法, （准确率得分）， 计算分类计算单的准确率


# 距离函数定义
def l1_distance(a, b):
    # a 的每一行都减去 b
    return np.sum(np.abs(a - b), axis=1)


def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


# 分类器的实现
class kNN(object):
    # 定义一个初始化方法， __init__ 是类的构造方法
    def __init__(self, n_neighbors=1, dist_func=l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 训练模型的方法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # 模型预测算法
    def predict(self, x):
        # 初始化预测分类数组, 初始化零数组, 行列
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        # 遍历 x 输入的数据点, 取出每个数据点的i， x_test
        for i, x_test in enumerate(x):
            # x_test 跟所有训练数据计算距离
            distances = self.dist_func(self.x_train, x_test)
            # 得到距离按照由近到远排序,取出索引值
            nn_index = np.argsort(distances)
            # 选举最近的K 个点， 保存他们的分类类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel()
            # 统计类别中出现频率最高的那个， 赋给y_pred[i]  , bincount, 统计每个值出现的次数, argmax：最大值的索引值
            y_pred[i] = np.argmax(np.bincount(nn_y))

        return y_pred


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # pandas 中的 DF是可以改变的, Spark中的DF是没法改变的
    df['class'] = iris.target
    df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
    # print(df.describe())
    x = iris.data
    y = iris.target.reshape(-1, 1)
    # print(x.shape, y.shape)
    # 划分训练集合测试集
    # random_state 随机测试的种子,stratify 保持 等比例分割, x 是完全随机的
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)

    # 2. 核心算法实现
    # x_test[0].reshape(1, -1).shape
    # print(x_test)
    # print(x_test[0].reshape(1, -1))
    # np.sum(np.abs(x_train - x_test[0].reshape(1, -1), axis = 1))、

    # print(np.zeros((1, 2)))
    # dist = np.array([2, 3, 43, 23, 23432, 234, 234, 21, 3, 53, 234, 23, 53, 353])
    # # np_index = np.sort(dist)
    # np_index = np.argsort(dist)
    # aa = arange(12).reshape(3, 4)
    # bb = aa.ravel()
    # print("aa:")
    # print(aa)
    # print("bb:", bb)

    # 测试
    # 定义一个kNN 实例
    knn = kNN(n_neighbors=3)
    # 训练模型
    knn.fit(x_train, y_train)

    # 传入测试数据，做预测
    y_pred = knn.predict(x_test)

    # 求出预测准确率
    accuracy = accuracy_score(y_test, y_pred)

    print("预测准确率：", accuracy)

    # 保存结果list
    result_list = []

    # 针对不同的参数选取，做预测
    for p in [1, 2]:
        knn.dist_func = l1_distance if p == 1 else l2_distance

        # 考虑不同 k 的取值, 步长 为 2
        for k in range(1, 10, 2):
            knn.n_neighbors = k
            # 传入测试数据，做预测
            y_pred = knn.predict(x_test)
            # 求出预测准确率
            accuracy  = accuracy_score(y_test, y_pred)
            result_list.append([k,  'l1_distance' if p == 1 else  'l2_distance', accuracy])
    df = pd.DataFrame(result_list, columns=['k', '距离函数', '预测准确率'])
    print(df)



