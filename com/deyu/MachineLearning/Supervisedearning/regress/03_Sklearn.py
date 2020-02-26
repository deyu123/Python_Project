import numpy as np
import matplotlib.pyplot as plt
#  1.导入数据 （data.csv)
points = np.genfromtxt("data.csv", delimiter=',')
x = points[:, 0]
y = points[:, 1]
# print(points)
# 用plt 来画散点图
plt.scatter(x, y)
plt.cbook
plt.show()


#  2.定义损失函数
# 定义计算损失函数,损失函数是系数的函数，还要传入数据的x,y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)
    # 驻点计算平方损失误差，然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2
    return total_cost / M


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    # -1 行数不限， 1，最后变成 1列
    x_new = x.reshape(-1, 1)
    y_new = y.reshape(-1, 1)
    lr.fit(x_new, y_new)
    # _ 下划线内置的意思
    # 从训练好的模型中提取系数和截距
    w = lr.coef_[0][0]
    b = lr.intercept_[0]
    print("w is:", w)
    print("b is:", b)

    cost = compute_cost(w, b, points)

    print("cost is:", cost)
    # 5.画出拟合曲线
    plt.scatter(x, y)
    # 针对每一个x, 预测 y 的值
    pred_y = w * x + b
    plt.plot(x, pred_y, c='r')
    plt.show()

