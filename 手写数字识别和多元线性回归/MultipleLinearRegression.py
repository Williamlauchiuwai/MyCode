import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn import metrics


mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def predict():
    data = datasets.load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = pd.DataFrame(data.target, columns=['MEDV'])
    X = df
    y = target


    # 去除MEDV=50的异常值
    drop_index = y[y['MEDV'] == 50].index.values
    X = X.drop(drop_index)
    y = y.drop(drop_index)
    # 这里划分训练集和测试集的参数random_state都是1，表示随机分配的数据是同一组
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.coef_)
    print(lr.intercept_)

    y_pred = lr.predict(X_test)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print('MSE:', MSE)
    print('RMSE:', RMSE)

    plt.figure(figsize=(15, 5))
    plt.plot(range(len(y_test)), y_test.values, 'r', label='测试数据')
    plt.plot(range(len(y_test)), y_pred, 'b', label='预测数据')
    plt.title('测试结果拟合直线图')
    plt.legend()
    plt.show()

    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('测试结果拟合散点图')
    plt.show()

    # 自变量对因变量的影响
    plt.plot(data.feature_names, lr.coef_.flatten())
    plt.xlabel('自变量名')
    plt.ylabel('影响大小')
    plt.title('自变量对因变量影响因素图')
    plt.show()






if __name__ =='__main__':
    predict()