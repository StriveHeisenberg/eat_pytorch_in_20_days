# !/usr/bin/env python
# _*_coding:utf-8_*_

import os
import datetime
import importlib
import torchkeras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 时间戳


def print_bar():
    """

    :return:
    """
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % now_time)


#%% 载入数据并可视化
df = pd.read_csv("./data/covid-19.csv", sep='\t')
df.plot(x='date', y=['confirmed_num', 'cured_num', 'dead_num'])
# plt.xticks(rotation=60)
dfdata = df.set_index('date')
dfdiff = dfdata.diff(periods=1).dropna()
dfdiff = dfdiff.reset_index('date')
dfdiff.plot(x='date', y=['confirmed_num', 'cured_num', 'dead_num'], figsize=(10, 6))
# plt.xticks(rotation=60)
dfdiff = dfdiff.drop('date', axis=1).astype('float32')
plt.show()

#%% 数据预处理:继承torch.utils.data.Dataset实现自定义时间序列数据集
# torch.utils.data.Dataset是一个抽象类，用户想要加载自定义数据集则只需继承该类，并覆写其中两个方法
# __len__和__getitem__，必须覆写，否则返回错误
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

WINDOW_SIZE = 8


class Covid19Dataset(Dataset):
    """
    继承Dataset, 覆写__len__和__getitem__
    """
    def __init__(self):
        super(Covid19Dataset, self).__init__()

    def __len__(self):
        return len(dfdiff) - WINDOW_SIZE

    def __getitem__(self, item):
        x = dfdiff.iloc[[item, item+WINDOW_SIZE - 1], :]
        feature = torch.tensor(x.values)
        y = dfdiff.iloc[[item+WINDOW_SIZE], :]
        label = torch.tensor(y.values)
        return (feature, label)


ds_train = Covid19Dataset()
dl_train = DataLoader(ds_train, batch_size=38)

#%% 定义模型
torch.random.seed()


class Block(nn.Module):
    """
    继承nn.Module基类构建自定义模型
    """
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, x, x_input):
        x_out = torch.max((1+x)*x_input[:, -1, :], torch.tensor(0.0))
        return x_out


class Net(nn.Module):
    """
    继承nn.Module基类
    """
    def __init__(self):
        super(Net, self).__init__()
        # 3 layers LSTM
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=5, batch_first=True)
        self.linear = nn.Linear(3, 3)
        self.block = Block()

    def forward(self, x_input):
        x = self.lstm(x_input)[0][:, -1, :]
        x = self.linear(x)
        y = self.block(x, x_input)
        return y


net = Net()
model = torchkeras.Model(net)
model.summary(input_shape=(8, 3), input_dtype=torch.FloatTensor)
# print(model)

#%% 训练模型，仿照Keras定义了一个高阶的模型接口Model,
# 实现 fit, validate，predict, summary 方法，相当于用户自定义高阶API


def mspe(y_pred, y_true):
    """
    正确率
    :param y_pre:
    :param y_true:
    :return:
    """
    err_percent = (y_true - y_pred)**2 / (torch.max(y_true**2, torch.tensor(1e-7)))
    return torch.mean(err_percent)


model.compile(loss_func=mspe, optimizer=torch.optim.Adagrad(model.parameters(), lr=0.1))
dfhistory = model.fit(100, dl_train, log_step_freq=10)

#%% 评估模型，仅可视化损失函数在训练集上的迭代情况


def plot_metric(dfhistory, metric):
    """

    :param dfhistory:
    :param metric:
    :return:
    """
    train_metrics = dfhistory[metric]
    epochs = range(1, len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(['train_' + metric])
    plt.show()


plot_metric(dfhistory, 'loss')

#%% 使用模型
dfresult = dfdiff[['confirmed_num', 'cured_num', 'dead_num']].copy()
dfresult.tail()

for i in range(200):
    arr_input = torch.unsqueeze(torch.from_numpy(dfresult.values[-38:, :]), axis=0)
    arr_predict = model.forward(arr_input)
    dfpredict = pd.DataFrame(torch.floor(arr_predict).data.numpy(), columns=dfresult.columns)
    dfresult = dfresult.append(dfpredict, ignore_index=True)

dfresult.query('confirmed_num==0').head()
dfresult.query('cured_num==0').head()
dfresult.query('dead_num==0').head()

#%% 保存模型
torch.save(model.net.state_dict(), "./model/model_parameter1_4.pth")
