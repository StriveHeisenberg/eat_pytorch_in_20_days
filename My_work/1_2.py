# !/usr/bin/env python
# _*_coding:utf-8_*_

#%%
import os
import datetime
from abc import ABC

# 设置环境变量
os.environ["KMP_DUPLCATE_LIB_OK"] = "TRUE"


def print_bar():
    """
    输出隔断横线和时间
    :return:
    """
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=========="*8 + "%s" % now_time)


# print_bar()

#%% 数据准备和导入
# 使用torchvision datasets.ImageFolder & DataLoader 读取 & 加载图片数据
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

transform_train = transforms.Compose([transforms.ToTensor()])
transform_valid = transforms.Compose([transforms.ToTensor()])

ds_train = datasets.ImageFolder("./data/cifar2/train/",
                                transform=transform_train,
                                target_transform=lambda t: torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("D:/Projects/eat_pytorch_in_20_days/data/cifar2/test/",
                                transform=transform_valid,
                                target_transform=lambda t: torch.tensor([t]).float())
# print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=0)
dl_valid = DataLoader(ds_valid, batch_size=50, shuffle=True, num_workers=0)

#%% 查看部分样本
from matplotlib import pyplot as plt

plt.figure(figsize=(8, 8))
for i in range(9):
    img, label = ds_train[i]
    img = img.permute(1, 2, 0)
    ax = plt.subplot(3, 3, i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label.item())
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

#%% 当DataLoader(num_workers=0)时，dl_train为可迭代对象
for x, y in dl_train:
    print(x.size(), y.size())
    break

#%% 创建模型，继承nn.Module基类构建自定义模型
# pool = nn.AdaptiveAvgPool2d((1, 1))
# t = torch.randn(10, 8, 32, 32)
# pool(t).shape


class Net(nn.Module, ABC):
    """
    继承nn.Module的自定义模型
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y


net = Net()
# print(net)

#%%
import torchkeras

torchkeras.summary(net, input_shape=(3, 32, 32))

#%% 训练模型，函数形式循环
import pandas as pd
from sklearn.metrics import roc_auc_score

model = net
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = torch.nn.BCELoss()
model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
model.metric_name = "auc"


def train_step(model, features, labels):
    """

    :param model:
    :param features:
    :param labels:
    :return:
    """
    # 训练模型, dropout发生作用
    model.train()
    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)
    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    return loss.item(), metric.item()


def valid_step(model, features, labels):
    """

    :param model:
    :param features:
    :param labels:
    :return:
    """
    # 预测模型, dropout不作用
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels)
        metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()


features, labels = next(iter(dl_valid))
train_step(model, features, labels)

#%%


def train_model(model, epochs, dl_train, dl_valid, log_step_freq):
    """

    :param model:
    :param epochs:
    :param dl_train:
    :param dl_valid:
    :param log_step_freq:
    :return:
    """
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print("Start Training: ")
    print_bar()

    for epoch in range(1, epochs+1):
        # 开始循环训练
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):
            loss, metric = train_step(model, features, labels)
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum/step, metric_sum/step))
        # 验证循环
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, labels)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 记录日志
        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info
        # 输出日志
        print(("\nepoch = %d, loss = %.3f, " + metric_name + " = %.3f, val_loss = %.3f, " +
               "val_"+metric_name+" = %.3f") % info)
        print_bar()
    print("Finished Training!")
    return dfhistory


epochs = 50
dfhistory = train_model(model, epochs, dl_train, dl_valid, log_step_freq=50)

#%% 评估模型
# print(dfhistory)
import matplotlib.pyplot as plt


def plot_metric(dfhistory, metric):
    """

    :param dfhistory:
    :param metric:
    :return:
    """
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title("Training and validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, "val_"+metric])
    plt.show()


plot_metric(dfhistory, "loss")
plot_metric(dfhistory, "auc")

#%% 使用模型


def predict(model, dl):
    """

    :param model:
    :param dl:
    :return:
    """
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in dl])
    return result.data


# y_pre_probs = predict(model, dl_valid)

#%% 保存模型参数
torch.save(model.state_dict(), "./model/model_parameter1_2.pth")

#%% 使用保存模型参数
net_clone = Net()
net_clone.load_state_dict(torch.load("./model/model_parameter1_2.pth"))
predict(net_clone, dl_valid)