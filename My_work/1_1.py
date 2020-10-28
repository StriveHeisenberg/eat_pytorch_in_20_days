# !/usr/bin/env python
# _*_coding:utf-8_*_


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

#%%
dftrain_raw = pd.read_csv('./data/train.csv')
dftest_raw = pd.read_csv('./data/test.csv')
# print(dftrain_raw.head(10))

#%%
def preprocessing(dfdata):
    '''数据预处理'''
    dfresult = pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)


x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw[['Survived']].values

x_test = preprocessing(dftest_raw).values
y_test = dftest_raw[['Survived']].values

# print("x_train.shape =", x_train.shape )
# print("x_test.shape =", x_test.shape )
#
# print("y_train.shape =", y_train.shape )
# print("y_test.shape =", y_test.shape )

dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).float()),
                     shuffle = True, batch_size = 8)
dl_valid = DataLoader(TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).float()),
                     shuffle = False, batch_size = 8)

#%%
def create_net():
    '''使用Sequential按层顺序定义模型'''
    net = nn.Sequential()
    net.add_module("linear1", nn.Linear(15, 20))
    net.add_module("relu1", nn.ReLU())
    net.add_module("linear2", nn.Linear(20, 15))
    net.add_module("relu2", nn.ReLU())
    net.add_module("linear3", nn.Linear(15, 1))
    net.add_module("sigmoid", nn.Sigmoid())
    return net

net = create_net()
print(net)


#%%
# from torchkeras import summary
# summary(net, input_shape=(15,))
from torchsummary import summary
summary(net, input_size=(15,))


#%% 训练模型
from sklearn.metrics import accuracy_score

loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
metric_name = "accuracy"

#%% 脚本循环
import datetime

epochs = 10
log_step_freq = 30

dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("=======" * 8 + "%s"%nowtime)

for epoch in range(1, epochs+1):
    ''' '''
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(dl_train, 1):
        # 梯度清零
        optimizer.zero_grad()
        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions, labels)
        metric = metric_func(predictions, labels)
        # 反向传播求梯度
        loss.backward()
        optimizer.step()
        # print日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") % \
                  (step, loss_sum/step, metric_sum/step))
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features, labels) in enumerate(dl_valid, 1):
        # 停止梯度计算
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions, labels)
            val_metric = metric_func(predictions, labels)
        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 记录日志并打印
    info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
    dfhistory.loc[epoch-1] = info
    print(("\nEPOCH = %d, loss = %.3f, " + metric_name + \
           " = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f") % info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "========" * 8 + "%s" % nowtime)

print("Finished Training.")

#%% 评估模型
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    ''' '''
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend("train_" + metric, "val_" + metric)
    plt.show()

plot_metric(dfhistory, "loss")
plot_metric(dfhistory, "accuracy")

#%% 使用模型
y_pred_probs = net(torch.tensor(x_test[0:10]).float()).data

y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), \
                     torch.zeros_like(y_pred_probs))

#%% 保存模型参数
torch.save(net.state_dict(), "./data/net_parameter.pkl")
net_clone = create_net()
net_clone.load_state_dict(torch.load("./data/net_parameter.pkl"))
net_clone.forward(torch.tensor(x_test[0:10]).float()).data

#%% 保存完整模型
torch.save(net, "./data/net_parameter.pkl")
net.loaded = torch.load("./data/net_parameter.pkl")
net.loaded(torch.tensor(x_test[0:10]).float()).data