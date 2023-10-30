import pickle
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from Dataloader_Pytorch import *
from torch.utils.data import DataLoader
from LSTM_architectures import LSTMModel, LSTM_multitask

# RMSE loss
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# MAE loss
class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, yhat, y):
        return torch.mean(torch.abs(yhat - y))

# MAPE loss
class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, yhat, y):
        return torch.mean(torch.abs((y - yhat) / y)) * 100

# R2 loss
class R2Loss(torch.nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, yhat, y):
        ybar = torch.mean(y)
        ssreg = torch.sum((yhat - ybar) ** 2)
        sstot = torch.sum((y - ybar) ** 2)
        return 1 - ssreg / sstot

def single_exponential_smoothing(data, alpha):
    # create an empty list to store the smoothed values
    smoothed = []
    # set the first smoothed value equal to the first data point
    smoothed.append(data[0])
    # loop through the remaining data points
    for i in range(1, len(data)):
        # calculate the smoothed value using the previous smoothed value and the current data point
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[-1]
        # add the smoothed value to the list of smoothed values
        smoothed.append(smoothed_value)
    # convert the list of smoothed values to a pandas Series
    # smoothed = pd.Series(smoothed)

    return smoothed
def holt_smoothing(data, alpha):
    smoothed = [data[0]] # 首项直接等于原数据第一项
    level = data[0]
    trend = data[1] - data[0]
    for i in range(1, len(data)):
        last_level, level = level, alpha*data[i] + (1-alpha)*(level+trend)
        trend = alpha*(level-last_level) + (1-alpha)*trend
        smoothed.append(level+trend)
    return smoothed

# Magic numbers
epochs = 30
batch_size = 32
hidden_size = 128
layers = 4
learning_rate = 1e-5
window_size = 300
loss_weight = [0.5, 0.5]  # regression, classification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
# 加载数据集
# './SingleDayDataAfterNormalization/04-26_Time AM.csv'
#csv_name = '/home/minziran/Documents/GitHub/LSTM/DataPreProcessing/SrcIP/0-10.200.7.218.csv'

dropped_features = ['Source.IP', 'Destination.IP', 'L7Protocol', 'Protocol', 'ProtocolName', 'Source.Port', 'Destination.Port']
csv_name = '/home/minziran/Documents/GitHub/LSTM/DataPreProcessing/DataAfterCleaning.csv'
data = pd.read_csv(csv_name)
print(data.info())
data.drop(columns=dropped_features, inplace=True)
date_format = '%d/%m/%Y%H:%M:%S'
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format=date_format)

data = data.sort_values(by=['Timestamp'])
print(len(data.columns))
# plt.plot(data['Timestamp'][:1000], data['Flow.Packets.s'][:1000])
# plt.show()
# data = data[["DateTime", "Traffic"]]
data.set_index('Timestamp', inplace=True)
feature_len = len(data.columns)



scaler = MinMaxScaler(feature_range=(0.1, 0.9))
new_data = scaler.fit_transform(data)
# data.set_index('DateTime', inplace=True)
data_scaled = pd.DataFrame(new_data, columns=data.columns, index=data.index)
# data_scaled['Traffic'] = single_exponential_smoothing(data_scaled['Traffic'], 0.5)

# data.set_index('DateTime', inplace=True)
# debug: data_scaled = data_scaled.head(10)
print("data_scaled", data_scaled.head())

for key in data_scaled.columns.tolist():
    data_scaled[key] = holt_smoothing(data_scaled[key], 0.3)
    # data_scaled[key] = single_exponential_smoothing(data_scaled[key], 0.5)

# plt.figure()
# data['Traffic'].plot.box()
# plt.show()

# if data_scaled.isna().values.any():
#     print("DataFrame contains NaN")
# else:
#     print("DataFrame does not contain NaN")
dataset = SequenceDataset_from_dataframe(data=data_scaled, window_size=window_size)
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)  # 划分训练集和测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loss_list = []
val_loss_list = []
train_rmse_list = []
val_rmse_list = []
train_mae_list = []
val_mae_list = []
train_mape_list = []
val_mape_list = []
train_r2_list = []
val_r2_list = []








# 训练 LSTM 模型
model = LSTMModel(input_features=feature_len, hidden_features=hidden_size, output_features=feature_len, num_layers=layers)
#model = LSTMModel(input_features=1, hidden_features=64, output_features=1, num_layers=1)
model.to(device)

mse_loss = torch.nn.MSELoss()
rmse_loss = RMSELoss()
mae_loss = MAELoss()
mape_loss = MAPELoss()
r2_loss = R2Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    # Training
    model.train()
    running_loss, running_rmse, running_mae, running_mape, running_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
    for whole_input, target in train_loader:
        batch_size = target.shape[0]
        whole_input, target, = whole_input.to(device), target.to(device)
        output = model(whole_input)
        loss = mse_loss(output, target)
        # loss = criterion(output, target)
        rmse = rmse_loss(output, target)
        mae = mae_loss(output, target)
        mape = mape_loss(output, target)
        r2 = r2_loss(output, target)

        optimizer.zero_grad()
        mae.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_rmse += rmse.item() * batch_size
        running_mae += mae.item() * batch_size
        running_mape += mape.item() * batch_size
        running_r2 += r2.item() * batch_size

    epoch_loss_train = running_loss / len(train_dataset)
    epoch_rmse_train = running_rmse / len(train_dataset)
    epoch_mae_train = running_mae / len(train_dataset)
    epoch_mape_train = running_mape / len(train_dataset)
    epoch_r2_train = running_r2 / len(train_dataset)

    train_loss_list.append(epoch_loss_train)
    train_rmse_list.append(epoch_rmse_train)
    train_mae_list.append(epoch_mae_train)
    train_mape_list.append(epoch_mape_train)
    train_r2_list.append(epoch_r2_train)

    print('Epoch [%d/%d], Total train Loss: %.4f, rmse: %.4f, mae: %.4f, mape: %.4f, r2: %.4f' % (epoch + 1, epochs,
                                                                                                  epoch_loss_train,
                                                                                                  epoch_rmse_train,
                                                                                                  epoch_mae_train,
                                                                                                  epoch_mape_train,
                                                                                                  epoch_r2_train))

    model.eval()
    running_loss, running_rmse, running_mae, running_mape, running_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for whole_input, target in val_loader:
            batch_size = target.shape[0]
            whole_input, target = whole_input.to(device), target.to(device)
            output = model(whole_input)
            loss = mse_loss(output, target)
            rmse = rmse_loss(output, target)
            mae = mae_loss(output, target)
            mape = mape_loss(output, target)
            r2 = r2_loss(output, target)
            # r2 = r2_score(target, output)

            running_loss += loss.item() * batch_size
            running_rmse += rmse.item() * batch_size
            running_mae += mae.item() * batch_size
            running_mape += mape.item() * batch_size
            running_r2 += r2.item() * batch_size

    epoch_loss_val = running_loss / len(val_dataset)
    epoch_rmse_val = running_rmse / len(val_dataset)
    epoch_mae_val = running_mae / len(val_dataset)
    epoch_mape_val = running_mape / len(val_dataset)
    epoch_r2_val = running_r2 / len(val_dataset)

    val_loss_list.append(epoch_loss_val)
    val_rmse_list.append(epoch_rmse_val)
    val_mae_list.append(epoch_mae_val)
    val_mape_list.append(epoch_mape_val)
    val_r2_list.append(epoch_r2_val)
    print('Epoch [%d/%d], Total val Loss: %.4f, rmse: %.4f, mae: %.4f, mape: %.4f, r2: %.4f' % (
        epoch + 1, epochs, epoch_loss_val, epoch_rmse_val, epoch_mae_val, epoch_mape_val, epoch_r2_val))

torch.save(model, "./HoltLSTMBig.pth")
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 12))

# 在第一个子图中画出arr1
axes[0].plot(train_loss_list, label='mse train')
axes[0].plot(val_loss_list, label='mse val')

# 在第二个子图中画出arr2
axes[1].plot(train_rmse_list, label='rmse train')
axes[1].plot(val_rmse_list, label='rmse val')
# 在第三个子图中画出arr3
axes[2].plot(train_mae_list, label='mae train')
axes[2].plot(val_mae_list, label='mae val')

# 在第四个子图中画出arr4
axes[3].plot(train_mape_list, label='mape train')
axes[3].plot(val_mape_list, label='mape val')

# 在第五个子图中画出arr5
axes[4].plot(train_r2_list, label='r2 train')
axes[4].plot(val_r2_list, label='r2 val')

# 设置每个子图的标题
axes[0].set_title('MSE Loss')
axes[1].set_title('RMSE Loss')
axes[2].set_title('MAE Loss')
axes[3].set_title('MAPE Loss')
axes[4].set_title('R^2 Loss')

# 调整子图之间的间距
fig.tight_layout()

# 显示图形

plt.legend()
# show the plot
plt.show()
print("Train Loss List", train_loss_list)
print("Var Loss List", val_loss_list)


# # 测试 LSTM 模型
# with torch.no_grad():
#     test_outputs = model(test_X)
#     test_loss = criterion(test_outputs, test_y)
#     print('Test Loss: %.4f' % test_loss.item())

# # 输出 LSTM 模型的预测结果
# predict_X = data_scaled.iloc[-window_size:].values.reshape((1, window_size))
# predict_y = model(torch.from_numpy(predict_X).float())
# predict_y = scaler.inverse_transform(predict_y.numpy())
# print('Network flow prediction for next time step: %.2f' % predict_y[0, 0])
#
# Epoch [30/30], Total train Loss: 0.0018, rmse: 0.0410, mae: 0.0256, mape: 15.2798, r2: 0.5823
# Epoch [30/30], Total val Loss: 0.0019, rmse: 0.0416, mae: 0.0256, mape: 15.0061, r2: 0.5743
# SES + LSTM
# Epoch [30/30], Total train Loss: 0.0005, rmse: 0.0205, mae: 0.0128, mape: 7.7988, r2: 0.2588
# Epoch [30/30], Total val Loss: 0.0005, rmse: 0.0205, mae: 0.0128, mape: 7.7515, r2: 0.2795
# # SES + LSTM big dataset
# Epoch [30/30], Total train Loss: 0.0041, rmse: 0.0638, mae: 0.0172, mape: 5.8562, r2: 0.1644
# Epoch [30/30], Total val Loss: 0.0041, rmse: 0.0639, mae: 0.0172, mape: 5.8188, r2: 0.1742
#LSTM Big
# Epoch [8/30], Total train Loss: 0.0169, rmse: 0.1298, mae: 0.0337, mape: 13.1749, r2: 0.4183
# Epoch [8/30], Total val Loss: 0.0170, rmse: 0.1300, mae: 0.0337, mape: 12.8686, r2: 0.4304
# Holt +LSTM
# Epoch [14/30], Total train Loss: 0.0025, rmse: 0.0497, mae: 0.0146, mape: 8.1327, r2: 0.1277
# Epoch [14/30], Total val Loss: 0.0025, rmse: 0.0497, mae: 0.0145, mape: 8.1537, r2: 0.1267