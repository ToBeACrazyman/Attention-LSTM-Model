import numpy as np
import pandas as pd
import datetime
from lstm_model import *
from train_data import *
from pyplot_make import *
from sklearn.metrics import r2_score

USE_MULTI_GPU = True

# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]
else:
    MULTI_GPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader_train, dataloader_test, features_num, out_size = data_Normalization()
# net = nn.Sequential()  # Request a sequential container 申请一个 顺序容器
# net.add_module('0', var_attention_layer(features_num))  # Add a variable attention layer 添加一个变量注意力层
# net.add_module('1', LSTM_bi(features_num, 128, 1))
# # You can add more Grus outside to achieve the effect of decreasing sampling layer by layer 可以在外部添加更多的gru，达到逐层递减的采样效果
# net.add_module('2', LSTM_bi_toLinear(128, 64, 1))
# # This is necessary. It can complete the connection with the thread layer, and its internal is also a Gru  这个是必须的，这个能够完成与线程层的相连，其内部也是一个gru
# net.add_module('3', nn.Linear(64, 32))
# net.add_module('4', nn.ReLU())
# net.add_module('5', nn.Linear(32, 5))
net = lstm_bi_attention(features_num, hidden_size, int(hidden_size / 2), num_layers=num_layers, pre_length=pre_len)
criterion = nn.MSELoss().to(device)
# optimizer = torch.optim.SGD(net.parameters(), lr=tf_lr, momentum=0.99)
optimizer = torch.optim.Adam(net.parameters(), lr=tf_lr, weight_decay=0.001)  # weight_decay=0.001 范数，提高泛化能力


def train(TModel, loader):
    TModel = TModel.train()
    epoch_loss = 0
    y_pre = []
    y_true = []
    # torch.backends.cudnn.enabled = False
    for X, y in loader:  # X--[batch,seq,feature_size]  y--[batch,seq]
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output, wei = TModel(X)
        output = output[:, -pre_len]
        y = y[:]
        loss = criterion(output, y)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(TModel.parameters(), 0.10)
        optimizer.step()
        epoch_loss += loss.item()
        pres = output.detach().cpu().numpy()  # [seq,batch,out_size]
        pres = pres.reshape(-1, 1)  # [none,out_size]
        tru = y.detach().cpu().numpy()
        tru = tru.reshape(-1, 1)
        y_pre.append(pres)
        y_true.append(tru)

    pre = np.concatenate(y_pre, axis=0)
    true = np.concatenate(y_true, axis=0)
    # 算准确率
    # acc = r2_score(true, pre)
    acc = 1 - (np.abs(pre[:, 0] - true[:, 0]) / true[:, 0]).mean()
    return acc, epoch_loss


# 测试函数
def test(TModel, tf_loader):
    epoch_loss = 0
    y_pre = []
    y_true = []
    wei_size = []

    for x, y in tf_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            output, wei = TModel(x)
            output = output[:, -pre_len]
            y = y[:]
            pres = output.detach().cpu().numpy()  # [seq,batch,out_size]
            tru = y.detach().cpu().numpy()
            mse_loss = ((pres - tru) ** 2).mean()
            epoch_loss += mse_loss.item()
            wei_size.append(wei.detach().cpu().numpy())
            y_pre.append(pres.reshape(-1, 1))
            y_true.append(tru.reshape(-1, 1))

    # t = pd.DataFrame(np.concatenate(true_csv, axis=0), columns=['真实值日产液', '真实值日产油'])
    # p = pd.DataFrame(np.concatenate(pre_csv, axis=0), columns=['预测值日产液', '预测值日产油'])
    # csv_t = pd.concat([t, p], axis=1)
    # csv_t.to_csv('./test/data_33.csv', sep=",", index=True)
    pre = np.concatenate(y_pre, axis=0)
    true = np.concatenate(y_true, axis=0)
    pre_df = pd.DataFrame(pre[:, 0], columns=['预测ROP'])
    true_df = pd.DataFrame(true[:, 0], columns=['真实ROP'])
    tt = pd.concat([pre_df, true_df], axis=1)

    acc = 1 - (np.abs(pre[:, 0] - true[:, 0]) / true[:, 0]).mean()

    wei = pd.DataFrame(wei_size)
    wei = np.mean(wei.values, axis=0)
    # acc = r2_score(true, pre)
    return acc, epoch_loss, wei, tt


def initiate():
    train_acc_size = []
    test_acc_size = []
    current_datetime = datetime.datetime.now()
    print("time = ", datetime.datetime.now())
    attn_size_test = []
    attn_size_train = []
    for epoch in range(401):
        net.train()
        net.to(device)
        train_acc, train_loss = train(net, dataloader_train)

        # print('Epoch:', '%04d' % epoch, 'loss =', '{:.6f}'.format(train_loss), ' acc =',
        #       '{:.6f}'.format(train_acc))

        net.eval()
        train_acc, train_loss, train_wei, train_tt = test(net, dataloader_train)
        train_acc_size.append(train_acc)
        attn_size_train.append(train_wei)
        test_acc, test_loss, attn_test, test_tt = test(net, dataloader_test)
        attn_size_test.append(attn_test)
        test_acc_size.append(test_acc)
        print('Epoch:', '%04d' % epoch, 'loss =', '{:.6f}'.format(train_loss), ' acc =',
              '{:.6f}'.format(train_acc))
        print('TEST:', 'loss =', '{:.6f}'.format(test_loss), ' acc =',
              '{:.6f}'.format(test_acc))
        print("time = ", datetime.datetime.now())
        if epoch % 20 == 0:
            distance_chart_plt(train_acc_size, test_acc_size, './')
        if epoch % 50 == 0:
            wei_att = pd.DataFrame(attn_size_test, columns=attn_elements)
            wei_att.to_csv('./parameters/test/test_atten.csv', sep=",", index=True)
            test_tt.to_csv('./parameters/test/test_pre.csv', sep=",", index=True)
            train_tt.to_csv('./parameters/train/train_pre.csv', sep=",", index=True)
            att_train = pd.DataFrame(attn_size_train, columns=attn_elements)
            att_train.to_csv('./parameters/train/train_attn.csv', sep=",", index=True)


initiate()
