import torch
from supervised_learn import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from until import *


# 取数据
def single_data():

    dataset_02 = pd.read_excel("./data_file/data_02.xls")
    dataset_02.drop(elements_2, axis=1, inplace=True)
    dataset_02 = dataset_02.rename(columns={'TD TRQ': 'Torque', 'ROP m/hr': 'ROP', 'FLWpmps': 'Pump flow'})
    # 将ROP m/hr这一列挪到最后一列
    cols_02 = list(dataset_02)
    cols_02.insert(12, cols_02.pop(cols_02.index('ROP')))
    cols_02.insert(5, cols_02.pop(cols_02.index('Pump flow')))
    # 重组df对象排列顺序
    dataset_02.drop(dataset_02.tail(1).index, inplace=True)
    dataset_02 = dataset_02.loc[0:, cols_02]
    dataset_05 = pd.read_excel("./data_file/data_05.xls")
    dataset_05.drop(elements_5, axis=1,
                    inplace=True)  # 删除“TVD”、“ROPA”列
    dataset_05 = dataset_05.rename(columns={'ROP mhr': 'ROP'})
    # 将ROP m/hr这一列挪到最后一列
    cols_05 = list(dataset_05)
    # cols_05.insert(12, cols_05.pop(cols_05.index('ROP')))
    # 重组df对象排列顺序
    dataset_05 = dataset_05.loc[0:, cols_05]

    dataset_06 = pd.read_excel("./data_file/data_06.xlsx")
    dataset_06.drop(
        elements_6,
        axis=1, inplace=True)  # 删除“TVD”、“ROPA”列
    dataset_06 = dataset_06.rename(columns={'ROP mhr': 'ROP'})
    cols_06 = list(dataset_06)
    # cols_06.insert(16, cols_06.pop(cols_06.index('ROP')))
    dataset_06 = dataset_06.loc[0:, cols_06]

    dataset_07 = pd.read_excel("./data_file/data_07.xlsx")
    dataset_07.drop(elements_7, axis=1, inplace=True)  # 删除“TVD”、“ROPA”列
    dataset_07['Torque'] = dataset_07['Torque'].map(lambda x: x * 1000)
    dataset_07 = dataset_07.rename(columns={'ROP mhr': 'ROP'})
    cols_07 = list(dataset_07)
    # cols_07.insert(16, cols_07.pop(cols_07.index('ROP')))
    dataset_07 = dataset_07.loc[0:, cols_07]
    return dataset_02, dataset_05, dataset_06, dataset_07


# 做归一化
def data_Normalization():
    dataset_02, dataset_05, dataset_06, dataset_07 = single_data()
    # 训练集合并
    frames = [dataset_05, dataset_06]
    dataset_train = pd.concat(frames)

    # 测试集
    dataset_test = dataset_07

    # 转化有监督的学习问题
    dataset_train_af_sl = fun_01(dataset_train)
    dataset_test_af_sl = fun_02(dataset_test, dataset_train)

    # 滑动窗口
    train_x, train_y = data_zip(dataset_train_af_sl.values, seq_len, pre_len)
    train_x = np.delete(train_x, len(train_x)-1, axis=0)
    train_y = np.delete(train_y, 0, axis=0)
    train_x, train_y = torch.from_numpy(train_x).type(torch.float32), torch.from_numpy(train_y).type(torch.float32)
    train_y = train_y.squeeze()
    test_x, test_y = data_zip(dataset_test_af_sl.values, seq_len, pre_len)
    test_x, test_y = torch.from_numpy(test_x).type(torch.float32), torch.from_numpy(test_y).type(torch.float32)
    test_y = test_y.squeeze()
    features_num = train_x.shape[-1]
    out_size = 1

    data_train = TensorDataset(train_x, train_y)
    dataloader_train = DataLoader(data_train, batch_size=train_batch, shuffle=True)
    data_test = TensorDataset(test_x, test_y)
    dataloader_test = DataLoader(data_test, batch_size=test_batch, shuffle=False)
    return dataloader_train, dataloader_test, features_num, out_size


