import pandas as pd
import numpy as np

# 训练集
def fun_01(dataset):
    # 转换有监督的学习问题函数实现
    def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = [], []
        # i: n_in, n_in-1, ..., 1
        # 代表t-n_in, ... ,t-1
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [(dataset.columns[j] + '(t-%d)' % i) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [(dataset.columns[j] + '%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [(dataset.columns[j] + '(t+%d)' % i) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if drop_nan:
            agg.dropna(inplace=True)
        return agg

    # 确保所有的数据是float
    dataset = dataset.astype('float32')

    # MinMaxScaler归一化特征
    # scaler_test = MinMaxScaler()
    # scaler_test = scaler_test.fit_transform(dataset_test)

    # dataset = dataset / (dataset.max())

    # 转换有监督的学习问题
    reframed = series_to_supervised(dataset, 1, 1)  # 每次用过去的1列值预测未来的1列值

    # 删除不想预测的列
    # reframed.drop(reframed.columns[[0,1,2,3,4,5, 7, 8,9,10,11,12,14,15,16,17,18,19,21,22,23,24,25,26,27,
    # 28,29,30,31,32,33,34,35,36,37,38,39,40,6,13,20]], axis=1, inplace=True)
    reframed.drop(reframed.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True)

    return reframed


# 测试集
def fun_02(dataset, dataset_train):
    # 转换有监督的学习问题函数实现
    def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = [], []
        # i: n_in, n_in-1, ..., 1
        # 代表t-n_in, ... ,t-1
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [(dataset.columns[j] + '(t-%d)' % i) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [(dataset.columns[j] + '%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [(dataset.columns[j] + '(t+%d)' % i) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if drop_nan:
            agg.dropna(inplace=True)
        return agg

    # 确保所有的数据是float
    dataset = dataset.astype('float32')

    # MinMaxScaler归一化特征
    # scaler_test = MinMaxScaler()
    # scaler_test = scaler_test.fit_transform(dataset_test)

    # dataset = dataset / (dataset_train.max())

    # 转换有监督的学习问题
    reframed = series_to_supervised(dataset, 1, 1)  # 每次用过去的1列值预测未来的1列值

    # 删除不想预测的列
    # reframed.drop(reframed.columns[[0,1,2,3,4,5, 7, 8,9,10,11,12,14,15,16,17,18,19,21,22,23,24,25,26,27,
    # 28,29,30,31,32,33,34,35,36,37,38,39,40,6,13,20]], axis=1, inplace=True)
    reframed.drop(reframed.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8]], axis=1, inplace=True)
    return reframed


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-n_steps:end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def data_zip(train_var, seq_x_len, seq_y_len):
    train_seq_nums_x = train_var.shape[0] - seq_y_len
    index = 0
    for i in range(0, train_seq_nums_x, seq_y_len):
        index = index + 1
    index = index-seq_x_len-1
    train_x = np.zeros((index, seq_x_len + seq_y_len, train_var.shape[1]))
    temp = 0
    for i in range(0, train_seq_nums_x, seq_y_len):
        if i + seq_x_len + seq_y_len >= train_seq_nums_x:
            break
        elif temp >= index:
            break
        else:
            train_x[temp] = train_var[i:i + seq_x_len + seq_y_len, :]
        temp = temp + 1
    # 制作标签集
    train_y = np.zeros((index, seq_y_len, 1))

    for i in range(train_x.shape[0]):

        train_y[i] = train_x[i, seq_x_len:, -1:]


    # train_x_new = np.zeros((index, seq_x_len, train_var.shape[1]))
    # item = 0
    #
    # for i in range(train_x.shape[0]):
    #     train_x_new[item] = train_x[i, 0:seq_x_len, :]
    #     # print(train_x_new[item])
    #     # print(train_x[i, 0:seq_x_len, :])
    #     item = item + 1

    return train_x, train_y
