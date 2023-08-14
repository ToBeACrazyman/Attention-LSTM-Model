import numpy as np
import pandas as pd


# 参数
elements_2 = ['ROP', 'Pump 1', 'Pump 2', 'Pump 3', 'BitRun', 'BitTime', 'Return', 'T GAS main']
elements_5 = [ 'FLOWOUT', 'ROP', 'Pump 1', 'Pump 2', 'Pump 3', 'ROP_log m/hr', 'BitTime', 'Total RPM']
elements_6 = [ 'Vertical Depth', 'ROP', 'FlowOut', 'HKH', 'Pump 1', 'Pump 2', 'Pump 3', 'BitTime', 'BitRun', 'DH RPM', 'Total RPM']
elements_7 = ['Pump 1', 'Pump 2', 'Pump 3', 'HKH', 'FlowOut', 'MW out', 'DEXPONENT', 'DH RPM', 'Total RPM',
                     'ROP', 'BitTime', 'BitRun']

attn_elements = ['ROP (t-1)', 'Depth', 'WOH', 'WOB', 'RPM', 'Torque', 'Pump flow', 'SPP', 'PumpTime', 'Pump_total','ROP']
data_02_elements = ['Depth', 'WOH', 'WOB', 'RPM', 'Torque', 'FLWpmps', 'SPP', 'PumpTime', 'Pump_total', 'ROP']

# 模型路径
model_path = './file/model_all_data.pkl'

# 超参数
max_min_distance = 500
train_batch = 16
test_batch = 16
pre_len = 1
seq_len = 1
interval = 365
tf_lr = 0.0002
times = 100

# 检查网络，定义网络的参数hidden_size和num_layers，并且查看网络结构，测试网络是否跑的通
# 定义网络的参数hidden_size和num_layers
hidden_size = 256
num_layers = 3


# 做标准化归一化
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData


# 获取最用保存数据路径
def get_train_result_path(epoch):
    path = "./pre_data/" + str(epoch) + ".csv"
    return path
