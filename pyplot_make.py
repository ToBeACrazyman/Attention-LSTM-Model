import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from until import *

# fig = plt.figure()
# fig.set_size_inches(10, 4)  # 整个绘图区域的宽度10和高度4

# ax = fig.add_subplot(1, 2, 1)
font = {
        'weight': '400',
        'size': 15,
        }


def rop_prediction_plt(true, pre, epoch, path):
    regressor = LinearRegression()
    regressor = regressor.fit(np.reshape(true, (-1, 1)), np.reshape(pre, (-1, 1)))
    print(regressor.coef_, regressor.intercept_)  # 打印拟合结果(参数)
    # 画出数据和拟合直线的图
    plt.scatter(true, pre)
    plt.plot(np.reshape(true, (-1, 1)), regressor.predict(np.reshape(true, (-1, 1))), 'r')
    plt.xlabel("actual value")
    plt.ylabel("predictive value")
    plt.title("Fitting results")
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()


def distance_chart_plt(train_acc_size, test_acc_size, path):

    unit = ["Relative_error", "Distance_length"]
    plt.figure(figsize=(24, 8))
    plt.plot(train_acc_size)
    plt.plot(test_acc_size)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.legend(["train_ACC", "test_ACC"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    # plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, 'acc.png'))
    plt.close()


def line_chart_plt(true, pre, md, epoch, path):
    true = true.flatten()
    pre = pre.flatten()
    md = md.flatten()
    unit = ["ROP(m/hr)", "Depth(m)"]
    plt.figure(figsize=(24, 8))
    plt.plot(md, true)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.plot(md, pre)
    plt.legend(["real", "pre"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()


def rel_error_plt(md, r, epoch, path):
    plt.scatter(r, md)
    # plt.plot(np.reshape(x, (-1, 1)), regressor.predict(np.reshape(x, (-1, 1))), 'r')
    plt.xlabel("relative_error")
    plt.ylabel("md")
    plt.title("Fitting results")
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()
