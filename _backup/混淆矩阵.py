import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels, label_names):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # 增大图形尺寸
    plt.figure(figsize=(10, 10))
    # 设置 annot_kws 来加大混淆矩阵中元素的字号
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={"size": 18},
                linewidths=0.1,  # 设置格子边框宽度
                square=True)  # 确保格子为正方形
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    # 设置 x 轴和 y 轴标签名的字号
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


# 示例标签名
label_names = ['bus', 'car', 'person', 'motobike', 'rect', 'square', 'truck', 'background']
labels = np.arange(8)

# 各行数据
row_0 = [629, 23, 0, 1, 0, 0, 19, 91]
row_1 = [12, 5944, 1, 2, 0, 0, 29, 810]
row_2 = [0, 0, 1174, 27, 0, 0, 0, 350]
row_3 = [0, 10, 94, 6191, 0, 0, 0, 1495]
row_4 = [0, 0, 0, 0, 1161, 1, 0, 45]
row_5 = [0, 0, 0, 0, 5, 1163, 0, 186]
row_6 = [25, 35, 0, 1, 0, 0, 971, 125]
row_7 = [45, 289, 432, 649, 30, 181, 66, 0]
y_true = []
y_pred = []


# 定义一个函数用于根据行数据填充 y_true 和 y_pred
def fill_data(y_true, y_pred, true_label, row):
    for pred_label, count in enumerate(row):
        y_true.extend([true_label] * count)
        y_pred.extend([pred_label] * count)
    return y_true, y_pred


# 填充第一行数据
y_true, y_pred = fill_data(y_true, y_pred, 0, row_0)
# 填充第二行数据
y_true, y_pred = fill_data(y_true, y_pred, 1, row_1)
# 填充第三行数据
y_true, y_pred = fill_data(y_true, y_pred, 2, row_2)
y_true, y_pred = fill_data(y_true, y_pred, 3, row_3)
# 填充第四行数据
y_true, y_pred = fill_data(y_true, y_pred, 4, row_4)
# 填充第五行数据
y_true, y_pred = fill_data(y_true, y_pred, 5, row_5)
# 填充第六行数据
y_true, y_pred = fill_data(y_true, y_pred, 6, row_6)
# 填充第七行数据
y_true, y_pred = fill_data(y_true, y_pred, 7, row_7)

plot_confusion_matrix(y_true, y_pred, labels, label_names)
