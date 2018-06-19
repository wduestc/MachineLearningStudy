# coding:utf-8

import numpy as np

# 使用logistic回归在简单数据集上的分类
def load_data_set():
  """
  加载数据集
  :return:返回两个数组 普通数组
    data_arr - 原始数据的特征
    label_arr - 原始数据的标签,也就是每条样本对应的类别
  """
  data_arr = []
  label_arr = []
  f = open('data/TestSet.txt')
  for line in f.readlines():
    line_arr = line.strip().split()
    data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
    label_arr.append(int(line_arr[2]))
  return data_arr, label_arr


def sigmoid(x):
  return 1.0/(1 + np.exp(-x))

def grad_ascent(data_arr, class_labels):
  """
  梯度上升法
  :param data_arr
  :param class_labels: class_labels
  :return:
  """
  data_mat = np.mat(data_arr)
  # 变成矩阵后进行转置
  label_mat = np.mat(class_labels).transpose()
  # m->数据量,样本数 n->特征值
  m, n = np.shape(data_mat)
  # 学习率: learning rate
  alpha = 0.001
  # 最大迭代次数 假装迭代
  max_cycles = 500
  # weights = np.ones((n, 1))
  weights = np.ones((n, 1))
  for k in range(max_cycles):
    h = sigmoid(data_mat * weights)
    error = label_mat - h
    weights = weights + alpha * data_mat.transpose()*error
  return weights

def plot_best_fit(weights):
  """
  可视化
  :param weights
  :return
  """
  import matplotlib.pyplot as plt
  data_mat, label_mat = load_data_set()
  data_arr = np.array(data_mat)
  n = np.shape(data_mat)[0]
  x_cord1 = []
  y_cord1 = []
  x_cord2 = []
  y_cord2 = []
  for i in range(n):
    if int(label_mat[i]) == 1:
      x_cord1.append(data_arr[i, 1])
      y_cord1.append(data_arr[i, 2])
    else:
      x_cord2.append(data_arr[i, 1])
      x_cord2.append(data_arr[i, 2])
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
  ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
  x = np.arrange(-3.0, 3.0, 0.1)
  y = (-weights[0] - weights[1]*x)/weights[2]
  ax.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()


def test():
 data_arr, label_arr = load_data_set()
 print (data_arr)
 print (label_arr)


if __name__ == "__main__":
  test()  



