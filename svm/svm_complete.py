# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

class optStruct:
  """
  建立数据结构来保存所有重要值
  """
  def __init__(self, dataMatIn, classLabels, C, toler, kTup):
    """
    Args:
     dataMatIn 数据集
     classLabels 类别标签
     C 松弛变量
     toler 容错率
     kTup
    """
    self.X = dataMatIn
    self.labelMat = classLabels
    self.C = C
    self.tol = toler

    #数据行数
    self.m = shape(dataMatIn)[0]
    self.alphas = mat(zeros((self.m, 1)))
    self.b = 0

    #误差缓存
    self.eCache = mat(zeros((self.m, 2)))

    #m行m列的矩阵
    self.K = mat(zeros((self.m, self.m)))
    for i in range(self.m):
      self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)
  
  def kernelTrans(X, A, kTup):
    """
    核转换函数
    Args:
       X  dataMatIn 数据集
       A  dataMatIn 数据集第i行的数据
       kTup 核函数的信息
    Returns:
    """
    m, n = shape(X)
    K = mat(zeros(m, 1))
    if kTup[0] == 'lin':
      K = X * A.T
    elif kTup[0] == 'rbf':
      for j in range(m):
        deltaRow = X[j, :] - A
        K[j] = deltaRow * deltaRow.T
      K = exp(K / (-1 * kTup[1] ** 2))
    else:
      raise NameError('Houston We Have a Problem')
    return K
  
  def loadDataSet(fileName):
    """
    Desc:
      对文件进行逐行解析,从而得到第i行的类标签和整个数据矩阵
    Args:
      fileName - 文件名
    Returns:
      dataMat  数据矩阵
      labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
      lineArr = line.strip().split('\t')
      dataMat.append([float(lineArr[0]), float(lineArr[1])])
      labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
  
  def calcEk(oS, k):
    """
    该过程在完整版的SMO算法中陪出现次数较多,从而得到第行的类标签和整个数据矩阵
    Args:
     oS optStruct对象
     k  具体的某一行
    Returns:
     Ek 预测结果与真实结果进行比对 计算误差Ek
    """
    fXk = multiply(oS.alphas, oS.labelMat).T*oS.K[:, K] + os.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
  
  def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
      i 第一个alpha的下标
      m 所有alpha的数目
    Returns:
      j 返回一个不为i的随机数 在0~m之间的整数值
    """
    j = i
    while j == i:
      j = random.randint(0, m-1)
    return j
    

