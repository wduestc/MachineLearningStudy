# coding:utf-8

import numpy as np
from sklearn import svm

def loadDataSet(fileName):
  dataMat = []
  labelMat = []
  with open(fileName, 'r') as fr:
    for line in fr.readlines():
      lineArr = line.strip().split('\t')
      dataMat.append([float(lineArr[0]), float(lineArr[1])])
      labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
  
def sklearn_svm(fileName):
  X, Y = loadDataSet(fileName)
  X = np.mat(X)
  # 拟合一个svm模型
  clf = svm.SVC(kernel='linear')
  clf.fit(X, Y)
  return clf

def ml_predict(clf, X):
  pre_y = clf.predict(X)
  return pre_y

if __name__ == "__main__":
"""
 进行预测
"""
 result = ml_predict(sklearn_svm(r'C:\Users\wd\Documents\GitHub\MachineLearningStudy\data\SVM\testSet.txt'), [[7, 1]])
 print (result)