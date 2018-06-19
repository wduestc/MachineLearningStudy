# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:47:46 2018

@author: wd
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# 参数
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# 加载数据
iris = load_iris()

X = iris.data
Y = iris.target
print (X)
print (Y)

