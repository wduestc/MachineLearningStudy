# coding:utf-8
print (__doc__)
import operator
from math import log
from collections import Counter

def createDataSet():
    """
    Desc:
     创建数据集
    parameters:
    :return:
    """

    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [1, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 将当前的实例的标签进行存储　每一行最后一个代表一个标签
        #print (featVec)
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典 如果当前的键值不存在　则扩展字典并将当前键值加入字典
        #print (currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #对于label标签的占比　求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使所有类标签发生的频率计算类比出现的概率
        prob = float(labelCounts[key])/numEntries
        # 计算香农
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 将指定特征的特征值等于value的行剩下的列作为子数据集
def splitDataSet(dataSet, index, value):
    """
    desc:
     依据index列进行分类　如果index列的值等于value的时候　就要将index划分到我们创建的新的数据集中
    :param dataSet:
    :param index:
    :param value:
    :return:
    """

    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
        reducedFeatVec.extend(featVec[index+1:])
        retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    desc:
      选择最好的特征
    parameters:
      dataSet 数据集
    returns:
      bestFeature 最优特征列
    """
    numFeatures = len(dataSet[0]) - 1
    # 原始数据熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优信息增益值
    bestInfoGain, bestFeature = 0.0, -1
    # iterator over all the features
    for i in range(numFeatures):
        # create a list of all the example of this feature
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print ("infoGain=", infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    desc:
      选择出现次数最多的一个结果
    parameters:
      classlist label的列集合
    return:
      bestFeature
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    Desc:
      创建决策树
    parameters:
      dataSet - 要创建决策树的训练数据集
      labels - 训练数据集中对应含义的labels
    returns:
      myTree
    """
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量　也就是说只有一个类别　则直接返回结果就可以了
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件:使用完所有的特征　仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeat:{}}
    del(labels[bestFeat])
    # 取出最优列
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    """
    Desc:
      对新的数据进行分类
    Parameters:
      inputTree -- 已经训练好的决策树模型
      featLables -- Feature标签对应的名称
      testVec -- 测试输入的数据
    returns:
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    # 测试数据
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print ('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel






    
    
    

if __name__ == "__main__":
    testdata,testlabels = createDataSet()
    print (calcShannonEnt(testdata))
    #print (calcShannonEnt(createDataSet()))

