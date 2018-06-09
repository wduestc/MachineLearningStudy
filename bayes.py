#!/usr/bin/env python
# coding:utf-8
# @Time     : 18-6-9 下午11:16
# @Author   : w_di_sc
# @Site     : 
# @File     : bayes.py
# @Software : PyCharm

import numpy as np

"""
贝叶斯公式
p(xy) = p(x|y)p(y)=p(y|x)p(x)
p(x|y) = p(y|x)p(x)/p(y)
"""

def load_data_set():
    """
    desc:
    　创建数据集
    :return:
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] # 1- 侮辱性文字　0-　正常文字
    return posting_list, class_vec

def create_vocab_list(data_set):
    """
    获取所有单词的集合
    :param dataset:
    :return:
    """
    vocab_set = set()
    for item in data_set:
        vocab_set = vocab_set | set(item)
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    """
    遍历该单词出现次数　出现该单词则将该单词置为１
    :param vocab_list:
    :param input_set:
    :return:
    """
    result = [0] * len(vocab_list)
    for word in input_set:
        #判断在哪个位置
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            pass
    return result
"""
listOPosts, listClasses = load_data_set()
myVocabList = create_vocab_list(listOPosts)
print (myVocabList)
print (listOPosts[0])
result = set_of_words2vec(myVocabList, listOPosts[0])
print (result)
"""

def train_naive_bayes(train_mat, train_category):
    """

    :param train_mat: type is ndarray
      输入文本　大致是[[0, 1, 0, 1], [], []]
    :param train_category:
      文件对应类别分类 [0, 1, 0]
    :return:
    """
    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])
    #侮辱性文件出现的概率
    pos_abusive = np.sum(train_category)/train_doc_num
    p0num = np.ones(words_num)
    p1num = np.ones(words_num)
    p0num_all = 2.0
    p1num_all = 2.0

    for i in range(train_doc_num):
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])
    p1vec = np.log(p1num/p1num_all)
    p0vec = np.log(p0num/p0num_all)
    return p0vec, p1vec, pos_abusive





