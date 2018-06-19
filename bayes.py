#!/usr/bin/env python
# coding:utf-8
# @Time     : 18-6-9 涓嬪崍11:16
# @Author   : w_di_sc
# @Site     : 
# @File     : bayes.py
# @Software : PyCharm

import numpy as np

"""
璐濆彾鏂叕寮?
p(xy) = p(x|y)p(y)=p(y|x)p(x)
p(x|y) = p(y|x)p(x)/p(y)
"""

def load_data_set():
    """
    desc:
    銆€鍒涘缓鏁版嵁闆?
    :return:
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] # 1- 渚颈鎬ф枃瀛椼€€0-銆€姝ｅ父鏂囧瓧
    return posting_list, class_vec

def create_vocab_list(data_set):
    """
    鑾峰彇鎵€鏈夊崟璇嶇殑闆嗗悎
    :param dataset:
    :return:
    """
    vocab_set = set()
    for item in data_set:
        vocab_set = vocab_set | set(item)
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    """
    閬嶅巻璇ュ崟璇嶅嚭鐜版鏁般€€鍑虹幇璇ュ崟璇嶅垯灏嗚鍗曡瘝缃负锛?
    :param vocab_list:
    :param input_set:
    :return:
    """
    result = [0] * len(vocab_list)
    for word in input_set:
        #鍒ゆ柇鍦ㄥ摢涓綅缃?
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
      杈撳叆鏂囨湰銆€澶ц嚧鏄痆[0, 1, 0, 1], [], []]
    :param train_category:
      鏂囦欢瀵瑰簲绫诲埆鍒嗙被 [0, 1, 0]
    :return:
    """
    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])
    #渚颈鎬ф枃浠跺嚭鐜扮殑姒傜巼
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





