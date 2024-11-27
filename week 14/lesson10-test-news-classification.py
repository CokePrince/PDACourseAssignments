"""
@Description: 第十三周课后作业——鸢尾花示例代码结果复现
@Author: 王宁远
@Date: 2024/11/26 16:24
@Version: 1.0
"""

# todo:

# 比较不同拉普拉斯平滑的α值对朴素贝叶斯分类的影响，并比较朴素贝叶斯分类与KNN的准确率

#coding=UTF-8
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

#1)获取数据
news = fetch_20newsgroups(subset='all')

#2)划分数据集
x_train, x_test, y_train, y_test =train_test_split(news.data, news.target)

#3)特征工程：文本特征抽取-tf_idf
transfer = TfidfVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# nbc stands for classification by native bayes algorithm
def nbc(alpha_value):

    #4)朴素贝叶斯算法分类器
    model = MultinomialNB(alpha = alpha_value)
    model.fit(x_train, y_train)
    #5)模型评估
    #计算准确率
    score =model.score(x_test, y_test)
    print(f"α为{alpha_value}时，准确率为：\n{score}\n")
    return None

# nbc stands for classification by native bayes algorithm
def knnc(k_value):

    #4)KNN算法分类器
    model = KNeighborsClassifier(n_neighbors = k_value)
    model.fit(x_train, y_train)
    #5)模型评估
    # 计算准确率
    score = model.score(x_test, y_test)
    print(f"邻居数为{k_value}时，准确率为：\n{score}\n")
    return None

alpha_value_list = [0.1, 0.3, 0.5, 0.7, 0.9]
k_value_list = [9, 7, 5, 3, 1]

for index in range(len(alpha_value_list)):
    nbc(alpha_value_list[index])
    knnc(k_value_list[index])