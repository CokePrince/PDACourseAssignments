"""
@Description: 第十三周课后作业——鸢尾花示例代码结果复现
@Author: 王宁远
@Date: 2024/11/26 16:24
@Version: 1.0
"""

# todo:

# 改变邻居数并尝试对特征进行标准化，观察结果的变化。

# coding=utf-8
import random
from sklearn.datasets import load_wine
import math
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
wine = load_wine()
data = wine['data']
target = wine['target']

# 将数据和标签整合为字典结构，便于操作
datas = []
for i in range(len(data)):
    row = {"data": data[i], "target": target[i]}
    datas.append(row)

# 查看数据结构
print("数据样本示例：", datas[:2])

# 2. 数据分组
random.shuffle(datas)  # 随机打乱数据
n = len(datas) // 3  # 测试集占三分之一
test_set = datas[:n]
train_set = datas[n:]

# 提取训练集和测试集的数据部分以便标准化
X_train = [item["data"] for item in train_set]
X_test = [item["data"] for item in test_set]

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 更新训练集和测试集的数据部分（标准化）
train_set_scaled = [{"data": X_train_scaled[i], "target": train_set[i]["target"]} for i in range(len(train_set))]
test_set_scaled = [{"data": X_test_scaled[i], "target": test_set[i]["target"]} for i in range(len(test_set))]

# 3. 定义距离计算函数
def distance(d1, d2):
    """
    计算两个样本之间的欧几里得距离
    """
    res = 0
    for i in range(len(d1["data"])):
        res += (float(d1["data"][i]) - float(d2["data"][i])) ** 2
    return math.sqrt(res)

# 4. 定义KNN函数
def knn(data, dataset, k=5):
    """
    使用K近邻算法预测测试数据的类别
    """
    # 1) 计算距离
    res = [{"target": train["target"], "distance": distance(data, train)} for train in dataset]

    # 2) 按距离排序
    res = sorted(res, key=lambda item: item["distance"])

    # 3) 取前K个
    res2 = res[:k]

    # 4) 加权平均
    result = {0: 0, 1: 0, 2: 0}  # Wine 数据集有 3 类 (0, 1, 2)
    sum_distances = sum(1 / (r["distance"] + 1e-5) for r in res2)  # 距离加权的归一化

    for r in res2:
        weight = 1 / (r["distance"] + 1e-5)  # 距离的倒数作为权重
        result[r["target"]] += weight / sum_distances

    # 返回预测结果（权重最大的类别）
    predicted_class = max(result, key=result.get)
    return predicted_class

# 5. 测试阶段
def evaluate_knn(dataset, dataset_name, k_values):
    accuracies = {}
    for k in k_values:
        correct = 0
        for test in test_set:
            actual = test["target"]
            predicted = knn(test, dataset, k=k)

            if actual == predicted:
                correct += 1

        accuracy = 100 * correct / len(test_set)
        accuracies[k] = accuracy
        print(f"{dataset_name}, 邻居数: {k}, 准确率: {accuracy:.2f}%")
    return accuracies

# 尝试不同的k值
k_values = [1, 3, 5, 7, 9]

# 非标准化结果
print("非标准化结果:")
non_normalized_accuracies = evaluate_knn(train_set, "非标准化", k_values)

# 标准化结果
print("\n标准化结果:")
normalized_accuracies = evaluate_knn(train_set_scaled, "标准化", k_values)