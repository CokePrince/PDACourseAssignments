"""
@Description: 第八周课后作业
@Author: 王宁远
@Date: 2024/10/16 11:29
@Version: 1.0
"""

"""
todo:

实战演练 1：读取白葡萄酒品质数据集

#该数据集的数据形式如下：
#首先，我们需要将存储在本地的数据集white_wine.csv读取入内存中。
说明：引入csv模块，打开文件
将数据保存于列表content中
打印content前5行
"""

import csv

path = "white_wine.csv"
with open(path, "r") as f:  # 使用 with 语句自动管理文件关闭
    reader = csv.reader(f)
    content = []
    for row in reader:
        # 使用制表符将列表中的元素连接成一个字符串，使之呈现正常的制表效果
        content.append('\t'.join(row))

# 打印前5行内容
for i in range(0, 5):
    print(content[i])

"""
todo:

实战演练 2 查看白葡萄酒中总共分为几个品质

品质quality变量在数据中是一个离散变量，而不是连续的，所以它只会有固定的几个等级。那么我们用Python中自带的集合set来查看白葡萄酒中总共的品质等级 
说明：
使用集合set查看白葡萄酒总共分为几个品质，并将所有品质等级保存在集合unity_quality中
其中，品质等级数据在最后一列
"""

qualities = []
for row in content[1:]:
    qualities.append(int(row[-1]))
unity_quality = set(qualities)
print (unity_quality)

"""
todo:

实战演练 3 : 按白葡萄酒等级将数据集划分为7个子集

将数据按白葡萄酒等级quality进行切分为7个子集，保存到一个字典中，字典的键为quality具体数值，值为归属于该quality的样本列表
说明

按白葡萄酒等级将数据集划分为7个子集，用字典保存每个子集，字典变量名为content_dict，变量的关键词key为品质，值value为每个品质子集的数据列表。
"""

content_dict = {}

for row in content[1:]:
    quality = int(row[-1])
    if quality not in content_dict.keys():
        content_dict[quality] = [row]
    else:
        content_dict[quality].append(row)
# 按等级排序顺序输出子集
for key in sorted(content_dict.keys()):
    print(f"Quality: {key}, List: {content_dict[key]}")

"""
todo:

实战演练4: 统计在每个品质的样本量
那么，你会不会好奇每个品质的样本是不是有多有少呢？哪个品质的样本量多一些，哪个又少一点？
说明:
统计每个品质下的样本数，保存为number_tuple，该变量为一个列表，每个元素是一个二元元组，元组第一个元素是品质，第二个元素是样本数
"""

number_tuple = []

for quality, list in content_dict.items():
    number_tuple.append((quality, len(list)))

print(number_tuple)

"""
todo:

实战演练5: 求每个数据集中fixed acidity的均值
既然白葡萄酒有品质的区别，那么是不是每个品质的fixed acidity区别会很大呢？
说明:
计算每个品质下变量fixed acidity的均值，并保存于列表mean_tuple中
要求列表中每一个元组的第一个元素为quality，第二个元素为该品质下fix acidity的均值
"""

mean_tuple = []

for quality, list in content_dict.items():
    sum_ = 0
    for row in list:
        sum_ += float(row[0])
    mean_tuple.append((quality, sum_/len(list)))

print (mean_tuple)

