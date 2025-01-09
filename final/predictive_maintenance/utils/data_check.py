"""
@Description: 期末大作业——构建预测性维护模型的数据集查看工具
@Author: 王宁远
@Date: 2025/01/09 15:22
@Version: 1.0.0
"""

# todo: 查看数据集概况

import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# 读取 CSV 文件
data = pd.read_csv("../resources/predictive_maintenance.csv")

# 查看数据集的整体信息
data.info()

# 查看前20条数据
print(data.head(20))

# 验证 Product ID 列的第一个字符是否与 Type 列的字符串一致
data['Product ID First Char'] = data['Product ID'].str[0]
is_consistent = (data['Product ID First Char'] == data['Type']).all()

if is_consistent:
    print("All the first character of 'Product ID' is the same with 'Type'.")

# 将 Target 列各值的分布可视化
target_counts = data['Target'].value_counts()
plt.figure(figsize=(6, 6))
target_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
plt.title('Distribution of Target Values')
plt.ylabel('')  # 隐藏 y 轴标签

# 显示图表
plt.show()