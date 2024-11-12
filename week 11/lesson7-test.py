"""
@Description: 第十一周课后作业
@Author: 王宁远
@Date: 2024/11/12 15:33
@Version: 1.0
"""

"""
todo:

使用第八周课程中提供的葡萄酒品质数据，求fixed acidity、density、pH等参数和质量的相关度。
"""

import pandas as pd

# 读取 CSV 文件
file_path = 'white_wine.csv'
df = pd.read_csv(file_path, delimiter='\t')  # 使用制表符作为分隔符

# 显示数据集的前几行，检视数据是否被正确读取
print(df.head())

# 计算相关系数矩阵
correlation_matrix = df.corr()

# 提取与 quality 相关的列
quality_correlations = correlation_matrix['quality']

# 打印 fixed acidity, density, pH 与 quality 的相关度
print("\nCorrelation of 'fixed acidity' with 'quality':", quality_correlations['fixed acidity'])
print("Correlation of 'density' with 'quality':", quality_correlations['density'])
print("Correlation of 'pH' with 'quality':", quality_correlations['pH'])