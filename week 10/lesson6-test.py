"""
@Description: 第十周课后作业
@Author: 王宁远
@Date: 2024/10/30 11:34
@Version: 1.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 为后续任务保存图表而创建目录
directory = 'images'
if not os.path.exists(directory):
    os.makedirs(directory)

# todo:

"""
1.	数据可视化类型匹配
使用以下数据，绘制一个 2x2 布局的图形，每张子图选择适合的数据可视化类型：
o	数据1：10个班级的学生人数 [25, 30, 35, 40, 20, 15, 10, 50, 45, 30]
o	数据2：每天温度的随机变化，范围在20℃到35℃之间 (生成长度为100的数据)
o	数据3：使用从正态分布生成的1000个随机数据
o	数据4：假设4个不同产品的销售份额比例 [30, 25, 25, 20]
要求：为每种数据选择一个合适的图表类型，并在子图中分别绘制（提示：可以考虑条形图、散点图、直方图和饼图）。
"""

# 数据准备
# 数据1: 10个班级的学生人数
class_sizes = [25, 30, 35, 40, 20, 15, 10, 50, 45, 30]
classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5',
           'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10']

# 数据2: 每天温度的随机变化
np.random.seed(0)  # 设置随机种子以确保结果可重复
daily_temperatures = 20 + 15 * np.random.rand(100)

# 数据3: 正态分布的随机数据
normal_distribution_data = np.random.randn(1000)

# 数据4: 四个不同产品的销售份额比例
product_shares = [30, 25, 25, 20]
products = ['Product A', 'Product B', 'Product C', 'Product D']

# 创建2x2的子图布局
fig, axs = plt.subplots(2, 2)

# 子图1: 条形图 - 班级学生人数
axs[0, 0].bar(classes, class_sizes)
axs[0, 0].set_title('Student Count by Class')
axs[0, 0].set_xticks(np.arange(len(classes)))
axs[0, 0].set_xticklabels(classes, rotation=45)

# 子图2: 散点图 - 温度随时间的变化
axs[0, 1].scatter(range(100), daily_temperatures, s=10, alpha=0.5)
axs[0, 1].set_title('Daily Temperature Variation')
axs[0, 1].set_xlabel('Day')
axs[0, 1].set_ylabel('Temperature (°C)')

# 子图3: 直方图 - 正态分布数据
axs[1, 0].hist(normal_distribution_data, bins=30, color='green', alpha=0.7)
axs[1, 0].set_title('Histogram of Normal Distribution')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')

# 子图4: 饼图 - 产品销售份额
axs[1, 1].pie(product_shares, labels=products, autopct='%1.1f%%', startangle=140)
axs[1, 1].set_title('Market Share by Product')

# 调整子图间距
plt.tight_layout()

# 保存图表
plt.savefig("images/task1.png")

# 显示图表
plt.show()


# todo:

"""
2.	自定义样式和标签
创建一组 3x1 的子图布局，其中包含以下三种图表：箱形图、小提琴图和茎叶图。
o	数据：使用正态分布生成长度为500的随机数据。
o	要求：自定义每张图的颜色、线条样式和标签，如使用绿色箱形图、蓝色的小提琴图和红色的茎叶图。
"""

# 生成符合正态分布的数据
data = np.random.randn(500)

# 创建一个3x1的子图布局
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 绘制箱形图
box_plot = axs[0].boxplot(data, patch_artist=True)
# 自定义颜色
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(box_plot[element], color='blue')
# 自定义填充颜色
plt.setp(box_plot['boxes'], facecolor='green')
axs[0].set_title('Box Plot')
axs[0].set_ylabel('Values')

# 绘制小提琴图
violin_plot = axs[1].violinplot(data, showmeans=True, showmedians=True)
# 自定义颜色
for body in violin_plot['bodies']:
    body.set_facecolor('blue')
    body.set_edgecolor('black')
    body.set_alpha(1)
plt.setp(violin_plot['cbars'], color='black')
plt.setp(violin_plot['cmeans'], color='blue')
plt.setp(violin_plot['cmedians'], color='red')
plt.setp(violin_plot['cmins'], color='black')
plt.setp(violin_plot['cmaxes'], color='black')
axs[1].set_title('Violin Plot')
axs[1].set_ylabel('Values')

# 绘制茎叶图
stem_plot = axs[2].stem(data, linefmt='r-', markerfmt='ro', basefmt='k-')
axs[2].set_title('Stem Plot')
axs[2].set_ylabel('Values')

# 调整子图间距
plt.tight_layout()

# 保存图表
plt.savefig("images/task2.png")

# 显示图表
plt.show()


# todo:

"""
3.	组合正弦和余弦曲线
绘制一个 1x2 布局的图形，左侧为正弦函数的折线图，右侧为余弦函数的散点图。
o	数据：x = np.linspace(0, 2 * np.pi, 100)，y1 = np.sin(x)，y2 = np.cos(x)。
o	要求：为正弦和余弦曲线分别选择不同的颜色，并添加适当的标题和轴标签。
"""

# 创建数据
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)  # 正弦函数
y2 = np.cos(x)  # 余弦函数

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 左侧子图 - 正弦函数的折线图
ax1.plot(x, y1, color='blue', label='sin(x)')
ax1.set_title('Sine Wave')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')
ax1.legend()

# 右侧子图 - 余弦函数的散点图
ax2.scatter(x, y2, color='red', label='cos(x)')
ax2.set_title('Cosine Scatter')
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')
ax2.legend()

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig("images/task3.png")

# 显示图表
plt.show()


# todo:

"""
4.	综合图表展示
创建一个 3x2 的子图布局，展示以下六种图表：折线图、条形图、直方图、饼图、箱形图和小提琴图。
o	使用以下数据集：
	数据1：x = np.linspace(0, 10, 50)，y = np.sin(x)
	数据2：五种不同商品的销量 [20, 15, 30, 35, 25]
	数据3：从正态分布生成的长度为500的随机数据
	数据4：四种类别的市场份额 [40, 30, 20, 10]
o	要求：为每个图表选择合适的数据和适当的自定义样式（如颜色、网格），确保每个图表的视觉风格一致。
"""

# 数据准备
x = np.linspace(0, 10, 50)
y = np.sin(x)

sales = [20, 15, 30, 35, 25]

normal_data = np.random.randn(500)

market_share = [40, 30, 20, 10]

labels = ['Product A', 'Product B', 'Product C', 'Product D']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

# 折线图
axes[0, 0].plot(x, y, color='blue')
axes[0, 0].set_title('Line Plot')
axes[0, 0].grid(True)

# 条形图
axes[0, 1].bar(range(len(sales)), sales, color='green')
axes[0, 1].set_xticks(range(len(sales)))
axes[0, 1].set_xticklabels(['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'])
axes[0, 1].set_title('Bar Chart')

# 直方图
axes[1, 0].hist(normal_data, bins=20, color='orange', edgecolor='black')
axes[1, 0].set_title('Histogram')

# 饼图
axes[1, 1].pie(market_share, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'blue', 'green', 'yellow'])
axes[1, 1].set_title('Pie Chart')

# 箱形图
axes[2, 0].boxplot(normal_data, patch_artist=True, boxprops=dict(facecolor="green"))
axes[2, 0].set_title('Box Plot')

# 小提琴图
axes[2, 1].violinplot(normal_data, showmeans=True, showmedians=True)
axes[2, 1].set_title('Violin Plot')

# 自动调整子图布局
plt.tight_layout()

# 保存图表
plt.savefig("images/task4.png")

# 显示图表
plt.show()