"""
@Description: 第十六周课后作业——探寻泰坦尼克号上各因素（客舱等级、年龄、性别、上船港口等）的特征的生还的可能性
@Author: 王宁远
@Date: 2024/12/11 11:43
@Version: 1.0
"""

# todo:

# 探寻泰坦尼克号上各因素（客舱等级、年龄、性别、上船港口等）的特征的生还的可能性，用决策树实现

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1) 获取数据
data = pd.read_csv("titanic.csv", encoding='gb2312')

# 确定特征值与目标值
x = data[["pclass", "age", "sex"]]
y = data["survived"]

# 2、数据处理
# 1）缺失值处理
x_copy = x.copy()  # 创建一个副本以避免SettingWithCopyWarning
x_copy.fillna(x_copy['age'].mean(), inplace=True)

# 将性别转换为数值型
x_copy['sex'] = x_copy['sex'].map({'male': 0, 'female': 1})

# 转换为字典列表
x_dict = x_copy.to_dict(orient="records")

# 3 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x_dict, y, random_state=22)

# 5 字典特征抽取
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 6 构建决策树模型
model = DecisionTreeClassifier(criterion="entropy")
model.fit(x_train, y_train)

# 7 模型评估
score = model.score(x_test, y_test)
print("准确率为：\n", score)

# 可视化决策树
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=transfer.get_feature_names_out(), class_names=['Not Survived', 'Survived'])
plt.title("Decision Tree for Titanic Survival Prediction")

# 保存图形到PDF文件
output_filename = "decision_tree_titanic.pdf"
plt.savefig(output_filename, format='pdf')
print(f"决策树已保存为 {output_filename}")
plt.show()