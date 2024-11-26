"""
@Description: 第十三周课后作业——PPT上的鸢尾花任务
@Author: 王宁远
@Date: 2024/11/26 16:08
@Version: 1.0
"""

# todo:

"""
1.采用load_iris读取数据并查看 
2.分割数据，产生75%的训练样本，25%的测试样本
3.标准化数据
4.导入K近邻分类模块
5.测试与性能评估，生成评估报告
"""

# 引入数据集，sklearn包含众多数据集
from sklearn import datasets
# 将数据分为测试集和训练集
from sklearn.model_selection import train_test_split
# 数据标准化处理
from sklearn.preprocessing import StandardScaler
# 利用邻近点方式训练数据
from sklearn.neighbors import KNeighborsClassifier
# 评估模型性能
from sklearn.metrics import classification_report

# 引入数据,本次导入鸢尾花数据，iris数据包含4个特征变量
iris = datasets.load_iris()
# 特征变量
iris_X = iris.data
# 目标值
iris_y = iris.target

# 利用train_test_split进行训练集和测试机进行分开，test_size占25%
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.25)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 循环遍历不同的邻居数
for n_neighbors in range(1, 9):
    # 导入K近邻分类模块
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # 进行填充测试数据进行训练
    knn.fit(X_train, y_train)

    # 测试与性能评估
    y_pred = knn.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    print(f"邻居数: {n_neighbors}\n评估报告:\n{report}\n")