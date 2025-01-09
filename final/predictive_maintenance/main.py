"""
@Description: 期末大作业——构建预测性维护模型的主文件
@Author: 王宁远
@Date: 2025/01/09 10:26
@Version: 1.0.0
"""

# todo: 构建预测性维护模型

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

plt.rcParams['font.family'] = 'Times New Roman'

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 去除无关特征
def drop_extraneous_columns(data):
    logging.info("Dropping extraneous columns...")
    # 删除不需要的列（如 UDI 和 Product ID）
    data = data.drop(columns=['UDI', 'Product ID'])
    logging.info("Columns 'UDI', 'Product ID' dropped.")
    return data

# 可视化并剔除异常值
def reject_outliers(data):

    logging.info("Showing and rejecting outliers...")

    # 仅选择数值列，并忽略 Target
    numeric_data = data.select_dtypes(include=['int64', 'float64']).drop(columns=['Target'])

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_data)
    plt.xticks(rotation=0)
    plt.show()

    # 计算 IQR
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1

    # 标记异常值
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR)))

    # 统计每列的异常值数量
    outliers_per_column = outliers.sum()
    print("Outliers per column: ")
    print(outliers_per_column)

    # 标记整体异常值（任意一列存在异常值即为异常样本）
    outliers_any_column = outliers.any(axis=1)
    print("Total outliers: ", outliers_any_column.sum())

    # 查看异常值样本
    print("Sample of outliers: ")
    print(data[outliers_any_column])

    # 删除异常值
    data_cleaned = data[~outliers_any_column].copy()  # 显式创建副本
    print("Data amount after cleaning: ", len(data_cleaned))

    logging.info("Rejecting outliers completed.")

    return data_cleaned

# 特征转换
def feature_label(data):

    logging.info("Performing feature labeling...")

    # 对 Failure Type 列进行标签编码
    label_encoder = LabelEncoder()
    data['Failure Type'] = label_encoder.fit_transform(data['Failure Type'])

    # 对 Type 列进行独热编码
    data = pd.get_dummies(data, columns=['Type'], drop_first=True)

    logging.info("Feature labeling completed.")
    return data, label_encoder

# 训练前处理
def pre_operate(data):

    logging.info("Processing data before training...")

    # 分离特征和目标变量
    x = data.drop(columns=['Target', 'Failure Type'])
    y = data['Failure Type']

    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)  # 对训练集进行拟合和转换
    x_test = scaler.transform(x_test)  # 对测试集仅进行转换

    # 将标准化后的数据重新转换为 DataFrame，并恢复列名（后续使用）
    x_train = pd.DataFrame(x_train, columns=x.columns)
    x_test = pd.DataFrame(x_test, columns=x.columns)

    # 获取特征信息（可用于确定模型构建时的参数）
    print("Number of features after scaling: ", x_train.shape[1])
    print("Info of features:")
    print(data.info())

    logging.info("Processing data before training completed.")
    return x_train, x_test, y_train, y_test

# 训练随机森林模型
def random_forest(x_train, y_train):

    logging.info("Training random forest classifier...")

    rf = RandomForestClassifier(random_state=42)

    # 记录随机森林训练时长
    start_time = time.time()  # 记录训练开始时间
    rf.fit(x_train, y_train)  # 训练随机森林模型
    rf_training_time = time.time() - start_time  # 计算训练时长
    print(f"Training time of Random Forest classifier: {rf_training_time:.2f} second.")

    logging.info("Training completed.")
    return rf

# 基于 GridSearchCV 超参数调优的随机森林
def random_forest_with_grid_search(x_train, y_train):

    logging.info("Training random forest classifier with GridSearchCV...")

    rf = RandomForestClassifier(random_state=42)

    # 超参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # 使用 GridSearchCV 进行超参数调优
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    start_time = time.time()
    grid_search.fit(x_train, y_train)

    # 最佳模型
    best_rf = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    rf_training_time = time.time() - start_time
    print(f"Training time of Random Forest classifier with GridSearchCV: {rf_training_time:.2f} second.")

    logging.info("Training completed.")
    return best_rf

# 基于 RandomizedSearchCV 超参数调优的随机森林
def random_forest_with_randomized_search(x_train, y_train):

    logging.info("Training random forest classifier with RandomizedSearchCV...")

    rf = RandomForestClassifier(random_state=42)

    # 超参数分布
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # 使用 RandomizedSearchCV 进行超参数调优
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,  # 随机搜索的迭代次数
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    start_time = time.time()
    random_search.fit(x_train, y_train)

    # 最佳模型
    best_rf = random_search.best_estimator_
    print("Best Parameters:", random_search.best_params_)

    rf_training_time = time.time() - start_time
    print(f"Training time of Random Forest classifier with RandomizedSearchCV: {rf_training_time:.2f} seconds.")

    logging.info("Training completed.")
    return best_rf

# 训练梯度提升树模型
def gradient_boost(x_train, y_train):

    logging.info("Training gradient boosting classifier...")

    gb = GradientBoostingClassifier(random_state=42)
    start_time = time.time()
    gb.fit(x_train, y_train)
    gb_training_time = time.time() - start_time
    print(f"Training time of Gradient Boost classifier: {gb_training_time:.2f} second.")

    logging.info("Training completed.")
    return gb

# 训练神经网络模型
def neural_network(x_train, y_train):

    logging.info("Training neural network classifier...")

    nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    start_time = time.time()
    nn.fit(x_train, y_train)
    nn_training_time = time.time() - start_time
    print(f"Training time of Neural Network classifier: {nn_training_time:.2f} second.")

    logging.info("Training completed.")
    return nn

# 通用的模型训练函数，支持超参数调优
def train_model_with_search(model_type, x_train, y_train, search_type=None, param_grid=None, **kwargs):
    """
    :param model_type: 模型类型（如 'random_forest', 'gradient_boost', 'neural_network'）
    :param search_type: 超参数调优类型（如 'grid', 'random'）
    :param param_grid: 超参数网格或分布
    :param kwargs: 其他可选参数（如超参数）
    """
    logging.info(f"Training {model_type} classifier with {search_type} search...")

    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, **kwargs)
    elif model_type == 'gradient_boost':
        model = GradientBoostingClassifier(random_state=42, **kwargs)
    elif model_type == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if search_type == 'grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    else:
        search = model

    start_time = time.time()
    search.fit(x_train, y_train)
    training_time = time.time() - start_time

    if search_type in ['grid', 'random']:
        print(f"Best Parameters: {search.best_params_}")
        model = search.best_estimator_

    print(f"Training time of {model_type} classifier with {search_type} search: {training_time:.2f} seconds.")

    logging.info("Training completed.")
    return model

# 使用模型预测并评估模型准确性
def predict(model, x_test, y_test, model_name, visualization, label_encoder):
    """
        :param model: 训练好的模型
        :param model_name: 模型名称，用于生成提示和图像
        :param visualization: 是否生成特征重要性排序图
        :param label_encoder: 用于显示图像中的坐标轴文字
    """
    logging.info("Evaluating model " + model_name + "...")

    y_pred = model.predict(x_test)

    print("Accuracy of", model_name, ":", accuracy_score(y_test, y_pred))
    print("Classification Report of", model_name, ":\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of", model_name, ":\n", np.array2string(cm)[1:-1]) # [1:-1]用于去除混淆矩阵最外侧的一对方括号

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of ' + model_name)
    plt.show()

    if visualization:
        # 获取特征重要性
        feature_importances = model.feature_importances_
        feature_names = x_test.columns

        # 按重要性排序
        indices = np.argsort(feature_importances)[::-1]

        # 可视化特征重要性
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances Sorted by " + model_name)
        plt.bar(range(x_test.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(x_test.shape[1]), [feature_names[i] for i in indices], rotation=90, fontsize=10)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.subplots_adjust(bottom=0.3)
        plt.tight_layout()
        plt.show()

    logging.info("Evaluating model completed.")
    return accuracy_score(y_test, y_pred)

if __name__ == '__main__':

    data = pd.read_csv("resources/predictive_maintenance.csv")
    data = drop_extraneous_columns(data)
    data = reject_outliers(data)
    data, label_encoder = feature_label(data)
    x_train, x_test, y_train, y_test = pre_operate(data)

    param_grid_rf = {
        'n_estimators': [100, 200, 300],  # 树的数量，中等范围
        'max_depth': [None, 10, 20, 30],  # 树的最大深度，None 表示不限制
        'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶节点所需的最小样本数
        'bootstrap': [True, False]  # 是否使用bootstrap采样
    }

    param_grid_gb = {
        'n_estimators': [100, 200, 300],  # 树的数量，中等范围
        'learning_rate': [0.01, 0.1, 0.2],  # 学习率，控制每棵树的贡献
        'max_depth': [3, 5, 7],  # 树的最大深度，限制树的复杂度
        'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶节点所需的最小样本数
        'subsample': [0.8, 1.0]  # 样本采样比例
    }

    param_grid_nn = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # 隐藏层结构，中等规模
        'activation': ['relu', 'tanh'],  # 激活函数，relu 和 tanh 是常用选择
        'solver': ['adam'],  # 优化器，adam 是默认且高效的选择
        'alpha': [0.0001, 0.001, 0.01],  # 正则化参数，控制过拟合
        'learning_rate': ['constant', 'adaptive'],  # 学习率策略
        'max_iter': [200, 300]  # 最大迭代次数，中等范围
    }

    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    param_dist_gb = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 11),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'subsample': uniform(0.8, 0.2)
    }

    param_dist_nn = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': uniform(0.0001, 0.01),
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': randint(200, 500)
    }

    rf = train_model_with_search("random_forest", x_train, y_train)
    predict(rf, x_test, y_test, "rf", True, label_encoder)

    rf_grid = train_model_with_search("random_forest", x_train, y_train, "grid", param_grid_rf)
    predict(rf_grid, x_test, y_test, "rf_grid", True, label_encoder)

    rf_random = train_model_with_search("random_forest", x_train, y_train, "random", param_dist_rf)
    predict(rf_random, x_test, y_test, "rf_random", True, label_encoder)

    gb = train_model_with_search("gradient_boost", x_train, y_train)
    predict(gb, x_test, y_test, "gb", True, label_encoder)

    # gb_grid = train_model_with_search("gradient_boost", x_train, y_train, "grid", param_grid_gb)
    # predict(gb_grid, x_test, y_test, "gb_grid", True, label_encoder)

    gb_random = train_model_with_search("gradient_boost", x_train, y_train, "random", param_dist_gb)
    predict(gb_random, x_test, y_test, "gb_random", True, label_encoder)

    nn = train_model_with_search("neural_network", x_train, y_train)
    predict(nn, x_test, y_test, "nn", False, label_encoder)

    nn_grid = train_model_with_search("neural_network", x_train, y_train, "grid", param_grid_nn)
    predict(nn_grid, x_test, y_test, "nn_grid", False, label_encoder)

    nn_random = train_model_with_search("neural_network", x_train, y_train, "random", param_dist_nn)
    predict(nn_random, x_test, y_test, "nn_random", False, label_encoder)