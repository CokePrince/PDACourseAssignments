"""
@Description: 期末大作业——预测歌曲是否有显式标记的主文件
@Author: 王宁远
@Date: 2025/01/10 20:57
@Version: 1.0.1
"""

# todo: 预测歌曲是否有显式标记

import logging
import os
import time

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Dense, Dropout
from scipy.stats import zscore, randint, uniform
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

plt.rcParams['font.family'] = 'Times New Roman'

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载数据并进行预处理
def preprocess(filepath):

    logging.info("Loading and preprocessing data...")
    data = pd.read_csv(filepath, encoding="ISO-8859-1")

    # 一些列在 csv 中以字符串形式存储数值，需要转换
    columns_to_convert = [
        "All Time Rank", "Track Score", "Spotify Streams", "Spotify Playlist Count",
        "Spotify Playlist Reach", "Spotify Popularity", "YouTube Views", "YouTube Likes",
        "TikTok Posts", "TikTok Likes", "TikTok Views", "YouTube Playlist Reach",
        "Apple Music Playlist Count", "AirPlay Spins", "SiriusXM Spins", "Deezer Playlist Count",
        "Deezer Playlist Reach", "Amazon Playlist Count", "Pandora Streams", "Pandora Track Stations",
        "Soundcloud Streams", "Shazam Counts"
    ]

    # 定义转换函数
    def convert(value):
        if pd.isna(value) or value == "N/A" or value == "NaN":
            return np.nan
        if isinstance(value, str):
            value = value.replace(",", "").replace(" ", "")
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                return np.nan
        return value

    for col in columns_to_convert:
        data[col] = data[col].apply(convert)

    # 剔除无关特征 ISRC 和 TIDAL Popularity（所有该列有值的数据的 Explicit Track 均缺失）
    data.drop(columns=["ISRC", "TIDAL Popularity"], inplace=True)

    # 删除 Explicit Track 为空的行
    data = data.dropna(subset=["Explicit Track"])

    # 处理缺失值
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)

    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        mode_value = data[col].mode()[0]
        data[col] = data[col].fillna(mode_value)

    # 细化 Release Date 字段信息
    data['Release Date'] = pd.to_datetime(data['Release Date'], errors='coerce')
    data['Release Year'] = data['Release Date'].dt.year
    data['Release Month'] = data['Release Date'].dt.month
    data['Release Quarter'] = data['Release Date'].dt.quarter
    data['Release Day of Week'] = data['Release Date'].dt.dayofweek

    # 删除原始的 Release Date 字段
    data.drop(columns=['Release Date'], inplace=True)

    logging.info("Data preprocessing completed.")
    return data

# 定义几种异常值处理方法处理数值列

# 孤立森林
def iso_forest(data, contamination_value, columns_to_process):
    logging.info("Rejecting outliers with IsolationForest...")
    iso_forest = IsolationForest(contamination=contamination_value, random_state=42)  # contamination 参数控制异常值的比例
    outliers = iso_forest.fit_predict(data[columns_to_process])
    logging.info("Rejecting outliers completed.")
    return data[outliers == 1]

# Z-score
def z_score(data, threshold, columns_to_process):

    logging.info("Rejecting outliers with Z-Score...")

    # 创建一个空的 DataFrame 来存储异常值标签
    outlier_flags = pd.DataFrame(index=data.index)

    for col in columns_to_process:
        z_scores = zscore(data[col])  # 计算 Z-score
        outlier_flags[col] = np.abs(z_scores) > threshold  # 标记异常值

    combined_outlier_flags = outlier_flags.any(axis=1)

    logging.info("Rejecting outliers completed.")
    return data[~combined_outlier_flags]

# 对分类变量统一使用频率编码的特征工程
def feature_convert(data):
    logging.info("Performing feature convertion...")

    # 对分类变量使用频率编码
    categorical_columns = ["Artist", "Track", "Album Name", 'Release Year', 'Release Month', 'Release Quarter']
    for col in categorical_columns:
        freq = data[col].value_counts(normalize=True)
        data[f'{col}_freq'] = data[col].map(freq)
        data.drop(columns=[col], inplace=True)

    logging.info("Feature convertion completed.")
    return data

# 对发布时间信息单独使用独热编码的特征工程
def feature_convert_onehot(data):
    logging.info("Performing feature convertion...")

    month_encoder = OneHotEncoder(drop=None)
    quarter_encoder = OneHotEncoder(drop=None)

    # 对 Release Month 进行编码
    month_encoded = month_encoder.fit_transform(data[['Release Month']]).toarray()

    # 对 Release Quarter 进行编码
    quarter_encoded = quarter_encoder.fit_transform(data[['Release Quarter']]).toarray()

    # 将编码结果转换为 DataFrame
    month_encoded_df = pd.DataFrame(month_encoded, columns=[f'Month_{i}' for i in range(month_encoded.shape[1])])
    quarter_encoded_df = pd.DataFrame(quarter_encoded,
                                      columns=[f'Quarter_{i}' for i in range(quarter_encoded.shape[1])])

    # 合并到原始数据
    data.drop(columns=['Release Month', 'Release Quarter'], inplace=True)
    data = pd.concat([data, month_encoded_df, quarter_encoded_df], axis=1)

    # 对分类变量使用频率编码
    categorical_columns = ["Artist", "Track", "Album Name"]
    for col in categorical_columns:
        freq = data[col].value_counts(normalize=True)
        data[f'{col} Freq'] = data[col].map(freq)
        data.drop(columns=[col], inplace=True)

    logging.info("Performing feature convertion completed.")
    return data

# 训练前处理
def pre_operate(data):

    logging.info("Processing data before training...")

    # 划分特征和目标变量
    x = data.drop(columns=["Explicit Track"])
    y = data["Explicit Track"]

    # 删除包含 NaN 值的样本（使用独热编码大量数据时概率性丢失数据）
    x = x.dropna()
    y = y[x.index]  # 同时删除 y 中对应的样本

    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    # 标准化数值型特征
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 将标准化后的数据重新转换为 DataFrame，并恢复列名（后续使用）
    x_train = pd.DataFrame(x_train, columns=x.columns)
    x_test = pd.DataFrame(x_test, columns=x.columns)

    # 获取特征信息（可用于确定模型构建时的参数）
    print("Number of features after scaling: ", x_train.shape[1])
    print("Info of features:")
    print(data.info())

    logging.info("Processing data before training completed.")
    return x_train, x_test, y_train, y_test

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

# 额外尝试深度学习
def tensor_flow(x_train, y_train, x_test, y_test):
    logging.info("Training tensor flow classifier...")

    tf.random.set_seed(42)
    np.random.seed(42)

    # 定义模型构建函数
    def build_model(hp):
        model = Sequential([
            Input(shape=(x_train.shape[1],)),
            Dense(
                units=hp.Int('units_1', min_value=128, max_value=512, step=64),
                activation='relu'
            ),
            Dropout(
                rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
            ),
            Dense(
                units=hp.Int('units_2', min_value=64, max_value=256, step=32),
                activation='relu'
            ),
            Dropout(
                rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
            ),
            Dense(
                units=hp.Int('units_3', min_value=32, max_value=128, step=16),
                activation='relu'
            ),
            Dropout(
                rate=hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)
            ),
            Dense(1, activation='sigmoid')
        ])

        # 选择优化器并设置学习率
        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')  # 学习率范围
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        # 编译模型
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        logging.info(f"Built model: {model}")
        return model

    start_time = time.time()

    # 初始化随机搜索
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,  # 最大尝试次数
        executions_per_trial=3,  # 每次尝试的训练次数
        directory='keras_tuner',  # 保存调优结果的目录
        project_name='trial_for_tensorflow_training'  # 项目名称
    )

    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=10,         # 如果 10 个 epoch 后验证集损失没有改善，则停止训练
        restore_best_weights=True  # 恢复最佳权重
    )

    # 学习率调度器
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # 监控验证集损失
        factor=0.2,          # 学习率乘以 0.2
        patience=5,          # 如果 5 个 epoch 后验证集损失没有改善，则降低学习率
        min_lr=1e-6          # 最小学习率
    )

    # 搜索最佳超参数
    tuner.search(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=100,  # 增加搜索时的训练轮数
        batch_size=64,  # 批量大小
        callbacks=[early_stopping, reduce_lr],  # 添加早停和学习率调度器
        verbose=1
    )

    # 获取最佳模型
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 打印最佳超参数
    print("Best Parameters: %s", best_hyperparameters.values)

    # 使用最佳超参数重新训练模型
    history = best_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=200,  # 最终训练的轮数
        batch_size=64,  # 批量大小
        callbacks=[early_stopping, reduce_lr],  # 添加早停和学习率调度器
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"Training time of Tensor Flow model: {training_time:.2f} seconds.")

    # 评估模型
    loss, accuracy = best_model.evaluate(x_test, y_test)
    print(f"Test Set Loss: {loss:.4f}, Test Set Accuracy: {accuracy:.4f}")

    logging.info("Training tensor_flow Model completed.")
    return best_model

# 使用模型预测并评估模型准确性
def predict(model, x_test, y_test, model_name, visualization):
    """
            :param model: 训练好的模型
            :param model_name: 模型名称，用于生成提示和图像
            :param visualization: 是否生成特征重要性排序图
    """
    logging.info("Evaluating model " + model_name + "...")

    y_pred = model.predict(x_test)

    print("Accuracy of", model_name, ":", accuracy_score(y_test, y_pred))
    print("Classification Report of", model_name, ":\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of", model_name, ":\n", np.array2string(cm)[1:-1])

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'],  # 手动设置刻度标签
                yticklabels=['0', '1'])  # 手动设置刻度标签
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
        plt.figure(figsize=(10, 8))  # 增加图像高度
        plt.title("Feature Importances Sorted by " + model_name)
        plt.bar(range(x_test.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(x_test.shape[1]), [feature_names[i] for i in indices], rotation=90, fontsize=10)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.subplots_adjust(bottom=0.3)  # 增加底部边距
        plt.tight_layout()
        plt.show()

    logging.info("Evaluating model completed.")
    return accuracy_score(y_test, y_pred)

# 主函数
if __name__ == "__main__":

    filepath = "resources/most_streamed_spotify_songs_2024.csv"

    # 需要被清洗的数据列
    columns_for_process = ["All Time Rank", "Track Score", "Spotify Streams", "Spotify Playlist Count",
        "Spotify Playlist Reach", "Spotify Popularity", "YouTube Views", "YouTube Likes",
        "TikTok Posts", "TikTok Likes", "TikTok Views", "YouTube Playlist Reach",
        "Apple Music Playlist Count", "AirPlay Spins", "SiriusXM Spins", "Deezer Playlist Count",
        "Deezer Playlist Reach", "Amazon Playlist Count", "Pandora Streams", "Pandora Track Stations",
        "Soundcloud Streams", "Shazam Counts"]

    data = preprocess(filepath)
    data = iso_forest(data, 0.05, columns_to_process=columns_for_process)
    # data = z_score(data, 3, columns_for_process)
    data = feature_convert(data)
    if not os.path.exists('temp'):
        os.makedirs('temp')
    data.to_csv("temp/data_after_feature_engineering.csv", index=False)
    x_train, x_test, y_train, y_test = pre_operate(data)

    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }

    param_grid_nn = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500, 1000]
    }

    param_dist_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    param_dist_gb = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'subsample': uniform(0.8, 0.2),
        'max_features': ['sqrt', 'log2']
    }

    param_dist_nn = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': uniform(0.0001, 0.1),
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': uniform(0.0001, 0.01),
        'max_iter': randint(500, 1000)
    }

    # rf = train_model_with_search("random_forest", x_train, y_train)
    # predict(rf, x_test, y_test, "rf", True)
    #
    # rf_grid = train_model_with_search("random_forest", x_train, y_train, "grid", param_grid_rf)
    # predict(rf_grid, x_test, y_test, "rf_grid", True)
    #
    # rf_random = train_model_with_search("random_forest", x_train, y_train, "random", param_dist_rf)
    # predict(rf_random, x_test, y_test, "rf_random", True)
    #
    # gb = train_model_with_search("gradient_boost", x_train, y_train)
    # predict(gb, x_test, y_test, "gb", True)
    #
    # gb_grid = train_model_with_search("gradient_boost", x_train, y_train, "grid", param_grid_gb)
    # predict(gb_grid, x_test, y_test, "gb_grid", True)
    #
    # gb_random = train_model_with_search("gradient_boost", x_train, y_train, "random", param_dist_gb)
    # predict(gb_random, x_test, y_test, "gb_random", True)
    #
    # nn = train_model_with_search("neural_network", x_train, y_train)
    # predict(nn, x_test, y_test, "nn", False)
    #
    # nn_grid = train_model_with_search("neural_network", x_train, y_train, "grid", param_grid_nn)
    # predict(nn_grid, x_test, y_test, "nn_grid", False)
    #
    # nn_random = train_model_with_search("neural_network", x_train, y_train, "random", param_dist_nn)
    # predict(nn_random, x_test, y_test, "nn_random", False)

    tensor_flow(x_train, y_train, x_test, y_test)