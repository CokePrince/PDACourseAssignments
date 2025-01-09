"""
@Description: 期末大作业——预测歌曲是否有显式标记任务的数据集查看工具
@Author: 王宁远
@Date: 2025/01/09 16:12
@Version: 1.0.0
"""

# todo: 查看数据集概况

import pandas as pd

# 读取 CSV 文件
data = pd.read_csv("../resources/most_streamed_spotify_songs_2024.csv", encoding="ISO-8859-1")

data.info()

# 验证所有 TIDAL Popularity 有值的数据的 Explicit Track 均缺失
tidal_popularity_not_null = data[data['TIDAL Popularity'].notnull()]
explicit_track_missing = tidal_popularity_not_null['Explicit Track'].isnull().all()

if explicit_track_missing:
    print("All data not null in TIDAL Popularity missed Explicit Track.")