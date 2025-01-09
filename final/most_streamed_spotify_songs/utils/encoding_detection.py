"""
@Description: 期末大作业——预测歌曲是否有显式标记任务的数据集编码格式确认工具
@Author: 王宁远
@Date: 2025/01/07 15:15
@Version: 1.0.0
"""

# todo: 确认 most_streamed_spotify_songs_2024.csv 的编码格式

import chardet
import pandas as pd

# 打开文件并读取内容
with open("../resources/most_streamed_spotify_songs_2024.csv", 'rb') as f:
    raw_data = f.read()

# 检测文件编码
result = chardet.detect(raw_data)

# 打印检测结果
print(f"Encoding detected: {result['encoding']}")
print(f"Confidence: {result['confidence']}")

# 使用检测到的编码读取文件
if result['encoding'] is not None:
    encoding = result['encoding']
    try:
        data = pd.read_csv("../resources/most_streamed_spotify_songs_2024.csv", encoding=encoding)
        print("Successfully read the file.")
    except Exception as e:
        print(f"Error occurred when reading the file: {e}")
else:
    print("Failed to detect the encoding.")