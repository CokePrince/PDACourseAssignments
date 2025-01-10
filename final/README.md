# README

本项目为课程《Python基础与互联网编程》（实际上应为《机器学习》）的最终作业。

本项目采用`Python 3.11.9`作为开发环境，未使用`3.12`及以上版本的原因是受到核心依赖库`TensorFlow 2.18.0`（已为最新稳定版）的兼容性限制，后者支持的Python版本范围为`3.7~3.11`。

项目依赖的外部Python软件包如下：

| 软件包名称           | 引用名 | 版本 |
|----------------------|--------|------------|
| imblearn             | -      | 0.13.0     |
| keras                | -      | 3.8.0      |
| keras-tuner          | kt     | 1.4.7      |
| matplotlib           | plt    | 3.10.0     |
| numpy                | np     | 2.0.2      |
| pandas               | pd     | 22.3       |
| scipy                | -      | 1.15.0     |
| seaborn              | sns    | 0.13.2     |
| sklearn              | -      | 1.6.0      |
| tensorflow           | tf     | 2.18.0     |

实际上，机器学习任务可由GPU加速。但为了保证兼容性，确保代码能够在不同终端顺利运行，本项目全程仅用CPU运算。本文中的实验数据均在`Intel Core i7-12700H`处理器环境下获得。

本项目所使用的随机数指定由`42`号随机种子生成，即设置`random_state=42`，保持此设置可以获得与本文相同的模型表现（部分模型除外）。事实上，只要随机种子相同，就可以保证每次生成的随机数相同，无论是42或其他整数。但使用42是机器学习的惯例。这里有一个有趣的典故，在科幻小说《银河系漫游指南》（The Hitchhiker's Guide to the Galaxy）中，超级计算机“深思”（Deep Thought）被问到“生命、宇宙以及万物的终极答案”时，经过长时间计算后给出的答案是42。

读者可以从以下链接获得项目文件：

https://github.com/CokePrince/PDACourseAssignments/tree/main/final

项目包含两个任务，即`构建用于预测性维护的机器学习模型`和`构建用于预测预测歌曲是否有显式标记的机器学习模型`。

建议为两个任务各自分配单独的工作区。两个任务文件夹根目录下的`main.py`是完成任务的完整代码，其它为工具代码。在运行任务代码前，工作区文件树应分别如下：

```
most_streamed_spotify_songs/
├── main.py
├── resources/
│   └── most_streamed_spotify_songs_2024.csv
└── utils/
    ├── data_overview.py
    └── encoding_detection.py
```

```
predictive_maintenance/
├── main.py
├── resources/
│   └── predictive_maintenance.csv
└── utils/
    └── data_check.py
```

代码运行后，工作区可能有新文件产生。
