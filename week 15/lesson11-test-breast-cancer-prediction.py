"""
@Description: 第十五周课后作业——通过乳腺癌数据集比较三种模型的精确度（逻辑回归、随机森林、KNN）
@Author: 王宁远
@Date: 2024/12/10 16:49
@Version: 1.0
"""

# todo:

# 通过乳腺癌数据集比较三种模型的精确度（逻辑回归、随机森林、KNN）

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
columns = [
    "Sample code number",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "Class",
]

# Step 2: Preprocessing
data = pd.read_csv(url, header=None, names=columns, na_values='?')
data = data.dropna()
X = data.iloc[:, 1:-1]
y = data['Class']
y = y.apply(lambda x: 0 if x == 2 else 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'K Nearest Neighbors': KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")

# Compare accuracies
print("Model Comparison based on Accuracy:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")
