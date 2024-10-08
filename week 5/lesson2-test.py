# ### 1. 列表（List）练习
#
# 1. 创建一个包含数字 1 到 5 的列表，并打印列表。
# 2. 访问并打印列表中的第三个元素。
# 3. 将列表中的第四个元素改为 10，并打印整个列表。
# 4. 将列表中的第一个到第三个元素切片出来并打印。
# 5. 向列表末尾添加数字 6，并打印整个列表。
# 6. 从列表中删除数字 2，并打印整个列表。
# 7. 打印列表的长度。
# 8. 检查数字 5 是否在列表中，结果为 True 或 False。
#
# ### 2. 元组（Tuple）练习
#
# 1. 创建一个包含字符串 "apple", "banana", "cherry" 的元组，并打印元组。
# 2. 访问并打印元组中的第二个元素。
# 3. 尝试修改元组中的第一个元素为 "orange"，观察会发生什么。
# 4. 将元组切片，提取前两个元素并打印。
# 5. 创建一个只有一个元素的元组（例如 "apple"），并打印。
# 6. 打印元组的长度。
#
# ### 3. 字典（Dictionary）练习
#
# 1. 创建一个包含姓名和年龄的字典，例如 {"name": "Tom", "age": 25}，并打印字典。
# 2. 访问并打印字典中的 "name" 对应的值。
# 3. 将字典中的 "age" 值改为 30，并打印整个字典。
# 4. 向字典中添加一个新的键值对 "gender": "male"，并打印整个字典。
# 5. 从字典中删除键 "age" 对应的项，并打印整个字典。
# 6. 打印字典中的所有键。
# 7. 打印字典中的所有值。
# 8. 检查字典中是否存在键 "name"，结果为 True 或 False。

"""
@Description: 第五周课后作业
@Author: 王宁远
@Date: 2024/10/08 21:42
@Version: 1.0
"""

# todo:
# ### 1. 列表（List）练习
# 1. 创建一个包含数字 1 到 5 的列表，并打印列表。
numbers = [1, 2, 3, 4, 5]
print(numbers)

# 2. 访问并打印列表中的第三个元素。（索引从0开始）
print(numbers[2])

# 3. 将列表中的第四个元素改为 10，并打印整个列表。
numbers[3] = 10
print(numbers)

# 4. 将列表中的第一个到第三个元素切片出来并打印。
sliced_numbers = numbers[0:3]
print(sliced_numbers)

# 5. 向列表末尾添加数字 6，并打印整个列表。
numbers.append(6)
print(numbers)

# 6. 从列表中删除数字 2，并打印整个列表。
numbers.remove(2)
print(numbers)

# 7. 打印列表的长度。
print(len(numbers))

# 8. 检查数字 5 是否在列表中，结果为 True 或 False。
print(5 in numbers)

# ### 2. 元组（Tuple）练习
# 1. 创建一个包含字符串 "apple", "banana", "cherry" 的元组，并打印元组。
fruits = ("apple", "banana", "cherry")
print(fruits)

# 2. 访问并打印元组中的第二个元素。（索引从0开始）
print(fruits[1])

# 3. 尝试修改元组中的第一个元素为 "orange"，观察会发生什么。
# fruits[0] = "orange"  # 这行代码会抛出错误，因为元组是不可变的

# 4. 将元组切片，提取前两个元素并打印。
sliced_fruits = fruits[:2]
print(sliced_fruits)

# 5. 创建一个只有一个元素的元组（例如 "apple"），并打印。
single_fruit = ("apple",)  # 逗号表示这是一个元组而不是字符串
print(single_fruit)

# 6. 打印元组的长度。
print(len(fruits))

# ### 3. 字典（Dictionary）练习
# 1. 创建一个包含姓名和年龄的字典，例如 {"name": "Tom", "age": 25}，并打印字典。
person = {"name": "John", "age": 20}
print(person)

# 2. 访问并打印字典中的 "name" 对应的值。
print(person["name"])

# 3. 将字典中的 "age" 值改为 30，并打印整个字典。
person["age"] = 30
print(person)

# 4. 向字典中添加一个新的键值对 "gender": "male"，并打印整个字典。
person["gender"] = "male"
print(person)

# 5. 从字典中删除键 "age" 对应的项，并打印整个字典。
del person["age"]
print(person)

# 6. 打印字典中的所有键。
print(person.keys())

# 7. 打印字典中的所有值。
print(person.values())

# 8. 检查字典中是否存在键 "name"，结果为 True 或 False。
print("name" in person)