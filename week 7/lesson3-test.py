"""
@Description: 第七周课后作业
@Author: 王宁远
@Date: 2024/10/09 11:30
@Version: 1.0
"""
from math import factorial


# todo:
# 1. 输入一个整数n，输出从1到n的数字

def print_to_n(n):
    for i in range(1, n+1):
        print(i, end='\t')
    print('\n')

print_to_n(5)

# todo:
# 2. 请输入一个整数n，输出n!的值

def my_factorial(n):
    factorial = 1
    for i in range(1, n+1):
        factorial *= i
    return factorial

print(my_factorial(5))

# todo:
# 3. 请输入一个正整数n,计算m=1-2+3-4..+(-)n

def calculate_m(n):
    m = 0
    for i in range(1, n+1):
        if i % 2:
            m += i
        else:
            m -= i
    return m

print(calculate_m(5))

# todo:
# 4. 字符串统计，输入一个字符串，输出数字字符的个数

import re


def count_digits(str):
    # 使用正则表达式匹配所有的数字字符
    digits = re.findall(r'\d', str)
    return len(digits)

print(count_digits('123Hello world456'))

# todo:
# 5. 字符分类统计，输入一个字符串，分别统计字母、数字、下划线、其他字符的统计结果

def classify_char_in_str(str):
    count = {"alpha":0, "digit":0, "underline":0, "other_char":0}

    # 使用正则表达式查找所有字母
    count["alpha"] = len(re.findall(r'[a-zA-Z]', str))

    # 查找所有数字
    count["digit"] = len(re.findall(r'\d', str))

    # 查找所有下划线
    count["underline"] = len(re.findall(r'_', str))

    # 计算其他字符，可以先从总长度减去已知的字符数量
    total_length = len(str)
    count["other_char"] = total_length - (count["alpha"] + count["digit"] + count["underline"])

    return count

print(classify_char_in_str('123Hello_world!456'))

# todo:
# 6. 打印水仙花数
# 水仙花数是指一个3位数，它的每个数位上的数字的3次幂之和等于它本身。例如:1^3+5^3+3^3=153.

def find_narcissus(n):
    # 使用断言，当n不是三位数时程序报错
    assert n.is_integer() and 999 >= n >= 100
    for i in range(100, n+1):
        # 分别取个、十、百位
        if (i%10)**3 + ((i%100)//10)**3 + (i//100)**3 == i:
            print(i, end='\t')
    print('\n')

find_narcissus(999)