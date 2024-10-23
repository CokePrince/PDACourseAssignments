"""
@Description: 第八周课后作业
@Author: 王宁远
@Date: 2024/10/23 11:36
@Version: 1.0
"""

# todo:
# 3. 编写一个自己经常用到的功能函数
# 判断输入是否为自然数，因为这一函数将被用在任务1和任务2中，故提前定义

def is_natural(n):

    try:
        n = int(n)
    # int() 方法自带了异常抛出，对于非整型输入将会抛出异常，利用此特性可判断输入是否为整数
    # 对非整型输入的异常，这里进行捕获并将异常信息转换为中文
    except ValueError:
        raise ValueError("输入须是一个整数")

    if n >= 0:
        return
    else:
        # 如果输入小于0，则抛出异常
        raise ValueError("输入须大于等于0")

# todo:
# 1. 判断一个数是不是质数

def is_prime(n):

    try:
        is_natural(n)
    except ValueError as e:
        raise e

    is_natural(n)
    # 若在上一语句没有异常抛出，说明输入确为自然数，可以安全地进行转换
    n = int(n)

    # 检查数字是否小于2，因为质数定义为大于1的自然数
    if n < 2:
        return False

    # 检查从2到n/2是否有能整除n的数
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


# 使用示例
try:
    number = input("请输入一个整数：")
    if is_prime(number):
        print(f"{number} 是质数。")
    else:
        print(f"{number} 不是质数。")
except ValueError as e:
    print(e)

# todo:
# 2. 判断一年是不是闰年

def is_leap(year):

    is_natural(year)

    year = int(year)

    # 亦可使用 calendar.isleap() 方法判断是否为闰年
    """
    import calendar
    if calendar.isleap(year):
        return True
    else:
        return False
    """

    # 闰年定义为年份能被4整除但不能被100整除，或能被400整除的年
    if year%4 == 0 and year%100 != 0 or year%400 == 0:
        return True

    return False

# 使用示例
try:
    year = input("请输入一个年份：")
    if is_leap(year):
        print(f"{year} 是闰年。")
    else:
        print(f"{year} 不是闰年。")
except ValueError as e:
    print(e)
