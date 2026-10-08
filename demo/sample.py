"""演示用的“问题代码”：包含常见坏味道，供 Code Agent 审查。

故意埋入的问题：
- 裸异常捕获
- 未使用导入
- 缺少类型标注与 docstring
- TODO 遗留
- 潜在除零与可变默认参数
- 不必要的重复计算
"""

import json  # 未使用的导入
import time


def calculate_average(nums=[]):  # 可变默认参数 + 无类型标注
    total = 0
    for n in nums:
        total = total + n
    return total / len(nums)  # 空列表会 ZeroDivisionError


def load_config(path):
    f = open("config.json")  # 未使用 path 参数，且未关闭文件句柄
    return json.load(f)


def process(records):
    result = []
    for r in records:
        try:
            result.append(calculate_average(r["values"]))
        except:  # 裸异常：吞掉所有错误
            pass
    return result


# TODO: 这里性能有问题，重复排序了三次，需要优化
def top_three(items):
    a = sorted(items)
    b = sorted(items)
    c = sorted(items)
    return [a[-1], b[-2], c[-3]]


def debug_print(msg):
    print("debug:", msg)  # 调试 print 残留


if __name__ == "__main__":
    print(top_three([3, 1, 2, 5, 4]))
    time.sleep(0)
