"""NumPy 入门示例。

运行方式：
    python3 examples/numpy_intro_example.py

适合刚学 Python、准备进入科研/数据处理方向的学习者。
本文件重点演示：
1. 创建数组
2. 查看形状和数据类型
3. 切片与索引
4. 向量化运算
5. 布尔筛选
6. 常用统计
7. reshape 变形
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 60)
    print("1) 创建数组")
    print("=" * 60)
    temperatures = np.array([18.5, 20.0, 21.3, 19.8, 22.1, 23.0])
    print("temperatures =", temperatures)

    matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    print("\nmatrix =")
    print(matrix)

    print("\n" + "=" * 60)
    print("2) 数组属性")
    print("=" * 60)
    print("shape:", matrix.shape)
    print("ndim:", matrix.ndim)
    print("dtype:", matrix.dtype)
    print("size:", matrix.size)

    print("\n" + "=" * 60)
    print("3) 索引与切片")
    print("=" * 60)
    print("第一行:", matrix[0])
    print("第二列:", matrix[:, 1])
    print("右下角 2x2 子矩阵:\n", matrix[1:, 1:])
    print("左上角 2x2 子矩阵:\n", matrix[:2, :2])

    print("\n" + "=" * 60)
    print("4) 向量化运算")
    print("=" * 60)
    print("原始温度:", temperatures)
    print("全部加 1.5 度:", temperatures + 1.5)
    print("全部乘 2:", temperatures * 2)

    print("\n说明：这里没有写 for 循环，而是一次性对整个数组运算。")

    print("\n" + "=" * 60)
    print("5) 布尔筛选")
    print("=" * 60)
    hot_days = temperatures[temperatures >= 21.0]
    print("温度 >= 21.0 的数据:", hot_days)

    print("\n" + "=" * 60)
    print("6) 常用统计")
    print("=" * 60)
    print(f"平均值: {temperatures.mean():.2f}")
    print(f"最大值: {temperatures.max():.2f}")
    print(f"最小值: {temperatures.min():.2f}")
    print(f"标准差: {temperatures.std():.2f}")

    print("\n" + "=" * 60)
    print("7) reshape 变形")
    print("=" * 60)
    values = np.arange(1, 13)
    print("一维数组:", values)
    reshaped = values.reshape(3, 4)
    print("reshape 成 3x4 后:\n", reshaped)

    print("\n" + "=" * 60)
    print("8) 一个小练习思路")
    print("=" * 60)
    print("你可以尝试自己改写：")
    print("- 求每一行的和: reshaped.sum(axis=1)")
    print("- 求每一列的平均值: reshaped.mean(axis=0)")
    print("- 找出所有偶数: reshaped[reshaped % 2 == 0]")


if __name__ == "__main__":
    main()
