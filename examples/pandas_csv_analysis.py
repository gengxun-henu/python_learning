"""pandas 读取 CSV 并做基础分析。

运行方式：
    python3 examples/pandas_csv_analysis.py

本文件重点演示：
1. 读取 CSV
2. 查看表结构
3. 新增一列
4. 条件筛选
5. 排序
6. 分组统计
7. 保存分析结果
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).with_name("sample_scores.csv")
OUTPUT_FILE = Path(__file__).with_name("sample_scores_with_total.csv")


def main() -> None:
    print("=" * 60)
    print("1) 读取 CSV")
    print("=" * 60)
    df = pd.read_csv(DATA_FILE)
    print(df)

    print("\n" + "=" * 60)
    print("2) 查看基础信息")
    print("=" * 60)
    print("数据形状:", df.shape)
    print("列名:", list(df.columns))
    print("\n前 3 行:")
    print(df.head(3))

    print("\n" + "=" * 60)
    print("3) 新增总分列")
    print("=" * 60)
    df["total"] = df["math"] + df["english"] + df["python"]
    print(df[["student", "total"]])

    print("\n" + "=" * 60)
    print("4) 条件筛选")
    print("=" * 60)
    high_math = df[df["math"] >= 85]
    print("数学 >= 85 分的学生:")
    print(high_math[["student", "class_name", "math"]])

    print("\n" + "=" * 60)
    print("5) 排序")
    print("=" * 60)
    sorted_df = df.sort_values(by="total", ascending=False)
    print("按总分从高到低排序:")
    print(sorted_df[["student", "total"]])

    print("\n" + "=" * 60)
    print("6) 分组统计")
    print("=" * 60)
    class_summary = df.groupby("class_name")[["math", "english", "python", "total"]].mean()
    print("按班级求平均分:")
    print(class_summary.round(2))

    print("\n" + "=" * 60)
    print("7) 找出总分最高的学生")
    print("=" * 60)
    top_student = df.loc[df["total"].idxmax()]
    print(
        f"最高分学生: {top_student['student']}，"
        f"班级: {top_student['class_name']}，"
        f"总分: {top_student['total']}"
    )

    print("\n" + "=" * 60)
    print("8) 保存结果")
    print("=" * 60)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"已保存带 total 列的新文件到: {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("9) 你可以继续练习")
    print("=" * 60)
    print("- 找出缺勤天数 > 2 的学生")
    print("- 按 math 单列排序")
    print("- 增加 average 平均分列")
    print("- 把班级平均分保存为新的 CSV")


if __name__ == "__main__":
    main()
