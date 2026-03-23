"""基于 SIFT + RANSAC 的图像特征匹配示例。

运行方式：
    python3 examples/sift_image_matching.py

本文件重点演示：
1. 用 OpenCV 构造两幅测试图像（原图 + 透视变换图）
2. 用 SIFT 检测关键点并提取描述子
3. 用 BFMatcher（暴力匹配）对描述子进行匹配
4. 用 Lowe's ratio test 筛选良好匹配
5. 用 RANSAC 估计单应矩阵，过滤离群点（outliers）
6. 打印各阶段匹配数量，观察 RANSAC 的过滤效果
"""

from __future__ import annotations

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def make_test_images(
    size: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成一对测试图像：原图和透视变换后的图像。

    为了让示例不依赖外部文件，图像由代码生成：
    - 灰度背景 + 若干随机几何图形（圆、矩形、线段）
    - 透视变换模拟真实场景中的视角变化

    Returns
    -------
    img1 : np.ndarray
        原始灰度图（query 图）。
    img2 : np.ndarray
        经过透视变换的灰度图（train 图）。
    H_true : np.ndarray
        生成 img2 时所用的真实单应矩阵（3×3）。
    """
    rng = np.random.default_rng(42)

    # 底色
    canvas = np.full((size, size), 30, dtype=np.uint8)

    # 绘制若干随机圆形
    for _ in range(20):
        cx = int(rng.integers(20, size - 20))
        cy = int(rng.integers(20, size - 20))
        radius = int(rng.integers(5, 30))
        color = int(rng.integers(80, 240))
        cv2.circle(canvas, (cx, cy), radius, color, -1)

    # 绘制若干随机矩形
    for _ in range(10):
        x1 = int(rng.integers(0, size - 40))
        y1 = int(rng.integers(0, size - 40))
        x2 = x1 + int(rng.integers(20, 80))
        y2 = y1 + int(rng.integers(20, 80))
        color = int(rng.integers(80, 240))
        cv2.rectangle(canvas, (x1, y1), (min(x2, size - 1), min(y2, size - 1)), color, 2)

    # 绘制若干随机线段
    for _ in range(15):
        pt1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        pt2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        color = int(rng.integers(80, 240))
        cv2.line(canvas, pt1, pt2, color, 2)

    img1 = canvas.copy()

    # 透视变换：模拟轻微视角倾斜
    src_pts = np.float32(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
    )
    dst_pts = np.float32(
        [
            [20, 15],
            [size - 25, 10],
            [size - 15, size - 20],
            [10, size - 10],
        ]
    )
    H_true = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img2 = cv2.warpPerspective(img1, H_true, (size, size))

    return img1, img2, H_true


def detect_and_compute(
    img: np.ndarray,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """用 SIFT 检测关键点并计算描述子。

    Parameters
    ----------
    img : np.ndarray
        输入灰度图。

    Returns
    -------
    kp : list[cv2.KeyPoint]
        检测到的关键点列表。
    des : np.ndarray
        对应的描述子矩阵，形状 (N, 128)。
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> tuple[int, list[cv2.DMatch]]:
    """用 BFMatcher + Lowe's ratio test 进行特征匹配。

    Parameters
    ----------
    des1, des2 : np.ndarray
        两幅图像的描述子矩阵。
    ratio_threshold : float
        Lowe 比例测试的阈值，默认 0.75。

    Returns
    -------
    num_candidates : int
        BFMatcher knnMatch 返回的候选匹配对数（query 关键点数量）。
    good_matches : list[cv2.DMatch]
        通过 ratio test 筛选后的良好匹配。
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    all_pairs = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches: list[cv2.DMatch] = []
    for pair in all_pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return len(all_pairs), good_matches


def filter_with_ransac(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    ransac_threshold: float = 5.0,
) -> tuple[np.ndarray | None, list[cv2.DMatch]]:
    """用 RANSAC 估计单应矩阵，过滤离群点。

    Parameters
    ----------
    kp1, kp2 : list[cv2.KeyPoint]
        两幅图像的关键点列表。
    good_matches : list[cv2.DMatch]
        通过 ratio test 的良好匹配。
    ransac_threshold : float
        RANSAC 重投影误差阈值（像素），默认 5.0。

    Returns
    -------
    H : np.ndarray | None
        估计到的单应矩阵（3×3），若匹配点不足则为 None。
    inlier_matches : list[cv2.DMatch]
        RANSAC 认定为内点的匹配。
    """
    if len(good_matches) < 4:
        print("  警告：良好匹配点不足 4 个，无法估计单应矩阵。")
        return None, []

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_threshold,
    )

    if mask is None:
        return H, []

    inlier_matches = [
        m for m, flag in zip(good_matches, mask.ravel()) if flag
    ]
    return H, inlier_matches


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("1) 生成测试图像")
    print("=" * 60)
    img1, img2, H_true = make_test_images(size=400)
    print(f"原图尺寸: {img1.shape}")
    print(f"变换图尺寸: {img2.shape}")
    print("真实单应矩阵 H_true:")
    print(np.round(H_true, 4))

    print("\n" + "=" * 60)
    print("2) SIFT 特征检测与描述子提取")
    print("=" * 60)
    kp1, des1 = detect_and_compute(img1)
    kp2, des2 = detect_and_compute(img2)
    print(f"图像 1 关键点数量: {len(kp1)}")
    print(f"图像 2 关键点数量: {len(kp2)}")
    print(f"描述子维度: {des1.shape[1]}")

    print("\n" + "=" * 60)
    print("3) BFMatcher 暴力匹配 + Lowe's ratio test")
    print("=" * 60)
    all_matches, good_matches = match_features(des1, des2, ratio_threshold=0.75)
    print(f"原始候选匹配数 (knnMatch k=2): {all_matches} 对")
    print(f"通过 ratio test 的良好匹配数:    {len(good_matches)} 对")

    print("\n" + "=" * 60)
    print("4) RANSAC 过滤离群点")
    print("=" * 60)
    H_est, inlier_matches = filter_with_ransac(
        kp1, kp2, good_matches, ransac_threshold=5.0
    )
    if H_est is not None:
        print(f"RANSAC 内点匹配数:  {len(inlier_matches)} 对")
        print(f"RANSAC 过滤离群点:  {len(good_matches) - len(inlier_matches)} 对")
        print("估计的单应矩阵 H_est:")
        print(np.round(H_est, 4))

        # 评估估计误差
        diff = np.abs(H_true - H_est)
        print(f"\nH_true 与 H_est 的最大绝对差: {diff.max():.4f}")
    else:
        print("单应矩阵估计失败。")

    print("\n" + "=" * 60)
    print("5) 小结")
    print("=" * 60)
    print(f"  关键点（img1 / img2）: {len(kp1)} / {len(kp2)}")
    print(f"  ratio test 后良好匹配: {len(good_matches)}")
    print(f"  RANSAC 后内点匹配:     {len(inlier_matches)}")
    inlier_rate = len(inlier_matches) / len(good_matches) * 100 if good_matches else 0
    print(f"  内点率:                {inlier_rate:.1f}%")
    print()
    print("你可以继续练习：")
    print("  - 修改 ratio_threshold，观察匹配数量变化")
    print("  - 修改 ransac_threshold，观察内点率变化")
    print("  - 替换为真实图像对（同一场景不同角度/光照）")
    print("  - 将结果可视化: cv2.drawMatches(...) 保存为 PNG")


if __name__ == "__main__":
    main()
