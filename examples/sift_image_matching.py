"""基于 SIFT + RANSAC 的图像特征匹配示例。

运行方式（两种模式）：

  # 模式 1：使用内置合成图像（默认，无需任何参数）
  python3 examples/sift_image_matching.py

  # 模式 2：使用用户提供的真实图像
  python3 examples/sift_image_matching.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg

  # 使用 FLANN 匹配器
  python3 examples/sift_image_matching.py --matcher flann

  # 使用 cross-check 匹配模式（适用于非重叠区域图像）
  python3 examples/sift_image_matching.py --match-mode crosscheck

  # 输出匹配结果图像
  python3 examples/sift_image_matching.py --draw-matches output.png

  # 使用基础矩阵进行 RANSAC 过滤
  python3 examples/sift_image_matching.py --ransac-method fundamental

本文件重点演示：
1. 用 OpenCV 构造两幅测试图像（原图 + 透视变换图），或读取用户提供的图像
2. 用 SIFT 检测关键点并提取描述子
3. 根据灰度阈值和半径过滤关键点（过滤掉靠近低灰度像素的特征点）
4. 用 BFMatcher 或 FLANN 匹配器对描述子进行匹配
5. 用 Lowe's ratio test 或 cross-check 模式筛选匹配
6. 用 RANSAC 估计单应矩阵或基础矩阵，过滤离群点（outliers）
7. 可选输出匹配结果可视化图像
8. 打印各阶段匹配数量，观察过滤效果

自动化测试：
- 单元测试：`tests/unit/test_sift_image_matching_unit.py`
- 功能测试：`tests/functional/test_sift_image_matching_functional.py`
- 建议在仓库根目录并激活 `conda asp360_new` 后运行：
    `python -m pytest tests/unit/test_sift_image_matching_unit.py tests/functional/test_sift_image_matching_functional.py -q`

修改记录:
- Gengxun, 2026-03-24: 增加 ratio-threshold 参数自定义功能，以及相应的功能测试和非法值测试。
- Gengxun, 2026-03-24: 增加 invalid-gray-threshold 和 invalid-radius 参数，用于过滤靠近低灰度像素的特征点。
- Gengxun, 2026-03-24: 增加 FLANN 匹配器、cross-check 模式、多种 RANSAC 方法、匹配结果可视化输出。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def load_image_as_gray(path: str) -> np.ndarray:
    """从文件加载图像并转换为灰度图。

    Parameters
    ----------
    path : str
        图像文件路径（支持 JPG、PNG、BMP 等 OpenCV 可读格式）。

    Returns
    -------
    gray : np.ndarray
        灰度图（dtype=uint8）。

    Raises
    ------
    SystemExit
        若文件不存在或无法读取，打印错误信息后退出程序。
    """
    img = cv2.imread(path)
    if img is None:
        print(f"错误：无法读取图像文件 '{path}'")
        print("请检查：")
        print("  1. 文件路径是否正确")
        print("  2. 文件是否为有效的图像格式（JPG / PNG / BMP 等）")
        sys.exit(1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def filter_keypoints_by_gray(
    img: np.ndarray,
    kp: list[cv2.KeyPoint],
    des: np.ndarray,
    invalid_gray_threshold: float = 10.0,
    invalid_radius: float = 5.0,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """根据灰度阈值和半径过滤关键点。

    过滤掉那些距离无效灰度值（低于阈值的像素）太近的特征点。

    Parameters
    ----------
    img : np.ndarray
        输入灰度图。
    kp : list[cv2.KeyPoint]
        待过滤的关键点列表。
    des : np.ndarray
        对应的描述子矩阵，形状 (N, 128)。
    invalid_gray_threshold : float
        灰度阈值，像素值小于此阈值时认为是无效值，默认 10.0。
    invalid_radius : float
        无效值半径（像素），特征点距离无效值小于此半径时将被过滤，默认 5.0。

    Returns
    -------
    filtered_kp : list[cv2.KeyPoint]
        过滤后的关键点列表。
    filtered_des : np.ndarray
        过滤后的描述子矩阵。
    """
    if des is None or len(kp) == 0:
        return kp, des

    # 创建无效像素掩码（灰度值 < 阈值的像素）
    invalid_mask = img < invalid_gray_threshold

    # 如果没有无效像素，直接返回原始关键点
    if not np.any(invalid_mask):
        return kp, des

    # 获取所有无效像素的坐标
    invalid_y, invalid_x = np.where(invalid_mask)
    invalid_coords = np.column_stack((invalid_x, invalid_y))

    # 过滤关键点
    filtered_kp = []
    filtered_indices = []

    for i, keypoint in enumerate(kp):
        x, y = keypoint.pt

        # 计算该关键点到所有无效像素的距离
        distances = np.sqrt(
            (invalid_coords[:, 0] - x) ** 2 + (invalid_coords[:, 1] - y) ** 2
        )

        # 如果最近距离大于等于半径，则保留该关键点
        if distances.min() >= invalid_radius:
            filtered_kp.append(keypoint)
            filtered_indices.append(i)

    # 过滤描述子
    if len(filtered_indices) > 0:
        filtered_des = des[filtered_indices]
    else:
        filtered_des = np.array([]).reshape(0, des.shape[1]) if des is not None else None

    return filtered_kp, filtered_des


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


def match_features_flann(
    des1: np.ndarray,
    des2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> tuple[int, list[cv2.DMatch]]:
    """用 FLANN 匹配器 + Lowe's ratio test 进行特征匹配。

    FLANN（Fast Library for Approximate Nearest Neighbors）使用 KD-Tree
    索引加速最近邻搜索，适合大规模特征匹配。

    Parameters
    ----------
    des1, des2 : np.ndarray
        两幅图像的描述子矩阵。
    ratio_threshold : float
        Lowe 比例测试的阈值，默认 0.75。

    Returns
    -------
    num_candidates : int
        FLANN knnMatch 返回的候选匹配对数。
    good_matches : list[cv2.DMatch]
        通过 ratio test 筛选后的良好匹配。
    """
    index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE = 1
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    all_pairs = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches: list[cv2.DMatch] = []
    for pair in all_pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return len(all_pairs), good_matches


def match_features_crosscheck(
    des1: np.ndarray,
    des2: np.ndarray,
) -> tuple[int, list[cv2.DMatch]]:
    """用 BFMatcher + cross-check 模式进行特征匹配。

    cross-check 模式不依赖 ratio test，适用于两幅图像可能不属于
    同一区域的场景。只有当 A 的最佳匹配是 B 且 B 的最佳匹配也是 A 时
    才保留该匹配。

    Parameters
    ----------
    des1, des2 : np.ndarray
        两幅图像的描述子矩阵。

    Returns
    -------
    num_candidates : int
        参与匹配的描述子数量（des1 的行数）。
    good_matches : list[cv2.DMatch]
        通过 cross-check 筛选后的匹配。
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    return des1.shape[0], list(matches)


def draw_match_image(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: str,
) -> None:
    """将匹配结果绘制到一幅拼接图像中并保存。

    Parameters
    ----------
    img1, img2 : np.ndarray
        两幅灰度图像。
    kp1, kp2 : list[cv2.KeyPoint]
        两幅图像的关键点列表。
    matches : list[cv2.DMatch]
        匹配结果列表。
    output_path : str
        输出图像文件路径。
    """
    result_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(output_path, result_img)
    print(f"匹配结果图像已保存到: {output_path}")


def filter_with_ransac(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    good_matches: list[cv2.DMatch],
    ransac_threshold: float = 5.0,
    method: str = "homography",
) -> tuple[np.ndarray | None, list[cv2.DMatch]]:
    """用 RANSAC 估计几何变换矩阵，过滤离群点。

    Parameters
    ----------
    kp1, kp2 : list[cv2.KeyPoint]
        两幅图像的关键点列表。
    good_matches : list[cv2.DMatch]
        通过 ratio test 或 cross-check 的良好匹配。
    ransac_threshold : float
        RANSAC 重投影误差阈值（像素），默认 5.0。
    method : str
        RANSAC 方法，可选 ``"homography"``（单应矩阵，默认）或
        ``"fundamental"``（基础矩阵）。

    Returns
    -------
    matrix : np.ndarray | None
        估计到的变换矩阵，若匹配点不足则为 None。
        homography 返回 3×3 单应矩阵，fundamental 返回 3×3 基础矩阵。
    inlier_matches : list[cv2.DMatch]
        RANSAC 认定为内点的匹配。
    """
    min_matches = 4 if method == "homography" else 8
    if len(good_matches) < min_matches:
        print(f"  警告：良好匹配点不足 {min_matches} 个，无法估计变换矩阵。")
        return None, []

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    if method == "fundamental":
        matrix, mask = cv2.findFundamentalMat(
            src_pts,
            dst_pts,
            cv2.FM_RANSAC,
            ransac_threshold,
        )
    else:
        matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransac_threshold,
        )

    if mask is None:
        return matrix, []

    inlier_matches = [
        m for m, flag in zip(good_matches, mask.ravel()) if flag
    ]
    return matrix, inlier_matches


def ratio_threshold_type(value: str) -> float:
    """解析并校验 Lowe ratio test 阈值。"""
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio_threshold 必须是浮点数。") from exc

    if not (0.0 < ratio < 1.0):
        raise argparse.ArgumentTypeError("ratio_threshold 必须在 0 到 1 之间。")

    return ratio

def ransac_threshold_type(value: str) -> float:
    """解析并校验 RANSAC 重投影误差阈值。
    Parameters:
    value : str
        输入的字符串值，应该能转换为正浮点数。
    Returns:
    float
        转换后的 RANSAC 阈值。
    Raises:
    argparse.ArgumentTypeError
        如果输入无法转换为正浮点数，或值不合法（<=0），则抛出异常。
    """
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ransac_threshold 必须是浮点数。") from exc

    if threshold <= 0.0:
        raise argparse.ArgumentTypeError("ransac_threshold 必须是正数。")

    return threshold


def invalid_gray_threshold_type(value: str) -> float:
    """解析并校验灰度阈值参数。

    Parameters
    ----------
    value : str
        命令行传入的灰度阈值字符串。

    Returns
    -------
    float
        转换并校验后的灰度阈值，要求位于 [0, 255] 区间内。

    Raises
    ------
    argparse.ArgumentTypeError
        当输入不是合法浮点数或超出允许范围时抛出。
    """
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid_gray_threshold 必须是浮点数。") from exc

    if not (0.0 <= threshold <= 255.0):
        raise argparse.ArgumentTypeError("invalid_gray_threshold 必须在 0 到 255 之间。")

    return threshold


def invalid_radius_type(value: str) -> float:
    """解析并校验无效值半径参数。

    Parameters
    ----------
    value : str
        命令行传入的半径字符串。

    Returns
    -------
    float
        转换并校验后的半径值，要求大于 0。

    Raises
    ------
    argparse.ArgumentTypeError
        当输入不是合法浮点数或小于等于 0 时抛出。
    """
    try:
        radius = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid_radius 必须是浮点数。") from exc

    if radius <= 0.0:
        raise argparse.ArgumentTypeError("invalid_radius 必须是正数。")

    return radius

# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "基于 SIFT + RANSAC 的图像特征匹配示例。\n\n"
            "不传任何参数时，使用内置合成图像演示全流程。\n"
            "同时指定 --img1 和 --img2 时，对用户提供的真实图像执行匹配。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--img1",
        metavar="路径",
        default=None,
        help="第一幅图像的文件路径（query 图）",
    )
    parser.add_argument(
        "--img2",
        metavar="路径",
        default=None,
        help="第二幅图像的文件路径（train 图）",
    )
    parser.add_argument(
        "--ratio-threshold",
        metavar="浮点数",
        type=ratio_threshold_type,
        default=0.75,
        help="Lowe ratio test 的阈值，必须在 0 到 1 之间，默认 0.75",
    )
    parser.add_argument(
        "--ransac-threshold",
        metavar="浮点数",
        type=ransac_threshold_type,
        default=5.0,
        help="RANSAC 重投影误差阈值（像素），默认 5.0",
    )
    parser.add_argument(
        "--invalid-gray-threshold",
        metavar="浮点数",
        type=invalid_gray_threshold_type,
        default=10.0,
        help="灰度阈值，像素值小于此阈值时认为是无效值（0-255），默认 10.0",
    )
    parser.add_argument(
        "--invalid-radius",
        metavar="浮点数",
        type=invalid_radius_type,
        default=5.0,
        help="无效值半径（像素），特征点距离无效值小于此半径时将被过滤，默认 5.0",
    )
    parser.add_argument(
        "--matcher",
        choices=["bf", "flann"],
        default="bf",
        help="匹配器类型：bf（BFMatcher，默认）或 flann（FLANN 匹配器）",
    )
    parser.add_argument(
        "--match-mode",
        choices=["ratio", "crosscheck"],
        default="ratio",
        help="匹配模式：ratio（Lowe ratio test，默认）或 crosscheck（交叉验证，适用于非重叠区域图像）",
    )
    parser.add_argument(
        "--ransac-method",
        choices=["homography", "fundamental"],
        default="homography",
        help="RANSAC 方法：homography（单应矩阵，默认）或 fundamental（基础矩阵）",
    )
    parser.add_argument(
        "--draw-matches",
        metavar="路径",
        default=None,
        help="将匹配结果可视化并保存到指定文件路径（如 output.png）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    use_real_images = args.img1 is not None or args.img2 is not None

    # 若只提供了一张图，报错退出
    if (args.img1 is None) != (args.img2 is None):
        print("错误：--img1 和 --img2 必须同时提供，或都不提供。")
        print("示例：python3 examples/sift_image_matching.py --img1 a.jpg --img2 b.jpg")
        sys.exit(1)

    # ── 步骤 1：准备图像 ────────────────────────────────────────
    print("=" * 60)
    if use_real_images:
        print("1) 加载用户提供的真实图像")
        print("=" * 60)
        img1 = load_image_as_gray(args.img1)
        img2 = load_image_as_gray(args.img2)
        print(f"图像 1: {args.img1}  尺寸: {img1.shape}")
        print(f"图像 2: {args.img2}  尺寸: {img2.shape}")
        H_true = None  # 真实图像无已知单应矩阵
    else:
        print("1) 生成合成测试图像（未传入 --img1/--img2，使用默认演示）")
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
    print(f"图像 1 关键点数量（检测后）: {len(kp1)}")
    print(f"图像 2 关键点数量（检测后）: {len(kp2)}")

    if des1 is None or des2 is None:
        print("错误：SIFT 未能提取到描述子，图像可能过于简单或分辨率过低。")
        sys.exit(1)

    print(f"描述子维度: {des1.shape[1]}")

    # 灰度过滤步骤
    print("\n" + "=" * 60)
    print("3) 根据灰度阈值过滤关键点")
    print("=" * 60)
    print(f"invalid_gray_threshold: {args.invalid_gray_threshold}")
    print(f"invalid_radius: {args.invalid_radius}")
    kp1, des1 = filter_keypoints_by_gray(
        img1, kp1, des1, args.invalid_gray_threshold, args.invalid_radius
    )
    kp2, des2 = filter_keypoints_by_gray(
        img2, kp2, des2, args.invalid_gray_threshold, args.invalid_radius
    )
    print(f"图像 1 关键点数量（过滤后）: {len(kp1)}")
    print(f"图像 2 关键点数量（过滤后）: {len(kp2)}")

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("错误：灰度过滤后没有剩余关键点，请尝试调整 invalid_gray_threshold 或 invalid_radius。")
        sys.exit(1)

    print("\n" + "=" * 60)
    match_mode = args.match_mode
    matcher_type = args.matcher
    if match_mode == "crosscheck":
        print("4) BFMatcher 交叉验证匹配（cross-check 模式）")
        print("=" * 60)
        all_matches, good_matches = match_features_crosscheck(des1, des2)
        print(f"参与匹配的描述子数量: {all_matches}")
        print(f"通过 cross-check 的匹配数: {len(good_matches)} 对")
    elif matcher_type == "flann":
        print("4) FLANN 匹配器 + Lowe's ratio test")
        print("=" * 60)
        print(f"ratio_threshold: {args.ratio_threshold}")
        all_matches, good_matches = match_features_flann(
            des1, des2, ratio_threshold=args.ratio_threshold
        )
        print(f"原始候选匹配数 (knnMatch k=2): {all_matches} 对")
        print(f"通过 ratio test 的良好匹配数:    {len(good_matches)} 对")
    else:
        print("4) BFMatcher 暴力匹配 + Lowe's ratio test")
        print("=" * 60)
        print(f"ratio_threshold: {args.ratio_threshold}")
        all_matches, good_matches = match_features(
            des1, des2, ratio_threshold=args.ratio_threshold
        )
        print(f"原始候选匹配数 (knnMatch k=2): {all_matches} 对")
        print(f"通过 ratio test 的良好匹配数:    {len(good_matches)} 对")

    print("\n" + "=" * 60)
    ransac_method = args.ransac_method
    method_label = "单应矩阵" if ransac_method == "homography" else "基础矩阵"
    print(f"5) RANSAC 过滤离群点（{method_label}）")
    print("=" * 60)
    H_est, inlier_matches = filter_with_ransac(
        kp1, kp2, good_matches,
        ransac_threshold=args.ransac_threshold,
        method=ransac_method,
    )
    if H_est is not None:
        print(f"RANSAC 内点匹配数:  {len(inlier_matches)} 对")
        print(f"RANSAC 过滤离群点:  {len(good_matches) - len(inlier_matches)} 对")
        print(f"估计的{method_label} ({ransac_method}):")
        print(np.round(H_est, 4))

        # 仅在合成图像 + homography 模式下才有真实单应矩阵可供对比
        if H_true is not None and ransac_method == "homography":
            diff = np.abs(H_true - H_est)
            print(f"\nH_true 与 H_est 的最大绝对差: {diff.max():.4f}")
    else:
        print(f"{method_label}估计失败。")

    # 绘制匹配结果图像
    if args.draw_matches is not None:
        print("\n" + "=" * 60)
        print("*) 绘制匹配结果图像")
        print("=" * 60)
        matches_to_draw = inlier_matches if inlier_matches else good_matches
        draw_match_image(img1, img2, kp1, kp2, matches_to_draw, args.draw_matches)

    print("\n" + "=" * 60)
    print("6) 小结")
    print("=" * 60)
    print(f"  匹配器: {matcher_type}")
    print(f"  匹配模式: {match_mode}")
    print(f"  RANSAC 方法: {ransac_method}")
    print(f"  关键点（img1 / img2）: {len(kp1)} / {len(kp2)}")
    print(f"  匹配后良好匹配: {len(good_matches)}")
    print(f"  RANSAC 后内点匹配:     {len(inlier_matches)}")
    inlier_rate = len(inlier_matches) / len(good_matches) * 100 if good_matches else 0
    print(f"  内点率:                {inlier_rate:.1f}%")
    print()
    print("你可以继续练习：")
  
if __name__ == "__main__":
    main()
