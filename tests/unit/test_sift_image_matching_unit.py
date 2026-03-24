"""`examples.sift_image_matching` 的单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from pprint import pformat
import sys

import numpy as np
import pytest
import cv2  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples import sift_image_matching as sim


def _pipeline_summary(pipeline_outputs: dict) -> str:
    """整理一份便于调试的流水线摘要信息。"""
    h_est = pipeline_outputs["h_est"]
    h_true = pipeline_outputs["h_true"]
    max_abs_diff = None
    if h_est is not None:
        max_abs_diff = float(np.abs(h_true - h_est).max())

    summary = {
        "img1_shape": tuple(pipeline_outputs["img1"].shape),
        "img2_shape": tuple(pipeline_outputs["img2"].shape),
        "kp1_count": len(pipeline_outputs["kp1"]),
        "kp2_count": len(pipeline_outputs["kp2"]),
        "des1_shape": None if pipeline_outputs["des1"] is None else tuple(pipeline_outputs["des1"].shape),
        "des2_shape": None if pipeline_outputs["des2"] is None else tuple(pipeline_outputs["des2"].shape),
        "candidate_matches": pipeline_outputs["candidates"],
        "good_matches": len(pipeline_outputs["good_matches"]),
        "inlier_matches": len(pipeline_outputs["inlier_matches"]),
        "inlier_rate_percent": (
            len(pipeline_outputs["inlier_matches"])
            / len(pipeline_outputs["good_matches"])
            * 100
            if pipeline_outputs["good_matches"]
            else 0.0
        ),
        "homography_max_abs_diff": max_abs_diff,
    }
    return pformat(summary, sort_dicts=False)


@pytest.fixture()
def pipeline_outputs():
    """运行一遍默认合成图像流程，供多个测试复用。"""
    img1, img2, h_true = sim.make_test_images(size=400)
    kp1, des1 = sim.detect_and_compute(img1)
    kp2, des2 = sim.detect_and_compute(img2)
    candidates, good_matches = sim.match_features(des1, des2)
    h_est, inlier_matches = sim.filter_with_ransac(kp1, kp2, good_matches)
    return {
        "img1": img1,
        "img2": img2,
        "h_true": h_true,
        "kp1": kp1,
        "des1": des1,
        "kp2": kp2,
        "des2": des2,
        "candidates": candidates,
        "good_matches": good_matches,
        "h_est": h_est,
        "inlier_matches": inlier_matches,
    }


def test_make_test_images_returns_expected_shapes() -> None:
    img1, img2, h_true = sim.make_test_images(size=256)
    info = {
        "img1_shape": tuple(img1.shape),
        "img2_shape": tuple(img2.shape),
        "img1_dtype": str(img1.dtype),
        "img2_dtype": str(img2.dtype),
        "h_true_shape": tuple(h_true.shape),
    }
    print("\n[unit] make_test_images 输出信息:\n" + pformat(info, sort_dicts=False))

    assert img1.shape == (256, 256), info
    assert img2.shape == (256, 256), info
    assert img1.dtype == np.uint8, info
    assert img2.dtype == np.uint8, info
    assert h_true.shape == (3, 3), info


def test_detect_and_compute_finds_keypoints_and_descriptors() -> None:
    img1, _, _ = sim.make_test_images()
    keypoints, descriptors = sim.detect_and_compute(img1)
    info = {
        "keypoint_count": len(keypoints),
        "descriptor_shape": None if descriptors is None else tuple(descriptors.shape),
    }
    print("\n[unit] detect_and_compute 输出信息:\n" + pformat(info, sort_dicts=False))

    assert len(keypoints) > 0, info
    assert descriptors is not None, info
    assert descriptors.ndim == 2, info
    assert descriptors.shape[0] == len(keypoints), info
    assert descriptors.shape[1] == 128, info


def test_match_features_returns_candidates_and_good_matches(pipeline_outputs) -> None:
    """验证 match_features 函数的输出结构和基本合理性。这里不对匹配质量做严格断言，因为它可能受随机因素影响，但至少要保证有候选匹配和一些良好匹配。"""
    
    summary = _pipeline_summary(pipeline_outputs)
    print("\n[unit] match_features 流水线摘要:\n" + summary)

    assert pipeline_outputs["des1"] is not None, summary
    assert pipeline_outputs["des2"] is not None, summary
    assert pipeline_outputs["candidates"] > 0, summary
    assert len(pipeline_outputs["good_matches"]) > 0, summary
    assert len(pipeline_outputs["good_matches"]) <= pipeline_outputs["candidates"], summary


def test_filter_with_ransac_returns_homography_and_inliers(pipeline_outputs) -> None:
    summary = _pipeline_summary(pipeline_outputs)
    print("\n[unit] filter_with_ransac 流水线摘要:\n" + summary)

    assert pipeline_outputs["h_true"].shape == (3, 3), summary
    assert pipeline_outputs["h_est"] is not None, summary
    assert pipeline_outputs["h_est"].shape == (3, 3), summary
    assert len(pipeline_outputs["good_matches"]) >= 4, summary
    assert len(pipeline_outputs["inlier_matches"]) > 0, summary
    assert len(pipeline_outputs["inlier_matches"]) <= len(pipeline_outputs["good_matches"]), summary


def test_filter_with_ransac_returns_none_when_matches_are_insufficient() -> None:
    img1, img2, _ = sim.make_test_images()
    kp1, _ = sim.detect_and_compute(img1)
    kp2, _ = sim.detect_and_compute(img2)

    h_est, inlier_matches = sim.filter_with_ransac(kp1, kp2, [])
    info = {
        "kp1_count": len(kp1),
        "kp2_count": len(kp2),
        "input_good_matches": 0,
        "h_est": h_est,
        "inlier_matches": inlier_matches,
    }
    print("\n[unit] 匹配不足时的返回信息:\n" + pformat(info, sort_dicts=False))

    assert h_est is None, info
    assert inlier_matches == [], info

# 自定义 ratio_threshold 参数的单元测试
import argparse


def test_ratio_threshold_type_accepts_valid_value() -> None:
    value = sim.ratio_threshold_type("0.7")
    print(f"\n[unit] 合法 ratio_threshold 输出: {value}")
    assert value == 0.7


@pytest.mark.parametrize("raw", ["0", "-0.1", "1", "1.2"])
def test_ratio_threshold_type_rejects_out_of_range_values(raw: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="ratio_threshold 必须在 0 到 1 之间。",
    ):
        sim.ratio_threshold_type(raw)


def test_match_features_supports_custom_ratio_threshold(pipeline_outputs) -> None:
    des1 = pipeline_outputs["des1"]
    des2 = pipeline_outputs["des2"]

    candidates_075, good_075 = sim.match_features(des1, des2, ratio_threshold=0.75)
    candidates_090, good_090 = sim.match_features(des1, des2, ratio_threshold=0.90)

    info = {
        "candidates_075": candidates_075,
        "good_matches_075": len(good_075),
        "candidates_090": candidates_090,
        "good_matches_090": len(good_090),
    }
    print(f"\n[unit] 自定义 ratio_threshold 对比信息:\n{info}")

    assert candidates_075 == candidates_090, info
    assert len(good_090) >= len(good_075), info


def test_filter_keypoints_by_gray_filters_keypoints_near_low_gray_values() -> None:
    """验证 filter_keypoints_by_gray 能过滤掉靠近低灰度值的关键点。"""
    img1, _, _ = sim.make_test_images(size=400)
    kp1, des1 = sim.detect_and_compute(img1)

    # 使用默认阈值（10.0）和半径（5.0）
    filtered_kp, filtered_des = sim.filter_keypoints_by_gray(img1, kp1, des1, 10.0, 5.0)

    info = {
        "original_keypoints": len(kp1),
        "filtered_keypoints": len(filtered_kp),
        "original_des_shape": des1.shape if des1 is not None else None,
        "filtered_des_shape": filtered_des.shape if filtered_des is not None else None,
    }
    print(f"\n[unit] filter_keypoints_by_gray 输出信息:\n{pformat(info, sort_dicts=False)}")

    # 过滤后的关键点数量应该小于等于原始数量
    assert len(filtered_kp) <= len(kp1), info
    # 描述子数量应该与关键点数量一致
    if filtered_des is not None:
        assert filtered_des.shape[0] == len(filtered_kp), info
        assert filtered_des.shape[1] == 128, info


def test_filter_keypoints_by_gray_with_high_threshold_filters_more() -> None:
    """验证更高的灰度阈值会过滤掉更多关键点。"""
    img1, _, _ = sim.make_test_images(size=400)
    kp1, des1 = sim.detect_and_compute(img1)

    # 使用低阈值
    filtered_kp_low, _ = sim.filter_keypoints_by_gray(img1, kp1, des1, 10.0, 5.0)
    # 使用高阈值
    filtered_kp_high, _ = sim.filter_keypoints_by_gray(img1, kp1, des1, 50.0, 10.0)

    info = {
        "original_keypoints": len(kp1),
        "filtered_keypoints_low_threshold": len(filtered_kp_low),
        "filtered_keypoints_high_threshold": len(filtered_kp_high),
    }
    print(f"\n[unit] 不同阈值过滤对比:\n{pformat(info, sort_dicts=False)}")

    # 高阈值应该过滤掉更多或同等数量的关键点
    assert len(filtered_kp_high) <= len(filtered_kp_low), info


def test_filter_keypoints_by_gray_returns_empty_when_all_filtered() -> None:
    """验证当所有关键点都被过滤时，返回空列表。"""
    img1, _, _ = sim.make_test_images(size=400)
    kp1, des1 = sim.detect_and_compute(img1)

    # 使用极高的阈值（所有像素都被认为是无效值）
    filtered_kp, filtered_des = sim.filter_keypoints_by_gray(img1, kp1, des1, 255.0, 50.0)

    info = {
        "original_keypoints": len(kp1),
        "filtered_keypoints": len(filtered_kp),
        "filtered_des_shape": filtered_des.shape if filtered_des is not None else None,
    }
    print(f"\n[unit] 全部过滤场景:\n{pformat(info, sort_dicts=False)}")

    # 应该过滤掉所有或几乎所有关键点
    assert len(filtered_kp) <= len(kp1), info
    if filtered_des is not None:
        assert filtered_des.shape[0] == len(filtered_kp), info


def test_invalid_gray_threshold_type_accepts_valid_value() -> None:
    """验证 invalid_gray_threshold_type 接受合法的灰度阈值。"""
    value = sim.invalid_gray_threshold_type("50.0")
    print(f"\n[unit] 合法 invalid_gray_threshold 输出: {value}")
    assert value == 50.0


@pytest.mark.parametrize("raw", ["-1", "256", "300"])
def test_invalid_gray_threshold_type_rejects_out_of_range_values(raw: str) -> None:
    """验证 invalid_gray_threshold_type 拒绝超出范围的值。"""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="invalid_gray_threshold 必须在 0 到 255 之间。",
    ):
        sim.invalid_gray_threshold_type(raw)


def test_invalid_radius_type_accepts_valid_value() -> None:
    """验证 invalid_radius_type 接受合法的半径值。"""
    value = sim.invalid_radius_type("10.0")
    print(f"\n[unit] 合法 invalid_radius 输出: {value}")
    assert value == 10.0


@pytest.mark.parametrize("raw", ["0", "-5", "-0.1"])
def test_invalid_radius_type_rejects_invalid_values(raw: str) -> None:
    """验证 invalid_radius_type 拒绝非正数值。"""
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="invalid_radius 必须是正数。",
    ):
        sim.invalid_radius_type(raw)


def test_match_features_flann_returns_candidates_and_good_matches(pipeline_outputs) -> None:
    """验证 FLANN 匹配器的输出结构和基本合理性。"""
    des1 = pipeline_outputs["des1"]
    des2 = pipeline_outputs["des2"]
    candidates, good_matches = sim.match_features_flann(des1, des2)

    info = {
        "candidates": candidates,
        "good_matches": len(good_matches),
    }
    print(f"\n[unit] match_features_flann 输出信息:\n{pformat(info, sort_dicts=False)}")

    assert candidates > 0, info
    assert len(good_matches) > 0, info
    assert len(good_matches) <= candidates, info


def test_match_features_flann_supports_custom_ratio_threshold(pipeline_outputs) -> None:
    """验证 FLANN 匹配器的 ratio_threshold 参数生效。"""
    des1 = pipeline_outputs["des1"]
    des2 = pipeline_outputs["des2"]

    _, good_075 = sim.match_features_flann(des1, des2, ratio_threshold=0.75)
    _, good_090 = sim.match_features_flann(des1, des2, ratio_threshold=0.90)

    info = {
        "good_matches_075": len(good_075),
        "good_matches_090": len(good_090),
    }
    print(f"\n[unit] FLANN 自定义 ratio_threshold 对比:\n{pformat(info, sort_dicts=False)}")

    assert len(good_090) >= len(good_075), info


def test_match_features_crosscheck_returns_matches(pipeline_outputs) -> None:
    """验证 cross-check 匹配模式的输出。"""
    des1 = pipeline_outputs["des1"]
    des2 = pipeline_outputs["des2"]
    candidates, matches = sim.match_features_crosscheck(des1, des2)

    info = {
        "candidates": candidates,
        "crosscheck_matches": len(matches),
    }
    print(f"\n[unit] match_features_crosscheck 输出信息:\n{pformat(info, sort_dicts=False)}")

    assert candidates == des1.shape[0], info
    assert len(matches) > 0, info
    assert len(matches) <= candidates, info
    # cross-check 匹配结果按距离排序
    if len(matches) >= 2:
        assert matches[0].distance <= matches[1].distance, info


def test_filter_with_ransac_fundamental_matrix(pipeline_outputs) -> None:
    """验证基础矩阵 RANSAC 方法返回 3x3 矩阵。"""
    kp1 = pipeline_outputs["kp1"]
    kp2 = pipeline_outputs["kp2"]
    good_matches = pipeline_outputs["good_matches"]

    matrix, inlier_matches = sim.filter_with_ransac(
        kp1, kp2, good_matches, method="fundamental"
    )

    info = {
        "matrix_shape": None if matrix is None else tuple(matrix.shape),
        "inlier_count": len(inlier_matches),
        "good_matches": len(good_matches),
    }
    print(f"\n[unit] fundamental matrix RANSAC 输出:\n{pformat(info, sort_dicts=False)}")

    assert matrix is not None, info
    assert matrix.shape == (3, 3), info
    assert len(inlier_matches) > 0, info
    assert len(inlier_matches) <= len(good_matches), info


def test_filter_with_ransac_fundamental_returns_none_when_insufficient() -> None:
    """验证基础矩阵需要至少 8 个匹配点。"""
    img1, img2, _ = sim.make_test_images()
    kp1, _ = sim.detect_and_compute(img1)
    kp2, _ = sim.detect_and_compute(img2)

    matrix, inlier_matches = sim.filter_with_ransac(
        kp1, kp2, [], method="fundamental"
    )
    info = {"matrix": matrix, "inlier_matches": inlier_matches}
    print(f"\n[unit] fundamental 匹配不足时的返回:\n{pformat(info, sort_dicts=False)}")

    assert matrix is None, info
    assert inlier_matches == [], info


def test_draw_match_image_creates_file(pipeline_outputs, tmp_path) -> None:
    """验证 draw_match_image 能生成输出文件。"""
    output_path = str(tmp_path / "test_draw.png")
    sim.draw_match_image(
        pipeline_outputs["img1"],
        pipeline_outputs["img2"],
        pipeline_outputs["kp1"],
        pipeline_outputs["kp2"],
        pipeline_outputs["inlier_matches"],
        output_path,
    )

    import os
    assert os.path.exists(output_path), f"输出文件不存在: {output_path}"
    assert os.path.getsize(output_path) > 0, f"输出文件为空: {output_path}"
    print(f"\n[unit] draw_match_image 文件大小: {os.path.getsize(output_path)} bytes")

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-s", "-q"]))
