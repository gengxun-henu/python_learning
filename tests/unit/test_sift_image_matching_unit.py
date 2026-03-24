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

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-s", "-q"]))
