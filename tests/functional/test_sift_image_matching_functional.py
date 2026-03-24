"""`examples/sift_image_matching.py` 的功能测试。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import cv2  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "sift_image_matching.py"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    """统一运行脚本，便于复用与输出调试信息。"""
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _important_stdout_lines(stdout: str) -> list[str]:
    """抽取脚本输出中的关键摘要行。"""
    keywords = (
        "生成合成测试图像",
        "加载用户提供的真实图像",
        "图像 1 关键点数量",
        "图像 2 关键点数量",
        "原始候选匹配数",
        "通过 ratio test",
        "RANSAC 内点匹配数",
        "RANSAC 后内点匹配",
        "内点率",
        "错误：",
    )
    return [line for line in stdout.splitlines() if any(keyword in line for keyword in keywords)]


def _format_result_info(case_name: str, result: subprocess.CompletedProcess[str]) -> str:
    """把功能测试结果整理为便于阅读的调试信息。"""
    key_lines = _important_stdout_lines(result.stdout)
    parts = [
        f"case={case_name}",
        f"command={' '.join(result.args)}",
        f"returncode={result.returncode}",
        "stdout_key_lines:",
        *(f"  {line}" for line in key_lines or ["<none>"]),
    ]
    if result.stderr.strip():
        parts.append("stderr:")
        parts.extend(f"  {line}" for line in result.stderr.strip().splitlines())
    return "\n".join(parts)


def test_script_runs_successfully_with_default_synthetic_images() -> None:
    result = _run_script()
    info = _format_result_info("default_synthetic", result)
    print("\n[functional] 默认合成图像运行摘要:\n" + info)

    assert result.returncode == 0, info
    assert "生成合成测试图像" in result.stdout, info
    assert "SIFT 特征检测与描述子提取" in result.stdout, info
    assert "RANSAC 过滤离群点" in result.stdout, info
    assert "RANSAC 后内点匹配" in result.stdout, info


def test_script_fails_when_only_one_image_argument_is_provided() -> None:
    """
    docstring: 这个测试验证了当用户只提供 --img1 参数而没有提供 --img2 参数时，脚本是否能够正确地检测到这个错误并返回非零的退出码，同时在输出中包含明确的错误信息提示用户需要同时提供两个图像路径。
    """
    result = _run_script("--img1", "only_one_image.jpg")
    info = _format_result_info("missing_second_image", result)
    print("\n[functional] 参数错误场景摘要:\n" + info)

    assert result.returncode != 0, info
    assert "--img1 和 --img2 必须同时提供" in result.stdout, info


def test_script_runs_successfully_with_real_images() -> None:
    """使用仓库内置真实图像验证脚本的真实数据路径。"""
    img1_path = PROJECT_ROOT / "tests" / "data" / "hayabusa2" / "hyb2_onc_20180710_060508_tvf_l2a.tif"
    img2_path = PROJECT_ROOT / "tests" / "data" / "hayabusa2" / "hyb2_onc_20180710_060852_tvf_l2a.tif"

    result = _run_script("--img1", str(img1_path), "--img2", str(img2_path))
    info = _format_result_info("real_images", result)
    print("\n[functional] 真实图像运行摘要:\n" + info)

    assert result.returncode == 0, info
    assert "加载用户提供的真实图像" in result.stdout, info
    assert "SIFT 特征检测与描述子提取" in result.stdout, info
    assert "RANSAC 过滤离群点" in result.stdout, info
    assert "RANSAC 后内点匹配" in result.stdout, info

def test_script_runs_successfully_with_custom_ratio_threshold() -> None:
    result = _run_script("--ratio-threshold", "0.7")
    info = _format_result_info("custom_ratio_threshold", result)
    print("\n[functional] 自定义 ratio_threshold 场景摘要:\n" + info)

    assert result.returncode == 0, info
    assert "ratio_threshold: 0.7" in result.stdout, info
    assert "通过 ratio test 的良好匹配数" in result.stdout, info


@pytest.mark.parametrize("bad_value", ["0", "-0.1", "1", "1.2"])
def test_script_fails_with_invalid_ratio_threshold(bad_value: str) -> None:
    result = _run_script("--ratio-threshold", bad_value)
    info = _format_result_info(f"invalid_ratio_threshold_{bad_value}", result)
    print("\n[functional] 非法 ratio_threshold 运行摘要:\n" + info)

    assert result.returncode != 0, info
    assert "ratio_threshold 必须在 0 到 1 之间。" in result.stderr or "ratio_threshold 必须在 0 到 1 之间。" in result.stdout, info



if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-s", "-q"]))
