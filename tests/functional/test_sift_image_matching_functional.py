"""`examples/sift_image_matching.py` 的功能测试。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None
pytestmark = pytest.mark.skipif(
    not CV2_AVAILABLE,
    reason="opencv-contrib-python 未安装，跳过 SIFT 功能测试。",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "sift_image_matching.py"


def test_script_runs_successfully_with_default_synthetic_images() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "生成合成测试图像" in result.stdout
    assert "SIFT 特征检测与描述子提取" in result.stdout
    assert "RANSAC 过滤离群点" in result.stdout
    assert "RANSAC 后内点匹配" in result.stdout


def test_script_fails_when_only_one_image_argument_is_provided() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--img1", "only_one_image.jpg"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--img1 和 --img2 必须同时提供" in result.stdout


def test_script_runs_successfully_with_real_images() -> None:
    """使用仓库内置真实图像验证脚本的真实数据路径。"""
    img1_path = PROJECT_ROOT / "tests" / "data" / "hayabusa2" / "hyb2_onc_20180710_060508_tvf_l2a.tif"
    img2_path = PROJECT_ROOT / "tests" / "data" / "hayabusa2" / "hyb2_onc_20180710_060852_tvf_l2a.tif"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--img1",
            str(img1_path),
            "--img2",
            str(img2_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "加载用户提供的真实图像" in result.stdout
    assert "SIFT 特征检测与描述子提取" in result.stdout
    assert "RANSAC 过滤离群点" in result.stdout
    assert "RANSAC 后内点匹配" in result.stdout
