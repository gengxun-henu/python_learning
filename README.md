# Python_Tests

这个仓库包含若干 Python 学习脚本、示例程序与配套测试。当前与计算机视觉相关的示例是 `examples/sift_image_matching.py`，它演示了基于 **SIFT + BFMatcher + Lowe's ratio test + RANSAC** 的图像特征匹配流程。

## 环境准备

推荐使用你当前已经验证通过的 conda 环境：`asp360_new`。

### 激活环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asp360_new
```

### 安装依赖

首次使用时，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

当前示例与测试依赖至少包括：

- `numpy`
- `opencv-contrib-python`
- `pytest`

## 运行 SIFT 图像匹配示例

请先进入仓库根目录：

```bash
cd /home/gengxun/Code/Python_Tests
```

### 方式 1：使用脚本内置的合成图像

这是最简单的运行方式，不依赖外部图像文件：

```bash
python examples/sift_image_matching.py
```

运行后你会在终端看到：

- 图像尺寸
- SIFT 检测到的关键点数量
- BFMatcher 候选匹配数量
- Lowe ratio test 后的良好匹配数量
- RANSAC 内点匹配数量
- 估计单应矩阵与简单统计摘要

### 方式 2：使用你自己的两张图像

当两张图像来自同一场景、但拍摄角度或光照略有差异时，更适合观察真实匹配效果：

```bash
python examples/sift_image_matching.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

注意：

- `--img1` 和 `--img2` 必须同时提供
- 图像应尽量包含足够纹理与重叠区域
- 支持 OpenCV 可读取的常见格式，如 `jpg`、`png`、`bmp`、`tif`

### 方式 3：使用仓库内置的真实测试影像

仓库自带了一组 `Hayabusa2` 测试影像，可直接用于功能验证：

```bash
python examples/sift_image_matching.py \
  --img1 tests/data/hayabusa2/hyb2_onc_20180710_060508_tvf_l2a.tif \
  --img2 tests/data/hayabusa2/hyb2_onc_20180710_060852_tvf_l2a.tif
```

## 运行测试

与该示例对应的自动化测试分为两类：

- 单元测试：`tests/unit/test_sift_image_matching_unit.py`
- 功能测试：`tests/functional/test_sift_image_matching_functional.py`

### 运行全部相关测试

```bash
python -m pytest tests/unit/test_sift_image_matching_unit.py tests/functional/test_sift_image_matching_functional.py -q
```

### 只运行单元测试

```bash
python -m pytest tests/unit/test_sift_image_matching_unit.py -q
```

### 只运行功能测试

```bash
python -m pytest tests/functional/test_sift_image_matching_functional.py -q
```

## 这些测试分别测什么

### 单元测试

单元测试主要验证脚本里的核心函数是否具备预期行为，例如：

- `make_test_images()` 是否能生成正确尺寸的灰度图和单应矩阵
- `detect_and_compute()` 是否能提取关键点和 128 维 SIFT 描述子
- `match_features()` 是否能返回候选匹配和良好匹配
- `filter_with_ransac()` 是否能在匹配充足时估计单应矩阵，并在匹配不足时正确返回 `None`

### 功能测试

功能测试把脚本当成一个整体来运行，验证：

- 默认无参数运行是否成功
- 参数错误时是否给出正确提示
- 使用仓库内置真实影像时整条流程是否可正常执行

## 常见问题

### 1. 运行时报 `No module named cv2`

说明当前 Python 环境里还没有安装 OpenCV。请确认已经激活 `asp360_new`，然后执行：

```bash
python -m pip install -r requirements.txt
```

### 2. 运行时报 `No module named pytest`

说明测试框架未安装，同样执行：

```bash
python -m pip install -r requirements.txt
```

### 3. 为什么脚本能运行，但匹配数量不多？

可能原因包括：

- 图像纹理太少
- 两张图像重叠区域不足
- 视角变化过大
- 光照差异太强

可以尝试：

- 调整 `ratio_threshold`
- 调整 `ransac_threshold`
- 换一对重叠区域更明显的图像

## 相关文件

- `examples/sift_image_matching.py`：SIFT 匹配示例脚本
- `tests/unit/test_sift_image_matching_unit.py`：单元测试
- `tests/functional/test_sift_image_matching_functional.py`：功能测试
- `tests/data/hayabusa2/`：真实测试影像数据

## 快速命令清单

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asp360_new
cd /home/gengxun/Code/Python_Tests
python -m pip install -r requirements.txt
python examples/sift_image_matching.py
python -m pytest tests/unit/test_sift_image_matching_unit.py tests/functional/test_sift_image_matching_functional.py -q
```
