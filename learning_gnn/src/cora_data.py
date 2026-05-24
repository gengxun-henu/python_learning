"""Cora 数据集下载与解析。

这里使用公开的 ``cora.content`` 和 ``cora.cites`` 文本文件。相比 Planetoid pickle
格式，这种格式不依赖 SciPy，便于用纯 PyTorch 学习 GNN。
"""

from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


CONTENT_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content"
CITES_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites"


@dataclass
class GraphData:
    """训练脚本需要的图数据。"""

    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    label_names: list[str]


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"download {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def load_cora(data_dir: str = "data/cora", seed: int = 42) -> GraphData:
    """下载并加载 Cora。

    Cora 是一个论文引用网络：节点是论文，边是引用关系，节点特征是词袋向量，标签是
    论文主题。节点分类任务是：只看少量有标签论文，预测其他论文的主题。
    """

    data_path = Path(data_dir)
    content_path = data_path / "cora.content"
    cites_path = data_path / "cora.cites"
    _download(CONTENT_URL, content_path)
    _download(CITES_URL, cites_path)

    paper_ids: list[str] = []
    features: list[list[float]] = []
    raw_labels: list[str] = []

    for line in content_path.read_text().splitlines():
        parts = line.strip().split("\t")
        paper_ids.append(parts[0])
        features.append([float(v) for v in parts[1:-1]])
        raw_labels.append(parts[-1])

    id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    label_names = sorted(set(raw_labels))
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor([label_to_idx[label] for label in raw_labels], dtype=torch.long)

    edges: set[tuple[int, int]] = set()
    for line in cites_path.read_text().splitlines():
        src_id, dst_id = line.strip().split("\t")
        if src_id not in id_to_idx or dst_id not in id_to_idx:
            continue
        src = id_to_idx[src_id]
        dst = id_to_idx[dst_id]
        # 引用关系天然有方向，但多数基础 GNN 会把它当作无向邻接关系使用。
        edges.add((src, dst))
        edges.add((dst, src))

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    train_mask, val_mask, test_mask = _make_masks(y, seed)

    return GraphData(x, y, edge_index, train_mask, val_mask, test_mask, label_names)


def _make_masks(
    labels: torch.Tensor,
    seed: int,
    train_per_class: int = 20,
    val_size: int = 500,
    test_size: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造常见的半监督节点分类划分。

    每个类别只给 20 个训练节点，模拟 GNN 论文中常见的低标签场景；验证集和测试集从
    剩余节点中随机抽取。
    """

    rng = random.Random(seed)
    num_nodes = labels.numel()
    train_indices: list[int] = []

    for class_id in labels.unique().tolist():
        class_indices = (labels == class_id).nonzero(as_tuple=False).flatten().tolist()
        rng.shuffle(class_indices)
        train_indices.extend(class_indices[:train_per_class])

    remaining = [idx for idx in range(num_nodes) if idx not in set(train_indices)]
    rng.shuffle(remaining)
    val_indices = remaining[:val_size]
    test_indices = remaining[val_size : val_size + test_size]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask
