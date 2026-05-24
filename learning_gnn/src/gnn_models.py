"""从零实现三种经典 GNN：GCN、GAT、GraphSAGE。

所有模型都遵循消息传递范式：

1. message：邻居节点产生要传来的信息。
2. aggregate：目标节点聚合邻居信息，例如求和、平均或注意力加权。
3. update：用聚合结果更新目标节点表示。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """给每个节点增加自环，让节点在聚合邻居时也保留自身信息。"""

    device = edge_index.device
    loops = torch.arange(num_nodes, device=device)
    loops = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loops], dim=1)


class GCNLayer(nn.Module):
    """Graph Convolutional Network 层。

    GCN 使用归一化邻接矩阵传播消息：
    H' = D^{-1/2} A D^{-1/2} H W

    直觉：每个节点接收邻居表示，但高阶数节点不能因为邻居多而数值过大，所以用度数
    做对称归一化。
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index

        h = self.linear(x)
        degree = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
        norm = degree[src].pow(-0.5) * degree[dst].pow(-0.5)

        out = torch.zeros_like(h)
        # index_add_ 是纯 PyTorch 的聚合操作：把每条边的消息加到目标节点。
        out.index_add_(0, dst, h[src] * norm.unsqueeze(-1))
        return out


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        self.conv1 = GCNLayer(in_features, hidden)
        self.conv2 = GCNLayer(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGELayer(nn.Module):
    """GraphSAGE mean 聚合层。

    GraphSAGE 的核心是“采样并聚合邻居”。这里为了代码清晰没有做邻居采样，而是对
    全部邻居求平均，再把自身表示和邻居平均表示拼接后更新。
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index

        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, dst, x[src])
        degree = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
        neighbor_mean = neighbor_sum / degree.unsqueeze(-1)

        return self.linear(torch.cat([x, neighbor_mean], dim=-1))


class GraphSAGE(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float):
        super().__init__()
        self.conv1 = GraphSAGELayer(in_features, hidden)
        self.conv2 = GraphSAGELayer(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GATLayer(nn.Module):
    """Graph Attention Network 层。

    GAT 不预先固定邻居权重，而是给每条边学习一个注意力分数。目标节点会对自己的
    入边做 softmax，重要邻居获得更大的消息权重。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout

        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index

        h = self.linear(x).view(num_nodes, self.heads, self.out_features)
        # 每条边的注意力 logit，只依赖源节点、目标节点和当前 head 的参数。
        scores = (h[src] * self.attn_src).sum(-1) + (h[dst] * self.attn_dst).sum(-1)
        scores = self.leaky_relu(scores)

        out = torch.zeros(num_nodes, self.heads, self.out_features, device=x.device)
        for node in range(num_nodes):
            incoming = (dst == node).nonzero(as_tuple=False).flatten()
            if incoming.numel() == 0:
                continue
            alpha = torch.softmax(scores[incoming], dim=0)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            messages = h[src[incoming]] * alpha.unsqueeze(-1)
            out[node] = messages.sum(dim=0)

        if self.concat:
            return out.reshape(num_nodes, self.heads * self.out_features)
        return out.mean(dim=1)


class GAT(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int,
        classes: int,
        dropout: float,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATLayer(in_features, hidden, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATLayer(
            hidden * heads, classes, heads=1, concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def build_model(
    name: str,
    in_features: int,
    hidden: int,
    classes: int,
    dropout: float,
    heads: int,
) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        return GCN(in_features, hidden, classes, dropout)
    if name == "gat":
        return GAT(in_features, hidden, classes, dropout, heads=heads)
    if name in {"sage", "graphsage"}:
        return GraphSAGE(in_features, hidden, classes, dropout)
    raise ValueError(f"未知模型: {name}")
