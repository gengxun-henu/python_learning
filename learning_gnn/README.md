# 学习 GNN 图神经网络

本分支用纯 PyTorch 从零实现三种经典图神经网络：GCN、GAT、GraphSAGE，并提供
Cora 论文引用网络上的节点分类示例。所有代码都在 `src/` 目录下。

## 运行方式

推荐使用本机 conda 环境 `deep-learning`。本分支已在该环境中验证过：

- `torch==2.5.1`
- `torch.cuda.is_available() == False`，示例可在 CPU 上运行

```bash
conda activate deep-learning
python src/train_cora.py --model gcn
python src/train_cora.py --model gat
python src/train_cora.py --model sage
```

如果不使用该环境，也可以自行安装依赖：

```bash
pip install -r requirements.txt
python src/train_cora.py --model gcn
python src/train_cora.py --model gat
python src/train_cora.py --model sage
```

第一次运行会自动下载 Cora 的 `cora.content` 和 `cora.cites` 到 `data/cora/`。

## 文件结构

- `src/cora_data.py`：下载、解析 Cora，并构造训练/验证/测试 mask。
- `src/gnn_models.py`：从零实现 GCN、GAT、GraphSAGE。
- `src/train_cora.py`：节点分类训练入口。
- `requirements.txt`：运行示例所需依赖。

## GNN 的核心概念

图神经网络的目标是学习节点、边或整张图的表示。和普通神经网络不同，GNN 的输入
不是固定网格或纯序列，而是由节点和边组成的图结构。

大多数 GNN 都可以理解为消息传递：

1. 每个节点从邻居节点接收消息。
2. 节点把邻居消息聚合起来。
3. 节点用聚合结果更新自己的表示。

经过多层堆叠后，节点可以获得多跳邻域的信息。例如 2 层 GNN 中，一个节点能间接看到
2 跳以内的节点。

## 三种模型的区别

### GCN

GCN 使用归一化邻接矩阵传播信息。它的邻居权重由图结构和节点度数决定，不额外学习
每条边的重要性。优点是简单稳定，适合理解 GNN 基础。

### GAT

GAT 使用注意力机制为每个邻居学习不同权重。一个节点可以更关注有用邻居，弱化噪声
邻居。优点是表达能力更强，缺点是计算更慢，实现也更复杂。

### GraphSAGE

GraphSAGE 强调从邻居采样并聚合，适合大图归纳学习。这里实现的是 mean aggregator：
先对邻居表示求平均，再把自身表示和邻居平均表示拼接更新。相比 GCN，它更清楚地区分
“自身信息”和“邻居信息”。

## 学习心得

GCN、GAT、GraphSAGE 的共同点是都在做邻居信息聚合；区别在于“怎么聚合”：

- GCN：用固定的度数归一化权重。
- GAT：用可学习的注意力权重。
- GraphSAGE：用聚合函数总结邻居，再和自身表示组合。

理解这三个模型后，再学习更复杂的图模型时，可以先问两个问题：消息是什么，聚合方式
是什么。这个视角能快速看清大多数 GNN 变体。
