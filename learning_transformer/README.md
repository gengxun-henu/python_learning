# 学习 Transformer 架构与实现

本分支用纯 PyTorch 从零实现一个简化版 Transformer Encoder，并提供一个可运行的
序列预测训练示例。所有核心代码都在 `src/` 目录下。

## 运行方式

推荐使用本机 conda 环境 `deep-learning`。本分支已在该环境中验证过：

- `torch==2.5.1`
- `torch.cuda.is_available() == False`，示例可在 CPU 上运行

```bash
conda activate deep-learning
python src/train_sequence_prediction.py
```

如果不使用该环境，也可以自行安装依赖：

```bash
pip install -r requirements.txt
python src/train_sequence_prediction.py
```

如果想更快看到结果，可以减少训练轮数：

```bash
python src/train_sequence_prediction.py --epochs 3 --samples 256
```

## 文件结构

- `src/transformer.py`：Self-Attention、Multi-Head Attention、位置编码、前馈网络和简化 Transformer。
- `src/train_sequence_prediction.py`：递增序列预测训练示例。
- `requirements.txt`：运行示例所需依赖。

## 核心概念

### Self-Attention

Self-Attention 让序列中的每个位置都能直接读取其他位置的信息。它使用三组向量：

- Query：当前位置想找什么信息。
- Key：每个位置提供什么可匹配特征。
- Value：匹配成功后真正被聚合的信息。

缩放点积注意力公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

除以 `sqrt(d_k)` 是为了避免点积值过大，使 softmax 不至于过早变得非常尖锐。

### Multi-Head Attention

多头注意力把向量拆成多个子空间并行计算注意力。不同 head 可以学习不同关系，例如
局部邻近、长距离依赖或重复模式。最后把多个 head 的结果拼接，再通过线性层融合。

### Positional Encoding

Attention 不天然知道顺序。位置编码把位置相关的正弦/余弦信号加到 token embedding
上，让模型区分“同一个数字出现在第 1 位”和“同一个数字出现在第 5 位”。

### Feed-Forward

Feed-Forward 网络对每个位置独立做非线性变换。Attention 负责位置间通信，Feed-Forward
负责提升每个位置表示的表达能力。

### 残差连接与 LayerNorm

每个子层外面都有残差连接和 LayerNorm。残差连接让梯度更容易传回早期层，LayerNorm
稳定每层激活分布，使训练更可靠。

## 学习心得

Transformer 的关键不是某个单独公式，而是一个清晰分工：

1. Embedding 把离散 token 变成向量。
2. Positional Encoding 注入顺序信息。
3. Self-Attention 在序列内部传递上下文。
4. Multi-Head 让模型同时观察多种关系。
5. Feed-Forward 对每个位置做更强的非线性表达。
6. 残差和归一化让深层堆叠可训练。

这个实现只包含 Encoder，并用 toy 任务验证训练闭环。理解它之后，再看完整的
Encoder-Decoder Transformer、GPT 类 decoder-only 模型或 BERT 类 encoder-only 模型
会更容易。
