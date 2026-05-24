"""从零实现一个简化版 Transformer。

这个文件刻意不直接使用 ``torch.nn.Transformer``，而是把核心积木拆开：
Self-Attention、Multi-Head Attention、Positional Encoding、Feed-Forward、
EncoderLayer 和一个用于序列预测的小模型。
"""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码。

    Self-Attention 本身只看 token 之间的内容相似度，不知道第几个 token 在前、
    第几个 token 在后。位置编码把“第几个位置”的信息加到词向量上，让模型能学习
    顺序模式。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer 表示它不是可训练参数，但会随模型一起保存/移动到 GPU。
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """给输入加位置编码。

        Args:
            x: [batch, seq_len, d_model]
        """

        return x + self.pe[:, : x.size(1)]


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力。

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Q 表示“我想找什么”，K 表示“我有什么特征可被匹配”，V 表示“匹配后要取走的
    信息”。除以 sqrt(d_k) 是为了避免点积过大导致 softmax 过早饱和。
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))

        if mask is not None:
            # mask 为 False 的位置不能被关注，填成极小值后 softmax 约等于 0。
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = attn @ value
        return output, attn


class MultiHeadAttention(nn.Module):
    """多头注意力。

    单个注意力头只能在一个表示空间里计算相似度。多头注意力把 d_model 切成多个
    子空间，每个头学习一种关系，例如局部邻近、长距离依赖或特定模式，最后再拼回
    原来的维度。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        if mask is not None:
            # 扩展 head 维度，让所有注意力头使用同一份 mask。
            mask = mask.unsqueeze(1)

        x, attn = self.attention(q, k, v, mask)
        x = self._merge_heads(x)
        return self.out_proj(self.dropout(x)), attn


class FeedForward(nn.Module):
    """逐位置前馈网络。

    Attention 负责在序列位置之间交换信息；Feed-Forward 负责对每个位置独立做非线性
    变换，提升模型表达能力。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """一个 Encoder 层：Self-Attention + Feed-Forward。

    每个子层都使用“残差连接 + LayerNorm”。残差让梯度更容易传递，LayerNorm 稳定
    激活分布，使训练更平滑。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x, attn


class MiniTransformer(nn.Module):
    """用于序列预测的简化 Transformer Encoder 模型。

    输入是一段整数序列，模型预测每个位置的下一个整数类别。训练脚本中会构造
    ``[a, a+1, a+2, ...]`` 这样的 toy 数据，让模型学习“下一个数”的规律。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(
        self, token_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)

        logits = self.classifier(x)
        return logits, attentions
