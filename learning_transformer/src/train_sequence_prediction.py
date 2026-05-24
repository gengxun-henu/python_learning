"""训练一个简化 Transformer 做序列预测。

运行方式：
    python src/train_sequence_prediction.py

任务：给定长度为 seq_len 的递增序列，预测每个位置的下一个数字。
例如输入 [3, 4, 5, 6]，目标是 [4, 5, 6, 7]。
"""

from __future__ import annotations

import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformer import MiniTransformer


class CountingDataset(Dataset):
    """动态生成简单序列数据。

    这个数据集的目标不是追求真实业务复杂度，而是让学习者能快速验证 Transformer 的
    前向传播、损失计算和参数更新流程。
    """

    def __init__(self, size: int, seq_len: int, vocab_size: int):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        start = random.randint(0, self.vocab_size - self.seq_len - 2)
        values = torch.arange(start, start + self.seq_len + 1) % self.vocab_size
        return values[:-1].long(), values[1:].long()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = CountingDataset(args.samples, args.seq_len, args.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MiniTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=args.d_ff,
        num_layers=args.layers,
        max_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            # CrossEntropyLoss 期望形状为 [N, C]，所以把 batch 和 seq 维度压平。
            loss = criterion(logits.reshape(-1, args.vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        print(f"epoch={epoch:02d} loss={avg_loss:.4f} accuracy={accuracy:.3f}")

    model.eval()
    demo = torch.tensor([[2, 3, 4, 5, 6, 7]], device=device)
    with torch.no_grad():
        logits, _ = model(demo)
    print("demo input:     ", demo.cpu().tolist()[0])
    print("model predicts: ", logits.argmax(dim=-1).cpu().tolist()[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个简化版 Transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
