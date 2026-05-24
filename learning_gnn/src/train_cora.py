"""在 Cora 数据集上训练 GCN/GAT/GraphSAGE。

运行方式：
    python src/train_cora.py --model gcn
    python src/train_cora.py --model gat
    python src/train_cora.py --model sage
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from cora_data import load_cora
from gnn_models import build_model


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    predictions = logits[mask].argmax(dim=-1)
    return (predictions == labels[mask]).float().mean().item()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    data = load_cora(args.data_dir, args.seed)
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model = build_model(
        args.model,
        in_features=x.size(1),
        hidden=args.hidden,
        classes=len(data.label_names),
        dropout=args.dropout,
        heads=args.heads,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            train_acc = accuracy(logits, y, train_mask)
            val_acc = accuracy(logits, y, val_mask)
            test_acc = accuracy(logits, y, test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}"
            )

    print(f"best val={best_val:.3f}, paired test={best_test:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cora 节点分类 GNN 示例")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "graphsage"], default="gcn")
    parser.add_argument("--data-dir", default="data/cora")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
