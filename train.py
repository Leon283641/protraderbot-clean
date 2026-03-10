"""
train.py — Train and compare all 4 models.

Usage:
    python train.py                        # compare all 4
    python train.py --model baseline       # just one
    python train.py --model transformer
    python train.py --model gat
    python train.py --model combined
    python train.py --epochs 50
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data import make_datasets
from graph import build_graph
from models import LogisticBaseline, SimpleTransformer, SimpleGAT, TransformerGAT


# ── Training loop ──────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, edge_index=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = (logits > 0).float()
        correct += (preds == y).sum().item()
        total += y.numel()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, edge_index=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_logits, all_labels = [], []

    for x, y in loader:
        logits = model(x, edge_index)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        preds = (logits > 0).float()
        correct += (preds == y).sum().item()
        total += y.numel()
        all_logits.append(logits)
        all_labels.append(y)

    loss = total_loss / len(loader.dataset)
    acc = correct / total
    return loss, acc


# ── Main comparison ────────────────────────────────────

def run_all(epochs=None):
    if epochs is None:
        epochs = config.EPOCHS

    print("=" * 60)
    print("STOCK MOVEMENT PREDICTION — ARCHITECTURE COMPARISON")
    print("=" * 60)

    # Data
    print("\nDownloading data...")
    train_ds, test_ds, n_stocks, log_ret_df, info = make_datasets()
    print(f"  Stocks: {info['tickers']}")
    print(f"  Days: {info['n_days']}  |  Train: {info['train_size']}  |  Test: {info['test_size']}")
    print(f"  Class balance: {info['class_balance']}")
    print(f"  Window: {config.WINDOW_SIZE} days  |  d_model: {config.D_MODEL}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Graph
    edge_index = build_graph(log_ret_df)
    print(f"  Graph edges: {edge_index.shape[1]} (threshold={config.CORR_THRESHOLD})")

    # Models
    models = {
        "Logistic Baseline": (LogisticBaseline(), False),
        "Transformer": (SimpleTransformer(), False),
        "GAT": (SimpleGAT(), True),
        "Transformer+GAT": (TransformerGAT(), True),
    }

    loss_fn = nn.BCEWithLogitsLoss()
    results = {}

    for name, (model, uses_graph) in models.items():
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        ei = edge_index if uses_graph else None
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, ei)
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                test_loss, test_acc = evaluate(model, test_loader, loss_fn, ei)
                print(f"  Epoch {epoch:3d} | Train Acc: {train_acc:.3f} | "
                      f"Test Acc: {test_acc:.3f} | Loss: {test_loss:.4f}")

        test_loss, test_acc = evaluate(model, test_loader, loss_fn, ei)
        results[name] = {
            "accuracy": round(test_acc, 4),
            "loss": round(test_loss, 4),
            "params": n_params,
        }

    # Summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Loss':>10} {'Params':>10}")
    print(f"{'─' * 55}")
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:10.3f} {r['loss']:10.4f} {r['params']:10,}")
    print(f"\nRandom guess baseline: ~0.530 (class imbalance)")

    # Save results to JSON for presentations
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results.json")

    return results


def run_single(model_name, epochs=None):
    if epochs is None:
        epochs = config.EPOCHS

    train_ds, test_ds, n_stocks, log_ret_df, info = make_datasets()
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    edge_index = build_graph(log_ret_df)

    model_map = {
        "baseline": (LogisticBaseline(), False),
        "transformer": (SimpleTransformer(), False),
        "gat": (SimpleGAT(), True),
        "combined": (TransformerGAT(), True),
    }
    model, uses_graph = model_map[model_name]
    ei = edge_index if uses_graph else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, ei)
        if epoch % 5 == 0 or epoch == 1:
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, ei)
            print(f"  Epoch {epoch:3d} | Train: {train_acc:.3f} | Test: {test_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=["baseline", "transformer", "gat", "combined"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.model:
        run_single(args.model, args.epochs)
    else:
        run_all(args.epochs)
