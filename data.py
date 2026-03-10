"""
data.py — Download stock prices, compute log returns, build datasets.

Single feature per stock: log return = ln(P_t / P_{t-1}).
Label: 1 if next-day return > 0 (UP), 0 if <= 0 (DOWN).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import config


def download_prices():
    """Download adjusted close prices for all stocks."""
    import yfinance as yf
    df = yf.download(config.TICKERS, start=config.START_DATE,
                     end=config.END_DATE, auto_adjust=True)["Close"]
    df = df.dropna()
    return df


def compute_log_returns(prices):
    """ln(P_t / P_{t-1}) — the only feature we use."""
    return np.log(prices / prices.shift(1)).dropna()


class StockDataset(Dataset):
    """
    Sliding window dataset for binary classification.

    Each sample:
        x: [N_stocks, window_size]  — past log returns
        y: [N_stocks]               — 1 if next return > 0, else 0
    """
    def __init__(self, log_returns_norm, log_returns_raw, window_size):
        self.data = log_returns_norm   # normalized (for model input)
        self.raw = log_returns_raw     # raw (for labels)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        w = self.window_size
        x = self.data[idx : idx + w]             # [window, N_stocks]
        y_raw = self.raw[idx + w]                 # [N_stocks] — next day raw return
        y = (y_raw > 0).astype(np.float32)        # 1=UP, 0=DOWN

        return (
            torch.tensor(x, dtype=torch.float32).T,   # [N, window]
            torch.tensor(y, dtype=torch.float32),      # [N]
        )


def make_datasets():
    """
    Download data, split temporally, normalize, return datasets.
    Returns: (train_dataset, test_dataset, n_stocks, info_dict)
    """
    prices = download_prices()
    log_ret = compute_log_returns(prices)
    arr = log_ret.values     # [T, N]
    n_stocks = arr.shape[1]

    split = int(len(arr) * config.TRAIN_RATIO)
    train_raw = arr[:split]
    test_raw = arr[split:]

    # Normalize using training stats only
    mean = train_raw.mean()
    std = train_raw.std()
    train_norm = (train_raw - mean) / std
    test_norm = (test_raw - mean) / std

    train_ds = StockDataset(train_norm, train_raw, config.WINDOW_SIZE)
    test_ds = StockDataset(test_norm, test_raw, config.WINDOW_SIZE)

    # Class balance info
    all_labels = (arr[config.WINDOW_SIZE:, :] > 0).mean()

    info = {
        "n_days": len(arr),
        "n_stocks": n_stocks,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "class_balance": f"{all_labels:.1%} positive",
        "tickers": config.TICKERS,
    }

    return train_ds, test_ds, n_stocks, log_ret, info
