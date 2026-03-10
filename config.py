"""
config.py — All shared settings in one place.
Every model uses the same window, split, stocks, and training setup.
"""

# Data
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# Shared across ALL models
WINDOW_SIZE = 30          # 30 trading days of history
TRAIN_RATIO = 0.8         # 80% train, 20% test (temporal split)
BATCH_SIZE = 32

# Training
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Model dimensions (shared where applicable)
D_MODEL = 32              # embedding dimension
N_HEADS = 4               # attention heads (Transformer + GAT)
N_LAYERS = 2              # encoder layers (Transformer)

# Graph
CORR_THRESHOLD = 0.3      # minimum |correlation| for an edge
