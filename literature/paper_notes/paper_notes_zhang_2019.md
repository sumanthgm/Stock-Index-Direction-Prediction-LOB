# Paper Notes: DeepLOB — Deep Convolutional Neural Networks for Limit Order Books

**Full title:** DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
**Authors:** Zihao Zhang, Stefan Zohren, Stephen Roberts (University of Oxford)
**Published:** IEEE Transactions on Signal Processing 67(11):3001–3012, June 2019
**arXiv:** 1808.03668 | **IEEE:** 10.1109/TSP.2019.2907260
**File:** `[15] DeepLOB_Zhang_2019.pdf`

---

## Why This Paper Matters

This is *the* foundational architecture paper for LOB forecasting with deep learning. Every subsequent paper in this reading list — Briola 2024, Lucchese 2022, Kolm 2021 — either uses DeepLOB directly as a baseline or builds on top of its ideas. You cannot understand any of the other papers without first understanding this one.

---

## The Prediction Task

Same as Briola 2024 (because Briola 2024 *uses* DeepLOB):

- **Input:** 100 consecutive LOB snapshots → a `100 × 40` matrix
- **Output:** Predict whether mid-price goes Up (+1), Down (-1), or stays Stable (0)
- **Horizons:** k ∈ {10, 20, 30, 50, 100} events ahead

---

## The DeepLOB Architecture

```
Input: 100 × 40 tensor
   ↓
[CNN Block 1] — 32 filters, LeakyReLU
   ↓
[CNN Block 2] — 32 filters
   ↓
[Inception Module] — multi-scale (1×1, 3×1, 5×1 parallel)
   ↓
[LSTM, 64 hidden units]
   ↓
[Dense, 3 outputs] — Softmax → P(Down), P(Stable), P(Up)
```

### CNN Blocks: Spatial Feature Extraction

The LOB snapshot (at any one moment) is a row of 40 numbers: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ..., bid_v10]. The CNN treats this row like a 1D signal and slides filters *across the level dimension*.

What the CNN learns: relationships between adjacent price levels. For example:
- "Volume at best bid is much larger than at levels 2-3" (imbalance building)
- "Ask side levels 1-4 all have small volume" (shallow ask book — move imminent)

Two CNN blocks are stacked with LeakyReLU activation. Each block uses 32 filters.

### Inception Module: Multi-Scale Features

Borrowed from GoogLeNet (computer vision). Three parallel filter sizes are applied simultaneously and their outputs concatenated:
- **1×1 filters:** point features (single-level)
- **3×1 filters:** local patterns spanning 3 levels
- **5×1 filters:** broader patterns spanning 5 levels

This lets the model look at the LOB at different "zoom levels" at once — some signals are narrow (best bid vs. best ask), others are broad (pressure building across first 5 levels).

### LSTM: Temporal Pattern Learning

After CNN+Inception, each of the 100 time steps has been reduced to a feature vector. The LSTM processes this sequence of 100 feature vectors.

- 64 hidden units
- Only the **last hidden state** is used (not all 100 — we want one prediction)
- The LSTM captures dynamics that build over time: "bid side volume has been steadily draining for the last 20 snapshots — a move is coming"

### Dense + Softmax

Final layer with 3 outputs. Softmax normalizes to a probability distribution over Down/Stable/Up.

---

## Normalization: The Critical Design Choice

**5-day rolling window z-score** — this is one of DeepLOB's key contributions, often overlooked.

### Why NOT global normalization

If you compute mean and std over the entire dataset and normalize everything by those values:
- The normalization statistics include information from the future
- You leak tomorrow's data into today's training — this is **lookahead bias**
- This is exactly the flaw in the FI-2010 benchmark dataset

### Why rolling 5-day window

For each feature at time t, compute:
```
x_normalized = (x - mean(last 5 trading days)) / std(last 5 trading days)
```

This gives you: "how far is this value from the recent local average, relative to recent local variability." That's a meaningful, stationary signal. Five days ≈ one trading week, which is enough data for stable statistics.

The same rolling z-score is applied to all 40 features (prices and volumes at all 10 levels).

---

## Training Details

| Parameter | Value |
|---|---|
| Lookback window T | 100 LOB snapshots |
| Prediction horizons k | {10, 20, 30, 50, 100} events |
| Batch size | 32 |
| Optimizer | Adam |
| Max epochs | 200, early stopping |
| Framework | Keras / TensorFlow |
| Parameters | ~60,000 |

~60K parameters is small by modern standards — this model can be trained on a consumer GPU (or even a MacBook with MPS) in hours.

---

## Datasets Used

### FI-2010 (Finnish Stocks)

Standard benchmark: 5 Finnish stocks, Helsinki exchange (NASDAQ Nordic), June 1-14 2010, ~395K events. All subsequent papers in this reading list reference this benchmark.

DeepLOB achieves state-of-the-art on FI-2010, outperforming:
- SVM
- Plain MLP
- Ridge Regression
- Previous CNN models

(See the FI-2010 paper notes for why you should treat these numbers with skepticism.)

### London Stock Exchange (LSE) Data

5 UK stocks, different exchange from FI-2010. This tests generalization — can a model trained on Finnish stocks perform well on UK stocks?

**Result:** Yes, reasonably well. The architecture generalizes across exchanges.

---

## Interpretability: LIME Analysis

LIME (Local Interpretable Model-agnostic Explanations) works by: given a specific prediction, perturb the input and see which features most change the output. These are the "important" features for that prediction.

**Key finding from LIME:**
- **Level 1 features (best bid and ask prices/volumes) are by far the most important**
- Level 2-4 features contribute some information
- Levels 5-10 contribute very little

This makes intuitive sense. In the short term, a price move happens when the queue at the best bid (or ask) is depleted. You can see this coming by watching level 1 volume drain. Levels 8-10 are rarely touched in a H=10 or H=50 horizon.

**Implication:** If compute is a constraint, using only L1 data (4 features instead of 40) may sacrifice relatively little performance — this is exactly what Lucchese 2022 tests.

---

## Transfer Learning Experiment

The paper trains a model on 4 stocks and tests on an unseen 5th stock.

**Result:** Transfer learning works — the model predicts the unseen stock significantly better than chance.

This means DeepLOB learns general LOB dynamics, not just stock-specific quirks. However, Kolm 2021 later shows that performance is still strongly tied to stock microstructure (large-tick stocks are easier regardless). Transfer is better within the same liquidity class.

---

## What DeepLOB Gets Right vs. Later Limitations

| Strength | Limitation (identified by later papers) |
|---|---|
| Rolling normalization — correct approach | Small R²: Actual predictive power is low; real trading gains are uncertain |
| Multi-scale features via Inception | Spatial structure of LOB invalid for small-tick stocks (Briola 2024) |
| Transfer across stocks works | Performance highly tied to tick size / liquidity (Kolm 2021) |
| ~60K params — lightweight | Using non-stationary LOB prices as input is suboptimal vs. OFI (Kolm 2021) |
| FI-2010 SOTA | FI-2010 benchmark itself has lookahead bias — inflated numbers |

---

## Indian Market Relevance

| NASDAQ / LSE (paper) | Indian NSE context |
|---|---|
| L=10 levels, 40 features | NSE L2 data: same format |
| 5-day rolling window | May need shorter window for more volatile NSE conditions |
| ~60K parameters | Trainable without HPC; GPU not strictly required for a few stocks |
| Level 1 most important (LIME) | Likely same for NSE large-tick stocks (HDFC, SBI, etc.) |
| Transfer learning across exchanges | Potential transfer within same NSE sector |
