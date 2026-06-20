# Paper Notes: The Short-Term Predictability of Returns in Order Book Markets

**Full title:** The Short-Term Predictability of Returns in Order Book Markets
**Authors:** Lucchese, Pakkanen, Veraart (Imperial College London)
**Published:** arXiv 2211.13777, 2022
**File:** `[74] ShortTermPredictability_Lucchese_2022.pdf`

---

## Why This Paper Matters

This is the most methodologically rigorous paper in the reading list. It answers the question "does LOB data actually help predict returns?" using proper statistical inference (Model Confidence Sets) rather than just reporting point estimates. It also introduces **deepVOL** — a volume-based representation that is more robust than price-level representations — and a **seq2seq encoder-decoder** for multi-horizon forecasting.

**Key distinction from Briola 2024:** This paper asks *whether* LOB data is predictive and *which representation works best*. Briola 2024 asks *which stocks* are predictable and how to bridge the gap to actual trading.

---

## Four Research Questions

The paper explicitly frames four questions:

1. Do high-frequency returns display predictability? If so, how far ahead?
2. Which LOB representation performs best?
3. Can a single model serve multiple prediction horizons (multi-horizon)?
4. Can a single universal model (trained on multiple stocks) generalize to unseen stocks?

---

## Three LOB Representations

This is the paper's central methodological contribution. The same underlying data can be represented three ways, with very different model performance.

### L1 / L2 — deepLOB (Standard Price-Volume)

You already know this one from Zhang 2019 and Briola 2024.

- **L1:** Only level 1 (best bid/ask) → 4 features per snapshot
- **L2:** All 10 levels → 40 features per snapshot
- Normalization: 5-day rolling z-score

### deepOF — Order Flow Representation

Instead of raw prices and volumes, compute the *change* between consecutive LOB snapshots.

For each level i, define:

**Bid Order Flow (bOF):**
```
bOF_t^i =  v_t^{i,bid}              if bid price moved UP (new quote)
           v_t^{i,bid} - v_{t-1}^{i,bid}  if bid price UNCHANGED
          -v_{t-1}^{i,bid}           if bid price moved DOWN (quote removed)
```

Symmetrically for ask: aOF_t^i

Concatenate: OF_t = [bOF_t^1..10, aOF_t^1..10] ∈ ℝ^20

**Why this is better than raw LOB:**
- Raw prices are non-stationary (they drift with stock price)
- Order flow is stationary — it measures the net change, not the absolute level
- Stationary inputs → better-behaved gradients → better training
- OFI (bOF - aOF) is well-known to contemporaneously predict returns (Cont et al.)

### deepVOL — Volume/Price-Grid Representation

The problem with L2 deepLOB: the spatial structure of the LOB is inconsistent across time for small-tick stocks.

Today: spread = 5 ticks. "Level 3" = 3 ticks from best quote.
Tomorrow: spread = 15 ticks. "Level 3" = 3 ticks from best quote, but the market has emptied levels 4-8 between levels 3 and 9.

For wide-spread stocks, the LOB has empty levels, and the spatial meaning of "position 3 in the array" changes day to day. The CNN, which assumes spatial locality means something, is confused.

**deepVOL fixes this by using a price-grid representation:**

Instead of "levels 1-10 from best quote," use a window of W consecutive price ticks centered on the mid-price. Each cell in the window has a fixed price distance from mid:
```
Cell 1 = mid - W/2 ticks
Cell 2 = mid - W/2 + 1 tick
...
Cell W = mid + W/2 ticks
```

The value in each cell = total volume resting at that price tick. If no orders exist there, the cell is zero.

```
deepVOL input shape: T × W × 2
  T = 100 time steps
  W = window size (e.g., 10 price ticks on each side)
  2 = bid and ask sides
```

**Why deepVOL is more robust:**
- "Position 3" always means "3 ticks from mid" — consistent spatial meaning
- Empty cells represent genuine absence of liquidity, not a data artifact
- The representation is robust to small perturbations (volume at neighboring ticks is correlated)
- This last property helps universal models (trained on multiple stocks) generalize better

### L3 deepVOL — Queue-Level Data

Extends deepVOL to track not just total volume per price tick but the individual orders in the queue:
- At price tick j, instead of one aggregate volume, you have a list of order sizes: [2516, 2000, 1484, ...]
- The paper uses the first D orders in each queue (D=10)
- A CNN aggregates the queue into a weighted sum before passing to the standard deepVOL pipeline

L3 data requires the full ITCH message feed, not just reconstructed order book snapshots.

---

## Multi-Horizon Forecasting: Seq2Seq Architecture

### The Problem with Single-Horizon Models

Training a separate model per horizon h is wasteful:
- If you want predictions at h ∈ {10, 20, 30, 50, 100, 200, 300, 500, 1000}, you need 9 separate models
- Each model is trained in isolation and ignores the fact that predictions across horizons should be *coherent*

### The Encoder-Decoder Solution

```
Input LOB sequence (100 time steps)
          ↓
    CNN + Inception
          ↓
      LSTM Encoder (64 hidden units)
          ↓
    Context vector z_t  (last LSTM hidden state)
          ↓
   ┌──── Decoder (LSTM) ────────────────────────┐
   │   z'_0 = z_t                               │
   │   For k = 1, 2, ..., K:                    │
   │     z'_k = LSTM(z'_{k-1}, p_{k-1})         │
   │     p_k  = softmax(Dense(z'_k))            │
   └────────────────────────────────────────────┘
   Output: p_1, p_2, ..., p_K  (one distribution per horizon)
```

The decoder autoregressively rolls forward: prediction at horizon k feeds into the state for predicting horizon k+1. This means the model learns that predictions at consecutive horizons are related.

This architecture is borrowed from neural machine translation (Cho et al. 2014, "Learning phrase representations using RNN encoder-decoder").

**Why this helps:** For a given input, predicting multiple horizons simultaneously gives the model more supervision signal per training step. The model learns a richer map from inputs to "order book regimes" than any single-horizon model can.

---

## Statistical Framework: Model Confidence Sets (MCS)

Standard ML practice: "Model A achieves MCC=0.36, Model B achieves MCC=0.34, therefore A is better." But is the difference statistically significant? Could it be noise from the specific test period?

MCS provides a rigorous answer.

### The Setup

For W time windows and a set of models M₀:
- Define L_{i,w} = test loss of model i in window w (categorical cross-entropy)
- For any two models i, j: d_{ij,w} = L_{i,w} - L_{j,w} (relative performance)
- Under the null: E[d_{ij}] = 0 for all pairs (no model is systematically better)

Use bootstrap resampling to test this null for the full set M₀.

### The Algorithm

1. Start with all models in M (initially = M₀)
2. Test if all models in M are equivalent (null: E[d_{i.}] = 0 for all i)
3. If null rejected: eliminate the worst model (argmax of t-statistic)
4. Repeat until null is accepted
5. The remaining set is the **Model Confidence Set** at level 1-α

The MCS is the set of models that cannot be statistically ruled out as best performers.

### What MCS P-values Mean

Each model gets an MCS p-value. If p_{i}^MCS < α, model i is eliminated from the superior set at confidence level 1-α.

**For the unpredictive benchmark** (a model that always predicts the empirical class distribution): if its MCS p-value < 0.01, then at least one order-book model is statistically significantly better. This is the definition of "there is predictability."

---

## Data

- **Universe:** 10 NASDAQ stocks covering a range of liquidity (Table 1 in paper)
  - From LILAK (very illiquid, spread 15.92 bps) to AAPL (spread 0.99 bps)
- **Period:** Jan 2019 – Jan 2020 (1 year)
- **Data source:** LOBSTER (10-level data)
- **Split:** W=11 five-week windows; each window: 4 weeks train+val, 1 week test
- **Horizons:** h ∈ {10, 20, 30, 50, 100, 200, 300, 500, 1000} events

### Normalization

- Price and order flow features: 5-day rolling z-score
- Volume features: max-scaling over the full input array (different from z-score, because volume has natural zero and is heavy-tailed)

### Label

Uses smoothed mid-price return:
```
r_{t,t+h} = (m̄_{t+h}^(5) - m_t) / m_t

where m̄_{t+h}^(5) = (1/11) × sum of mid-prices from t+h-5 to t+h+5
```

Quantile-based threshold γ: empirically chosen per stock, per window, per horizon as the average of the 33rd and 66th percentile of training returns. This ensures roughly balanced classes.

---

## Key Results

### Q1: Is there predictability?

**Yes.** For most stocks, the unpredictive benchmark has MCS p-value < 0.01 up to h=50-300 events. The most illiquid stock (LILAK) shows weaker evidence of predictability.

Horizons of 50-300 events correspond to roughly 0.5 – 10+ seconds of wall time, depending on the stock. This is above HFT latency requirements for most market participants.

### Q2: Best representation?

| Model | % in superior set (α=0.01) |
|---|---|
| deepLOB(L1) | 11% |
| deepOF(L1) | 22% |
| deepLOB(L2) | 11% |
| **deepOF(L2)** | **89%** |
| **deepVOL(L2)** | **84%** |
| deepVOL(L3) | 86% |

**Clear winner: L2 data with order flow or volume representation.** Standard deepLOB (price-based) is rarely in the superior set.

Key insight: **L3 data does not add meaningful value over L2 for stock-specific models.** The extra queue depth granularity doesn't improve predictions. This has practical implications — you don't need the full ITCH feed; the standard reconstructed order book at 10 levels is sufficient.

### Q3: Multi-horizon (seq2seq)?

Yes — seq2seq models almost always outperform single-horizon counterparts. deepOF(L2, seq2seq) is in the superior set 97% of the time (vs. 89% for deepOF(L2) single-horizon).

### Q4: Universal models?

Remarkable result: **a universal model trained on 5 stocks can predict the other 5 unseen stocks.** deepVOL(L3, universal) is in the superior set 100% of the time for universal models.

| Universal model | % in superior set (α=0.01) |
|---|---|
| deepLOB(L1, universal) | 0% |
| deepOF(L2, universal) | 62% |
| deepVOL(L2, universal) | 81% |
| **deepVOL(L3, universal)** | **100%** |

**Why does volume representation generalize better universally?** Volume-based features are robust to small perturbations in price levels. Patterns at "3 ticks from mid" are similar across stocks that have similar liquidity profiles. Order flow, being price-differenced, is also fairly portable — but raw LOB prices are completely stock-specific.

---

## Key Conclusions Summary

| Finding | What It Means |
|---|---|
| Predictability up to ~h=300 | Short-term LOB patterns are real signals, not noise |
| deepOF(L2) >> deepLOB(L2) | Stationarity of input representation matters critically |
| L3 ≈ L2 for stock-specific | Don't overcomplicate data collection |
| seq2seq > single-horizon | Build multi-horizon model from the start |
| Universal models work | One model can serve a universe of stocks |
| Volume representation wins universally | deepVOL is the right architecture for multi-stock models |

---

## Indian Market Implications

| Finding | NSE application |
|---|---|
| deepOF(L2) is consistently best | Compute order flow features from NSE data; don't use raw prices |
| L2 >> L1 | Collect full 10-level LOB, not just best bid/ask |
| L3 ≈ L2 (stock-specific) | No need for queue-level data; NSE L2 is sufficient |
| Universal models work with deepVOL | One model for all Nifty 50 stocks is feasible |
| Predictability at h≈50-300 | At NSE (lower activity than NASDAQ), this may correspond to longer clock time (5-30 seconds?) |
| seq2seq recommended | Build multi-horizon model: predicting h=10, 30, 50 simultaneously |
