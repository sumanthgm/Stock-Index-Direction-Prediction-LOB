# Paper Notes: FI-2010 — Benchmark Dataset for Mid-Price Forecasting

**Full title:** Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods
**Authors:** Ntakaris et al.
**Published:** Journal of Forecasting 37(8):852–866, 2018
**arXiv:** 1705.03233
**File:** `[70] FI2010_Ntakaris_2018.pdf`

---

## Why This Paper Matters

FI-2010 is the **standard benchmark dataset** for LOB mid-price prediction. Every paper in this reading list reports results on it or explicitly comments on its limitations. You need to understand both what it is *and* what's wrong with it.

The short version: FI-2010 made the field comparable for the first time, but it has serious methodological flaws that inflate all performance numbers. Never treat FI-2010 results as ground truth for real-world predictability.

---

## The Dataset

### Raw data properties

| Property | Value |
|---|---|
| Exchange | NASDAQ Nordic (Helsinki, Finland) |
| Stocks | 5 Finnish companies: Kesko (KESBV), Outokumpu (OUT1V), Sampo (SAMPO), Nokia (NOK1V), Rautaruukki (RTRKS) |
| Period | June 1–14, 2010 — just **10 trading days** |
| Total events | ~395,000 after sampling |
| LOB depth | 10 price levels (same as Briola 2024) |
| Sampling rate | Every **10 order book events** → 10-event interval snapshots |

### Data format

Each record is one LOB snapshot:
```
[ask_price_1, ask_vol_1, bid_price_1, bid_vol_1,
 ask_price_2, ask_vol_2, bid_price_2, bid_vol_2,
 ...                                           ,
 ask_price_10, ask_vol_10, bid_price_10, bid_vol_10]
= 40 features
+ labels for 5 horizons
```

---

## Prediction Horizons

FI-2010 uses 5 prediction horizons, all defined in *event count*:

| Horizon k | Events ahead | Approximate wall time |
|---|---|---|
| k=1 | 10 events | < 1 second |
| k=2 | 20 events | ~ 1 second |
| k=3 | 30 events | ~ 1–2 seconds |
| k=5 | 50 events | ~ 2–5 seconds |
| k=10 | 100 events | ~ 5–10 seconds |

These map directly to the H10/H50/H100 notation used in Briola 2024 (which uses raw LOB updates, not 10-event batches).

---

## Label Definition

The label uses **smoothed mid-price**:

```
m_t = (best_ask + best_bid) / 2   at time t

m_smooth(t, k) = average of mid-price over the next k snapshots

Label:
  UP     (+1)  if  m_smooth(t,k) - m_t  > α
  DOWN   (-1)  if  m_smooth(t,k) - m_t  < -α
  STABLE  (0)  otherwise
```

The smoothing (averaging over k future values rather than taking one future point) reduces noise in labels. A single future mid-price can spike due to a large temporary order; the smoothed version better represents the true short-term direction.

The threshold α is chosen to balance classes.

---

## Three Normalization Schemes — And Why All Three Are Flawed

The dataset was released in three pre-normalized versions:

| Scheme | Method | Problem |
|---|---|---|
| **Z-score (v0)** | Global mean and std over entire dataset | **Lookahead bias** |
| **Min-Max (v1)** | Scale to [0,1] using global min and max | **Lookahead bias** |
| **Decimal Precision (v2)** | Divide by a power of 10 | Least affected but still global |

### The lookahead bias problem

All three schemes compute normalization statistics over the *entire dataset*, including the future. This means:

When normalizing the data at time t on day 1, the statistics already incorporate prices from days 2-10. In practice, you wouldn't have access to this information. It is as if your model on Monday knows the average price of the stock for the entire next two weeks.

This makes models look much better than they would in real deployment. Since the raw LOB data cannot be reconstructed from the normalized FI-2010 format, you cannot retrospectively fix this with rolling normalization.

**DeepLOB (Zhang 2019) introduced the 5-day rolling z-score specifically to fix this problem on fresh data.**

---

## Baseline Models

The paper establishes initial baselines:

| Model | MCC (best horizon) |
|---|---|
| Ridge Regression | ~0.40 on z-score version |
| RBF SVM | ~0.43 |
| 2-layer MLP | ~0.42 |

These were quickly overtaken. DeepLOB achieved MCC ~0.60+ on the same dataset — but again, inflated by global normalization.

---

## Cross-Validation Strategy: Anchored Day-Based CV

The paper uses **anchored expanding-window CV**:

```
Fold 1: Train on day 1, test on day 2
Fold 2: Train on days 1-2, test on day 3
...
Fold 9: Train on days 1-9, test on day 10
```

The window always starts at day 1 and expands. This is called "anchored" because the start is fixed.

**Why not rolling window CV?** Rolling window uses a fixed-length window (e.g., always last 5 days for training). This prevents the model from benefiting from older data. Anchored CV uses all available history, which is more realistic for financial models (more historical data is generally better).

---

## The Five Known Limitations of FI-2010

Identified across subsequent papers (Briola 2024, Lucchese 2022):

1. **Only 10 trading days** — one unusual day can dominate results. Not representative of normal market conditions.

2. **Only 5 Finnish stocks** — Helsinki exchange is a small, relatively inactive market. Dynamics may not generalize to NASDAQ, LSE, NSE.

3. **Global normalization with lookahead bias** — inflates all reported numbers by an unknown but significant amount.

4. **Cannot be reconstructed** — the pre-processed format doesn't expose raw LOB data. You cannot apply better normalization (like rolling z-score) retroactively.

5. **The benchmark has been "overfit" by the community** — thousands of models have been tuned and selected specifically for FI-2010. Some improvement in reported numbers is architectural; some is just overfitting to idiosyncrasies of these specific 10 days of Finnish stocks.

---

## What FI-2010 Contributed (Despite Limitations)

Before FI-2010, every paper used different stocks, time periods, preprocessing, and evaluation setups. No two papers could be meaningfully compared. FI-2010:

- Gave the community a common ground for comparison
- Drove rapid development of progressively better architectures (CNN-LSTM, attention, etc.)
- Standardized the prediction task: 3-class, event-based horizons, 40 LOB features

The field now recognizes the need for better benchmarks. Briola 2024's LOBFrame is the current best attempt at a rigorous, reproducible framework.

---

## Comparison: FI-2010 vs. LOBSTER-based datasets

| Property | FI-2010 | LOBSTER (Briola 2024 / Lucchese 2022) |
|---|---|---|
| Period | 10 days | 1-3 years |
| Stocks | 5 Finnish | 10-15 NASDAQ |
| Normalization | Global (flawed) | Rolling 5-day (correct) |
| Raw data available | No | Yes |
| Reproducibility | Fixed format | Replicable pipeline (LOBFrame) |
| Known ceiling effect | Yes | No |

---

## Indian Market Connection

There is currently **no Indian equivalent of FI-2010**. This is both a gap and an opportunity:

**Gap:** You can't benchmark your NSE model against existing literature numbers — no common reference exists.

**Opportunity:** Creating a clean, properly normalized benchmark dataset from NSE LOB data — and releasing the evaluation pipeline — would itself be a genuine contribution to the literature.

If your team can obtain NSE L2 data (even 3 months of data for 5-10 liquid Nifty 50 stocks) and build a clean train/val/test evaluation pipeline with rolling normalization, you would be building exactly what FI-2010 failed to provide: a rigorous, reproducible benchmark for Indian markets.
