# Paper Notes: Deep Order Flow Imbalance — Extracting Alpha from the Limit Order Book

**Full title:** Deep Order Flow Imbalance: Extracting Alpha from the Limit Order Book
**Authors:** Petter N. Kolm, Jeremy Turiel, Nicholas Westray (NYU Courant / Man Group)
**Published:** Working Paper, SSRN 3900141, 2021
**File:** `[45] DeepOrderFlowImbalance_Kolm_2021.pdf`

*Note: This file is a conference presentation slide deck (46 slides), not the full paper PDF. The full paper on SSRN contains detailed mathematical derivations and extended results tables. These notes cover what was presented in the slides.*

---

## Why This Paper Matters

This paper makes two critical contributions:

1. **Stationarity of inputs matters more than architecture.** Order flow representation (stationary, derived from differencing LOB states) dramatically outperforms raw LOB prices (non-stationary) as model input. This is perhaps the single most practically actionable insight in this entire reading list.

2. **Stock predictability is explained by a simple microstructural characteristic: Log(Updates/PriceChg).** This provides a quantitative framework for selecting which stocks to trade/model in any market — including Indian NSE.

---

## The Core Problem: Non-Stationarity of LOB Inputs

Consider the regression you're setting up:

```
r_{t+h} = g(x_t, x_{t-1}, ..., x_{t-W}) + ε_t
```

- **LHS (r_{t+h}):** Future return — this is stationary (bounded, mean-reverting)
- **RHS (x_t):** LOB snapshot — contains prices, which are **non-stationary** (they drift with the stock price)

This is econometrically problematic. You have a stationary dependent variable on a non-stationary independent variable. Standard regression theory breaks down. The model may find spurious correlations driven by the price trend rather than the true LOB signal.

**The fix:** Use Order Flow (OF) as input instead of raw LOB. Order flow is computed by differencing consecutive LOB states, which removes the non-stationary price component and yields a stationary signal.

---

## Computing Order Flow

For each price level i, compare two consecutive LOB snapshots (time t and t-1):

**Bid Order Flow (bOF):**
```
bOF_t^i = 
  v_t^{i,bid}                      if b_t^i > b_{t-1}^i  (bid price moved UP)
  v_t^{i,bid} - v_{t-1}^{i,bid}    if b_t^i = b_{t-1}^i  (bid price UNCHANGED, volume changed)
 -v_{t-1}^{i,bid}                   if b_t^i < b_{t-1}^i  (bid price moved DOWN)
```

**Ask Order Flow (aOF):** Symmetric for ask side.

**Intuition:**
- Bid price jumps up → new aggressive bid arrived → positive flow
- Bid price drops → bid was cancelled/consumed → negative flow
- Bid price stable but volume increased → new limit order added at existing level → positive flow
- Bid price stable but volume decreased → cancellation or trade at existing level → negative flow

**Order Flow vector:** OF_t = [bOF_t^1, ..., bOF_t^10, aOF_t^1, ..., aOF_t^10] ∈ ℝ^20

**Order Flow Imbalance:** OFI_t = bOF_t - aOF_t ∈ ℝ^10

OFI measures *net pressure* at each level: positive = more buying pressure than selling at that level.

Both OF and OFI are stationary inputs — they don't drift with the stock price.

---

## Model Universe

Six models, each tested with both OF and raw LOB inputs:

| Model | Notes |
|---|---|
| **ARX** | Linear: autoregressive with exogenous features |
| **MLP** | 4-layer, fully connected |
| **LSTM** | 128 hidden units |
| **LSTM-MLP** | LSTM (128) → MLP head (64) |
| **LSTM(3)** | Deep LSTM, 150 hidden units (Sirignano & Cont architecture) |
| **CNN-LSTM** | DeepLOB (Zhang 2019) |

Total: 12 model variants (6 models × 2 input types: OF vs LOB)

---

## Data (Massive Scale)

| Property | Value |
|---|---|
| Universe | 115 NASDAQ stocks |
| Period | Jan 2019 – Jan 2020 (1 year) |
| Data source | LOBSTER + WRDS |
| Total size | ~10TB uncompressed |
| Infrastructure | NYU Greene HPC (32K CPU cores, 332 NVIDIA GPUs), NYU Hudson |

Training configuration:
```
(1, 4, 1) rolling window:
  1 week validation (early stopping)
  4 weeks training
  1 week test
  Step forward by 3 weeks
```

Result: 12 models × 115 stocks × 18 time periods = **24,840+ trained networks**.

This is a much larger empirical study than any previous paper (Briola 2024 uses 15 stocks, Lucchese uses 10). The scale provides statistical power for the cross-sectional analysis.

Computation: training a single model takes 10-60 minutes on one GPU, depending on model and stock.

---

## Prediction Task: Regression, Not Classification

Unlike the other papers, this paper uses **regression**:
- Predict forward returns r_{t,t+h} simultaneously at H horizons: r_t = (r_{t,1}, ..., r_{t,H}) ∈ ℝ^H
- Metric: **out-of-sample R² (R²_OS)**

```
R²_OS = 1 - MSE(model) / MSE(unpredictive benchmark)
```

Where the "unpredictive benchmark" always predicts the training set mean return. If R²_OS > 0, the model explains variance beyond the naive mean. If R²_OS < 0, the model is *worse* than just predicting the mean.

**Horizon definition (stock-specific):**
```
Δt = 2.34 × 10^7 / N
```
where N = average number of non-zero tick-by-tick returns per day. Horizons are fractions of Δt: 1/5 × k × Δt for k = 1, ..., 10. This normalizes for activity differences between stocks.

Winsorization and z-scoring of all inputs and targets.

---

## Key Result #1: OF >> LOB (Figure 3 in slides)

When plotting R²_OS against horizon (in units of average price changes):

**OF models (left panel):** R²_OS consistently positive (0.25–1.25%) across all models and horizons for most stocks.

**LOB models (right panel):** R²_OS **negative for many stocks and horizons** — raw LOB input produces predictions *worse than just predicting the mean*.

This is the most striking result in the paper. Non-stationary LOB prices as input can actively harm model performance. This is not a small degradation — the LOB models frequently achieve negative out-of-sample R².

**Practical lesson:** In any LOB prediction project, never use raw prices as neural network input. Always transform to order flow first.

---

## Key Result #2: LSTM > Non-LSTM, but CNN Adds Little (Slide 33)

Among OF-input models:
- LSTM-based models (LSTM, LSTM-MLP, CNN-LSTM) consistently outperform non-LSTM (ARX, MLP)
- Sequential processing of the 100-step input window is important
- **But CNN-LSTM ≈ plain LSTM** once OF is used

Why does CNN add little over plain LSTM for OF inputs?

The CNN in DeepLOB was designed to extract spatial structure from the price-level grid. With raw LOB input, the CNN provides some implicit stationarization — local differencing in the convolution partially cancels out the non-stationary price component. With explicit OF transformation, this benefit is gone, and the CNN just adds parameters without meaningful benefit.

---

## Key Result #3: Cross-Sectional Analysis — What Predicts Predictability?

This is the most analytically novel part of the paper.

**Setup:** Average each stock's R²_OS across horizons. This gives 115 data points — one per stock. Correlate with four stock characteristics:

| Characteristic | Definition |
|---|---|
| **Tick Size** | Fraction of time the bid-ask spread equals exactly $0.01 (the minimum tick) |
| **LogUpdates** | log(average LOB updates per day) |
| **LogTrades** | log(average trades per day) |
| **LogPriceChg** | log(average price changes per day) |
| **Log(Updates/PriceChg)** | log(updates/day ÷ price changes/day) |

**Correlation matrix reveals (Figure 4):**
- Tick Size ↔ Log(Updates/PriceChg): correlation = 0.95 — nearly identical
- Tick Size ↔ LogPriceChg: correlation = -0.47 — large-tick stocks have fewer price changes per update

**Scatter plots (Figure 5):**
All four characteristics show positive correlation with R²_OS, but the relationship is clearest for Tick Size and Log(Updates/PriceChg).

**Regression analysis (Figure 6):**
```
R²_OS = 0.0089 × log(Updates/PriceChg) - 0.0089

Adjusted R² = 75%
```

**75% of the cross-sectional variation in predictability is explained by this single variable.**

### Why Does Log(Updates/PriceChg) Explain Performance?

High Log(Updates/PriceChg) means: many LOB events happen per price change. This is characteristic of:
- **Large-tick stocks** — the spread is almost always 1 tick; the price can only move by depletion of the level-1 queue. Before the price moves, there is substantial LOB activity (cancellations, new limits) that is *predictive* of the impending move.
- **Stable, orderly books** — the LOB has consistent spatial structure; patterns learned on Monday transfer to Tuesday.

Low Log(Updates/PriceChg) means: price changes happen with few intervening LOB events. This is characteristic of:
- **Small-tick stocks** — spread fluctuates, moves are sudden, less "preparation" is visible in the LOB before a price move.
- **Sparse books** — empty levels, inhomogeneous spatial structure, patterns don't transfer.

**This directly connects to Briola 2024's tick-size classification.** The correlation between Tick Size and Log(Updates/PriceChg) is 0.95 — they are measuring the same underlying property of the stock's microstructure.

---

## Key Conclusions

1. **Stationarity of inputs is critical.** OF >> LOB. This is not architectural — it's about choosing the right input representation.

2. **LSTM > linear > MLP** for sequential LOB data. The sequential structure of the 100-step lookback window matters; don't use MLP for LOB regression.

3. **CNN adds little once you use OF.** The spatial feature extraction of CNN is less valuable when inputs are already stationary and well-structured.

4. **Large-tick stocks are significantly more predictable.** Log(Updates/PriceChg) explains 75% of cross-sectional R² variation.

5. **Alpha exists at all tested horizons.** Even at 2 average price changes ahead, OF models have positive R²_OS for large-tick stocks.

6. **Simple models with the right input > complex models with the wrong input.** This is the "bitter lesson" applied to LOB forecasting.

---

## Indian Market Implications

| Finding | NSE capstone implication |
|---|---|
| OF >> LOB inputs | Always compute order flow from NSE LOB data — never input raw prices |
| LSTM > MLP for regression | Use LSTM (or LSTM-MLP) as your primary architecture |
| CNN adds little after OF transform | Don't over-engineer with CNN if you use OF input |
| Log(Updates/PriceChg) predicts R² | Select NSE stocks using this metric: banking stocks (HDFC, SBI, ICICI) likely have high values |
| 75% of cross-sectional variance explained | Your capstone can replicate this analysis on NSE stocks |
| Alpha at all horizons | Even if your model has small R², it may be actionable — don't dismiss low R² models |
| Horizons normalized by price changes | Define your NSE horizons in terms of price changes, not clock time or event count |
