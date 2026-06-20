# Paper Notes: Deep Limit Order Book Forecasting

**Full title:** Deep Limit Order Book Forecasting — A Microstructural Guide
**Authors:** Antonio Briola, Silvia Bartolucci, Tomaso Aste (UCL)
**arXiv:** 2403.09267v4, June 2024
**File:** `Deep Limit Order Book Forecasting.pdf`

---

## What is a Limit Order Book?

Think of the LOB as the full, live queue of all unfilled buy and sell orders for a stock at any moment. Every modern exchange (NSE, NASDAQ, etc.) runs one. Unlike looking at just the last traded price, the LOB shows you *intent* — how much demand and supply exists at every nearby price level right now.

### Three types of orders

| Order Type | What it means | Cost | Effect on LOB |
|---|---|---|---|
| **Limit order** | "I'll buy/sell X shares, but only at price P or better" | Low (no guarantee of fill) | Adds liquidity — adds to the queue |
| **Market order** | "Buy/sell X shares at whatever the best current price is" | High (pays the spread) | Takes liquidity — removes from the queue |
| **Cancellation order** | "Delete my pending limit order" | None | Removes from the queue |

### Key prices

- **Best ask**: the lowest price any seller is currently offering (`p1_ask`)
- **Best bid**: the highest price any buyer is currently willing to pay (`p1_bid`)
- **Mid-price**: `(best_ask + best_bid) / 2` — the "theoretical fair price"
- **Bid-ask spread**: `best_ask - best_bid` — the cost of immediately buying *and* selling. If spread = $0.01, that's 1 tick.
- **Tick size (θ)**: the smallest allowed price increment. On NASDAQ, fixed at $0.01.

### The LOB as a data structure

The paper uses 10 price levels on each side. Each LOB snapshot at time τ is:

```
[ask_price_1, ask_volume_1, bid_price_1, bid_volume_1,
 ask_price_2, ask_volume_2, bid_price_2, bid_volume_2,
 ... 10 levels ...] → 40 numbers per snapshot
```

Level 1 = best quotes. Level 2 = second-best, and so on going deeper into the book. The deeper you go, the less likely those prices are to be hit immediately.

---

## The Prediction Task

**Goal:** Given a sequence of 100 consecutive LOB snapshots (the recent past), predict whether the mid-price will go **Up**, **Down**, or stay **Stable** after the next H LOB updates.

The label is defined as:

```
if (m_{τ+H} - m_τ) ≤ -θ  →  Down   (-1)
if |change| < θ            →  Stable  (0)
if (m_{τ+H} - m_τ) ≥ +θ  →  Up     (+1)
```

Three horizons tested: **H10, H50, H100** — meaning 10, 50, or 100 LOB update events into the future. These are measured in "tick-time" (number of order events), not clock time. H10 might happen in under 1 second; H100 might take 10+ seconds, depending on how actively traded the stock is.

---

## The Model: DeepLOB

The key model is **DeepLOB** (Zhang et al. 2019 — key paper to read next). The architecture processes each input as a `100 × 40` matrix:

1. **CNN layers** — Learn *spatial* patterns across the 10 price levels (e.g., "there is a lot of volume building on the bid side at levels 2–4"). CNNs are effective here because they can detect local patterns without being sensitive to their exact position.
2. **Inception module** — Multi-scale convolution: looks at patterns at different "zoom levels" simultaneously.
3. **LSTM (64 units)** — Learns *temporal* dependencies across the 100 time steps. Captures patterns that build over several seconds.
4. **Dense layer (3 outputs)** — Softmax → probability of Down / Stable / Up.

Training details:
- Batch size: 32
- Optimizer: AdamW, learning rate = 6×10⁻⁵
- Max 100 epochs, early stopping with patience = 15
- Framework: PyTorch (via LOBFrame)

---

## The Central Finding: Tick Size Drives Predictability

The paper's most important insight is that a stock's **tick size relative to its spread** determines how predictable it is. They define three classes:

| Class | Rule | Examples (NASDAQ) | LOB behaviour |
|---|---|---|---|
| **Large-tick** | spread ≤ 1.5θ | BAC, CSCO, KO, ORCL, PFE, VZ | Dense, orderly, tight spreads |
| **Medium-tick** | 1.5θ < spread ≤ 3θ | AAPL, ABBV, PM | Borderline |
| **Small-tick** | spread > 3θ | GOOG, CHTR, GS, IBM, MCD, NVDA | Sparse, wide, volatile spreads |

### Why large-tick stocks are more predictable

- Their spread is almost always exactly 1 tick. When the volume at the best bid depletes, the mid-price *must* jump by exactly 1 tick — this is mechanical and learnable.
- Volume at best quotes is high and informative: you can see a price move approaching as volume drains from one side.
- The LOB has a consistent spatial structure — price level 3 always means roughly the same distance from the best quote.

### Why small-tick stocks are hard

- Spread fluctuates widely (can be 20–50 ticks). Empty levels in the LOB are common.
- The spatial structure of LOB snapshots is inhomogeneous — level 3 today might be at a completely different price distance than level 3 tomorrow.
- Patterns the model learns from training do not transfer well to test data.

### MCC results summary

Matthews Correlation Coefficient (MCC) ranges from -1 (inverse prediction) to +1 (perfect), 0 = random.

| Stock type | H10 | H50 | H100 |
|---|---|---|---|
| **Large-tick** | **0.29** | **0.36** | **0.26** |
| Medium-tick | 0.13 | 0.09 | 0.04 |
| Small-tick | 0.11 | 0.04 | ~0.01 (near random) |

Large-tick stocks stay predictable across all horizons. Small-tick becomes essentially random at H100.

---

## The "Simulation-to-Reality" Gap — The Paper's Critical Contribution

This is where the paper goes beyond a standard ML benchmark paper and makes a genuinely important argument.

### The problem with standard metrics

Standard metrics (accuracy, F1, MCC) measure *how many* predictions are correct. For trading, *where* errors occur in the sequence matters far more.

### Illustrative example (Figure 11 in the paper)

| Scenario | MCC | F1 | p_T (correct trades) |
|---|---|---|---|
| A | 0.59 | 0.64 | **0.00** |
| B | 0.11 | 0.29 | **1.00** |

Scenario A has far better ML metrics but executes zero profitable trades. Why? Its errors fall precisely at the transition points (where the signal switches from Down to Up or vice versa) — the exact moments when you need to open or close a position. The model never correctly identifies both the entry *and* exit of a trade.

### The p_T metric

The paper proposes a new metric: **probability of executing a correct transaction**.

```
p_T = CT / (PT + TT - CT)
```

Where:
- **PT** = potential transactions (count of complete open→close cycles in the ground truth labels)
- **TT** = transactions the model attempted based on its predictions
- **CT** = correctly executed transactions (model and ground truth agree on both open and close)

### What p_T reveals in practice

- Small and medium-tick stocks: p_T drops to **0** above a confidence threshold of 0.5 — the model cannot execute a single profitable trade under any reasonable confidence filter.
- Large-tick stocks: p_T remains non-zero even at threshold 0.9, and the probability of a correct trade at threshold 0.7 is ~0.34–0.50 (H50).

### Key takeaway

High accuracy on a test set does not mean your model is useful for trading. You must evaluate whether correct predictions are clustered at the *right* moments in the sequence, not just that there are enough of them in aggregate.

---

## Data Pipeline Design (Practical Notes)

### Normalization
- Uses a **5-day rolling window z-score** per feature — not global normalization.
- The LOB is highly non-stationary. Global normalization leaks future statistics and inflates performance on easy benchmark datasets (a major flaw in papers using FI-2010).

### Balanced sampling
- During training: up to 5000 random samples per class per trading day.
- Prevents the model from just predicting "Stable" all the time (class imbalance problem, especially at shorter horizons for large-tick stocks).

### Data cleaning
- Remove first and last 10 minutes of each trading day (auction dynamics are different).
- Collapse multiple events at the same nanosecond timestamp to the last state.
- Remove crossed quotes (data anomalies where best bid ≥ best ask).

### Train / Validation / Test split (per year)
- **Training**: 45 consecutive trading days
- **Validation**: 5 days randomly sampled (non-consecutive) from within the training period
- **Test**: 10 consecutive trading days after training

---

## Physical Time vs. Tick-Time

All horizons are defined in LOB updates (tick-time), not seconds. The mapping to real time varies by stock:

| Stock type | H10 physical time | H50 physical time | H100 physical time |
|---|---|---|---|
| Large-tick | < 1 second (most) | 1–10 seconds | 1–10 seconds |
| Small-tick | < 1 second (most) | ≥ 10 seconds (most) | ≥ 10 seconds (most) |

**Practical implication:** Acting on H10 forecasts requires low-latency infrastructure (sub-second). This is the domain of HFT. H50 and H100 are more tractable for non-HFT actors.

---

## LOBFrame (Open-Source Framework)

The authors release an open-source Python/PyTorch codebase:
- GitHub: https://github.com/FinancialComputingUCL/LOBFrame
- Includes: data loading, preprocessing, training pipeline, validation, test, trading simulation
- Designed to be modular — you can plug in different forecasting models

---

## What This Means for the Indian Market Capstone

| NASDAQ (paper) | Indian Markets (our project) |
|---|---|
| Fixed tick size $0.01 | Variable tick size bands on NSE (depends on price range of the security) |
| 15 large, liquid stocks | Need to assess liquidity of Nifty 50 / Sensex constituents |
| LOBSTER data provider | NSE's own data feed or third-party vendors |
| 9:40 am – 3:50 pm ET | 9:15 am – 3:30 pm IST |
| No price bands in scope | Indian market has circuit breakers: 5%, 10%, 20% bands |

**Key research question for adaptation:** Do Indian large-tick index stocks (e.g., banking stocks like HDFC, SBI that trade with 1-tick spreads) show the same predictability advantage over small-tick stocks? If so, the paper's framework provides a direct methodology to follow.

---

## Papers to Read Next (Priority Order)

1. **DeepLOB** — Zhang, Zohren, Roberts (2019), IEEE Transactions on Signal Processing 67(11):3001–3012
   - The base model architecture used throughout this paper (CNN + LSTM)
   - arXiv: https://arxiv.org/abs/1808.03668
   - IEEE: https://ieeexplore.ieee.org/document/8673598/

2. **FI-2010 benchmark dataset** — Ntakaris et al. (2018), Journal of Forecasting 37(8):852–866
   - Standard benchmark: 5 Finnish stocks, NASDAQ Nordic, 10 days, ~395K events
   - Limitation: pre-processed and pre-normalized — raw LOB cannot be reconstructed, causes overfitting
   - arXiv: https://arxiv.org/abs/1705.03233

3. **Short-Term Predictability of Returns in Order Book Markets** — Lucchese, Pakkanen, Veraart (2022), arXiv 2211.13777
   - Introduces rolling normalization; key factors for successful LOB forecasting; 'high-frequency stocks'; L2 data
   - arXiv: https://arxiv.org/abs/2211.13777

4. **Deep Order Flow Imbalance** — Kolm, Turiel, Westray (SSRN 3900141)
   - Introduces 'information richness' (IR) score and 'information-rich stocks' concept
   - The Briola 2024 paper shows IR score maps directly to tick size — tick size is the simpler proxy
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141

5. **Deep Learning Modeling of LOB: A Comparative Perspective** — Briola, Turiel, Aste (2020), arXiv 2007.07319
   - Same first author as the Briola 2024 paper; earlier work comparing multiple architectures
   - Key finding: MLP performs comparably to CNN-LSTM, suggesting spatial/temporal decomposition is approximate
   - arXiv: https://arxiv.org/abs/2007.07319
