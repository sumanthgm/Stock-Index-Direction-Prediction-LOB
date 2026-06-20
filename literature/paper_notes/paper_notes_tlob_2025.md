# Paper Notes: TLOB — A Novel Transformer Model with Dual Attention for LOB Price Trend Prediction

**Full title:** TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data
**Authors:** Leonardo Berti (Sapienza University of Rome), Gjergji Kasneci (TU Munich)
**Published:** arXiv 2502.15757, May 2025
**File:** `[N1] TLOB_Berti_2025.pdf`

---

## Why This Paper Matters

This paper does two things that matter directly for your capstone:

1. **Introduces TLOB and MLPLOB** — new SOTA architectures that beat every prior model on FI-2010 and on recent NASDAQ and BTC data. TLOB is the new reference architecture for "recent AI/ML" in LOB forecasting.

2. **Empirically shows predictability is declining.** When testing the same model on Intel data from 2012 vs. 2015, F1 dropped by **6.68 points**. This confirms what Briola 2024 theorized: as patterns become known and arbitraged, alpha disappears over time.

---

## What Problem Does It Solve?

Prior SOTA models (Axial-LOB, BINCTABL, DeepLOB) had two issues:
- **Robustness:** They often fail to replicate on new datasets (shown by Prata 2023's benchmark).
- **Architectural complexity:** CNN-LSTM or axial-attention models were tuned for the specific FI-2010 structure.

This paper proposes simpler, cleaner alternatives grounded in recent deep learning advances.

---

## The New Labeling Method

Standard labeling (Zhang 2019) smooths only future prices:
```
l(t) = (avg future mid-prices − current mid-price) / current mid-price
```

**The problem:** The smoothing window k coincides with the prediction horizon h. For h=2, k=2 means almost no denoising. This introduces "horizon bias" — model performance is artificially linked to the window length.

**The fix (this paper):** Separate k and h:
```
w+(t, h, k) = (1/(k+1)) × sum of mid-prices from t+h to t+h+k  [future window]
w−(t, h, k) = (1/(k+1)) × sum of mid-prices from t-k to t      [past window]
l(t, h, k) = (w+(t,h,k) − w−(t,h,k)) / w−(t,h,k)
```

This makes k and h independently tunable — you can use a large denoising window k even at short horizons h, and vice versa.

**For FI-2010:** They retain the original labels for direct comparison with prior work. For TSLA and INTC, they use the new method.

---

## Model 1: MLPLOB

**Motivation:** Prata 2023's benchmark showed that simpler models often match complex ones when properly tuned. Inspired by MLP-Mixer (Tolstikhin 2021) which achieves near-ViT performance in vision with only MLPs.

### Architecture

Input: sequence of T LOB snapshots × N features → tensor **X** ∈ ℝ^(T×N)

Two types of MLP layers alternate:

**Feature-Mixing MLP** (applied row by row, i.e., per time step):
```
U_{i,*} = σ(LayerNorm(σ(X_{i,*} W1) W2 + X_{i,*}))   for i = 1...T
```
This mixes features across the N LOB dimensions at each timestep.

**Temporal-Mixing MLP** (applied column by column, i.e., per feature):
```
Z_{*,j} = σ(LayerNorm(σ(U_{*,j} W3) W4 + U_{*,j}))   for j = 1...N
```
This mixes information across the T time steps for each feature.

After several such alternating blocks:
- Dimensionality reduction: flatten all features to a single vector
- Fully connected layers gradually reduce dimension
- Softmax over 3 classes (Up/Down/Stable)

**Key design choice: Isotropic.** Each block maintains the same dimensionality (N is constant throughout). This contrasts with CNN pyramids that shrink spatial resolution while increasing channels.

**Bilinear Normalization Layer** (borrowed from BiNCTABL, Tran 2021): Applied as the first layer. Unlike z-score normalization which uses fixed training statistics, bilinear normalization adapts to batch-specific statistics. This makes the model robust to distribution shifts at test time — a key advantage when deploying on new data.

### Why MLPLOB Is Interesting

The fundamental insight: "Spatial and temporal relationships in LOB data are critical" (Sirignano & Cont), and an MLP can capture them explicitly through separate feature-mixing and temporal-mixing, without needing the inductive biases of CNN spatial locality or LSTM sequential processing.

---

## Model 2: TLOB

### Architecture

Each TLOB block contains three components:

**1. Temporal Self-Attention:**
```
Input: T LOB snapshots as sequence
Query/Key/Value projections over the T-dimension
Output: attention weights between different time steps
```
This captures *when* patterns matter — e.g., which of the 100 past snapshots is most relevant to the current prediction.

**2. Spatial (Feature) Self-Attention:**
```
Input: N LOB features at each snapshot
Query/Key/Value projections over the N-dimension
Output: attention weights between different LOB features
```
This captures *which* price-volume pairs are most informative — e.g., which of the 40 features (10 ask prices, 10 ask volumes, 10 bid prices, 10 bid volumes) carry predictive information.

**3. MLPLOB block:**
The standard Transformer feed-forward network is replaced by an MLPLOB block. This gives the model both the structured attention of a Transformer and the mixing capability of the MLPLOB.

**Additional components:**
- **Bilinear Normalization:** Applied before each TLOB block to handle non-stationarity.
- **Sinusoidal Positional Encoding:** Added before attention to preserve chronological order (since self-attention is permutation-invariant without this).
- **Skip connections:** Standard residual connections around each block.

The architecture is stacked N times (N=4 in experiments; ablation showed no benefit from more heads — 1 head optimal, chosen for simplicity).

### Why Dual Attention?

LOB data varies along two axes:
- **Time:** Snapshot at t is different from snapshot at t-50 (temporal dependencies)
- **Features:** Ask price level 1 is different from bid volume level 5 (spatial dependencies)

Standard Transformers process tokens along a single axis. TLOB explicitly models both. This is similar in spirit to Axial-LOB's factored attention, but TLOB applies it as separate self-attention modules rather than axial attention blocks.

**Ablation (Table 9):** Removing either attention mechanism reduces performance:
```
TLOB w/o Spatial Attention:  h=10: 79.59, h=50: 87.51, h=100: 91.40
TLOB w/o Temporal Attention: h=10: 80.27, h=50: 87.72, h=100: 91.42
Full TLOB:                   h=10: 81.55, h=50: 90.03, h=100: 92.81
```
Both attention types contribute complementary information.

---

## Datasets

| Dataset | Description | Stocks | Period |
|---|---|---|---|
| FI-2010 | Finnish stocks, NASDAQ Nordic | 5 stocks | June 2010 |
| TSLA-INTC | LOBSTER data | Tesla + Intel | January 2015 |
| BTC | Binance perpetual futures | BT-CUSDT.P | Jan 9–20, 2023 |

TSLA and INTC are sampled every 500 trades (volume-based sampling, not event-based) to maintain temporal consistency while avoiding noise from ultra-high-frequency sampling.

---

## Key Results

### FI-2010 Results (Table 3)

| Model | h=10 | h=20 | h=50 | h=100 |
|---|---|---|---|---|
| DeepLOB | 71.62 | 75.4 | 87.1 | 77.6 |
| AXIALLOB | 73.2 | 63.4 | 78.3 | 79.2 |
| BINCTABL | 81.1 | 71.5 | 87.7 | 92.1 |
| **MLPLOB** | **81.64** | **84.88** | **91.39** | **92.62** |
| **TLOB** | **81.55** | **82.68** | **90.03** | **92.81** |

Both MLPLOB and TLOB outperform all prior work. MLPLOB is stronger at short horizons; TLOB catches up at longer horizons.

### TSLA Results (Table 4)

| Model | h=10 | h=20 | h=50 | h=100 |
|---|---|---|---|---|
| DeepLOB | 36.25 | 36.58 | 35.29 | 34.43 |
| BINCTABL | 58.69 | 48.83 | 42.23 | 38.77 |
| MLPLOB | 60.72 | 50.25 | 38.97 | 32.95 |
| **TLOB** | 60.50 | 49.74 | **43.48** | **39.84** |

All models perform much worse on TSLA (small-tick, volatile) than on FI-2010. TLOB better handles the longer horizons.

### INTC Results (Table 5)

| Model | h=10 | h=20 | h=50 | h=100 |
|---|---|---|---|---|
| DeepLOB | 68.13 | 63.70 | 40.3 | 30.1 |
| BINCTABL | 72.65 | 66.57 | 53.99 | 41.08 |
| **MLPLOB** | **81.15** | **73.25** | 55.74 | 43.18 |
| **TLOB** | 80.15 | 72.75 | **62.07** | **50.14** |

INTC is a large-tick stock — much higher performance. TLOB's advantage at longer horizons (h=50, 100) is clear.

### BTC Results (Table 6)

| Model | h=10 | h=20 | h=50 | h=100 |
|---|---|---|---|---|
| BINCTABL | 73.4 | 61.34 | 47.05 | 40.59 |
| **MLPLOB** | 74.6 | 61.02 | 42.74 | 36.97 |
| **TLOB** | **74.7** | **61.74** | **48.54** | **41.49** |

TLOB consistently wins on BTC — the newest and most efficient dataset (2023).

### Historical Comparison: Is Predictability Declining? (Table 7)

| Period | INTC F1 (h=50) |
|---|---|
| 2012 | 66.87 |
| 2015 | 60.19 |
| Decline | **−6.68** |

This confirms the **Efficient Market Hypothesis prediction**: as trading algorithms exploit patterns, predictability decays. The Indian market may actually show higher predictability than NASDAQ currently, if it's less efficient.

### Alternative Threshold: Setting θ = Average Spread (Table 8)

For TSLA with θ set to average spread (reflecting actual transaction costs):
```
h=50: 41.39 → significant drop from 38.97 (when θ balances classes)
h=100: 36.48
h=200: 30.82
```

When the threshold reflects real transaction costs rather than class balance, performance drops significantly. This underscores the gap between ML accuracy and trading profitability — same message as Briola 2024's p_T metric.

---

## Model Complexity

| Model | Parameters | Inference Time (ms) |
|---|---|---|
| DeepLOB | 1.4×10⁵ | 1.31 |
| BINCTABL | 1.1×10⁴ | 0.71 |
| **MLPLOB** | **6.3×10⁷** | **4.79** |
| **TLOB** | **1×10⁷** | **2.24** |

MLPLOB and TLOB have more parameters but are competitive on inference time. They converge in less than half the epochs required by BINCTABL and DeepLOB.

---

## Key Conclusions

1. **MLPLOB and TLOB set new SOTA** on FI-2010, TSLA-INTC, and BTC — beating all prior architectures.

2. **MLPLOB wins at short horizons (h=10, 20); TLOB wins at longer horizons (h=50, 100).** This is consistent with Transformers being better at long-range dependencies.

3. **Dual attention (temporal + spatial) is necessary.** Ablation confirms removing either component hurts.

4. **Predictability has declined -6.68 F1 over 3 years** for Intel. Markets are self-correcting.

5. **When θ = average spread (realistic cost), performance drops substantially.** Classification accuracy ≠ trading profitability.

6. **Bilinear normalization is critical** for generalizability across market conditions.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| TLOB is new SOTA | Implement TLOB (or simplified version) as "recent AI/ML" component |
| MLPLOB is competitive simple baseline | Replace the "MLP" baseline in your proposal with MLPLOB |
| Dual attention (temporal + spatial) | The spatial attention is especially interesting for NSE — different bid-ask structure may require different spatial patterns |
| Predictability declined in NASDAQ | NSE may have *higher* predictability currently (less algorithmic trading, less efficient market) — test this hypothesis |
| θ = average spread test | Compute NSE average spread and set θ accordingly — crucial for practical significance |
| TLOB converges faster | Practical advantage given limited compute for capstone |
| FI-2010 performance ≠ real-world performance | Don't report only FI-2010 numbers; test on NSE data which is the whole point |
