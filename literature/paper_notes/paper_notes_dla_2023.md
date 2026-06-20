# Paper Notes: DLA — Dual-Stage Temporal Attention for LOB Mid-Price Forecasting

**Full title:** Forecasting the Mid-price Movements with High-Frequency LOB: A Dual-Stage Temporal Attention-Based Deep Learning Architecture
**Authors:** Yanhong Guo, Xinxin Chen
**Published:** Arabian Journal for Science and Engineering, Vol. 48, pp. 9597–9618, 2023
**URL:** https://link.springer.com/article/10.1007/s13369-022-07197-3

---

## Why This Paper Matters

DLA introduces a two-stage temporal attention mechanism for LOB forecasting that improves on the single-attention approach. It appears in the Prata 2023 benchmark (LOBCAST) as one of 15 models and ranks **2nd overall on FI-2010** (generalizability score 76.9% — second only to BINCTABL at 73.5%). It's a direct predecessor to the TLOB architecture (Berti 2025), which also uses dual attention but extends it to the spatial dimension as well.

Understanding DLA is useful because:
1. It established that **two attention stages beat one** — validated by the benchmark
2. TLOB can be understood as "DLA extended to also attend over features, not just time"
3. DLA is already implemented in LOBCAST — easy to use as a baseline

---

## The Problem DLA Solves

Prior LSTM-based LOB models (DeepLOB) process the 100-step input window sequentially. The final hidden state is a compressed summary of all 100 steps — but not all steps are equally informative. A price-moving event at step 95 is more predictive of near-term mid-price change than a quiet period at step 10.

**Single-stage attention (DEEPLOBATT, Zhang 2021)** addresses this by adding attention over encoder hidden states — the model learns to weight each of the 100 time steps by importance when computing the final representation.

**DLA's innovation:** Apply attention at *two* stages, not one:
1. **Before encoding (input attention):** Weight the raw input features before they enter the LSTM encoder
2. **After encoding (hidden state attention):** Weight the encoder's hidden states when constructing the final representation

This is inspired by DA-RNN (Dual-Stage Attention RNN) from the general time-series literature, adapted here for LOB data.

---

## Architecture

```
Input: 100 LOB snapshots × 40 features = X ∈ ℝ^(100×40)
           ↓
┌─────────────────────────────────────────────┐
│  Stage 1: Input Attention                    │
│  For each time step t:                       │
│    α_t = softmax(score(X_t, hidden state))   │
│    X̃_t = α_t ⊙ X_t  (weighted input)        │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│  LSTM Encoder                               │
│  Processes weighted inputs X̃_1...X̃_100      │
│  Outputs hidden states h_1...h_100          │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│  Stage 2: Hidden State Attention            │
│  β_t = softmax(score(h_t, context))         │
│  context = Σ_t β_t × h_t  (weighted sum)   │
└─────────────────────────────────────────────┘
           ↓
Dense(3) → Softmax → Up/Down/Stable
```

**Stage 1 (Input Attention):**
At each time step, an attention score is computed over the 40 input features based on the current LSTM hidden state. This produces a weighted input: features that are currently informative (given the evolving state) are amplified; noisy/irrelevant features are suppressed. The weights change dynamically over the 100-step sequence.

**Stage 2 (Hidden State Attention):**
After encoding all 100 steps, attention scores are computed over all 100 hidden states. This produces a context vector that is a weighted sum of all hidden states — emphasizing the time steps where the most relevant dynamics occurred. This is similar to the attention mechanism in sequence-to-sequence models (e.g., in DEEPLOBATT).

**Key difference from DEEPLOBATT (single-stage):** DEEPLOBATT only has Stage 2 (hidden state attention). DLA adds Stage 1 on top, which pre-selects which features to focus on before the LSTM even processes them. The two stages handle complementary information: Stage 1 focuses *what* to encode, Stage 2 focuses *when* the encoding was most relevant.

---

## Benchmark Results

From Prata 2023's LOBCAST benchmark (the most comprehensive evaluation):

**FI-2010 performance:**
- DLA F1: 73.4 ± 4.1 (ranks 2nd alongside DeepLOB in the benchmark)
- Robustness score: 93.2 (high — results are reproducible)

**Generalizability (LOB-2021/2022):**
- Generalizability score: **76.9%** — 2nd best overall (BINCTABL is 73.5%)
- DLA drops from FI-2010 to new data, but less severely than most models
- The dual attention helps: attention-based models consistently generalize better than pure CNN/LSTM

**Parameter count:** ~1.2×10^5 (comparable to DeepLOB). Inference time: 0.23ms.

---

## Relationship to Other Architectures

| Model | Attention stages | Time attention | Feature attention |
|---|---|---|---|
| DeepLOB | None | — | — |
| DEEPLOBATT (Zhang 2021) | 1 stage | Hidden states only | No |
| **DLA (Guo 2023)** | **2 stages** | **Input + hidden states** | **No** |
| TLOB (Berti 2025) | 2 separate SA blocks | Temporal SA | Spatial/feature SA |

DLA is the direct precursor to TLOB. The key evolution from DLA to TLOB:
- DLA: dual attention **over time** (two stages, both temporal)
- TLOB: dual attention **over both axes** (temporal self-attention + spatial/feature self-attention)

---

## Key Conclusions

1. **Two temporal attention stages beat one** — DLA outperforms DEEPLOBATT (single stage) consistently.

2. **High generalizability** — 2nd best in the Prata 2023 benchmark, meaning DLA's patterns learned on FI-2010 transfer to new NASDAQ data better than most models.

3. **Input attention (Stage 1) adds value** beyond just hidden state attention alone — empirically confirmed by the benchmark ranking.

4. **Limitation:** Both attention stages operate over the *time* dimension only. DLA doesn't explicitly model spatial dependencies between LOB features — this gap is what TLOB addresses.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| 2nd best generalizable model in benchmark | DLA is a strong baseline for NSE — likely transfers better than pure CNN models |
| Already in LOBCAST framework | No implementation needed — directly usable for NSE experiments |
| Dual temporal attention > single temporal attention | Confirms the attention architecture direction; sets expectation that TLOB (spatial + temporal) will do better |
| Input attention adapts feature weighting per time step | For NSE, different LOB features may be important at different times (e.g., during FII activity vs. retail hours) — DLA can capture this |
