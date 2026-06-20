# Paper Notes: The Intraday Dynamics Predictor — TrioFlow CNN+GRU for HF Price Forecasting

**Full title:** The Intraday Dynamics Predictor: A TrioFlow Fusion of Convolutional Layers and Gated Recurrent Units for High-Frequency Price Movement Forecasting
**Authors:** Ilia Zaznov, Julian Martin Kunkel, Atta Badii, Alfonso Dufour
**Affiliation:** University of Reading, UK
**Published:** Applied Sciences (MDPI), Vol. 14(7), Article 2984, April 2024
**DOI:** 10.3390/app14072984
**URL:** https://www.mdpi.com/2076-3417/14/7/2984
**Note:** Same team as the 2022 LOB Survey (Zaznov et al.) — this is their follow-up implementation paper.

---

## Why This Paper Matters

This paper does two important things for the LOB prediction field:
1. **Introduces a joint LOB + Order Flow (OF) input** — most prior models (DeepLOB, FI-2010 baselines) use only LOB snapshots. This paper is among the first to systematically incorporate executed trade data (order flow) as a co-equal input, motivated by Kolm et al. (2021)'s finding that OF is informationally richer than the static LOB
2. **Introduces the first non-NASDAQ dataset with both LOB and OF data** — the MICEX (Moscow Exchange) dataset, making Russian equities the closest available analog to an Indian market study

For the capstone: the LOB + OF joint input design is directly applicable to NSE data (which includes both order book and trade tick feeds), and the MICEX results provide a precedent for non-US, non-NASDAQ LOB prediction.

---

## The TrioFlow Architecture

"TrioFlow" = **three parallel convolutional branches** (inception-style multi-scale) fused before a GRU layer. The full model is formally named **TFF-CL-GRU** (TrioFlow Fusion of Convolutional Layers and GRU).

```
Input: [T timesteps × 70 features]
           ↓
   ┌───────────┬───────────┬───────────┐
   │ Branch 1  │ Branch 2  │ Branch 3  │
   │ Conv2D    │ Conv2D    │ Conv2D    │
   │ (kernel A)│ (kernel B)│ (kernel C)│
   └───────────┴───────────┴───────────┘
           ↓ concatenate + reshape
   GRU (64 units)
           ↓
   Dense(3) → Softmax
   → Up / Flat / Down
```

**Three branches:** Each uses different Conv2D kernel sizes to capture different spatial scales across the 70-feature × T-timestep input. This is analogous to the Inception module in DeepLOB, but with three parallel full convolutional streams rather than a single inception block.

**Key design choices:**
- **PReLU** (Parametric ReLU) activation throughout — slope for negative values is learned; avoids "dying ReLU" and outperforms standard ReLU empirically on this task
- **GRU with 64 units** over LSTM — empirically chosen; GRU matched or outperformed LSTM for this task at lower computational cost
- **Class imbalance handling** — dataset is not class-balanced; weighted F1 is the primary metric

---

## Input Representation: LOB + Order Flow (70 features)

The key innovation over DeepLOB is combining two data streams:

| Feature Group | Count | Description |
|---|---|---|
| LOB features (R1–R40) | 40 | Prices and volumes at top 10 bid and ask levels |
| Order Flow features (R41–R70) | 30 | Last 10 actual transactions: price, volume, direction per trade |
| **Total** | **70** | Per LOB snapshot timestamp |

**Why add order flow?** Following Kolm et al. (2021) — executed trades reveal information that resting limit orders don't. A large market sell order (OF feature) signals aggression that the static bid queue (LOB feature) doesn't capture until prices move. Combining both gives the model information about both the current state of supply/demand and the active direction of liquidity consumption.

**Note:** FI-2010 contains only LOB data, making it incompatible with this feature set. This is why the authors created their own dataset.

---

## Datasets

### Dataset 1: MICEX LOB+OF (new, introduced by this paper)

| Property | Value |
|---|---|
| Exchange | Moscow Exchange (MICEX/MOEX) |
| Stocks | Sberbank (SBER), VTB Bank (VTBR), Gazprom (GAZP) |
| Total records | 1.5M+ LOB+OF records |
| Availability | Public — Kaggle: https://www.kaggle.com/datasets/izaznov/micex-lob-of |

This is the first publicly available benchmark dataset combining LOB and OF data. It also covers a non-US, non-NASDAQ market — directly relevant for papers working on non-NASDAQ exchanges (like NSE).

### Dataset 2: LOBSTER (existing)

| Property | Value |
|---|---|
| Exchange | NASDAQ |
| Stocks | AAPL, AMZN, GOOGL |
| Source | LOBSTER system (standard) |

---

## Results vs. DeepLOB

TFF-CL-GRU outperforms DeepLOB on all 6 stocks:

| Stock | Exchange | TFF-CL-GRU F1 | DeepLOB F1 (approx.) |
|---|---|---|---|
| VTB | MICEX | **~65%** | ~60% |
| Gazprom | MICEX | **~51%** | lower |
| Sberbank | MICEX | **~45%** | lower |
| Amazon | LOBSTER | **~60%** | lower |
| Google | LOBSTER | **~55%** | lower |
| Apple | LOBSTER | **~49%** | lower |

Improvement over DeepLOB: **4–5+ F1 percentage points** on all stocks.

Additionally, **simulated trading experiments** show positive returns when using TFF-CL-GRU signals for trade execution — bridging statistical prediction to practical trading utility. This addresses the simulation-to-reality gap critique from the authors' own 2022 survey.

---

## Relationship to Other Papers

| Paper | Input | Architecture | Data |
|---|---|---|---|
| DeepLOB (Zhang 2019) | LOB only (40 feat.) | 1D CNN + Inception + LSTM | FI-2010 |
| Axial-LOB (Kisiel 2022) | LOB only (40 feat.) | Factored 2D attention | FI-2010 |
| TLOB (Berti 2025) | LOB only (40 feat.) | Temporal + Spatial SA | FI-2010 + LOBSTER |
| **TFF-CL-GRU (this paper)** | **LOB + OF (70 feat.)** | **3-branch CNN + GRU** | **MICEX + LOBSTER** |

The unique contribution is the input feature set (LOB + OF) and the non-NASDAQ benchmark dataset. The architecture itself (multi-scale CNN + GRU) is an incremental evolution of DeepLOB.

---

## Key Conclusions

1. **LOB + Order Flow jointly outperforms LOB alone.** The 30 order flow features (last 10 trades) add predictive signal beyond what the static LOB snapshot captures — consistent with Kolm et al. (2021).

2. **TrioFlow (multi-scale CNN) outperforms single-scale CNN + LSTM (DeepLOB).** Multi-branch feature extraction at different spatial scales captures more of the LOB+OF structure.

3. **GRU ≈ LSTM for this task** but at lower cost — a practical finding for implementation choices.

4. **Non-NASDAQ markets (MICEX) show similar prediction structure.** The model generalizes from NASDAQ patterns to Russian equities — the first evidence that LOB prediction approaches transfer beyond NASDAQ Nordic and US markets.

5. **Simulated trading validation is achievable.** The paper demonstrates positive P&L from model signals — rare in the LOB literature; most papers stop at statistical metrics.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| LOB + OF input outperforms LOB alone | NSE provides both order book and trade tick data — using both jointly is feasible and likely beneficial |
| MICEX results show non-NASDAQ generalization | MICEX (Russian market, emerging market dynamics) is the closest analog to NSE in the existing literature — strong citation for Indian market work |
| First publicly available LOB+OF dataset | Gap: no Indian market equivalent exists — NSE LOB+OF dataset would be another contribution |
| Trading simulation shows positive P&L | Applicable methodology for NSE evaluation — report both F1 and simulated returns |
| GRU performs as well as LSTM | Consider GRU as an alternative to LSTM in NSE experiments — fewer parameters, faster training |
