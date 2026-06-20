# Paper Notes: LOB Benchmark Study (ICLR 2025) — LOB Models vs. Time Series Models

**Full title:** A Benchmark Study For Limit Order Book (LOB) Models and Time Series Forecasting Models on LOB Data
**Authors:** Unknown (ICLR 2025 submission, anonymous peer review)
**Published:** ICLR 2025 (Learning on time series and dynamical systems track)
**URL:** https://openreview.net/forum?id=MhD9rLeU31

---

## Why This Paper Matters

This is a **second benchmark paper** (distinct from Prata 2023 / LOBCAST). Where Prata 2023 benchmarked 15 LOB-specific models against each other on FI-2010 + new LOBSTER data, this 2025 paper asks a different question: **how do LOB-specific models compare against general time-series forecasting models (PatchTST, iTransformer, TimesNet, etc.)?**

It also introduces two new dimensions not in Prata 2023:
1. **Asset transferability** — train on one asset, test on a completely different one
2. **Mid-price return forecasting (MPRF)** — predicting a continuous return value, not just Up/Down/Stable direction

This paper matters for your capstone because it provides context on whether general time-series models (which are state-of-the-art in other domains) are worth trying on LOB data — and the answer is: not without modification.

---

## Four Key Contributions

### 1. Asset Transferability Analysis

Tests whether LOB models can generalize **across assets** (not just across time periods). This is different from the Prata 2023 generalizability test (same model applied to different stocks in the same period). Here: train on stock A, test on stock B.

**Why this matters:** In real trading, you want a model that works on any instrument, not one retrained from scratch for each stock. Asset transferability is the harder, more practical test.

**Uses proprietary futures data** (not publicly available FI-2010 or LOBSTER) — this is a limitation for reproducibility but a strength for real-world relevance.

### 2. Mid-Price Return Forecasting (MPRF) — New Task

Prior LOB benchmarks evaluate **direction classification** (Up/Down/Stable). This paper introduces **mid-price return forecasting** — predicting the actual magnitude of the price change (a continuous regression target), not just the sign.

This matters because:
- Direction accuracy ≠ profit (you need to know *how much* the price moves, not just which way)
- Standard F1/MCC metrics don't capture the economic value of a prediction
- MPRF is a harder task — bridging closer to the actual trading utility question

### 3. Cross-Field Evaluation: General Time Series vs. LOB-Specific Models

State-of-the-art time-series models from the broader ML community (PatchTST, iTransformer, TimesNet, DLinear, etc.) are tested on LOB data.

**Key finding:** Generic time-series models **underperform LOB-specific models** without adaptation. The paper explicitly states: "LOB-aware model design is essential for achieving optimal prediction performance on LOB datasets."

This validates the entire body of LOB-specific architecture work (DeepLOB, BINCTABL, TLOB, etc.) — you cannot simply apply a state-of-the-art general sequence model to LOB data and expect SOTA results.

### 4. CVML Architecture — New Contribution

The paper introduces **CVML (Convolutional Cross-Variate Mixing Layers)** as an architectural enhancement:
- Applied to general time-series models to adapt them for LOB data
- Mixes information across the variate/feature dimension (similar in spirit to TLOB's spatial attention, but as a convolutional operation)
- Achieves **+244.9% average improvement** on MPRF over vanilla time-series models

This is a strong result: adding LOB-aware feature mixing to general models recovers much of the performance gap. The implication is that **cross-feature mixing is the critical missing component** in generic time-series models when applied to LOBs.

---

## Relationship to Prata 2023 (LOBCAST)

| Dimension | Prata 2023 (LOBCAST) | This paper (ICLR 2025) |
|---|---|---|
| Models benchmarked | 15 LOB-specific | LOB-specific + general TS models |
| Data | FI-2010 + LOBSTER (NASDAQ stocks) | Proprietary futures data |
| Task | Direction classification (3-class) | Direction + return regression (MPRF) |
| Generalization test | Same stocks, different time period | Different assets entirely |
| New architecture | None | CVML |
| Key finding | FI-2010 is overfit; BINCTABL generalizes best | LOB-aware design essential; CVML bridges the gap |

The two benchmark papers are **complementary**, not redundant. Prata 2023 is the go-to for comparing LOB-specific models. This paper is the go-to for understanding how LOB models relate to the broader time-series ML community.

---

## The CVML Architecture

CVML (Convolutional Cross-Variate Mixing Layers) adds feature-axis convolutions to any existing time-series model:

```
Standard time-series model:
  sequence → temporal modeling → prediction

With CVML enhancement:
  sequence → temporal modeling → CVML (cross-feature mixing) → prediction
```

The CVML layer applies convolutions across the variate/feature dimension to capture inter-feature dependencies. For LOB data, this means mixing information across the 40 features (10 ask prices, 10 ask volumes, 10 bid prices, 10 bid volumes) — the cross-feature pattern that standard 1D temporal models ignore.

**Connection to other papers:**
- Similar motivation to TLOB's spatial attention (attend over 40 features)
- Similar motivation to HLOB's HCNN (learn non-consecutive feature dependencies)
- But CVML is a **plug-in module** for existing models rather than a standalone architecture

**Why +244.9%?** Generic TS models have no feature-mixing component for LOB data. Adding CVML essentially "teaches" them what LOB-specific models already know structurally — dramatically improving their performance.

---

## Key Conclusions

1. **LOB-aware design is not optional.** General SOTA time-series models (PatchTST, iTransformer) significantly underperform LOB-specific models without adaptation.

2. **Cross-variate/feature mixing is the critical LOB inductive bias.** CVML +244.9% improvement shows that mixing across LOB features (the spatial dimension) is the key thing generic models are missing.

3. **Asset transferability is the hardest LOB challenge.** Models trained on one asset fail to transfer cleanly to another — even harder than the Prata 2023 cross-time-period generalizability test.

4. **MPRF (continuous return prediction) is a more practically relevant metric than direction classification.** F1 score on 3-class problems doesn't capture whether you're predicting large or small moves.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Generic TS models underperform without LOB-aware design | Don't be tempted to use general TS models (LSTMs from generic tutorials) — use LOB-specific architectures |
| CVML adds feature mixing to any model | If you use a simple LSTM baseline, add a feature-mixing layer — likely cheap improvement |
| Asset transferability is hard | Testing a model trained on HDFC on ICICI data is a genuine research contribution — test this |
| MPRF as additional metric | Consider reporting not just F1/MCC but a regression metric on mid-price returns — more practically meaningful for NSE |
| Proprietary futures data used | Gap: this benchmark doesn't use Indian data — your NSE study fills this gap |
