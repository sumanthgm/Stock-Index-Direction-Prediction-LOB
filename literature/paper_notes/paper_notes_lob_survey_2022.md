# Paper Notes: Predicting Stock Price Changes Based on the Limit Order Book — A Survey

**Full title:** Predicting Stock Price Changes Based on the Limit Order Book: A Survey
**Authors:** Ilia Zaznov, Julian Kunkel, Alfonso Dufour, Atta Badii
**Affiliations:** University of Reading (CS + Finance); University of Göttingen (HPC/GWDG)
**Published:** Mathematics (MDPI), Vol. 10(8), Article 1234, April 2022. Open access.
**DOI:** 10.3390/math10081234
**URL:** https://www.mdpi.com/2227-7390/10/8/1234

---

## Why This Paper Matters

This is the first systematic survey of LOB-based mid-price prediction. It covers the full arc from classical statistical models through traditional ML to deep learning, and provides the most comprehensive critique of the field's experimental methodology available at the time.

For the capstone, this paper is valuable for two reasons:
1. **Literature framing:** It provides the taxonomy and comparative context needed to position any new study — showing what categories of models exist and how they've been evaluated
2. **Methodology critique:** Its diagnosis of the "simulation-to-reality gap" and FI-2010 over-reliance pre-dates and anticipates the empirical evidence later confirmed by Prata 2023 (LOBCAST) and Briola 2024 — providing the conceptual framework for why those findings matter

---

## Survey Scope

- Models and methods for stock price change prediction from LOB and order flow data
- Publications from approximately 2010–2021 (the full deep learning era for LOB)
- Mid-price direction classification as the canonical task
- Both LOB snapshot features and order flow representations
- Evaluation methodology critique — not just "what works" but "how we know if it works"

---

## Taxonomy of LOB Prediction Approaches

### By Model Family

**Classical Statistical Models**
- Autoregressive models (AR, ARIMA) on mid-price time series
- Point process / Hawkes process models for order arrival dynamics
- Queue-reactive models (LOB state transitions via stochastic processes)
- Econometric regression on order flow features

**Classical Machine Learning**
- SVM: Kercheval and Zhang (2015) — first major SVM application on LOB data; used engineered features from LOB levels
- Random Forest / XGBoost: ensemble methods on handcrafted LOB features
- Logistic Regression: standard baseline

**Deep Learning — Recurrent**
- LSTM / GRU: Sirignano (2019), Sirignano and Cont (2021) — deep LSTM on raw LOB snapshots; showed necessity of nonlinear models on large LOBSTER datasets
- Vanilla RNN: predecessor to LSTM baselines

**Deep Learning — Convolutional**
- CNN: Tsantekidis et al. (2017/2018) — early spatial CNN for LOB; treated LOB snapshot as 2D input
- Tensor/Image representation: Tran et al. (2017) — LOB as image

**Deep Learning — Hybrid CNN+LSTM**
- **DeepLOB (Zhang et al. 2019):** CNN spatial extraction + Inception modules + LSTM; trained on FI-2010; the dominant benchmark reference at time of writing
- Various CNN+RNN hybrids

**Deep Learning — Attention / Transformer**
- **TransLOB (Wallbridge 2020):** Transformer applied to LOB sequences; attention over LOB levels and time steps
- Dual-stage attention architectures (emerging at time of writing)

**Encoder-Decoder / Multi-Horizon**
- Seq2Seq architectures for simultaneous multi-horizon prediction

### By Input Representation

| Representation | Description | Typical Use |
|---|---|---|
| L1 (best bid/ask only) | Simplest — just the top-of-book | Basic models |
| L2 (full depth snapshot) | Prices + volumes at all levels | DeepLOB, most DL models |
| Order Flow Imbalance (OFI) | Derived from consecutive LOB changes | Kolm 2021; "information-rich" stocks |
| Engineered features | Spread, depth imbalance, midprice lags | Traditional ML |
| Image/tensor | LOB snapshot treated as 2D image | CNN-based models |

The survey concludes that **L2 + OFI** provides the strongest predictive signal, consistent with Kolm et al. (2021).

---

## Key Findings

### What Works (Positive Results)
1. **DL models consistently outperform classical ML** on FI-2010 — LSTM, CNN, and CNN+LSTM all dominate SVM/RF on the standard benchmark
2. **High-frequency stocks (liquid, active)** are more predictable — more order flow activity creates stronger signal
3. **L2 data beats L1** — deeper book information improves predictions
4. **Order flow representation** often outperforms raw LOB snapshots — OFI is a more stationary signal
5. **Adequate dataset size** is necessary — small datasets like FI-2010 risk overfitting

### The Simulation-to-Reality Gap (Critical Finding)

The survey's most important contribution is its methodological critique. Several layers of unrealism compound to make benchmark results misleading:

**Layer 1 — Mid-price execution assumption:**
All models are evaluated assuming trades execute at the mid-price. Real execution happens at the bid (to sell) or ask (to buy), imposing half-spread cost per trade. For any market with a spread > 0 (i.e., all markets), this systematically overstates returns.

**Layer 2 — Transaction costs ignored:**
Brokerage fees (~0.5% per round trip for retail; smaller but nonzero for institutional) are never deducted in benchmark evaluations. At high frequency, these erode claimed profits to near-zero or negative.

**Layer 3 — Market impact ignored:**
Large orders move the market. A model's signal may be correct, but executing at scale changes prices adversely. None of the surveyed models account for this.

**Layer 4 — FI-2010 overfitting:**
FI-2010 is 5 Finnish stocks over 10 days from 2010. Models achieving >80% accuracy on FI-2010 show performance degradation of up to ~19.6% F1-score on fresh LOBSTER data (confirmed later by Prata 2023). The survey identifies this as "overfit to benchmark" before empirical confirmation.

**Conclusion:** "Although considerable progress was achieved in this direction, even the state-of-the-art models cannot guarantee a consistent profit in active trading."

---

## Gaps Identified

### Input Data Gaps
- FI-2010 is outdated and too small; larger, newer, diverse datasets needed
- Normalization must prevent lookahead bias (leakage from test-period statistics)
- Level 3 / MBO (message-level) data is under-utilized
- Most work is NASDAQ-only; other exchanges, asset classes, and regions needed
- **Indian market data (NSE/BSE) is completely absent from the literature**

### Architecture Gaps
- Attention/Transformer architectures underexplored in 2022 (TLOB, Axial-LOB not yet published)
- Microstructure priors (tick size, stock type, spread regime) rarely built into architectures
- Interpretability methods (SHAP, attention visualization) absent from LOB DL work

### Experimental Design Gaps
- **Trading simulation as evaluation metric** — replace accuracy/F1 with simulated P&L and Sharpe ratio
- **Realistic execution modeling** — model bid-ask spread costs, partial fills, market impact
- **Mandatory out-of-sample generalization** testing across stocks and time periods
- **Statistical significance testing** of model comparisons
- **Reproducibility** — open-source code and data sharing largely absent

---

## Positioning vs. Other Papers in This Review

| Survey Aspect | Zaznov et al. 2022 | Prata et al. 2023 (LOBCAST) |
|---|---|---|
| Type | Literature survey | Empirical benchmark |
| Generalizability crisis | Predicted/argued | Empirically confirmed |
| FI-2010 critique | Conceptual | Measured (up to 19.6% F1 drop) |
| Coverage | 2010–2021 | 2023 focus |
| New architecture | None | BINCTABL |
| New dataset | None | LOB-2021/2022 (LOBSTER) |

The survey pre-dates and frames many issues that Prata 2023, Briola 2024, and the ICLR 2025 benchmark later confirm quantitatively.

---

## Key Conclusions

1. **DL models outperform classical ML on benchmarks, but benchmark results do not imply real-world profitability.** The gap between reported accuracy and actual trading utility is the field's central unresolved problem.

2. **FI-2010 is a flawed benchmark that the field has overfit.** More realistic evaluation requires larger, fresher datasets — this is now confirmed by LOBCAST (Prata 2023).

3. **Order flow + full L2 depth** provides the best predictive signal — consistent with Kolm et al. (2021).

4. **Evaluation methodology is broken.** Mid-price execution assumption, ignored transaction costs, and lack of trading simulation testing invalidate most published profitability claims.

5. **Indian and emerging market LOB data are completely absent** from the literature — this gap alone makes a study on NSE data a genuine novel contribution.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Mid-price execution assumption is unrealistic | For NSE, even more so — wider spreads, lower liquidity for mid-cap stocks amplify this gap |
| FI-2010 overfit — need fresh data | NSE/BSE data is a genuine new benchmark with no prior overfitting |
| Transaction costs erode claimed profits | NSE STT (Securities Transaction Tax) + brokerage makes this even more acute for Indian markets |
| L2 + OFI beats L1 | Confirms direction: use full LOB depth and order flow imbalance features for NSE experiments |
| Indian market gap identified | Your capstone fills an explicit gap this 2022 survey identified — first known LOB prediction study on NSE data |
