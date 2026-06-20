# Paper Notes: Mid-Price Forecasting in Limit Order Books with Gaussian Process Models

**Full title:** Mid-Price Forecasting in Limit Order Books with Gaussian Process Models
**Author:** Rui Liu
**Supervisors:** Juho Kanniainen, Puneet Pasricha (Tampere University)
**Published:** MSc Thesis, Tampere University (Faculty of IT and Communication Sciences), November 2024
**URL:** https://trepo.tuni.fi/bitstream/handle/10024/162261/LiuRui.pdf?sequence=2

---

## Why This Paper Matters

This is the first work to apply Gaussian Process (GP) models to the full 40-dimensional LOB feature space for mid-price prediction. Prior GP applications in finance used low-dimensional OHLCV data; this paper is the first to test GPs at the scale of HFT LOB data.

It matters for the capstone because:
1. It provides a **probabilistic baseline** distinct from all DL baselines — GPs give calibrated uncertainty intervals, not just point predictions
2. GPs are **interpretable**: the kernel design explicitly encodes assumptions about price vs. volume relationships
3. The **regression task** (continuous mid-price forecasting) complements the classification task — aligns with the MPRF direction in the ICLR 2025 benchmark paper
4. The paper demonstrates that **for long-horizon regression, GPs beat TABL** (prior SOTA) by a large margin — GPs aren't just a weak baseline

---

## Problem Setup

**Two tasks on the same dataset:**

1. **GPC (GP Classification):** Predict mid-price movement direction — Up/Down/Stationary  
   - Horizons: H = 1, 5, 10 (number of future LOB events)  
   - Label formula: `l_i = (avg_future_mid - avg_past_mid) / avg_past_mid`  
   - Threshold α separates Up/Down/Stationary classes

2. **GPR (GP Regression):** Predict the actual future mid-price  
   - Horizons: H = 10,000; 13,000; 15,000 (average mid-price over next N events)  
   - Long-horizon regression, not short-term direction

**Why long horizons for regression?** At very short horizons, the mid-price is dominated by the stationary class (barely changes), making regression metrics (MSE, R²) poorly informative. Long horizons give meaningful variation.

---

## Dataset

**Source:** NASDAQ TotalView-ITCH feed → H5 files (same approach as FI-2010 builder but different stocks)  
**Stocks:** AAPL, FB, GOOG, INTC, MSFT (5 US large-caps)  
**Period:** Sept 14–25, 2015 (10 trading days)  
**Events extracted:** ~780,000 (filtered to 9:40–15:50 to exclude opening/closing noise)  
**Input features:** 40 features = top 10 bid/ask price and volume levels (standard LOB representation)  
**Train/test split:** 7 days training, 3 days test  
**Normalization:** Z-score per feature (column-wise), statistics computed on training set only  
**Input window:** Rolling window of 10 consecutive events → each sample is a 10×40 matrix

**Note:** This is NOT FI-2010 — different stocks, different exchange period, different year. Results are not directly comparable to FI-2010 benchmarks (BINCTABL, DeepLOB, etc.).

---

## Methodology

### Gaussian Process Regression (GPR)

A GP places a distribution over functions: f(x) ~ GP(m(x), k(x, x')). Posterior after conditioning on training data gives both a predicted mean and a variance (uncertainty interval).

**Prediction:**
```
μ* = m(x*) + k(x*, x)(K + σ²I)⁻¹(y - m(x))
Σ* = k(x*, x*) - k(x*, x)(K + σ²I)⁻¹k(x, x*)
```

The predicted output y* ~ N(μ*, Σ* + σ²I) — a full Gaussian, not just a point.

Implementation: scikit-learn GaussianProcessRegressor.

### Gaussian Process Classification (GPC)

For 3-class prediction, GPC places a GP prior on latent functions f_c(x) per class, then converts to probabilities via softmax. Posterior inference uses Laplace approximation (exact posterior is intractable for classification).

Implementation: scikit-learn GaussianProcessClassifier.

### Kernel Design (Key Insight)

The thesis uses a **combined kernel** that encodes domain knowledge about LOB structure:

```
K(x₁, x₂) = k_linear(x₁_price, x₁_price) + k_RBF(x₂_vol, x₂_vol)
```

- **Price features (x₁):** Linear kernel — captures the known linear relationship between LOB price levels and future mid-price
- **Volume features (x₂):** RBF kernel — captures nonlinear relationships between volume distributions and price changes

This split is principled: there is a near-deterministic linear relationship between current best bid/ask prices and the near-future mid-price (regression in the figures shows tight linear scatter for all 5 stocks). Volume-to-price relationships are nonlinear and stock-specific.

### Ensemble Method (Addressing O(N³) Cost)

GPs have O(N³) training complexity and O(N²) storage — prohibitive for ~780,000 samples.

**Solution:** Split training data into chunks of N=8,000 samples, train a separate GP per chunk in parallel, aggregate predictions by averaging.

```
Training: data → chunks (8,000 each) → parallel GPR/GPC per chunk → M models
Inference: test point → prediction from each of M models → average
```

Computation on Google TPU v2-8 (8 cores, 64GB):

| Chunk Size | Total Time | Per-Sample |
|---|---|---|
| 8,000 | 1,259s | 11.6ms |
| 16,000 | 2,947s | 27.0ms |
| 32,000 | 9,003s | 82.6ms |
| 40,000 | 13,724s | 125.9ms |

Cubic scaling is clearly visible — 5× the chunk size → ~10× the time. Ensemble with 8,000-sample chunks was chosen as the practical compromise.

---

## Results

### Classification (GPC) — F1 Score

| Model | H=1 F1 | H=5 F1 | H=10 F1 |
|---|---|---|---|
| SVM | 30.79 | 37.13 | 30.44 |
| SLFN | 43.01 | 44.36 | 43.49 |
| BL (3-layer) | **56.00** | 63.65 | 60.73 |
| TABL (3-layer) | 55.37 | 62.52 | **61.54** |
| **GPC** | 30.52 | **59.70** | 51.29 |

**Interpretation:**
- H=1: GPC performs poorly (30.52%) — at very short horizons, the LOB is too noisy for GP's smoothness assumptions; 74% of samples are stationary class, biasing all models
- H=5: GPC jumps to 59.70% — 3rd best overall, ~25% above SLFN, competitive with 3-layer BL
- H=10: GPC = 51.29% — middle of the pack

**Win-rate at H=1 (excluding stationary class):** GPC = **78.97%** vs SLFN = 62.66%. When forced to predict directional moves only, GPC is far more accurate — the issue is GPC predicts "stationary" too rarely.

**Uncertainty (entropy):** GPC entropy is concentrated between 0.8–1.0 (higher confidence, narrower distribution) vs TABL which ranges broadly from 0.0–1.05. GPC has more calibrated uncertainty despite lower F1.

### Regression (GPR) — R² Score

| Model | H=10,000 R² | H=13,000 R² | H=15,000 R² |
|---|---|---|---|
| LR | 87.11 | 82.06 | 66.51 |
| RR | 93.56 | 92.90 | 86.00 |
| SLFN | 88.14 | 82.32 | 74.70 |
| TABL | 90.44 | 88.54 | 76.58 |
| **GPR** | **94.47** | **93.87** | **90.47** |

**GPR improvement over TABL in MAE:**
- H=10,000: +25.17% better
- H=13,000: +23.86% better
- H=15,000: +32.41% better

All differences are statistically significant (p<0.001 for H=10,000 and 13,000; p=0.04 for H=15,000 vs RR).

**Why GPR > TABL for regression?** The combined linear+RBF kernel is well-matched to the LOB regression structure. TABL overfits the linear price signal while adding complexity from temporal attention. RR (ridge regression) actually performs close to GPR — the dominant signal in long-horizon regression is linear, and GPR captures this through the linear kernel while adding nonlinear volume effects via RBF.

**Confidence intervals:** GPR naturally produces 95% confidence intervals for mid-price predictions. The actual mid-prices consistently fall within the predicted intervals — well-calibrated, not just wide bands.

---

## GP vs. DL: Key Tradeoffs

| Dimension | GP | DL (TABL/BL) |
|---|---|---|
| Short-horizon classification (H=1) | Poor (30%) | Strong (56%) |
| Medium-horizon classification (H=5,10) | Competitive (51–60%) | Best (61–64%) |
| Long-horizon regression | **Best (90–94% R²)** | Weaker (76–90%) |
| Uncertainty quantification | Built-in, calibrated | No / requires MC dropout |
| Training time | O(N³), very slow | Fast (GPU-parallelizable) |
| Interpretability | High (kernel encodes priors) | Low (black box) |
| Scalability | Poor without approximation | Good |

---

## Key Conclusions

1. **GPs are competitive for medium-horizon classification and best for long-horizon regression.** They are not a strong baseline for short-horizon (H=1) classification tasks due to class imbalance and noise sensitivity.

2. **Kernel design matters enormously.** Separating price features (linear kernel) from volume features (RBF kernel) leverages known LOB structure and outperforms using a single kernel type.

3. **Uncertainty quantification is GP's unique contribution.** Well-calibrated confidence intervals are unavailable in standard DL models — this is a practical advantage for risk management, even if F1 score is lower.

4. **Computational cost is the critical limitation.** O(N³) scaling requires ensemble approximation; even then, training takes 1,000+ seconds on TPU hardware for LOB-scale data.

5. **First application of GPs to full 40-dimensional LOB features.** Prior GP finance work used OHLCV (4–5 features); scaling to 40 required the ensemble approach and the principled kernel split.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| GPR beats TABL for long-horizon regression | If adding a regression evaluation metric (like MPRF from ICLR 2025 paper), GPR is a stronger regression baseline than TABL |
| Calibrated uncertainty intervals | NSE data is noisier (circuit breakers, price bands, lower liquidity) — uncertainty estimates have higher practical value |
| O(N³) cost limits scalability | For NSE with potentially fewer data points than NASDAQ, GP might be more feasible than for large US datasets |
| H=1 classification is weak | Confirms that short-horizon LOB prediction is noisy universally — NSE short-horizon likely equally hard |
| Linear kernel for prices | The price-level linearity finding likely holds for NSE too — useful prior for any kernel or normalization design |
