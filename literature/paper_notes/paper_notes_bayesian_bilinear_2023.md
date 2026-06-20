# Paper Notes: Bayesian Bilinear Neural Network for Predicting Mid-Price Dynamics in LOB Markets

**Full title:** Bayesian Bilinear Neural Network for Predicting the Mid-Price Dynamics in Limit-Order Book Markets
**Authors:** Martin Magris, Mostafa Shabani, Alexandros Iosifidis
**Affiliation:** DIGIT Group, Department of Electrical and Computer Engineering, Aarhus University, Denmark
**arXiv:** 2203.03613 (March 2022)
**Published:** Journal of Forecasting, Vol. 42(6), September 2023
**DOI:** 10.1002/for.2955
**URL:** https://onlinelibrary.wiley.com/doi/full/10.1002/for.2955
**Note:** Magris is also a co-author of the original FI-2010 benchmark dataset paper (Ntakaris et al. 2018).

---

## Why This Paper Matters

Most LOB prediction papers output a single point prediction (argmax of softmax). This paper asks: *what is the full probability distribution over {Up, Down, Stationary} given the current LOB state, and how confident should we be?* It answers by applying **Bayesian Neural Network** training to the TABL architecture.

This matters for the capstone for two reasons:
1. **Uncertainty quantification is practically important.** A model that says "70% Up, 20% Stationary, 10% Down" is more actionable than one that says "Up" — especially in Indian markets with wider spreads and higher noise
2. **This paper establishes that TABL-class architectures can be Bayesianized.** The TABL (Temporal Attention-Augmented Bilinear Layer) family is the same lineage as BINCTABL — the best-generalizing model in the Prata 2023 benchmark. Understanding the probabilistic variant informs the architecture choice

---

## Background: What is a Bilinear Layer?

Standard fully-connected (FC) layers treat the flattened LOB input as a 1D vector, losing the 2D structure (features × time). The bilinear layer instead applies **separate linear projections along each axis**:

For input X ∈ ℝ^(D×T) (D=40 features, T=10 time steps):
```
Bilinear projection:
  Step 1 (feature axis): X̄ = W₁X    where W₁ ∈ ℝ^(D'×D)
  Step 2 (time axis):    Y = φ(X̃W₂ᵀ + B)    where W₂ ∈ ℝ^(T'×T)
```

The temporal attention vector a ∈ ℝ^T is learned and weights the time steps:
```
X̃ = λ(X̄ ⊙ A) + (1−λ)X̄
```
where λ is a learned blending scalar and A broadcasts attention weights across features.

**Why bilinear over standard FC?** Parameter count scales linearly with D and T separately (O(D'D + T'T)), not quadratically (O(D·T·hidden)). A 2-layer TABL network with ~5,000–10,000 parameters matches deep LSTM/CNN models on FI-2010. The bilinear structure also encodes the right inductive bias: LOB features and time steps should be treated as separate axes with different transformation semantics.

---

## The Bayesian Treatment: VOGN

**VOGN = Variational Online Gauss-Newton** — a natural-gradient variational inference algorithm that places a Gaussian posterior over all network weights.

### How it works

1. **Prior:** Gaussian prior over all weights p(w)
2. **Posterior approximation:** Variational family q(w) = N(μ, Σ), optimized to minimize KL(q‖p(w|data))
3. **Gradient update:** Uses natural gradient (Fisher information matrix approximation via Gauss-Newton) rather than Euclidean gradient — accounts for the geometry of the probability manifold
4. **Cost:** Comparable per-epoch cost to Adam — practically viable for the compact TABL architecture

### At inference time

Multiple weight samples w₁,...,wₙ ~ q(w) are drawn and the predictive distribution is:
```
P(y|x) = (1/N) Σᵢ Softmax(f(x; wᵢ))
```
This gives a full distribution over {Up, Stationary, Down}, not just the argmax.

### Why VOGN over MC Dropout?

MC Dropout (Gal & Ghahramani, 2016) is a common cheap approximation to Bayesian inference — retain dropout at test time and average multiple forward passes. The paper shows VOGN produces **better-calibrated uncertainty** than MC Dropout applied to Adam-trained weights, with a more principled posterior. MC Dropout posteriors tend to be overconfident.

---

## Architecture: 2-Layer Bayesian TABL

```
Input: X ∈ ℝ^(40×10)  (40 LOB features × 10 most recent events)
           ↓
   Bayesian TABL Layer 1 (VOGN)
   - Temporal attention over T=10 timesteps
   - Bilinear projection: feature D=40→40, time T=10→10
   - Dropout
           ↓
   Bayesian TABL Layer 2 (VOGN)
   - Further temporal + feature projection
           ↓
   Bayesian Dense(3) (VOGN)
   - W₃ sampled from posterior at test time
           ↓
   Monte Carlo averaging over N weight samples
   → P(Up), P(Stationary), P(Down)
```

All layers have variational Gaussian posteriors learned via VOGN. Very compact — the deterministic TABL architecture fits in ~5,000–10,000 parameters.

---

## Dataset and Task

**Dataset:** FI-2010 — standard benchmark  
- 5 Finnish stocks (NASDAQ Nordic Helsinki), 10 trading days, June 2010  
- Setup 2 convention: 7 days train, 3 days test

**Input:** 40 LOB features × 10 timesteps (standard setup)

**Normalization:** Z-score over non-overlapping 10-event blocks

**Prediction task:** 3-class mid-price direction  
**Horizons:** k = 10, 20, 30, 50, 100 LOB update events

---

## Results

### Predictive Accuracy

The VOGN-trained Bayesian TABL achieves **competitive or superior F1 and accuracy** compared to deterministic variants:

| Training Method | Accuracy | F1 | Uncertainty |
|---|---|---|---|
| SGD (deterministic) | Baseline | Baseline | None |
| Adam (deterministic) | ≥SGD | ≥SGD | None |
| Adam + MC Dropout | ≈Adam | ≈Adam | Approximate |
| **VOGN (Bayesian)** | **≥ all** | **≥ all** | **Calibrated** |

Results reported per stock and aggregated across 5 FI-2010 stocks, all horizons k=10–100.

### Uncertainty Calibration

The paper's key contribution is calibration quality:
- **Entropy analysis:** VOGN's predictive distribution has lower entropy on correct predictions (model is more confident when right) and higher entropy on wrong predictions (model knows when it doesn't know)
- **MC Dropout comparison:** VOGN posteriors are better calibrated — MC Dropout tends to be overconfident, assigning high softmax probability to incorrect predictions
- **Decision threshold application:** A trader using VOGN can set a confidence threshold (e.g., only trade when P(Up) > 0.70) and filter out low-confidence signals — this is impossible with deterministic models

---

## Relationship to Other Papers in This Review

| Paper | Architecture | Uncertainty | Notes |
|---|---|---|---|
| DeepLOB (Zhang 2019) | CNN + Inception + LSTM | None | Point prediction |
| TABL (Tran, Iosifidis 2019) | Bilinear + temporal attention | None | Foundation for this paper |
| BINCTABL (Prata 2023 benchmark best) | Bilinear + channel attention | None | Best generalizable model |
| **This paper (Magris 2023)** | **Bayesian TABL (VOGN)** | **Calibrated posterior** | **Probabilistic TABL** |
| GP models (Liu 2024) | Gaussian Process | Built-in | Non-parametric alternative |

The Bayesian TABL and the GP models (Liu 2024) represent two complementary approaches to uncertainty quantification in LOB prediction — deep learning with Bayesian inference vs. non-parametric Bayesian methods.

---

## Key Conclusions

1. **Bayesian training (VOGN) improves both accuracy and calibration over deterministic TABL.** The improvement in accuracy is modest; the improvement in uncertainty quality is substantial.

2. **Calibrated confidence enables better trading decisions.** A model that outputs "55% Up" vs "92% Up" provides actionable information that deterministic models discard.

3. **VOGN outperforms MC Dropout for uncertainty quantification.** MC Dropout is computationally cheap but produces overconfident posteriors; VOGN is slightly more expensive but principled.

4. **Bilinear structure is parameter-efficient.** The compact TABL architecture is well-suited to Bayesian treatment — fewer parameters = more tractable posterior inference.

5. **Connects econometrics and DL.** The paper explicitly bridges the probabilistic tradition in financial econometrics (where prediction uncertainty is standard) with the accuracy-focused DL tradition in LOB prediction.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Calibrated uncertainty via VOGN | For NSE, where circuit breakers and price bands create discontinuous dynamics, knowing when the model is uncertain is especially valuable |
| TABL lineage = best generalizable model family | BINCTABL is best in Prata 2023 benchmark — the Bayesian version of the same family adds uncertainty |
| Threshold-based trading | NSE STT (Securities Transaction Tax) makes low-confidence trades especially costly — confidence thresholds directly reduce transaction drag |
| Compact parameter count | Low parameter count is beneficial on NSE where training data will be more limited than US markets |
| FI-2010 basis | Same benchmark as DeepLOB, HLOB, BINCTABL — results directly comparable in literature context, but NSE validation is the capstone's novel contribution |
