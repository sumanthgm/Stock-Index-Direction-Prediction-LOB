# Paper Notes: LOB-Based Deep Learning Models for Stock Price Trend Prediction — A Benchmark Study

**Full title:** LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study
**Authors:** Matteo Prata, Giuseppe Masi, Leonardo Berti, Viviana Arrigoni, Andrea Coletta, Irene Cannistraci, Svitlana Vyetrenko, Paola Velardi, Novella Bartolini (Sapienza + JPMorgan AI Research)
**Published:** arXiv 2308.01915, September 2023 (Artificial Intelligence Review 2024)
**File:** `[N4] LOB_Benchmark_Prata_2023.pdf`

---

## Why This Paper Matters

This paper is the **reality check** the LOB field needed. It takes 15 SOTA models that all claimed strong performance, replicates them on FI-2010, then tests them on completely new LOBSTER data. The finding is stark: **almost every model that looked good on FI-2010 fails badly on unseen data.**

This paper matters for your capstone because:
1. It tells you which models actually generalize — not just which ones fit FI-2010.
2. It introduces **LOBCAST**, an open-source framework you can use directly.
3. It defines the "generalizability problem" that is central to why testing on NSE data is a meaningful research contribution.

---

## The Core Finding in One Sentence

FI-2010 is overfit — a model achieving 80%+ F1 on FI-2010 can fail to 48-61% F1 on other NASDAQ stocks from different time periods.

---

## The Experimental Setup

### Two Questions Being Answered

**Robustness:** Can the model reproduce its *own* claimed performance when re-implemented and retested on FI-2010?

**Generalizability:** Does the model perform similarly on *new* unseen LOB data?

### Datasets

**FI-2010:** The standard benchmark (5 Finnish stocks, 10 days, June 2010). Already normalized and labeled. Used as the robustness test.

**LOB-2021:** Constructed from LOBSTER data.
- Period: July 1–15, 2021 (10 trading days)
- 6 NASDAQ stocks (selected via t-SNE clustering to be representative): SOFI, NFLX, CSCO, WING, SHLS, LSTR
- Market conditions: Relatively stable period
- Structure: Same as FI-2010 (event-based sampling every 10 events, 10 LOB levels, same split)

**LOB-2022:** Same 6 stocks, same structure.
- Period: February 1–15, 2022 (10 trading days)
- Higher volatility due to Ukraine war impact
- Class distributions are different from LOB-2021 (stocks like CSCO are more balanced; NFLX and LSTR are imbalanced)

### Performance Scores Defined

**Robustness Score** = 100 − (|Δ| + S)

Where Δ = difference between claimed F1 (from original paper) and measured F1 on FI-2010, S = standard deviation across 5 seeds × 5 horizons. A score of 100 = perfectly reproducible; low score = the model is sensitive to random initialization or hyperparameters.

**Generalizability Score** = 100 − (|Δ| + S)

Same formula but Δ = drop from FI-2010 to LOB-2021/2022.

---

## The 15 Models Benchmarked

| Model | Year | Architecture | Params | Inference (ms) |
|---|---|---|---|---|
| MLP | baseline | Multi-layer perceptron | 1.6×10⁴ | 0.08 |
| LSTM | baseline | LSTM | 1.6×10⁴ | 0.21 |
| CNN1 (Tsantekidis 2017) | 2017 | CNN | 3.5×10⁴ | 0.36 |
| CTABL (Tran 2018) | 2018 | Temporal-Attention + Bilinear | 1.1×10⁴ | 0.48 |
| DAIN (Passalis 2019) | 2019 | Deep Adaptive Input Norm + MLP | 5.3×10⁵ | 0.50 |
| CNN2 (Tsantekidis 2020) | 2020 | CNN with dilated conv | 2.8×10⁵ | 0.49 |
| CNNLSTM | 2020 | CNN + LSTM | (merged) | 0.49 |
| TransLOB (Wallbridge 2020) | 2020 | Dilated CNN + Transformer | 1.3×10⁵ | 3.90 |
| TLoNBoF (Passalis 2020) | 2020 | Temporal Logistic NBoF | - | 1.73 |
| DEEPLOBATT (Zhang 2021) | 2021 | DeepLOB + Seq2Seq + Attention | 1.8×10⁵ | 0.71 |
| DLA (Guo 2022) | 2022 | Temporal Attn + Stacked GRU | 1.2×10⁵ | 0.23 |
| DeepLOB (Zhang 2019) | 2019 | CNN + Inception + LSTM | 1.4×10⁵ | 1.31 |
| ATNBoF (Tran 2022) | 2022 | Attention + NBoF + 2D-Attention | 2.0×10⁵ | 1.91 |
| **BINCTABL (Tran 2021)** | 2021 | CTABL + Bilinear Norm | **1.1×10⁴** | **0.71** |
| AXIALLOB (Kisiel 2022) | 2022 | Gated Axial Attention | 2.0×10⁵ | 1.91 |
| METALOB (ensemble) | - | Meta-classifier (MLP on all models) | - | - |
| MAJORITY (ensemble) | - | F1-weighted voting | - | - |

---

## Key Results (Table 2)

### Robustness on FI-2010

| Model | F1 Claim | F1 LOBCAST | F1 Rank | Robustness Score |
|---|---|---|---|---|
| MLP | 51.8 | 48.0±2.6 | 14 | **91.8** |
| LSTM | 63.4 | 63.4±3.6 | 7 | **97.5** (best) |
| CTABL | 74.3 | 69.6±4.3 | 5 | 91.3 |
| DEEPLOBATT | 78.8 | 71.4±5.3 | 4 | 87.6 |
| DeepLOB | 78.8 | 73.4±4.1 | - | 93.2 |
| **BINCTABL** | **80.1** | **82.6±7.0** | **1** | **99.7** |
| DLA | 78.7 | 73.4±4.1 | 2 | 93.2 |
| AXIALLOB | 82.0±3.7 | 73.4±5.7 | 3 | 88.2 |
| TransLOB | 87.3±4.0 | 59.4±2.6 | 9 | **69.9** (worst) |
| ATNBoF | 67.1±5.5 | 40.9±7.7 | 15 | **66.1** (worst) |

Key observations:
- **BINCTABL** actually *exceeds* its claimed performance (82.6 vs. 80.1) — genuine robustness.
- **TransLOB** collapses from 87.3 to 59.4 — a 28 point drop, worst robustness.
- **ATNBoF** is worst overall, failing both robustness and generalizability.
- **CNNLSTM** achieves the greatest improvement (20.9%) when switching to raw LOB features — the original model used hand-engineered stationary features, but raw LOB works better here.
- Half of all models had standard deviation > 5 F1 points across seeds — extreme sensitivity to initialization.

### Generalizability on LOB-2021/2022

All models drop significantly from FI-2010 performance:
```
FI-2010 best (BINCTABL): 82.6%
LOB-2021 best (BINCTABL): 61.2%   → −21.4 points
LOB-2022 best (BINCTABL): 59.5%   → −23.1 points
```

Overall LOB-2021/2022 performance range: **48–61% F1** (vs. 48–82% on FI-2010).

**Generalizability rankings (LOB-2021/2022):**
1. BINCTABL: 73.5% generalizability
2. DLA: 76.9%
3. DEEPLOBATT: 74.5%
4. AXIALLOB: 70.7%

Five of the best six models on LOB-2021/2022 incorporate **attention mechanisms**. The attention-based models consistently generalize better than pure CNN or LSTM models.

---

## Critical Observations

### 1. None of the top 3 models use 100-step input windows

The top performing models on LOB-2021/2022 (BINCTABL, DLA, DEEPLOBATT) use:
- BINCTABL: 10 time steps
- DLA: 15 time steps
- DEEPLOBATT: 300 time steps

Models using 100 steps (the dominant choice in the literature) are *not* among the top performers when generalization is tested. This suggests the standard 100-step window may be overfitted to FI-2010.

### 2. Ensemble methods don't help

Both METALOB and MAJORITY fail to exceed the performance of the best individual models. High agreement rate among base models means the ensemble just repeats the same errors.

### 3. ATNBoF worst overall

Despite being from 2022, ATNBoF performs worst on both robustness and generalizability. Its complex 2D attention mechanism apparently overfits severely.

### 4. TransLOB's architecture doesn't generalize

TransLOB combines dilated convolutions + Transformer but drops from 87.3 to 59.4 on FI-2010 alone (before even testing generalizability). The dilated convolutions may be tuned too specifically to FI-2010's event structure.

### 5. CSCO generalizes best among LOB-2021/2022 stocks

CSCO (Cisco Systems) has a class distribution of 18-65-17% in the training set — very stable (mostly "stationary"). Models consistently perform best on CSCO. This aligns with the Kolm 2021 finding: predictability is highest for large-tick stocks with stable, orderly books.

---

## The LOBCAST Framework

LOBCAST (github.com/matteoprata/LOBCAST) is a Python framework for LOB stock market trend prediction, built on PyTorch Lightning.

**What it provides:**
- Data preprocessing (normalization, splitting, labeling)
- All 15 benchmarked models implemented in PyTorch (not TensorFlow)
- Training, validation, and test infrastructure
- Integration with WANDB for hyperparameter tuning
- Backtesting for profit analysis via Backtesting.py
- F1, Accuracy, Recall metric reports

**Why this matters for your capstone:** You can fork LOBCAST and add your NSE data pipeline + TLOB/MLPLOB models. This saves months of implementation work.

---

## Why Models Generalize Poorly

Three root causes identified:

**1. FI-2010 is too simple.** Finnish NASDAQ Nordic stocks are less liquid and efficient than US NASDAQ stocks. LOB-2021/2022 have approximately 3× more events in the same period. Models trained on the simpler Finnish data fail on more complex US stocks.

**2. Overfitting to time period.** LOB-2022 (war-period volatility) is harder than LOB-2021. Models overfit to the specific regime of FI-2010 (June 2010).

**3. Labeling threshold sensitivity.** FI-2010's θ was chosen to balance classes at k=5. For other stocks with different return distributions, this θ creates imbalanced classes. Models biased toward predicting "stable" on FI-2010 fail on other distributions.

---

## Key Conclusions

1. **FI-2010 is a poor benchmark for real-world applicability.** High F1 on FI-2010 does not imply generalizability.

2. **BINCTABL is the most reliable model overall.** Best on FI-2010, best generalizability score. Key ingredient: Bilinear Normalization that adapts to batch-level statistics.

3. **Attention mechanisms generalize better than CNN/LSTM alone.** 5 of 6 best generalizability models use attention.

4. **All models show large performance drops on new data.** The field has a systematic overfitting problem.

5. **LOBCAST is the standard open-source framework for this research.**

6. **Profits from trading are not guaranteed.** Trading simulation on LOB-2021 confirms models are not practically profitable.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| All models degrade on new data | NSE data is inherently "new" — expect lower performance than FI-2010 numbers suggest |
| BINCTABL most generalizable | Include BINCTABL as a strong baseline alongside DeepLOB |
| Attention models generalize better | TLOB (dual attention) should generalize to NSE better than DeepLOB |
| LOBCAST framework available | Fork LOBCAST; add your NSE data pipeline and TLOB implementation |
| FI-2010 ≠ real market | This strengthens the motivation for your NSE study — proper out-of-sample testing |
| Stable stocks (CSCO-like) are more predictable | NSE banking sector (HDFC, ICICI) = "CSCO equivalent" — test these first |
| Threshold θ selection matters | For NSE, compute stock-specific θ or use TLOB's average-spread approach |
| Ensemble methods don't improve | Don't waste time on ensemble methods for this project |
| Need long training data (months, not 10 days) | LOB-2021/2022 uses 10 days; still showed generalizability issues — NSE needs more |
