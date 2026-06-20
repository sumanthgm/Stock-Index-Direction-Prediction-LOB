# Paper Notes: Deep Learning Modeling of Limit Order Book — A Comparative Perspective

**Full title:** Deep Learning Modeling of Limit Order Book: A Comparative Perspective
**Authors:** Antonio Briola, Jeremy Turiel, Tomaso Aste (UCL)
**Published:** arXiv 2007.07319, 2020
**File:** `[14] DeepLOB_Comparative_Briola_2020.pdf`

*Note: This is the **earlier** Briola paper (2020). The Briola 2024 paper you started with is a follow-up from the same group. Read this one to understand what motivated the 2024 work.*

---

## Why This Paper Matters

This paper asks a provocative question: **Is the sophisticated CNN-LSTM architecture of DeepLOB actually necessary?** The answer: **no — a simple MLP performs statistically equivalently.**

This challenges the dominant assumption in the field that the spatial-temporal inductive bias of CNN+LSTM is the right choice for LOB data. It also introduces rigorous pairwise model comparison via Bayesian statistics, and it's the origin of the architectural benchmark that Briola 2024 later builds on.

---

## Models Compared

Six architectures are systematically compared:

| Model | Architecture | Key Characteristic |
|---|---|---|
| **Random** | Uniform random prediction | Lower floor |
| **Naive** | Always predict the most common class | Lower floor |
| **Logistic Regression** | Linear, L2-regularized, multinomial | Simple, no hidden layers |
| **Shallow LSTM** | Single LSTM layer | Captures temporal structure only |
| **Self-Attention LSTM** | LSTM + self-attention module | Temporal + selective attention |
| **CNN-LSTM (DeepLOB)** | CNN → Inception → LSTM | Spatial + temporal (state-of-art) |
| **MLP** | Multi-layer perceptron (4 layers) | Fully connected, no explicit structure |

---

## Data

- **Stock:** Intel Corporation (INTC) — a large-tick NASDAQ stock (liquid, tight spread, dense order book)
- **Source:** LOBSTER, 2017-2018
- **Lookback window:** 100 LOB snapshots (same as DeepLOB)
- **LOB depth:** 10 levels → 40 features per snapshot

One stock only. This is a limitation acknowledged in the paper — results may not generalize to small-tick stocks (Briola 2024 specifically extends this study to 15 stocks to address this).

---

## Label Design: Quantile-Based Labeling

Rather than using a fixed threshold (like tick size θ), this paper defines the three classes using **quantiles of the empirical return distribution**:

```
q_{-1}: bottom 33rd percentile of returns → "Down"
q_0:    middle ~34% → "Stable"
q_{+1}: top 33rd percentile → "Up"
```

The threshold is empirically set per training set: the boundary between Stable and Up is the 66th percentile of training returns; between Down and Stable is the 33rd percentile.

**Why this is better than fixed-threshold labeling:**

For short horizons (H=10), most mid-price changes are tiny — often much less than 1 tick. If you use tick size as your threshold, 85-90% of samples are labeled "Stable," and any model that just predicts "Stable" achieves 85% accuracy for free. The model learns nothing meaningful.

Quantile-based labeling forces **balanced classes by construction** — each class gets 33% of samples. The model must actually learn to distinguish which 33% are up-moves vs. which are down-moves.

This approach is also used in Lucchese 2022 and is now considered best practice for LOB classification.

---

## Statistical Comparison: Bayesian Correlated t-Test

### The Problem with Naive Comparison

If Model A achieves MCC = 0.36 and Model B achieves MCC = 0.34 across 5 test folds, is A better? Standard t-test says: check if the mean difference is significantly non-zero.

But the standard t-test assumes **independent samples**. Test-set results from consecutive folds of the same stock are not independent — they come from the same market, the same stock, the same regime. Correlation between folds inflates the Type I error rate.

### The Bayesian Correlated t-Test (Benavoli et al. 2017)

This test accounts for the correlation between folds. Given paired per-fold performance metrics (MCC or F-score for model A vs. B), it computes:

- P(A is better than B)
- P(B is better than A)  
- P(they are practically equivalent) — within a practical equivalence region (rope)

**Output:** Models can be grouped into statistical clusters where within-cluster differences are not significant, and between-cluster differences are significant.

---

## Results

### Model Performance Clusters

At all three horizons H∆τ ∈ {10, 50, 100}:

**Top cluster (statistically equivalent):**
```
CNN-LSTM ≈ MLP
```

**Middle-top cluster:**
```
Self-Attention LSTM  (at H=10 only; degrades at H=50, 100)
```

**Middle cluster:**
```
Shallow LSTM
Logistic Regression
```

**Bottom:**
```
Naive / Random
```

The ranking representation (Tables 8-9 in the paper):

| Rank | H∆τ=10 | H∆τ=50 | H∆τ=100 |
|---|---|---|---|
| Best | MLP = CNN-LSTM | MLP = CNN-LSTM | MLP = CNN-LSTM |
| 2nd | Self-Attention LSTM | Shallow LSTM | Logistic Regression |
| 3rd | Shallow LSTM / Log. Reg. | Log. Reg. | Self-Attention LSTM |
| Worst | Naive / Random | Random | Random |

### The Key Claim: MLP ≈ CNN-LSTM

This is the provocative finding. A plain MLP, which:
- Has no concept of "which features are adjacent price levels"
- Treats all 100×40=4000 input values symmetrically
- Applies no temporal sequencing — sees all 100 timesteps simultaneously

...performs statistically identically to CNN-LSTM, which:
- Explicitly models spatial locality via convolutional filters
- Explicitly models temporal sequence via LSTM

**What does this imply?**

Three possible interpretations:

1. **The CNN's spatial assumption is wrong.** LOB levels may not have consistent spatial relationships — so learning "level 3 is always 3 ticks from mid" is not useful. The MLP just learns direct feature → label mappings.

2. **The LSTM's temporal assumption adds little.** The signal at short horizons is concentrated in recent snapshots, not spread across 100 steps. The MLP approximates this with its first layer.

3. **The signal is simple enough that any architecture with sufficient capacity finds it.** The CNN and LSTM provide useful inductive biases, but the dataset is small enough that they don't provide a regularization advantage over MLP.

The paper leans toward interpretations 1 and 3: "time and space are good approximations of the LOB's inner structure, but they should not be considered the real, necessary underlying dimensions."

---

## Self-Attention LSTM: The Interesting Middle Case

### What Self-Attention Does

Standard LSTM processes the 100-step sequence and produces one final hidden state. In self-attention, *every* hidden state can "attend to" every other hidden state:

```
For step t, compute attention weights α over all 100 steps:
  α_i = softmax(score(h_t, h_i))  for i = 1..100

Context vector: c_t = sum(α_i × h_i)  weighted average of all states

Final representation: concat(h_t, c_t) → prediction
```

This is the same mechanism as the Transformer (Vaswani et al., "Attention is All You Need"), but applied on top of an LSTM rather than replacing it.

### Results

At H∆τ=10: Self-Attention LSTM ≈ CNN-LSTM ≈ MLP (all in same top cluster)
At H∆τ=50: Performance drops — outside top cluster
At H∆τ=100: Similar drop

**Interpretation:** At short horizons, the relevant signal is concentrated in the most recent snapshots. Attention can learn to focus on those. At longer horizons, the signal is more diffuse and the LSTM baseline becomes less effective; the attention mechanism amplifies this weakness.

This suggests that **pure Transformer architectures** (no LSTM, just attention) might work differently — they can model longer-range dependencies more effectively through multi-head attention without sequential bottleneck. This is exactly the direction post-2021 work has taken.

---

## Key Conclusions

1. **Don't over-engineer for a single large-tick stock.** A well-tuned MLP can match DeepLOB on INTC. If compute is a constraint, MLP is a valid choice.

2. **The spatial-temporal decomposition of CNN-LSTM may not be the right inductive bias for all stocks.** It works (competitive performance), but it's not uniquely correct.

3. **Self-attention is promising but unstable.** Degrades at longer horizons for this stock. Post-2021 work on full Transformers may overcome this.

4. **For practical selection, use Bayesian correlated t-test.** Don't just compare mean metrics across folds — account for temporal correlation.

5. **Only tested on one large-tick stock.** Results on small-tick stocks are unknown. Briola 2024 shows that nothing predicts small-tick stocks well anyway, so the comparison may be moot there.

---

## Indian Market Implications

| Finding | NSE capstone implication |
|---|---|
| MLP ≈ CNN-LSTM on large-tick stock | Include MLP as a strong baseline — fast to train, competitive |
| Self-attention works at short horizons | Full Transformer (no LSTM) is worth testing as "recent AI/ML" component |
| Quantile-based labeling recommended | Use quantile labels for NSE data to avoid class imbalance |
| Bayesian t-test for comparison | Use MCS (Lucchese) or Bayesian t-test rather than just point MCC |
| Only large-tick (INTC) tested | Test on both large-tick (HDFC, SBI) and small-tick (GOOGL-equivalent NSE) stocks |
| Logistic Regression is solid baseline | Always include Logistic Regression — it performs surprisingly well |
