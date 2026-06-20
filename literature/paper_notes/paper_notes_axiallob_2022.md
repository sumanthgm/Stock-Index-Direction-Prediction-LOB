# Paper Notes: Axial-LOB — High-Frequency Trading with Axial Attention

**Full title:** Axial-LOB: High-Frequency Trading with Axial Attention
**Authors:** Damian Kisiel, Denise Gorse (University College London)
**Published:** arXiv 2212.01807, December 2022 / IEEE SSCI 2022
**File:** `[N3] AxialLOB_Kisiel_2022.pdf`

---

## Why This Paper Matters

Axial-LOB answers a specific question: can we get global (long-range) attention over LOB data without the O(h²w²) quadratic cost of full 2D self-attention?

The answer is **axial attention** — factorize 2D attention into two sequential 1D attention operations (one along the width/feature axis, one along the height/time axis). This gives global receptive field at linear complexity O(hwm).

Two additional contributions:
1. **9,615 parameters** — 10× smaller than DeepLOB (142,435 params), yet SOTA performance.
2. **Feature-order invariant** — unlike CNN-based models, it doesn't require features to be in a specific spatial arrangement.

---

## The Core Problem with CNN-Based Models

DeepLOB and its variants extract features using convolutions. Convolutions assume **spatial locality**: filters only combine information from nearby elements. This works well when the assumption holds — e.g., "bid level 1 is next to bid level 2."

But LOB data may not have consistent spatial relationships:
- Spread can vary: today "level 3 is 3 ticks from mid," tomorrow "level 3 is 5 ticks from mid"
- For small-tick stocks, spatial structure is even more inhomogeneous
- CNNs require a specific ordering of input features (prices before volumes, ask before bid) — changing the order breaks the model

**Axial-LOB is designed to work without these spatial assumptions.**

---

## Background: From 2D Attention to Axial Attention

### Full 2D Self-Attention (Too Expensive)

For an input X ∈ ℝ^(C_in × H × W), standard self-attention computes:
```
y_ij = sum_{h=1}^{H} sum_{w=1}^{W} softmax(q_ij^T k_hw) v_hw
```

Complexity: O(h²w²) — for a 100×40 LOB input, this is O(100²×40²) = 16 million operations per layer.

### Axial Attention (Global Receptive Field, Cheaper)

Factorize into two sequential 1D operations:
1. **Width axis attention** (attends over W=40 features at each time step independently)
2. **Height axis attention** (attends over H=100 time steps for each feature independently)

Complexity: O(hwm) where m = number of features — linear in both dimensions.

The result still achieves a global receptive field: every position can "see" every other position after both attention passes. The factorization just means it happens in two stages instead of one expensive 2D pass.

---

## Gated Positional Embeddings

Standard axial attention (Wang 2020) adds learned relative positional encodings to attention scores:
```
y_ij = sum_h softmax(q_ij^T k_hj + q_ij^T r^q_hj + k_hj^T r^k_hj)(v_hj + r^v_hj)
```

Where r^q, r^k, r^v ∈ ℝ^(H×H) are learnable positional bias terms.

**The problem with financial data:** Positional encodings assume "position 3 is always 3 time steps ago." But financial time series are noisy — the positional bias can mislead the model if learned inaccurately from noisy data.

**Gated positional embeddings** (Valanarasu 2021, medical imaging) add learnable gates g^q, g^k, g^v to control how much positional information flows:
```
y_ij = sum_h softmax(q_ij^T k_hj + g_q b_q + g_k b_k)(v_hj + g_v b_v)
```

Where b^q = q^T_ij r^q_hj, b^k = k^T_hj r^k_hj, b^v = r^v_hj.

**Intuition:** The gates learn whether positional information is helpful for this particular dataset. If the data is noisy (financial data), the gates can suppress irrelevant positional biases.

**Design choice in Axial-LOB:** Unlike the original, gates only control the *positional bias* terms, not the full value vectors. This is a deliberate choice — positional noise should be gated, but the content of the values (actual LOB price/volume data) should not be suppressed.

The gate training is delayed until epoch 5 to allow stable attention weights to form first.

---

## Architecture

```
Input: X ∈ ℝ^(H×W×1) = 40 time steps × 40 features × 1 channel
  [Note: paper uses 40 time steps, not 100 like DeepLOB]
  ↓
Conv 1×1 + Batch Norm  [adjust channel count]
  ↓
┌─────────────────────────────────────────────────────┐
│  Gated Axial Attention Block (repeated 2 layers)     │
│  ┌─────────────────────────────────────────────────┐│
│  │ Layer 1:                                        ││
│  │   Gated Multi-Head Axial Attention (Width Axis) ││
│  │   → Gated Multi-Head Axial Attention (Height Axis)││
│  │   + residual connection                         ││
│  │ Layer 2: same                                   ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
  ↓
Conv 1×1 + Batch Norm  [channel pooling]
  ↓
Adaptive Average Pooling
  ↓
Fully Connected Layer
  ↓
Softmax → 3 classes
```

**Input:** Only 40 LOB snapshots (vs. 100 in DeepLOB). X = [p^ask_i, v^ask_i, p^bid_i, v^bid_i]_{i=1}^{10} ∈ ℝ^{40×1} per snapshot.

**Width axis attention** processes all 40 features at a given time step simultaneously — captures spatial relationships between price levels and volumes.

**Height axis attention** processes all 40 (or fewer) time steps for each feature — captures temporal evolution of each feature.

**1×1 convolutions** do not perform spatial convolutions. They only do channel-wise mixing (changing the number of channels), not spatial filtering. This is what preserves the permutation invariance.

---

## Training Details

- **Optimizer:** SGD with momentum
- **Batch size:** 64
- **Epochs:** 100 with early stopping (stop if validation loss doesn't improve for 10 epochs)
- **Learning rate schedule:** Cosine annealing (LR_decay = ½ × (1 + cos(π × T_cur/T_total)))
- **Gate training:** Delayed to epoch 5 for stability
- **Hardware:** NVIDIA Tesla P100 16GB GPU
- **Hyperparameter search:** 100 iterations of random grid search

---

## Results on FI-2010

Using α = 0.002 threshold for labeling (same as prior work):

| Model | k=10 | k=20 | k=30 | k=50 | k=100 |
|---|---|---|---|---|---|
| CNN | 55.21 | 59.17 | 65.72 | 59.44 | 65.05 |
| B(TABL) | 69.20 | 62.22 | 67.08 | 73.64 | 69.14 |
| C(TABL) | 77.63 | 66.93 | 69.34 | 78.44 | 74.94 |
| DeepLOB | 83.40 | 72.82 | 75.33 | 80.35 | 76.76 |
| DeepLOB-Seq2Seq | 81.51 | 72.99 | 75.75 | 77.99 | 79.16 |
| DeepLOB-Attention | 82.37 | 73.73 | 76.94 | 79.38 | 81.49 |
| **Axial-LOB** | **85.14** | **75.78** | **80.08** | **83.27** | **85.93** |

Axial-LOB achieves SOTA at all prediction horizons (at time of publication, 2022). Results reported as mean of 5 independent runs.

---

## Model Complexity Comparison

| Model | Parameters |
|---|---|
| CNN | 17,635 |
| B(TABL) | 5,844 |
| C(TABL) | 11,344 |
| DeepLOB | 142,435 |
| DeepLOB-Seq2Seq | 176,419 |
| DeepLOB-Attention | 177,699 |
| **Axial-LOB** | **9,615** |

Axial-LOB has fewer parameters than even the simplest TABL models, while outperforming the much larger DeepLOB family.

---

## Feature Permutation Robustness

**Experiment:** Start from the same weights, run 5 trials with different random permutations of the 40 input features. Record mean ± std of F1 change.

| Horizon | Axial-LOB (ΔF1) | DeepLOB-Attention (ΔF1) |
|---|---|---|
| k=10 | −0.94 ± 0.22 | −2.53 ± 0.49 |
| k=20 | −0.56 ± 0.33 | −2.48 ± 0.44 |
| k=30 | −1.05 ± 0.30 | −3.09 ± 0.50 |
| k=50 | −0.68 ± 0.36 | −4.01 ± 0.86 |
| k=100 | −0.37 ± 0.35 | −4.81 ± 0.83 |

**Interpretation:**
- DeepLOB-Attention loses up to 4.81 F1 points when features are reordered — it has learned to depend on the specific input layout.
- Axial-LOB loses only ~0.37–1.05 F1 points — its attention-based feature mixing doesn't assume a specific ordering.

**Why this matters for your capstone:** When adapting to NSE data, the ordering of features in the input tensor may differ from NASDAQ conventions. Axial-LOB will be more robust to such differences.

---

## Key Conclusions

1. **Axial attention achieves global receptive field with fewer parameters than CNN-LSTM.** The 10× parameter reduction while outperforming DeepLOB is significant.

2. **Gating positional encodings is important for noisy financial data.** Without gates, noisy positional biases can hurt performance.

3. **Feature permutation robustness is a real advantage.** Axial-LOB can incorporate additional LOB features (e.g., order flow features alongside raw prices) without redesigning the network.

4. **Short window (40 snapshots) suffices.** Axial-LOB uses only 40 timesteps vs. DeepLOB's 100, suggesting the most predictive information is in recent snapshots.

5. **Limitation: Only tested on FI-2010.** The paper acknowledges this limitation and calls for testing on larger datasets. Prata 2023's benchmark subsequently includes Axial-LOB and finds it generalizes reasonably well (rank 3 on LOB-2021/2022, F1≈71.3%).

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Feature permutation robustness | Can easily add order flow features or NSE-specific features to input without architectural changes |
| Only 40 input timesteps needed | Shorter input window → faster training and inference |
| 9,615 parameters | Very fast training — suitable for capstone compute constraints |
| Global attention with linear complexity | Scales better than CNN-LSTM to longer sequences if needed |
| Outperforms DeepLOB on FI-2010 (SOTA at 2022) | Strong baseline for comparison; now surpassed by TLOB but still relevant |
| Width-axis attention = feature interactions | For NSE, may discover different bid-ask relationships than NASDAQ |
| Limitation: not tested on out-of-sample data | Prata benchmark shows it does generalize reasonably; important context |
