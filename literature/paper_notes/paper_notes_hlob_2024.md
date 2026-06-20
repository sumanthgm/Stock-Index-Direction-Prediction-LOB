# Paper Notes: HLOB — Information Persistence and Structure in Limit Order Books

**Full title:** HLOB — Information Persistence and Structure in Limit Order Books
**Authors:** Antonio Briola, Silvia Bartolucci, Tomaso Aste (University College London)
**Published:** arXiv 2405.18938, June 2024
**File:** `[N2] HLOB_Briola_2024.pdf`

---

## Why This Paper Matters

HLOB makes a fundamental observation that no prior model had addressed properly: **the dependency structure of a LOB is not between consecutive levels — it's between non-consecutive levels, and those dependencies are stock-specific.**

DeepLOB uses convolutions with a (1×2) stride to combine each level's price and volume. This implicitly assumes that level 1 is most similar to level 2, level 2 to level 3, and so on. But in real LOBs, especially for small-tick stocks, the "distance" between consecutive price levels varies wildly. Two orders listed as "level 1" and "level 2" might be 50 ticks apart, while levels 4 and 7 might carry more correlated information. CNN models are blind to this.

This paper's answer: compute the actual mutual information (MI) between every pair of the 20 volume levels, build a TMFG (Triangulated Maximally Filtered Graph) from the MI matrix, then design the neural network specifically around the topological structures the TMFG reveals — tetrahedra, triangles, and edges.

---

## The Core Problem with Consecutive-Level CNN Models

Standard CNN-based LOB models convolve across consecutive LOB levels. This only works reliably when:
1. The concept of "level" is spatially consistent (levels 1-2 are always roughly the same distance apart)
2. Most information is concentrated near the best bid/ask (level 1)

Neither holds for small-tick stocks. The "actual LOB depth" metric Ξ^{Bid, Ask} (from the Briola 2024 main paper) measures how many distinct price levels are actively populated. For CHTR (small-tick), mean Ξ^Ask = 53.83 — meaning on average 54 active ask price levels, crammed into 10 standardized "level" bins. The bins are arbitrary; consecutive bins may contain vastly different amounts of information.

For large-tick stocks (BAC, CSCO): mean Ξ ≈ 9. All 10 standard levels are always populated, and the concept of "level" is real and stable. CNN models work well here.

---

## Background: Information Filtering Networks (IFNs) and the TMFG

An IFN is a graph built to represent the true dependency structure in a multivariate dataset. It finds the network G* that best approximates the joint distribution f(L) by minimizing the KL divergence — equivalently, minimizing cross-entropy H:

```
G* = argmin_G D_KL(f(L) || f̃(L|G))
   = argmin_G H(L|G)
```

The TMFG algorithm builds this graph incrementally: at each step, it joins the two nodes sharing the largest mutual information, subject to topological constraints (planar and chordal). The result is a sparse graph that retains the statistically strongest dependencies while pruning weak ones.

**Key TMFG properties:**
- **Planar:** embeddable on a sphere without edge crossings
- **Chordal:** every cycle of 4+ nodes has a chord, decomposing into triangles. Enables exact Gaussian likelihood computation.
- Captures up to 4-variable clique interactions — richer than trees (MST) or planar graphs alone (PMFG)

---

## The TMFG Building Process (Section 4.1)

**Step 1: Volume-only representation**

Remove price levels from the LOB snapshots. Each snapshot reduces from ℝ^{4L} to ℝ^{2L} (L=10 → 20 volume levels: 10 ask volumes + 10 bid volumes). Prices are dropped because they vary continuously and introducing them into MI computation adds noise.

Volume levels are binned into equally-spaced bins to reduce noise from small quantity fluctuations. The number of bins is optimized on training/validation and stays constant across tick-size classes; bin size is computed per-stock per-trading-day.

**Step 2: Mutual information matrix**

For each stock, for each training day, compute all pairwise MI between the 20 volume levels → (20×20) symmetric MI matrix. A bootstrapping procedure is applied daily for reliability. Final MI matrix = average of daily MI matrices over all training days.

**Step 3: TMFG algorithm**

Apply TMFG using the average MI matrix as the similarity matrix. The result is a stock-specific graph capturing its unique volume dependency structure.

**Output — 3 simplicial families (topological structures):**

With 20 volume levels, the TMFG always produces:
- **Tetrahedra** (4-cliques, 3D simplices): **17 tetrahedra** → shape (17×4)
- **Triangles** (3-cliques, 2D simplices): **52 triangles** → shape (52×3)
- **Edges** (2-cliques, 1D simplices): **54 edges** → shape (54×2)

The TMFG is computed once from training data and stays fixed for all test-time predictions. It encodes the "average spatial structure" of that stock's LOB.

---

## From TMFG to HLOB Architecture (Section 4.2)

For each of the 100 timestamps in the input window, the simplicial families are "flattened" and price levels are reinserted alongside the volume levels:

Each simplex member identifies a volume level → insert the corresponding price level alongside:
- Tetrahedron: 4 volume levels + 4 corresponding price levels = 8 features → 17 tetrahedra × 8 = 136 features
- Triangle: 3 volume levels + 3 price levels = 6 features → 52 triangles × 6 = 312 features
- Edge: 2 volume levels + 2 price levels = 4 features → 54 edges × 4 = 216 features

Result for 100 timestamps:
- **Tetrahedra input:** 100 × 136
- **Triangles input:** 100 × 312
- **Edges input:** 100 × 216

These three tensors feed into three parallel CNN heads:

```
Input[100×136]     Input[100×312]     Input[100×216]
(Tetrahedra)       (Triangles)          (Edges)
      ↓                   ↓                  ↓
  CNN Head 1          CNN Head 2          CNN Head 3
      ↓                   ↓                  ↓
                     Concatenate
                          ↓
                    LSTM @ 32 units
                          ↓
                    Dense @ 3 units
                    (Up/Down/Stable)
```

**Each CNN head has 3 convolutional layers:**

*Layer 1:* `1×2@32, stride (1×2)` — combines the price and volume of each node (same design as DeepLOB's first conv layer, which pairs p_ask and v_ask etc.). Parameters: 96 per head.

*Layer 2 (within-simplex):* Captures dependencies between the nodes of each simplex:
- Tetrahedra head: `1×4@32, stride (1×4)` — combines all 4 nodes. Params: 12,384.
- Triangles head: `1×3@32, stride (1×3)` — combines all 3 nodes. Params: 11,360.
- Edges head: `1×2@32, stride (1×2)` — combines both nodes. Params: 10,336.

*Layer 3 (cross-simplex):* Captures relationships between different simplices of the same family (this layer goes beyond the TMFG structure, hence dropout @0.35 is applied for regularization):
- Tetrahedra head: `1×17@32` (17 = number of tetrahedra). Params: 17,440.
- Triangles head: `1×52@32`. Params: 53,280.
- Edges head: `1×54@32`. Params: 55,328.

After 3 layers, each head outputs feature maps of shape (100×1). Concatenated → **LSTM@32 units** (16,640 params, captures temporal dynamics) → **Dense@3** (3 output classes).

**Total: ~1.8×10^5 parameters. Inference time: 0.16ms.**

---

## Datasets and Training

**15 NASDAQ stocks (Jan 2017 – Dec 2019) from LOBSTER:**

| Tick class | Stocks | Criterion |
|---|---|---|
| Small-tick | CHTR, GOOG, GS, IBM, MCD, NVDA | avg spread ≥ 3θ |
| Medium-tick | AAPL, ABBV, PM | 1.5θ ≤ avg spread < 3θ |
| Large-tick | BAC, CSCO, KO, ORCL, PFE, VZ | avg spread ≤ 1.5θ |

θ = $0.01 on NASDAQ.

**Data split per year:** 40 training days, 5 validation days, 10 test days. Training days are mostly consecutive but not all; validation days randomly sampled from the training period.

**Labeling (different from Zhang 2019):** Simple tick-difference at the prediction horizon:
```
m_{τ+Δτ} − m_τ ≤ −θ → Down (−1)
|m_{τ+Δτ} − m_τ| < θ → Stable (0)
m_{τ+Δτ} − m_τ ≥ +θ → Up (+1)
```
No smoothing window — uses raw mid-price difference. This gives finer control over the signal amplitude at each horizon.

**Training details:**
- Optimizer: AdamW; lr = 6×10^{-5}, β1 = 0.90, β2 = 0.95
- Max epochs: 100; early stopping if validation loss doesn't drop by 0.003 for 15 consecutive epochs
- Batch size: 32 (random balanced sampling — 5000 examples per class per day; if fewer available, use minimum class count)
- Loss: categorical cross-entropy
- Total compute: 1,350 experiments × 7,192 GPU-hours across 10 different GPU types

**10 benchmark models:** CNN1 (2017), CNN2 (2020), DLA (2022), BinBTabl (2021), BinCTabl (2021), DeepLOB (2019), Transformer (vanilla), iTransformer (2023), LobTransformer (new), HLOB.

---

## Key Results

### H∆τ = 10 (Table 4) — HLOB best in 73.3% of cases

- Small-tick: best in 4/6 (CHTR, GS, IBM, MCD); 2nd in GOOG; 3rd in NVDA
- Medium-tick: best in all 3 (AAPL, ABBV, PM)
- Large-tick: best in 4/6 (BAC, CSCO, KO, PFE); 2nd in ORCL, VZ

| Metric | Small-tick avg | Medium-tick avg | Large-tick avg |
|---|---|---|---|
| F1 | 0.42 | 0.41 | 0.48 |
| MCC | 0.16 | 0.16 | 0.33 |
| p_T | 0.11 | 0.14 | 0.09 |

Gain vs. DeepLOB at H10: F1 +0.03 (small), +0.02 (medium), +0.003 (large); MCC +0.04, +0.02, +0.02.

### H∆τ = 50 (Table 5) — HLOB best in 60% of cases

- Small-tick: degrades significantly — best only for IBM; 3rd in most others
- Medium-tick: still best in all 3
- Large-tick: best in 5/6 (BAC, CSCO, KO, ORCL, VZ); 2nd in PFE

| Metric | Small-tick avg | Medium-tick avg | Large-tick avg |
|---|---|---|---|
| F1 | 0.36 (−16.7% vs H10) | 0.40 (−2.5%) | 0.58 (**+17.2%**) |
| MCC | 0.09 (−77.8%) | 0.11 (−45.5%) | 0.38 (+13.2%) |
| p_T | 0.07 (−57.1%) | 0.10 (−40%) | 0.14 (−35.7%) |

**Large-tick stocks actually improve F1 and MCC from H10 to H50** — the spatial structure captured by the TMFG is persistent over medium horizons for stable LOBs.

### H∆τ = 100 (Table 6) — HLOB best in 33% of cases

- Small-tick: best in 1/6 (IBM only)
- Medium-tick: best in 3/3, but margin over BinCTabl/BinBTabl shrinks
- Large-tick: best in 4/6 (BAC, CSCO, KO, ORCL); 2nd in PFE; 3rd in VZ

At H100, BinCTabl and BinBTabl (dual-attention models over both spatial and temporal dimensions) overtake HLOB for medium-tick stocks because the IFN only models spatial structure — at long horizons, temporal dynamics dominate.

### Trading Performance (Figure 4 — p_T quadrant analysis)

Three consistent groups across all horizons:

**Group 1 (upper-right quadrant): BinBTabl, BinCTabl, HLOB**
Most effective at correctly predicting round-trip transactions; relatively active traders.

**Group 2 (lower-left quadrant): iTransformer, LobTransformer**
Worst performers across all horizons and metrics. Naive Transformer adaptation to LOBs fails.

**Group 3 (mixed): CNN1, CNN2, DLA, Transformer, DeepLOB**
Borderline behavior, often between quadrants III and IV.

---

## Spatial Distribution of Information (Section 5.2)

The MI matrices reveal fundamentally different LOB structures:

**Small-tick stocks:**
- Low non-normalized average MI: CHTR=0.35, GOOG=0.26 (IBM=0.74 is an outlier)
- MI concentrated along same-side contiguous levels (near-diagonal only)
- Weak cross-side dependencies (ask volumes barely correlate with bid volumes)
- High Ξ values (CHTR: mean 53.83 ask levels) — the "level" concept is nearly arbitrary

**Medium-tick stocks:**
- AAPL: MI=0.41, level 1 detached from others (best bid/ask is too noisy for medium-tick)
- ABBV, PM: MI=0.59, 0.63 — similar to small-tick, top 7 levels most informative
- AAPL's lower MI compensated by high LOB stability (low Ξ ≈ 9)

**Large-tick stocks:**
- High non-normalized MI: BAC=1.18, CSCO=1.00, KO=0.83, PFE=0.90
- Clear hierarchical structure in 3 clusters: level 1 (lowest MI), levels 2-3 (intermediate), levels 4-10 (highest)
- Strong cross-side dependencies (ask volumes correlate with bid volumes across levels)
- All stocks have Ξ ≈ 9 — all 10 standard levels are always populated

**Why this explains HLOB's differential performance:**

For large-tick stocks, the TMFG MI matrix has rich, stable, hierarchical structure. The HCNN head can learn meaningful non-consecutive dependencies (e.g., ask level 5 correlates with bid level 7). For small-tick stocks, the MI matrix is nearly flat and concentrated near the diagonal — the TMFG adds little over simple consecutive-level convolution.

The HLOB advantage persists at longer horizons for large-tick stocks because the spatial structure itself is stable over time (Ξ ≈ 9 consistently). For small-tick stocks, the structure "drifts" as the book sparsity changes, invalidating the frozen TMFG.

---

## Key Conclusions

1. **LOB spatial structure requires higher-order dependency modeling.** Consecutive-level convolution (as in DeepLOB) is suboptimal — the TMFG reveals non-trivial dependencies between non-adjacent volume levels.

2. **The persistence of spatial structure is tick-size dependent:**
   - Large-tick: compact, stable MI structure → HLOB holds its advantage across all horizons
   - Small-tick: sparse, drifting structure → HLOB advantage disappears after H10

3. **HLOB is universally best at H10** regardless of tick size. The averaged TMFG captures short-term patterns even in volatile small-tick LOBs.

4. **At longer horizons (H50, H100), dual-attention models pull ahead** because the IFN captures only spatial dynamics — BinCTabl/BinBTabl add temporal attention that HLOB lacks.

5. **Naive Transformer for LOBs fails.** iTransformer and LobTransformer are consistently the worst models, placing in the lower-left quadrant. This contrasts with TLOB (Berti 2025), which uses specially designed dual attention — standard Transformer adaptation is not enough.

6. **Code is in LOBFrame** (github.com/FinancialComputingUCL/LOBFrame) with full reproducibility.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Large-tick stocks have rich cross-side MI structure | NSE banking stocks (HDFC, ICICI, SBI) are likely large-tick equivalents — test MI matrices; HLOB should work well |
| Small-tick stocks: TMFG structure is sparse and unstable | NSE small-cap stocks — don't expect HLOB gains; stick to BINCTABL or TLOB |
| Large-tick F1 actually improves from H10 to H50 | For NSE large-tick stocks, medium horizons may be the sweet spot |
| Level concept unreliable for high-Ξ stocks | NSE tick sizes vary (₹0.05–₹1.00 by price band) — compute Ξ for your NSE stocks before choosing models |
| LOBFrame has full HLOB implementation | Fork LOBFrame for the NSE pipeline — saves significant implementation effort |
| iTransformer/LobTransformer worst performers | Use TLOB's specific dual-attention rather than any standard Transformer adapter |
| Computing MI matrix for NSE stocks is novel | Visualizing the MI matrix for NSE stocks and comparing it to NASDAQ patterns is a publishable contribution |
| Frozen TMFG assumption | For NSE, test whether the TMFG structure changes across market regimes (pre-COVID vs. post-COVID); temporally evolving TMFG is future work even in this paper |
