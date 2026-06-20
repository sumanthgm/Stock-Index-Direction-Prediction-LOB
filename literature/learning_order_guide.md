# Recommended Learning Order — LOB Literature Survey

19 papers, organized into 7 stages. Each stage builds on the previous.  
Papers marked **[CSV]** are in the team tracking CSV. Papers marked **[Prior]** are from the earlier foundational review.

---

## Stage 1 — What Is LOB Prediction? (4 papers)

Read these to understand the field, the benchmark, and the baseline before any architecture paper.

### 1. LOB Survey — Zaznov et al. 2022 **[CSV #3]**
`paper_notes_lob_survey_2022.md`  
**Read first.** Full taxonomy of LOB prediction (statistical → classical ML → CNN → LSTM → attention). Most importantly: the simulation-to-reality gap, mid-price execution assumption, and FI-2010 over-reliance — critiques you'll need to understand every other paper in context.

### 2. FI-2010 Dataset — Ntakaris et al. 2018 **[Prior]**
`paper_notes_ntakaris_2018.md`  
The benchmark dataset every paper uses. Understand its construction (5 Finnish stocks, 10 days, 144 features, 3 labeling horizons), its normalization convention (z-score), and its limitations. You cannot read DeepLOB or LOBCAST without this.

### 3. DeepLOB — Zhang et al. 2019 **[Prior]**
`paper_notes_zhang_2019.md`  
The baseline CNN+LSTM architecture. The 100×40 input tensor format, 5-day rolling z-score normalization, and 3-class labeling are all defined here and referenced by every subsequent paper. Every comparison table in this survey uses DeepLOB as a reference point.

### 4. Comparative (MLP ≈ CNN-LSTM) — Briola et al. 2020 **[Prior]**
`paper_notes_briola_2020.md`  
**Reality check.** On FI-2010, a simple MLP matches DeepLOB in performance — suggesting the dataset is too easy, not that attention or CNNs don't matter. Prepares you for the generalizability crisis papers.

---

## Stage 2 — What Makes LOB Data Predictable? (3 papers)

Understand the microstructure signals before studying architectures designed to exploit them.

### 5. Deep Order Flow Imbalance — Kolm et al. 2021 **[Prior]**
`paper_notes_kolm_2021.md`  
Order flow (executed trades) is informationally richer than the static LOB snapshot. Introduces the "information richness" concept: stocks with more updates per price change are more predictable. Foundation for understanding why TrioFlow (Stage 6) adds OF as an input and why LOB-only models have limits.

### 6. Short-Term Predictability of Returns in LOB Markets — Lucchese et al. 2022 **[Prior]**
`paper_notes_lucchese_2022.md`  
deepOF and deepVOL as distinct input representations; seq2seq multi-horizon architecture; MCS framework for model comparison. Introduces the question of *when* and *why* short-term LOB prediction works — predictability varies by stock, horizon, and representation. Essential context for the main capstone paper.

### 7. Main Paper — Briola et al. 2024 **[CSV #1, #24]**
`paper_notes_briola_2024.md`  
Read after Lucchese (same group). Synthesizes microstructure priors into architecture design: tick-size-dependent normalization, deepOF vs deepVOL comparison, the p_T practical metric. This is the capstone's primary reference — framing all experiments on NSE data.

---

## Stage 3 — What Actually Generalizes? (2 papers)

Understand the evaluation crisis before studying SOTA models.

### 8. LOB Benchmark + LOBCAST — Prata et al. 2023 **[CSV #7]**
`paper_notes_lob_benchmark_2023.md`  
Empirically confirms the FI-2010 crisis: models achieving 80%+ F1 on FI-2010 drop by up to 19.6% on fresh LOBSTER data. BINCTABL is the most generalizable model. The LOBCAST framework is what you should use to implement and compare models. After reading this, you'll understand why benchmark results in earlier papers are unreliable.

### 9. ICLR 2025 Benchmark — Anonymous 2025 **[CSV #5]**
`paper_notes_iclr_benchmark_2025.md`  
Extends the benchmark question: can general time-series models (PatchTST, iTransformer) replace LOB-specific models? Answer: no without adaptation. CVML plug-in module recovers most of the gap. Introduces MPRF (continuous return prediction) as a more practically meaningful metric than F1.

---

## Stage 4 — Attention Architectures (3 papers)

Three different takes on attention for LOB, building toward TLOB.

### 10. DLA — Guo & Chen 2023 **[CSV #4]**
`paper_notes_dla_2023.md`  
Dual temporal attention: Stage 1 weights input features before LSTM, Stage 2 weights hidden states after. 2nd most generalizable model in LOBCAST. Read this to understand the temporal attention lineage before Axial-LOB and TLOB.

### 11. Axial-LOB — Kisiel & Gorse 2022 **[CSV #8]**
`paper_notes_axiallob_2022.md`  
Factored 2D attention: separate attention over the time axis and the feature axis. Only 9,615 parameters — the most parameter-efficient model in the literature. Contrast with DLA (both focus on different dimensions). Feature-order invariance is a useful inductive bias for LOB data.

### 12. TLOB + MLPLOB — Berti & Kasneci 2025 **[Prior]**
`paper_notes_tlob_2025.md`  
Current SOTA. Dual independent self-attention: temporal SA (like DLA) + spatial SA (like Axial-LOB's feature axis) applied simultaneously. Bilinear Normalization stabilizes training. MLPLOB shows that even without attention, careful normalization + residual MLP is competitive. Read after DLA and Axial-LOB — TLOB makes most sense as a synthesis of both.

---

## Stage 5 — Alternative Input Representations (4 papers)

Non-attention approaches: different ways to represent LOB data.

### 13. Imaging LOB — Ye et al. 2023 **[CSV #2]**
`paper_notes_imaging_lob_2023.md`  
Convert the LOB tensor to a 2D image and apply CNN image classification. The normalization step (image-space) handles cross-stock price-scale heterogeneity in a different way from z-score. Useful for NSE where stocks range from ₹10 to ₹10,000+.

### 14. Deep Hybrid LOB — Nguyen et al. 2022 **[CSV #16]**
`paper_notes_deep_hybrid_lob_2022.md`  
ResNet50 (ImageNet pretrained) + LSTM applied to LOB. Shows that transfer learning from image classification to LOB data is feasible — though not architecturally motivated. Read as a contrast to the LOB-specific designs; highlights what happens without microstructure priors.

### 15. HLOB — Briola, Bartolucci, Aste 2024 **[Prior]**
`paper_notes_hlob_2024.md`  
TMFG graph from mutual information between LOB volume levels → HCNN processes graph simplices (tetrahedra/triangles/edges). Captures non-consecutive level dependencies that standard CNNs miss. Large-tick stocks show hierarchical structure; small-tick stocks near-diagonal. The most structurally innovative architecture in the survey.

---

## Stage 6 — Probabilistic Methods and Non-NASDAQ Markets (3 papers)

Uncertainty quantification and generalization beyond NASDAQ.

### 16. Bayesian TABL — Magris et al. 2023 **[CSV #9]**
`paper_notes_bayesian_bilinear_2023.md`  
VOGN Bayesian inference on the TABL (bilinear + temporal attention) architecture. Produces calibrated probability distributions over {Up, Stationary, Down}. First paper in this review to treat prediction uncertainty as a first-class output. Outperforms MC Dropout on calibration quality.

### 17. GP Models for LOB — Liu 2024 **[CSV #6]**
`paper_notes_gp_lob_2024.md`  
Non-parametric Bayesian alternative to DL. GPR beats TABL by 25–32% MAE for long-horizon mid-price regression. GPC competitive at H=5,10 but weak at H=1. Built-in confidence intervals; combined linear (price) + RBF (volume) kernel encodes LOB domain knowledge. Read alongside Bayesian TABL as two approaches to the same uncertainty problem.

### 18. TrioFlow CNN+GRU — Zaznov et al. 2024 **[CSV #22]**
`paper_notes_trioflow_2024.md`  
Three-branch inception-style CNN + GRU with joint LOB + Order Flow input (70 features). The MICEX Russian dataset (Sberbank, VTB, Gazprom) is the closest existing analog to Indian NSE market conditions — non-US, emerging market dynamics, different liquidity profiles. Outperforms DeepLOB on all 6 stocks.

---

## Stage 7 — Frontier: Generative Models (1 paper)

Most conceptually advanced. Read last.

### 19. LOBS5 — Nagy et al. 2023 **[Prior]**
`paper_notes_lobs5_2023.md`  
First autoregressive generative model for raw LOB messages (not prices/returns). 12,011-token vocabulary; 22 tokens per message; S5 state space model. Generates synthetic LOB sequences that match real mid-price return distributions. Framing: LOB as a world model for RL trading agents. Conceptually distinct from all other papers — read last.

---

## Quick Reference: Dependency Map

```
Stage 1 (Foundation)          Stage 2 (Predictability)
  Survey (1)                    OFI (5)
  FI-2010 (2)  ←──────────      Lucchese (6)
  DeepLOB (3)  ←──┐         ↗  Main Paper (7)
  Comparative (4) │       ↗
                  │     ↗
Stage 3 (Benchmarks)            Stage 4 (Attention)
  LOBCAST (8) ←──┘              DLA (10)
  ICLR 2025 (9)                 Axial-LOB (11)
                                TLOB (12) ← synthesis of 10+11

Stage 5 (Representations)       Stage 6 (Probabilistic)
  Imaging (13)                  Bayesian TABL (16)
  Hybrid (14)                   GP Models (17)
  HLOB (15)                     TrioFlow (18) ← adds OF signal (5)

Stage 7 (Frontier)
  LOBS5 (19)
```

---

## Suggested Reading Plan by Role

**For understanding the full literature (all 19):** Follow stages 1→7 in order.

**For implementation focus (minimum viable for experiments):**
Read: FI-2010 (2) → DeepLOB (3) → LOBCAST (8) → Main Paper (7) → TLOB (12)

**For writing the related work section:**
Read: Survey (1) → LOBCAST (8) → ICLR 2025 (9) → then skim architecture papers 10–15 for 1-paragraph each

**For Indian market framing:**
Focus on: Survey (1, section on gaps) → Main Paper (7, Indian market sections) → TrioFlow (18, MICEX analog) → ICLR 2025 (9, asset transferability)
