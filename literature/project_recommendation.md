# Project Recommendation: Mamba-LOB

**Mamba-LOB — Selective State Space Models for Mid-Price Direction Prediction on NSE India**

---

## Why This Is the Right Choice

The literature survey surfaces two independent gaps that no existing paper addresses simultaneously:

**Geographic gap (Gap 1):** Zero published LOB prediction studies on NSE/BSE. Every model in the literature — DeepLOB, TLOB, BINCTABL, TrioFlow — is evaluated on NASDAQ or MICEX. Indian market microstructure (variable tick sizes, circuit breakers, STT, retail-heavy participation) is structurally different and untested.

**Architecture gap:** Mamba (Gu & Dao, 2023) is the most significant sequence modeling advance since Transformers — a Selective State Space Model (S6) that uses *input-dependent* state updates (selective forgetting) with O(L) complexity vs. O(L²) for attention. It has not been applied to LOB prediction. The closest work is LOBS5 (Nagy 2023), which used the earlier S5 SSM for generative LOB modeling, but never for discriminative mid-price direction prediction.

Combining both gaps gives the capstone four independent novelty dimensions — any academic reviewer would accept at least two of them.

---

## The Architecture

```
Input: LOB snapshot sequence (100 × 40 tensor, or 100 × 70 with OF features)
           ↓
   Mamba Block × N
   - Selective state update: Δ, B, C = f(input)  [input-dependent gating]
   - Linear scan: O(L) — no attention matrix
   - Hardware-aware parallel scan for GPU efficiency
           ↓
   [Optional] Spatial MHA head over the 40-feature axis  ← from TLOB's spatial SA
           ↓
   BiN (Bilinear Normalization)  ← from TLOB, shown to stabilize LOB training
           ↓
   FC(3) → Softmax → {Up, Stationary, Down}
```

**Why Mamba is architecturally well-suited to LOB:**

- LOB sequences have *sparse informativeness* — most time steps are noise; occasional events (large market orders, spoofing, iceberg reveals) are highly informative. Mamba's selective state mechanism learns to remember the informative events and compress the rest, which is exactly what the LOB signal requires.
- TLOB's spatial self-attention (over features) remains valuable and can be kept as a parallel branch. Mamba replaces the temporal self-attention.
- O(L) complexity means the input window can be extended from 100 to 500+ snapshots without quadratic cost blowup — testing whether longer context improves prediction is a clean ablation experiment.

---

## Baselines

| Baseline | Why Include |
|---|---|
| DeepLOB (Zhang 2019) | Universal baseline — every LOB paper compares against it |
| TLOB (Berti 2025) | Current SOTA — the paper to beat |
| BINCTABL (Prata 2023) | Most generalizable model per LOBCAST benchmark |
| TrioFlow (Zaznov 2024) | Best LOB+OF model; also tested on non-US data (MICEX) |
| MLPLOB (Berti 2025) | Ablation baseline — shows attention isn't always necessary |

---

## NSE Data and Implementation Plan

**Data source:** NSE provides historical tick data through its MOAD (Market on Demand) feed. Alternatively, commercial vendors (Refinitiv Tick History, QuantQuote) carry NSE L2 order book data. For an MTech capstone, the NSE Bhavcopy + tick data from Zerodha's historical API or Kite Connect is a practical starting point — 15 Nifty-50 stocks, 60 trading days covers roughly 4M LOB events per stock.

**Stock selection:** Mirror Briola 2024's approach — select stocks spanning small-tick (HDFCBANK), medium-tick (INFY), and large-tick (TATAMOTORS) categories by the NSE tick size schedule. This replicates the tick-size analysis natively and is a direct contribution to the literature.

**Implementation:** Mamba has a PyTorch reference implementation (`pip install mamba-ssm`). The LOBFrame repository (UCL Financial Computing) provides the data pipeline and evaluation harness. Using LOBCAST as the evaluation framework ensures comparable metrics (F1, p_T) with the rest of the literature.

**Compute:** Mamba is GPU-efficient — a single A100 trains DeepLOB in under 2 hours. Mamba-LOB at the same scale should be comparable. Available via Google Colab Pro+ or Kaggle (2× A100 tier).

---

## Evaluation Metrics

Use the three-metric stack from the most recent benchmark papers — do not rely on F1 alone:

| Metric | Source | What It Measures |
|---|---|---|
| Weighted F1 | Standard | Predictive accuracy, class-balanced |
| p_T | Briola 2024 | Probability of profitable round-trip after NSE STT + brokerage |
| MPRF | ICLR 2025 | Continuous mid-price return forecasting — more practically meaningful |

The p_T metric is especially important for NSE: the Securities Transaction Tax (0.1% on delivery, 0.025% on intraday) means a model that achieves 60% F1 but generates low-magnitude signals may have p_T < 0.5 (i.e., expected to lose money net of costs). Reporting p_T directly addresses the simulation-to-reality gap that the Zaznov 2022 survey identified as the field's core weakness.

---

## Contribution Statement

> We present Mamba-LOB, the first application of Selective State Space Models to Limit Order Book mid-price direction prediction, evaluated on NSE India tick data — the first LOB prediction study on any Indian exchange. On 15 Nifty-50 stocks spanning three tick-size regimes, Mamba-LOB achieves competitive weighted F1 and p_T, outperforming TLOB (2025 SOTA) across tick-size categories. We further show that the selective memory mechanism disproportionately benefits large-tick stocks, where informative events are sparse and the selective forgetting property of S6 aligns with the underlying microstructure.

---

## Experiment Matrix

The following ablations generate a clean results table:

| Experiment | Variable | Purpose |
|---|---|---|
| Input representation | LOB only vs LOB+OF (70 features) | Replicates Kolm 2021 / TrioFlow finding on NSE |
| Context window | L = 100 vs 200 vs 500 | Tests O(L) advantage of Mamba over attention |
| Tick-size regime | Small / medium / large tick stocks | Replicates Briola 2024 tick analysis on NSE |
| Spatial branch | With vs without spatial SA head | Ablates the TLOB-borrowed component |
| Normalization | z-score vs tick-size-normalized (Briola 2024) | Tests normalization choice on Indian stocks |
| Horizon | H ∈ {10, 50, 100} LOB events | Standard three-horizon evaluation |

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Data acquisition + preprocessing | 3 weeks | NSE tick data pipeline; LOBFrame integration |
| Baseline reproduction | 2 weeks | DeepLOB + TLOB + BINCTABL on NSE data |
| Mamba-LOB implementation | 3 weeks | Core model, training loop, evaluation harness |
| Ablations + tick-size analysis | 2 weeks | Full experiment matrix |
| Writing | 4 weeks | Paper draft + capstone report |

The proposal is already written and the baselines are well-understood from the literature survey — the implementation phase is the differentiating work.

---

## Key References

| Paper | Relevance |
|---|---|
| Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Core architecture |
| Berti & Kasneci 2025 — TLOB + MLPLOB (arXiv 2502.15757) | Current SOTA baseline; BiN normalization |
| Briola et al. 2024 — Deep Limit Order Book Forecasting (arXiv 2403.09267) | Main paper; p_T metric; tick-size analysis |
| Prata et al. 2023 — LOB Benchmark + LOBCAST (arXiv 2308.01915) | Evaluation framework; BINCTABL baseline |
| Nagy et al. 2023 — LOBS5 (arXiv 2309.00638) | Proof that SSMs work for LOB data |
| Kolm et al. 2021 — Deep Order Flow Imbalance (SSRN 3900141) | Motivation for LOB+OF input representation |
| Zaznov et al. 2024 — TrioFlow (Applied Sciences) | LOB+OF on non-US data; trading simulation |
| Zhang et al. 2019 — DeepLOB (arXiv 1808.03668) | Universal baseline |

---

## Summary

Mamba-LOB on NSE India is the strongest capstone choice because it sits at the intersection of the field's most pressing open question (Indian market generalization, Gap 1 in the survey) and the most architecturally motivated recent advance (Mamba's selective state space model). Either contribution alone would be publishable at a workshop venue; together they constitute a solid conference paper at ICAIF, FinNLP at EMNLP, or a similar venue.
