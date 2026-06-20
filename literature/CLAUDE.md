# Capstone Project: LOB-Based Index Direction Prediction

## Project Overview
MTech (DSAI) capstone project at PES University.
**Topic:** Short-term index direction prediction using Limit Order Book (LOB) data in Indian Stock Markets.
**Goal:** Literature survey + experimental implementation.

## Team Context
- User: Joseph Raj — Data Engineering Lead, strong analytics/DE background, solid DSAI fundamentals
- Team: Group 5 — multi-member; literature survey split across team members
- Focus: Indian markets (NSE/BSE), contrast with global LOB literature

## Current Phase
Literature survey complete. Moving to implementation planning.

## Key Documents
| File | Purpose |
|---|---|
| `literature_survey.md` | Comprehensive academic literature survey (~5,000 words, all 19 papers) |
| `literature_survey_tracking.csv` | Full paper tracking: 27 papers, inclusion status, learning order, notes file |
| `learning_order_guide.md` | 7-stage recommended reading sequence with dependency map and reading plans |
| `project_recommendation.md` | Architecture recommendation: Mamba-LOB on NSE India |
| `capstone_project_proposal.md` | Broader project plan: 4-phase structure, baselines, evaluation, team division |

## Chosen Project Direction
**Mamba-LOB — Selective State Space Models for Mid-Price Direction Prediction on NSE India**

Core novelty: first application of Mamba (Gu & Dao 2023, selective SSM, O(L) complexity) to LOB prediction, evaluated on NSE India — the first LOB study on any Indian exchange.

Four novelty dimensions:
1. New architecture (Mamba on LOB = no prior work; LOBS5 used S5 for generative LOB, not discriminative prediction)
2. New market (NSE India = Gap 1 in survey, no prior work)
3. Tick-size analysis (NSE variable tick rules, mirrors Briola 2024 methodology)
4. Asset transferability (cross-sector NSE generalization)

**Baselines:** DeepLOB, TLOB (current SOTA), BINCTABL (most generalizable), TrioFlow (LOB+OF)

**Evaluation metrics:** Weighted F1, p_T (Briola 2024), MPRF (ICLR 2025)

## Papers Reviewed (with learning docs)

### Foundational Papers (Round 1)
- `paper_notes_briola_2024.md` — Main paper (Briola et al. 2024, arXiv 2403.09267). deepVOL + deepOF + seq2seq; microstructure priors; p_T metric; tick-size dependent normalization.
- `paper_notes_zhang_2019.md` — DeepLOB architecture (Zhang 2019). CNN+Inception+LSTM; 100×40 input; 5-day rolling z-score; universal baseline.
- `paper_notes_ntakaris_2018.md` — FI-2010 dataset (Ntakaris 2018). 5 Finnish stocks, NASDAQ Nordic, 10 days June 2010; benchmark limitations.
- `paper_notes_lucchese_2022.md` — Short-term predictability (Lucchese 2022). deepOF/deepVOL; seq2seq multi-horizon; MCS framework.
- `paper_notes_briola_2020.md` — Comparative study (Briola 2020). MLP ≈ CNN-LSTM on FI-2010; Bayesian t-test; FI-2010 too easy.
- `paper_notes_kolm_2021.md` — Deep OFI (Kolm 2021). OF >> LOB; Log(Updates/PriceChg) explains 75% of cross-sectional predictability.

### Recent SOTA Papers (Round 2)
- `paper_notes_tlob_2025.md` — TLOB + MLPLOB (Berti & Kasneci 2025, arXiv 2502.15757). Current SOTA; dual attention (temporal + spatial); Bilinear Normalization; predictability declining in NASDAQ.
- `paper_notes_axiallob_2022.md` — Axial-LOB (Kisiel & Gorse 2022). Factored 2D attention; 9,615 params; feature-order invariant.
- `paper_notes_lob_benchmark_2023.md` — LOB Benchmark + LOBCAST (Prata et al. 2023). FI-2010 generalizability crisis; BINCTABL best generalizable model; LOBCAST evaluation framework.
- `paper_notes_hlob_2024.md` — HLOB (Briola, Bartolucci, Aste 2024, arXiv 2405.18938). TMFG + HCNN; non-consecutive LOB level dependencies; tick-size dependent structure.
- `paper_notes_lobs5_2023.md` — LOBS5 (Nagy et al. 2023, arXiv 2309.00638). First autoregressive generative model for LOB messages; S5 SSM; world model for RL.

### Additional Papers (Round 3 — from team CSV review)
- `paper_notes_lob_survey_2022.md` — LOB Survey (Zaznov et al. 2022, Mathematics MDPI). First systematic survey; simulation-to-reality gap; FI-2010 critique; taxonomy.
- `paper_notes_dla_2023.md` — DLA (Guo & Chen 2023). Dual-stage temporal attention (input + hidden); 2nd best generalizable in LOBCAST; direct precursor to TLOB.
- `paper_notes_iclr_benchmark_2025.md` — ICLR 2025 LOB Benchmark (anon). LOB-specific vs general TS models; CVML +244.9% on MPRF; asset transferability; MPRF metric.
- `paper_notes_gp_lob_2024.md` — GP Models for LOB (Liu 2024, MSc thesis Tampere U). GPR best for long-horizon regression (+25-32% over TABL); calibrated uncertainty; combined linear+RBF kernel.
- `paper_notes_imaging_lob_2023.md` — Imaging LOB (Ye et al. 2023). LOB as 2D image; CNN image classifier; handles price-scale heterogeneity across stocks.
- `paper_notes_deep_hybrid_lob_2022.md` — Deep Hybrid LOB (Nguyen et al. 2022, FDSE). ResNet50+LSTM; ImageNet transfer learning; second-level aggregation; no FI-2010 comparison.
- `paper_notes_trioflow_2024.md` — TrioFlow CNN+GRU (Zaznov et al. 2024, Applied Sciences). LOB+OF joint input (70 features); 3-branch inception-style CNN + GRU; MICEX Russian dataset; beats DeepLOB.
- `paper_notes_bayesian_bilinear_2023.md` — Bayesian TABL (Magris et al. 2023, J. Forecasting). VOGN Bayesian inference on TABL; calibrated uncertainty; outperforms MC Dropout; threshold-based trading decisions.

## Papers NOT Included (from CSV review)
15 papers excluded. Categories:
- **Different task:** fill probability, manipulation detection, pairs trading, continuous-time modeling, RL trading, broker identities
- **Generic stock prediction:** OHLCV-based, HMM, stacked LSTM tutorials — no LOB data
- **Different methodology:** fuzzy numbers, stochastic process modeling
- **Errors/Duplicates:** 3 entries (#12/#19 same URL + year 2026 placeholders; #27 medical cholesterol paper)

## Key Research Gaps (from literature survey)
1. **Indian market LOB prediction** — no published study on NSE/BSE (the capstone's primary contribution)
2. LOB+OF joint input on non-US, non-European markets
3. Tick-size analysis outside NASDAQ/LOBSTER datasets
4. Uncertainty quantification under regulatory constraints (circuit breakers, price bands)
5. Foundation models (TimesFM, Chronos, Lag-Llama) not yet tested on LOB data

## Indian Market Adaptation Notes
- NSE/BSE tick size varies by price band (₹0.05 to ₹1.00), unlike fixed $0.01 on NASDAQ
- Circuit breakers (5%/10%/20% price bands) create discontinuous dynamics
- STT (Securities Transaction Tax): 0.025% intraday, 0.1% delivery — makes p_T metric critical
- Trading hours: 9:15 am – 3:30 pm IST
- Higher retail participation vs. NASDAQ; different liquidity profiles
- Data sources: NSE MOAD feed, Refinitiv, Zerodha Kite Connect (for academic proof-of-concept)

## Core Concepts
- LOB = Limit Order Book: electronic queue of all unmatched buy/sell orders
- Mid-price = (best_ask + best_bid) / 2
- Tick size (θ) = smallest price increment; stock classification: small/medium/large-tick
- Prediction horizons: H ∈ {10, 50, 100} LOB update events (tick-time, not clock-time)
- DeepLOB input: 100×40 tensor (100 consecutive LOB snapshots × 40 features)
- Order Flow Imbalance (OFI): derived from consecutive LOB changes; stationary; more predictive than raw prices
- p_T metric: probability of profitable round-trip after transaction costs (Briola 2024)
- MPRF: Mid-Price Return Forecasting — continuous regression version of the direction task (ICLR 2025)

## Repo / Tools
- LOBFrame (open-source): https://github.com/FinancialComputingUCL/LOBFrame
- LOBCAST (benchmark framework): https://github.com/matteoprata/LOBCAST
- Mamba (reference implementation): `pip install mamba-ssm`
- Data source in papers: LOBSTER (NASDAQ tick-by-tick), FI-2010, MICEX (Kaggle)
