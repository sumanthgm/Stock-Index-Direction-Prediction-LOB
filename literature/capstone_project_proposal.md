# Capstone Project Proposal

**Proposed title:** LOB-Based Mid-Price Direction Forecasting for NSE Stocks: Comparing Input Representations and Architectures with Microstructural Analysis

**Team context:** MTech DSAI, PES University, Sem 3 capstone

---

## What the Literature Establishes (The Setup)

Ten papers collectively tell a coherent story about LOB forecasting on NASDAQ stocks:

**Foundational findings:**
1. **DeepLOB (Zhang 2019):** CNN+LSTM with 5-day rolling normalization is the reference architecture
2. **FI-2010 (Ntakaris 2018):** The benchmark dataset has serious normalization flaws — don't trust its numbers
3. **Kolm 2021:** Stationary inputs (order flow) dramatically outperform non-stationary inputs (raw LOB prices); large-tick stocks are more predictable; Log(Updates/PriceChg) explains 75% of cross-sectional predictability
4. **Lucchese 2022:** L2 + order flow or volume representation is consistently best; seq2seq multi-horizon is better than single-horizon; universal models work
5. **Briola 2024:** Large-tick classification drives predictability; p_T metric reveals that high ML accuracy ≠ actionable trading signal

**Recent SOTA findings:**
6. **LOB Benchmark (Prata 2023):** FI-2010 is overfit — models achieving 80%+ F1 there fail to 48-61% on new NASDAQ data. BINCTABL is the most generalizable model. LOBCAST provides an open-source framework with all 15 benchmarked models.
7. **Axial-LOB (Kisiel 2022):** Factored 2D attention achieves global receptive field with only 9,615 parameters; feature-order invariant; SOTA at publication.
8. **TLOB + MLPLOB (Berti 2025):** Current SOTA. Dual attention (temporal + spatial) + Bilinear Normalization. Predictability has declined 6.68 F1 points over 3 years for NASDAQ — Indian markets may be more predictable due to lower algorithmic trading.
9. **HLOB (Briola 2024):** TMFG + HCNN captures non-consecutive LOB volume level dependencies; best at H10 universally; effectiveness tied to tick-size class. Large-tick stocks have rich spatial MI structure.
10. **LOBS5 (Nagy 2023):** First autoregressive generative model for LOB messages; LOB as a "world model" for RL; return forecasting emerges as byproduct of generation.

**The gap:** All of this work is on NASDAQ (US) or Finnish stocks. None of it has been replicated on the NSE (Indian) market. The Indian market differs in tick size structure, liquidity, price band circuit breakers, trading session times, and participant composition. Whether these findings transfer is an open empirical question.

---

## Proposed Research Objective

**Test whether the microstructure-driven predictability findings from NASDAQ transfer to NSE Nifty-50 stocks, and identify which LOB input representation and model architecture works best in the Indian market context.**

Specifically:
- Do NSE large-tick stocks (tightly-spread, high-liquidity stocks like banking sector) show significantly higher LOB predictability than small-tick stocks?
- Does Log(Updates/PriceChg) explain cross-sectional predictability variation in NSE?
- Does order flow representation outperform raw LOB prices on NSE data?
- Which architecture performs best on NSE: CNN-LSTM, LSTM, MLP, or Transformer?

---

## Recommended Project Structure

### Phase 1: Data Acquisition and Pipeline (3-4 weeks)

**What to get:**
- NSE L2 order book data for 10-15 Nifty-50 stocks
- Aim for at least 3-6 months of historical LOB data
- Cover both large-tick (HDFC, SBI, ICICI, RELIANCE, TCS) and small-tick stocks

**Data sources to investigate:**
- NSE's own data products (NSE Data Feed / Historical Data)
- Third-party vendors: Tickertape, True Data, QuantInsti (for academic access)
- Ask your faculty supervisor — academic institutions sometimes have data agreements

**Pipeline to build (modeled after LOBFrame):**
1. Load raw NSE LOB snapshots (message file + order book file)
2. Clean: remove first/last 10 min of trading day, remove crossed quotes, collapse same-nanosecond events
3. Apply 5-day rolling z-score normalization per feature
4. Compute order flow features from consecutive snapshots
5. Compute deepVOL volume representation
6. Generate labels (quantile-based, balanced 3 classes)
7. Split: per stock, rolling window (4 weeks train, 1 week test)

**Key decisions:**
- NSE tick size is price-band dependent (varies from ₹0.05 to ₹1.00 based on price range). You need to compute per-stock tick size and classify stocks accordingly.
- NSE circuit breakers (5%, 10%, 20% price bands) — events near circuit breaker limits should be handled carefully or excluded.
- Trading hours: 9:15 am – 3:30 pm IST (vs. 9:30 am – 4:00 pm ET for NASDAQ)

---

### Phase 2: Baseline Replication (2-3 weeks)

**Implement the known-good architectures from the literature:**

| Model | Input | Notes |
|---|---|---|
| Logistic Regression | OF | Strong simple baseline (Briola 2020) |
| MLPLOB | LOB/OF | Competitive with TLOB at short horizons; new simple baseline (Berti 2025) |
| DeepLOB CNN-LSTM | LOB (raw) | Reference architecture (Zhang 2019) |
| BINCTABL | LOB | Most generalizable in benchmark study (Prata 2023); use as strong baseline |
| deepOF CNN-LSTM | OF | Best single-stock model (Lucchese 2022) |

Use the **LOBFrame** codebase (https://github.com/FinancialComputingUCL/LOBFrame) as your starting point — it implements HLOB, DeepLOB, BinCTabl, and the full Briola 2024 pipeline in PyTorch. The **LOBCAST** framework (https://github.com/matteoprata/LOBCAST) has all 15 benchmark models (BINCTABL, DLA, etc.) in PyTorch Lightning.

**Evaluation:**
- Primary metric: MCC (Matthews Correlation Coefficient) — same as literature
- Also compute F1 (weighted) and accuracy for comparability
- Use rolling window CV, not global train/test split
- Report results separately per stock AND aggregated by tick size class

---

### Phase 3: Architecture Extension — Transformer for LOB (2-3 weeks)

This is the "recent AI/ML technology" component your faculty is looking for.

**The Transformer for LOB Forecasting:**

The Briola 2020 paper showed that Self-Attention LSTM (partial transformer) works at short horizons. A full Transformer encoder (with multi-head attention, no LSTM) has not been systematically compared on the same benchmark as deepOF. This is a genuine gap.

**Proposed architecture:** TLOB adaptation (the 2025 current SOTA)

TLOB (Berti & Kasneci 2025) is now the established SOTA and directly applicable. Rather than inventing a new architecture, implement TLOB and demonstrate it on NSE data — the architectural novelty is already established; your contribution is the NSE application and analysis.

```
Input: 100 LOB snapshots × 40 features
   ↓
Bilinear Normalization (batch-adaptive — handles NSE distribution shifts)
   ↓
TLOB Block (repeated N=4 times):
  ├── Temporal Self-Attention (100 time steps attend to each other)
  ├── Spatial Self-Attention (40 features attend to each other)
  └── MLPLOB block (Feature-Mixing MLP + Temporal-Mixing MLP)
   ↓
Dense (3 outputs) → Softmax (Up/Down/Stable)
```

**Why TLOB is the right choice for the "recent AI/ML" requirement:**

- Dual attention (temporal + spatial) is architecturally motivated: LOB varies along both time and feature axes
- Bilinear Normalization handles distribution shift — important for NSE where market regimes differ from NASDAQ
- New SOTA on FI-2010, TSLA/INTC, and BTC (2025) — clearly "recent AI/ML"
- Ablation study shows both attention types are necessary (removes uncertainty about whether it's just one mechanism)
- TLOB converges faster than BINCTABL and DeepLOB — practical advantage for capstone compute

**Why testing TLOB on NSE is a genuine open question:**

The BERTI 2025 paper only tests on NASDAQ and Binance data. NSE has different microstructure (varying tick sizes, circuit breakers, FII participation). Whether TLOB's dual attention generalizes to Indian market dynamics is an untested empirical question.

**Simpler alternative if TLOB is too complex:** MLPLOB — same paper, simpler architecture (pure MLP, no attention), competitive at short horizons (H10, H20), much easier to implement and debug.

---

### Phase 4: Analysis and Evaluation (2 weeks)

**Microstructural Analysis (replicating Kolm 2021 on NSE):**

For each of your 10-15 NSE stocks, compute:
- Tick Size metric (fraction of time spread = minimum tick)
- Log(Updates/PriceChg)
- Log(Updates per day)
- Log(Trades per day)

Correlate these with your best model's MCC. Reproduce the scatter plots from Kolm Figure 5-6 for NSE data.

**Hypothesis to test:** Do the NSE results show the same pattern — large-tick stocks significantly more predictable, Log(Updates/PriceChg) explaining most cross-sectional variance?

**Statistical comparison:**
- Use Model Confidence Sets (Lucchese 2022) for rigorous model comparison across multiple test windows
- Do not just report the model with highest average MCC — report which models are statistically indistinguishable from the best

**Trading utility evaluation (optional but strong):**
- Implement the p_T metric from Briola 2024
- This bridges the gap between "ML accuracy" and "actionable signal"
- Show whether your best model on NSE can actually execute profitable trades under realistic conditions

---

## The Core Contribution (What Makes This Novel)

This project is a **structured replication + extension study** on a new market. That is a legitimate and publishable contribution.

Specifically, the novel contributions are:

1. **First systematic LOB prediction study on NSE data** — establishing baseline predictability in Indian markets
2. **Cross-sectional microstructural analysis for NSE** — does Log(Updates/PriceChg) explain NSE predictability the same way it does for NASDAQ?
3. **Systematic comparison of input representations on NSE** — OF vs. raw LOB, with rigorous statistical comparison (MCS)
4. **Transformer for LOB** — comparing full Transformer vs. LSTM and CNN-LSTM on the same benchmark, with proper statistical inference

---

## Architecture Summary: What to Build

```
Data Pipeline:
  NSE LOB → Clean → Normalize (rolling z-score) → OF features + deepVOL features

Models (in order of complexity):
  1. Logistic Regression baseline
  2. MLP baseline  
  3. DeepLOB (CNN-LSTM, raw input) — reference
  4. deepOF-LSTM — expected best from literature
  5. LOBformer (Transformer, OF input) — novel contribution

Evaluation:
  - MCC, F1 per stock and tick-size class
  - MCS for statistical comparison
  - Microstructural correlation analysis
  - (Optional) p_T trading utility metric

Output:
  - Learning docs for all papers (done)
  - Clean NSE LOB pipeline + dataset description
  - Trained model weights for 5 architectures across 10-15 stocks
  - Results tables showing whether NASDAQ findings transfer to NSE
  - Scatter plot: Log(Updates/PriceChg) vs. MCC for NSE stocks
```

---

## Answering the Faculty's "Recent AI/ML" Requirement

The Transformer architecture (Phase 3) directly addresses this:

- Transformers are the dominant architecture in NLP (GPT, BERT), vision (ViT), and increasingly in time-series forecasting (PatchTST, TimesNet, iTransformer — all 2022-2024)
- Applying Transformers to LOB data is an active research direction — but the specific comparison against the deepOF baseline with order flow inputs has not been done cleanly in the literature
- Multi-head attention has a natural interpretation for LOB data: different heads might learn to attend to bid-side imbalance patterns, ask-side depletion patterns, and longer-range trend patterns simultaneously
- Self-supervised pre-training of a Transformer on LOB sequences is a genuinely novel extension (though it may be out of scope for one semester)

**If you want to go further:** Fine-tuning a pretrained financial LLM (like FinBERT or similar) is probably not directly applicable to LOB data (it's structured numerical data, not text). But training a BERT-style masked prediction model on LOB sequences as pre-training, then fine-tuning for direction prediction, would be academically interesting and novel.

---

## Realistic Scope for One Semester

**Minimum viable project (guaranteed achievable):**
- 5-10 NSE stocks, 3 months of LOB data
- DeepLOB (reference), deepOF-LSTM (best baseline), Transformer (novel)
- MCC comparison + microstructural correlation analysis
- Clear replication or refutation of the large-tick predictability finding

**Stretch goals:**
- 15+ stocks, 6-12 months of data
- Full MCS statistical framework
- p_T trading utility metric
- Universal model (single model trained on all stocks)
- Multi-horizon seq2seq Transformer

**What to scope out:**
- Live trading implementation — out of scope, and requires regulatory/infrastructure complexity
- Building HFT infrastructure for H10 horizons — focus on H50/H100 where latency is manageable
- Complete novel architecture from scratch — build on DeepLOB / LOBFrame codebase

---

## Suggested Team Division (if applicable)

| Member | Focus |
|---|---|
| Data & Pipeline | NSE data acquisition, cleaning, normalization pipeline |
| Baselines | Implement and tune MLP, LSTM, deepOF-LSTM |
| Architecture | Implement and tune Transformer (LOBformer) |
| Analysis | MCS comparison, microstructural correlation, p_T metric |

---

## Key References for Each Component

| Component | Primary reference |
|---|---|
| Data pipeline | Briola 2024 + LOBFrame codebase |
| deepOF-LSTM baseline | Kolm 2021, Lucchese 2022 |
| DeepLOB reference | Zhang 2019 |
| MCS evaluation | Lucchese 2022 (Appendix A.2) |
| Microstructural analysis | Kolm 2021 (Sections 4-5) |
| Transformer architecture | "Attention is All You Need" (Vaswani 2017) + PatchTST (2023) |
| p_T metric | Briola 2024 (Section 5) |
