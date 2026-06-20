# Literature Survey: Deep Learning for Mid-Price Direction Prediction in Limit Order Book Markets

**Capstone Project — MTech DSAI, PES University (Group 5)**  
**Focus: Indian Stock Market (NSE/BSE) Context**

---

## Abstract

This survey examines the state of the art in machine learning-based mid-price direction prediction using Limit Order Book (LOB) data. We systematically review 19 papers spanning 2018–2025, covering foundational architectures, benchmark datasets, evaluation methodology, and the most recent SOTA models. We trace the evolution from simple CNN+LSTM baselines to dual-attention transformers, graph-structured CNNs, and probabilistic Bayesian models. A central finding is that the field faces a reproducibility and generalizability crisis — models achieving >80% F1 on the canonical FI-2010 benchmark show substantial degradation on out-of-sample data. We identify five open research gaps, the most significant being the complete absence of LOB prediction studies on Indian equity markets (NSE/BSE), which motivates this capstone project.

---

## 1. Introduction

Modern financial markets operate through the Limit Order Book (LOB): an electronic queue that records all unexecuted buy and sell orders at different price levels. The LOB provides a real-time, high-resolution picture of supply and demand dynamics, updated continuously as traders submit, cancel, and execute orders. Predicting short-term mid-price movements — the average of the best bid and ask prices — from LOB data has become one of the core problems in quantitative finance and machine learning.

The LOB prediction problem is formally stated as: given the current state of the LOB (prices and volumes at the top L levels on both bid and ask sides), predict whether the mid-price will increase, decrease, or remain stable over the next H LOB updates. This is cast as a three-class classification problem with classes {Up, Down, Stationary}, evaluated at prediction horizons H ∈ {10, 50, 100} updates in tick-time.

Why LOB data rather than OHLCV (open/high/low/close/volume) time series? The LOB contains information about the full depth of market liquidity — not just where trades happened, but where traders are willing to trade. This depth is known to contain predictive signal about future price movements [Kolm et al. 2021; Lucchese et al. 2022]. As Zaznov et al. (2022) note, LOB-based models consistently outperform OHLCV-based models when correctly evaluated.

The field has progressed through several distinct phases: (1) traditional statistical and classical ML approaches (2010–2017), (2) deep learning baselines anchored by the FI-2010 benchmark (2017–2021), (3) architectural innovation and benchmark critique (2022–2024), and (4) emerging directions in uncertainty quantification, generative models, and cross-market generalization (2023–2025). This survey covers phases 2–4 in depth and phase 1 selectively.

The Indian market context is of particular relevance. The National Stock Exchange (NSE) is the world's largest derivatives exchange by volume and operates a pure LOB trading mechanism. Yet no published study applies LOB-specific DL models to NSE data. This gap is the central motivation for our capstone project.

---

## 2. The LOB as a Predictive Signal

### 2.1 Structure and Notation

A LOB snapshot at time t is a tensor X(t) ∈ ℝ^(2L×2) containing the prices P^a_l(t), P^b_l(t) and volumes V^a_l(t), V^b_l(t) at levels l = 1,...,L on the ask and bid sides respectively. The standard input for deep learning models follows the FI-2010 convention: L=10 levels, giving 40 features per snapshot, with T=100 consecutive snapshots forming the input window X ∈ ℝ^(100×40).

The mid-price at time t is:
```
m(t) = (P^a_1(t) + P^b_1(t)) / 2
```

The prediction target at horizon H is the smoothed label:
```
l_H(t) = sign(m̄(t+H) − m̄(t))   where m̄(t) = (1/H)Σ_{i=1}^{H} m(t+i)
```

A threshold α is applied to create the Stationary class: if |m̄(t+H) − m̄(t)| / m̄(t) < α, label is Stationary.

### 2.2 Order Flow as an Alternative Representation

Kolm, Turiel, and Westray (2021) demonstrate that Order Flow Imbalance (OFI) — derived from changes in the LOB rather than its static state — is a more stationary and predictive signal than raw LOB snapshots. OFI captures the net pressure of orders entering and exiting the book between consecutive snapshots. Their key empirical finding is that `log(LOB updates / price changes)` — a measure of market activity per price tick — explains a significant portion of LOB predictability. Stocks with higher this ratio (more active but less volatile) are more predictable. This concept of "information-rich" stocks has influenced subsequent architecture design.

Lucchese, Pakkanen, and Veraart (2022) extend this by introducing deepOF and deepVOL as competing input representations: deepOF encodes order flow imbalance across price levels, while deepVOL encodes the raw volume distribution at each level. Their seq2seq multi-horizon architecture produces simultaneous predictions for all horizons rather than independently training for each. The Multiple Comparison Sequences (MCS) statistical framework is applied to rigorously compare models — a methodological practice absent from most prior LOB work.

### 2.3 Predictability and Market Microstructure

Briola, Bartolucci, and Aste (2024) [arXiv 2403.09267] establish the link between microstructure and predictability most explicitly. They introduce the practical metric p_T: the probability that a model-informed round-trip transaction (buy on predicted Up, sell on next trade) yields positive returns after transaction costs. Unlike F1 score, p_T captures economic value directly. Their key finding: predictability is stock-specific, tick-size-dependent, and horizon-dependent. The LOB's structural properties — measured by the mean number of active price levels Ξ^{Bid,Ask} — determine how much predictive signal is available.

---

## 3. The FI-2010 Benchmark Era (2018–2022)

### 3.1 The FI-2010 Dataset

Ntakaris et al. (2018) established the canonical LOB prediction benchmark: the FI-2010 dataset containing LOB data for five Finnish stocks traded on NASDAQ Nordic Helsinki exchange over ten trading days (June 1–14, 2010). The dataset includes pre-constructed LOB snapshots with 144 handcrafted features normalized using three methods (z-score, min-max, decimal precision) and labels at five prediction horizons (k=1, 2, 3, 5, 10 events). The "Basic Set" of 40 features (prices and volumes at L=10 levels) became the standard input format.

FI-2010's contributions were substantial: it standardized the input format, normalization, and evaluation protocol for the field. However, its limitations — identified explicitly by Ntakaris et al. themselves and later confirmed empirically — are equally important:

- **Scale:** 5 stocks, 10 trading days represents a tiny fraction of market microstructure diversity
- **Geography:** Finnish large-caps on a Nordic exchange differ substantially from US NASDAQ stocks and dramatically from emerging market stocks (India, Russia, Brazil)
- **Age:** Pre-2010 market microstructure differs from modern HFT-dominated LOBs
- **Pre-processing:** The original ITCH data cannot be fully recovered from the published dataset, preventing replication of the preprocessing pipeline

### 3.2 DeepLOB: The Baseline Architecture

Zhang et al. (2019) introduced DeepLOB, which became the primary benchmark model for the next six years. The architecture consists of: (1) convolutional layers that extract spatial features across the 40-feature LOB representation, treating the 10-level depth as a spatial dimension; (2) Inception modules that capture multi-scale temporal patterns; (3) an LSTM layer that models temporal dependencies across the 100-snapshot input window; (4) a three-class softmax output.

DeepLOB achieves over 80% accuracy and weighted F1 on FI-2010, with the best results at H=10 (shortest prediction horizon). The model's success established two important design principles that subsequent architectures built on: (a) the LOB should be processed as a 2D spatial structure (features × levels), not a 1D vector; and (b) temporal modeling via LSTM is necessary beyond pure convolutional processing.

However, Briola et al. (2020) show that on FI-2010, a simple four-layer MLP with ~1.8×10^5 parameters performs comparably to DeepLOB. Their Bayesian t-test comparison across five stocks and three horizons finds no statistically significant difference between MLP and CNN-LSTM models — a finding that reveals FI-2010 is insufficiently challenging to discriminate between architectures.

### 3.3 The Dual-Stage Attention Direction

Guo and Chen (2023) introduced DLA (Dual-Stage Attention LSTM) — two temporal attention stages operating at different points in the LSTM processing pipeline:

- **Stage 1 (Input Attention):** Before the LSTM encoder, attention weights across the 40 input features based on the current hidden state, selectively amplifying informative LOB features
- **Stage 2 (Hidden State Attention):** After encoding, attention weights across the 100 hidden states to construct a context vector emphasizing the most informative time steps

DLA ranks second in generalizability (76.9%) in the LOBCAST benchmark (Prata et al. 2023). Its dual-stage structure directly anticipates TLOB, which can be understood as extending DLA's temporal attention principle to also include a spatial (feature-axis) dimension.

---

## 4. The Benchmark Critique and Generalizability Crisis (2022–2024)

### 4.1 LOBCAST: Empirical Evidence of FI-2010 Overfit

Prata et al. (2023) conducted the most comprehensive empirical benchmark of LOB models: the LOBCAST study. Fifteen models were trained on FI-2010 and evaluated on two criteria: (1) absolute performance on FI-2010, and (2) generalizability — performance on fresh LOBSTER data covering the same stocks in 2021–2022 (LOB-2021/2022 dataset).

The central finding is stark: a model's performance on FI-2010 is a poor predictor of its real-world generalizability. The model with the highest FI-2010 F1 score (TransLOB) shows one of the largest generalizability gaps. Conversely, BINCTABL — a bilinear network with channel attention — achieves only moderate FI-2010 performance but the highest generalizability score (73.5%). Performance degradation across models ranges from 8% to 19.6% in F1.

This finding has major methodological implications: the field has been optimizing for FI-2010 overfitting rather than real predictive capability. The LOBCAST framework — an open-source Python package implementing all 15 models with consistent preprocessing — addresses this by enabling standardized cross-dataset evaluation.

### 4.2 The ICLR 2025 Extension

The ICLR 2025 benchmark (Anonymous, 2025) extends this critique in two directions. First, it tests whether general time-series models from the broader ML community (PatchTST, iTransformer, TimesNet, DLinear) can match LOB-specific models. The answer is no — without architectural adaptation, SOTA time-series models significantly underperform LOB-specific architectures. Second, the paper introduces Mid-Price Return Forecasting (MPRF): predicting the continuous magnitude of the price change, not just its sign.

The paper's architectural contribution is CVML (Convolutional Cross-Variate Mixing Layers): a plug-in module that adds cross-feature mixing to any time-series model. Adding CVML to generic models yields a +244.9% average improvement in MPRF — identifying cross-feature mixing (the spatial dimension of LOB data) as the critical inductive bias that generic models lack.

A third dimension is asset transferability: training on one asset and testing on a completely different one. This is harder than the LOBCAST generalizability test (same stocks, different time) and remains an open challenge for all models.

### 4.3 The Evaluation Gap: From F1 to Economic Metrics

The Zaznov et al. (2022) survey identified what they term the "simulation-to-reality gap" — a systematic divergence between reported benchmark metrics and actual trading utility. Three layers of unrealism compound in standard LOB evaluation:

1. **Mid-price execution assumption:** All models assume trades execute at the mid-price. Real execution at the bid or ask imposes half-spread cost per trade — fatal for frequently trading models
2. **Transaction costs ignored:** Securities Transaction Tax (STT), brokerage, and market impact are universally omitted. For Indian markets, STT alone (0.025% for intraday equity) significantly reduces model-claimed returns
3. **No trading simulation:** Accuracy and F1 do not measure P&L. Models that are "right" about small moves while being "wrong" about large moves may be useless or harmful in practice

Briola et al. (2024) directly address this with the p_T metric. The ICLR 2025 paper addresses it with MPRF. Zaznov et al.'s own follow-up work (TrioFlow, 2024) validates predictions through simulated trading experiments on MICEX data — a practice rare in the field but necessary for credibility.

---

## 5. Architectural Evolution

### 5.1 The CNN+LSTM Baseline

DeepLOB (Zhang 2019) established CNN+LSTM as the default LOB architecture. The spatial CNN extracts features across the L price levels (treating them as a 1D spatial dimension), and the LSTM models temporal dependencies. Nguyen et al. (2022) extended this with ResNet50 (a pretrained ImageNet backbone) in place of the custom CNN, showing that transfer learning from image classification is feasible but architecturally unmotivated — ResNet50's spatial statistics assumptions (natural image correlations) do not correspond to LOB data structure.

### 5.2 Attention Mechanisms

The attention trajectory in LOB models has proceeded through three stages:

**Single-stage temporal attention (DEEPLOBATT, 2021):** Attention over the LSTM's 100 hidden states weights time steps by importance. A significant but partial improvement over raw LSTM.

**Dual-stage temporal attention (DLA, 2023):** Two attention stages — one over input features (pre-encoding), one over hidden states (post-encoding). Higher generalizability than single-stage attention, as validated by LOBCAST.

**Full 2D attention (Axial-LOB, TLOB, 2022–2025):** 
- Axial-LOB (Kisiel & Gorse, 2022) factorizes 2D attention into separate row-attention (time) and column-attention (features), achieving competitive results with only 9,615 parameters. Factored attention is also feature-order invariant — an useful inductive bias since LOB feature ordering (bid level 1, 2, ... ask level 1, 2, ...) is arbitrary.
- TLOB (Berti & Kasneci, 2025) applies independent self-attention blocks over both the temporal and spatial axes, combined with Bilinear Normalization (BiN). BiN simultaneously normalizes along both axes, addressing the non-stationarity of LOB features without the lookahead bias risk of rolling window normalization. TLOB is the current SOTA — it achieves the best generalizability scores on both FI-2010 and LOBSTER datasets.

### 5.3 Graph-Structured and Image-Based Approaches

HLOB (Briola, Bartolucci, Aste 2024) takes a fundamentally different approach to the spatial structure of LOB data. Rather than treating price levels as equally-spaced spatial neighbors, HLOB uses mutual information between volume levels to construct a Triangulated Maximally Filtered Graph (TMFG): a sparse, planar graph that retains only the most information-rich pairwise dependencies. The HCNN (Homological CNN) then processes graph simplices — tetrahedra, triangles, and edges — in parallel, capturing non-consecutive level dependencies that standard 1D CNN cannot.

The key empirical finding is tick-size dependent: for large-tick stocks (spread ≈ tick size, e.g., BAC, CSCO), the TMFG reveals a rich hierarchical structure with high mutual information across non-adjacent levels. For small-tick stocks (spread >> tick size, e.g., GOOG, CHTR), the MI matrix is nearly diagonal — only adjacent levels are informative, and HCNN offers little advantage over standard CNN.

Ye et al. (2023) propose a different structural approach: imaging the LOB. Each LOB snapshot window is converted to a standardized 2D image where pixel intensity encodes price and volume values through intra-snapshot normalization. This image-space normalization handles the cross-stock price-scale heterogeneity problem (a ₹2 stock and a ₹2,000 stock should produce comparable image patterns). A standard CNN image classifier applied to the resulting images outperforms raw-feature CNN baselines.

### 5.4 Bilinear Networks

The TABL family (Temporal Attention-Augmented Bilinear Layer) represents an alternative architectural philosophy. Rather than applying a single flat transformation to a 1D feature vector, the bilinear layer applies separate learned projections along each axis of the input matrix:

```
Y = φ(W₁ · X · W₂ᵀ + B)
```

where W₁ ∈ ℝ^(D'×D) projects the feature dimension and W₂ ∈ ℝ^(T'×T) projects the temporal dimension. A learned temporal attention vector further reweights time steps. This factored structure is parameter-efficient (parameter count scales linearly with D and T separately), interpretable (spatial and temporal transformations are explicit), and BINCTABL proves the most generalizable model in the LOBCAST benchmark.

Magris, Shabani, and Iosifidis (2023) extend TABL to a Bayesian neural network via VOGN (Variational Online Gauss-Newton) — a natural-gradient variational inference algorithm. The resulting Bayesian TABL produces a full predictive distribution P(Up, Down, Stable | LOB) at each timestamp, enabling confidence-threshold-based trading strategies (trade only when model assigns >70% probability to a direction). VOGN outperforms both deterministic TABL and the common MC Dropout approximation on calibration quality.

---

## 6. Probabilistic Methods

### 6.1 Gaussian Process Models

Liu (2024) provides the first application of Gaussian Process models to the full 40-dimensional LOB feature space. Prior GP finance applications used low-dimensional OHLCV data; scaling to LOB required both a principled kernel design and an ensemble approximation to address the O(N³) computational cost.

The kernel design is notable: a combined linear kernel on price features (encoding the known near-linear relationship between LOB price levels and future mid-price) and an RBF kernel on volume features (capturing nonlinear volume-price relationships). GPR with this combined kernel achieves R² = 94.47% at H=10,000 event horizon — outperforming TABL (90.44%) and Ridge Regression (93.56%) significantly. For classification, GPC is competitive at H=5 and H=10 (F1 ≈ 60%) but weak at H=1 due to class imbalance domination.

The uniquely valuable output of GP models is the 95% confidence interval on mid-price predictions — well-calibrated and practically useful for risk management. GP predictions consistently contain the true mid-price within their intervals even when the point estimate is wrong.

### 6.2 Generative Models: LOBS5

Nagy et al. (2023) take a fundamentally different approach: rather than predicting mid-price direction, they generate synthetic LOB messages autoregressively. LOBS5 uses a 12,011-token vocabulary encoding all LOB message types (order submissions, cancellations, executions), with each message tokenized into 22 tokens covering event type, direction, price, size, and timing. A two-branch S5 (Simplified Structured State Space) architecture processes message sequences and maintains an internal representation of the current book state.

LOBS5 achieves perplexity of 3.63 (GOOG) and 4.04 (INTC) — well below the uniform-distribution baseline. Return correlations between synthetic and real sequences reach ρ ≈ 0.1–0.2 at 80–100 message horizons. This generative framing opens LOB prediction to reinforcement learning: a trained LOBS5 can serve as a "world model" for RL trading agents, enabling simulation-based training without requiring live market access.

---

## 7. Multi-Modal Input: Order Flow + LOB

Zaznov et al. (2024) introduce TFF-CL-GRU (TrioFlow): the first model to systematically use LOB and Order Flow data as co-equal inputs. The 70-feature input combines 40 LOB features (standard) with 30 order flow features (price, volume, direction of the 10 most recent executed trades). Three parallel convolutional branches (inception-style, different kernel sizes) extract multi-scale spatial features, followed by a 64-unit GRU.

On both MICEX (Moscow Exchange, Russian equities) and LOBSTER (NASDAQ) datasets, TrioFlow outperforms DeepLOB by 4–5 F1 points. The MICEX dataset — the first publicly available LOB+OF dataset for a non-US market — is itself a contribution. Russian equities on MICEX share structural similarities with Indian equities on NSE: emerging market dynamics, lower liquidity than US large-caps, different participant composition.

The simulated trading validation is equally important: TrioFlow signals produce positive P&L when used for execution decisions, directly addressing the simulation-to-reality gap identified by the same team's 2022 survey.

---

## 8. Key Findings and Trends

### 8.1 Architectural Lessons

Five consistent architectural lessons emerge across the literature:

1. **The spatial dimension matters.** Treating the L price levels as a meaningful spatial structure (not a flat feature vector) consistently improves performance. CNN, HLOB's HCNN, TABL's bilinear projection, and TLOB's spatial SA all exploit this.

2. **Temporal attention improves generalizability.** Models with explicit temporal attention (DLA, TLOB) generalize better to new data than pure CNN-LSTM models. The 2nd and 1st ranked generalizable models in LOBCAST both use attention.

3. **Cross-feature mixing is the critical missing piece.** The ICLR 2025 benchmark shows that generic TS models fail precisely because they lack cross-feature mixing. CVML (+244.9%) and HLOB's HCNN both solve variants of this problem.

4. **Normalization is underappreciated.** TLOB's BiN and DeepLOB's rolling z-score are not interchangeable. Improper normalization is a major source of inter-study inconsistency and can introduce lookahead bias.

5. **Order flow adds signal beyond the static LOB.** Kolm et al. (2021) theoretically and TrioFlow (2024) empirically confirm that executed trade data complements the resting order book.

### 8.2 Evaluation Lessons

1. **FI-2010 performance does not predict generalizability.** Correlation between FI-2010 F1 and LOBCAST generalizability score across 15 models is low. Optimizing for FI-2010 is counterproductive for real-world deployment.

2. **F1 score alone is insufficient.** MPRF, p_T, and simulated trading P&L are more practically meaningful. Future work should report at least one economic metric.

3. **Asset transferability is the hardest test.** Cross-time-period generalizability (LOBCAST) is already difficult. Cross-asset generalizability (ICLR 2025) is harder. No model currently handles both well.

4. **Statistical significance testing is rarely done.** Only GP Models (Liu 2024) and Lucchese et al. (2022) use rigorous statistical tests (Kruskal-Wallis + Dunn; MCS). Most performance comparisons in the literature cannot be verified as statistically significant.

### 8.3 Trajectory

The field is moving in three simultaneous directions:
- **Toward uncertainty quantification:** From point predictions (DeepLOB) → probabilistic classification (Bayesian TABL) → continuous return regression with confidence intervals (GPR, MPRF)
- **Toward structural priors:** From generic CNN (DeepLOB) → topology-aware processing (HLOB) → kernel-encoded domain knowledge (GPR with linear+RBF split)
- **Toward broader generalization:** From single benchmark (FI-2010) → multiple exchanges (LOBCAST) → non-NASDAQ markets (TrioFlow/MICEX) → emerging markets (gap, 2025+)

---

## 9. Research Gaps

### Gap 1: Indian Market LOB Prediction (Critical Gap)
No published study applies LOB-specific deep learning to Indian equity markets. The NSE operates a pure LOB mechanism with distinct microstructure properties: variable tick-size rules (not fixed $0.01 like NASDAQ), circuit breakers and price bands (±5%/10%/20% per day), different liquidity profiles, strong retail participation, and FII activity creating intraday seasonal patterns. Whether NASDAQ-trained architectural insights transfer to NSE is entirely unknown.

### Gap 2: LOB + Order Flow on Non-US Markets
TrioFlow (2024) shows that LOB+OF joint inputs outperform LOB-only on MICEX. However, MICEX and LOBSTER cover only two market types. The benefit of adding order flow has not been validated on Asian, South Asian, or African exchanges where market microstructure differs fundamentally.

### Gap 3: Tick-Size Analysis Outside NASDAQ
HLOB (2024) demonstrates that tick-size classification (large/medium/small) fundamentally changes the optimal architecture. For NSE, tick sizes vary by price range (₹0.05 for prices up to ₹50, ₹0.10 for ₹50–₹100, ₹0.25 for ₹100–₹250, etc.), creating a natural experiment. No study has applied tick-size-aware architecture selection to an exchange with variable tick-size rules.

### Gap 4: Uncertainty Quantification for Risk-Managed Trading
Both Bayesian TABL and GP models demonstrate that calibrated uncertainty estimates improve decision quality. However, neither has been applied to markets with structural risk constraints (circuit breakers, price bands, mandatory delivery). Indian regulatory mechanisms create natural thresholds where uncertainty quantification is practically valuable.

### Gap 5: Foundation Models for LOB
No published work applies large pre-trained time-series foundation models (TimesFM, Chronos, Moirai, Lag-Llama) to LOB data, either zero-shot or fine-tuned. The ICLR 2025 benchmark shows generic sequence models fail without LOB-specific adaptation — but these models are not foundation models and were not pre-trained on financial data. A pre-trained model with broader temporal pattern knowledge may generalize differently.

---

## 10. Conclusion

The LOB prediction literature has matured substantially since DeepLOB (2019). The architectural consensus is clear: spatial processing of price levels, temporal attention, and cross-feature mixing are the essential inductive biases. The evaluation consensus is equally clear: FI-2010 alone is insufficient, and F1 score alone is insufficient.

The most significant open problem is geographic generalization. Every model in this survey was developed and validated on NASDAQ or NASDAQ Nordic data. The MICEX experiment (TrioFlow, 2024) provides the first evidence that LOB prediction methods transfer to non-US markets — but a single experiment on three Russian stocks does not establish the principle broadly.

Indian equity markets — specifically NSE — represent both the largest and most structurally interesting unexplored context. With 1,600+ listed equities across vastly different liquidity profiles, tick-size regimes, and market cap ranges, NSE provides a natural laboratory for the generalizability questions the field most urgently needs to answer.

---

## References

1. Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark dataset for mid‐price forecasting of limit order book data with machine learning methods. *Journal of Forecasting*, 37(8), 852–866. arXiv:1705.03233

2. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*, 67(11), 3001–3012. arXiv:1808.03668

3. Briola, A., Turiel, J., & Aste, T. (2020). Deep learning modeling of limit order books: A comparative perspective. arXiv:2007.07319

4. Kolm, P. N., Turiel, J., & Westray, N. (2021). Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book. *SSRN* 3900141.

5. Lucchese, L., Pakkanen, M. S., & Veraart, A. E. D. (2022). Short-term predictability of returns in limit order book markets. arXiv:2211.13777

6. Zaznov, I., Kunkel, J., Dufour, A., & Badii, A. (2022). Predicting stock price changes based on the limit order book: A survey. *Mathematics*, 10(8), 1234.

7. Guo, Y., & Chen, X. (2023). Forecasting the mid-price movements with high-frequency LOB: A dual-stage temporal attention-based deep learning architecture. *Arabian Journal for Science and Engineering*, 48, 9597–9618.

8. Kisiel, D., & Gorse, D. (2022). Axial-LOB: High-frequency trading with axial attention. *IEEE SSCI 2022*. DOI:10.1109/SSCI51031.2022.10022284

9. Magris, M., Shabani, M., & Iosifidis, A. (2023). Bayesian bilinear neural network for predicting the mid-price dynamics in limit-order book markets. *Journal of Forecasting*, 42(6). arXiv:2203.03613

10. Nagy, P., Frey, N., Wattenhofer, R., & Trimpe, S. (2023). LOBS5: Generative modelling of limit order book sequences. arXiv:2309.00638

11. Ye, W., Yang, J., & Chen, P. (2023). Short-term stock price trend prediction with imaging high frequency limit order book data. *International Journal of Forecasting*, 40(3), 1189–1205.

12. Prata, M., et al. (2023). LOBCAST: An open-source library for LOB feature extraction, modelling, testing and forecasting. arXiv:2308.01915. *Artificial Intelligence Review* (2024 journal version).

13. Briola, A., Bartolucci, S., & Aste, T. (2024). HLOB: Information filtering networks and temporal self-attention for limit order book forecasting. arXiv:2405.18938

14. Briola, A., et al. (2024). Deep limit order book forecasting. arXiv:2403.09267. *Quantitative Finance* (2025 journal version).

15. Liu, R. (2024). Mid-price forecasting in limit order books with Gaussian process models. MSc Thesis, Tampere University. November 2024.

16. Nguyen, D.-P., et al. (2022). Deep hybrid models for forecasting stock midprices from the high-frequency limit order book. *FDSE 2022*, Springer CCIS 1688, pp. 393–406.

17. Berti, G., & Kasneci, G. (2025). TLOB: A new transformer model with dual attention for superior generalizability in stock price trend prediction. arXiv:2502.15757

18. Zaznov, I., Kunkel, J. M., Badii, A., & Dufour, A. (2024). The intraday dynamics predictor: A TrioFlow fusion of convolutional layers and gated recurrent units for high-frequency price movement forecasting. *Applied Sciences*, 14(7), 2984.

19. Anonymous. (2025). A benchmark study for limit order book (LOB) models and time series forecasting models on LOB data. *ICLR 2025* (Learning on Time Series and Dynamical Systems Track). OpenReview: MhD9rLeU31.
