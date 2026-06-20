# Paper Notes: LOBS5 — Generative AI for End-to-End Limit Order Book Modelling

**Full title:** Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network
**Authors:** Peer Nagy, Sascha Frey, Silvia Sapora, Kang Li, Anisoara Calinescu, Stefan Zohren, Jakob Foerster (Oxford-Man Institute, University of Oxford)
**Published:** arXiv 2309.00638, August 2023 (ICAIF 2023)
**File:** `[N5] LOBS5_Nagy_2023.pdf`

---

## Why This Paper Matters

All other papers in this review treat LOB forecasting as a **classification problem**: given a snapshot history, predict direction (Up/Down/Stable). LOBS5 takes a fundamentally different approach: **generate the next LOB message**.

This is the difference between a model that classifies tomorrow's weather and a simulator that generates actual atmospheric states. The generative approach gives you:
1. A "world model" of the LOB — full LOB trajectories, not just direction labels
2. Counterfactual scenarios (what if a large sell order arrived next?)
3. Forecasted mid-price returns as a byproduct — not explicitly trained for
4. A foundation for reinforcement learning trading agents

This paper is the first autoregressive end-to-end generative model for LOB **messages** (not prices or returns) — the LOB equivalent of GPT for financial microstructure data.

---

## What Problem Does It Solve?

**Prior generative approaches and their limitations:**

| Prior approach | Limitation |
|---|---|
| GAN-based models (TimeGAN, FinGAN) | Generate price return sequences, not order-level data; mode collapse; no tractable distribution over sequences |
| RNN-based LOB generator (Hultin 2023) | Models level-2 aggregates with binned values — loses precision; cannot reference individual orders |
| Discriminative classifiers (DeepLOB, etc.) | Only predict a label; cannot generate new market states |

**The key insight:** LOB message sequences are structurally similar to language. Each "message" is like a "sentence" with structured fields (event type, direction, price, size, timestamp). A language model trained to predict the next word can be adapted to predict the next order — using the same tokenization and autoregressive training paradigm.

**Why autoregressive models are better than GANs for this task:**
- Define a tractable probability distribution over sequences of any length
- No mode collapse
- Naturally interpretable: the model defines p(next token | all previous tokens)
- Scale better with data (evidence from LLMs)

---

## The LOB Message Structure

LOBSTER provides 6-field messages:
1. **Timestamp** — nanoseconds after midnight (15-digit integer)
2. **Event type** — 1: new limit order, 2: partial cancellation, 3: full deletion, 4: visible execution
3. **Order ID** — identifier (not used directly due to non-stationarity)
4. **Size** — shares
5. **Price** — in cents
6. **Direction** — buy or sell

**Pre-processing steps:**
- Use only event types 1-4 (exclude hidden orders, auction trades, trading halts)
- Convert prices from dollar values to **tick distances from current mid-price** — makes prices stationary over time
- Truncate extreme values: prices beyond ±999 ticks, sizes > 9999 (affects < 0.1% of data)
- Add **inter-arrival time Δt** between consecutive messages (more stationary than raw timestamps)
- For referential messages (cancellations, deletions, executions): append the **reference fields** — modified price, size, and timestamp of the original order being referenced
- Reorder fields so higher-entropy fields appear later (conditions on more certain fields first)

**Final field order per message:**
event type → direction → price sign → price tick distance → size → Δt (4 tokens) → arrival time (5 tokens) → reference price sign → reference price distance → reference size → reference time (5 tokens)

---

## Tokenization (Section 4.2)

The vocabulary has **12,011 distinct tokens**. Each message is encoded as **22 tokens**:

| Field | # Tokens | Notes |
|---|---|---|
| Event type | 1 | 4 possible values |
| Direction | 1 | buy or sell |
| Price sign | 1 | above or below mid-price |
| Price tick distance | 1 | 0–999 ticks |
| Size | 1 | 1–9999 shares |
| Δt (inter-arrival time) | 4 | tokenized in groups of 3 digits |
| Arrival time | 5 | 15-digit nanosecond timestamp → 5 groups of 3 digits |
| Ref: price sign | 1 | sign of original order's price |
| Ref: price distance | 1 | tick distance of original order |
| Ref: size | 1 | size of original order |
| Ref: arrival time | 5 | timestamp of original order |
| **Total** | **22** | For new limit orders: ref fields = NA |

**Important design choice:** Non-overlapping token ranges for different field types. Even if event type and price level share the same raw integer value, they receive different tokens. This lets the model learn field-specific conditional distributions and makes semantic structure explicit.

Δt and arrival time share the same vocabulary (both are time durations tokenized as 3-digit groups). Reference time tokens are structurally easier to predict — the referenced order's timestamp usually already appeared in the input sequence.

---

## Model Architecture (Section 5.1)

### Background: S5 (Simplified Structured State Space)

S5 is a sequence model based on state space models (SSMs). The SSM defines a dynamical system:
```
x'(t) = Ax(t) + Bu(t)   [state transition]
y(t)  = Cx(t) + Du(t)   [output]
```

Matrices A, B, C, D are learned. Combined with the HiPPO initialization framework (which initializes A to preserve polynomial projections of the input history), SSMs excel at long-range dependencies.

**S5 vs. S4 vs. Transformer:**
- **Transformer:** O(L²) complexity — quadratic in sequence length. Impractical for 11,000-token sequences.
- **S4:** Many single-input single-output SSMs; O(H²) per step at inference.
- **S5:** One multi-input multi-output SSM; **O(L) complexity at inference** via parallel scan; better performance on Long Range Arena benchmarks. Allows varying sampling time steps.

For LOBs where message sequences can exceed 10,000 tokens, S5's linear complexity is essential.

### Two-Branch Architecture

```
Masked message sequence           Order book volume images
n×22 tokens = 11,000 tokens       n=500 book snapshots
         ↓                                 ↓
  Linear embedding              S5 layer (P+1 features → H)
  (one-hot → hidden dim H)
         ↓                                 ↓
       S5 layers                      project to dim H
         ↓                                 ↓
       project to sequence length L ──────┘
                          ↓
                  concatenate branches
                          ↓
               Combined S5 Module (6 S5 layers)
                          ↓
             Average over sequence dimension
                          ↓
              v=12,011 output neurons (logits)
                       (softmax)
```

**Message branch:** Flattened token sequences. One-hot token vectors → linear embedding (dim H) → S5 layers.

**Book branch:** Level-2 order book state as a sparse "volume image": P volume features centered around mid-price + 1 mid-price change feature = P+1 dimensional vector per snapshot. This representation preserves the price structure (empty price levels are explicitly represented as zeros, unlike standard LOB representations that collapse them). Book states feed through an S5 layer → projected to hidden dimension H.

Both branches projected to common sequence length L, concatenated, then processed by the Combined S5 module (6 S5 layers). Final output: average over sequence dimension → 12,011 logit outputs.

**Total: ~6.3×10^6 parameters.**

**Training procedure (masked language modeling style):**
- Mask one random token in the last message (MSK token); replace all subsequent tokens with HID tokens
- Model predicts only the masked token
- Time tokens of new messages are not prediction targets (computed from Δt); reference time tokens of non-new-order messages are prediction targets
- Input: n=500 messages per sequence; random starting offset each epoch
- Optimizer: Adam; Jax framework (GPU-accelerated via JIT compilation)

---

## Inference Loop (Section 5.2)

The model is embedded in a simulation loop with two nested loops:

**Token Generation Loop (inner):**
1. Feed (masked message sequence, book sequence) to model
2. Sample next token from softmax over logits; restrict to syntactically valid tokens for the current field
3. Repeat until all 22 tokens of the new message are generated
4. Once Δt tokens are generated, compute: arrival time = previous timestamp + Δt

**Message Generation Loop (outer):**
1. Decode complete generated message
2. Apply **error correction** for referential messages (cancellations, deletions, executions):
   - Search input sequence for existing order matching generated reference (direction + price + size + time)
   - If no match: search without time field (time has highest error rate)
   - If still no match: apply to initial volume at correct price level (handles orders before context window)
3. Submit corrected message to **Jax-LOB simulator** (GPU-accelerated, JIT-compiled)
4. Simulator: new_book_state = sim(current_book_state, generated_message)
5. Add generated message and new book state to input sequences
6. Return to step 1

The Jax-LOB simulator is essential — it deterministically updates the LOB state given each generated message and feeds back the new level-2 volume images into the book branch.

---

## Results (Section 6)

### Perplexity

Overall per-token PPL (lower is better = model assigns higher probability to actual data):

| Stock | PPL | Std. err |
|---|---|---|
| GOOG | **3.63** | 0.0047 |
| INTC | **4.04** | 0.0043 |

GOOG has lower PPL because it has higher trading volume → more training data. Both are well below 12,011 (which would be random guessing over all tokens), indicating meaningful learning.

Per-field perplexity breakdown (Table 1, selected):

| Field | GOOG PPL | INTC PPL | Interpretation |
|---|---|---|---|
| Event type | 2.15 | 1.92 | Easy — only 4 possible types |
| Direction | 1.71 | 1.55 | Easy — buy or sell |
| Price tick distance | **5.55** | **2.41** | GOOG harder: small-tick, sparse price distribution |
| Size | **3.18** | **13.51** | INTC harder: wider spread in order sizes |
| Δt (4 tokens) | 1.00–6.21 | 1.01–5.45 | Later tokens harder (higher entropy for longer intervals) |
| Reference arrival time | 405–770 | 122–524 | Hardest: references old events potentially outside context |

The GOOG model has higher perplexity for price prediction because GOOG is a small-tick stock — its order book is sparse and prices can be placed anywhere in a wide range (many ticks from mid-price). INTC, as a large-tick stock, has a denser and more predictable order book, so price placements are more constrained.

The reference arrival time tokens have very high PPL because they reference orders that may have been submitted before the 500-message context window, making exact timestamp recovery impossible from the sequence alone.

### Return Correlation (Figure 8)

Pearson correlation ρ between **generated** mid-price returns and **realized** mid-price returns, for s = 1...100 messages into the future:

| Stock | ρ | Significant up to |
|---|---|---|
| GOOG | ~0.1 | ~80 messages |
| INTC | ~0.2 | ≥100 messages |

The model was **not trained to forecast returns** — it was trained to predict the next token. The return correlation emerges as a byproduct of learning accurate message distributions. This is competitive with DeepLOB-style discriminative models trained explicitly for this task.

INTC (large-tick, less liquid) remains predictable further into the future — consistent with findings across all papers in this review.

### Distribution Matching (Figures 4–6)

- **Mid-price returns (Figure 4):** 95% confidence intervals of generated distributions overlap well with realized distributions for both stocks. No drift or systematic bias.
- **Event type frequencies (Figure 5):** Approximately matched, but execution events are overestimated by ~5–10% — a known effect of cross-entropy loss's mode-covering tendency (the model spreads probability mass over more outcomes than actually occur).
- **Inter-arrival times Δt (Figure 6):** Well matched — P-P plots show near-diagonal curves; histograms on log-x scale show the heavy-tailed distribution is captured.

---

## Key Conclusions

1. **First autoregressive end-to-end generative model for LOB messages.** Prior work generated level-2 aggregated data or price returns — this generates individual orders at full level-3 resolution.

2. **S5 (state space model) handles long message sequences efficiently.** Linear O(L) complexity at inference vs. Transformer's O(L²) is essential for 11,000-token sequences.

3. **Return forecasting emerges from generation** without explicit training. ρ≈0.1–0.2, significant up to 80–100 messages ahead — competitive with discriminative models.

4. **Large-tick stocks (INTC) are more predictable** (ρ≈0.2 vs. 0.1 for GOOG). Consistent with every other paper in this review.

5. **The model defines a world model for the LOB.** Given initial state + context, it simulates future order flow. This enables RL agents for trading, market making, or order execution optimization.

6. **Scaling potential.** NASDAQ has 2500 stocks × millions of daily messages. The tokenization approach is directly scalable — this is the foundation for "large financial models."

7. **Current limitations:**
   - Referential order hallucinations require computationally intensive error correction
   - Only trained on 2 stocks — cross-stock generalization not tested
   - Context window limited to 500 messages; references outside window are imprecisely handled

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Autoregressive LOB generation enables simulation | For NSE: a generative model could simulate circuit-breaker events or large FII block-order impacts — novel application |
| S5 model: O(L) complexity | Feasible for NSE data which may have different message rates than NASDAQ |
| Return forecasting emerges as byproduct | Framing: if you built LOBS5 for NSE, you get forecasting "for free" from generation |
| Large-tick (INTC) more predictable (ρ≈0.2) | NSE banking stocks likely ρ>0.2 given less algorithmic trading → less efficient market |
| 22 tokens per message structure | NSE uses the same 6-field message structure — tokenization directly applicable |
| Execution messages overestimated in output | NSE has circuit breakers that may create different execution patterns — worth testing |
| Only 2 stocks, 102 training days | Too complex for capstone implementation; cite as future direction showing awareness of frontier |
| World model for RL | Position as "next step beyond forecasting" in project proposal — shows research depth |
| GANs have mode collapse problems | Justify using discriminative classifiers (not generative) for your forecasting task — appropriate for capstone scope |
| LLM-style scaling to all 2500 NASDAQ stocks | If Indian market data becomes available at scale: "large NSE model" is a genuine future research direction |
