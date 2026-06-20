# Paper Notes: Deep Hybrid Models for Forecasting Stock Midprices from the High-Frequency Limit Order Book

**Full title:** Deep Hybrid Models for Forecasting Stock Midprices from the High-Frequency Limit Order Book
**Authors:** Duc-Phu Nguyen, Nhat-Tan Le, Tien-Thinh Nguyen, Thanh-Phuong Nguyen, Tien-Duc Van, Son-Tu Phan, Khuong Nguyen-An
**Affiliation:** Ho Chi Minh City University of Technology and related Vietnamese institutions
**Published:** FDSE 2022 (9th Int'l Conference on Future Data and Security Engineering), Springer CCIS Vol. 1688, pp. 393–406, November 2022
**DOI:** 10.1007/978-981-19-8069-5_26
**URL:** https://link.springer.com/chapter/10.1007/978-981-19-8069-5_26

---

## Why This Paper Matters

This paper represents a common class of applied LOB work: taking standard off-the-shelf DL architectures (here, ResNet50 from image classification + LSTM) and adapting them to LOB mid-price direction prediction. Its value for the capstone is less in its specific results (which aren't directly comparable to FI-2010 benchmarks) and more in what it illustrates:
1. **Transfer learning from image models to LOB data** is a viable experimental direction — treating LOB as an image-like 2D input
2. **Hybrid CNN+LSTM beats each alone** — empirically validates the spatial+temporal decomposition that DeepLOB and TLOB formalize more rigorously
3. **Contrast case:** Shows what a study without LOB-specific architectural priors looks like, making the design choices in DeepLOB, HLOB, TLOB clearer by contrast

---

## The "Hybrid" Architecture

"Hybrid" = combining ResNet50 (deep residual CNN) and LSTM in a sequential pipeline. Four variants are evaluated:

```
Standalone: ResNet50 only → Dense(3) → Softmax
Standalone: LSTM only → Dense(3) → Softmax

Hybrid 1 (ResNet → LSTM):
  LOB input → ResNet50 (spatial features) → LSTM (temporal) → Dense(3) → Softmax

Hybrid 2 (LSTM → ResNet):
  LOB input → LSTM (temporal sequence) → ResNet50 → Dense(3) → Softmax
```

**Key design choice:** ResNet50 is applied via TensorFlow Hub as a pretrained KerasLayer (ImageNet-pretrained), then fine-tuned on LOB data. The LOB data is treated as a 2D input (image-like) for the ResNet50 component.

This is notable because ResNet50 was originally designed for 224×224 RGB images. Adapting it to LOB input requires reshaping the LOB tensor into a compatible format — the paper doesn't fully explain this reshape, which is a methodological gap.

---

## Input Representation

**Non-standard pipeline — NOT compatible with FI-2010's 40-feature format.**

| Step | Description |
|---|---|
| Raw source | WRDS TAQ (Trade and Quote) data, ~15GB, NASDAQ US equities |
| Resolution | Millisecond tick data |
| Aggregation | Aggregated to **second-level** (one snapshot per second) |
| Features | LOB prices and volumes at multiple levels |
| Input format | 2D tensor treated as image-like input for ResNet |

**Critical difference from core LOB literature:** Aggregating from millisecond to second-level discards the tick-by-tick dynamics that define LOB predictability in papers like DeepLOB, HLOB, and TLOB. The core LOB literature uses tick-time (event-by-event) rather than clock-time aggregation precisely because market microstructure dynamics are event-driven.

---

## Dataset

**Source:** WRDS TAQ (Wharton Research Data Services — Trade and Quote database)  
**Market:** NASDAQ, US equities  
**Size:** ~15GB  
**Resolution:** Millisecond, aggregated to second-level  
**Stocks:** Multiple US stocks (exact list not specified in available sources)  
**NOT FI-2010** — no comparison to DeepLOB benchmarks on Finnish NASDAQ Nordic stocks  

This makes the paper's results non-comparable to most other LOB papers. There is no common benchmark reference point.

---

## Results

The paper compares its four models internally:

| Model | Description |
|---|---|
| ResNet50 | Standalone spatial CNN |
| LSTM | Standalone temporal model |
| ResNet50+LSTM | CNN features → LSTM (Hybrid 1) |
| LSTM+ResNet50 | LSTM features → ResNet (Hybrid 2) |

**Direction of results:** The hybrid models outperform standalone ResNet50 and LSTM — the combination of spatial feature extraction with temporal modeling provides better predictions than either alone.

**No comparison to DeepLOB, BINCTABL, TLOB, or any standard benchmark.** The paper is self-contained in its experimental setup and does not position against the mainstream LOB literature. It does not appear in the LOBCAST benchmark study.

---

## Relationship to Core LOB Architecture Literature

| Architecture | Spatial (LOB levels) | Temporal | How Combined |
|---|---|---|---|
| DeepLOB (Zhang 2019) | Custom CNN + Inception | LSTM | CNN output → LSTM |
| HLOB (Briola 2024) | TMFG-structured HCNN | LSTM | 3 HCNN heads → LSTM |
| TLOB (Berti 2025) | Spatial self-attention | Temporal self-attention | Dual independent SA |
| **This paper (Hybrid 1)** | **ResNet50 (pretrained)** | **LSTM** | **CNN output → LSTM** |

Structurally, Hybrid 1 (ResNet50→LSTM) is the same paradigm as DeepLOB (CNN→LSTM), but using a pretrained ImageNet backbone instead of a LOB-specific CNN with Inception modules. The lack of LOB-specific design (tick-size awareness, feature ordering) is the key architectural weakness.

---

## Key Limitations

1. **No FI-2010 benchmark:** Results cannot be compared to DeepLOB, HLOB, TLOB, or any of the standard LOB models
2. **Second-level aggregation loses microstructure:** The tick-by-tick dynamics that make LOB data informative are discarded by aggregating to 1-second resolution
3. **ResNet50 on LOB is unmotivated:** The ResNet50 architecture was designed for natural image statistics (spatial correlation in 2D pixel grids). LOB data has different structural properties — adapting it without theoretical motivation is ad hoc
4. **Regional conference venue:** FDSE 2022 is not a top-tier ML or finance venue; the paper has not been reproduced or extended by the broader community
5. **No microstructure priors:** Ignores tick size, stock type, information richness, normalization considerations that the core literature identifies as essential

---

## Key Conclusions

1. **Hybrid CNN+LSTM outperforms either standalone** on their dataset — consistent with the broader finding (DeepLOB, TLOB) that spatial+temporal decomposition is valuable for LOB data.

2. **Transfer learning from image classification to LOB is possible** — but the paper doesn't establish that ImageNet-pretrained features are meaningfully relevant to LOB patterns (this is an unverified assumption).

3. **Not a core LOB architecture paper.** This is an applied engineering study, not a contribution to LOB-specific architecture design.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Hybrid CNN+LSTM beats standalone | Validates the CNN+LSTM paradigm empirically — confirms DeepLOB-style architecture is a sensible baseline for NSE |
| No FI-2010 testing | Gap: this paper's approach is untested on standard benchmarks — can't compare to BINCTABL or TLOB |
| Second-level aggregation | NSE data, if sampled at second-level, would face the same microstructure loss — use event-level snapshots instead |
| Transfer learning from images | Possible exploratory idea for NSE — but requires careful LOB-to-image representation design (see Imaging LOB paper, Ye et al. 2023) |
