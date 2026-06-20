# Paper Notes: Short-term Stock Price Trend Prediction with Imaging High Frequency LOB Data

**Full title:** Short-term stock price trend prediction with imaging high frequency limit order book data
**Authors:** Wuyi Ye, Jinting Yang, Pengzhan Chen
**Published:** International Journal of Forecasting, Vol. 40, Issue 3, pp. 1189–1205 (published online 2023, in print 2024)
**URL:** https://www.sciencedirect.com/science/article/abs/pii/S0169207023001073

---

## Why This Paper Matters

This paper contributes a different answer to the question: *what is the right input representation for LOB prediction?*

Most papers (DeepLOB, TLOB, HLOB) use the raw LOB tensor directly — a time × features matrix fed into CNNs or attention models. This paper proposes **imaging the LOB**: converting the raw LOB data into standardized 2D images and treating mid-price prediction as an **image classification problem**.

The core claim: by normalizing LOB data into a consistent visual representation (removing scale differences between stocks, handling price/volume heterogeneity), the image encoding reveals patterns that raw feature vectors obscure. A CNN processing the image can learn spatial LOB patterns in the same way it learns to recognize objects in photographs.

---

## The Imaging Approach

### The Problem with Raw LOB Representations

Standard LOB data has two heterogeneity problems:
1. **Price scale varies:** A LOB snapshot for a $2,000 stock (like GOOG) has very different raw price values from a $20 stock, making cross-stock generalization hard
2. **Volume distribution varies:** Volume per level differs enormously across stocks and across time — raw volume features are non-stationary

DeepLOB's solution (5-day rolling z-score normalization) handles this partially, but per-feature normalization doesn't capture the spatial relationship between price levels within a snapshot.

### How Imaging Works

The imaging framework transforms each LOB snapshot window into a 2D matrix that can be treated as a grayscale or multi-channel image:

**Step 1:** Take the standard T×4L LOB tensor (T time steps × 4L features for L levels)

**Step 2:** Apply a normalization that maps price and volume values into a standardized pixel intensity range (e.g., 0-255). The key design choice is that normalization is done relative to the **intra-snapshot structure** — not just temporal rolling statistics. This preserves the relative shapes of the bid/ask volume distribution.

**Step 3:** Reshape into a 2D "image" format suitable for standard CNN image classifiers (e.g., ResNet, VGG, or custom CNN designed for this image size)

**Intuition:** A LOB image at time t has a visual "shape" — the distribution of bid and ask volumes across price levels forms a visible profile. At a liquid large-tick stock, this profile is compact and symmetric. At a small-tick stock, it's sparse and irregular. A CNN trained on these images learns to recognize these structural patterns visually.

---

## Architecture

The paper uses a CNN-based image classifier applied to the 2D LOB images. The specific architecture is built on top of the imaging pre-processing step:

```
LOB snapshot window (T steps × 4L features)
           ↓
   Imaging transformation
   (normalization → 2D matrix → image)
           ↓
   CNN image classifier
   (convolutional layers → pooling → dense)
           ↓
   Softmax → 3 classes (Up/Down/Stable)
```

The contribution is primarily in the **representation** (imaging step) rather than the network architecture — the same CNN applied to raw LOB features performs worse than CNN applied to imaged LOB features.

---

## Key Results

- The image-based CNN **outperforms traditional ML methods** (SVM, Random Forest) on raw LOB features
- Also **outperforms basic DL baselines** (MLP, LSTM) on raw LOB features
- The improvement is attributed to the imaging normalization revealing structural LOB patterns that raw features don't expose directly
- The additional information implicit in the visual LOB structure contributes measurably to short-term trend prediction
- Tested on standard LOB benchmark datasets

---

## Relationship to Other Papers in This Review

| Aspect | This paper | DeepLOB (Zhang 2019) | HLOB (Briola 2024) |
|---|---|---|---|
| Input representation | 2D image from LOB | Raw LOB tensor (100×40) | TMFG-structured LOB |
| Normalization | Image-space normalization | 5-day rolling z-score | 5-day rolling z-score |
| Architecture | CNN image classifier | CNN + Inception + LSTM | HCNN + LSTM |
| Key insight | Visual structure in LOB | Spatial+temporal patterns | Non-consecutive dependencies |

The imaging approach is complementary to the TMFG/HCNN approach in HLOB: both recognize that raw consecutive LOB features don't capture the true structure. This paper uses visual normalization; HLOB uses mutual information graph structure.

---

## Key Conclusions

1. **Representation matters as much as architecture.** The same CNN produces better results on imaged LOB data than raw LOB data — suggesting the imaging normalization itself extracts useful inductive biases.

2. **Visual patterns in LOBs are learnable.** Treating LOB prediction as image classification is a valid and productive framing.

3. **Limitation:** Like all papers tested only on standard (NASDAQ-style) benchmarks, generalizability to new markets is not established.

---

## Indian Market Implications

| Finding | NSE Capstone Implication |
|---|---|
| Image normalization handles price-scale heterogeneity | Useful for NSE where stocks range from ₹10 to ₹10,000+ — imaging normalization may generalize better than raw values |
| Visual structure in LOB is learnable | Could be a simple baseline for your capstone — easier to implement than TLOB |
| Beats raw-feature CNN baselines | Establishes a stronger baseline than MLP/LSTM on raw features |
| Not compared to BINCTABL/TLOB | Gap: this approach hasn't been benchmarked against the SOTA models. For NSE, this could be worth trying as an additional comparison |
