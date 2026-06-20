# Phase 2 Plan — deepening the Data-Science contribution

*Created 2026-06-13. To be confirmed with mentor before starting (planned: 2026-06-14).*

## Why
The project to date is strong on **data engineering + reproduction + transfer**, but the genuinely *new*
Data-Science contribution is thin (it reads as "we collected data and ran existing models"). Phase 2 adds
real DS novelty so the contribution is unambiguous. **Hyperparameter tuning alone is rigour, not novelty** —
the novelty comes from feature engineering and architecture design; HPO/significance/calibration wrap it.

## Tier A — Microstructure feature engineering (DE → DS bridge; **novel**)
Construct and **ablate** engineered channels on top of the raw 40-feature LOB window:
- **Order-Flow Imbalance (OFI)** (Cont et al.) and **depth imbalance** — literature shows these carry
  directional signal raw prices don't.
- **Micro-price** (volume-weighted mid) and **relative spread**.
- **Order-count field** — *unique to our Dhan data*, absent in FI-2010/Kite → a 60-feature variant.
- **20-level depth** (80 features) vs 10-level.
- **RQ:** how much / which microstructure info does the NSE index-futures signal need?
- **Deliverable:** `modeling/features.py` + an ablation table (40 vs 40+OFI vs 60 vs 80).
- **Why it matters:** highest-leverage way to actually *improve* the weak NSE numbers; directly uses our DE work.

## Tier B — Improved MambaLOB architecture (**headline novelty**)
Beyond the vanilla SSM stack:
- **Bidirectional Mamba** — forward + time-reversed scans concatenated (valid for classification; BiLSTM analog).
- **ConvMambaLOB** — DeepLOB-style convolutions (spatial LOB structure) → linear-time Mamba temporal core.
- **Goal:** match/beat TLOB at a fraction of its compute → turns "we applied Mamba" into "we designed a better LOB model."
- **Deliverable:** new variants in `modeling/models.py` (a `bidirectional` flag + a `ConvMambaLOB` class).

## Tier C — Rigour layer (expected practice; strengthens, not novelty)
- **HPO (Optuna)** on the best variant; **equal budget** for baselines (fairness).
- **Class-imbalance study:** focal loss vs class-weighted CE vs class-balanced sampling vs label smoothing.
- **Statistical significance:** multi-seed bootstrap CIs (already scaffolded in `stats.py`) + **Diebold–Mariano /
  permutation tests** for pairwise model comparison.
- **Probability calibration:** temperature scaling → feeds the cost-aware backtest (ties ML confidence to economic τ).

## Carried over from the interim plan
Scheme-B confirmatory training; cost-aware backtest (E8); walk-forward robustness; multi-seed CIs; final report consolidation.
(See `notebooks/nse_extras.ipynb` — already split out for these.)

## Recommended scope / order (time-bounded)
1. **Tier A** (feature engineering) — biggest accuracy lever + novelty.
2. **Tier B** (ConvMambaLOB + bidirectional) — headline architecture novelty.
3. **Tier C** rigour: HPO on the winner, significance tests, calibration. Skip exhaustive HPO on everything.

## Reframed contribution (one-liner for the report/mentor)
> "We engineered an NSE index-futures LOB dataset, then used it to (1) construct microstructure-aware
> features — including an order-count channel unavailable in any prior benchmark — and (2) design an improved
> selective-SSM architecture, tuned and evaluated with statistical significance and cost-aware calibration,
> achieving competitive accuracy at a fraction of the transformer's compute."

Maps to report sections: features → §3.3; architecture → §3.5/§3.7; HPO/imbalance/significance/calibration → §3.5.2/§3.6/§3.9.
