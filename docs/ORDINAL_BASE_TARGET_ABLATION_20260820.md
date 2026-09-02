# Ordinal Base-Target Ablation — 2024

## Scope

This is a leakage-safe base-layer target comparison, not a stack or policy
promotion.  Both sides use the same chronological contract:

- training decisions: 2024-01-01 through 2024-03-31;
- held decisions: 2024-04-01 through 2024-06-30;
- labels admitted to fitting only when their H12 outcome was available before
  2024-04-01;
- feature selection: frozen 120-field side-local base contracts; coverage is
  measured on target-free, decision-time executable candidates only;
- model: the frozen current R3 base LGBM parameters (140 trees, learning rate
  0.05, 31 leaves, minimum child 350, 0.8 row/feature fractions, L2 8);
- H12 labels: exact decision-time entry at signal close + one hour, TP +6 ATR,
  SL -4 ATR, adverse precedence on a same-minute conflict, cost 100 bps once;
- every arm ranks all OOS executable candidates before missing future outcomes
  are inspected.

The control is the current R3 score:

```text
P(robust clear) - 0.5 × P(adverse)
```

Ordinal arms have four classes:

```text
net <= lower; lower < net <= 0; 0 < net <= upper; net > upper
```

For ordinal arms, the score is the predicted class probability weighted by the
**training-only** class median net outcome, monotonically repaired if needed.
It is not fitted or calibrated on April--June.

## Long result

Long support was 96,916 valid training rows and 188,872 scored OOS candidates
(104,760 resolved H12 labels).  All 120 frozen features passed the causal
coverage gate; minimum coverage was 94.44%.

| Long target | H12 net top 1% | Top 2% | Top 5% | Net-score Spearman |
|---|---:|---:|---:|---:|
| **Current R3 control** | **+88.60** | **+66.24** | **−0.67** | +0.0518 |
| Ordinal −150 / 0 / +25 | +7.44 | −18.01 | −28.73 | +0.0561 |
| Ordinal −200 / 0 / +50 | +11.84 | −18.06 | −24.50 | **+0.0623** |
| Ordinal −250 / 0 / +50 | +47.55 | +13.00 | −21.20 | +0.0526 |
| Ordinal −250 / 0 / +75 | +22.86 | +10.53 | −15.04 | +0.0452 |
| Ordinal −300 / 0 / +100 | +23.78 | −9.37 | −35.59 | +0.0519 |

Values are net bps/trade.  Although two ordinal labels slightly improve
unconditional net rank correlation, neither matches R3 where the base layer is
used: the top 1--2% tail.  **Keep the current long R3 target.**

## Short result

Short support was 96,916 valid training rows and 188,770 scored OOS candidates
(135,275 resolved H12 labels).  All 120 frozen features passed the causal
coverage gate; minimum coverage was 90.02%.

| Short target | H12 net top 1% | Top 2% | Top 5% | Net-score Spearman |
|---|---:|---:|---:|---:|
| Current R3 control | −50.58 | −71.09 | −83.43 | −0.0094 |
| Ordinal −150 / 0 / +25 | −33.82 | −50.19 | **−52.12** | **+0.0383** |
| Ordinal −200 / 0 / +50 | −16.45 | −44.92 | −74.55 | +0.0183 |
| Ordinal −250 / 0 / +50 | **−9.23** | −44.63 | −77.28 | +0.0180 |
| Ordinal −250 / 0 / +75 | −45.10 | −73.92 | −82.67 | +0.0159 |
| Ordinal −300 / 0 / +100 | −22.78 | −47.14 | −70.80 | +0.0145 |
| −200 / 0 / +50, boundary certainty | −20.56 | −44.10 | −77.85 | +0.0221 |
| −200 / 0 / +50, mild class balance | −40.34 | −65.99 | −81.85 | +0.0271 |
| −200 / 0 / +50, hybrid | −21.88 | **−40.24** | −64.59 | +0.0265 |
| −250 / 0 / +75, hybrid | −30.57 | −60.23 | −87.43 | +0.0277 |

Training-only weighting details:

- `boundary certainty`: `0.25 + 0.75 × sigmoid(distance_to_nearest_edge / 50)`;
- `mild class`: a mean-normalised square-root inverse-frequency factor;
- `hybrid`: their product, capped to [0.25, 4] then mean-normalised.

Weights never enter features, OOS scoring, ranking, or policy replay.

The ordinal labels repair the reverse net relationship of the R3 control and
reduce tail harm, but they do not yield a positive short base.  The provisional
short candidate is **−250 / 0 / +50, uniform** for a future *separate*
base-to-meta experiment because it is best at the extreme tail; it is not
canonical or promoted.  The −150 / 0 / +25 label is the strongest broad-rank
control and should accompany that experiment.

## Frozen short policy diagnostic

To avoid judging short target candidates only against TP6/SL4, the selected
OOS tails were replayed with a fixed side-correct exact-one-minute parent
SimplePolicy:

```text
entry: exact decision-minute open
stop: 3 ATR
trail activation: 0.5 ATR
giveback: 0.25 ATR
timeout: 12 hours
cost: 100 bps once
```

No short policy HPO was conducted.  This is therefore an economic diagnostic,
not a promoted short exit policy.  The exact-1m source covered 67.5--68.5% of
the post-ranking selected tails; unavailable paths were excluded rather than
encoded as failures.

| Short target | Policy net top 1% | Top 2% | Top 5% |
|---|---:|---:|---:|
| Current R3 control | −86.42 | −83.90 | −87.67 |
| Ordinal −150 / 0 / +25 | −79.20 | −85.27 | **−81.03** |
| Ordinal −200 / 0 / +50, hybrid | −80.20 | −80.85 | −81.60 |
| Ordinal −250 / 0 / +50 | −82.62 | −92.55 | −89.50 |
| Ordinal −300 / 0 / +100 | **−77.66** | **−78.45** | −83.19 |

No arm is positive under this policy either.  A short-policy HPO must use a
strictly earlier OOF score population, then be frozen before any later short
target comparison; it must not select a policy from this April--June evidence.

## Artifacts

- Runner: `scripts/run_strict_r3_ordinal_base_target_ablation.py`
- Long control: `data_perp/artifacts/strict_r3_long_ordinal_target_arms_3m_oos_2024_20260820_v1/R3_current_control`
- Long ordinal arms: `data_perp/artifacts/strict_r3_long_ordinal_target_arms_3m_oos_2024_20260820_v2/`
- Short arms: `data_perp/artifacts/strict_r3_short_ordinal_target_arms_3m_oos_2024_20260820_v2/`

Each arm contains its feature/label hashes, frozen parameters, per-month H12
metrics, full OOS predictions, and—for short—the post-ranking exact-1m policy
outcome/coverage receipt.

## Decision

- **Long:** retain current R3.
- **Short:** do not promote a base target.  Carry `−250/0/+50 uniform` and
  `−150/0/+25 uniform` into a later strictly OOF base+meta comparison, after a
  short-only frozen policy is selected from an earlier OOF development window.
