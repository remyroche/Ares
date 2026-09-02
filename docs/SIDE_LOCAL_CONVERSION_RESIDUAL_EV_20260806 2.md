# Side-local conversion + residual-specialist ablation

Date: 2026-08-06  
Artifact: `data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1`

## What was tested

This is a strict OOS ablation of the requested stack:

1. The base score (`P(clear) - P(adverse)`) is ranked within each 4-hour × side query and admitted at the top 40%.
2. Seven frozen specialist views are retrained per side/fold, but their target is now the ordinalized residual
   `H12 exact net bps - causal side-local base EV map bps`, rather than direct net or an unrelated event target.
3. The conversion model is a native LambdaRank model trained on the same residual target, per side and per 4-hour query. It receives the base outputs, map-support/trust fields, seven residual-specialist outputs, and 85 causal regime/context fields.
4. The base score is mapped to expected net bps with a prior-resolved, side-local monotone PAVA map. The conversion score is mapped separately to expected residual bps with the same side-local, prior-resolved procedure. The strict boundary is `label_available_ts < decision_timestamp`.
5. Final score: `base_EV + lambda × mapped_residual_EV` for admitted rows, with `lambda ∈ {0, .25, .5, .75, 1}`. Lambda is selected on the calibration slice per side/fold at top-10% net; the selected OOS score is then ranked globally.

The three transport folds use one year of training and one-month calibration before each two-month (or final one-month) test window:

| Fold | Train | Calibration | OOS |
|---|---|---|---|
| Jul–Aug | Jul 2023–May 2024 | Jun 2024 | Jul–Aug 2024 |
| Sep–Oct | Jul 2023–Jul 2024 | Aug 2024 | Sep–Oct 2024 |
| Nov partial | Jul 2023–Sep 2024 | Oct 2024 | Nov 2024 |

There are 388,494 OOS candidates (194,247 long and 194,247 short). The residual LambdaRank queries contain 1,593–2,325 queries per side/fold, with median 92–93 candidates/query (minimum 64–65).

## True globally ranked OOS result

These figures rank the complete two-sided candidate population, not separate per-side tails. Net includes the 100-bps cost floor.

| Score | Top 1% net | Top 5% net | Top 10% net | Global rank IC |
|---|---:|---:|---:|---:|
| Base EV only (`λ=0`) | −22.33 | **+11.52** | −43.33 | 0.0395 |
| `λ=.25` | +4.95 | −21.62 | −47.00 | 0.0510 |
| `λ=.50` | **+8.15** | −62.37 | −57.34 | 0.0498 |
| `λ=.75` | −42.16 | −61.14 | −76.58 | 0.0457 |
| `λ=1` | −92.61 | −64.21 | −79.93 | 0.0427 |
| OOF-selected side/fold λ | −59.56 | −52.09 | −54.88 | 0.0417 |

The residual conversion component does not advance the base system. A fixed small residual weight improves only the global top-1% point estimate; it damages the top-5% and top-10% ranking. The OOF-selected mixture is worse than the no-op base at all three tails.

## Side-local result

| Side / score | Top 1% net | Top 5% net | Top 10% net | Rank IC |
|---|---:|---:|---:|---:|
| Long, base EV | +48.16 | +1.15 | +11.52 | 0.0439 |
| Long, OOF-selected λ | −51.66 | −51.92 | −36.62 | 0.0354 |
| Short, base EV | −148.84 | −178.86 | −159.11 | −0.0068 |
| Short, OOF-selected λ | −117.80 | −132.30 | −136.81 | 0.0352 |

Per-side lambda selection partially repairs the short rank ordering but substantially damages the long side. This means “side-local” mapping by itself is not enough to make the two side scores safely comparable for a pooled global ranking; the map needs a separate cross-side calibration/decision gate or a common-bps shrinkage layer.

## Month and transport stability of the selected OOS score

| Period | Top 1% net | Top 5% net | Top 10% net | Rank IC |
|---|---:|---:|---:|---:|
| Jul 2024 | −163.63 | −170.80 | −118.27 | −0.0603 |
| Aug 2024 | −109.10 | −199.13 | −144.15 | +0.0048 |
| Sep 2024 | −42.30 | −95.20 | −93.26 | −0.0171 |
| Oct 2024 | −61.19 | −93.97 | −120.57 | +0.0185 |
| Nov 2024 | **+110.06** | +12.00 | +17.39 | +0.1422 |

By transport fold, top-10% net is −139.30 bps (Jul–Aug), −101.65 bps (Sep–Oct), and +17.39 bps (Nov). The apparent pooled result is therefore dominated by a late positive month and is not portable.

## Residual learnability diagnostics

On the 156,126 admitted OOS rows, the mapped residual score has only +0.035 Spearman correlation with realised residual net. It is +0.029 on long and +0.029 on short. The admitted residual mean is +14.1 bps long and −5.3 bps short, so the residual problem is asymmetric even after side-local base mapping.

## Decision

`NO_RESIDUAL_CONVERSION_ADVANCE` for this contract. The requested mechanics are implemented and leakage-safe, but the conversion layer is too weak and the side-local maps do not produce stable cross-side global comparability. Keep the base-only side-local EV map as the control. Do not promote the specialist/residual mixture to policy.

The next targeted repair should be a calibration-only experiment before another residual model sweep:

- fit a held-out side × month (or side × volatility/cost-to-ATR) shrinkage map in common bps;
- require a positive absolute mapped EV threshold per side before global ranking;
- evaluate `base_EV`, residual EV, and combined EV with a cross-side calibration intercept/slope learned only from prior resolved rows;
- reject the residual correction unless it improves both top-10% pooled net and the worst transport month.

## Artifacts and verification

- [conversion contract](../data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1/conversion_contract.json)
- [OOF lambda selections](../data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1/lambda_selection.parquet)
- [true global/side/month metrics](../data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1/global_metrics.parquet)
- [predictions](../data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1/predictions.parquet)
- [specialist target audit](../data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1/specialist_target_audit.parquet)

Focused regression suite: **15 passed** (`test_side_local_conversion_residual_ev.py`, top-40 reliability, gated residual, multiview specialists, and residual HPO contracts).
