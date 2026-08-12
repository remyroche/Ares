# OOF GAM simplification and portability-aware feature selection

Date: 2026-08-12  
Population: identical prior-OOF GAM panel, 154,072 scored rows, April–December 2024  
Ranking: one pooled global top-10% ranking after causal mapping; no timestamp-local selection

## What was tested

The sealed control uses fixed additive cubic splines:

```text
SplineTransformer(n_knots=5, degree=3) + Ridge(alpha=2)
```

The ablation tested:

| Variant | GAM | Feature selection |
|---|---|---|
| simple_quadratic | 3 knots, degree 2, Ridge α=10 | fixed existing field list |
| portable_simple | 3 knots, degree 2, Ridge α=10 | prior-only portable selection |
| portable_linear | 2 knots, degree 1, Ridge α=20 | prior-only portable selection |

For each held trust fold, portable selection used three expanding validation
blocks inside the earlier training population.  Each field was scored using:

```text
median(validation top-10 net EV)
- 0.75 × MAD(validation top-10 net EV)
- max(0, −worst validation top-10 net EV)
```

`raw_trust_score` was always retained.  Regime/transition arms were capped at
four fields and combined arms at six.  No held-fold outcome was used.

## Results

Values below are net bps/trade; the same-side control is the sealed current GAM
with the corresponding source arm.

| Arm | Top-10 net | Execution rank IC | Worst month | Portability score | Positive months |
|---|---:|---:|---:|---:|---:|
| Control baseline | −107.78 | 0.0936 | −167.95 | −317.59 | 0/9 |
| Control regime | −116.53 | 0.0774 | −158.60 | −310.51 | 0/9 |
| Simple quadratic regime | **−110.18** | 0.0782 | −156.41 | **−300.16** | 0/9 |
| Portable simple regime | −114.32 | 0.0836 | −159.68 | −302.01 | 0/9 |
| Portable linear regime | **−110.06** | **0.0853** | **−155.63** | −301.12 | 0/9 |
| Control combined + adverse | −112.17 | 0.0841 | −180.03 | −323.40 | 1/9 |
| Portable linear combined + adverse | −113.92 | **0.0954** | **−164.18** | −304.87 | 1/9 |

The best portability-aware model is `portable_linear__regime_gam`: compared
with the matched regime control, it improves pooled top-10 net by 6.47 bps,
worst month by 2.97 bps, and execution rank IC by 0.0079.  Its month-level
portability score improves by 9.39 bps-equivalent points.

The fixed-list `simple_quadratic__regime_gam` has the best absolute portability
score (−300.16), but it is not itself a portability-selected feature contract.
It is therefore a useful low-complexity control, not the preferred portable
contract.

No arm clears costs at top-10.  None is promoted to production.

## What portability selected

For the portable regime arm, `raw_trust_score` and
`regime_state_entropy` were retained in every fold.  The remaining selected
fields varied among regime OOD score, uncertainty, margin, and state posterior
coordinates.  The mean pairwise selected-set Jaccard was approximately 0.42.

This means the portability-aware selector improved economic stability, but it
has not yet produced a fully persistent semantic regime contract.  The gain is
consistent with lower GAM variance plus removal of unstable context terms,
not proof that any one regime coordinate is universally portable.

## Correctness

- Selection uses only earlier rows inside each trust fold.
- Labels are required to resolve before each nested validation block.
- Features have 100% finite coverage in the selection audit.
- No target-like or action-layer field was selected.
- Global ranking is performed after score generation.
- Focused tests: `3 passed`.

Artifacts:

- [ablation artifacts](/Users/remyroche/Documents/Ares/data_perp/artifacts/oof_gam_portability_simplification_20260812_v2/)
- [runner](/Users/remyroche/Documents/Ares/scripts/run_oof_gam_portability_ablation.py)
- [tests](/Users/remyroche/Documents/Ares/tests/test_oof_gam_portability_ablation.py)
