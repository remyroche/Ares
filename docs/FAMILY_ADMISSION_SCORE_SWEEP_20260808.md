# Causal 21-day admission score sweep

## Scope

The same side-local, pooled-parent 21-day map was applied to every already
materialized base/residual score. The map uses strict prior label availability,
20 fixed rank bins, 5% trimmed means, 500-row side shrinkage, and no test-outcome
selection. Evaluation is restricted to the 275,540 outer-test rows.

## Maximum mapped test expected net

| score | max mapped bps | median mapped bps | rows mapped |
|---|---:|---:|---:|
| Cap-120 / A | 18.94 | −68.74 | 274,142 |
| C family state | 16.25 | −71.95 | 274,142 |
| D MLP state | 29.99 | −71.37 | 274,142 |
| E near-tie | 15.17 | −68.71 | 274,142 |
| G recent reliability | 30.00 | −69.10 | 274,142 |
| H support/OOD | 30.65 | −69.17 | 274,142 |
| J dynamic family MLP | 22.03 | −69.12 | 274,142 |

No existing score reaches the required +50-bps mapped domain. Therefore the
declared admission floor admits zero rows for every arm.

## Lower-floor diagnostic

At a +10-bps floor, the strongest arm was D (1,174 eligible rows; realized
net +32.94 bps/trade). At +25 bps, D and G each had 310 eligible rows but only
realized +3.65 bps/trade. H had 315 eligible rows with realized +6.47 bps.
At a zero floor, the best 0.5% mapped tail was D at +25.07 bps realized, but
the full admitted set was negative.

These lower floors are diagnostic only. They do not establish a valid +50-bps
policy admission rule.

## Decision

The failure is common to the whole existing score family, not a defect unique
to H or to leaf-family weighting. A conversion-only selection among existing
arms cannot satisfy the causal admission gate. The next repair must change the
target/score semantics or the side-specific economic calibration, with a fresh
outer validation period afterward.

Artifact: [score sweep](/Users/remyroche/Documents/Ares/data_perp/artifacts/family_admission_score_sweep_20260808_v1)
