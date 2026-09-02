# Strict-R3 base-grid / F72 reconstruction — 2026-08-26

## Scope and decision status

This is an offline, long-only, causal research receipt.  It does not alter the
live stack or promote a challenger.

The requested incumbent B/E/T blend grid was completed on the January–June
2026 strict held period using four fully resolved consensus-training months,
a separate 28-day embargo, strict prior-only MC1 fits, dual current/BCF MC1
admission at +50 bps, and one chronological portfolio state.

The early causal F72 history was rebuilt and a new strict-OOF F72 B ledger was
successfully generated.  It cannot yet enter a *qualifying* six-held-month
4-month-consensus / 4-month-MC1 grid: canonical policy outcomes first exist in
April 2025, the earliest legal router score is July 2025, and the earliest
legal F72 B held score is November 2025.  Backfilling earlier supervised folds
would require missing policy paths and would violate the requested chronology.

## Rebuilt causal F72 lineage

Artifacts:

- Early target-free candidate/features: `data_perp/artifacts/strict_r3_f72_early_router_features_20260826_v3_{jan,feb,mar}_identity`
- Composite feature history: `data_perp/artifacts/strict_r3_f72_router_composite_features_20260826_v2_full`
- Rebuilt July router: `data_perp/artifacts/strict_r3_f72_router_july_rebuilt_20260826_v1`
- Strict-OOF F72 B blocks: `data_perp/artifacts/strict_r3_f72_b_rebuilt_{novjan,febapr,mayjul}_20260826_v2`
- Frozen B + incumbent E/T target-free downstream source: `data_perp/artifacts/strict_r3_f72_bonly_incumbent_et_geometry_20260826_v2`

January–March source reconstruction retained every point-in-time candidate,
including invalid/unresolved label rows, but used those label files only as an
identity ledger.  Each panel contains the candidate identity plus causal
features and no policy/path/target fields.

The rebuilt F72 B folds are target-free and strict:

| Held months | Folds | Minimum train rows | Held matrix finite fraction |
|---|---:|---:|---:|
| Nov 2025–Jul 2026 | 9 | 46,926 | 100% |

The reconstructed July router has exact candidate identity parity with the old
router.  Its rank Spearman correlation is 0.9866 rather than bit parity because
the old source silently omitted March candidates before applying its
target-free training cap.  The rebuilt source includes them; therefore it is a
new causal lineage and is not presented as a reproduction of the old model.

## Completed 4-month / 6-month incumbent grid

All rows below are portfolio-constrained, outcome-complete, and use the
dual-MC1 +50-bps admission.  `Coverage-only` lets an arm use all of its
admitted candidates.  It is the appropriate total-economic-contribution
measure, but not a same-trade-count comparison against the live source.

| Upstream coordinate | MC1 admits | Entries | Net EV/trade (bps) | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Incumbent E/T raw | 23,507 | 4,052 | +176.19 | +713,923 | +133.47 | +63.60 | -21.44% |
| B30 / E56 / T14 | 12,644 | 3,544 | +193.06 | +684,187 | +139.18 | +86.09 | -21.83% |
| B30 / E35 / T35 | 10,203 | 3,229 | +202.61 | +654,233 | +133.38 | +75.99 | -21.37% |
| E/T rank-50 | 9,726 | 3,173 | **+202.96** | +643,994 | +152.14 | **+86.80** | **-19.32%** |
| B15 / E42.5 / T42.5 | 9,793 | 3,180 | +202.22 | +643,059 | **+153.66** | +73.96 | -20.08% |
| B45 / E27.5 / T27.5 | 10,977 | 3,342 | +193.44 | +646,481 | +136.07 | +68.78 | -22.87% |

### Matched live-source comparison

The live historical baseline selected 3,627 entries at +155.60 bps/trade,
+564,362 total net bps, +128.42 worst-month EV, +43.58 worst-week EV, and
-27.81% maximum drawdown.  The following arms are re-auctioned to that exact
3,627-entry count where possible; this is the fairest comparison with live.

| Arm | Entries | Net EV/trade | Delta EV/trade | Delta total net bps | Delta worst month | Delta worst week | Delta max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Incumbent E/T raw | 3,627 | +162.62 | +7.02 | +25,472 | -7.96 | +42.87 | +5.87 pp |
| B30 / E56 / T14 | 3,030 | +186.82 | +31.22 | +1,695 | +10.28 | +59.55 | +3.89 pp |
| E/T rank-50 | 2,702 | +200.94 | +45.34 | -21,425 | +24.12 | +73.32 | +8.49 pp |

## Interpretation

No candidate is promoted.  The raw incumbent coordinate maximises total
contribution, while E/T rank-50 is the strongest precision/risk arm but loses
enough entries that its matched total falls below the live baseline.  B30/E56/T14
is the most balanced weight challenger in the matched comparison: positive
total delta, materially better per-trade EV, and stronger worst-period/DD
metrics.  It remains selection-period research evidence rather than live
evidence.

F72 is now a valid strict-OOF base-score artifact, but its available score
history is insufficient for the requested six-month fully prequential
downstream evaluation.  It must remain excluded from any live or canonical
promotion until enough post-November 2025 score history exists under one
unchanged F72/router lineage.

