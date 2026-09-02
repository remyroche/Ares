# Causal Profile Value-Area Group Ablation — 31 August 2026

## Decision

**Retain the full causal value-area context block.  Remove no group.**

The apparent June–July case for dropping HVN/LVN does not survive the sealed
August holdout.  The fields are complementary downstream inputs to the causal
S/R head and MC1 mapper; the evidence does not support treating any one group
as redundant.

This is offline research only.  It does not change the canonical or live
feature contract.

## Causal source

All variants use the same source engine and strict OOF head contract:

* profile context is computed from same-or-earlier completed states;
* an S/R head for a held month uses only interaction labels resolved before its
  train boundary;
* profile snapshots are known before the decision, never from the subsequent
  eight-hour path;
* downstream MC1 keeps the paired BCF/current maps, source-aligned parent
  policy, dual admission and constrained BCF-priority auction fixed.

The source manifests declare `causal-sr-heads-oof-v1` and explicitly prohibit
future interaction state.

## Groups tested

| Group | Fields | Leave-one-group-out source |
|---|---|---|
| POC | `profile_poc_distance_atr` | `...without_poc...` |
| VAH/VAL | `profile_vah_distance_atr`, `profile_val_distance_atr` | `...without_vah_val...` |
| HVN/LVN | `profile_hvn_distance_atr`, `profile_lvn_distance_atr` | `...without_hvn_lvn...` |
| Value-area geometry | `profile_inside_value_area`, `profile_value_area_width_atr` | `...without_value_area_geometry...` |

The full block includes all seven fields above.  No Bollinger, Keltner,
Donchian, OI-at-price, time-balance, or other profile fields are part of this
decision.

## Constrained portfolio evidence

All rows use the C1 refit-core-plus-causal-S/R arm, dual +50-bps admission,
BCF MC1 priority, and the fixed 7x / 10%-slot portfolio auction.

### June–July 2026 selection slice

| Value-area source | Entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| Full seven fields | 465 | +197.74 | +91,948 | +167.77 | +120.76 | −31.48% |
| Without POC | 476 | +196.62 | +93,592 | +161.71 | +106.70 | −33.67% |
| Without VAH/VAL | 458 | +197.33 | +90,376 | +166.75 | +118.98 | −31.86% |
| Without HVN/LVN | 470 | +197.39 | +92,772 | +169.81 | +134.21 | −31.07% |
| Without value-area geometry | 470 | +189.44 | +89,036 | +162.17 | +110.23 | −31.45% |

HVN/LVN removal looked locally attractive: slightly more total contribution,
better maximum drawdown, and stronger worst-period metrics, at a negligible
EV/trade reduction.  This was a hypothesis, not a promotion decision.

### August 2026 sealed holdout

| Value-area source | Entries | Net EV/trade | Total net bps | Max DD |
|---|---:|---:|---:|---:|
| Full seven fields | 98 | **+277.61** | **+27,206** | **−4.71%** |
| Without POC | 94 | +203.73 | +19,150 | −27.49% |
| Without VAH/VAL | 96 | +203.68 | +19,554 | −27.49% |
| Without HVN/LVN | 92 | +203.61 | +18,732 | −27.49% |
| Without value-area geometry | 94 | +217.29 | +20,425 | −27.49% |

Every removal fails the holdout.  Deleting HVN/LVN gives up 74.0 bps/trade,
8,474 total bps, and 22.8 percentage points of maximum-drawdown quality
relative to the full context.  The other removals show the same direction.

## Conclusion

The appropriate compact contract remains:

```text
POC + VAH/VAL + HVN/LVN + inside-value-area + value-area width
```

Do not remove a group or add it to the canonical/live stack solely from this
short holdout.  A later untouched block should re-test the same leave-one-
group-out design before any simplification.

## Artifacts

* Full source/replay: `data_perp/artifacts/canonical_sr_profile_levels_c1_mc1_junjul_20260831_v2/` and `..._august_20260831_v1/`
* POC removal: `...profile_levels_without_poc...`
* VAH/VAL removal: `...profile_levels_without_vah_val...`
* HVN/LVN removal: `...profile_levels_without_hvn_lvn...`
* Value-area geometry removal: `...profile_levels_without_value_area_geometry...`
