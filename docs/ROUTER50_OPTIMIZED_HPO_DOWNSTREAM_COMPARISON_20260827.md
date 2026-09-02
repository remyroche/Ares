# Optimized Router50: corrected HPO and downstream comparison

Status: offline research only. The optimized router is **not promoted**.

## Question

Does the optimized 30-field Router50 improve the retained P8u router when the
only downstream change is route membership?

The answer is split deliberately:

1. **Router layer:** yes, materially and consistently.
2. **Base → Meta → dual-MC1 → constrained portfolio:** no. The improvement is
   not preserved at the canonical 50-bps admission gate.

## Fixed comparison contract

Both arms use the exact same point-in-time candidate identities, the same
30-field causal feature contract, strict prequential labels, 28-day reserve,
the same E/T Base structure, the same two Meta heads (cap80 ordinary and
cap120 equal-month), the same separate Current and BCF MC1 maps, the same
dual-MC1 admission, rich-policy labels, and one chronological portfolio.

The router has **only boolean authority**: it retains the top 50% of finite
candidates inside each exact UTC decision timestamp. Its numeric score is not
an input to Base, Meta, or MC1. Consequently, the result isolates the economic
effect of changing the routed training/inference population.

## Corrected optimized HPO

Frozen config:
[`strict_r3_router50_optimized_u50p050c300_hpo_weekly_20260827_v1.json`](../config/strict_r3_router50_optimized_u50p050c300_hpo_weekly_20260827_v1.json)
(`SHA-256 a195f179b6e6f98e7923262c2928519c6d3c811cab46d8099fb43795a4e085b0`).

The prior selection shortcut was replaced with the declared timestamp-local
objective:

```text
utility = (clip(max(policy_net_bps - 50, 0), 0, 300) / 300)^0.5
S_router = 0.70 × R50_utility + 0.15 × recall(net > 50) + 0.15 × recall(net > 100)
S_stable = weekly mean(S_router within Q20–Q80) + 0.5 × mean(Q15, Q10, Q5)
```

The optimized candidate is `U50_p050_c300`, `positive_125`, native
`rank_xendcg`, exact-timestamp query, route Top-50%. It won the 12 isolated
HPO shards on the three frozen validation eras (2025-11, 2026-03, 2026-07):

| Metric | P8u anchor | Optimized #1732 | Delta |
|---|---:|---:|---:|
| `S_stable` | 1.11985 | 1.18431 | +0.06446 |
| Top-50 utility recall | 0.72783 | 0.79497 | +6.71 pp |
| Recall net > 50 bps | 0.71401 | 0.79020 | +7.62 pp |
| Recall net > 100 bps | 0.78358 | 0.83794 | +5.44 pp |
| Worst validation-era `S_router` | 0.73447 | 0.76170 | +0.02723 |

The frozen winning ranker is depth 4, 15 leaves, learning rate 0.06256,
minimum-child fraction 0.00485 with an 800-row floor, feature fraction 0.7447,
subsample 0.7632, L1 0.00498, L2 21.8484, max-bin 127, truncation 12, and
early stopping after 30 rounds. It uses a six-month train window, 28-day
prequential reserve and 120,000-row cap.

## Router result

This is a target-free score comparison over the common July-2025 through
July-2026 score ledger (9,459 decision timestamps, 752,571 Top-50 selections
per arm). Policy outcomes are joined only after route membership has been
computed.

| Router metric | P8u | Optimized | Delta |
|---|---:|---:|---:|
| Top-50 utility recall | 0.75200 | 0.78244 | +3.04 pp |
| Recall net > 50 bps | 0.73144 | 0.77520 | +4.38 pp |
| Recall net > 100 bps | 0.80912 | 0.82973 | +2.06 pp |
| `S_router` | 0.75748 | 0.78845 | +0.03097 |
| Weekly `S_stable` | 1.10604 | 1.15421 | +0.04817 |
| Worst weekly `S_router` | 0.65920 | 0.69224 | +0.03304 |
| Q25 weekly `S_router` | 0.71165 | 0.76848 | +0.05683 |

The optimized router improves `S_router` in every one of the 13 held score
months. Its Top-50-only substitutions are still economically weak in absolute
terms, but are less bad than P8u-only substitutions (`−92.46` versus `−99.67`
net bps/row) and capture substantially more opportunity utility (17,126 versus
9,924 utility units). This is valid router evidence, not a claim of portfolio
improvement.

## Downstream result: April–July 2026

The downstream evaluation is a separate, matched four-month OOS replay. It was
not used to select the winning HPO shard, except that July also belongs to the
historical HPO era set; it is therefore **not untouched promotion evidence**.

### Timestamp-local layer economics

All figures are average canonical rich-policy net bps after selecting Top-k
within each timestamp, then averaging timestamps.

| Layer / Top-k | P8u | Optimized | Delta |
|---|---:|---:|---:|
| Base Top-1 | +181.82 | +180.19 | −1.63 |
| Base Top-2 | +156.96 | +151.53 | −5.43 |
| Base Top-5 | +104.43 | +101.17 | −3.26 |
| Base Top-10 | +62.29 | +60.44 | −1.85 |
| BCF Meta Top-1 | +194.41 | +193.65 | −0.76 |
| BCF Meta Top-2 | +162.67 | +163.67 | +1.00 |
| BCF Meta Top-5 | +108.15 | +107.87 | −0.28 |
| BCF Meta Top-10 | +62.95 | +62.83 | −0.12 |

The Current Meta Top-1 improves (+11.65 bps), but it deteriorates at Top-2,
Top-5 and Top-10. Thus no later score layer provides a broad, reliable transfer
of the router-layer gain.

### Dual-MC1 admissions before portfolio constraints

| Dual-MC1 threshold | P8u admissions | P8u realized EV | Optimized admissions | Optimized realized EV | Delta admissions | Delta EV |
|---|---:|---:|---:|---:|---:|---:|
| 30 bps | 15,146 | +126.06 | 16,628 | +120.79 | +1,482 | −5.27 |
| 40 bps | 12,719 | +143.51 | 13,880 | +138.08 | +1,161 | −5.43 |
| 50 bps | 10,806 | +160.48 | 11,433 | +156.26 | +627 | −4.21 |

The optimized router creates more MC1 admissions, but their realized policy EV
is lower. This is the decisive transmission failure.

### One chronological constrained portfolio

| Gate | Arm | Entries | Net EV / trade | Total net bps | Worst month | Worst week | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| 30 | P8u | 3,404 | +133.59 | +454,752 | +102.79 | +70.25 | −27.41% |
| 30 | Optimized | 3,344 | +133.57 | +446,644 | +111.05 | +63.72 | −25.18% |
| 40 | P8u | 3,203 | +138.51 | +443,640 | +107.39 | +60.13 | −26.24% |
| 40 | Optimized | 3,145 | +138.72 | +436,266 | +114.64 | +71.30 | −27.38% |
| 50 | P8u | 2,943 | +151.58 | +446,102 | +120.81 | +74.04 | −20.21% |
| 50 | Optimized | 2,877 | +148.49 | +427,219 | +122.08 | +71.03 | −27.38% |

At the canonical 50-bps gate, the optimized arm has 66 fewer constrained
entries, −3.09 bps/trade and −18,883 total net bps versus P8u. Its slightly
better worst month does not compensate for lower aggregate EV, a worse worst
week and a 7.17-point deeper maximum drawdown.

## Decision

Keep **P8u** as the retained router. Freeze optimized #1732 as a
router-layer-only research challenger, not as an inference or live change.

The next legitimate research question is not another HPO sweep. It is why the
optimized router’s additional valid utility does not translate through E/T,
the two Meta heads, and the dual-MC1 admission map. Test that as a held-out
mechanism study with fixed router #1732 rather than tuning it again on these
same months.

## Artifacts

- Optimized router config: `config/strict_r3_router50_optimized_u50p050c300_hpo_weekly_20260827_v1.json`
- P8u target-free router ledger: `data_perp/artifacts/strict_r3_router30_final_matched_p8u_202507_202607_20260827_v3`
- Optimized target-free router ledger: `data_perp/artifacts/strict_r3_router30_optimized_u50_hpo1732_matched_202507_202607_20260827_v1`
- Matched comparison tables: `data_perp/artifacts/strict_r3_router30_optimized_u50_hpo1732_comparison_20260827_v6`
- Comparison runner: `scripts/report_strict_r3_router_hpo_comparison.py`
