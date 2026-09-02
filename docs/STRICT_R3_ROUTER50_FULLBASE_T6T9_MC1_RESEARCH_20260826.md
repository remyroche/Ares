# Strict-R3 Router-50 / Full-Base / T6–T9 / MC1 Research Receipt

**Published:** 2026-08-26  
**Status:** `OFFLINE_RESEARCH_CHALLENGER — NOT LIVE`  
**Scope:** long-only; strict-OOF, canonical reconciled rich-policy outcomes; April–July 2026 evaluation.

This document supersedes the preliminary router-50 metrics whose run contracts
described the base as route-trained. The final receipt instead makes the
intended architecture explicit: it **reuses immutable, full-population,
strict-OOF enhanced-base coordinates**, applies a target-free timestamp-local
top-50% route, and only then refits T6/T9 and the two MC1 mappers.

It does not change the deployed trader or any live bundle.

## Frozen research contract

```text
full causal candidate universe
  -> strict-OOF full-trained enhanced three-way base
  -> router top-50% at each timestamp
  -> T6 / T9 correction heads, with router ranks as inputs
  -> Current and BCF MC1 mappers, with router ranks as inputs
  -> dual MC1 expected-EV floor
  -> one global constrained chronological portfolio
```

### Base

The base is an equal common-bps mean of three independently strict-OOF
coordinates, each mapped chronologically to the same rich-policy-net scale:

```text
enhanced base = mean(
  Strict-R3 R3 P(clear) - 0.5 P(adverse),
  direct policy-conversion efficiency,
  direct policy-conversion timing
)
```

All components use the frozen 120-field causal base contract. The base is
trained on the full eligible causal population; it is **not** trained only on
the routed half. The router is a downstream selection gate based on the
strict-OOF P8u economic-recall model and retains
`max(1, ceil(0.50 × candidates at the timestamp))`.

The route is target-free at score time. Router outputs have no numerical base
authority in this selected arm; they are consumed only by the downstream T6,
T9, Current-MC1 and BCF-MC1 models.

### T6/T9 correction layer

Only these two heads are used:

| Head | Physical slot | Purpose |
|---|---|---|
| T6 | `cap80_ordinary` | Ordinal rank-error correction |
| T9 | `cap120_equal_month` | Ordinal exit-quality correction |

They refit with three preceding resolved calendar months and a 28-day reserve.
Their inputs contain the full enhanced-base context plus the target-free router
rank coordinates. They are correction models, not standalone alpha models.

### Dual MC1 admission

Current and BCF are two separately prequential expected-policy-net mappers.
They are fitted with at most three preceding scored calendar months, only on
labels resolved before the held month. The router ranks are available as
additional target-free MC1 inputs. A trade is admissible only when **both**
mapped values clear the same selected EV floor. The portfolio priority is the
BCF mapped EV, not the raw score.

The outcome is the canonical reconciled rich policy: trailing profit, smooth
capital protection, 100-bps cost exactly once, and the frozen constrained
global auction.

## Base ablation: where routing belongs

All four arms evaluate the identical strict-OOF router top-50% at each
timestamp and the same rich-policy labels. “Top-10 precision” means the
timestamp-average fraction of selected rows with realised policy net above
+50 bps.

| Base variant | Top-10 EV | Top-10 precision | Top-2 EV | Top-2 constrained EV/trade | Max DD | Daily Sortino |
|---|---:|---:|---:|---:|---:|---:|
| B0, full trained then routed | +40.41 | 49.78% | +79.64 | +66.26 | −35.47% | 55.54 |
| **Enhanced three-way, full trained then routed** | **+72.42** | 47.96% | **+137.76** | **+120.76** | **−17.38%** | **380.73** |
| Enhanced three-way, trained only on routed rows | +48.82 | 43.82% | +106.88 | +85.09 | −31.99% | 91.36 |
| Routed-row base with router inputs | +48.56 | 44.37% | +106.14 | +90.27 | −31.68% | 73.14 |

The full-trained enhanced base passes the predeclared precision floor (within
three percentage points of B0) and wins on the decisive timestamp-local
top-2, constrained top-2, drawdown and Sortino criteria. The evidence does
not support refitting the base on routed rows.

## T6/T9 and MC1 input ablations

With the selected full-trained enhanced-base coordinate fixed, feeding the
router ranks to the correction layer improves its timestamp-local top-2 score
quality while preserving top-10 precision. The router-aware T6/T9 arm is
therefore selected before MC1.

At the MC1 stage, adding the same target-free router coordinates is beneficial
at the useful admission floors. The following are the final, clean,
fully-refitted router-aware results:

| Dual MC1 floor | Portfolio entries | Net EV/trade | Total net bps | Worst month | Worst week | Max DD | Daily Sortino |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 bps | 3,080 | +139.95 | +431,055 | +115.17 | +80.69 | −22.79% | 667.56 |
| **40 bps** | **2,829** | +154.91 | **+438,254** | +130.58 | **+95.42** | −13.27% | 1,651.77 |
| **50 bps** | 2,616 | **+164.82** | +431,161 | **+134.25** | +89.58 | **−13.25%** | **2,204.33** |

The results create a real decision trade-off rather than one universal
winner:

- **40 bps** maximises total realised net and has the best worst week.
- **50 bps** maximises per-trade EV, worst-month EV and risk-adjusted score.
- 30 bps is dominated on drawdown and risk-adjusted measures.

No live threshold is changed by this receipt. A threshold must be frozen
before a later untouched-forward validation.

## Matched B0/T6/T9 control (April–July subset)

The previously completed B0/T6/T9 control began in February. To avoid
confounding the comparison with its February–March data, its completed
decision ledger is restricted below to the same April–July calendar window.
The selected rich-policy outcomes and portfolio constraints are unchanged,
but these rows are still a **matched research control**, not a claim of
inference parity with the deployed stack.

| Arm | Entries | Net EV/trade | Total net bps | Worst month | Worst week |
|---|---:|---:|---:|---:|---:|
| B0 + T6/T9 control, dual 50 bps | 1,847 | +166.97 | +308,398 | +150.99 | +106.96 |
| Router-50 full-base + T6/T9 + MC1, dual 40 bps | 2,829 | +154.91 | +438,254 | +130.58 | +95.42 |
| Router-50 full-base + T6/T9 + MC1, dual 50 bps | 2,616 | +164.82 | +431,161 | +134.25 | +89.58 |

Relative to the B0 control, router-50/50 adds 769 constrained entries and
+122,763 bps total, with only −2.16 bps/trade. Router-50/40 maximises total
contribution by adding 982 entries and +129,856 bps, at a larger
−12.06-bps/trade precision cost. The router challenger has lower worst-period
averages in this matched window, so it is not a promotion result.

## Causality and lineage receipt

The completed final receipt is:

```text
data_perp/artifacts/
  strict_r3_router50_fullbase_t6t9_meta_mc1_router_final_20260826_v1/
```

Its correctness report proves:

- base coordinates were copied from the immutable full-trained enhanced-base
  source and then restricted to exact top-50% router membership;
- 584,280 routed target-free base rows were preserved;
- T6/T9 use only `cap80_ordinary` and `cap120_equal_month`;
- all Current and BCF target-free score panels contain no outcome columns;
- Current/BCF candidate identities are exact matches;
- labels are joined only after score production;
- router ranks are model inputs only for T6/T9 and MC1, never post-score
  selection leakage;
- the evaluation window is April–July 2026, despite legacy filenames which
  label the broad ledger `2026_marjul`.

## Reproduction references

| Purpose | Path |
|---|---|
| Final selected stack runner | `scripts/run_strict_r3_router_routed_base_stack.py` |
| Frozen full-base router source | `scripts/materialize_strict_r3_router50_fullbase_source.py` |
| Base ablation report | `scripts/report_strict_r3_router50_base_metrics.py` |
| T6/T9 input ablation report | `scripts/report_strict_r3_router50_meta_metrics.py` |
| MC1 input/floor report | `scripts/report_strict_r3_router50_mc1_metrics.py` |
| Router OOF source | `data_perp/artifacts/strict_r3_economic_recall_router_selected_p8u100c250_h6_uniform_oof13_202607_20260825_v1/` |
| Enhanced base source | `data_perp/artifacts/strict_r3_enhanced_base_threeway_targetfree_20260824_v1/` |
| Rich-policy ledger | `data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet` |
| Final clean receipt | `data_perp/artifacts/strict_r3_router50_fullbase_t6t9_meta_mc1_router_final_20260826_v1/` |

## Next required work

1. Run the router → base → T6/T9 → MC1 → portfolio waterfall on the same
   April–July identities, including an oracle diagnostic that is explicitly
   excluded from selection.
2. Predeclare either the 40-bps total-EV objective or the 50-bps
   precision/risk objective; do not select after observing later data.
3. Validate the frozen selected contract on a temporally later untouched
   period before any inference or live-stack change.
