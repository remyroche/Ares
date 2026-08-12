# TP6/SL4 conditional cluster/path layer handoff

## Scope

Long-only conditional path/cluster layer on top of the TP6/SL4/H12 base contract.
The layer discovers clusters from frozen signed/absolute structural-family
contributions, gives each row a soft membership in each cluster, and fits one
residual target per cluster:

```text
cluster_residual[c] = soft_membership[c] × (exact_net_bps − base_expected_bps)
```

`base_expected_bps` is a train-only cost-aware bps map. It is never replaced by
a percentile rank.

## Implementation

- Representation utilities: `extreme_price_movements/conditional_cluster_residual.py`
- Long-only runner: `scripts/run_tp6_sl4_conditional_cluster_residual.py`
- Canonical meta-path materializer: `scripts/materialize_tp6_sl4_canonical_meta_paths_20260808.py`
- Full causal meta-pool materializer: `scripts/materialize_tp6_sl4_canonical_meta_pool_20260808.py`
- Strict OOF regime/transition pool augmentation: `scripts/augment_tp6_sl4_canonical_meta_pool_with_oof_regime.py`
- Frozen correction-weight ablation: `scripts/ablate_tp6_sl4_cluster_correction_weight.py`
- Final strict diagnostic: `data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260807_v6/`
- Canonical Base+Consensus handoff: `data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/canonical_cluster_input_2025.parquet`
- Canonical meta-path contract: `data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260808_bands4_v1/meta_family_contract.json`
- Canonical full-pool evaluation: `data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260808_bands4_fullpool_v1/`
- Regime-augmented full-pool evaluation: `data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260811_regime_fullpool_v1/`
- Extended-history family contract/replay: `data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1/` and `data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260811_extended_regime_v1/`
- Causal reliability/OOD gate overlay: `scripts/run_tp6_sl4_cluster_reliability_gate.py` and `data_perp/artifacts/tp6_sl4_cluster_reliability_gate_20260811_v1/`
- Frozen cross-fold cluster residual replay: `scripts/run_tp6_sl4_frozen_cluster_residual.py` and `data_perp/artifacts/tp6_sl4_frozen_cluster_residual_20260812_v1/`
- OOF per-cluster reliability learner: `scripts/run_tp6_sl4_frozen_cluster_reliability.py` and `data_perp/artifacts/tp6_sl4_frozen_cluster_reliability_20260812_v1/`
- Earlier-era stable-contract transport audit: `data_perp/artifacts/tp6_sl4_frozen_cluster_transport_20260812_v1/`

The runner supports both base and meta path contracts through `--family-prefix`;
the default is `base_structural_family__`. A meta-family store can be supplied
with a predeclared prefix once its OOF contribution artifact exists.

## Leakage contract

The default path policy is `meta_partition=test`. This is deliberate: rows
marked `meta_train` are not accepted as OOF path features for the same row.
Clusters, CMI selection, quantile condition edges, and residual models are fit
on rows before the held-out month. Missing path rows remain in coverage audits;
they are never encoded as zero evidence.

Correctness checks are in `tests/test_conditional_cluster_residual.py` and the
run-level report is `correctness_test_report.json`.

## Meta/context pool

The pool is assembled from all configured meta, regime, residual, context,
transition, reliability, uncertainty, and market-cross-sectional keys in
`config.py`, intersected with the available family/ledger schema. The strict
run had 103 available fields before CMI selection, including regime state
probabilities, entropy, transition onset, state age, OI/funding, breadth,
cross-asset, liquidity, and structural-price fields. Each cluster selected at
most 16 fields using train-only active/inactive weighted binned MI.

The full pool is audited before reduction; selection does not widen the
inference contract. The original canonical source store provides 104 usable
numeric causal fields before CMI selection. A strict OOF hourly sidecar now
adds 209 finite five-state, transition, geometry, continuous-rank, and
relationship-break fields for the entire 2025 population, bringing the
regime-augmented pre-selection pool to 313 fields. State IDs and provenance
remain audit-only; fold-local padded centroid distances are excluded rather
than imputed.

## Strict diagnostic result

The available historical family store covers 81,293/255,823 base rows (31.8%)
under the strict OOF/test-partition policy. It supplies only one usable
chronological training/held-out transition in this panel: May-trained →
June-held-out. March–April are absent from the source sidecar, not treated as
zero path evidence.

On the covered June held-out rows:

| Arm | Top 0.5% net | Top 1% net | Top 5% net | Rank IC |
|---|---:|---:|---:|---:|
| Base bps anchor | −111.22 | −116.36 | −145.88 | −0.0612 |
| Cluster-only residual | −94.39 | −124.77 | −145.23 | −0.0600 |
| Cluster + context residual | −77.54 | −92.81 | −140.95 | −0.0495 |

The conditional context arm improves the covered diagnostic relative to the
base control at the narrowest tails, but remains economically negative. This
is not promotion evidence: the panel/base itself is a weak historical control
and the path coverage is incomplete.

The held-out condition table is
`cluster_condition_economics.parquet`. It reports train-edge-binned
membership-weighted net/residual means and active-minus-inactive deltas for
each selected context field. The June report shows, for example, that
`loc_swing_range_pos_24`, `trend_pct_mkt_resid`, market return, funding, order
book, and regime probabilities materially change cluster-conditioned residuals;
their signs and economics must be rechecked on a broader canonical panel.

## Canonical meta-path materialization

The canonical downstream replay now persists:

- `base_expected_bps`: train-only isotonic TP6/SL4 net anchor;
- R3 probabilities and base score;
- causal context fields;
- consensus/residual rank outputs;
- exact labels for evaluation;
- `label_available_ts` for causal joining.

The canonical residual LambdaRank head is refit before each held 2025 month
with the exact TP6/SL4 residual target and 4-hour UTC × side queries. Its strict
OOF meta-path artifacts cover all 10,224 long 2025 rows. Raw leaf tokens remain
inside per-month strict artifacts; the handoff matrix contains only structural
family memberships and contribution shares. The selected four-band taxonomy
has 83 recurrent families and covers 43.8% of native meta-path absolute
contribution mass. The original ten-band/20-family control covered 10.6%; the
four-band representation is retained for the canonical comparison, while the
remaining mass is explicitly tracked as unassigned rather than imputed.

The cluster runner accepts the canonical `__ts__` and `r3_meta_p_*` aliases,
uses the full available pool before train-only CMI reduction, and evaluates
global top-k ranking. Path and regime join coverage is 100% for every held
month with `meta_partition=test`. January has no prior canonical path fold and
is therefore retained in the coverage population but not used as a supervised
cluster test fold.

## Canonical results

Full-pool four-band result, 11 evaluated held months (February–December 2025):

| Arm | Top 0.5% net | Top 1% net | Top 2% net | Top 5% net | Top 10% net |
|---|---:|---:|---:|---:|---:|
| Base bps anchor | −36.16 | −68.77 | −3.13 | **+11.40** | −4.48 |
| Cluster-only correction | **+18.97** | −130.78 | +1.95 | +8.50 | −10.97 |
| Cluster + full-pool context | −3.00 | −137.31 | **+3.46** | +8.91 | −10.56 |

The cluster/context layer improves the narrowest top-0.5% tail by 33.2 bps
and makes March top-5 positive (+23.73 versus −32.67 bps for the base), but it
does not improve pooled top-5 or worst-month economics. It is therefore a
validated conditional diagnostic feature layer, not a promoted replacement for
the canonical base ranking.

The frozen correction-weight ablation shows that smaller corrections improve
the narrowest tail (cluster-only λ=0.25: +29.58 bps at top-0.5%) but still do
not beat the base at pooled top-5 (+4.21 versus +11.40). λ=0 is the base
control; no correction multiplier is promoted.

The regime-augmented replay selected regime/transition/continuous fields in
738 of 1,008 cluster-field instances (73.2%), confirming that they are being
used by the train-only selectors. It did not improve the economics: the
cluster-context top-5 result moved to +5.52 bps, with −50.53, −162.71, and
−159.53 bps in September, October, and August respectively. The complete
analysis is in
`docs/TP6_SL4_CONDITIONAL_CLUSTER_REGIME_AUGMENTATION_20260811.md`.

An extended-history replay then added strict path/base rows from April–November
2024 before scoring January–December 2025 (19 evaluated held months; the source
has no December 2024 rows). The taxonomy contained 100 recurrent families and
represented 38.5% of native path mass. Its pooled top-5 net results were
−7.74 bps for the base, −11.79 for cluster-only, and −20.68 for
cluster+regime/context. This confirms that the narrow 2025-only improvement is
not portable across a longer chronology.

## Remaining work before promotion

1. Test whether a contribution-mass-preserving path abstraction can raise
   recurrent coverage beyond 43.8% without admitting fold-local semantics.
2. Build a reliability/OOD gate for path corrections and require it to improve
   the extended-history top-5 and worst-month gates.
3. Promote only if the cluster/context correction improves pooled global net
   EV and worst-month economics; current evidence does not pass that gate.

The first predeclared reliability/OOD overlay has now been evaluated on the
19-month extended panel.  The best arm (cluster-only correction, represented
path mass ≥0.25, λ=0.50) produced −7.01 bps/trade at global top-5 versus
−7.74 bps for the base control.  It improved the number of positive months
from 11/19 to 12/19, but its worst month remained −175.31 bps and pooled net
EV stayed negative.  The overlay is therefore not promoted.  Gate coverage,
weights, and all monthly rows are persisted in the linked artifact; the next
step should be a learned prior-resolved path reliability/conversion target,
not another threshold sweep.

Before that learned reliability layer, the cluster identity contract itself was
repaired.  The earlier replay rediscovered `cluster_00`–`cluster_06` per fold,
so those names did not represent the same specialists over time.  The frozen
replay discovers five clusters once on 2024 development rows and reuses their
exact family memberships for all 2025 held months.  With the same 313-field
causal/meta pool exposed before train-only CMI, the stable per-cluster soft
residual layer reaches +6.81 net bps/trade at global top-5 versus +2.57 for
the base control, with rank IC 0.0652 versus 0.0636.  Monthly mean top-5 is
3.84 versus 2.58 bps and positive months are 7/12 versus 6/12, but the worst
month deteriorates to −186.99 bps.  This is a promising development arm, not
yet a promotion: it still needs untouched-era transport and a causal
reliability/conversion learner that predicts when each frozen cluster’s
correction should be trusted.

The OOF per-cluster reliability learner has now been run.  Its target is the
OOF soft-cluster residual minus the OOF cluster prediction, trained from
leave-one-month-out 2024 predictions and scored on 2025.  At α=0.75 it reaches
+10.39 net bps/trade at global top-5 (base +2.57; frozen cluster +6.81), but
the worst month is −204.67 bps and top-1 is −110.92 bps.  No strength passes
the combined pooled/worst-month/top-1 promotion gate.  Keep the reliability
outputs as diagnostic per-cluster features, not as a production correction.

An additional transport audit discovered a separate stable contract on
label-matured April 2024 rows only and evaluated May–November 2024.  The
frozen cluster arm reached −12.03 net bps/trade at top-5 versus +3.27 for the
base, despite a slightly better unweighted monthly mean (−12.41 versus
−23.08).  Its worst month was −122.32 versus −90.14.  This means the +6.81
top-5 result on the 2025 OOS period does not transport reliably to the earlier
era.  The cluster/reliability layer remains research-only.

The remaining native path mass was then exposed as a separate
`frozen_unassigned` soft residual head, using the same frozen 313-field
causal/meta pool and train-only CMI selection. It did not advance: 2025
top-5 net was +1.72 bps/trade versus +6.81 for frozen clusters (base +2.57),
and earlier-era transport was −14.49 versus −12.03 for frozen clusters and
+3.27 for the base. It improves the earlier-era monthly mean and worst month,
but fails the pooled promotion gates. It remains a diagnostic coverage
feature. See the [unassigned ablation](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_unassigned_ablation_20260812_v1/)
and [transport replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_frozen_unassigned_transport_20260812_v1/).
