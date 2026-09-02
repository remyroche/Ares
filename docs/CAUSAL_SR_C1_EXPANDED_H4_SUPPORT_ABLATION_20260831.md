# Causal SR C1 Expanded H4 Support Ablation — 2026-08-31

## Decision

Do **not** promote an H4 continuation overlay or alter the live/canonical
parent policy from this result.  The larger causal sample fixes the prior
near-zero label pathology and demonstrates that an F25 reduction is
economically indistinguishable from F45 *within this H4 experiment*.  However,
every H4 arm remains below the unchanged rich exact-1m parent policy on the
matched June--August 2026 portfolio replay.

F25 is therefore a valid computational simplification candidate for a future
H4 experiment, not a deployed feature contract.  No feature/model/live bundle
is superseded by this document.

## Immutable expanded population

The target-free C1 route uses dual BCF/current MC1 expected EV >= +40 bps,
before reading any future one-minute path.  It joins outcome paths only after
selection.

| Item | Value |
|---|---:|
| Route candidates | 20,610 |
| Valid exact 1m paths | 20,557 |
| Invalid paths after routing | 53 |
| Required source parts fully readable | 10,739 |
| Portfolio entries before H4, Jun--Aug 2026 | 682 |
| Exact state rows | 481,307 |
| Context materialised state rows | 481,303 |
| Deterministic MFE-ready label states | 27,453 |
| MFE-ready labelled candidates | 8,386 |

The 53 unavailable exact outcomes never affect route membership, H4 training,
or portfolio capacity.  One symbol contributed four states without a readable
15-minute context source; those states fail closed.

The exact 1m source repair receipt is
`data_perp/artifacts/causal_sr_c1_expanded_h4_source_audit_t40_20260831_v1/quarantine_receipt.json`.
It moved exactly 242 verified corrupt parts to reversible quarantine.  The
post-recovery full-Parquet read audit found zero unreadable parts.

## Causality contract

- Source route: target-free, dual MC1 >= 40 bps before exact path access.
- Parent policy: frozen rich exact-one-minute policy, +5-minute entry and
  100-bps cost exactly once.
- H4 label: latched activation-50/giveback-20 exact continuation net minus
  unchanged exact parent net.
- State sample: the first, evenly spaced, and last available MFE-ready
  completed 15-minute states, capped at four states per candidate.  Membership
  uses no label or outcome value.
- A held month sees only labelled candidates from the preceding nine calendar
  months with `policy_label_available_ts < held_month`.
- H4 authority: strictly-positive predictions at a completed MFE-ready state
  may latch a 20% giveback tightening for the next interval.  It cannot loosen
  a stop, promote a candidate, change sizing, or act on the same bar.
- Portfolio: unchanged global chronological auction and parent constraints.

The prior sparse-label defect is repaired: labels are 56.78% zero, 34.16%
positive, and 9.05% negative.  This replaces the earlier 99.03% zero target.

Strict-prior label support remains substantial but is not uniform because no
immutable C1 panel exists for January--May 2026:

| Held month | Prior label states | Prior candidates |
|---|---:|---:|
| 2026-06 | 13,289 | 4,084 |
| 2026-07 | 10,846 | 3,328 |
| 2026-08 | 8,061 | 2,496 |

The evaluation is therefore June--August 2026 only.  It is research evidence,
not an untouched promotion test.

## Feature and geometry arms

F45 is the frozen May-2026 C4 contract: nine mandatory position-state fields
plus 36 contextual fields.  F35 and F25 retain those nine mandatory fields and
remove 10/20 contextual fields by prior-window availability, robust scale and
Spearman redundancy.  That pruning reads no label, prediction, outcome or
portfolio field.  D4 is the original L2 H4 geometry (depth 4, 15 leaves,
minimum child 5%, L2 20, LR .025, 420 trees); D2 is a support-first compact
control (depth 2, 7 leaves; all other settings unchanged).

## Matched portfolio result

| Arm | Entries | Net bps/trade | Total net bps | Max DD | Sortino | Delta bps/trade vs parent |
|---|---:|---:|---:|---:|---:|---:|
| Parent exact 1m | 682 | +118.77 | +80,998.84 | -10.82% | 0.6684 | — |
| F45 / D4 | 682 | +117.81 | +80,344.59 | -10.82% | 0.6760 | -0.96 |
| F35 / D4 | 682 | +117.82 | +80,355.37 | -10.82% | 0.6760 | -0.94 |
| F25 / D4 | 682 | +117.82 | +80,352.28 | -10.82% | 0.6760 | -0.95 |
| F45 / D2 | 682 | +115.24 | +78,591.53 | -10.82% | 0.6632 | -3.53 |

The feature-removal conclusion is narrow but useful: F25 is only -0.01
bps/trade from F45/D4, while F35 is +0.02 bps/trade from F45/D4.  Thus 10--20
of the non-mandatory fields can be removed without meaningful incremental loss
in this experiment.  This does **not** overcome H4's -0.94/-0.95 bps/trade
loss versus the untouched parent.

| Month | Parent | F45/D4 | F35/D4 | F25/D4 |
|---|---:|---:|---:|---:|
| 2026-06 | +183.37 | +184.24 | +184.26 | +184.25 |
| 2026-07 | +98.39 | +97.46 | +97.48 | +97.48 |
| 2026-08 | +78.36 | +74.81 | +74.79 | +74.79 |

Although D4 schedules 1,229/1,222/505 enabled states in June/July/August,
only 16 of the 682 accepted positions receive a different exit.  June gains
+169.9 bps in total, but July loses -319.4 and August loses -504.8.  This
is a time-transfer problem rather than an insufficient-field problem.

## Relevant artifacts

- Route: `data_perp/artifacts/causal_sr_c1_expanded_h4_route_t40_20260831_v1`
- Exact parent panel: `data_perp/artifacts/causal_sr_c1_exact1m_parent_t40_expanded_20260831_v1`
- H4 expanded ablation: `data_perp/artifacts/causal_sr_c1_h4_expanded_support_20260831_v1`
- Runner: `scripts/run_causal_sr_c1_h4_expanded_support_ablation.py`
