# E2/H4 Giveback-20 exact release gate — 2026-09-01

## Decision

**Do not activate E2 + H4 Giveback-20 in the exchange-writing stack.**

The September no-order candidate bundle is source- and inference-parity
correct, but its H4 continuation controller does not clear the exact-path
economic gate.  The result is intentionally a release rejection, not a
parameter-selection opportunity.

## Contract repaired before evaluation

The earlier September candidate trained the H4 model on the historical
activation-only advantage panel while applying both:

- 50% earlier trailing activation; and
- 20% tighter trailing giveback.

That is a model/action mismatch: activation-only advantage does not estimate
the value of the Giveback-20 action.  The new producer rejects this mismatch.
It requires the exact counterfactual label:

```text
latched_activation50_giveback20_advantage_bps
= net(policy with the fixed H4 next-interval action)
 - net(unchanged rich parent policy)
```

The exact path materialiser also stores `MC1_expected_bps=0` by design so
that MC1 cannot affect an exit-path counterfactual.  The refit producer now
joins the immutable target-free route to restore the causal BCF MC1 value for
the H4 eligibility gate and H4 feature.  It neither derives that value from a
future outcome nor lets it affect the label.

## September no-order bundle

| Item | Value |
|---|---|
| Bundle | `strict_r3_p8u_e2_h4_live_parity_bundle_20260901_v2_action_aligned` |
| Cutoff | 2026-09-01 00:00 UTC |
| Training window | 2026-05-01 through 2026-08-31, labels resolved before cutoff |
| E2 pair labels | 1,010 |
| H4 action-aligned states used | 1,665 |
| H4 target | `latched_activation50_giveback20_advantage_bps` |
| Runtime authority | H4 prediction >= 0; 50% earlier activation and 20% tighter giveback, next completed 15-minute interval only |
| Bundle status | `SEALED_NO_ORDER_LIVE_PARITY_CANDIDATE` |

The target-free inference audit on 2026-08-15 is exact:

| Check | Result |
|---|---:|
| E2 H0 prediction delta | 0.0 |
| E2 H3 prediction delta | 0.0 |
| H4 prediction delta | 0.0 |
| Outcome columns read by audit | 0 |
| Exchange/order calls | 0 |

The audit is not an economic test; it proves only scoring parity and correct
input lineage.

## Strict-prior exact one-minute economic test

The action-aligned L1 replay uses the same H4 geometry as the September
bundle: L1 mean, depth 4, 15 leaves, 5% child-support floor, L2 regularisation
20, learning rate 0.025, 420 trees.  Each held month uses at most nine earlier
calendar months and only labels resolved before that month.  It uses:

- target-free C1 dual-40 route;
- rich parent policy, entry at decision +5 minutes;
- exact one-minute exits and a single 100-bps cost;
- unchanged global chronological portfolio constraints; and
- the fixed activation-plus-Giveback-20 next-interval action.

| Arm | Portfolio entries | Net EV/trade | Total net bps | Delta vs parent |
|---|---:|---:|---:|---:|
| Exact rich parent | 682 | +118.77 | +80,998.84 | — |
| **H4 L1, F45/d4** | 682 | **+115.31** | **+78,641.36** | **−3.46 bps/trade; −2,357.48 bps** |

The compact F35, F25 and shallow F45/d2 variants are all similarly negative.
This confirms that the negative result is not an activation-only label bug,
nor a feature-contract sparsity issue.

## Why the old +23/+25 bps H4 evidence cannot promote this controller

The historical headline was measured on a different 15-minute,
source-valid policy substrate, with a larger selected population and broader
activation-only controller.  It also benefited partly from portfolio-capacity
recycling.  On the common exact-one-minute / +5-minute-entry substrate,
archived activation-only H4 retains only a modest effect; the actual
Giveback-20 action does not port.  See
[matched 2x2 reconciliation](CAUSAL_SR_E2_C1_H4_MATCHED_2X2_20260831.md).

## Live-execution conclusion

The code verifies that an H4-modified trailing threshold is passed through the
stored full-size exit VWAP/impact conversion before a native protective stop
is placed.  This does not cure an economically negative controller.

For a long, the verified conversion is:

```text
H4 policy stop
  -> native exchange trigger = policy stop / (1 - stored exit-VWAP impact)
  -> expected full-size exit VWAP = native trigger * (1 - stored impact)
  = H4 policy stop
```

The stored impact is the directional full-size exit impact captured for that
position.  It is deliberately retained across a stop amendment, rather than
cancelling protection to fetch a fresh book; the actual Kraken fill, expected
VWAP, impact and policy threshold are then recorded in close reconciliation.
The focused `test_h4_giveback_stop_is_converted_through_the_persisted_exit_vwap_contract`
asserts this bridge.  Thus, H4 does affect the executable stop and its
spread/impact/VWAP accounting; it is not a paper-only policy adjustment.

No exchange-writing session, live model hash, candidate admission, or exit
policy has changed from this result.  E2/H4 Giveback-20 requires a new,
separately predeclared exact-path controller that improves an untouched period
before a future activation request can be considered.

## C1 S/R package status

The September C1-LVA source bundle is sealed and its source-only unit test
passes.  It has no candidate, MC1, portfolio, exchange, or order authority by
design.  The current exchange-writing gateway has no C1 feature materialiser
or MC1 consumer, so C1 cannot yet be claimed as part of a live successor.

That is deliberately fail-closed: adding C1 would require a hash-bound
completed-bar S/R/profile materialiser, a current C1-augmented MC1 refit, and
an exact target-free inference/replay parity receipt.  Those release steps are
not performed while the required H4 Giveback-20 component itself fails its
economic gate.

The S/R materialiser must also preserve its state semantics.  Its zone
lifecycle, resolved-touch history and parent-strength statistics are
stateful; a short rolling reconstruction is not equivalent to the historical
producer.  A valid live implementation must either persist that state with an
append-only receipt or replay deterministically from the frozen source origin
on every refit.  It must not substitute a shorter warm-up merely to reduce
runtime.

### C1 append-state repair (no-order)

The first of those C1 requirements is now implemented, but it has **not**
activated C1 in the exchange-writing gateway.  The new
`CausalSRC1AppendState` contract persists, per symbol:

- active candidate levels and merged zone lifecycle;
- bounded per-zone reaction / break histories;
- long-lived parent strength and break aggregates;
- unresolved eight-hour interactions, keyed by their resolution timestamp;
- dynamic-level deduplication state and counters; and
- a hash-bound 45-day completed-bar tail required by the structural recurrence;
  and
- a cumulative SHA-256 identity chain from the first locally available source
  bar through the latest processed bar.

The append operation rejects a changed historical bar, a non-15-minute gap,
or a missing exact completed decision bar.  A supplied overlap must be either
the full processed source history (validated by the cumulative identity chain)
or wholly inside the retained 45-day tail (validated bar-for-bar); a partial
off-tail overlap fails closed.  It resolves a pending interaction only at its
own eight-hour horizon; no outcome is attached to a candidate snapshot.  The
profile/value-area component has a declared bounded 21-day state (plus
session/week state), so it is reconstructed solely from completed bars and
merged backward/as-of; retained C1-LVA fields do not require the optional
OI-at-price values.

Focused tests show an initial source replay followed by save/load/append gives
an exactly identical later target-free S/R snapshot to one uninterrupted
causal replay; both an in-tail rewrite and an unverifiable off-tail overlap
fail closed.  This is a technical parity test, not an economic promotion
result.  The first v1 bootstrap predates this cumulative-chain schema and is
preserved only as invalid diagnostic evidence.  The fresh v2 bootstrap runs
from the 2025-01-01 source origin for the frozen live universe; it must then be
checked against archived C1 snapshots and paired with a newly sealed
C1-augmented MC1 bundle and exact portfolio replay before C1 can enter a
successor release.

Relevant no-order components:

- `extreme_price_movements/inference/causal_sr_c1_state.py`;
- `scripts/bootstrap_causal_sr_c1_append_state.py`; and
- `tests/test_causal_sr_ontology_contract.py`.

## Artifacts

- [Action-aligned bundle](../data_perp/artifacts/strict_r3_p8u_e2_h4_live_parity_bundle_20260901_v2_action_aligned/bundle_manifest.json)
- [Target-free parity receipt](../data_perp/artifacts/strict_r3_p8u_e2_h4_inference_replay_parity_20260901_v2_action_aligned/receipt.json)
- [Strict-prior L1 replay](../data_perp/artifacts/causal_sr_c1_h4_expanded_support_l1_action_aligned_20260901_v1/portfolio_summary.parquet)
