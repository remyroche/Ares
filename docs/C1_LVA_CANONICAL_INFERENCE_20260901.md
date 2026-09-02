# C1-LVA canonical inference contract — 2026-09-01

## Status

**Canonical long-only C1 source and dual-MC1 input contract.**  The C1 source
and mapper remain hash-bound and no-order.  The canonical combined C0/C1
admission order is now documented separately in
[`C0_C1_AGREEMENT_TIER_CANONICAL_INFERENCE_20260901.md`](C0_C1_AGREEMENT_TIER_CANONICAL_INFERENCE_20260901.md):
`both-admitted -> C1-only -> C0-only`.  Exchange-write authority remains a
separate explicit release decision; it is not inherited from canonical
research or inference status.

This document supersedes the research-status decision in
[`CAUSAL_SR_C1_MAIN_CHALLENGER_20260831.md`](CAUSAL_SR_C1_MAIN_CHALLENGER_20260831.md).
That document remains an evidence archive only.

## Canonical flow

```text
complete target-free BCF and Current score coordinates
  -> append-only completed-15m C1 S/R state
  -> causal value-area source snapshot
  -> verified C1-LVA source heads
  -> independently fitted BCF and Current C1-MC1 EV maps
  -> BCF EV >= 50 bps AND Current EV >= 50 bps
  -> normal constrained portfolio auction, priority = BCF MC1 EV
  -> exact rich parent execution policy
```

The C1 source never creates candidates, sees outcomes, filters an unavailable
snapshot, applies the admission threshold, ranks the auction, or submits an
order.  If a C1 snapshot is unavailable, the candidate remains present with
`sr_snapshot_available=0`; the frozen mapper uses its train-only medians for
the missing C1 values.  A missing map vintage, daily shift, hash, ordered
input, or changed completed bar fails closed.

### Ordered mapper inputs

Six existing target-free score coordinates:

1. `final_score`
2. `base_rank42`
3. `conditional_consensus_rank`
4. `upstream`
5. `ordinary_shadow_consensus_rank`
6. `correctness_rank`

Then eleven causal C1 S/R/value-area summaries plus availability:

1. `sr_long_support_hold_strength`
2. `sr_long_resistance_break_probability`
3. `sr_long_downside_break_probability`
4. `sr_long_resistance_rejection_strength`
5. `sr_long_structure_balance`
6. `sr_long_support_distance_atr`
7. `sr_long_resistance_distance_atr`
8. `sr_support_prior_strength`
9. `sr_resistance_prior_strength`
10. `sr_support_reaction_magnitude_q50`
11. `sr_resistance_reaction_magnitude_q50`
12. `sr_snapshot_available`

The mapper is a HistGradientBoosting regressor with depth 2, 80 iterations,
learning rate 0.04, L2 20, and minimum leaf support 100.  Each BCF and Current
vintage is fitted from six preceding complete months of policy labels resolved
strictly before the held month.  Its 21-day daily residual shift also contains
only labels resolved before each decision day.

## Hash-bound implementation

| Role | Canonical path | SHA-256 / binding |
|---|---|---|
| Contract | [`strict_r3_p8u_c1_lva_canonical_20260901_v1.json`](../config/strict_r3_p8u_c1_lva_canonical_20260901_v1.json) | `3173421008d2c7d59f616f97da4372989d4a97ce3b898ae02d9b2eec4a0b98ae` |
| Assembler | [`p8u_c1_lva_canonical_stack.py`](../extreme_price_movements/inference/p8u_c1_lva_canonical_stack.py) | `778cf544a5aa6cf08cf6845607693eb8edc3f291fd7384fa62a2d2d6cc0ccf3a` |
| C1 source bundle | [`causal_sr_c1_lva_inference_bundle_20260901_v1`](../data_perp/artifacts/causal_sr_c1_lva_inference_bundle_20260901_v1/bundle_manifest.json) | manifest `c3edb7cb2fed97ab9f192f64b9954fbd8eb6dd9de92385e756d8408a46f02524` |
| Append-only C1 state | [`causal_sr_c1_state.py`](../extreme_price_movements/inference/causal_sr_c1_state.py) | `0aaf21b33f016213ba9a323c90d37ac71538d98e4878fbee73e82ac393c19152` |
| Source bundle runtime | [`causal_sr_c1_lva_bundle.py`](../extreme_price_movements/inference/causal_sr_c1_lva_bundle.py) | `b6c351a4dfb8c9e5952baa321f0349b8d78523b6ae2718b90fe3231fde1ae997` |
| Mapper runtime | [`p8u_c1_mc1_inference_package.py`](../extreme_price_movements/inference/p8u_c1_mc1_inference_package.py) | `0412d62ad913124ddb70d1df577766eb156ceff6c0bf1c2fc751272dc67b40e8` |
| Vintage selector | [`p8u_c1_mc1_selector.py`](../extreme_price_movements/inference/p8u_c1_mc1_selector.py) | `514fb4d977ed67d6f5c6fe1d002ba9177d74a5fba3abcd2536b3a31b2849fca8` |
| Sealed May–Jul mapper packages | [`p8u_c1_full_coverage_dual_mc1_prequential_mayjul_20260901_v1`](../data_perp/artifacts/p8u_c1_full_coverage_dual_mc1_prequential_mayjul_20260901_v1/run_manifest.json) | package tree `bc37e267e33b03bf3702dccd47e19da287bcc1507ec9bc1a0a2ab34d1f7057e4` |

The canonical loader verifies all listed runtime and artifact hashes before it
loads a source model or mapper.  It then rejects any non-target-free field,
identity mismatch between BCF and Current, C1 reordering, non-causal daily
shift, or package vintage that does not cover the decision timestamp.

The canonical parity audit
[`c1_lva_canonical_parity_20260901_v1`](../data_perp/artifacts/c1_lva_canonical_parity_20260901_v1/receipt.json)
reconstructed all 65,656 sealed May–July target-free rows: BCF mapped EV,
Current mapped EV, and the dual-admission boolean each have exactly zero
numerical delta for every monthly vintage.

The included May–July validation index is a historical parity fixture, **not**
a future-vintage fallback.  Each future refit must write a new immutable
six-month package index and a matching C1 source bundle before its held month
can be scored.

## Execution successor status

[`strict_r3_p8u_c1_lva_live_execution_candidate_20260901_v1.json`](../config/strict_r3_p8u_c1_lva_live_execution_candidate_20260901_v1.json)
records the only permitted live preflight convention for a future C1 release:

```text
raw expected gross = BCF C1-MC1 expected net + 100 bps
execution-adjusted EV = raw expected gross
                        - adverse-only delay gap
                        - (1.2 × full live spread + 2 × entry VWAP impact + 10 bps)
```

There is **no 80-bps friction floor**.  This matches the sealed execution
implementation; an earlier gateway JSON that documented a `max(80, …)` floor
was inaccurate.  The successor remains no-order until it has a current-month
C1 mapper vintage, append-only C1 state, independent BCF coordinates, and an
end-to-end no-order parity receipt.  It never falls back to the non-C1 MC1
mapper or a stale monthly C1 package.

## Exact one-minute validation evidence

The canonical economic receipt is:
[`p8u_c1_lva_vs_core_exact1m_parent_mayjul_20260901_v9_all_active_sources_clean`](../data_perp/artifacts/p8u_c1_lva_vs_core_exact1m_parent_mayjul_20260901_v9_all_active_sources_clean/portfolio_summary.parquet).

It uses target-free dual admission, BCF priority, the normal global constrained
portfolio, entry at decision +5 minutes, exact one-minute rich-parent exits,
and exactly one 100-bps parent cost.  It deliberately excludes E2/H4; they
require their own route-specific refit and cannot be spliced into this source
contract.

| May–Jul 2026 | Accepted trades | Trades/day | Net EV/trade | Total net bps | Sortino | Max drawdown | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| Core without C1 | 959 | 10.50 | +121.23 | +116,261.66 | 0.6979 | −8.73% | +15.79% |
| **Canonical C1-LVA** | **1,174** | **12.77** | **+105.61** | **+123,987.74** | **0.6121** | **−12.59%** | **+22.30%** |
| C1 minus core | +215 | +2.27 | −15.62 | +7,726.07 | −0.0858 | −3.86 pp | +6.51 pp |

| Month | C1 trades | C1 net EV/trade | C1 total net bps |
|---|---:|---:|---:|
| May 2026 | 708 | +88.91 | +62,948.13 |
| June 2026 | 248 | +140.46 | +34,833.91 |
| July 2026 | 218 | +120.21 | +26,205.69 |

The earlier +197.74-bps C1 headline used a different 15-minute / +1-hour
policy proxy.  Replaying its legacy C1 score panel under this exact 1m +5m
contract yielded +110.03 bps/trade, so the headline difference is primarily
execution-contract measurement—not an accidentally superseded C1 model.

## Zero-fee empirical spread sensitivity

Receipt:
[`c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4`](../data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4/scenario_metrics.parquet),
run manifest SHA-256 `270ec6f6e0b42a3aeec4ae6389bf3d62bccf4e3ed9a60369b9b805d2d2a40ed9`.

For the same 1,174 selected trades and fixed constrained entry identities:

```text
zero-fee net bps = exact 1m gross policy bps − per-asset full bid/ask spread percentile
```

One full spread is entry half-spread plus exit half-spread.  The source outcome
already contains the +5-minute entry timing and exact one-minute exit path;
the sensitivity adds no fee, impact, or extra slippage.  It does **not** use a
spread scenario to change admission, ranking, capacity, or the selected trade
set.  It replays the recorded constrained position-size fraction against a
10,000-unit starting wallet so PnL reflects the same compounding convention as
the parent receipt.

| Cost scenario | Mean net bps/trade | Total net bps | Win rate | Mean spread cost | Simulated final wallet | Realized-wallet max DD |
|---|---:|---:|---:|---:|---:|---:|
| Parent fixed 100 bps | +105.61 | +123,987.74 | 68.06% | 100.00 bps | 37,634,546.34 | −12.58% |
| **0% fee, asset p50 spread** | **+162.16** | **+190,375.12** | **73.51%** | **43.45 bps** | **3,010,276,121.27** | **−9.20%** |
| 0% fee, asset p60 spread | +157.90 | +185,368.98 | 73.08% | 47.72 bps | 2,173,615,496.71 | −9.39% |
| 0% fee, asset p70 spread | +152.88 | +179,481.37 | 73.00% | 52.73 bps | 1,480,374,500.29 | −9.58% |
| 0% fee, asset p80 spread | +146.01 | +171,417.66 | 72.15% | 59.60 bps | 872,703,650.83 | −9.89% |

The wallet figures are **simulated quote units from a 10,000-unit starting
wallet under 7× / 10%-slot compounding**, not a claim about an account’s
current balance.  The bps, trade count, and fixed-identity net PnL are the
portable comparison values.

The per-asset ledger supplied coverage for every selected trade: 102 symbols,
minimum 117 observed spreads per selected asset (median 294).  It used 270
readable files / 88,089 deduplicated observed spreads from 2026-06-10 through
2026-08-15.  Fifty-eight malformed placeholder files were excluded with their
exact errors preserved in `spread_source_audit.parquet`; none was interpreted
as zero spread or imputed.

Detailed outputs are:

- [`scenario_metrics.parquet`](../data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4/scenario_metrics.parquet)
- [`monthly_metrics.parquet`](../data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4/monthly_metrics.parquet)
- [`daily_metrics.parquet`](../data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4/daily_metrics.parquet)
- [`trade_sensitivity.parquet`](../data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4/trade_sensitivity.parquet)
- [`per_asset_spread_quantiles.parquet`](../data_perp/artifacts/c1_lva_zero_fee_per_asset_spread_sensitivity_20260901_v4/per_asset_spread_quantiles.parquet)

## Required operational behavior

1. Materialise C1 only from the persisted append-only completed-15-minute
   state; do not approximate it with a short rolling reconstruction.
2. Verify all source/runtime/package hashes in the canonical config before
   scoring.  A hash mismatch blocks C1 mapping and therefore entry.
3. Create the C1 snapshot for every upstream candidate; absence remains a
   mapper input, never a candidate-side filter.
4. Refit both mapper families with six prior complete months and only
   prior-resolved rich-policy labels; write a new immutable package index.
5. Apply the two 50-bps MC1 gates independently.  Use only BCF mapped EV for
   auction priority after both gates pass.
6. Do not interpret this no-order canonicalization as permission to modify an
   exchange-writing gateway.  A separate fresh monthly bundle and end-to-end
   inference/execution parity release remains required for that operation.
