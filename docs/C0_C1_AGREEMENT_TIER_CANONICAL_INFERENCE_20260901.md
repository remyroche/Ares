# C0/C1 agreement-tier canonical inference contract — 2026-09-01

## Status

This is the user-directed, long-only **canonical inference successor** for
combining the frozen C0 and C1-LVA score families:

```text
same target-free Router50 population
  -> independently hash-bound C0 and C1 BCF/Current MC1 maps
  -> C0 dual-50 admission and C1 dual-50 admission
  -> both-admitted -> C0-only -> C1-only capacity gap fill
  -> normal constrained portfolio auction
  -> existing rich-policy execution and exit monitoring
```

It is the canonical inference hierarchy. It is not yet the active
exchange-writing release because current-vintage C1 packages and a current
end-to-end parity release remain required operational evidence.

The superseded v1 hard-tier route forced every C1-only candidate ahead of every
C0-only candidate; the superseded v2 raw-EV route placed both unpaired families
in one raw-EV class. The active canonical contract is
[`strict_r3_p8u_c0_c1_agreement_c0first_inference_20260901_v3.json`](../config/strict_r3_p8u_c0_c1_agreement_c0first_inference_20260901_v3.json),
SHA-256 `8cb57065a639606c2125b49b9312e9f05d3a37a730f940a8537779f5d9f5d7c6`.

## Exact inference semantics

Every C0 and C1 map receives the **same** complete target-free Router50
identities: `candidate_id`, `__decision_ts__`, `__symbol__`, and `side_name`.
An identity mismatch, duplicate, missing map, non-finite EV, or outcome-like
column fails closed.

Each family has its own fixed gate:

```text
family admitted = BCF MC1 expected EV >= +50 bps
                   AND Current MC1 expected EV >= +50 bps
```

The selector retains a candidate only if either family admits it and gives it
the following deterministic class:

| Tier | Provenance | Coordinate retained for execution EV |
|---:|---|---|
| Agreement | `both_admitted` | C0 BCF / Current MC1 values |
| Unpaired | `c1_only` | C1-LVA BCF / Current MC1 values |
| Unpaired | `c0_only` | C0 BCF / Current MC1 values |

The raw `auction_priority_bps` remains the selected source's BCF MC1 expected
EV. It is used for raw expected gross, execution-adjusted EV, and terminal
telemetry. The auction alone uses:

```text
portfolio_order_priority_bps = auction_priority_bps
                               + 20,000 for both-admitted rows
                               + 10,000 for C0-only rows
                               + 0 for C1-only rows
```

This makes agreement dominant, preserves C0 opportunities ahead of C1-only
opportunities, then permits C1 solely as a capacity gap fill. The synthetic
ordering offsets are kept out of entry economics, execution guards, PnL, and
subsequent calibration.

## Hash-bound components

| Role | Path | SHA-256 |
|---|---|---|
| C0-first canonical contract | [`strict_r3_p8u_c0_c1_agreement_c0first_inference_20260901_v3.json`](../config/strict_r3_p8u_c0_c1_agreement_c0first_inference_20260901_v3.json) | `8cb57065a639606c2125b49b9312e9f05d3a37a730f940a8537779f5d9f5d7c6` |
| Target-free selector | [`p8u_c0_c1_agreement_tier.py`](../extreme_price_movements/inference/p8u_c0_c1_agreement_tier.py) | `e443d351d9aab2f50924c88938ac8f3954730bbd5a04c205a7dbe28287d780a7` |
| Immutable score assembler | [`assemble_p8u_c0_c1_agreement_tier.py`](../scripts/assemble_p8u_c0_c1_agreement_tier.py) | `4aa45e67e29ddc768d285efab41eecc9e8d069d48823f5f7150914a28f62062a` |
| No-order portfolio adapter | [`p8u_execution_portfolio_adapter.py`](../extreme_price_movements/inference/p8u_execution_portfolio_adapter.py) | `bff089b5e7e2ad38c002f023f111316fa29cda237579d703e1415300431aac14` |
| C0 source contract | [`strict_r3_p8u_dual_mc1_sixmonth_inference_20260828_v2.json`](../config/strict_r3_p8u_dual_mc1_sixmonth_inference_20260828_v2.json) | `5a4a25c56409093e3d3d0f5e0bd1c6872377bae7d87dece943aa182e65250989` |
| C1-LVA source/map contract | [`strict_r3_p8u_c1_lva_canonical_20260901_v1.json`](../config/strict_r3_p8u_c1_lva_canonical_20260901_v1.json) | `3173421008d2c7d59f616f97da4372989d4a97ce3b898ae02d9b2eec4a0b98ae` |

The assembler writes an immutable target-free score panel and manifest with
all source hashes, tier counts, raw EV field, and ordering field.  It has no
label, policy, portfolio, exchange, or order-submission authority.

The existing no-order portfolio adapter has a separately validated
`c0_c1_agreement_tier` mode.  That mode recomputes both dual-admission
predicates from the persisted C0/C1 values, validates every tier and selected
coordinate, and sorts only by `portfolio_order_priority_bps`.  It keeps
`auction_priority_bps` unchanged for execution economics.

Its sealed v3 May--July target-free parity fixture is
[`p8u_c0_c1_agreement_c0first_target_free_assembly_mayjul_20260901_v3`](../data_perp/artifacts/p8u_c0_c1_agreement_c0first_target_free_assembly_mayjul_20260901_v3/run_manifest.json).
It has 4,205 target-free admitted rows: 3,160 both-admitted, 741 C1-only, and
304 C0-only.  The raw selected EV is never above 10,000 bps, confirming that
the offset remains confined to the separate auction-order field.

## Evidence and trade-off

The decisive evidence uses entry at decision +5 minutes, exact completed
one-minute rich-policy exits, the normal global constrained portfolio, and one
100-bps cost.  It does not use a coarse OHLC policy proxy.

| May--Jul 2026 exact route | Accepted trades | Net EV/trade | Total net bps | Sortino | Max drawdown |
|---|---:|---:|---:|---:|---:|
| C0 only | 959 | **+121.23** | +116,261.66 | **0.698** | **-8.73%** |
| C1-LVA only | 1,174 | +105.61 | +123,987.74 | 0.612 | -12.59% |
| Both -> C1-only -> C0-only (superseded) | 1,211 | +105.08 | +127,252.12 | 0.605 | -11.46% |
| Both -> unpaired by raw BCF EV (superseded v2) | 1,205 | +104.48 | +125,898.76 | 0.598 | -11.46% |
| **Both -> C0-only -> C1-only (canonical v3)** | **1,211** | **+105.25** | **+127,458.06** | **0.602** | **-11.46%** |

The C0-first route retains six more trades and 1,559 total bps than the raw-EV
variant, while slightly improving its risk-adjusted score. It is still not a
pure quality or risk-adjusted improvement over C0-only: C1 remains a bounded
capacity extension, not a C0 replacement.

Its stress point is May:

| May 2026 accepted tier | Trades | Net EV/trade | Total net bps |
|---|---:|---:|---:|
| Both-admitted | 593 | +105.22 | +62,395.63 |
| C0-only | 61 | **-8.45** | -515.44 |
| C1-only | 77 | **-10.62** | -817.43 |

Thus the high-quality agreement tier carries the month. The two appended
single-family cohorts were negative even under the canonical C0-first order.
The route must not be described as a lower-risk replacement for C0-only or as
evidence that C1 is purely better.

The immutable exact replays are
[`hard tier v1`](../data_perp/artifacts/c0_c1_agreement_tier_exact1m_mayjul_20260901_v1/portfolio_summary.parquet)
and [`agreement then raw EV v2`](../data_perp/artifacts/c0_c1_agreement_then_rawev_exact1m_mayjul_20260901_v1/portfolio_summary.parquet),
and [`canonical C0-first v3`](../data_perp/artifacts/c0_c1_agreement_then_c0_then_c1_exact1m_mayjul_20260901_v1/portfolio_summary.parquet).

## Execution and terminal controls

This selector changes admission provenance and auction ordering only.  It does
not change the existing rich-policy exit contract:

- native reduce-only protective stop is placed from the policy threshold after
  conversion through size-aware exit VWAP;
- the live monitor advances the rich policy on completed one-minute bars;
- a newly armed trailing/smooth state cannot exit in the same bar;
- execution-adjusted EV uses the raw mapped BCF EV selected above, never the
  tier offset;
- terminal telemetry must retain both family EVs, provenance, selected raw EV,
  spread/impact/delay fields, and fee-confirmed realised net bps.

## Exchange-release gate

Before any exchange-writing activation, all of the following are required:

1. current C0 and C1 monthly mapper packages, fitted only from their declared
   prior-resolved data;
2. append-only completed-15-minute C1 state and a current C1 source bundle;
3. a current no-order receipt proving C0/C1 identity equality, target-free
   score parity, tier provenance, raw-versus-order-field separation, portfolio
   order, execution preflight, and terminal-telemetry schema;
4. a separately sealed exchange authorization that binds every current hash.

Missing C1 data, a stale C1 package, or a hash mismatch must fail closed.  No
fallback to a different C0/C1 version is permitted.
