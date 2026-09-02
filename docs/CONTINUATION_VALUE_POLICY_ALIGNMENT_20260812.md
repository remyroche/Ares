# Continuation-value policy alignment audit

Date: 2026-08-12  
Side: long only  
Status: corrected OOF research challenger; not promoted to execution

## Conclusion

The continuation challenger now uses the same frozen policy identity as the
current schema-v5 canonical stack and Codex thread
`019fe7f0-c456-7d42-92be-596a601782f9`:

- policy artifact:
  `data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json`;
- SHA-256:
  `2dc9a145766ae383a4ab7c33e8a9f9e358175597e05582300ff0a05732673603`;
- stop: `4.15200064332387 ATR`;
- trailing activation: `2.326224919759605 ATR`;
- fixed giveback: `0.10237198997143725 ATR`;
- timeout: 12 hours / 48 complete 15-minute bars;
- entry: first 15-minute open at signal close plus one hour;
- adverse exit and hard take-profit: disabled;
- cost: flat 100 bps deducted exactly once after gross simulation;
- simulator: `extreme_price_movements.simple_policy_optimiser.simulate_and_score`.

The earlier 14.5-bps mean absolute difference was not evidence of a different
policy geometry. It combined two effects:

1. the stored canonical outcome ledger used a mixed exact-minute-resampled /
   15-minute historical data vintage, while the challenger intentionally used
   the current local 15-minute history for both continue and exit-now;
2. the earlier challenger passed a 0.5% per-side simulator fee while exit-now
   used the canonical flat 100-bps deduction. That accounting mismatch moved
   the paired continuation target by 1.35 bps on average and 4.21 bps at p95.

The fee mismatch is fixed. The market-data-resolution difference remains an
explicit source limitation because the complete historical bar vintage used by
the original canonical replay was not persisted. The existing frozen
minute-path archive covers only 751 of the current 12,000 selected strict-R3
candidates, so it cannot honestly repair the complete population.

## Reconciliation evidence

The source-aligned stored outcome ledger and the original canonical policy
replay are identical on their 9,592 overlapping selected IDs: zero mean and
maximum absolute difference.

The corrected local-15-minute replay contains 9,886 complete trades and 78,025
hourly decision states. Against the stored mixed-resolution outcome ledger:

| Metric | Difference |
|---|---:|
| Mean absolute net difference | 13.52 bps |
| Median absolute net difference | approximately 1 bps |
| P95 absolute net difference | 41.29 bps |
| Maximum absolute difference | 2,930.13 bps |

Most rows agree closely; a small number of different intrabar stop/trailing
paths drive the tail. A current exact-first diagnostic is not a valid repair:
May and June 2026 historical entry prices in the mutable exact store no longer
match the canonical replay vintage. January 2025 through April 2026 reproduces
the old exact-first entries essentially exactly, while entry mismatches rise to
40.8% in May and 88.7% in June.

Therefore the challenger keeps one internally coherent target:

```text
local 15-minute path
→ same optimized policy for continuation
→ next 15-minute open for exit-now
→ flat 100-bps round-trip cost on both alternatives
```

It does not mix a stored canonical continuation outcome with a current local
exit-now outcome.

## Corrected OOF continuation results

Artifact:
`data_perp/artifacts/continuation_value_challenger_long_hourly_canonicalpolicy_local15m_20260812_v6`

The model acts once per completed hour. Its LambdaRank query is four-hour ×
long-side, but query grouping does not reduce the hourly inference clock.

Winner:

- features: 26 causal path-state fields only;
- query: four-hour × side;
- OOF states: 66,012;
- OOF trades: 7,965;
- pooled rank IC: 0.1092.

| Tail | Continuation-delta uplift | Continued-trade net |
|---:|---:|---:|
| 1% | +72.51 bps | +65.16 bps |
| 2% | +74.23 bps | +58.96 bps |
| 5% | +69.48 bps | +11.72 bps |
| 10% | +115.43 bps | +58.07 bps |
| 20% | +81.69 bps | +27.69 bps |

The bounded target screen finds that downside-sensitive ordinal and binary
`delta > +50 bps` targets remain competitive, but no direct execution action is
selected from these target-oracle metrics.

## Downstream action gate

The continuation model remains one layer removed from execution. A separate
causal hourly D0-D5 classifier decides whether to preserve the incumbent policy
or exit at the next 15-minute open.

| Arm | Adaptive net | Uplift vs D0 | Intervention rate |
|---|---:|---:|---:|
| D0 frozen policy | -1.13 bps | 0.00 | 0.00% |
| D5 all compact outputs | -1.64 bps | -0.51 | 3.94% |
| D0 path/context relearner | -1.77 bps | -0.64 | 4.67% |
| D4 uncertainty/support/OOD | -1.96 bps | -0.83 | 3.50% |
| D1 continuation rank | -1.99 bps | -0.86 | 2.10% |
| D3 value probability | -2.59 bps | -1.46 | 5.16% |
| D2 expected bps | -2.87 bps | -1.74 | 4.60% |

No downstream arm clears the required +5-bps incremental gate. Nothing is
promoted.

## Code and tests

- `scripts/run_continuation_value_challenger.py`
- `extreme_price_movements/continuation_value/`
- `tests/test_continuation_value.py`
- `docs/TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md`

Validation:

```text
python3 -m pytest -q tests/test_continuation_value.py
9 passed
```

The canonical document's retained schema-v2 SL3/0.5/0.25 section is now
explicitly marked legacy so it cannot override the schema-v5 optimized-policy
contract.
