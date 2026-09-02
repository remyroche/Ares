# Short base feature selection and supportive labels — 2026-08-20

## Scope and decision boundary

This is long-side methodology applied separately to shorts. It is not a
promotion and it does not alter the live long stack.

- Selection window: 2024-01-01 through 2024-03-31.
- Inner chronological evidence: January → February and January–February →
  March, with labels required to be available before each validation month.
- Held target evaluation: 2024-04-01 through 2024-06-30.
- All scores use the frozen short 120-field causal base contract and frozen
  base LGBM parameters; no HPO or policy tuning occurred.

The original long selector artifact had been retired during prior cleanup, but
the retained Stage-I implementation specifies the process. The applied short
contract is:

```text
target-free executable coverage/variance
→ univariate screen + bounded ReliefF rescue (gain tie-break)
→ Spearman-0.95 feature-only representatives
→ chronological signed-economic permutation MDA
→ smallest prefix within one standard error
```

Apr–Jun candidate, feature, label, and outcome rows are not opened by the
selector. The complete immutable selection receipt is
`data_perp/artifacts/strict_r3_short_stagei_style_feature_selection_2024q1_20260820_v7`.

## Feature-selection result

The repaired short feature panel now has all **120/120** frozen short fields at
or above the target-free 90% coverage and variance gates. The selector reduced
this to 48 MDA candidates, 30 ranked fields and a 15-field one-SE prefix:

```text
xs_dispersion__ffd_amihud_04
q_tail_asym__amihud_z_peer_resid
xs_dispersion__funding_per_hour
state_spectral_eig_condition
q_upper_tail__xasset_ob_liquidity_ts_resid
eig_effective_rank__open_interest
state_spectral_eig_top3_share
q_iqr__ret48h_bench_resid
mkt_ret_eq_4h
mkt_pct_price_down_oi_up_1h
q_tail_asym__ob_depth_usd_l20_z
bars_to_support_daily_donchian
xasset_mkt_depth_to_qv_z
loc_session_pos_24
q_lower_tail__oi_7d_x_funding
```

The inner selector evidence is weak: 15 fields scored −143.18 bps on the
weighted Feb/March inner utility versus −129.74 for 20 fields; the one-SE rule
therefore chooses the smaller 15-field prefix. This makes the untouched
Apr–Jun test essential.

## Untouched Apr–Jun target comparison

All figures are exact H12 TP6/SL4 **net** bps per resolved trade, globally
ranked across the held population. Labels are only joined after scoring;
coverage is approximately 68–70% at the tails because invalid/incomplete paths
remain excluded from outcome averages rather than encoded as losses.

| Feature contract | Target | Top 1% | Top 2% | Top 5% |
|---|---|---:|---:|---:|
| Prior 120 fields | Ordinal ≤−250 / 0 / +50 | **−9.23** | **−44.63** | −77.28 |
| Prior 120 fields | Ordinal ≤−200 / 0 / +50 | −16.45 | −44.92 | −74.55 |
| Prior 120 fields | R3 control | −50.58 | −71.09 | −83.43 |
| Stage-I 15 fields | Ordinal ≤−250 / 0 / +75 | −46.50 | −48.25 | −86.02 |
| Stage-I 15 fields | Ordinal ≤−300 / 0 / +100 | −85.75 | −88.81 | −86.83 |
| Stage-I 15 fields | Ordinal ≤−250 / 0 / +50 | −97.46 | −95.75 | −86.62 |
| Stage-I 15 fields | Ordinal ≤−200 / 0 / +50 | −105.02 | −117.50 | −92.96 |
| Stage-I 15 fields | R3 control | −115.45 | −100.02 | −92.43 |
| Stage-I 15 fields | Ordinal ≤−150 / 0 / +25 | −178.49 | −121.31 | −104.98 |

Conclusion: the side-local Stage-I subset is a valid research result but does
**not** improve the short base over the full 120-field contract. Keep the
120-field short pool for the next base/meta experiment. The least poor tested
short target remains the prior 120-field ordinal ≤−250 / 0 / +50 control, but
it is still far below the current long R3 reference (+88.60 bps at top 1% on
the matching prior experiment). Nothing here advances shorts to trading.

## Short supportive path labels

`scripts/materialize_strict_r3_short_supportive_path_labels.py` produces the
sidecar:

`data_perp/artifacts/strict_r3_short_supportive_path_labels_2024_20260820_v3`

It uses each frozen short decision-time entry and ATR, opens only the complete
post-decision 720×1-minute high/low path, and calls the shared v6
side-normalized target kernel with `side_sign = -1`:

- favourable excursion = entry − low;
- adverse excursion = high − entry;
- label availability = decision + 12 hours;
- invalid source paths retain null economic targets and zero validity flags.

It exposes the five primary targets and 58 support fields, including
meaningful-MFE reachability/timing, pre-MFE MAE, adverse-turn/recovery,
multi-horizon slope and path-efficiency labels. They are training-only labels,
not inference features.

| Month | Rows | Valid auxiliary paths | Meaningful-MFE reached |
|---|---:|---:|---:|
| 2024-01 | 52,230 | 31,753 | 15,907 |
| 2024-02 | 50,998 | 30,906 | 11,466 |
| 2024-03 | 55,499 | 34,735 | 16,409 |
| 2024-04 | 55,869 | 42,127 | 22,861 |
| 2024-05 | 64,231 | 40,116 | 17,272 |
| 2024-06 | 68,745 | 53,114 | 25,337 |

For every month, the sidecar has exact candidate/timestamp/decision/availability
identity parity with the short label source, is short-only, has targets for
every valid path, and has null targets for every invalid path.

The v3 receipt also verifies the entry convention directly: all 232,751 valid
rows have a frozen `tp6_sl4_entry_price` exactly equal to the reopened
decision-minute 1-minute OHLCV open.  The five primary targets therefore have
the same shared v6 kernel, 12-hour/one-minute path convention, ATR
normalization, meaningful-MFE rule, censoring, and invalid-path handling as
the long target materializer.  The retained historic long sidecar did not
persist the 58 newer supporting columns (`include_supportive_columns=False`),
so those are same-kernel extensions rather than byte-for-byte historical long
columns.

## Recommended next experiment

Do not feed future-path labels directly to the base. Train short auxiliary
heads strictly OOF/prequentially from this sidecar, then test their OOF outputs
as meta/reliability inputs against the retained 120-field ordinal base. Begin
with a reachability head `P(meaningful MFE)` before conditional MFE magnitude;
the label coverage and event rate make that the cleanest short-side bottleneck
test.

## Correctness checks

- `extreme_price_movements/tests/test_ordinal_base_target_ablation.py`
- `extreme_price_movements/tests/test_short_target_augmentations.py`

Both focused suites pass (9 tests total). The new supportive-target test proves
that mirrored long and short paths produce identical side-relative target
values.
