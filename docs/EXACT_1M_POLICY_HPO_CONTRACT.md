# Exact 1m SimplePolicyOptimiser HPO contract

Status: offline research successor; not promoted to the live policy.

## Purpose

The frozen policy winner was selected on a 15-minute path contract. The live
long-only executor instead processes completed one-minute bars and submits a
reduce-only market close after a trailing threshold bar completes. This
contract removes that mismatch before another policy geometry is selected.

## Frozen execution semantics

```text
decision timestamp
→ entry at one uniform decision + 5 minute timestamp
→ 720 complete Kraken Futures one-minute bars
→ pre-existing hard protective stop
→ trailing state armed only by a prior completed MFE
→ crossing on a later completed minute
→ reduce-only exit fill proxy = that threshold bar close
→ H12 timeout at final completed-minute close
→ 100 bps round-trip policy cost exactly once
```

The historical archive has no reliable bid/ask snapshots for the 2024 HPO
population. The bar-close trailing fill is therefore explicit and deterministic;
it is not presented as historical order-book reconstruction. A later observed
live-fill audit can replace it only through a new versioned contract.

## Time units

The new HPO forbids legacy bar-based optimisation parameters. It uses:

- `trailing_activation_decay_half_life_minutes`;
- `trailing_activation_decay_start_minutes`;
- `adverse_exit_fast_minutes`;
- `adverse_exit_min_speed_per_hour`;
- `holding_rent_bps_per_hour`.

This prevents a parameter from changing its economic meaning by 15× when a
15-minute replay is changed to one-minute data.

## Sequential funnel

The default, deployable funnel re-fits the full current parent-policy
surface—exactly three live inputs—without introducing a model-only exit:

1. broad stop ATR / activation ATR / fixed giveback ATR;
2. local refinement of all three controls around the broad finalists; and
3. narrow polish of all three controls.

The frozen incumbent then enters the same final constrained-portfolio
tournament as the challengers; a challenger cannot win simply because the
control was omitted.  `--execution-surface rich_research` retains the earlier
time/geometry/protection search as a separate, explicitly non-deployable
diagnostic: the current minute executor does not consume those controls.

The final control and top challengers are replayed through the standard
portfolio auction with their explicit one-minute `exit_timestamp`; the
portfolio layer never infers exits as `holding_bars × 15 minutes`.

The winner is selected from those finalists by constrained-portfolio monthly
economics: median monthly net bps/trade minus one-half monthly MAD and the
full worst-negative-month penalty. It is not selected solely on unconstrained
per-row path outcomes.

The sequential search uses an equal-month, deterministic candidate-ID sample
(`500` rows/month by default) chosen before outcomes are read. Finalists are
then evaluated on the complete July–December path population and its actual
portfolio timestamps.

## Artifacts and commands

First write the target-free source request. It is score-routed before paths or
outcomes are inspected:

```bash
python3 scripts/materialize_strict_r3_exact_1m_policy_hpo_dataset.py \
  --out-dir data_perp/artifacts/strict_r3_exact_1m_policy_hpo_download_request_2024_v1 \
  --request-only
```

Download it through the pre-existing append-only Kraken producer. The request
records the required `6060` minute source warm-up and `725` minute decision window.
The policy still uses a 100-hour Wilder-14 state: the extra hour ensures a
five-minute-delayed entry has 100 *complete* preceding hourly bins rather than
99 plus an incomplete first bin.
(five-minute entry delay plus H12 path); no source bar is fabricated.

```bash
python3 scripts/download_policy_execution_1m.py \
  --candidates data_perp/artifacts/strict_r3_exact_1m_policy_hpo_download_request_2024_v1/candidate_download_request.parquet \
  --data-root data_perp --warmup-minutes 6060 --horizon-minutes 725 \
  --partition-count 16 --partition-id 0 \
  --manifest data_perp/artifacts/strict_r3_exact_1m_policy_hpo_download_request_2024_v1/download_partition_0.json \
  --stage-manifest data_perp/artifacts/strict_r3_exact_1m_policy_hpo_download_request_2024_v1/candidate_download_request.json
```

Run every partition ID once, then materialise the immutable HPO dataset and
join only complete paths:

```bash
python3 scripts/materialize_strict_r3_exact_1m_policy_hpo_dataset.py \
  --minute-root data_perp/exchanges/krakenfutures/execution_1m
```

Run the sequential HPO only after that immutable dataset has valid 2024 source
coverage:

```bash
python3 scripts/run_strict_r3_exact_1m_policy_hpo.py \
  --dataset-dir data_perp/artifacts/strict_r3_exact_1m_policy_hpo_dataset_202402_20260817_v2 \
  --out-dir data_perp/artifacts/strict_r3_exact_1m_policy_hpo_live_parent_long_20260817_v2 \
  --execution-surface live_parent
```

Implementation is in:

- `extreme_price_movements/exact_1m_policy_contract.py`;
- `scripts/materialize_strict_r3_exact_1m_policy_hpo_dataset.py`;
- `scripts/run_strict_r3_exact_1m_policy_hpo.py`.

## Required gates

- Future path availability never affects candidate routing.
- Invalid or incomplete paths are excluded from HPO rather than labelled as a
  zero-return failure.
- All 720 post-entry minutes, the causal ATR and the entry minute are finite.
- At least 90% of score-routed rows have complete paths; every compatible
  February–December month has at least 75 valid rows.
- Every path uses the same decision + 5m entry rule; no rank-dependent entry
  materialisation is permitted.
- No same-bar activation-and-trailing crossing is permitted.
- Gross minus net is exactly 100 bps on every valid path.
- Selection uses the compatible February–December 2024 period only.
  2025–2026 are untouched evaluation periods.
- The winner stays a challenger until a separately approved live bundle is
  created and its one-minute state machine is replayed against the executor.

## Completed 2024 run

The immutable source request contains 27,871 score-routed candidates; 27,754
have a complete exact one-minute path (99.58%). All valid rows use a five-minute
delayed entry and a 720-minute horizon. HPO calibrates on February–June 2024
and selects on July–December 2024; 2025 onward was not opened.

The deployable-surface tournament retained the incumbent parent contract:

```text
SL                  4.15200064 ATR
trailing activation 2.32622492 ATR
trailing giveback   0.10237199 ATR
```

On the same July–December constrained replay, the incumbent produced 2,469
accepted entries, +31.45 net bps/trade, +77,649 total net bps, and a −34.37
bps/trade worst month. The strongest new deployable challenger produced 2,434
entries, +29.51 net bps/trade, +71,832 total net bps, and a −39.97 bps/trade
worst month. It therefore did not advance.

The earlier rich-search artifact
`strict_r3_exact_1m_policy_hpo_long_20260817_v1` is diagnostic-only: it
selected dynamic/protection behaviour unsupported by the live parent policy
and must never be used as a live-policy candidate.

## Explicit dual-MC1 candidate mode

The exit-resolution research can materialise a separately sealed, target-free
candidate request rather than reusing the 2024 ledger's top-five-percent route.
This is required when the fixed admission contract is:

```text
BCF MC1 expected EV >= 30 bps
AND current-v5 MC1 expected EV >= 30 bps
→ BCF MC1 expected EV supplies the portfolio priority
```

The decision-entry request is:

`data_perp/artifacts/strict_r3_exact_1m_dual30_bcf_priority_candidates_decision_2025_2026_20260817_v1/`

It contains the 49,556 predeclared long candidates, not only candidates with
a valid policy outcome or an accepted historical portfolio position. Its
manifest binds the request hash, a zero-minute entry contract, target-free
selection inputs, and forbidden outcome inputs. `priority_bps` becomes the
portfolio ordering score; it is not a label.

After all 16 append-only downloader receipts exist, materialise it with:

```bash
python3 scripts/materialize_strict_r3_exact_1m_policy_hpo_dataset.py \
  --candidate-input data_perp/artifacts/strict_r3_exact_1m_dual30_bcf_priority_candidates_decision_2025_2026_20260817_v1/candidate_download_request.parquet \
  --candidate-manifest data_perp/artifacts/strict_r3_exact_1m_dual30_bcf_priority_candidates_decision_2025_2026_20260817_v1/candidate_download_request.json \
  --download-request-dir data_perp/artifacts/strict_r3_exact_1m_dual30_bcf_priority_candidates_decision_2025_2026_20260817_v1 \
  --entry-delay-minutes 0 \
  --start 2025-02-01T00:00:00Z --end 2026-08-01T00:00:00Z \
  --out-dir data_perp/artifacts/strict_r3_exact_1m_dual30_decision_dataset_2025_2026_v1
```

This mode rejects outcome-derived source columns, a non-target-free source
manifest, duplicate IDs, a non-long side, non-finite priority, an entry
timestamp other than the declared decision delay, a source hash mismatch, or
incomplete partition receipts. It only joins one-minute path validity after
candidate routing; invalid paths are audited and excluded from fitting/replay.
