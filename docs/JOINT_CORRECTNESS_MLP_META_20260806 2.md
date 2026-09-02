# Latent joint-correctness states and MLP meta layer

## Scope

This experiment uses the frozen long-side structural sidecar:

`data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4/tree_meta_candidate_sidecar.parquet`

The sidecar contains 136 causal raw/AE-GMM fields, fixed `meta_train`,
`meta_calibration`, and `test` partitions, realised fixed-policy gross/net
labels, and strict fold provenance.  The source is long-only; no short-side
result is implied.

## Architecture

For each outer fold and each target arm:

1. Split `meta_train` chronologically into `head_fit` (first 60%) and
   `state_train` (last 40%).
2. Fit ten side-local LambdaRank heads on `head_fit`:
   `cap40/60/80/100/120 × ordinary/equal_month`.
3. Extract each head's score percentile and tree-leaf posterior correctness.
   Leaf correctness is a smoothed empirical correctness map fit only on
   `head_fit` resolved outcomes.
4. Discover latent joint states from the ten soft correctness values, binary
   activations, summaries, and the highest co-activation-lift pairs.  K is
   selected from 3–6 on `state_train` only.
5. Train an MLP on causal context, ten head ranks, leaf correctness,
   consensus dispersion/agreement, OOD/health fields, and prior-only
   7/14/28-day correctness history.
6. Fit the MLP-to-net isotonic map on `meta_calibration` only, then score `test`.
7. Compare the MLP as a replacement, as a weight on consensus, and as an
   asymmetric blend with the matched cap-120 policy-correction control.

The final ranking is pooled global top-k, not per-timestamp top-k.  Net labels
already contain the single declared cost; no second cost is subtracted.

## Target ablation

| Target arm | Definition |
|---|---|
| `residual_ordinal` | residual net (`net_bps − base_expected_bps`) in −150/−50/+50/+150 bps grades |
| `net_ordinal` | exact net in the same ordinal bins |
| `clear_binary` | exact H12 net > +50 bps |
| `query_rank_ordinal` | query-relative residual quintiles |

## Pooled OOS results

The full ablation is in
`data_perp/artifacts/joint_correctness_mlp_meta_20260806_v1/comparison_metrics.parquet`.

The strongest target was `clear_binary`.  Its best consensus-weighting arm was
the MLP-weighted consensus at top-5; the best top-1 point was the 50/50 blend.

| Clear-binary score | Top 1% net | Top 5% net | Top 10% net |
|---|---:|---:|---:|
| Consensus EV alone | −100.75 | −102.83 | −98.64 |
| MLP state score replacing consensus | −94.39 | −92.83 | −95.38 |
| Cap-120 policy correction | −73.06 | −99.45 | −100.33 |
| MLP-weighted consensus | +13.08 | **−23.09** | −39.56 |
| 25% MLP / 75% consensus | +14.85 | −24.03 | −40.21 |
| 50% MLP / 50% consensus | **+16.25** | −23.33 | **−39.55** |

The other target arms were weaker.  Their best pooled top-5 net values were:

| Target | Best score arm | Top-5 net |
|---|---|---:|
| `clear_binary` | MLP-weighted consensus | −23.09 bps |
| `net_ordinal` | MLP-weighted consensus | −38.00 bps |
| `residual_ordinal` | 25% MLP / 75% consensus | −45.01 bps |
| `query_rank_ordinal` | 50% MLP / 50% consensus | −73.16 bps |

## Monthly stability

Clear-binary MLP-weighted consensus top-5 net EV by month:

| Month | Net bps/trade |
|---|---:|
| May 2024 | −46.18 |
| June 2024 | −197.66 |
| July 2024 | −31.04 |
| August 2024 | −74.18 |
| September 2024 | −45.32 |
| October 2024 | −87.29 |
| November 2024 | +78.19 |

The positive pooled top-1 result is not stable: only November is positive at
top-5.  The June failure is severe.  No arm passes execution readiness.

## Can the MLP recognize the states?

The explicit state-recognition audit is in
`data_perp/artifacts/joint_correctness_mlp_meta_clearbinary_20260806_v1/mlp_state_metrics.parquet`.

Across the three folds:

| Split | Accuracy | Log loss | State-EV rank IC |
|---|---:|---:|---:|
| State training | 99.22% | 0.023 | +0.032 |
| Calibration | 95.17% | 0.373 | −0.027 |
| Test | 93.51% | 0.706 | +0.003 |

This is the key result: the MLP recognizes the latent leaf-derived states, but
the states do not transport economically.  Test state-EV rank IC is effectively
zero.  The state representation is therefore mostly identifying structural
model patterns, not stable future net opportunity.

State economic separation also collapses across folds.  The first fold has
state means roughly −112, −101, −82, and −57 bps; the later folds compress to
approximately −112/−111/−111 and −96/−93/−91 bps.

## Decision

1. The MLP should not replace consensus.
2. MLP weighting/blending is materially better than raw consensus and the
   matched cap-120 control in this replay, especially for top-1/top-5.
3. `clear_binary` is the best of the tested head targets.
4. The improvement is not portable enough for promotion: top-5 remains
   negative in six of seven months and the latent-state EV signal has no test
   rank correlation.
5. The next repair should discover states using a target that is explicitly
   cross-fold economically stable, or add a second state layer that predicts
   regime-conditioned conversion failure rather than leaf correctness alone.

Correctness reports:

- `data_perp/artifacts/joint_correctness_mlp_meta_20260806_v1/correctness_test_report.json`
- `data_perp/artifacts/joint_correctness_mlp_meta_clearbinary_20260806_v1/correctness_test_report.json`

Both reports pass their declared integrity checks.
