# Dataset Contract

## 1. Row Identity And Time

Every model or policy row must have a stable identity including timestamp,
symbol/instrument, and side. Candidate IDs should remain stable across base,
meta, policy, and replay artifacts.

Timestamps represent observability time. For OHLCV, the timestamp is the bar
close unless a manifest explicitly states otherwise. Features use data at or
before the timestamp; targets and replay paths start after the decision point.

Higher-frequency and auxiliary data must be joined with causal backward/as-of
logic. Publication delay, stale limits, and timezone must be explicit.

UTC is canonical for persisted datasets, joins, model features, labels, replay,
inference, manifests, and artifact IDs. Every internal timestamp must be
timezone-aware UTC; naive legacy values are interpreted as UTC on ingest, never
as machine-local time. `Europe/Paris`/CEST may be used only after UTC
normalization for display, email, or UI text, and must not feed a stored value,
join key, split, rolling window, or calendar feature.

## 2. Side Semantics

- Use canonical `long` and `short` names plus an explicit numeric side sign.
- Returns, MFE, MAE, TP, SL, EV, and residuals must be transformed consistently
  into side-relative economic orientation.
- Never combine long and short rows before verifying that signs cannot cancel.
- Preserve side through labels, base OOS, meta handoff, policy, and inference.

## 3. Features And Targets

- `feature_t` uses only data observable by `t`.
- `target_t` uses the future executable path after `t`.
- Soft-binary labels, economic utility, clean/dirty path flags, geometry, horizon,
  and cost assumptions must be recorded in the label manifest.
- Outcome-derived archetypes may describe training labels or meta targets, but
  cannot be inference inputs unless predicted from pre-entry features.

## 4. AE/GMM And Archetype State

Fit scaler/AE/GMM exactly once per model cycle on sampled beginning/middle/end
rows from the designated feature-selection/HPO reference period. Reuse the
exact serialized state for base/meta growing windows, final refits, replay, and
inference. The fit may use later covariates but never outcomes; disclose that
representation-selection exception in manifests and OOS claims. Keep cluster
IDs, posteriors, entropy, distance, reconstruction error, speed, and acceleration
aligned to the same frozen cluster ordering and input-feature order.

Both base and meta datasets are archetype-aware. Base rows must preserve their
observable label/state archetype and frozen AE/GMM context. Meta rows must carry
those base archetypes and may add meta-feature regimes, reliability priors,
support drift, leaf drift, residual context, and recent-performance context.

## 5. Residual And Surprise Semantics

Name residuals with their units and reference prediction:

- probability residual: realized soft/hard outcome minus predicted probability
- economic residual: realized net EV minus mapped/predicted net EV

Residual autocorrelation must use time-ordered OOS residuals and state its lag,
window, grouping, and minimum support. Report its signed value with residual mean:
positive autocorrelation means errors persist; negative means they alternate.

Hit-rate surprise is signed:

`recent_resolved_hit_rate - train_derived_expected_hit_rate`

Preserve positive and negative surprise, its standardized value, effective
support, half-life, and lookback. Current standard horizons are 3, 7, and 14 days
with each source window capped at four times its half-life. Only outcomes resolved
before the row timestamp may contribute.

## 6. Handoff Contract

Base-to-meta and meta-to-policy rows must record:

- timestamp, symbol, side, candidate ID
- OOF/frozen base and meta scores
- score/rank basis and candidate frontier
- archetype/policy archetype/local side archetype
- label and geometry manifest IDs
- feature/model/policy hashes where available
- decision fold and training cutoff
- probability/economic residual definitions and lagged history provenance
- signed hit-rate surprise, expected/actual rate, half-life, and support

## 7. Universe And Missing Data

Universe selection must be deterministic and point-in-time safe. Future listings,
future liquidity, or full-period survivorship cannot alter historical rows.

Missing data behavior must be explicit: drop, mask, causal fill, or fail closed.
Required inference features may not be silently synthesized. Cached or imputed
values are allowed only when frozen in the training-time transform contract.

## 8. Storage And Precision

Prefer `float32`, bounded integer types, categorical encoding, Parquet, and
partitioned/batched reads. Preserve higher precision only where numerical error
would affect labels, costs, ordering, or execution.

Every published dataset must record run ID, source period, feature contract,
universe, bar frequency, code revision, and row/column counts.
