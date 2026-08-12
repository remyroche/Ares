"""Vectorised strict-OOF H1--H5 materialisation from a strict event store.

This is the production counterpart to :mod:`causal_leaf_health`.  The latter
remains the small-data reference implementation.  This module deliberately
does *not* recreate a Python ``_FamilyStats`` object for every rule-family
event.  It uses the immutable event-store's ordered candidate streams and
DuckDB window/as-of operations for the high-cardinality H1/H2 work, then
loads only the frozen H3/H4/H5 family subset into pandas.

There are two important physical consequences:

* H1/H2 are aggregated directly into candidate features.  A full 7M-row
  family-state audit table is no longer an intermediate requirement.
* The retained state audit is deliberately narrow: selected H3/H4/H5 family
  rows plus all fields required to reproduce their derived features.  The
  artifact manifest calls this out so it cannot be mistaken for a complete
  raw-leaf or family-event export.

Every outcome-bearing cumulative value is joined with the strict predicate
``label_available_ts < feature_generation_ts``.  Month-frozen H2 and H3
snapshots are also causal: their source labels must have resolved before the
snapshot time.  This module never persists a raw leaf identifier.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd

from .causal_leaf_covariance import covariance_feature_names
from .causal_leaf_health import (
    DIRECTIONS,
    HEADS,
    SCHEMA,
    STATUS,
    CausalLeafHealthConfig,
    CausalLeafHealthError,
    _config_payload,
    _fit_ridge,
    _materialise_h4_h5,
    _relationship_break_columns,
)
from .strict_event_store import (
    CANDIDATE_COLUMNS,
    CONTRIBUTION_COLUMNS,
    StrictEventStore,
    load_strict_event_store,
)


_IDENTITY = (
    "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
)
_FAMILY = (
    "feature_contract_sha256", "side_name", "head_name", "rule_signature", "contribution_direction",
)
_EVENT_STATE_KEY = tuple(dict.fromkeys((*_IDENTITY, "feature_generation_ts", "label_available_ts", *_FAMILY, "family_ensemble_tree_contribution")))
_H1_METRICS = (
    "posterior_correctness", "posterior_lower_95", "row_support", "timestamp_support",
    "day_support", "symbol_support", "support_score", "calibration_residual",
    "economic_residual_bps", "false_positive_loss_bps",
)
_H2_METRICS = (
    "period_count", "supported_period_count", "observed_variance", "sampling_variance",
    "excess_variance", "robust_z_excess_variance", "sign_reversal_rate", "worst_damage_bps",
    "calibration_drift", "support_failure", "instability",
)
_H3_METRICS = (
    "availability", "compatibility", "expected_error_bps", "confidence", "unexplained_break",
)


def _literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _glob(root: Path) -> str:
    return _literal(str(root / "**" / "*.parquet"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _asof_context_timeline(
    context: pd.DataFrame,
    columns: Sequence[str],
    config: CausalLeafHealthConfig,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate the compact shared causal timeline before registering it."""

    if "candidate_id" in context.columns:
        raise CausalLeafHealthError(
            "vectorized H1--H5 requires a shared causal context timeline; "
            "candidate-specific context would defeat the bounded event-store contract"
        )
    selected = tuple(map(str, columns[: int(config.covariance_max_fields)]))
    required = {"regime_available_utc", *selected}
    missing = sorted(required.difference(context.columns))
    if missing:
        raise CausalLeafHealthError(f"causal context lacks declared fields: {missing}")
    forbidden = (
        "target", "label", "outcome", "future", "realized", "realised", "pnl", "net_ev",
        "gross_ev", "mfe", "mae", "barrier", "timeout", "exit", "post_entry", "postentry",
    )
    bad = [str(name) for name in context.columns if any(token in str(name).lower() for token in forbidden)]
    if bad:
        raise CausalLeafHealthError(f"causal context includes outcome-derived columns: {bad[:8]}")
    result = context.loc[:, ["regime_available_utc", *selected]].copy()
    result["regime_available_utc"] = pd.to_datetime(result["regime_available_utc"], utc=True, errors="coerce")
    if result["regime_available_utc"].isna().any() or result["regime_available_utc"].duplicated().any():
        raise CausalLeafHealthError("causal context timestamps must be finite and unique")
    for name in selected:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if not np.isfinite(result.loc[:, list(selected)].to_numpy(dtype=float)).all():
        raise CausalLeafHealthError("causal context values must be finite")
    return result.sort_values("regime_available_utc", kind="stable"), selected


def _metric_expressions(section: str, metrics: Sequence[str], alias: str = "s") -> tuple[list[str], list[str]]:
    """Contribution-weighted direct-candidate feature expressions.

    The denominator intentionally includes all active contribution mass.  A
    selected-only H3/H4 state consequently has zero contribution outside the
    frozen subset rather than silently renormalising that subset to one.
    """

    expressions: list[str] = []
    names: list[str] = []
    for head in HEADS:
        for direction in DIRECTIONS:
            predicate = f"{alias}.head_name={_literal(head)} AND {alias}.contribution_direction={_literal(direction)}"
            denominator = f"SUM(CASE WHEN {predicate} THEN abs({alias}.family_ensemble_tree_contribution) ELSE 0.0 END)"
            for metric in metrics:
                column = f"{section}_{metric}"
                name = f"base_health__{section}__{head}__{direction}__{metric}"
                expressions.append(
                    f"CAST(COALESCE(SUM(CASE WHEN {predicate} THEN abs({alias}.family_ensemble_tree_contribution) * COALESCE({alias}.\"{column}\", 0.0) ELSE 0.0 END) / NULLIF({denominator}, 0.0), 0.0) AS FLOAT) AS \"{name}\""
                )
                names.append(name)
            name = f"base_health__{section}__{head}__{direction}__active_abs_contribution"
            expressions.append(f"CAST(COALESCE({denominator}, 0.0) AS FLOAT) AS \"{name}\"")
            names.append(name)
    return expressions, names


def _copy_direct_sections(
    connection: duckdb.DuckDBPyConnection, *, candidate_view: str, state_view: str,
    output: Path, sections: Sequence[tuple[str, Sequence[str]]],
) -> list[str]:
    expressions: list[str] = []
    names: list[str] = []
    for section, metrics in sections:
        current, current_names = _metric_expressions(section, metrics)
        expressions.extend(current); names.extend(current_names)
    identity = ", ".join(f"b.{item}" for item in _IDENTITY)
    grouped = ", ".join(f"b.{item}" for item in _IDENTITY)
    sql = (
        f"COPY (SELECT {identity}, {', '.join(expressions)} "
        f"FROM (SELECT DISTINCT {', '.join(_IDENTITY)} FROM {candidate_view}) b "
        f"LEFT JOIN {state_view} s USING ({', '.join(_IDENTITY)}) "
        f"GROUP BY {grouped}) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(sql)
    return names


def _candidate_context(connection: duckdb.DuckDBPyConnection, timeline: pd.DataFrame) -> None:
    connection.register("__context_timeline", timeline)
    # One context row per tradable candidate/side, not per semantic head.
    connection.execute(
        """
        CREATE TABLE candidate_context AS
        WITH base AS (
            SELECT candidate_id, decision_ts, side_name, fold_id, transport, meta_partition,
                   min(feature_generation_ts) AS feature_generation_ts
            FROM candidates
            GROUP BY candidate_id, decision_ts, side_name, fold_id, transport, meta_partition
        )
        SELECT b.*, r.*
        FROM base b
        ASOF LEFT JOIN __context_timeline r
          ON b.feature_generation_ts >= r.regime_available_utc
        """
    )
    absent = connection.execute("SELECT count(*) FROM candidate_context WHERE regime_available_utc IS NULL").fetchone()[0]
    if int(absent):
        raise CausalLeafHealthError("candidate lacks a prior available causal context row")


def _create_h1_state(connection: duckdb.DuckDBPyConnection, config: CausalLeafHealthConfig) -> None:
    """Create contribution-level H1 snapshots with windowed/as-of SQL only."""

    # Facts are never selected from an outcome-bearing contribution store: all
    # outcomes remain in the verified candidate part and are joined by full
    # strict identity here.
    connection.execute(
        """
        CREATE TEMP VIEW family_events AS
        SELECT f.candidate_id, f.__ts__ AS decision_ts, f.side_name, f.head_name,
               f.fold_id, f.transport, f.meta_partition, f.feature_contract_sha256,
               f.rule_signature, f.contribution_direction, f.family_ensemble_tree_contribution,
               c.feature_generation_ts, c.label_available_ts, c.semantic_label,
               c.head_prediction, c.net_bps, c.base_expected_bps, c.asset
        FROM contributions f
        INNER JOIN candidates c
          ON c.candidate_id=f.candidate_id AND c.decision_ts=f.__ts__
         AND c.side_name=f.side_name AND c.head_name=f.head_name
         AND c.fold_id=f.fold_id AND c.transport=f.transport
         AND c.meta_partition=f.meta_partition
         AND c.feature_contract_sha256=f.feature_contract_sha256
        """
    )
    # Marker columns make distinct time/day/symbol support cumulative without
    # requiring a Python set per family.
    connection.execute(
        """
        CREATE TEMP VIEW family_marked AS
        SELECT *,
          CASE WHEN row_number() OVER (PARTITION BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction, decision_ts ORDER BY label_available_ts, candidate_id) = 1 THEN 1 ELSE 0 END AS new_timestamp,
          CASE WHEN row_number() OVER (PARTITION BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction, CAST(decision_ts AS DATE) ORDER BY label_available_ts, candidate_id) = 1 THEN 1 ELSE 0 END AS new_day,
          CASE WHEN row_number() OVER (PARTITION BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction, asset ORDER BY label_available_ts, candidate_id) = 1 THEN 1 ELSE 0 END AS new_symbol
        FROM family_events
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE family_h1_resolution AS
        WITH grouped AS (
          SELECT feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction,
                 label_available_ts,
                 count(*) AS rows_delta, sum(semantic_label) AS success_delta,
                 sum(head_prediction) AS prediction_delta,
                 sum(net_bps) AS net_delta, sum(base_expected_bps) AS expected_delta,
                 sum(CASE WHEN head_prediction >= 0.5 AND semantic_label <= 0 THEN greatest(-net_bps, 0.0) ELSE 0.0 END) AS fpl_delta,
                 sum(new_timestamp) AS timestamp_delta, sum(new_day) AS day_delta, sum(new_symbol) AS symbol_delta
          FROM family_marked
          GROUP BY ALL
        )
        SELECT *,
          sum(rows_delta) OVER w AS rows, sum(success_delta) OVER w AS successes,
          sum(prediction_delta) OVER w AS predictions, sum(net_delta) OVER w AS nets,
          sum(expected_delta) OVER w AS expecteds, sum(fpl_delta) OVER w AS fpls,
          sum(timestamp_delta) OVER w AS timestamps, sum(day_delta) OVER w AS days,
          sum(symbol_delta) OVER w AS symbols
        FROM grouped
        WINDOW w AS (PARTITION BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction ORDER BY label_available_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE scope_h1_resolution AS
        WITH grouped AS (
          SELECT feature_contract_sha256, side_name, head_name, label_available_ts,
                 count(*) AS rows_delta, sum(semantic_label) AS success_delta
          FROM resolution_candidates GROUP BY ALL
        )
        SELECT *, sum(rows_delta) OVER w AS rows, sum(success_delta) OVER w AS successes
        FROM grouped
        WINDOW w AS (PARTITION BY feature_contract_sha256, side_name, head_name ORDER BY label_available_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE global_h1_resolution AS
        WITH grouped AS (
          SELECT feature_contract_sha256, label_available_ts, count(*) AS rows_delta, sum(semantic_label) AS success_delta
          FROM resolution_candidates GROUP BY ALL
        )
        SELECT *, sum(rows_delta) OVER w AS rows, sum(success_delta) OVER w AS successes
        FROM grouped
        WINDOW w AS (PARTITION BY feature_contract_sha256 ORDER BY label_available_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        """
    )
    alpha, beta = float(config.global_alpha), float(config.global_beta)
    side_strength, family_strength = float(config.side_head_prior_strength), float(config.family_prior_strength)
    connection.execute(
        f"""
        CREATE TEMP TABLE h1_states AS
        WITH joined AS (
          SELECT e.*, COALESCE(f.rows, 0) AS f_rows, COALESCE(f.successes, 0.0) AS f_successes,
                 COALESCE(f.predictions, 0.0) AS f_predictions, COALESCE(f.nets, 0.0) AS f_nets,
                 COALESCE(f.expecteds, 0.0) AS f_expecteds, COALESCE(f.fpls, 0.0) AS f_fpls,
                 COALESCE(f.timestamps, 0) AS f_timestamps, COALESCE(f.days, 0) AS f_days,
                 COALESCE(f.symbols, 0) AS f_symbols,
                 COALESCE(sh.rows, 0) AS sh_rows, COALESCE(sh.successes, 0.0) AS sh_successes,
                 COALESCE(g.rows, 0) AS g_rows, COALESCE(g.successes, 0.0) AS g_successes
          FROM family_events e
          ASOF LEFT JOIN family_h1_resolution f
            ON e.feature_contract_sha256=f.feature_contract_sha256 AND e.side_name=f.side_name
           AND e.head_name=f.head_name AND e.rule_signature=f.rule_signature
           AND e.contribution_direction=f.contribution_direction
           AND e.feature_generation_ts > f.label_available_ts
          ASOF LEFT JOIN scope_h1_resolution sh
            ON e.feature_contract_sha256=sh.feature_contract_sha256 AND e.side_name=sh.side_name
           AND e.head_name=sh.head_name AND e.feature_generation_ts > sh.label_available_ts
          ASOF LEFT JOIN global_h1_resolution g
            ON e.feature_contract_sha256=g.feature_contract_sha256
           AND e.feature_generation_ts > g.label_available_ts
        ), priors AS (
          SELECT *,
            (g_successes + {alpha}) / greatest(g_rows + {alpha + beta}, 1e-12) AS global_mean,
            (sh_successes + {side_strength} * ((g_successes + {alpha}) / greatest(g_rows + {alpha + beta}, 1e-12))) / greatest(sh_rows + {side_strength}, 1e-12) AS side_mean
          FROM joined
        ), posterior AS (
          SELECT *,
            f_successes + {family_strength} * side_mean AS post_alpha,
            (f_rows - f_successes) + {family_strength} * (1.0 - side_mean) AS post_beta
          FROM priors
        )
        SELECT candidate_id, decision_ts, feature_generation_ts, label_available_ts, side_name, head_name,
               fold_id, transport, meta_partition, feature_contract_sha256, rule_signature,
               contribution_direction, family_ensemble_tree_contribution,
               CAST(post_alpha / greatest(post_alpha + post_beta, 1e-12) AS FLOAT) AS h1_posterior_correctness,
               CAST(greatest(0.0, post_alpha / greatest(post_alpha + post_beta, 1e-12) - 1.96 * sqrt(greatest((post_alpha / greatest(post_alpha + post_beta, 1e-12)) * (1.0 - post_alpha / greatest(post_alpha + post_beta, 1e-12)) / (post_alpha + post_beta + 1.0), 0.0))) AS FLOAT) AS h1_posterior_lower_95,
               CAST(f_rows AS FLOAT) AS h1_row_support, CAST(f_timestamps AS FLOAT) AS h1_timestamp_support,
               CAST(f_days AS FLOAT) AS h1_day_support, CAST(f_symbols AS FLOAT) AS h1_symbol_support,
               CAST(least(1.0, least(f_timestamps / {float(config.min_timestamp_support)}, least(f_days / {float(config.min_day_support)}, f_symbols / {float(config.min_symbol_support)}))) AS FLOAT) AS h1_support_score,
               CAST(CASE WHEN f_rows > 0 THEN f_successes / f_rows - f_predictions / f_rows ELSE 0.0 END AS FLOAT) AS h1_calibration_residual,
               CAST(CASE WHEN f_rows > 0 THEN f_nets / f_rows - f_expecteds / f_rows ELSE 0.0 END AS FLOAT) AS h1_economic_residual_bps,
               CAST(CASE WHEN f_rows > 0 THEN f_fpls / f_rows ELSE 0.0 END AS FLOAT) AS h1_false_positive_loss_bps
        FROM posterior
        """
    )


def _create_h2_state(connection: duckdb.DuckDBPyConnection, config: CausalLeafHealthConfig) -> None:
    """Month-frozen portability snapshots; no event-wise Python update loop."""

    # A H12 label is available well inside the reference's 24h month-close
    # lag.  We nevertheless require resolution before the score snapshot in
    # the period source so a changed label contract fails causally rather than
    # accidentally introducing look-ahead.
    connection.execute(
        """
        CREATE TEMP TABLE family_period AS
        SELECT feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction,
               date_trunc('month', decision_ts) AS period_start,
               count(*) AS rows, count(DISTINCT decision_ts) AS timestamps,
               count(DISTINCT CAST(decision_ts AS DATE)) AS days, count(DISTINCT asset) AS symbols,
               avg(net_bps - base_expected_bps) AS effect_bps,
               CASE WHEN count(*) > 1 THEN sqrt(greatest(avg(power((net_bps - base_expected_bps), 2)) - power(avg(net_bps - base_expected_bps), 2), 0.0) / count(*)) ELSE NULL END AS effect_se_bps,
               avg(semantic_label - head_prediction) AS calibration_residual
        FROM family_events
        GROUP BY ALL
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE score_months AS
        SELECT DISTINCT feature_contract_sha256, side_name, head_name,
               date_trunc('month', feature_generation_ts) AS score_month
        FROM candidates
        """
    )
    # period ends must be before score-month start minus 24 hours, matching
    # the reference's month-boundary frozen snapshot exactly.
    connection.execute(
        f"""
        CREATE TEMP TABLE h2_base AS
        WITH closed AS (
          SELECT m.score_month, p.*,
                 (p.rows >= {int(config.min_period_rows)} AND p.timestamps >= {int(config.min_timestamp_support)}
                  AND p.days >= {int(config.min_day_support)} AND p.symbols >= {int(config.min_symbol_support)}) AS supported
          FROM score_months m
          JOIN family_period p
            ON m.feature_contract_sha256=p.feature_contract_sha256 AND m.side_name=p.side_name AND m.head_name=p.head_name
           AND p.period_start + INTERVAL 1 MONTH <= m.score_month - INTERVAL {int(config.period_close_lag_hours)} HOUR
        ), closed_with_prior AS (
          SELECT *,
                 lag(effect_bps) OVER w AS previous_effect_bps,
                 lag(effect_se_bps) OVER w AS previous_effect_se_bps,
                 lag(supported) OVER w AS previous_supported
          FROM closed
          WINDOW w AS (PARTITION BY score_month, feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction ORDER BY period_start)
        ), basic AS (
          SELECT score_month, feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction,
                 count(*) AS period_count, count(*) FILTER (WHERE supported) AS supported_period_count,
                 sum(rows) FILTER (WHERE supported) AS weight_sum,
                 sum(rows * effect_bps) FILTER (WHERE supported) AS effect_sum,
                 sum(rows * power(effect_bps, 2)) FILTER (WHERE supported) AS effect_sq_sum,
                 avg(power(effect_se_bps, 2)) FILTER (WHERE supported AND effect_se_bps IS NOT NULL) AS sampling_variance,
                 max(greatest(-effect_bps, 0.0)) FILTER (WHERE supported) AS worst_damage_bps,
                 sum(rows * abs(calibration_residual)) FILTER (WHERE supported) AS calibration_sum,
                 avg(CASE WHEN supported AND previous_supported
                                AND abs(effect_bps) > 1.96 * coalesce(effect_se_bps, 'Infinity'::DOUBLE)
                                AND abs(previous_effect_bps) > 1.96 * coalesce(previous_effect_se_bps, 'Infinity'::DOUBLE)
                          THEN CAST(sign(effect_bps) <> sign(previous_effect_bps) AS DOUBLE)
                     ELSE NULL END) AS sign_reversal_rate
          FROM closed_with_prior GROUP BY ALL
        )
        SELECT *,
          CASE WHEN supported_period_count > 0 THEN effect_sq_sum / weight_sum - power(effect_sum / weight_sum, 2) ELSE NULL END AS observed_variance,
          CASE WHEN supported_period_count > 0 THEN greatest(0.0, effect_sq_sum / weight_sum - power(effect_sum / weight_sum, 2) - coalesce(sampling_variance, 0.0)) ELSE NULL END AS excess_variance,
          CASE WHEN period_count > 0 THEN 1.0 - supported_period_count::DOUBLE / period_count ELSE 1.0 END AS support_failure,
          CASE WHEN supported_period_count > 0 THEN calibration_sum / weight_sum ELSE NULL END AS calibration_drift
        FROM basic
        """
    )
    # robust_z uses a support bucket just as the reference.  DuckDB's median
    # keeps it spillable; it avoids a huge per-month Python list of families.
    connection.execute(
        """
        CREATE TEMP TABLE h2_snapshot AS
        WITH bucketed AS (
          SELECT *, floor(log2(greatest(weight_sum, 1)))::INTEGER AS support_bucket
          FROM h2_base
        ), med AS (
          SELECT *, median(excess_variance) OVER (PARTITION BY score_month, feature_contract_sha256, side_name, head_name, support_bucket) AS excess_median
          FROM bucketed
        ), mad AS (
          SELECT *, median(abs(excess_variance - excess_median)) OVER (PARTITION BY score_month, feature_contract_sha256, side_name, head_name, support_bucket) AS excess_mad
          FROM med
        ), damage AS (
          SELECT *, median(worst_damage_bps) OVER (PARTITION BY score_month) AS damage_median
          FROM mad
        )
        SELECT *,
          CASE WHEN excess_variance IS NULL THEN NULL ELSE (excess_variance - excess_median) / greatest(1.4826 * coalesce(excess_mad, 0.0), 1e-8) END AS robust_z,
          coalesce(sign_reversal_rate, 0.0) AS sign_reversal_rate,
          CASE WHEN excess_variance IS NULL THEN NULL ELSE coalesce((excess_variance - excess_median) / greatest(1.4826 * coalesce(excess_mad, 0.0), 1e-8), 0.0) + coalesce(worst_damage_bps, 0.0) / greatest(coalesce(damage_median, 1.0), 1.0) + 0.5 * coalesce(support_failure, 1.0) END AS instability
        FROM damage
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE h2_states AS
        SELECT h.candidate_id, h.decision_ts, h.feature_generation_ts, h.label_available_ts,
               h.side_name, h.head_name, h.fold_id, h.transport, h.meta_partition,
               h.feature_contract_sha256, h.rule_signature, h.contribution_direction,
               h.family_ensemble_tree_contribution,
               CAST(coalesce(s.period_count, 0.0) AS FLOAT) AS h2_period_count,
               CAST(coalesce(s.supported_period_count, 0.0) AS FLOAT) AS h2_supported_period_count,
               CAST(coalesce(s.observed_variance, 0.0) AS FLOAT) AS h2_observed_variance,
               CAST(coalesce(s.sampling_variance, 0.0) AS FLOAT) AS h2_sampling_variance,
               CAST(coalesce(s.excess_variance, 0.0) AS FLOAT) AS h2_excess_variance,
               CAST(coalesce(s.robust_z, 0.0) AS FLOAT) AS h2_robust_z_excess_variance,
               CAST(coalesce(s.sign_reversal_rate, 0.0) AS FLOAT) AS h2_sign_reversal_rate,
               CAST(coalesce(s.worst_damage_bps, 0.0) AS FLOAT) AS h2_worst_damage_bps,
               CAST(coalesce(s.calibration_drift, 0.0) AS FLOAT) AS h2_calibration_drift,
               CAST(coalesce(s.support_failure, 1.0) AS FLOAT) AS h2_support_failure,
               CAST(coalesce(s.instability, 0.0) AS FLOAT) AS h2_instability
        FROM h1_states h
        LEFT JOIN h2_snapshot s
          ON h.feature_contract_sha256=s.feature_contract_sha256 AND h.side_name=s.side_name
         AND h.head_name=s.head_name AND h.rule_signature=s.rule_signature
         AND h.contribution_direction=s.contribution_direction
         AND date_trunc('month', h.feature_generation_ts)=s.score_month
        """
    )


def _h3_states(
    connection: duckdb.DuckDBPyConnection, *, selected: frozenset[tuple[str, str, str, str, str]],
    context_columns: Sequence[str], config: CausalLeafHealthConfig, temporary: Path,
) -> Path:
    """Fit the small month-frozen H3 ridge set only for selected families."""

    output = temporary / "h3_selected_states.parquet"
    empty_columns = [
        *_EVENT_STATE_KEY, *[f"h3_{name}" for name in _H3_METRICS],
    ]
    if not selected:
        pd.DataFrame(columns=empty_columns).to_parquet(output, index=False, compression="zstd")
        return output
    keys = pd.DataFrame(sorted(selected), columns=list(_FAMILY))
    connection.register("h3_keys", keys)
    selected_frame = connection.execute(
        f"""
        SELECT e.*, cc.regime_available_utc, {', '.join(f'cc."{name}"' for name in context_columns)}
        FROM family_events e
        INNER JOIN h3_keys k USING ({', '.join(_FAMILY)})
        INNER JOIN candidate_context cc USING ({', '.join(_IDENTITY)})
        ORDER BY e.feature_generation_ts, e.label_available_ts, e.candidate_id, e.rule_signature
        """
    ).fetch_df()
    if selected_frame.empty:
        pd.DataFrame(columns=empty_columns).to_parquet(output, index=False, compression="zstd")
        return output
    selected_frame["feature_generation_ts"] = pd.to_datetime(selected_frame["feature_generation_ts"], utc=True)
    selected_frame["label_available_ts"] = pd.to_datetime(selected_frame["label_available_ts"], utc=True)
    selected_frame["__score_month__"] = selected_frame["feature_generation_ts"].dt.strftime("%Y-%m")
    rows: list[pd.DataFrame] = []
    effective = (
        pd.Timestamp(pd.to_datetime(config.family_selection_effective_utc, utc=True))
        if config.family_selection_effective_utc is not None else None
    )
    # A maximum of selected_families x months small ridge fits.  This is the
    # only intentional Python loop in the production path; no full family
    # population is converted into Python objects.
    for month, scored in selected_frame.groupby("__score_month__", sort=True, observed=True):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        train = selected_frame.loc[selected_frame["label_available_ts"].lt(start)].copy()
        for key, target in scored.groupby(list(_FAMILY), sort=False, observed=True):
            out = target.loc[:, list(_EVENT_STATE_KEY)].copy()
            active = (
                np.ones(len(target), dtype=bool)
                if effective is None else target["feature_generation_ts"].ge(effective).to_numpy(bool)
            )
            prior = train.loc[(train[list(_FAMILY)] == pd.Series(key, index=list(_FAMILY))).all(axis=1)]
            if len(prior) > int(config.h3_max_rows_per_family):
                prior = prior.nlargest(int(config.h3_max_rows_per_family), "label_available_ts", keep="last")
            model = None
            if len(prior) >= int(config.h3_min_rows):
                values = prior.loc[:, list(context_columns)].to_numpy(dtype=np.float64)
                residual = (prior["net_bps"].to_numpy(dtype=float) - prior["base_expected_bps"].to_numpy(dtype=float))
                from collections import deque
                model = _fit_ridge(deque(zip(values, residual, strict=True)), config)
            matrix = target.loc[:, list(context_columns)].to_numpy(dtype=np.float64)
            valid = np.isfinite(matrix).all(axis=1) & active
            availability = np.zeros(len(target), dtype=np.float32)
            compatibility = np.zeros(len(target), dtype=np.float32)
            expected = np.zeros(len(target), dtype=np.float32)
            confidence = np.zeros(len(target), dtype=np.float32)
            unexplained = np.zeros(len(target), dtype=np.float32)
            if model is not None:
                prediction = np.asarray([model.predict(row) for row in matrix[valid]], dtype=np.float64)
                conf = float(model.rows / (model.rows + float(config.h3_min_rows)))
                comp = np.exp(-np.abs(prediction) / max(float(model.residual_scale), 1e-6))
                availability[valid] = 1.0; expected[valid] = prediction.astype(np.float32)
                confidence[valid] = conf; compatibility[valid] = comp.astype(np.float32)
                unexplained[valid] = ((1.0 - comp) * conf).astype(np.float32)
            out["h3_availability"] = availability; out["h3_compatibility"] = compatibility
            out["h3_expected_error_bps"] = expected; out["h3_confidence"] = confidence
            out["h3_unexplained_break"] = unexplained
            rows.append(out)
    pd.concat(rows, ignore_index=True).to_parquet(output, index=False, compression="zstd")
    return output


def _selected_state_audit(
    connection: duckdb.DuckDBPyConnection, *, selection: frozenset[tuple[str, str, str, str, str]],
    covariance_selection: frozenset[tuple[str, str, str, str, str]],
    relationship_selection: frozenset[tuple[str, str, str, str, str]],
    context_columns: Sequence[str], h3_path: Path, config: CausalLeafHealthConfig, output: Path,
) -> int:
    """Persist only source-filtered selected H3/H4/H5 state rows."""

    if not selection:
        pd.DataFrame().to_parquet(output, index=False, compression="zstd")
        return 0
    connection.register("selected_audit_keys", pd.DataFrame(sorted(selection), columns=list(_FAMILY)))
    connection.register("selected_covariance_keys", pd.DataFrame(sorted(covariance_selection), columns=list(_FAMILY)))
    connection.register("selected_relationship_keys", pd.DataFrame(sorted(relationship_selection), columns=list(_FAMILY)))
    h3_literal = _literal(str(h3_path))
    active_clause = "TRUE" if config.family_selection_effective_utc is None else f"h.feature_generation_ts >= TIMESTAMPTZ {_literal(config.family_selection_effective_utc)}"
    context_select = ", ".join(f"cc.\"{name}\"" for name in context_columns)
    context_join = " AND ".join(f"h.{name}=cc.{name}" for name in _IDENTITY)
    h3_join = " AND ".join(f"h.{name}=x.{name}" for name in _EVENT_STATE_KEY)
    connection.execute(
        f"""
        COPY (
          SELECT h.*, {context_select}, cc.regime_available_utc,
                 CAST({active_clause} AND ck.feature_contract_sha256 IS NOT NULL AS FLOAT) AS h4_selection_active,
                 CAST({active_clause} AND rk.feature_contract_sha256 IS NOT NULL AS FLOAT) AS h5_selection_active,
                 coalesce(x.h3_availability, 0.0)::FLOAT AS h3_availability,
                 coalesce(x.h3_compatibility, 0.0)::FLOAT AS h3_compatibility,
                 coalesce(x.h3_expected_error_bps, 0.0)::FLOAT AS h3_expected_error_bps,
                 coalesce(x.h3_confidence, 0.0)::FLOAT AS h3_confidence,
                 coalesce(x.h3_unexplained_break, 0.0)::FLOAT AS h3_unexplained_break
          FROM h2_states h
          INNER JOIN selected_audit_keys k USING ({', '.join(_FAMILY)})
          LEFT JOIN selected_covariance_keys ck USING ({', '.join(_FAMILY)})
          LEFT JOIN selected_relationship_keys rk USING ({', '.join(_FAMILY)})
          INNER JOIN candidate_context cc ON {context_join}
          LEFT JOIN read_parquet({h3_literal}) x
            ON {h3_join}
        ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])


def _merge_health(connection: duckdb.DuckDBPyConnection, paths: Sequence[Path], names: Sequence[Sequence[str]], output: Path) -> int:
    identity = ", ".join(_IDENTITY)
    aliases = [chr(ord("a") + index) for index in range(len(paths))]
    select = [f"{aliases[0]}.{item}" for item in _IDENTITY]
    for alias, group in zip(aliases, names, strict=True):
        select.extend(f"{alias}.\"{name}\"" for name in group)
    joins = f"read_parquet({_literal(str(paths[0]))}) {aliases[0]}"
    for alias, path in zip(aliases[1:], paths[1:], strict=True):
        joins += f" INNER JOIN read_parquet({_literal(str(path))}) {alias} USING ({identity})"
    connection.execute(
        f"COPY (SELECT {', '.join(select)} FROM {joins} ORDER BY {aliases[0]}.transport, {aliases[0]}.meta_partition, {aliases[0]}.decision_ts, {aliases[0]}.candidate_id) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])


def materialize_strict_oof_causal_leaf_health_vectorized(
    event_store: StrictEventStore | str | Path,
    output_dir: str | Path,
    *, causal_context: pd.DataFrame, context_feature_columns: Sequence[str],
    config: CausalLeafHealthConfig = CausalLeafHealthConfig(),
    threads: int = 2, memory_limit: str = "2GB",
    verify_event_store_parts: bool = False,
) -> Path:
    """Materialise the H1--H5 artifact through the canonical event store.

    The function is intentionally fail-closed: the store manifest is verified
    before any query; output is atomic; and selected-state source filtering is
    explicit in the artifact manifest.  ``threads=2`` avoids oversubscribing
    a host while DuckDB's external algorithms use the temporary directory.
    """

    started = time.monotonic()
    config.validate()
    # A completed store has already hashed every physical part before atomic
    # sealing.  Reuse checks the sealed manifest/index and strict-root lineage
    # by default rather than re-hashing multi-GB immutable files on every
    # ablation.  A caller can opt into a full physical checksum audit.
    store = event_store if isinstance(event_store, StrictEventStore) else load_strict_event_store(
        event_store, verify_parts=verify_event_store_parts, verify_source=True,
    )
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite causal leaf health artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    timeline, context_columns = _asof_context_timeline(causal_context, context_feature_columns, config)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        connection = duckdb.connect(database=str(temporary / "health_vectorized.duckdb"))
        connection.execute(f"PRAGMA threads={max(1, int(threads))}")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(temporary / 'duckdb_tmp'))}")
        # The canonical store sorted these two streams once.  Every health
        # consumer reuses them rather than paying another population-wide
        # score/resolution sort; family joins occur only where state is needed.
        connection.execute(f"CREATE VIEW candidates AS SELECT {', '.join(CANDIDATE_COLUMNS)} FROM read_parquet({_glob(store.root / 'score_order')}, hive_partitioning=false)")
        connection.execute(f"CREATE VIEW resolution_candidates AS SELECT {', '.join(CANDIDATE_COLUMNS)} FROM read_parquet({_glob(store.root / 'resolution_order')}, hive_partitioning=false)")
        connection.execute(f"CREATE VIEW contributions AS SELECT {', '.join(CONTRIBUTION_COLUMNS)} FROM read_parquet({_glob(store.root / 'contribution_parts')}, hive_partitioning=false)")
        invalid = connection.execute("SELECT count(*) FROM contributions WHERE rule_signature IS NULL OR trim(rule_signature)='' OR contribution_direction NOT IN ('positive', 'negative') OR family_ensemble_tree_contribution=0").fetchone()[0]
        if int(invalid):
            raise CausalLeafHealthError("event-store contributions violate the token-free numeric contract")
        late_horizon = connection.execute(
            "SELECT count(*) FROM candidates WHERE label_available_ts > decision_ts + INTERVAL 24 HOUR"
        ).fetchone()[0]
        if int(late_horizon):
            raise CausalLeafHealthError(
                "vectorised monthly H2 requires labels to resolve within its declared 24h close lag"
            )
        _candidate_context(connection, timeline)
        _create_h1_state(connection, config)
        _create_h2_state(connection, config)
        h1_path = temporary / "h1.parquet"
        h2_path = temporary / "h2.parquet"
        h1_names = _copy_direct_sections(connection, candidate_view="candidates", state_view="h1_states", output=h1_path, sections=(("h1", _H1_METRICS),))
        h2_names = _copy_direct_sections(connection, candidate_view="candidates", state_view="h2_states", output=h2_path, sections=(("h2", _H2_METRICS),))
        selected_context = config.selected_context_families
        h3_path = _h3_states(connection, selected=selected_context, context_columns=context_columns, config=config, temporary=temporary)
        connection.execute(f"CREATE TEMP VIEW h3_states AS SELECT * FROM read_parquet({_literal(str(h3_path))})")
        # All contribution rows are present in h2_states; H3 is a sparse
        # left join, preserving non-selected denominator mass as zeros.
        connection.execute(
            f"""
            CREATE TEMP VIEW h3_full AS
            SELECT h.*, coalesce(s.h3_availability, 0.0)::FLOAT AS h3_availability,
                   coalesce(s.h3_compatibility, 0.0)::FLOAT AS h3_compatibility,
                   coalesce(s.h3_expected_error_bps, 0.0)::FLOAT AS h3_expected_error_bps,
                   coalesce(s.h3_confidence, 0.0)::FLOAT AS h3_confidence,
                   coalesce(s.h3_unexplained_break, 0.0)::FLOAT AS h3_unexplained_break
            FROM h2_states h LEFT JOIN h3_states s
              USING ({', '.join(_EVENT_STATE_KEY)})
            """
        )
        h3_features_path = temporary / "h3.parquet"
        h3_names = _copy_direct_sections(connection, candidate_view="candidates", state_view="h3_full", output=h3_features_path, sections=(("h3", _H3_METRICS),))
        all_selected = frozenset(set(config.selected_context_families) | set(config.selected_covariance_families) | set(config.selected_relationship_families))
        selected_state_path = temporary / "base_leaf_family_candidate_states.parquet"
        selected_rows = _selected_state_audit(
            connection, selection=all_selected,
            covariance_selection=config.selected_covariance_families,
            relationship_selection=config.selected_relationship_families,
            context_columns=context_columns, h3_path=h3_path, config=config, output=selected_state_path,
        )
        selected_states = pd.read_parquet(selected_state_path) if selected_rows else pd.DataFrame()
        covariance, relationships, field_audit = _materialise_h4_h5(selected_states, context_columns=context_columns, config=config)
        covariance_path = temporary / "leaf_covariance_diagnostics.parquet"
        relationship_path = temporary / "leaf_relationship_breaks.parquet"
        covariance.to_parquet(covariance_path, index=False, compression="zstd")
        relationships.to_parquet(relationship_path, index=False, compression="zstd")
        # H4 follows the same full-mass direct aggregation rule as H3.
        if covariance.empty:
            h4_metrics = [name.removeprefix("base_health__h4__") for name in covariance_feature_names("base_health__h4")] + ["availability"]
            zeros = ", ".join(f"0.0::FLOAT AS \"h4_{metric}\"" for metric in h4_metrics)
            connection.execute(f"CREATE TEMP VIEW h4_full AS SELECT h2.*, {zeros} FROM h2_states h2")
        else:
            connection.execute(
                f"CREATE TEMP VIEW h4_selected AS "
                f"SELECT c.*, coalesce(s.h4_selection_active, 0.0)::FLOAT AS h4_selection_active "
                f"FROM read_parquet({_literal(str(covariance_path))}) c "
                f"LEFT JOIN read_parquet({_literal(str(selected_state_path))}) s "
                f"USING ({', '.join([*_IDENTITY, 'head_name', 'contribution_direction', 'rule_signature'])})"
            )
            h4_names_source = [name for name in covariance.columns if name.startswith("base_health__h4__")]
            # A selection fitted at its predecessor cutoff is unavailable
            # before that cutoff.  Retain those rows only in the narrow audit;
            # candidate features must be exactly zero until activation.
            select = ", ".join(
                f"(coalesce(c.\"{name}\", 0.0) * coalesce(c.h4_selection_active, 0.0))::FLOAT AS \"h4_{name.removeprefix('base_health__h4__')}\""
                for name in h4_names_source
            )
            connection.execute(
                f"CREATE TEMP VIEW h4_full AS SELECT h2.*, {select}, CAST(coalesce(c.h4_selection_active, 0.0) > 0.0 AS FLOAT) AS h4_availability FROM h2_states h2 LEFT JOIN h4_selected c USING ({', '.join([*_IDENTITY, 'head_name', 'contribution_direction', 'rule_signature'])})"
            )
            h4_metrics = [name.removeprefix("base_health__h4__") for name in h4_names_source] + ["availability"]
        h4_path = temporary / "h4.parquet"
        h4_names = _copy_direct_sections(connection, candidate_view="candidates", state_view="h4_full", output=h4_path, sections=(("h4", tuple(h4_metrics)),))
        # H5 retains its selected portable-economic denominator by design.
        h5_path = temporary / "h5.parquet"
        from .causal_leaf_health_streaming import _copy_h5  # local: no import cycle
        h5_names = _copy_h5(connection, candidate_view="candidates", relationship_path=relationship_path, output=h5_path, pairs=sorted(_relationship_break_columns(context_columns)))
        health_path = temporary / "base_leaf_health_features_oof.parquet"
        health_rows = _merge_health(connection, (h1_path, h2_path, h3_features_path, h4_path, h5_path), (h1_names, h2_names, h3_names, h4_names, h5_names), health_path)
        # Keep full selected audit only; final diagnostic tables remain
        # compact, not a duplicate of the large causal state table.
        period_output = temporary / "leaf_period_metrics.parquet"
        portability_output = temporary / "leaf_portability_scores.parquet"
        connection.execute(
            f"COPY (SELECT * FROM family_period ORDER BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction, period_start) TO {_literal(str(period_output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            f"COPY (SELECT * FROM h2_snapshot ORDER BY score_month, feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction) TO {_literal(str(portability_output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        explain_path = temporary / "covariance_explainability.parquet"
        pd.DataFrame({"status": ["NOT_FITTED_IN_STATE_MATERIALISATION"], "reason": ["C5/C6 held-out explanatory regressions belong to the later transport ablation, not prequential feature generation"], "uses_outcomes": [False]}).to_parquet(explain_path, index=False, compression="zstd")
        connection.close()
        files = (selected_state_path, health_path, period_output, portability_output, covariance_path, relationship_path, explain_path)
        manifest = {
            "schema": SCHEMA,
            "status": STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "strict_roots": list(store.manifest["source"]["strict_roots"]),
            "strict_root_manifest_sha256": dict(store.manifest["source"]["strict_root_manifest_sha256"]),
            "strict_event_store_manifest_sha256": _hash(store.manifest_path),
            "contract": {
                "family_identity": "feature_contract_sha256, side, head, rule_signature, contribution_direction",
                "raw_leaf_ids": "rejected; event store retains only token-free same-artifact family contributions",
                "history": "only label_available_ts < feature_generation_ts; same timestamp events score before resolution",
                "state_engine": "vectorized SQL H1/H2 direct candidate aggregates; selected-family month-frozen H3 and selected H4/H5",
                "family_state_audit": "only frozen H3/H4/H5 selected family rows are persisted; unselected event state is not materialised",
                "physical_types": "event-store dictionary dimensions and float32 continuous fields; health features float32",
                "event_store_reuse_validation": "sealed part-index and strict-root lineage verified; full physical part hashing is opt-in",
                "covariance": "H4 source-filtered to frozen selected families and uses only causal compact context",
                "relationship_breaks": "H5 source-filtered to frozen selected families and uses causal relationship residuals",
            },
            "config": _config_payload(config),
            "context_columns": list(context_columns),
            "covariance_field_audit": field_audit.to_dict("records"),
            "row_counts": {
                "family_candidate_states": selected_rows,
                "health_features": health_rows,
                "covariance_diagnostics": int(len(covariance)),
                "relationship_breaks": int(len(relationships)),
            },
            "performance": {"elapsed_seconds": round(time.monotonic() - started, 3), "threads": int(threads), "memory_limit": memory_limit, "full_event_store_part_checksum_audit": bool(verify_event_store_parts)},
            "sha256": {path.name: _hash(path) for path in files},
        }
        (temporary / "leaf_covariance_reference_manifest.json").write_text(json.dumps({"schema": SCHEMA, "status": "CAUSAL_COVARIANCE_REFERENCE_DECLARED", "contract": manifest["contract"]["covariance"], "context_columns": list(context_columns), "covariance_field_audit": manifest["covariance_field_audit"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "leaf_failure_classification.yaml").write_text(json.dumps({"schema": SCHEMA, "classification_counts": {}, "labels": ["STABLE_PORTABLE", "SUPPORT_SHIFT_ONLY", "CALIBRATION_DRIFT", "COMPOSITION_DRIFT", "REGIME_CONDITIONAL", "COVARIANCE_CONDITIONAL", "GLOBAL_PERIOD_SENSITIVE", "UNEXPLAINED_CONCEPT_BREAK", "LOW_SUPPORT_UNCERTAIN", "META_HARMFUL"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "health_materialization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["materialize_strict_oof_causal_leaf_health_vectorized"]
