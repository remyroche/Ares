"""Scope-bounded strict-OOF H1--H5 health materialisation.

This is the production implementation used when an event store is too large
for the original vectorised implementation to keep a joined contribution
population in one DuckDB database.  It deliberately has a different physical
plan, not a different causal contract:

* candidate-only resolution groups build the shared global H1 prior once;
* each ``(feature contract, side, head)`` contribution scope is opened,
  scored, reduced to direct candidate H1/H2/H3 features, and released;
* the only contribution-level records retained are frozen H3/H4/H5 family
  states and their per-candidate denominator.  H4/H5 are therefore source
  filtered at the first durable boundary.

No raw leaf identifier is accepted or emitted.  In particular, this module
does not make the temporary 423M-row joined ``family_events`` table used by
the first vectorised implementation.
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
from typing import Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from .causal_leaf_covariance import covariance_feature_names
from .causal_leaf_health import (
    DIRECTIONS, HEADS, SCHEMA, STATUS, CausalLeafHealthConfig,
    CausalLeafHealthError, _config_payload, _materialise_h4_h5,
    _relationship_break_columns,
)
from .causal_leaf_health_streaming import _copy_h5
from .causal_leaf_health_vectorized import (
    _EVENT_STATE_KEY, _FAMILY, _H1_METRICS, _H2_METRICS, _H3_METRICS,
    _IDENTITY, _asof_context_timeline, _copy_direct_sections,
    _h3_states, _literal,
)
from .strict_event_store import (
    CANDIDATE_COLUMNS, CONTRIBUTION_COLUMNS, StrictEventStore,
    load_strict_event_store,
)


_SCHEMA = "causal_leaf_health_scoped_v1"
_SCOPE = ("contract", "side", "head")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_literal(root: Path, values: Iterable[object]) -> str:
    paths = [str(root / str(value)) for value in values]
    if not paths:
        # DuckDB must not be asked to expand an empty parquet list.
        raise CausalLeafHealthError("event-store scope has no physical parts")
    return "[" + ", ".join(_literal(path) for path in paths) + "]"


def _parquet_columns(connection: duckdb.DuckDBPyConnection, path: Path) -> tuple[str, ...]:
    """Read Parquet schema only; never decode a scope just to inspect names."""
    rows = connection.execute(f"DESCRIBE SELECT * FROM read_parquet({_literal(str(path))})").fetchall()
    return tuple(str(row[0]) for row in rows)


def _scope_rows(store: StrictEventStore, dataset: str) -> pd.DataFrame:
    rows = store.part_index.loc[store.part_index["dataset"].eq(dataset)].copy()
    needed = {"contract", "side", "head", "path"}
    if not needed.issubset(rows.columns):
        raise CausalLeafHealthError("strict event-store index lacks scope fields")
    if rows.empty:
        raise CausalLeafHealthError(f"strict event-store has no {dataset} parts")
    return rows.sort_values([*_SCOPE, "path"], kind="stable")


def _scope_paths(rows: pd.DataFrame, key: tuple[str, str, str]) -> list[str]:
    contract, side, head = key
    subset = rows.loc[
        rows["contract"].astype(str).eq(contract)
        & rows["side"].astype(str).eq(side)
        & rows["head"].astype(str).eq(head),
        "path",
    ]
    return subset.astype(str).tolist()


def _context_keys(selection: frozenset[tuple[str, str, str, str, str]], key: tuple[str, str, str]) -> frozenset[tuple[str, str, str, str, str]]:
    contract, side, head = key
    return frozenset(value for value in selection if value[:3] == (contract, side, head))


def _candidate_global_resolution(connection: duckdb.DuckDBPyConnection, resolution_paths: str, output: Path) -> int:
    """Persist the compact shared global prior from candidate rows only.

    The output contains one record per resolution timestamp/contract rather
    than one record per family contribution.  It is intentionally written
    before any scope contribution is opened and is re-used read-only by every
    side/head scope.
    """

    connection.execute(
        f"CREATE VIEW all_resolution AS SELECT {', '.join(CANDIDATE_COLUMNS)} FROM read_parquet({resolution_paths}, hive_partitioning=false)"
    )
    connection.execute(
        f"""
        COPY (
          WITH grouped AS (
            SELECT feature_contract_sha256, label_available_ts,
                   count(*) AS rows_delta, sum(semantic_label) AS success_delta
            FROM all_resolution
            GROUP BY ALL
          )
          SELECT feature_contract_sha256, label_available_ts,
                 sum(rows_delta) OVER w AS rows,
                 sum(success_delta) OVER w AS successes
          FROM grouped
          WINDOW w AS (
            PARTITION BY feature_contract_sha256
            ORDER BY label_available_ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          )
        ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])


def _create_scoped_h1_state(connection: duckdb.DuckDBPyConnection, config: CausalLeafHealthConfig, global_resolution: Path) -> None:
    """Create H1 snapshots for the currently-open scope only.

    This is the H1 SQL from the vectorised reference with its global prior
    replaced by a candidate-only shared resolution group.  ``family_events``
    is scoped by the caller, so its largest temporary is one scope, never the
    entire event-store population.
    """

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
                 label_available_ts, count(*) AS rows_delta, sum(semantic_label) AS success_delta,
                 sum(head_prediction) AS prediction_delta, sum(net_bps) AS net_delta,
                 sum(base_expected_bps) AS expected_delta,
                 sum(CASE WHEN head_prediction >= 0.5 AND semantic_label <= 0 THEN greatest(-net_bps, 0.0) ELSE 0.0 END) AS fpl_delta,
                 sum(new_timestamp) AS timestamp_delta, sum(new_day) AS day_delta, sum(new_symbol) AS symbol_delta
          FROM family_marked GROUP BY ALL
        )
        SELECT *, sum(rows_delta) OVER w AS rows, sum(success_delta) OVER w AS successes,
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
    alpha, beta = float(config.global_alpha), float(config.global_beta)
    side_strength, family_strength = float(config.side_head_prior_strength), float(config.family_prior_strength)
    connection.execute(
        f"""
        -- A view is intentional: materialising a 70M-row H1 state table
        -- would defeat the scoped disk budget.  The direct candidate query
        -- consumes this plan and only selected H3/H4/H5 rows are copied.
        CREATE TEMP VIEW h1_states AS
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
          ASOF LEFT JOIN read_parquet({_literal(str(global_resolution))}) g
            ON e.feature_contract_sha256=g.feature_contract_sha256
           AND e.feature_generation_ts > g.label_available_ts
        ), priors AS (
          SELECT *,
            (g_successes + {alpha}) / greatest(g_rows + {alpha + beta}, 1e-12) AS global_mean,
            (sh_successes + {side_strength} * ((g_successes + {alpha}) / greatest(g_rows + {alpha + beta}, 1e-12))) / greatest(sh_rows + {side_strength}, 1e-12) AS side_mean
          FROM joined
        ), posterior AS (
          SELECT *, f_successes + {family_strength} * side_mean AS post_alpha,
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


def _candidate_context(connection: duckdb.DuckDBPyConnection, timeline: pd.DataFrame) -> None:
    connection.register("__scoped_context", timeline)
    connection.execute(
        """
        CREATE TEMP TABLE candidate_context AS
        SELECT c.*, r.*
        FROM candidates c
        ASOF LEFT JOIN __scoped_context r
          ON c.feature_generation_ts >= r.regime_available_utc
        """
    )
    if int(connection.execute("SELECT count(*) FROM candidate_context WHERE regime_available_utc IS NULL").fetchone()[0]):
        raise CausalLeafHealthError("candidate lacks a prior available causal context row")


def _create_scoped_h2_snapshot(connection: duckdb.DuckDBPyConnection, config: CausalLeafHealthConfig) -> None:
    """Create compact month-frozen H2 snapshots, never a full H2 state table."""
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
        FROM family_events GROUP BY ALL
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE score_months AS
        SELECT DISTINCT feature_contract_sha256, side_name, head_name,
               date_trunc('month', feature_generation_ts) AS score_month FROM candidates
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE h2_base AS
        WITH closed AS (
          SELECT m.score_month, p.*,
                 (p.rows >= {int(config.min_period_rows)} AND p.timestamps >= {int(config.min_timestamp_support)}
                  AND p.days >= {int(config.min_day_support)} AND p.symbols >= {int(config.min_symbol_support)}) AS supported
          FROM score_months m JOIN family_period p
            ON m.feature_contract_sha256=p.feature_contract_sha256 AND m.side_name=p.side_name AND m.head_name=p.head_name
           AND p.period_start + INTERVAL 1 MONTH <= m.score_month - INTERVAL {int(config.period_close_lag_hours)} HOUR
        ), prior AS (
          SELECT *, lag(effect_bps) OVER w AS previous_effect_bps,
                 lag(effect_se_bps) OVER w AS previous_effect_se_bps, lag(supported) OVER w AS previous_supported
          FROM closed WINDOW w AS (PARTITION BY score_month, feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction ORDER BY period_start)
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
                          THEN CAST(sign(effect_bps) <> sign(previous_effect_bps) AS DOUBLE) ELSE NULL END) AS sign_reversal_rate
          FROM prior GROUP BY ALL
        )
        SELECT *, CASE WHEN supported_period_count > 0 THEN effect_sq_sum / weight_sum - power(effect_sum / weight_sum, 2) ELSE NULL END AS observed_variance,
          CASE WHEN supported_period_count > 0 THEN greatest(0.0, effect_sq_sum / weight_sum - power(effect_sum / weight_sum, 2) - coalesce(sampling_variance, 0.0)) ELSE NULL END AS excess_variance,
          CASE WHEN period_count > 0 THEN 1.0 - supported_period_count::DOUBLE / period_count ELSE 1.0 END AS support_failure,
          CASE WHEN supported_period_count > 0 THEN calibration_sum / weight_sum ELSE NULL END AS calibration_drift
        FROM basic
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE h2_snapshot AS
        WITH bucketed AS (
          SELECT *, floor(log2(greatest(weight_sum, 1)))::INTEGER AS support_bucket FROM h2_base
        ), med AS (
          SELECT *, median(excess_variance) OVER (PARTITION BY score_month, feature_contract_sha256, side_name, head_name, support_bucket) AS excess_median FROM bucketed
        ), mad AS (
          SELECT *, median(abs(excess_variance - excess_median)) OVER (PARTITION BY score_month, feature_contract_sha256, side_name, head_name, support_bucket) AS excess_mad FROM med
        ), damage AS (
          SELECT *, median(worst_damage_bps) OVER (PARTITION BY score_month) AS damage_median FROM mad
        )
        SELECT *, CASE WHEN excess_variance IS NULL THEN NULL ELSE (excess_variance - excess_median) / greatest(1.4826 * coalesce(excess_mad, 0.0), 1e-8) END AS robust_z,
          coalesce(sign_reversal_rate, 0.0) AS sign_reversal_rate,
          CASE WHEN excess_variance IS NULL THEN NULL ELSE coalesce((excess_variance - excess_median) / greatest(1.4826 * coalesce(excess_mad, 0.0), 1e-8), 0.0) + coalesce(worst_damage_bps, 0.0) / greatest(coalesce(damage_median, 1.0), 1.0) + 0.5 * coalesce(support_failure, 1.0) END AS instability
        FROM damage
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW h2_states AS
        SELECT e.*, CAST(coalesce(s.period_count, 0.0) AS FLOAT) AS h2_period_count,
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
        FROM family_events e LEFT JOIN h2_snapshot s
          ON e.feature_contract_sha256=s.feature_contract_sha256 AND e.side_name=s.side_name
         AND e.head_name=s.head_name AND e.rule_signature=s.rule_signature
         AND e.contribution_direction=s.contribution_direction
         AND date_trunc('month', e.feature_generation_ts)=s.score_month
        """
    )


def _selected_denominators(connection: duckdb.DuckDBPyConnection, *, selection: frozenset[tuple[str, str, str, str, str]], output: Path) -> int:
    if not selection:
        pd.DataFrame(columns=[*_IDENTITY, "head_name", "contribution_direction", "full_abs_contribution"]).to_parquet(output, index=False, compression="zstd")
        return 0
    keys = pd.DataFrame(sorted(selection), columns=list(_FAMILY))
    connection.register("__selected_denominator_keys", keys)
    connection.execute(
        f"""
        COPY (
          WITH relevant_candidates AS (
            SELECT DISTINCT {', '.join(_IDENTITY)}
            FROM h2_states h INNER JOIN __selected_denominator_keys k USING ({', '.join(_FAMILY)})
          )
          SELECT h.{', h.'.join(_IDENTITY)}, h.head_name, h.contribution_direction,
                 CAST(sum(abs(h.family_ensemble_tree_contribution)) AS FLOAT) AS full_abs_contribution
          FROM h2_states h INNER JOIN relevant_candidates r USING ({', '.join(_IDENTITY)})
          GROUP BY h.{', h.'.join(_IDENTITY)}, h.head_name, h.contribution_direction
        ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])


def _write_empty_selected_state(path: Path, context_columns: Sequence[str]) -> None:
    """Write a typed, readable empty audit instead of a zero-column parquet."""
    columns = [
        *_EVENT_STATE_KEY,
        *[f"h1_{name}" for name in _H1_METRICS],
        *[f"h2_{name}" for name in _H2_METRICS],
        "regime_available_utc", *context_columns,
        "h4_selection_active", "h5_selection_active",
        *[f"h3_{name}" for name in _H3_METRICS],
    ]
    pd.DataFrame({name: pd.Series(dtype="float32") for name in columns}).to_parquet(path, index=False, compression="zstd")


def _write_selected_h1(
    connection: duckdb.DuckDBPyConnection, *, selection: frozenset[tuple[str, str, str, str, str]], output: Path,
) -> None:
    """Durably retain H1 only for frozen selected families before release."""
    if not selection:
        pd.DataFrame({name: pd.Series(dtype="float32") for name in [*_EVENT_STATE_KEY, *[f"h1_{metric}" for metric in _H1_METRICS]]}).to_parquet(output, index=False, compression="zstd")
        return
    connection.register("__scoped_selected_h1_keys", pd.DataFrame(sorted(selection), columns=list(_FAMILY)))
    connection.execute(
        f"COPY (SELECT h.* FROM h1_states h INNER JOIN __scoped_selected_h1_keys k USING ({', '.join(_FAMILY)})) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def _selected_state_audit_scoped(
    connection: duckdb.DuckDBPyConnection, *, selection: frozenset[tuple[str, str, str, str, str]],
    covariance_selection: frozenset[tuple[str, str, str, str, str]],
    relationship_selection: frozenset[tuple[str, str, str, str, str]], context_columns: Sequence[str],
    h1_path: Path, h3_path: Path, config: CausalLeafHealthConfig, output: Path,
) -> int:
    """Persist selected H1+H2 rows only, without materialising a full join."""
    if not selection:
        _write_empty_selected_state(output, context_columns)
        return 0
    connection.register("__scoped_selected_keys", pd.DataFrame(sorted(selection), columns=list(_FAMILY)))
    connection.register("__scoped_cov_keys", pd.DataFrame(sorted(covariance_selection), columns=list(_FAMILY)))
    connection.register("__scoped_rel_keys", pd.DataFrame(sorted(relationship_selection), columns=list(_FAMILY)))
    active = "TRUE" if config.family_selection_effective_utc is None else f"h.feature_generation_ts >= TIMESTAMPTZ {_literal(config.family_selection_effective_utc)}"
    context_select = ", ".join(f"cc.\"{name}\"" for name in context_columns)
    # H1 is a view, H2 is a compact snapshot join.  The predicate is applied
    # at the durable boundary: only frozen selected families are copied.
    connection.execute(
        f"""
        COPY (
          SELECT h.*, s.{', s.'.join([f'h2_{name}' for name in _H2_METRICS])},
                 {context_select}, cc.regime_available_utc,
                 CAST({active} AND ck.feature_contract_sha256 IS NOT NULL AS FLOAT) AS h4_selection_active,
                 CAST({active} AND rk.feature_contract_sha256 IS NOT NULL AS FLOAT) AS h5_selection_active,
                 coalesce(x.h3_availability, 0.0)::FLOAT AS h3_availability,
                 coalesce(x.h3_compatibility, 0.0)::FLOAT AS h3_compatibility,
                 coalesce(x.h3_expected_error_bps, 0.0)::FLOAT AS h3_expected_error_bps,
                 coalesce(x.h3_confidence, 0.0)::FLOAT AS h3_confidence,
                 coalesce(x.h3_unexplained_break, 0.0)::FLOAT AS h3_unexplained_break
          FROM read_parquet({_literal(str(h1_path))}) h
          INNER JOIN __scoped_selected_keys k USING ({', '.join(_FAMILY)})
          LEFT JOIN h2_states s USING ({', '.join(_EVENT_STATE_KEY)})
          INNER JOIN candidate_context cc USING ({', '.join(_IDENTITY)})
          LEFT JOIN __scoped_cov_keys ck USING ({', '.join(_FAMILY)})
          LEFT JOIN __scoped_rel_keys rk USING ({', '.join(_FAMILY)})
          LEFT JOIN read_parquet({_literal(str(h3_path))}) x USING ({', '.join(_EVENT_STATE_KEY)})
        ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])


def _merge_scope_sections(connection: duckdb.DuckDBPyConnection, inputs: Sequence[Path], output: Path) -> tuple[int, tuple[str, ...]]:
    """Join the three scope-local direct sections before releasing the scope."""
    columns: list[str] = []
    for path in inputs:
        columns.extend([name for name in _parquet_columns(connection, path) if name not in _IDENTITY])
    # The identities are identical within a scope.  We retain one copy of
    # every wide field; a later global SUM is exact because a given
    # head/direction belongs to exactly one scope.
    aliases = [f"p{i}" for i in range(len(inputs))]
    select = [f"{aliases[0]}.{item}" for item in _IDENTITY]
    for alias, path in zip(aliases, inputs, strict=True):
        for name in _parquet_columns(connection, path):
            if name not in _IDENTITY:
                select.append(f"{alias}.\"{name}\"")
    joins = f"read_parquet({_literal(str(inputs[0]))}) {aliases[0]}"
    for alias, path in zip(aliases[1:], inputs[1:], strict=True):
        joins += f" INNER JOIN read_parquet({_literal(str(path))}) {alias} USING ({', '.join(_IDENTITY)})"
    connection.execute(f"COPY (SELECT {', '.join(select)} FROM {joins}) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0]), tuple(columns)


def _write_h4_cells(
    selected_state: Path, covariance: Path, denominators: Sequence[Path], output: Path,
    *, memory_limit: str = "1GB", temp_disk_limit: str = "8GB", temp_dir: Path | None = None,
) -> tuple[tuple[str, ...], int]:
    """Build selected-only H4 candidate cells with the full-mass denominator."""
    metrics = tuple(name.removeprefix("base_health__h4__") for name in covariance_feature_names("base_health__h4"))
    if not selected_state.is_file() or not denominators:
        pd.DataFrame(columns=[*_IDENTITY, "head_name", "contribution_direction", *metrics, "availability"]).to_parquet(output, index=False, compression="zstd")
        return metrics, 0
    database = output.parent / "h4_cells.duckdb"
    spill = temp_dir or (output.parent / "h4_cells_tmp")
    connection = duckdb.connect(database=str(database))
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(spill))}")
        connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        covariance_columns = _parquet_columns(connection, covariance)
        if not set(f"base_health__h4__{name}" for name in metrics).issubset(covariance_columns):
            pd.DataFrame(columns=[*_IDENTITY, "head_name", "contribution_direction", *metrics, "availability"]).to_parquet(output, index=False, compression="zstd")
            return metrics, 0
        denominator_paths = "[" + ", ".join(_literal(str(path)) for path in denominators) + "]"
        metric_sql = ", ".join(
            f"CAST(COALESCE(sum(abs(s.family_ensemble_tree_contribution) * s.h4_selection_active * COALESCE(c.\"base_health__h4__{name}\", 0.0)) / NULLIF(max(d.full_abs_contribution), 0.0), 0.0) AS FLOAT) AS \"{name}\""
            for name in metrics
        )
        connection.execute(
            f"""
            COPY (
              SELECT s.{', s.'.join(_IDENTITY)}, s.head_name, s.contribution_direction,
                     {metric_sql},
                     CAST(COALESCE(sum(abs(s.family_ensemble_tree_contribution) * s.h4_selection_active) / NULLIF(max(d.full_abs_contribution), 0.0), 0.0) AS FLOAT) AS availability
              FROM read_parquet({_literal(str(selected_state))}) s
              LEFT JOIN read_parquet({_literal(str(covariance))}) c
                USING ({', '.join([*_IDENTITY, 'head_name', 'contribution_direction', 'rule_signature'])})
              INNER JOIN read_parquet({denominator_paths}, union_by_name=true) d
                USING ({', '.join([*_IDENTITY, 'head_name', 'contribution_direction'])})
              WHERE COALESCE(s.h4_selection_active, 0.0) > 0.0
              GROUP BY s.{', s.'.join(_IDENTITY)}, s.head_name, s.contribution_direction
            ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        return metrics, int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
        database.unlink(missing_ok=True)


def _pivot_partial(connection: duckdb.DuckDBPyConnection, *, input_path: Path, section: str, metrics: Sequence[str], output: Path) -> tuple[str, ...]:
    """Turn a sparse selected-family cell table into conventional wide fields."""
    expressions: list[str] = []
    names: list[str] = []
    for head in HEADS:
        for direction in DIRECTIONS:
            predicate = f"head_name={_literal(head)} AND contribution_direction={_literal(direction)}"
            for metric in metrics:
                name = f"base_health__{section}__{head}__{direction}__{metric}"
                expressions.append(f"CAST(COALESCE(sum(CASE WHEN {predicate} THEN \"{metric}\" ELSE 0.0 END), 0.0) AS FLOAT) AS \"{name}\"")
                names.append(name)
    connection.execute(
        f"""
        COPY (
          SELECT {', '.join(_IDENTITY)}, {', '.join(expressions)}
          FROM read_parquet({_literal(str(input_path))})
          GROUP BY {', '.join(_IDENTITY)}
        ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return tuple(names)


def _final_health(
    connection: duckdb.DuckDBPyConnection, *, candidate_paths: str, scope_health: Sequence[Path],
    h4_path: Path, h4_names: Sequence[str], h5_path: Path, h5_names: Sequence[str], output: Path,
) -> int:
    """Merge candidate-direct scope outputs without a contribution join."""
    connection.execute(
        f"CREATE TEMP VIEW base_candidates AS SELECT DISTINCT {', '.join(_IDENTITY)} FROM read_parquet({candidate_paths}, hive_partitioning=false)"
    )
    if scope_health:
        scope_paths = "[" + ", ".join(_literal(str(path)) for path in scope_health) + "]"
        # Sum is deliberate: feature columns are non-zero in only their own
        # side/head scope.  It preserves negative residual metrics unlike MAX.
        health_names = [name for name in _parquet_columns(connection, scope_health[0]) if name not in _IDENTITY]
        aggregate = ", ".join(f"CAST(COALESCE(sum(\"{name}\"), 0.0) AS FLOAT) AS \"{name}\"" for name in health_names)
        connection.execute(
            f"CREATE TEMP VIEW h123 AS SELECT {', '.join(_IDENTITY)}, {aggregate} FROM read_parquet({scope_paths}, union_by_name=true) GROUP BY {', '.join(_IDENTITY)}"
        )
    else:
        raise CausalLeafHealthError("scoped materialisation produced no H1/H2/H3 scope feature parts")
    selects = [f"b.{name}" for name in _IDENTITY]
    for name in [name for name in _parquet_columns(connection, scope_health[0]) if name not in _IDENTITY]:
        selects.append(f"COALESCE(h.\"{name}\", 0.0)::FLOAT AS \"{name}\"")
    for alias, path, names in (("h4", h4_path, h4_names), ("h5", h5_path, h5_names)):
        if not path.is_file():
            continue
        for name in names:
            # H4's direct aggregation retains the full active contribution
            # mass, even when the selected covariance subset is empty.  That
            # mass is exactly the H1 active mass for the same semantic
            # head/direction.  H4 cells intentionally retain only selected
            # rows, so recover the six stable full-mass fields from H1 rather
            # than materialising a contribution-wide H4 intermediate.
            if alias == "h4" and name.endswith("__active_abs_contribution"):
                h1_name = name.replace("base_health__h4__", "base_health__h1__", 1)
                selects.append(f"COALESCE(h.\"{h1_name}\", 0.0)::FLOAT AS \"{name}\"")
                continue
            selects.append(f"COALESCE({alias}.\"{name}\", 0.0)::FLOAT AS \"{name}\"")
    joins = f"base_candidates b LEFT JOIN h123 h USING ({', '.join(_IDENTITY)})"
    joins += f" LEFT JOIN read_parquet({_literal(str(h4_path))}) h4 USING ({', '.join(_IDENTITY)})"
    joins += f" LEFT JOIN read_parquet({_literal(str(h5_path))}) h5 USING ({', '.join(_IDENTITY)})"
    connection.execute(f"COPY (SELECT {', '.join(selects)} FROM {joins} ORDER BY b.transport, b.meta_partition, b.decision_ts, b.candidate_id) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])


def materialize_strict_oof_causal_leaf_health_scoped(
    event_store: StrictEventStore | str | Path,
    output_dir: str | Path,
    *, causal_context: pd.DataFrame, context_feature_columns: Sequence[str],
    config: CausalLeafHealthConfig = CausalLeafHealthConfig(),
    threads: int = 2, memory_limit: str = "2GB", verify_event_store_parts: bool = False,
    max_selected_state_rows: int = 3_000_000, temp_disk_limit: str = "16GB",
) -> Path:
    """Materialise compatible H1--H5 output one physical scope at a time.

    The temporary database is recreated for every scope.  The peak
    contribution working set is therefore bounded by the largest event-store
    ``(contract, side, head)`` partition; no global contribution table is
    read, joined, sorted, or persisted.
    """
    started = time.monotonic()
    config.validate()
    if int(max_selected_state_rows) <= 0:
        raise CausalLeafHealthError("max_selected_state_rows must be positive")
    if not str(temp_disk_limit).strip():
        raise CausalLeafHealthError("temp_disk_limit must be a non-empty DuckDB size")
    store = event_store if isinstance(event_store, StrictEventStore) else load_strict_event_store(
        event_store, verify_parts=verify_event_store_parts, verify_source=True,
    )
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite causal leaf health artifact: {target}")
    timeline, context_columns = _asof_context_timeline(causal_context, context_feature_columns, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    scope_dir = temporary / "scope_features"; scope_dir.mkdir()
    selected_dir = temporary / "selected_state_parts"; selected_dir.mkdir()
    denominator_dir = temporary / "selected_denominator_parts"; denominator_dir.mkdir()
    diagnostics_dir = temporary / "scope_diagnostics"; diagnostics_dir.mkdir()
    try:
        score_index = _scope_rows(store, "score_order")
        resolution_index = _scope_rows(store, "resolution_order")
        candidate_index = _scope_rows(store, "candidate")
        contribution_index = _scope_rows(store, "contribution")
        keys = [tuple(map(str, row)) for row in score_index.loc[:, list(_SCOPE)].drop_duplicates().itertuples(index=False, name=None)]
        if not keys:
            raise CausalLeafHealthError("strict event store has no score scopes")
        all_resolution = _paths_literal(store.root, resolution_index["path"].astype(str))
        all_candidates = _paths_literal(store.root, candidate_index["path"].astype(str))
        prior_connection = duckdb.connect(database=str(temporary / "global_priors.duckdb"))
        prior_connection.execute(f"PRAGMA threads={max(1, int(threads))}")
        prior_connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        prior_connection.execute(f"PRAGMA temp_directory={_literal(str(temporary / 'global_prior_tmp'))}")
        prior_connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        global_resolution = temporary / "global_h1_resolution.parquet"
        global_resolution_rows = _candidate_global_resolution(prior_connection, all_resolution, global_resolution)
        prior_connection.close()
        scope_health_paths: list[Path] = []
        selected_state_paths: list[Path] = []
        denominator_paths: list[Path] = []
        period_paths: list[Path] = []
        portability_paths: list[Path] = []
        selected_rows = 0
        max_scope_rows = 0
        all_selected = frozenset(set(config.selected_context_families) | set(config.selected_covariance_families) | set(config.selected_relationship_families))
        for index, key in enumerate(keys):
            contract, side, head = key
            candidate_paths = _paths_literal(store.root, _scope_paths(score_index, key))
            resolution_paths = _paths_literal(store.root, _scope_paths(resolution_index, key))
            contribution_paths = _paths_literal(store.root, _scope_paths(contribution_index, key))
            connection = duckdb.connect(database=str(temporary / f"scope_{index}.duckdb"))
            try:
                connection.execute(f"PRAGMA threads={max(1, int(threads))}")
                connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
                connection.execute(f"PRAGMA temp_directory={_literal(str(temporary / f'scope_{index}_tmp'))}")
                # The external SQL plan is deliberately fail-closed if it
                # would consume the volume reserved for immutable artifacts.
                # This is a safety ceiling, not a semantic/selection limit.
                connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
                connection.execute(f"CREATE VIEW candidates AS SELECT {', '.join(CANDIDATE_COLUMNS)} FROM read_parquet({candidate_paths}, hive_partitioning=false)")
                connection.execute(f"CREATE VIEW resolution_candidates AS SELECT {', '.join(CANDIDATE_COLUMNS)} FROM read_parquet({resolution_paths}, hive_partitioning=false)")
                connection.execute(f"CREATE VIEW contributions AS SELECT {', '.join(CONTRIBUTION_COLUMNS)} FROM read_parquet({contribution_paths}, hive_partitioning=false)")
                bad = int(connection.execute("SELECT count(*) FROM contributions WHERE rule_signature IS NULL OR trim(rule_signature)='' OR contribution_direction NOT IN ('positive', 'negative') OR family_ensemble_tree_contribution=0").fetchone()[0])
                if bad:
                    raise CausalLeafHealthError("event-store contributions violate the token-free numeric contract")
                late = int(connection.execute("SELECT count(*) FROM candidates WHERE label_available_ts > decision_ts + INTERVAL 24 HOUR").fetchone()[0])
                if late:
                    raise CausalLeafHealthError("scoped monthly H2 requires labels to resolve within the declared 24h close lag")
                _candidate_context(connection, timeline)
                _create_scoped_h1_state(connection, config, global_resolution)
                h1_path = scope_dir / f"scope_{index:03d}_h1.parquet"
                h2_path = scope_dir / f"scope_{index:03d}_h2.parquet"
                _copy_direct_sections(connection, candidate_view="candidates", state_view="h1_states", output=h1_path, sections=(("h1", _H1_METRICS),))
                scope_selected = _context_keys(all_selected, key)
                selected_h1_path = selected_dir / f"scope_{index:03d}_h1.parquet"
                _write_selected_h1(connection, selection=scope_selected, output=selected_h1_path)
                # Release all high-cardinality H1 intermediates before H2.
                # The selected H1 projection above is the only H1 state that
                # survives the scope boundary.
                connection.execute("DROP VIEW h1_states")
                connection.execute("DROP TABLE family_h1_resolution")
                connection.execute("DROP TABLE scope_h1_resolution")
                connection.execute("DROP VIEW family_marked")
                _create_scoped_h2_snapshot(connection, config)
                _copy_direct_sections(connection, candidate_view="candidates", state_view="h2_states", output=h2_path, sections=(("h2", _H2_METRICS),))
                selected_context = _context_keys(config.selected_context_families, key)
                h3_path = _h3_states(connection, selected=selected_context, context_columns=context_columns, config=config, temporary=temporary)
                connection.execute(f"CREATE TEMP VIEW h3_states AS SELECT * FROM read_parquet({_literal(str(h3_path))})")
                connection.execute(f"CREATE TEMP VIEW h3_full AS SELECT h.*, coalesce(s.h3_availability, 0.0)::FLOAT AS h3_availability, coalesce(s.h3_compatibility, 0.0)::FLOAT AS h3_compatibility, coalesce(s.h3_expected_error_bps, 0.0)::FLOAT AS h3_expected_error_bps, coalesce(s.h3_confidence, 0.0)::FLOAT AS h3_confidence, coalesce(s.h3_unexplained_break, 0.0)::FLOAT AS h3_unexplained_break FROM family_events h LEFT JOIN h3_states s USING ({', '.join(_EVENT_STATE_KEY)})")
                h3_feature_path = scope_dir / f"scope_{index:03d}_h3.parquet"
                _copy_direct_sections(connection, candidate_view="candidates", state_view="h3_full", output=h3_feature_path, sections=(("h3", _H3_METRICS),))
                scope_path = scope_dir / f"scope_{index:03d}.parquet"
                scope_count, _ = _merge_scope_sections(connection, (h1_path, h2_path, h3_feature_path), scope_path)
                max_scope_rows = max(max_scope_rows, scope_count)
                scope_health_paths.append(scope_path)
                state_path = selected_dir / f"scope_{index:03d}.parquet"
                _selected_state_audit_scoped(connection, selection=scope_selected, covariance_selection=_context_keys(config.selected_covariance_families, key), relationship_selection=_context_keys(config.selected_relationship_families, key), context_columns=context_columns, h1_path=selected_h1_path, h3_path=h3_path, config=config, output=state_path)
                state_count = int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(state_path))})").fetchone()[0])
                selected_rows += state_count
                if selected_rows > int(max_selected_state_rows):
                    raise CausalLeafHealthError("selected H3/H4/H5 state audit exceeds max_selected_state_rows")
                # Empty scopes have a deliberately typed placeholder audit.
                # Do not union that placeholder with populated scope files:
                # their physical Arrow types differ for timestamps/dimensions,
                # while an empty scope contributes no H4/H5 state.
                if state_count:
                    selected_state_paths.append(state_path)
                denominator_path = denominator_dir / f"scope_{index:03d}.parquet"
                _selected_denominators(connection, selection=scope_selected, output=denominator_path)
                if state_count:
                    denominator_paths.append(denominator_path)
                period_path = diagnostics_dir / f"scope_{index:03d}_period.parquet"
                portability_path = diagnostics_dir / f"scope_{index:03d}_portability.parquet"
                connection.execute(f"COPY (SELECT * FROM family_period ORDER BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction, period_start) TO {_literal(str(period_path))} (FORMAT PARQUET, COMPRESSION ZSTD)")
                connection.execute(f"COPY (SELECT * FROM h2_snapshot ORDER BY score_month, feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction) TO {_literal(str(portability_path))} (FORMAT PARQUET, COMPRESSION ZSTD)")
                period_paths.append(period_path); portability_paths.append(portability_path)
            finally:
                connection.close()
                for child in (temporary / f"scope_{index}_tmp",):
                    shutil.rmtree(child, ignore_errors=True)
                Path(temporary / f"scope_{index}.duckdb").unlink(missing_ok=True)
        selected_state = temporary / "base_leaf_family_candidate_states.parquet"
        if selected_state_paths:
            merge_connection = duckdb.connect(database=":memory:")
            selected_paths = "[" + ", ".join(_literal(str(path)) for path in selected_state_paths) + "]"
            merge_connection.execute(f"COPY (SELECT * FROM read_parquet({selected_paths}, union_by_name=true)) TO {_literal(str(selected_state))} (FORMAT PARQUET, COMPRESSION ZSTD)")
            merge_connection.close()
        else:
            _write_empty_selected_state(selected_state, context_columns)
        selected_states = pd.read_parquet(selected_state) if selected_rows else pd.DataFrame()
        covariance, relationships, field_audit = _materialise_h4_h5(selected_states, context_columns=context_columns, config=config)
        covariance_path = temporary / "leaf_covariance_diagnostics.parquet"; covariance.to_parquet(covariance_path, index=False, compression="zstd")
        relationship_path = temporary / "leaf_relationship_breaks.parquet"; relationships.to_parquet(relationship_path, index=False, compression="zstd")
        h4_cells = temporary / "h4_cells.parquet"
        h4_metrics, _ = _write_h4_cells(selected_state, covariance_path, denominator_paths, h4_cells)
        h4_path = temporary / "h4.parquet"
        h5_path = temporary / "h5.parquet"
        merge_connection = duckdb.connect(database=str(temporary / "final_merge.duckdb"))
        try:
            merge_connection.execute(f"PRAGMA threads={max(1, int(threads))}")
            merge_connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
            merge_connection.execute(f"PRAGMA temp_directory={_literal(str(temporary / 'final_merge_tmp'))}")
            merge_connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
            h4_partial_names = _pivot_partial(merge_connection, input_path=h4_cells, section="h4", metrics=(*h4_metrics, "availability"), output=h4_path)
            # Preserve the reference 300-field health contract.  The actual
            # values are injected from H1 during the final merge above, which
            # is semantically identical to the reference H4 denominator and
            # avoids re-opening every contribution just for six fields.
            h4_names_list: list[str] = []
            h4_fields_per_scope = len(h4_metrics) + 1  # metrics + availability
            for index, (head, direction) in enumerate((pair for head in HEADS for pair in ((head, "positive"), (head, "negative")))):
                start = index * h4_fields_per_scope
                h4_names_list.extend(h4_partial_names[start:start + h4_fields_per_scope])
                h4_names_list.append(f"base_health__h4__{head}__{direction}__active_abs_contribution")
            h4_names = tuple(h4_names_list)
            pairs = sorted(_relationship_break_columns(context_columns))
            if relationships.empty:
                # Keep the exact stable H5 feature contract even when no
                # frozen relationship family is selected.
                empty = pd.DataFrame(columns=[*_IDENTITY])
                empty.to_parquet(temporary / "h5_candidates.parquet", index=False, compression="zstd")
                _copy_h5(merge_connection, candidate_view=f"read_parquet({_literal(str(temporary / 'h5_candidates.parquet'))})", relationship_path=relationship_path, output=h5_path, pairs=pairs)
            else:
                merge_connection.execute(f"CREATE TEMP VIEW h5_candidates AS SELECT DISTINCT {', '.join(_IDENTITY)} FROM read_parquet({_literal(str(relationship_path))})")
                _copy_h5(merge_connection, candidate_view="h5_candidates", relationship_path=relationship_path, output=h5_path, pairs=pairs)
            h5_names = tuple(name for name in pd.read_parquet(h5_path, columns=None).columns if name not in _IDENTITY)
            health_path = temporary / "base_leaf_health_features_oof.parquet"
            health_rows = _final_health(merge_connection, candidate_paths=all_candidates, scope_health=scope_health_paths, h4_path=h4_path, h4_names=h4_names, h5_path=h5_path, h5_names=h5_names, output=health_path)
            period_output = temporary / "leaf_period_metrics.parquet"
            portability_output = temporary / "leaf_portability_scores.parquet"
            period_list = "[" + ", ".join(_literal(str(path)) for path in period_paths) + "]"
            portability_list = "[" + ", ".join(_literal(str(path)) for path in portability_paths) + "]"
            merge_connection.execute(f"COPY (SELECT * FROM read_parquet({period_list}, union_by_name=true) ORDER BY feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction, period_start) TO {_literal(str(period_output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
            merge_connection.execute(f"COPY (SELECT * FROM read_parquet({portability_list}, union_by_name=true) ORDER BY score_month, feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction) TO {_literal(str(portability_output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:
            merge_connection.close()
        explain_path = temporary / "covariance_explainability.parquet"
        pd.DataFrame({"status": ["NOT_FITTED_IN_STATE_MATERIALISATION"], "reason": ["C5/C6 held-out explanatory regressions belong to the later transport ablation, not prequential feature generation"], "uses_outcomes": [False]}).to_parquet(explain_path, index=False, compression="zstd")
        files = (selected_state, health_path, period_output, portability_output, covariance_path, relationship_path, explain_path)
        manifest = {
            "schema": SCHEMA, "status": STATUS, "created_utc": datetime.now(timezone.utc).isoformat(),
            "strict_roots": list(store.manifest["source"]["strict_roots"]),
            "strict_root_manifest_sha256": dict(store.manifest["source"]["strict_root_manifest_sha256"]),
            "strict_event_store_manifest_sha256": _hash(store.manifest_path),
            "contract": {
                "family_identity": "feature_contract_sha256, side, head, rule_signature, contribution_direction",
                "raw_leaf_ids": "rejected; event store retains only token-free same-artifact family contributions",
                "history": "only label_available_ts < feature_generation_ts; same timestamp events score before resolution",
                "state_engine": "scope-bounded vectorized SQL H1/H2/H3 direct candidate aggregates; selected-family H4/H5",
                "family_state_audit": "only frozen H3/H4/H5 selected family rows are persisted; unselected event state is not materialised",
                "physical_types": "event-store dictionary dimensions and float32 continuous fields; health features float32",
                "event_store_reuse_validation": "sealed part-index and strict-root lineage verified; full physical part hashing is opt-in",
                "covariance": "H4 source-filtered to frozen selected families; its denominator preserves full active candidate contribution mass",
                "relationship_breaks": "H5 source-filtered to frozen selected families and uses causal relationship residuals",
                "scope_plan": "candidate-only global H1 prior once; one (contract,side,head) contribution scope at a time; scope temporary is dropped before the next scope",
            },
            "config": _config_payload(config), "context_columns": list(context_columns),
            "covariance_field_audit": field_audit.to_dict("records"),
            "row_counts": {"family_candidate_states": selected_rows, "health_features": health_rows, "covariance_diagnostics": int(len(covariance)), "relationship_breaks": int(len(relationships)), "global_h1_resolution_groups": global_resolution_rows, "scopes": len(keys), "max_scope_candidate_rows": max_scope_rows},
            "performance": {"elapsed_seconds": round(time.monotonic() - started, 3), "threads": int(threads), "memory_limit": memory_limit, "temp_disk_limit": temp_disk_limit, "full_event_store_part_checksum_audit": bool(verify_event_store_parts), "global_contribution_join": False},
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


__all__ = ["materialize_strict_oof_causal_leaf_health_scoped"]
