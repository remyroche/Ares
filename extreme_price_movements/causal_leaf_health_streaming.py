"""Bounded-memory production materialiser for strict H1--H5 health states.

The public dataframe builder remains the reference implementation for small
tests.  This module consumes :class:`StrictOOFFamilyInputSpool` parts without
ever concatenating the full candidate/family long tables.  It makes two
ordered passes per ``(contract, side, head)`` scope: a label-resolution pass
and a feature-time scoring pass.  That is the exact prequential ordering used
by the in-memory builder, but has a bounded working set.

H4/H5 are deliberately handled only after H1--H3 state parts exist and only
for externally frozen selected families.  A hard selected-state limit fails
closed instead of allowing an unbounded dataframe allocation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .causal_leaf_covariance import covariance_feature_names
from .causal_leaf_health import (
    DIRECTIONS,
    HEADS,
    SCHEMA,
    STATUS,
    CausalLeafHealthConfig,
    CausalLeafHealthError,
    _FamilyStats,
    _config_payload,
    _family_h1,
    _family_selection_active,
    _materialise_h4_h5,
    _portability_metrics,
    _relationship_break_columns,
    _snapshot_portability,
    _context_model_snapshot,
)
from .causal_leaf_health_artifacts import StrictOOFFamilyInputSpool


_IDENTITY = ("candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition")
_CANDIDATE_FIELDS = (
    "candidate_id", "decision_ts", "feature_generation_ts", "label_available_ts",
    "side_name", "head_name", "fold_id", "transport", "meta_partition",
    "feature_contract_sha256", "semantic_label", "head_prediction", "net_bps",
    "base_expected_bps", "asset",
)


def _literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _glob(path: Path) -> str:
    return _literal(str(path / "*.parquet"))


def _read_spool(spool_root: str | Path) -> StrictOOFFamilyInputSpool:
    root = Path(spool_root)
    manifest_path = root / "strict_family_input_spool_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalLeafHealthError(f"invalid strict family input spool: {root}") from exc
    if manifest.get("status") != "STRICT_OOF_FAMILY_INPUT_SPOOL_COMPLETED":
        raise CausalLeafHealthError("strict family input spool is not complete")
    index = root / str(manifest.get("pair_index", ""))
    if not index.is_file():
        raise CausalLeafHealthError("strict family input spool lacks its paired part index")
    pairs = pd.read_parquet(index)
    required = {"part", "candidate_part", "contribution_part", "candidate_sha256", "contribution_sha256"}
    if not required.issubset(pairs.columns):
        raise CausalLeafHealthError("strict family input spool index lacks integrity fields")
    candidate_parts: list[Path] = []
    contribution_parts: list[Path] = []
    for row in pairs.sort_values("part", kind="stable").itertuples(index=False):
        candidate = root / "candidate_parts" / str(row.candidate_part)
        contribution = root / "contribution_parts" / str(row.contribution_part)
        if not candidate.is_file() or not contribution.is_file():
            raise CausalLeafHealthError("strict family input spool is missing a paired part")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != str(row.candidate_sha256):
            raise CausalLeafHealthError("strict family candidate part hash differs from its immutable index")
        if hashlib.sha256(contribution.read_bytes()).hexdigest() != str(row.contribution_sha256):
            raise CausalLeafHealthError("strict family contribution part hash differs from its immutable index")
        candidate_parts.append(candidate)
        contribution_parts.append(contribution)
    if not candidate_parts:
        raise CausalLeafHealthError("strict family input spool has no parts")
    return StrictOOFFamilyInputSpool(
        root=root, candidate_parts=tuple(candidate_parts), contribution_parts=tuple(contribution_parts),
        strict_roots=tuple(map(str, manifest.get("strict_roots", []))),
        strict_root_manifest_sha256=dict(manifest.get("strict_root_manifest_sha256", {})),
        manifest_path=manifest_path,
    )


def _context_timeline(
    context: pd.DataFrame, columns: Sequence[str], config: CausalLeafHealthConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate the shared as-of context form used by strict production runs."""

    if "candidate_id" in context.columns:
        raise CausalLeafHealthError(
            "bounded H1--H5 materialisation requires a shared causal context timeline; "
            "candidate-specific context is intentionally rejected rather than buffered in memory"
        )
    required = {"regime_available_utc", *map(str, columns)}
    missing = sorted(required.difference(context.columns))
    if missing:
        raise CausalLeafHealthError(f"causal regime context lacks declared fields: {missing}")
    if len(columns) > int(config.covariance_max_fields):
        columns = tuple(map(str, columns[: int(config.covariance_max_fields)]))
    forbidden = ("target", "label", "outcome", "future", "realized", "realised", "pnl", "net_ev", "gross_ev", "mfe", "mae", "barrier", "timeout", "exit", "post_entry", "postentry")
    found = [str(name) for name in context.columns if any(token in str(name).lower() for token in forbidden)]
    if found:
        raise CausalLeafHealthError(f"causal regime context contains outcome-derived fields: {found[:8]}")
    work = context.loc[:, ["regime_available_utc", *columns]].copy()
    work["regime_available_utc"] = pd.to_datetime(work["regime_available_utc"], utc=True, errors="coerce")
    if work["regime_available_utc"].isna().any() or work["regime_available_utc"].duplicated().any():
        raise CausalLeafHealthError("shared causal regime context has invalid or duplicate availability timestamps")
    work = work.sort_values("regime_available_utc", kind="stable")
    values = work.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return work["regime_available_utc"].astype("int64").to_numpy(), values, tuple(columns)


def _asof_context(times: np.ndarray, values: np.ndarray, timestamp: pd.Timestamp) -> tuple[pd.Timestamp, np.ndarray]:
    position = int(np.searchsorted(times, int(pd.Timestamp(timestamp).value), side="right") - 1)
    if position < 0:
        raise CausalLeafHealthError("every candidate requires a prior-available regime context row")
    return pd.Timestamp(times[position], tz="UTC"), values[position]


def _reader_events(connection: duckdb.DuckDBPyConnection, sql: str, batch_rows: int) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Yield one candidate/head and its already-collapsed family rows at once."""

    reader = connection.execute(sql).to_arrow_reader(batch_size=int(batch_rows))
    pending: list[dict[str, Any]] = []
    key: tuple[Any, ...] | None = None
    first: dict[str, Any] | None = None
    for batch in reader:
        frame = batch.to_pandas()
        for row in frame.to_dict("records"):
            next_key = (row["candidate_id"], row["decision_ts"], row["side_name"], row["head_name"], row["fold_id"], row["transport"], row["meta_partition"])
            if key is not None and next_key != key:
                assert first is not None
                yield first, pending
                pending = []
                first = None
            if first is None:
                first = {name: row[name] for name in _CANDIDATE_FIELDS}
                key = next_key
            pending.append({
                "rule_signature": str(row["rule_signature"]),
                "contribution_direction": str(row["contribution_direction"]),
                "family_ensemble_tree_contribution": float(row["family_ensemble_tree_contribution"]),
            })
    if first is not None:
        yield first, pending


def _global_snapshots(connection: duckdb.DuckDBPyConnection, candidate_view: str, batch_rows: int) -> dict[tuple[str, int], tuple[int, float]]:
    """Global H1 priors immediately before every decision timestamp."""

    times = connection.execute(
        f"SELECT DISTINCT feature_generation_ts FROM {candidate_view} ORDER BY feature_generation_ts"
    ).fetchall()
    reader = connection.execute(
        f"SELECT feature_contract_sha256, label_available_ts, semantic_label FROM {candidate_view} "
        "ORDER BY label_available_ts, candidate_id, head_name"
    ).to_arrow_reader(batch_size=int(batch_rows))
    label_iter = (row for batch in reader for row in batch.to_pandas().to_dict("records"))
    next_label = next(label_iter, None)
    totals: dict[str, list[float]] = {}
    result: dict[tuple[str, int], tuple[int, float]] = {}
    for (time,) in times:
        ts = pd.Timestamp(time)
        while next_label is not None and pd.Timestamp(next_label["label_available_ts"]) < ts:
            contract = str(next_label["feature_contract_sha256"])
            state = totals.setdefault(contract, [0.0, 0.0])
            state[0] += 1.0
            state[1] += float(next_label["semantic_label"])
            next_label = next(label_iter, None)
        for contract, state in totals.items():
            result[(contract, int(ts.value))] = (int(state[0]), float(state[1]))
    return result


def _write_state_rows(path: Path, rows: Iterator[dict[str, Any]], *, flush_rows: int = 25_000) -> int:
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    count = 0
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= flush_rows:
                frame = pd.DataFrame(buffer)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                count += len(frame)
                buffer.clear()
        if buffer:
            frame = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(frame)
        if writer is None:
            pd.DataFrame().to_parquet(path, index=False, compression="zstd")
    finally:
        if writer is not None:
            writer.close()
    return count


def _scope_state_rows(
    connection: duckdb.DuckDBPyConnection, *, candidate_view: str, contribution_view: str,
    contract: str, side: str, head: str, context_times: np.ndarray, context_values: np.ndarray,
    context_columns: Sequence[str], config: CausalLeafHealthConfig,
    global_snapshots: dict[tuple[str, int], tuple[int, float]], batch_rows: int,
) -> tuple[Iterator[dict[str, Any]], dict[tuple[str, str, str, str, str], _FamilyStats]]:
    """Produce exact H1--H3 rows for one independent side/head scope."""

    condition = (
        f"c.feature_contract_sha256={_literal(contract)} AND c.side_name={_literal(side)} "
        f"AND c.head_name={_literal(head)}"
    )
    select = ", ".join(f"c.{field}" for field in _CANDIDATE_FIELDS)
    join = (
        f"SELECT {select}, f.rule_signature, f.contribution_direction, f.family_ensemble_tree_contribution "
        f"FROM {candidate_view} c INNER JOIN {contribution_view} f "
        "ON c.candidate_id=f.candidate_id AND c.decision_ts=f.__ts__ "
        "AND c.side_name=f.side_name AND c.head_name=f.head_name AND c.fold_id=f.fold_id "
        f"WHERE {condition} "
    )
    score_events = _reader_events(connection, join + "ORDER BY c.feature_generation_ts, c.candidate_id, c.head_name, f.rule_signature, f.contribution_direction", batch_rows)
    update_events = _reader_events(connection, join + "ORDER BY c.label_available_ts, c.candidate_id, c.head_name, f.rule_signature, f.contribution_direction", batch_rows)
    families: dict[tuple[str, str, str, str, str], _FamilyStats] = {}
    side_stats = _FamilyStats()
    next_update = next(update_events, None)
    h2_snapshot: dict[tuple[str, str, str, str, str], dict[str, float | str]] = {}
    h3_snapshot: dict[Any, Any] = {}
    snapshot_period: str | None = None

    def update(event: tuple[dict[str, Any], list[dict[str, Any]]]) -> None:
        row, contributions = event
        args = dict(success=float(row["semantic_label"]), prediction=float(row["head_prediction"]), net_bps=float(row["net_bps"]), base_expected_bps=float(row["base_expected_bps"]), decision_ts=pd.Timestamp(row["decision_ts"]), asset=str(row["asset"]))
        side_stats.update(**args, context=None, h3_selected=False, max_h3_rows=int(config.h3_max_rows_per_family))
        _, context = _asof_context(context_times, context_values, pd.Timestamp(row["feature_generation_ts"]))
        for contribution in contributions:
            key = (str(row["feature_contract_sha256"]), str(row["side_name"]), str(row["head_name"]), contribution["rule_signature"], contribution["contribution_direction"])
            stats = families.setdefault(key, _FamilyStats())
            stats.update(**args, context=context, h3_selected=key in config.selected_context_families, max_h3_rows=int(config.h3_max_rows_per_family))

    def generate() -> Iterator[dict[str, Any]]:
        nonlocal next_update, h2_snapshot, h3_snapshot, snapshot_period
        for row, contributions in score_events:
            current_time = pd.Timestamp(row["feature_generation_ts"])
            while next_update is not None and pd.Timestamp(next_update[0]["label_available_ts"]) < current_time:
                update(next_update)
                next_update = next(update_events, None)
            period = str(current_time.strftime("%Y-%m"))
            if period != snapshot_period:
                h2_snapshot = _snapshot_portability(families, current_time, config)
                h3_snapshot = _context_model_snapshot(families, config)
                snapshot_period = period
            global_rows, global_successes = global_snapshots.get((contract, int(current_time.value)), (0, 0.0))
            global_stats = _FamilyStats(rows=int(global_rows), successes=float(global_successes))
            _, context = _asof_context(context_times, context_values, current_time)
            selection_active = _family_selection_active(current_time, config)
            for contribution in contributions:
                key = (contract, side, head, contribution["rule_signature"], contribution["contribution_direction"])
                stats = families.get(key, _FamilyStats())
                h1 = _family_h1(stats, side_stats, global_stats, config)
                h2 = h2_snapshot.get(key, _portability_metrics(stats, current_time, config))
                h3_selected = selection_active and key in config.selected_context_families
                h4_selected = selection_active and key in config.selected_covariance_families
                h5_selected = selection_active and key in config.selected_relationship_families
                model = h3_snapshot.get(key) if h3_selected else None
                if model is not None and np.isfinite(context).all():
                    error = model.predict(context)
                    compatibility = float(np.exp(-abs(error) / max(model.residual_scale, 1e-6)))
                    confidence = float(model.rows / (model.rows + float(config.h3_min_rows)))
                    h3 = (1.0, compatibility, error, confidence, (1.0 - compatibility) * confidence)
                else:
                    h3 = (0.0, 0.0, 0.0, 0.0, 0.0)
                output = {
                    **{name: row[name] for name in _CANDIDATE_FIELDS if name not in {"semantic_label", "head_prediction", "net_bps", "base_expected_bps", "asset"}},
                    "rule_signature": contribution["rule_signature"], "contribution_direction": contribution["contribution_direction"],
                    "family_ensemble_tree_contribution": contribution["family_ensemble_tree_contribution"],
                    "h4_selection_active": float(h4_selected), "h5_selection_active": float(h5_selected),
                    "h1_posterior_correctness": h1["posterior_correctness"], "h1_posterior_lower_95": h1["posterior_lower_95"], "h1_row_support": h1["row_support"], "h1_timestamp_support": h1["timestamp_support"], "h1_day_support": h1["day_support"], "h1_symbol_support": h1["symbol_support"], "h1_support_score": h1["support_score"], "h1_calibration_residual": h1["calibration_residual"], "h1_economic_residual_bps": h1["economic_residual_bps"], "h1_false_positive_loss_bps": h1["false_positive_loss_bps"],
                    "h2_period_count": h2["period_count"], "h2_supported_period_count": h2["supported_period_count"], "h2_observed_variance": h2["observed_variance"], "h2_sampling_variance": h2["sampling_variance"], "h2_excess_variance": h2["excess_variance"], "h2_robust_z_excess_variance": h2.get("robust_z_excess_variance", np.nan), "h2_sign_reversal_rate": h2["sign_reversal_rate"], "h2_worst_damage_bps": h2["worst_damage_bps"], "h2_calibration_drift": h2["calibration_drift"], "h2_support_failure": h2["support_failure"], "h2_instability": h2["instability"], "h2_classification": h2["classification"],
                    "h3_availability": h3[0], "h3_compatibility": h3[1], "h3_expected_error_bps": h3[2], "h3_confidence": h3[3], "h3_unexplained_break": h3[4], "regime_available_utc": _asof_context(context_times, context_values, current_time)[0],
                }
                output.update({column: float(value) for column, value in zip(context_columns, context, strict=True)})
                yield output
        while next_update is not None:
            update(next_update)
            next_update = next(update_events, None)
    return generate(), families


def _section_select(section: str, metrics: Sequence[str], *, source: str) -> tuple[list[str], list[str]]:
    """SQL expressions and output names for one contribution-weighted section."""

    expressions: list[str] = []
    names: list[str] = []
    for head in HEADS:
        for direction in DIRECTIONS:
            predicate = f"s.head_name={_literal(head)} AND s.contribution_direction={_literal(direction)}"
            denominator = f"SUM(CASE WHEN {predicate} THEN abs(s.family_ensemble_tree_contribution) ELSE 0.0 END)"
            for metric in metrics:
                name = f"base_health__{section}__{head}__{direction}__{metric}"
                if section in {"h1", "h2", "h3"}:
                    source_metric = f"{section}_{metric}"
                elif section == "h4" and metric != "availability":
                    source_metric = f"base_health__h4__{metric}"
                else:
                    source_metric = metric
                expressions.append(
                    f"CAST(COALESCE(SUM(CASE WHEN {predicate} THEN abs(s.family_ensemble_tree_contribution) * COALESCE(s.\"{source_metric}\", 0.0) ELSE 0.0 END) / NULLIF({denominator}, 0.0), 0.0) AS FLOAT) AS \"{name}\""
                )
                names.append(name)
            weight = f"base_health__{section}__{head}__{direction}__active_abs_contribution"
            expressions.append(f"CAST(COALESCE({denominator}, 0.0) AS FLOAT) AS \"{weight}\"")
            names.append(weight)
    return expressions, names


def _copy_aggregate_sections(
    connection: duckdb.DuckDBPyConnection, *, candidate_view: str, state_glob: str,
    output: Path, sections: Sequence[tuple[str, Sequence[str]]],
) -> list[str]:
    expressions: list[str] = []
    names: list[str] = []
    for section, metrics in sections:
        current, current_names = _section_select(section, metrics, source="s")
        expressions.extend(current)
        names.extend(current_names)
    identity = ", ".join(f"b.{name}" for name in _IDENTITY)
    group = ", ".join(f"b.{name}" for name in _IDENTITY)
    sql = (
        f"COPY (SELECT {identity}, {', '.join(expressions)} FROM "
        f"(SELECT DISTINCT {', '.join(_IDENTITY)} FROM {candidate_view}) b "
        f"LEFT JOIN read_parquet({_literal(state_glob)}, union_by_name=true) s USING ({', '.join(_IDENTITY)}) "
        f"GROUP BY {group}) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(sql)
    return names


def _copy_h5(
    connection: duckdb.DuckDBPyConnection, *, candidate_view: str, relationship_path: Path,
    output: Path, pairs: Sequence[str],
) -> list[str]:
    names: list[str] = []
    expressions: list[str] = []
    for head in HEADS:
        for direction in DIRECTIONS:
            prefix = f"base_health__h5__{head}__{direction}"
            predicate = f"r.head_name={_literal(head)} AND r.contribution_direction={_literal(direction)}"
            weight = "COALESCE(r.portable_economic_weight, 0.0)"
            denominator = f"SUM(CASE WHEN {predicate} THEN {weight} ELSE 0.0 END)"
            cells = {
                "weighted_break": f"SUM(CASE WHEN {predicate} THEN COALESCE(r.relationship_break, 0.0) * {weight} ELSE 0.0 END)",
                "material_break_share": f"SUM(CASE WHEN {predicate} THEN CAST(COALESCE(r.material_break, false) AS DOUBLE) * {weight} ELSE 0.0 END)",
                "worst_break": f"MAX(CASE WHEN {predicate} THEN COALESCE(r.relationship_break, 0.0) ELSE 0.0 END)",
            }
            for suffix, numerator in cells.items():
                name = f"{prefix}__{suffix}"
                if suffix == "worst_break":
                    expressions.append(f"CAST(COALESCE({numerator}, 0.0) AS FLOAT) AS \"{name}\"")
                else:
                    expressions.append(f"CAST(COALESCE({numerator} / NULLIF({denominator}, 0.0), 0.0) AS FLOAT) AS \"{name}\"")
                names.append(name)
            availability = f"{prefix}__availability"
            expressions.append(f"CAST(({denominator} > 0.0) AS FLOAT) AS \"{availability}\"")
            names.append(availability)
            for pair in pairs:
                name = f"{prefix}__{pair}__material_break_share"
                numerator = f"SUM(CASE WHEN {predicate} AND r.relationship_pair={_literal(pair)} THEN CAST(COALESCE(r.material_break, false) AS DOUBLE) * {weight} ELSE 0.0 END)"
                expressions.append(f"CAST(COALESCE({numerator} / NULLIF({denominator}, 0.0), 0.0) AS FLOAT) AS \"{name}\"")
                names.append(name)
    identity = ", ".join(f"b.{name}" for name in _IDENTITY)
    group = ", ".join(f"b.{name}" for name in _IDENTITY)
    connection.execute(
        f"COPY (SELECT {identity}, {', '.join(expressions)} FROM "
        f"(SELECT DISTINCT {', '.join(_IDENTITY)} FROM {candidate_view}) b "
        f"LEFT JOIN read_parquet({_literal(str(relationship_path))}) r USING ({', '.join(_IDENTITY)}) "
        f"GROUP BY {group}) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return names


def _family_summaries(families: dict[tuple[str, str, str, str, str], _FamilyStats], now: pd.Timestamp, config: CausalLeafHealthConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_rows: list[dict[str, Any]] = []
    for key, stats in families.items():
        contract, side, head, signature, direction = key
        for period, value in sorted(stats.periods.items()):
            period_rows.append({"feature_contract_sha256": contract, "side_name": side, "head_name": head, "rule_signature": signature, "contribution_direction": direction, "period": period, "rows": value.rows, "independent_timestamps": len(value.timestamps), "trading_days": len(value.days), "symbols": len(value.symbols), "mean_prediction": value.prediction_sum / max(value.rows, 1), "posterior_correctness_raw": value.successes / max(value.rows, 1), "calibration_residual": value.calibration_sum / max(value.rows, 1), "mean_net_bps": value.net_sum / max(value.rows, 1), "economic_residual_bps": value.effect_bps(), "economic_residual_se_bps": value.effect_se_bps(), "false_positive_loss_bps": value.false_positive_loss_sum / max(value.rows, 1)})
    portability = _snapshot_portability(families, now, config)
    portability_rows = [{"feature_contract_sha256": key[0], "side_name": key[1], "head_name": key[2], "rule_signature": key[3], "contribution_direction": key[4], **metrics} for key, metrics in portability.items()]
    return pd.DataFrame(period_rows), pd.DataFrame(portability_rows)


def materialize_strict_oof_causal_leaf_health_streaming(
    spool_root: str | Path, output_dir: str | Path, *, causal_context: pd.DataFrame,
    context_feature_columns: Sequence[str], config: CausalLeafHealthConfig = CausalLeafHealthConfig(),
    batch_rows: int = 25_000, max_selected_state_rows: int = 3_000_000,
) -> Path:
    """Materialise exact strict H1--H5 artifacts with bounded candidate input.

    ``max_selected_state_rows`` is a safety bound only for the H4/H5 selected
    dataframe.  Exceeding it is a hard error: silently dropping selected
    families would alter the frozen health contract.
    """

    config.validate()
    if int(batch_rows) <= 0 or int(max_selected_state_rows) <= 0:
        raise CausalLeafHealthError("streaming batch and selected-state limits must be positive")
    spool = _read_spool(spool_root)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite causal leaf health artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    context_times, context_values, context_columns = _context_timeline(causal_context, context_feature_columns, config)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    state_dir = temporary / "state_parts"
    state_dir.mkdir()
    try:
        connection = duckdb.connect(database=str(temporary / "health.duckdb"))
        connection.execute("PRAGMA threads=2")
        connection.execute("PRAGMA memory_limit='2GB'")
        connection.execute(f"CREATE VIEW candidates AS SELECT * FROM read_parquet({_glob(spool.root / 'candidate_parts')}, union_by_name=true)")
        connection.execute(f"CREATE VIEW contributions AS SELECT * FROM read_parquet({_glob(spool.root / 'contribution_parts')}, union_by_name=true)")
        invalid = connection.execute("SELECT count(*) FROM contributions WHERE family_ensemble_tree_contribution=0 OR rule_signature IS NULL OR trim(rule_signature)='' OR contribution_direction NOT IN ('positive','negative')").fetchone()[0]
        if int(invalid):
            raise CausalLeafHealthError("spooled family contributions fail the token-free numeric contract")
        unmatched = connection.execute("SELECT count(*) FROM contributions f LEFT JOIN candidates c ON c.candidate_id=f.candidate_id AND c.decision_ts=f.__ts__ AND c.side_name=f.side_name AND c.head_name=f.head_name AND c.fold_id=f.fold_id WHERE c.candidate_id IS NULL").fetchone()[0]
        if int(unmatched):
            raise CausalLeafHealthError("spooled family contribution cannot prove candidate/head provenance")
        snapshots = _global_snapshots(connection, "candidates", int(batch_rows))
        scopes = connection.execute("SELECT DISTINCT feature_contract_sha256, side_name, head_name FROM candidates ORDER BY 1,2,3").fetchall()
        period_parts: list[Path] = []
        portability_parts: list[Path] = []
        total_states = 0
        for index, (contract, side, head) in enumerate(scopes):
            rows, families = _scope_state_rows(connection, candidate_view="candidates", contribution_view="contributions", contract=str(contract), side=str(side), head=str(head), context_times=context_times, context_values=context_values, context_columns=context_columns, config=config, global_snapshots=snapshots, batch_rows=int(batch_rows))
            state_path = state_dir / f"scope_{index:03d}.parquet"
            total_states += _write_state_rows(state_path, rows)
            now = pd.Timestamp(connection.execute("SELECT max(feature_generation_ts) FROM candidates").fetchone()[0]) + pd.Timedelta(hours=int(config.period_close_lag_hours) + 1)
            period, portability = _family_summaries(families, now, config)
            if not period.empty:
                path = temporary / f"period_{index:03d}.parquet"; period.to_parquet(path, index=False, compression="zstd"); period_parts.append(path)
            if not portability.empty:
                portability["is_portable"] = portability["classification"].eq("STABLE_PORTABLE")
                path = temporary / f"portability_{index:03d}.parquet"; portability.to_parquet(path, index=False, compression="zstd"); portability_parts.append(path)

        state_glob = str(state_dir / "*.parquet")
        h123_path = temporary / "h123.parquet"
        h123_names = _copy_aggregate_sections(connection, candidate_view="candidates", state_glob=state_glob, output=h123_path, sections=(("h1", ("posterior_correctness", "posterior_lower_95", "row_support", "timestamp_support", "day_support", "symbol_support", "support_score", "calibration_residual", "economic_residual_bps", "false_positive_loss_bps")), ("h2", ("period_count", "supported_period_count", "observed_variance", "sampling_variance", "excess_variance", "robust_z_excess_variance", "sign_reversal_rate", "worst_damage_bps", "calibration_drift", "support_failure", "instability")), ("h3", ("availability", "compatibility", "expected_error_bps", "confidence", "unexplained_break"))))

        selections = set(config.selected_covariance_families) | set(config.selected_relationship_families)
        selected_path = temporary / "selected_states.parquet"
        if selections:
            key_rows = pd.DataFrame(sorted(selections), columns=["feature_contract_sha256", "side_name", "head_name", "rule_signature", "contribution_direction"])
            connection.register("selected_keys", key_rows)
            count = connection.execute(f"SELECT count(*) FROM read_parquet({_literal(state_glob)}, union_by_name=true) s INNER JOIN selected_keys k USING (feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction)").fetchone()[0]
            if int(count) > int(max_selected_state_rows):
                raise CausalLeafHealthError(f"selected H4/H5 state rows ({count}) exceed bounded limit ({max_selected_state_rows}); refusing an unbounded allocation")
            connection.execute(f"COPY (SELECT s.* FROM read_parquet({_literal(state_glob)}, union_by_name=true) s INNER JOIN selected_keys k USING (feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction)) TO {_literal(str(selected_path))} (FORMAT PARQUET, COMPRESSION ZSTD)")
            selected_states = pd.read_parquet(selected_path)
        else:
            selected_states = pd.DataFrame()
        covariance, relationships, field_audit = _materialise_h4_h5(selected_states, context_columns=context_columns, config=config)
        covariance_path = temporary / "leaf_covariance_diagnostics.parquet"; covariance.to_parquet(covariance_path, index=False, compression="zstd")
        relationship_path = temporary / "leaf_relationship_breaks.parquet"; relationships.to_parquet(relationship_path, index=False, compression="zstd")
        h4_path = temporary / "h4.parquet"
        h4_metrics = [name.removeprefix("base_health__h4__") for name in covariance_feature_names("base_health__h4")] + ["availability"]
        if covariance.empty:
            # An empty table still has the required metric surface.  The SQL
            # aggregate emits causal zeros, exactly like the reference builder.
            empty = pd.DataFrame(columns=[*_IDENTITY, "head_name", "contribution_direction", "family_ensemble_tree_contribution", *[f"base_health__h4__{name}" for name in h4_metrics if name != "availability"], "availability"])
            empty.to_parquet(covariance_path, index=False, compression="zstd")
        else:
            covariance["availability"] = 1.0
            covariance.to_parquet(covariance_path, index=False, compression="zstd")
        h4_names = _copy_aggregate_sections(connection, candidate_view="candidates", state_glob=str(covariance_path), output=h4_path, sections=(("h4", h4_metrics),))
        h5_path = temporary / "h5.parquet"
        h5_names = _copy_h5(connection, candidate_view="candidates", relationship_path=relationship_path, output=h5_path, pairs=sorted(_relationship_break_columns(context_columns)))
        health_path = temporary / "base_leaf_health_features_oof.parquet"
        extras = [*h4_names, *h5_names]
        connection.execute(f"COPY (SELECT h.*, {', '.join(f'x.\"{name}\"' for name in h4_names)}, {', '.join(f'y.\"{name}\"' for name in h5_names)} FROM read_parquet({_literal(str(h123_path))}) h JOIN read_parquet({_literal(str(h4_path))}) x USING ({', '.join(_IDENTITY)}) JOIN read_parquet({_literal(str(h5_path))}) y USING ({', '.join(_IDENTITY)}) ORDER BY h.transport, h.meta_partition, h.decision_ts, h.candidate_id) TO {_literal(str(health_path))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        # Compact all per-scope state parts only after H4/H5 diagnostics have
        # been frozen; the raw state table intentionally retains selection flags.
        state_output = temporary / "base_leaf_family_candidate_states.parquet"
        connection.execute(f"COPY (SELECT * FROM read_parquet({_literal(state_glob)}, union_by_name=true) ORDER BY transport, meta_partition, decision_ts, candidate_id, head_name, rule_signature, contribution_direction) TO {_literal(str(state_output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        period_output = temporary / "leaf_period_metrics.parquet"
        portability_output = temporary / "leaf_portability_scores.parquet"
        if period_parts: connection.execute(f"COPY (SELECT * FROM read_parquet({_literal(str(temporary / 'period_*.parquet'))}, union_by_name=true)) TO {_literal(str(period_output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        else: pd.DataFrame().to_parquet(period_output, index=False, compression="zstd")
        if portability_parts: connection.execute(f"COPY (SELECT * FROM read_parquet({_literal(str(temporary / 'portability_*.parquet'))}, union_by_name=true)) TO {_literal(str(portability_output))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        else: pd.DataFrame().to_parquet(portability_output, index=False, compression="zstd")
        explain = pd.DataFrame({"status": ["NOT_FITTED_IN_STATE_MATERIALISATION"], "reason": ["C5/C6 held-out explanatory regressions belong to the later transport ablation, not prequential feature generation"], "uses_outcomes": [False]})
        explain.to_parquet(temporary / "covariance_explainability.parquet", index=False, compression="zstd")
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (state_output, health_path, period_output, portability_output, covariance_path, relationship_path, temporary / "covariance_explainability.parquet")}
        health_rows = int(connection.execute(
            f"SELECT count(*) FROM read_parquet({_literal(str(health_path))})"
        ).fetchone()[0])
        manifest = {
            "schema": SCHEMA,
            "status": STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "strict_roots": list(spool.strict_roots),
            "strict_root_manifest_sha256": spool.strict_root_manifest_sha256,
            "strict_input_spool_manifest_sha256": hashlib.sha256(spool.manifest_path.read_bytes()).hexdigest(),
            "contract": {
                "family_identity": "feature_contract_sha256, side, head, rule_signature, contribution_direction",
                "raw_leaf_ids": "rejected; only token-free same-artifact rule-family contributions are accepted",
                "history": "only label_available_ts < feature_generation_ts; all same-timestamp candidates are scored before any update",
                "streaming": "ordered score/update passes are scope-local; global posterior snapshots are computed from all prior-resolved candidate/head labels",
                "covariance": "H4 uses selected families, compact predeclared causal context, two horizons and no outcome fields",
                "relationship_breaks": "H5 uses selected causal relationship residuals and frozen portability/economic weighting",
            },
            "config": _config_payload(config),
            "context_columns": list(context_columns),
            "covariance_field_audit": field_audit.to_dict("records"),
            "row_counts": {
                "family_candidate_states": int(total_states),
                "health_features": health_rows,
                "covariance_diagnostics": int(len(covariance)),
                "relationship_breaks": int(len(relationships)),
            },
            "sha256": hashes,
        }
        (temporary / "leaf_covariance_reference_manifest.json").write_text(json.dumps({"schema": SCHEMA, "status": "CAUSAL_COVARIANCE_REFERENCE_DECLARED", "contract": manifest["contract"]["covariance"], "context_columns": list(context_columns), "covariance_field_audit": manifest["covariance_field_audit"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "leaf_failure_classification.yaml").write_text(json.dumps({"schema": SCHEMA, "classification_counts": {}, "labels": ["STABLE_PORTABLE", "SUPPORT_SHIFT_ONLY", "CALIBRATION_DRIFT", "COMPOSITION_DRIFT", "REGIME_CONDITIONAL", "COVARIANCE_CONDITIONAL", "GLOBAL_PERIOD_SENSITIVE", "UNEXPLAINED_CONCEPT_BREAK", "LOW_SUPPORT_UNCERTAIN", "META_HARMFUL"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "health_materialization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        connection.close()
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["materialize_strict_oof_causal_leaf_health_streaming"]
