#!/usr/bin/env python3
"""Materialise the matched exact-H12 ledger for the controlled T0--T4 x S0--S5 study.

This is a data-only preparatory step.  It joins the authoritative exact-H12
primary/supportive target pack to the frozen raw causal panel, creates the
predeclared 12+4+4 chronological protocol folds, and writes no model or OOF
prediction.  A completed v2 target pack is preferred automatically; a v1
fallback is explicit in the manifest so it cannot silently masquerade as v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.controlled_target_supportive_ablation import ContractError, derive_economic_targets, validate_causal_raw_features

SCHEMA = "controlled_target_supportive_prepared_ledger_v1"
DEFAULT_RAW_PANEL = ROOT / "data_perp/artifacts/long_exact_h12_raw_base_panel_20260730_v2"
DEFAULT_TARGET_V2 = ROOT / "data_perp/artifacts/root_cause_exact_h12_execution_target_pack_20260801_v2"
DEFAULT_TARGET_V1 = ROOT / "data_perp/artifacts/root_cause_exact_h12_execution_target_pack_20260801_v1"
DEFAULT_NATIVE_SOURCES = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
    ROOT / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet",
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v1"
REQUIRED_PACK_FILES = ("primary_labels.parquet", "supportive_labels.parquet", "manifest.json", "execution_target_contract.json")
NATIVE_EVENT = "__soft_tb_first_event__"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _path_literal(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _duckdb() -> Any:
    """Import lazily so pure contract tests need no optional SQL engine."""
    try:
        import duckdb
    except ModuleNotFoundError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "materialization requires duckdb; use the project runtime that provides it"
        ) from error
    return duckdb


def _is_completed_target_pack(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_PACK_FILES)


def resolve_target_pack(explicit: Path | None = None) -> tuple[Path, str]:
    """Prefer only a *completed* v2 directory; never read/alter a staging dir."""
    if explicit is not None:
        if not _is_completed_target_pack(explicit):
            raise FileNotFoundError(f"target pack is incomplete: {explicit}")
        return explicit, "explicit"
    if _is_completed_target_pack(DEFAULT_TARGET_V2):
        return DEFAULT_TARGET_V2, "v2_preferred"
    if _is_completed_target_pack(DEFAULT_TARGET_V1):
        return DEFAULT_TARGET_V1, "v1_fallback_v2_not_completed"
    raise FileNotFoundError("neither a completed exact-H12 v2 nor v1 target pack exists")


def protocol_folds(timestamps: Iterable[object]) -> pd.DataFrame:
    """The fixed 12-month base / 4-month meta-fit / 4-month meta-OOS protocol."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    start = pd.Timestamp("2023-04-01T00:00:00Z")
    boundaries = (start, start + pd.DateOffset(months=12), start + pd.DateOffset(months=16), start + pd.DateOffset(months=20))
    names = ("base_train", "meta_train", "meta_oos")
    roles = ("base_fit_support_warmup", "base_oos_meta_fit", "base_oos_meta_heldout")
    output = pd.DataFrame({"__ts__": ts})
    assigned = np.full(len(output), None, dtype=object)
    for order, (name, role, left, right) in enumerate(zip(names, roles, boundaries[:-1], boundaries[1:], strict=True)):
        mask = output.__ts__.ge(left) & output.__ts__.lt(right)
        assigned[mask.to_numpy()] = name
    if pd.isna(assigned).any():
        raise ValueError("timestamps fall outside the fixed 20-month target protocol")
    output["oof_fold"] = assigned
    output["protocol_role"] = output.oof_fold.map(dict(zip(names, roles, strict=True)))
    output["fold_order"] = output.oof_fold.map(dict(zip(names, range(len(names)), strict=True))).astype(np.int8)
    return output.drop(columns="__ts__")


def fold_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oof_fold": ["base_train", "meta_train", "meta_oos"],
            "fold_order": [0, 1, 2],
            "start_utc": pd.to_datetime(["2023-04-01", "2024-04-01", "2024-08-01"], utc=True),
            "end_exclusive_utc": pd.to_datetime(["2024-04-01", "2024-08-01", "2024-12-01"], utc=True),
            "protocol_role": ["base_fit_support_warmup", "base_oos_meta_fit", "base_oos_meta_heldout"],
            "training_rule": ["no prior rows; retained solely to seed strict OOF support", "train only base_train labels available before 2024-04-01", "train only base_train/meta_train labels available before 2024-08-01"],
        }
    )


def _native_control_expression() -> str:
    # This source does not contain the archived native 24h first-touch target.
    # The alias is retained so the generic runner can execute T0, but the
    # manifest makes the H12 reconstruction caveat unambiguous.
    return f"CASE n.{_quote(NATIVE_EVENT)} WHEN 'favorable_first' THEN 1.0 WHEN 'timeout' THEN 0.5 WHEN 'adverse_first_or_conflict' THEN 0.0 ELSE NULL END"


def _required_support_columns() -> tuple[str, ...]:
    return (
        "__peak_mfe_atr_12h__",
        "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_meaningful_mfe_atr_12h__",
        "__future_slope_atr_per_hour_12h__",
        "__path_auxiliary_target_valid__",
        "clean_economic_favorable_first",
        "adverse_first",
        "timeout",
        # Grouped S1--S5 surface used by the attached roadmap.  These are
        # explicit aliases/composites of point-in-time-resolved path labels;
        # the formula is recorded in the run manifest below.
        "__peak_mfe_within_1h_12h__",
        "__peak_mfe_within_2h_12h__",
        "__peak_mfe_within_4h_12h__",
        "__peak_mfe_within_8h_12h__",
        "__meaningful_mfe_reached_12h__",
        "__pre_mfe_mae_event_12h__",
        "__adverse_trough_atr_12h__",
        "__mae_before_1_5atr_mfe__",
        "__peak_mfe_return_12h__",
        "__peak_mfe_atr_12h__",
        "__peak_mfe_ge_1_5atr_12h__",
        "__mfe_ratio_to_peak_at_2h_12h__",
        "__mfe_ratio_to_peak_at_4h_12h__",
        "__mfe_ratio_to_peak_at_8h_12h__",
        "__mfe_persistence_path_efficiency_12h__",
        "__adverse_trough_recovery_fraction_12h__",
        "__adverse_trough_recovered_80pct_12h__",
    )


def _validate_source_contract(
    raw_dir: Path,
    target_pack: Path,
    native_sources: tuple[Path, ...],
    feature_json: Path | None = None,
) -> tuple[Path, list[str]]:
    raw_file = raw_dir / "raw_base_panel.parquet"
    feature_file = raw_dir / "raw_feature_contract.json"
    if not raw_file.is_file() or not feature_file.is_file():
        raise FileNotFoundError("raw panel must contain raw_base_panel.parquet and raw_feature_contract.json")
    feature_source = feature_json if feature_json is not None else feature_file
    if not feature_source.is_file():
        raise FileNotFoundError(f"feature contract does not exist: {feature_source}")
    feature_payload = json.loads(feature_source.read_text(encoding="utf-8"))
    features = feature_payload.get("raw_feature_columns") or feature_payload.get("feature_columns")
    if not isinstance(features, list):
        raise ValueError("raw feature contract does not contain raw_feature_columns")
    features = list(validate_causal_raw_features(features))
    raw_names = set(pq.ParquetFile(raw_file).schema.names)
    missing = {"candidate_id", "__ts__", "side_name", "__decision_ts__", *features}.difference(raw_names)
    if missing:
        raise ValueError(f"raw panel lacks frozen causal columns: {sorted(missing)}")
    primary_names = set(pq.ParquetFile(target_pack / "primary_labels.parquet").schema.names)
    support_names = set(pq.ParquetFile(target_pack / "supportive_labels.parquet").schema.names)
    required_primary = {"candidate_id", "decision_ts", "label_end_ts", "label_available_ts", "execution_exact_h12_gross_bps", "execution_exact_h12_cost_bps", "execution_exact_h12_net_bps"}
    if required_primary - primary_names or set(_required_support_columns()) - support_names:
        raise ValueError("exact-H12 target pack misses primary/supportive fields required by the ablation")
    for source in native_sources:
        names = set(pq.ParquetFile(source).schema.names)
        if {"candidate_id", NATIVE_EVENT} - names:
            raise ValueError(f"native-control source lacks candidate ID or {NATIVE_EVENT}: {source}")
    return raw_file, features


def _validate_materialized_ledger(ledger: Path, raw_rows: int, raw_features: list[str]) -> dict[str, Any]:
    columns = set(pq.ParquetFile(ledger).schema.names)
    required = {
        "candidate_id", "__ts__", "side_name", "__label_available_at__", "__first_touch_target_soft__",
        "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "__opportunity_occurred_12h__",
        "favorable_first", "adverse_first", "timeout", "oof_fold", "protocol_role", "fold_order", *raw_features,
    }
    missing = required - columns
    if missing:
        raise ValueError(f"materialised ledger lacks fields: {sorted(missing)}")
    connection = _duckdb().connect(database=":memory:")
    try:
        quoted = str(ledger).replace("'", "''")
        query = f"SELECT count(*) AS rows, count(DISTINCT candidate_id) AS unique_rows, max(abs(execution_gross_ev_12h - execution_cost_return - execution_net_ev_12h)) AS accounting_error, sum(CASE WHEN __label_available_at__ <> __ts__ + INTERVAL 13 HOUR THEN 1 ELSE 0 END) AS bad_availability, sum(CASE WHEN favorable_first + adverse_first + timeout <> 1 THEN 1 ELSE 0 END) AS bad_competing, sum(CASE WHEN __path_auxiliary_target_valid__ <> 1 THEN 1 ELSE 0 END) AS invalid_path FROM read_parquet('{quoted}')"
        result = connection.execute(query).fetchdf().iloc[0].to_dict()
    finally:
        connection.close()
    if int(result["rows"]) != raw_rows or int(result["unique_rows"]) != raw_rows:
        raise ValueError("exact target join changed candidate population or uniqueness")
    if float(result["accounting_error"]) > 1e-10 or any(int(result[key]) != 0 for key in ("bad_availability", "bad_competing", "invalid_path")):
        raise ValueError(f"prepared-ledger contract validation failed: {result}")
    return {key: (int(value) if key != "accounting_error" else float(value)) for key, value in result.items()}


def materialize(
    *,
    raw_dir: Path,
    target_pack: Path | None,
    native_sources: tuple[Path, ...],
    output: Path,
    feature_json: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    pack, selection = resolve_target_pack(target_pack)
    raw_file, raw_features = _validate_source_contract(raw_dir, pack, native_sources, feature_json)
    raw_rows = pq.ParquetFile(raw_file).metadata.num_rows
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        ledger = stage / "prepared_target_supportive_ledger.parquet"
        # DuckDB keeps the 380-column raw feature join out-of-core and avoids
        # a duplicate 300+MB pandas frame.  Every selected field is explicit.
        raw_select = ", ".join(f"r.{_quote(column)}" for column in ("candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__", *raw_features))
        primary = "p.execution_exact_h12_gross_bps / 10000.0 AS execution_gross_ev_12h, p.execution_exact_h12_cost_bps / 10000.0 AS execution_cost_return, p.execution_exact_h12_net_bps / 10000.0 AS execution_net_ev_12h, p.label_end_ts AS __label_end_ts__, p.label_available_ts AS __label_available_at__"
        support = """
            s.clean_economic_favorable_first AS __opportunity_occurred_12h__,
            s.clean_economic_favorable_first AS favorable_first,
            s.adverse_first,
            s.timeout,
            s.__peak_mfe_atr_12h__,
            s.__time_to_first_meaningful_mfe_hours_12h__,
            s.__mae_before_meaningful_mfe_atr_12h__,
            s.__future_slope_atr_per_hour_12h__,
            s.__path_auxiliary_target_valid__,
            /* S1: reach-at-1/2/4/8/12h plus a 12h-normalized speed term. */
            (
                0.15 * COALESCE(s.__peak_mfe_within_1h_12h__, 0.0)
              + 0.15 * COALESCE(s.__peak_mfe_within_2h_12h__, 0.0)
              + 0.15 * COALESCE(s.__peak_mfe_within_4h_12h__, 0.0)
              + 0.15 * COALESCE(s.__peak_mfe_within_8h_12h__, 0.0)
              + 0.25 * COALESCE(s.__meaningful_mfe_reached_12h__, 0.0)
              + 0.15 * (1.0 - LEAST(GREATEST(COALESCE(s.__time_to_first_meaningful_mfe_hours_12h__, 12.0) / 12.0, 0.0), 1.0))
            ) AS __group_s1_opportunity_reach_time__,
            /* S2: a higher value means more adverse path risk. */
            (
                0.35 * COALESCE(s.adverse_first, 0.0)
              + 0.25 * COALESCE(s.__pre_mfe_mae_event_12h__, 0.0)
              + 0.20 * LEAST(GREATEST(COALESCE(s.__adverse_trough_atr_12h__, 0.0) / 2.0, 0.0), 1.0)
              + 0.20 * LEAST(GREATEST(COALESCE(s.__mae_before_1_5atr_mfe__, 0.0) / 2.0, 0.0), 1.0)
            ) AS __group_s2_adverse_path_risk__,
            /* S3: peak magnitude plus an approximate net-of-row-cost margin. */
            (
                0.35 * LEAST(GREATEST(COALESCE(s.__peak_mfe_atr_12h__, 0.0) / 4.0, 0.0), 1.0)
              + 0.20 * COALESCE(s.__peak_mfe_ge_1_5atr_12h__, 0.0)
              + 0.20 * LEAST(GREATEST((COALESCE(s.__peak_mfe_return_12h__, 0.0) * 10000.0 - p.execution_exact_h12_cost_bps) / 200.0, -1.0), 1.0) * 0.5 + 0.10
              + 0.25 * LEAST(GREATEST(COALESCE(s.__peak_mfe_return_12h__, 0.0) * 10000.0 / 400.0, 0.0), 1.0)
            ) AS __group_s3_magnitude_net_margin__,
            /* S4: persistence at 2/4/8h and path efficiency (giveback proxy). */
            (
                0.20 * COALESCE(s.__mfe_ratio_to_peak_at_2h_12h__, 0.0)
              + 0.20 * COALESCE(s.__mfe_ratio_to_peak_at_4h_12h__, 0.0)
              + 0.20 * COALESCE(s.__mfe_ratio_to_peak_at_8h_12h__, 0.0)
              + 0.40 * COALESCE(s.__mfe_persistence_path_efficiency_12h__, 0.0)
            ) AS __group_s4_persistence_giveback__,
            /* S5: early-adverse recovery, conditional on an adverse trough. */
            CASE WHEN COALESCE(s.adverse_first, 0.0) > 0.5 THEN
                0.60 * COALESCE(s.__adverse_trough_recovery_fraction_12h__, 0.0)
              + 0.40 * COALESCE(s.__adverse_trough_recovered_80pct_12h__, 0.0)
            ELSE 0.0 END AS __group_s5_early_adverse_recovery__
        """
        # DuckDB's parameter binder treats a Python list passed to
        # ``read_parquet(?)`` inconsistently when a COPY destination is also
        # parameterised.  Render the immutable source paths as a quoted list
        # literal; single quotes are escaped and no path is interpreted as
        # user SQL.
        native_files_sql = "[" + ", ".join(_path_literal(path) for path in native_sources) + "]"
        connection = _duckdb().connect(database=":memory:")
        try:
            connection.execute("PRAGMA threads=4")
            query = f"""
                COPY (
                    SELECT {raw_select}, {primary}, {support}, {_native_control_expression()} AS __first_touch_target_soft__
                    FROM read_parquet({_path_literal(raw_file)}) AS r
                    INNER JOIN read_parquet({_path_literal(pack / 'primary_labels.parquet')}) AS p USING (candidate_id)
                    INNER JOIN read_parquet({_path_literal(pack / 'supportive_labels.parquet')}) AS s USING (candidate_id)
                    INNER JOIN read_parquet({native_files_sql}) AS n USING (candidate_id)
                    ORDER BY r.__ts__, r.candidate_id
                ) TO {_path_literal(ledger)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
            connection.execute(query)
        finally:
            connection.close()
        # Add deterministic folds in a narrow second pass.  This stays small
        # relative to path decoding and lets tests exercise fold logic directly.
        work = pd.read_parquet(ledger, columns=["candidate_id", "__ts__"])
        folds = protocol_folds(work["__ts__"])
        fold_assignment = pd.DataFrame({"candidate_id": work.candidate_id, **folds})
        folds_path = stage / "fold_assignment.parquet"
        fold_assignment.to_parquet(folds_path, index=False, compression="zstd")
        connection = _duckdb().connect(database=":memory:")
        final_ledger = stage / "prepared_target_supportive_ledger_with_folds.parquet"
        try:
            connection.execute(
                f"COPY (SELECT l.*, f.oof_fold, f.protocol_role, f.fold_order FROM read_parquet({_path_literal(ledger)}) AS l INNER JOIN read_parquet({_path_literal(folds_path)}) AS f USING (candidate_id) ORDER BY l.__ts__, l.candidate_id) TO {_path_literal(final_ledger)} (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            connection.close()
        ledger.unlink()
        final_ledger.rename(ledger)
        validation = _validate_materialized_ledger(ledger, int(raw_rows), raw_features)
        # This also checks the expected targets/event simplex on a small
        # deterministic sample without loading the full 380-feature matrix.
        sample = pd.read_parquet(ledger, columns=["candidate_id", "__first_touch_target_soft__", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "__opportunity_occurred_12h__", "favorable_first", "adverse_first", "timeout"]).head(10_000)
        derive_economic_targets(sample)
        features_path = stage / "frozen_raw_causal_features.json"
        _write_json(features_path, {"raw_feature_columns": raw_features, "raw_feature_count": len(raw_features), "source": str(raw_dir / "raw_feature_contract.json")})
        table_path = stage / "chronological_protocol_folds.parquet"
        fold_table().to_parquet(table_path, index=False, compression="zstd")
        native_caveat = "T0 is a reconstructed 12-hour soft triple-barrier alpha control from __soft_tb_first_event__, exposed under the generic runner alias __first_touch_target_soft__. It is not the archived native 24-hour first-touch label; no T0 result may be interpreted as a native-24h parity or promotion claim."
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_RESEARCH_ONLY_PREPARED_INPUT_NO_MODEL_NO_PROMOTION",
            "target_pack": {"path": str(pack), "selection": selection, "manifest_sha256": _sha256(pack / "manifest.json")},
            "raw_panel": {"path": str(raw_file), "sha256": _sha256(raw_file), "raw_feature_count": len(raw_features)},
            "feature_contract": {
                "path": str((feature_json or (raw_dir / "raw_feature_contract.json"))),
                "sha256": _sha256(feature_json or (raw_dir / "raw_feature_contract.json")),
                "selection": "explicit causal feature manifest" if feature_json is not None else "raw panel contract",
            },
            "native_control_sources": {str(path): _sha256(path) for path in native_sources},
            "native_control_horizon_caveat": native_caveat,
            "targets": {"T0": "reconstructed_h12_soft_alpha_control_only", "T1": "clean_economic_favorable_first", "T2": "authoritative exact-H12 net replay", "T3": "row-cost-aware timeout/adverse/clean competing risk + exact net", "T4": "clean/positive/adverse hurdle decomposition + exact net"},
            "grouped_support_surface": {
                "S1": "0.15 reach by 1h + 0.15 by 2h + 0.15 by 4h + 0.15 by 8h + 0.25 reach by 12h + 0.15*(1-time_to_meaningful_mfe/12), clipped to [0,1]",
                "S2": "0.35 adverse-first + 0.25 pre-MFE adverse event + 0.20 clipped adverse trough ATR/2 + 0.20 clipped MAE before 1.5 ATR/2",
                "S3": "peak MFE ATR, 1.5 ATR reach, peak gross return minus exact row cost, and peak return; normalized/clipped",
                "S4": "MFE persistence ratios at 2/4/8h plus persistence path efficiency",
                "S5": "conditional early adverse recovery: 0.60 recovery fraction + 0.40 80%-recovered indicator; zero when no adverse trough",
                "semantics": "finite causal-support labels; all support predictions are strict chronological OOF before entering a later target model",
            },
            "fold_protocol": "12m base support warmup (2023-04..2024-03), 4m meta fit (2024-04..07), 4m untouched meta OOS (2024-08..11); labels available decision+12h",
            "validation": validation,
            "outputs": {},
        }
        for path in (ledger, folds_path, features_path, table_path):
            manifest["outputs"][path.name] = _sha256(path)
        _write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-panel", type=Path, default=DEFAULT_RAW_PANEL)
    parser.add_argument("--target-pack", type=Path, default=None, help="Optional completed exact-H12 target pack; default prefers completed v2")
    parser.add_argument("--features-json", type=Path, default=None, help="Optional explicit causal feature manifest; overrides the raw panel's broad feature contract")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(materialize(raw_dir=args.raw_panel, target_pack=args.target_pack, native_sources=DEFAULT_NATIVE_SOURCES, output=args.output, feature_json=args.features_json), indent=2, default=str))


if __name__ == "__main__":
    main()
