#!/usr/bin/env python3
"""Materialize source-separated exact-path and execution labels for a backcast.

The physical path targets are derived from the unadjusted 720-minute OHLC path.
The policy-economics targets are carried from the separately signed candidate-
local policy replay.  A joined artifact is emitted for multi-task learning, but
the manifest preserves the distinction and no historical execution-parity or
OOF claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_labels import (
    PATH_ARCHETYPE_RULE_VERSION,
    _deterministic_path_archetypes_batch,
    _deterministic_realization_strength_batch,
    _summarize_side_relative_path_batch,
)
from extreme_price_movements.path_auxiliary_targets import (
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    TARGET_COLUMNS,
    TARGET_SCHEMA,
    build_path_auxiliary_targets,
)
from scripts.materialize_febapr2025_exact1m_path_head_labels import (
    HORIZONS_HOURS,
    _decode_paths,
    _execution_adjusted_path,
    _side_sign,
    _soft_triple_barrier,
)


SCHEMA = "historical_backcast_exact1m_execution_path_labels_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
HORIZON_MINUTES = 720
POLICY_COLUMNS = (
    "execution_decision_utc",
    "policy_archetype",
    "execution_geometry_key",
    "execution_geometry_source",
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
    "execution_exit_reason",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_entry_price",
    "execution_exit_price",
    "execution_expected_spread_bps",
    "execution_entry_half_spread_bps",
    "execution_exit_half_spread_bps",
    "execution_label_end_utc",
    "execution_label_available_at",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    output = frame.copy()
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
    output["side_name"] = output["side_name"].astype(str).str.lower()
    if not output["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{source} has non-canonical side")
    if output.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError(f"{source} has duplicate exact identities")
    return output


def _load_context(
    path_targets_path: Path,
    timing_candidates_path: Path,
    policy_labels_path: Path,
) -> pd.DataFrame:
    targets = _canonical_identity(
        pd.read_parquet(
            path_targets_path,
            columns=[
                *IDENTITY,
                "__barrier_pct__",
                "__path_auxiliary_atr_fraction__",
            ],
        ),
        source="path targets",
    )
    timing = _canonical_identity(
        pd.read_parquet(
            timing_candidates_path,
            columns=[
                *IDENTITY,
                "__decision_ts__",
                "atr_fraction",
                "fee",
                "entry_spread",
                "exit_spread",
            ],
        ),
        source="timing candidates",
    )
    policy = _canonical_identity(
        pd.read_parquet(
            policy_labels_path,
            columns=[*IDENTITY, *POLICY_COLUMNS],
        ),
        source="policy labels",
    )
    frame = targets.merge(
        timing,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    ).merge(policy, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(frame) != len(targets) or len(frame) != len(timing) or len(frame) != len(policy):
        raise ValueError("target, timing, and policy identities do not match exactly")
    for column in (
        "__decision_ts__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_label_available_at",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    expected_decision = frame["__ts__"] + pd.Timedelta(hours=1)
    expected_end = expected_decision + pd.Timedelta(hours=12)
    if (
        not frame["__decision_ts__"].eq(expected_decision).all()
        or not frame["execution_decision_utc"].eq(expected_decision).all()
        or not frame["execution_label_end_utc"].eq(expected_end).all()
        or not frame["execution_label_available_at"].eq(expected_end).all()
    ):
        raise ValueError("joined sources violate signal/decision/resolution timing")
    atr = pd.to_numeric(
        frame["__path_auxiliary_atr_fraction__"], errors="raise"
    ).to_numpy(float)
    timing_atr = pd.to_numeric(frame["atr_fraction"], errors="raise").to_numpy(float)
    if not np.allclose(atr, timing_atr, rtol=0.0, atol=1e-8):
        raise ValueError("timing and path-target ATR values disagree")
    fee = pd.to_numeric(frame["fee"], errors="raise").to_numpy(float)
    policy_fee = pd.to_numeric(
        frame["execution_cost_return"], errors="raise"
    ).to_numpy(float)
    if not np.allclose(fee, policy_fee, rtol=0.0, atol=1e-8):
        raise ValueError("timing and policy fee values disagree")
    for timing_column, policy_column in (
        ("entry_spread", "execution_entry_half_spread_bps"),
        ("exit_spread", "execution_exit_half_spread_bps"),
    ):
        if not np.allclose(
            pd.to_numeric(frame[timing_column], errors="raise").to_numpy(float),
            pd.to_numeric(frame[policy_column], errors="raise").to_numpy(float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"{timing_column} disagrees with policy labels")
    return frame.set_index("candidate_id", verify_integrity=True)


def _batch(paths: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    ids = paths["candidate_id"].astype(str)
    if ids.duplicated().any() or not ids.isin(context.index).all():
        raise ValueError("exact path identities are incomplete or duplicated")
    frame = context.loc[ids].reset_index()
    for column in IDENTITY:
        if column == "candidate_id":
            continue
        if column == "__ts__":
            lhs = pd.to_datetime(paths[column], utc=True)
            rhs = pd.to_datetime(frame[column], utc=True)
        else:
            lhs = paths[column].astype(str)
            rhs = frame[column].astype(str)
        if not lhs.reset_index(drop=True).eq(rhs.reset_index(drop=True)).all():
            raise ValueError(f"path identity mismatch on {column}")

    open_, high, low, close = _decode_paths(paths["execution_future_path"])
    sign = _side_sign(frame["side_name"])
    atr = pd.to_numeric(
        frame["__path_auxiliary_atr_fraction__"], errors="raise"
    ).to_numpy(float)
    barrier = pd.to_numeric(frame["__barrier_pct__"], errors="raise").to_numpy(float)
    fee = pd.to_numeric(frame["fee"], errors="raise").to_numpy(float)
    entry_spread = pd.to_numeric(
        frame["entry_spread"], errors="raise"
    ).to_numpy(float)
    exit_spread = pd.to_numeric(frame["exit_spread"], errors="raise").to_numpy(float)
    aux = build_path_auxiliary_targets(
        entry_price=open_[:, 0],
        future_high=high,
        future_low=low,
        atr_fraction=atr,
        side_sign=sign,
        bar_minutes=1,
        horizon_hours=12,
        include_supportive_columns=True,
    ).as_columns()
    triple = _soft_triple_barrier(
        high,
        low,
        entry=open_[:, 0],
        atr_fraction=atr,
        side_sign=sign,
    )
    entry, exec_high, exec_low, exec_close = _execution_adjusted_path(
        open_,
        high,
        low,
        close,
        side_sign=sign,
        entry_spread_bps=entry_spread,
        exit_spread_bps=exit_spread,
    )
    summary = _summarize_side_relative_path_batch(
        exec_high,
        exec_low,
        exec_close,
        entry_price=entry,
        risk_distance=entry * barrier,
        atr_fraction=atr,
        side_sign=sign,
        bar_hours=1.0 / 60.0,
        horizons_hours=HORIZONS_HOURS,
        take_profit_r=np.full(len(frame), np.nan),
        trailing_trigger_r=np.full(len(frame), np.nan),
        stop_r=np.ones(len(frame)),
        cost_return=fee,
        archetype_cost_return=fee,
        activation_distance_return=np.full(len(frame), np.nan),
        prefix="path_arch_",
    )
    shape = _deterministic_path_archetypes_batch(summary, prefix="path_arch_")
    strength = _deterministic_realization_strength_batch(
        summary, prefix="path_arch_"
    )
    archetype = np.full(len(frame), None, dtype=object)
    valid_archetype = pd.notna(shape) & pd.notna(strength)
    archetype[valid_archetype] = np.char.add(
        np.char.add(shape[valid_archetype].astype(str), "__"),
        strength[valid_archetype].astype(str),
    )
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True)
    encoded_start = np.asarray(
        [
            json.loads(value)["timestamp"][0]
            for value in paths["execution_future_path"]
        ],
        dtype=np.int64,
    )
    if not np.array_equal(encoded_start, decision.astype("int64").to_numpy()):
        raise ValueError("first exact-path minute does not equal decision timestamp")

    output: dict[str, Any] = {
        column: frame[column].to_numpy() for column in IDENTITY
    }
    output.update(
        {
            "__decision_ts__": decision.to_numpy(),
            "__label_end_ts__": (decision + pd.Timedelta(hours=12)).to_numpy(),
            "__label_available_at__": (
                decision + pd.Timedelta(hours=12)
            ).to_numpy(),
            "__barrier_pct__": barrier.astype(np.float32),
            "__path_auxiliary_atr_fraction__": atr.astype(np.float32),
            "path_archetype_rule_version": PATH_ARCHETYPE_RULE_VERSION,
            "path_shape_archetype": pd.array(shape, dtype="string"),
            "path_realization_strength": pd.array(strength, dtype="string"),
            "path_archetype": pd.array(archetype, dtype="string"),
        }
    )
    output.update(aux)
    output.update(triple)
    output.update(
        {column: values.astype(np.float32) for column, values in summary.items()}
    )
    for column in POLICY_COLUMNS:
        output[column] = frame[column].to_numpy()
    result = pd.DataFrame(output)

    opportunity = result["__meaningful_mfe_reached_12h__"].astype(bool)
    peak = pd.to_numeric(result["__peak_mfe_return_12h__"], errors="coerce")
    net = pd.to_numeric(result["execution_net_ev_12h"], errors="coerce")
    gross = pd.to_numeric(result["execution_gross_ev_12h"], errors="coerce")
    result["__opportunity_occurred_12h__"] = opportunity.astype(np.int8)
    result["__favorable_payoff_return_12h__"] = peak.astype(np.float32)
    result["__adverse_competing_risk_12h__"] = (
        result["__soft_tb_first_event__"] == "adverse_first_or_conflict"
    ).astype(np.int8)
    result["__timeout_outcome_12h__"] = (
        result["execution_exit_reason"].astype(str).str.contains(
            "timeout", case=False, regex=False
        )
    ).astype(np.int8)
    result["__exit_conversion_loss_return_12h__"] = np.maximum(
        peak.to_numpy(float) - gross.to_numpy(float), 0.0
    ).astype(np.float32)
    result["__opportunity_scarcity_proxy_12h__"] = (~opportunity).astype(np.int8)
    result["__exit_conversion_failure_proxy_12h__"] = (
        opportunity & (net <= 0.0)
    ).astype(np.int8)
    result["__timeout_degradation_proxy_12h__"] = (
        result["__timeout_outcome_12h__"].astype(bool) & (net < 0.0)
    ).astype(np.int8)
    result["__adverse_payoff_expansion_proxy_12h__"] = (
        pd.to_numeric(result["execution_mae_return_12h"], errors="coerce")
        >= atr
    ).astype(np.int8)
    return result


def _physical_columns(columns: list[str]) -> list[str]:
    fixed = {
        *IDENTITY,
        "__decision_ts__",
        "__label_end_ts__",
        "__label_available_at__",
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
        "__opportunity_occurred_12h__",
        "__favorable_payoff_return_12h__",
        "__adverse_competing_risk_12h__",
        "__opportunity_scarcity_proxy_12h__",
    }
    prefixes = (
        "__peak_",
        "__time_",
        "__mae_",
        "__bars_",
        "__future_",
        "__log1p_",
        "__meaningful_",
        "__soft_tb_",
        "__path_",
        "__mfe_",
        "__pre_",
        "__reaches_",
        "__hits_",
        "__fraction_",
        "__favorable_path_",
        "__adverse_trough_",
        "__trough_",
    )
    return [
        column
        for column in columns
        if column in fixed or column.startswith(prefixes)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--paths-manifest", type=Path, required=True)
    parser.add_argument("--path-targets", type=Path, required=True)
    parser.add_argument("--label-input-manifest", type=Path, required=True)
    parser.add_argument("--timing-candidates", type=Path, required=True)
    parser.add_argument("--timing-candidates-manifest", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--policy-labels-manifest", type=Path, required=True)
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=256)
    args = parser.parse_args()

    label_inputs_manifest = _json(args.label_input_manifest)
    if label_inputs_manifest.get("schema") != (
        "historical_backcast_exact1m_label_inputs_v1"
    ):
        raise ValueError("label-input manifest schema is invalid")
    if (
        label_inputs_manifest.get("outputs", {})
        .get("path_targets", {})
        .get("sha256")
        != _sha256(args.path_targets)
    ):
        raise ValueError("label-input manifest does not bind path targets")
    source_evidence_scope = label_inputs_manifest.get(
        "evidence_scope", "frozen_backcast_diagnostic_not_oof"
    )
    source_lineage = label_inputs_manifest.get(
        "lineage", "historical_frozen_backcast_exact1m_research_only"
    )
    source_economics = label_inputs_manifest.get(
        "economics", "current_frozen_spread_counterfactual"
    )
    source_historical_l2 = bool(
        label_inputs_manifest.get("historical_l2_spread_available", False)
    )
    if (
        label_inputs_manifest.get("oof_status") != "not_oof"
        or bool(label_inputs_manifest.get("execution_parity_claim"))
        or bool(label_inputs_manifest.get("promotion_eligible"))
    ):
        raise ValueError("label-input manifest is not research-only")
    policy_manifest = _json(args.policy_labels_manifest)
    policy_lineage = policy_manifest.get("historical_lineage") or {}
    if (
        policy_manifest.get("schema")
        != "execution_ev_deployed_policy_1m_labels_v1"
        or policy_manifest.get("source_artifact_sha256")
        != _sha256(args.policy_labels)
        or policy_lineage.get("oof_status") != "not_oof"
        or policy_lineage.get("evidence_scope") != source_evidence_scope
        or policy_lineage.get("lineage") != source_lineage
        or policy_lineage.get("economics") != source_economics
    ):
        raise ValueError(
            "policy-label manifest is invalid or disagrees with label-input lineage"
        )
    timing_manifest = _json(args.timing_candidates_manifest)
    if (
        timing_manifest.get("schema") != "execution_entry_timing_candidates_v1"
        or timing_manifest.get("source_artifact_sha256")
        != _sha256(args.timing_candidates)
        or (timing_manifest.get("historical_lineage") or {}).get("lineage")
        != source_lineage
        or (timing_manifest.get("historical_lineage") or {}).get("economics")
        != source_economics
    ):
        raise ValueError("timing-candidate manifest is invalid or has wrong lineage")
    paths_manifest = _json(args.paths_manifest)
    if (
        paths_manifest.get("schema") != "execution_entry_timing_1m_paths_v1"
        or paths_manifest.get("source_artifact_sha256") != _sha256(args.paths)
        or int(paths_manifest.get("rows", {}).get("output", -1))
        != int(pq.ParquetFile(args.paths).metadata.num_rows)
    ):
        raise ValueError("exact-path manifest is invalid")
    coverage_manifest = _json(args.coverage_manifest)
    if (
        coverage_manifest.get("schema")
        != "historical_exact1m_candidate_coverage_v1"
        or coverage_manifest.get("status") != "complete"
        or float(coverage_manifest.get("candidate_coverage_fraction", 0.0))
        != 1.0
        or coverage_manifest.get(
            "lineage", "historical_frozen_backcast_exact1m_research_only"
        )
        != source_lineage
        or coverage_manifest.get(
            "evidence_scope", "frozen_backcast_diagnostic_not_oof"
        )
        != source_evidence_scope
    ):
        raise ValueError(
            "candidate-level exact-path coverage is incomplete or has wrong lineage"
        )

    context = _load_context(
        args.path_targets, args.timing_candidates, args.policy_labels
    )
    source = pq.ParquetFile(args.paths)
    required = {"execution_future_path", *IDENTITY}
    if not required.issubset(source.schema.names):
        raise ValueError("exact paths are missing identity/path fields")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    joined_path = output / "joined_multitask_labels.parquet"
    physical_path = output / "physical_path_labels.parquet"
    joined_tmp = output / ".joined.partial"
    physical_tmp = output / ".physical.partial"
    joined_writer: pq.ParquetWriter | None = None
    physical_writer: pq.ParquetWriter | None = None
    rows = 0
    seen: set[str] = set()
    try:
        for batch in source.iter_batches(
            batch_size=int(args.batch_rows),
            columns=[*IDENTITY, "execution_future_path"],
        ):
            labels = _batch(batch.to_pandas(), context)
            ids = labels["candidate_id"].astype(str)
            if ids.duplicated().any() or any(value in seen for value in ids):
                raise ValueError("duplicate exact path identity")
            seen.update(ids)
            joined_table = pa.Table.from_pandas(labels, preserve_index=False)
            physical_table = pa.Table.from_pandas(
                labels.loc[:, _physical_columns(labels.columns.tolist())],
                preserve_index=False,
            )
            if joined_writer is None:
                joined_writer = pq.ParquetWriter(
                    joined_tmp, joined_table.schema, compression="zstd"
                )
                physical_writer = pq.ParquetWriter(
                    physical_tmp, physical_table.schema, compression="zstd"
                )
            joined_writer.write_table(joined_table)
            assert physical_writer is not None
            physical_writer.write_table(physical_table)
            rows += len(labels)
    finally:
        if joined_writer is not None:
            joined_writer.close()
        if physical_writer is not None:
            physical_writer.close()
    if rows != len(context) or len(seen) != len(context):
        raise ValueError("label materialization did not preserve every identity")
    os.replace(joined_tmp, joined_path)
    os.replace(physical_tmp, physical_path)

    report_frame = pd.read_parquet(
        joined_path,
        columns=[
            "__ts__",
            "side_name",
            "__opportunity_occurred_12h__",
            "__adverse_competing_risk_12h__",
            "__timeout_outcome_12h__",
            "__exit_conversion_failure_proxy_12h__",
            "execution_net_ev_12h",
            "__peak_mfe_atr_12h__",
        ],
    )
    report_frame["month"] = pd.to_datetime(
        report_frame["__ts__"], utc=True
    ).dt.strftime("%Y-%m")
    report = (
        report_frame.groupby(["month", "side_name"], sort=True)
        .agg(
            rows=("side_name", "size"),
            opportunity_rate=("__opportunity_occurred_12h__", "mean"),
            adverse_first_rate=("__adverse_competing_risk_12h__", "mean"),
            timeout_rate=("__timeout_outcome_12h__", "mean"),
            conversion_failure_rate=(
                "__exit_conversion_failure_proxy_12h__",
                "mean",
            ),
            mean_policy_net_ev=("execution_net_ev_12h", "mean"),
            mean_peak_mfe_atr=("__peak_mfe_atr_12h__", "mean"),
        )
        .reset_index()
    )
    report_path = output / "support_by_month_side.csv"
    report.to_csv(report_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "materialized",
        "rows": int(rows),
        "identity": list(IDENTITY),
        "label_timing": {
            "signal_to_decision": "1h",
            "path": "[decision, decision+12h)",
            "label_available_at": "decision+12h",
        },
        "source_separation": {
            "physical_path_labels": (
                "unadjusted exact 1m decision-open OHLC path; no policy exit "
                "or spread adjustment"
            ),
            "policy_economics": (
                "separately signed candidate-local exit replay under "
                f"{source_economics}"
            ),
            "joined_multitask_labels": (
                "convenience exact-identity join; primary target remains direct "
                "execution_net_ev_12h and auxiliaries are representation tasks"
            ),
        },
        "multitask_targets": {
            "primary": "execution_net_ev_12h",
            "auxiliary": [
                "__opportunity_occurred_12h__",
                "__favorable_payoff_return_12h__",
                "execution_mae_return_12h",
                "__adverse_competing_risk_12h__",
                "__exit_conversion_loss_return_12h__",
                "__timeout_outcome_12h__",
                *TARGET_COLUMNS.values(),
            ],
            "opportunity_state_proxies_are_multilabel": True,
        },
        "auxiliary_kernel_schema": TARGET_SCHEMA,
        "supportive_label_columns": list(ALL_SUPPORTIVE_LABEL_COLUMNS),
        "soft_triple_barrier": {
            "upper": "max(1.5*ATR_fraction, 1.5%)",
            "lower": "1.0*ATR_fraction",
            "same_minute_conflict": "adverse_first_or_conflict",
        },
        "path_archetype": {
            "rule_version": PATH_ARCHETYPE_RULE_VERSION,
            "contract": (
                "execution-spread-adjusted 12h path, side retained; not "
                "bitwise comparable to the historical 24h v6 corpus"
            ),
        },
        "sources": {
            "exact_paths": {
                "path": str(args.paths.resolve()),
                "sha256": _sha256(args.paths),
            },
            "path_targets": {
                "path": str(args.path_targets.resolve()),
                "sha256": _sha256(args.path_targets),
            },
            "policy_labels": {
                "path": str(args.policy_labels.resolve()),
                "sha256": _sha256(args.policy_labels),
            },
            "candidate_coverage_manifest": {
                "path": str(args.coverage_manifest.resolve()),
                "sha256": _sha256(args.coverage_manifest),
            },
        },
        "outputs": {
            "physical_path_labels": {
                "path": str(physical_path.resolve()),
                "sha256": _sha256(physical_path),
                "rows": int(rows),
            },
            "joined_multitask_labels": {
                "path": str(joined_path.resolve()),
                "sha256": _sha256(joined_path),
                "rows": int(rows),
            },
            "support_by_month_side": {
                "path": str(report_path.resolve()),
                "sha256": _sha256(report_path),
                "rows": int(len(report)),
            },
        },
        "evidence_scope": source_evidence_scope,
        "lineage": source_lineage,
        "candidate_population_lineage": label_inputs_manifest.get(
            "candidate_population_lineage"
        ),
        "product_lineage": label_inputs_manifest.get("product_lineage"),
        "return_unit": label_inputs_manifest.get(
            "return_unit", "decimal_notional_return"
        ),
        "bootstrap_barrier_data_acquisition_only": label_inputs_manifest.get(
            "bootstrap_barrier_data_acquisition_only", False
        ),
        "parent_policy_binding": label_inputs_manifest.get(
            "parent_policy_binding"
        ),
        "oof_status": "not_oof",
        "economics": source_economics,
        "historical_l2_spread_available": source_historical_l2,
        "execution_parity_claim": False,
        "promotion_eligible": False,
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
