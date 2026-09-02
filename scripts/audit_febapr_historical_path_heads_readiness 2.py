#!/usr/bin/env python3
"""Audit Feb--Apr 2025 readiness for auxiliary heads and CatBoost path labels.

This is deliberately a data/provenance gate: it neither fits a model nor writes
training inputs.  It separates (a) labels that are already causally usable for
the five auxiliary heads from (b) the additional cost/geometry target and
side-local selection/HPO evidence required for a historical CatBoost head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meaningful_mfe_event_ablation import (
    atr_soft_triple_barrier_labels,
    competing_risk_targets,
)
from extreme_price_movements.path_auxiliary_model_families import (
    HEAD_SPECS,
    build_role_targets,
    validate_canonical_auxiliary_labels,
)

DEFAULT_RESIDUAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet"
DEFAULT_LABEL_DIR = ROOT / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels"
DEFAULT_POLICY_INPUTS = ROOT / "data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1/path_targets.parquet"
DEFAULT_RESIDUAL_SHARDS = ROOT / "data_perp/artifacts/febapr2025_residual_shards_20260727_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_historical_path_heads_readiness_20260727_v2"

IDENTITY = ("candidate_id", "side_name", "__ts__")
AUX_COLUMNS = (
    "candidate_id",
    "side_name",
    "__ts__",
    "__label_end_ts__",
    "__path_auxiliary_atr_fraction__",
    "__peak_mfe_atr_12h__",
    "__peak_mfe_atr_clip_8__",
    "__time_to_first_meaningful_mfe_hours_12h__",
    "__mae_before_meaningful_mfe_atr_12h__",
    "__bars_before_price_stops_decreasing_12h__",
    "__bars_to_confirmed_adverse_trough__",
    "__future_slope_atr_per_hour_12h__",
    "__meaningful_mfe_reached_12h__",
    "__path_auxiliary_target_valid__",
    "__time_to_first_meaningful_mfe_target_valid__",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _utc(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    return result


def _label_path(label_dir: Path, side: str) -> Path:
    return label_dir / f"train_global_{side}_3.parquet"


def _role_support(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate fixed role semantics and report side-local usable support."""

    validate_canonical_auxiliary_labels(frame)
    targets = build_role_targets(frame)
    report: dict[str, Any] = {}
    for name, target in targets.items():
        values = target.target[target.train_mask]
        item: dict[str, Any] = {
            "source_column": target.source_column,
            "valid_rows": int(target.valid_mask.sum()),
            "train_rows": int(target.train_mask.sum()),
            "task": target.role.task,
            "target_condition": target.role.target_condition,
        }
        if target.role.task == "binary":
            item["positive_rows"] = int((values > 0.5).sum())
            item["positive_fraction"] = float(np.mean(values > 0.5)) if len(values) else None
        elif len(values):
            item["minimum"] = float(np.min(values))
            item["maximum"] = float(np.max(values))
            item["standard_deviation"] = float(np.std(values))
        report[name] = item
    return report


def _residual_feature_contracts(shard_root: Path) -> dict[str, Any]:
    """Read the residual predictor provenance without treating it as head HPO."""

    result: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for path in sorted(shard_root.glob("*/coverage_economics_gate.json")):
        gate = json.loads(path.read_text())
        for fold in gate.get("folds", []):
            side = str(fold["side"]).lower()
            result[side].append(
                {
                    "fold": fold["fold"],
                    "feature_count": len(fold.get("features", [])),
                    "features_sha256": hashlib.sha256(
                        json.dumps(fold.get("features", []), separators=(",", ":")).encode()
                    ).hexdigest(),
                    "hpo_sha256": fold.get("hpo_sha256"),
                    "train_resolution_max": fold.get("train_resolution_max"),
                    "purge": fold.get("purge"),
                }
            )
    return result


def _expected_auxiliary_contract() -> dict[str, Any]:
    return {
        head.name: {
            "deployment_status": head.deployment_status,
            "description": head.description,
            "promotion_requirement": head.promotion_requirement,
            "roles": [
                {
                    "name": role.name,
                    "task": role.task,
                    "target_columns": list(role.target_columns),
                    "target_condition": role.target_condition,
                    "quantile": role.quantile,
                    "deployment_status": role.deployment_status,
                }
                for role in head.roles
            ],
        }
        for head in HEAD_SPECS
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    residual = _utc(pd.read_parquet(args.residual), ("__ts__", "__decision_ts__", "execution_label_end_utc", "native_label_resolution_utc"))
    strict = residual.loc[residual["residual_is_oof"].astype(bool)].copy()
    warmup = residual.loc[~residual["residual_is_oof"].astype(bool)].copy()
    if len(strict) != 140_682 or len(warmup) != 64_512:
        raise ValueError("unexpected strict/warm-up population size")
    if strict.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("strict residual OOF identity is not unique")
    if set(strict["side_name"].unique()) != {"long", "short"}:
        raise ValueError("strict residual population must include both canonical sides")

    sources: list[pd.DataFrame] = []
    for side in ("long", "short"):
        path = _label_path(args.label_dir, side)
        source = _utc(pd.read_parquet(path, columns=list(AUX_COLUMNS)), ("__ts__", "__label_end_ts__"))
        if source.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"auxiliary label identity is duplicated for {side}")
        if not source["side_name"].eq(side).all():
            raise ValueError(f"auxiliary source contains wrong side rows for {side}")
        sources.append(source)
    source = pd.concat(sources, ignore_index=True)
    merged = strict.loc[:, [*IDENTITY, "__decision_ts__", "execution_label_end_utc", "native_label_resolution_utc"]].merge(
        source,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    join_ok = bool(merged["_merge"].eq("both").all())
    if not join_ok:
        raise ValueError("strict residual rows are missing canonical auxiliary labels")
    merged = merged.drop(columns="_merge")
    auxiliary_resolution_ok = bool(
        merged["__label_end_ts__"].eq(merged["__decision_ts__"] + pd.Timedelta(hours=12)).all()
        and merged["execution_label_end_utc"].eq(merged["__label_end_ts__"]).all()
    )
    native_resolution_is_later = bool(
        merged["native_label_resolution_utc"].eq(merged["__decision_ts__"] + pd.Timedelta(hours=24)).all()
    )

    side_reports: dict[str, Any] = {}
    for side in ("long", "short"):
        side_frame = merged.loc[merged["side_name"].eq(side)].copy()
        support = _role_support(side_frame)
        triple = atr_soft_triple_barrier_labels(side_frame)
        risk = competing_risk_targets(pd.concat([side_frame, triple], axis=1))
        side_reports[side] = {
            "strict_rows": int(len(side_frame)),
            "auxiliary_label_end_matches_decision_plus_12h": bool(
                side_frame["__label_end_ts__"].eq(side_frame["__decision_ts__"] + pd.Timedelta(hours=12)).all()
            ),
            "valid_auxiliary_path_rows": int(side_frame["__path_auxiliary_target_valid__"].eq(1).sum()),
            "role_support": support,
            "meaningful_mfe_event_classifier": {
                "target": "__meaningful_mfe_reached_12h__",
                "valid_rows": int(side_frame["__path_auxiliary_target_valid__"].eq(1).sum()),
                "positive_rows": int(((side_frame["__meaningful_mfe_reached_12h__"] == 1) & (side_frame["__path_auxiliary_target_valid__"] == 1)).sum()),
                "requires": ["side-local features", "side-local feature selection", "side-local HPO", "strict 12h label-end purge"],
            },
            "soft_triple_barrier_competing_risk": {
                "upper": "max(1.5 ATR, 1.5% return)",
                "lower": "1.0 ATR",
                "timeout_hours": 12,
                "same_hour_conflict": "adverse first conservatively",
                "risk_valid_rows": int(risk["risk_valid"].sum()),
                "favorable_first_rows": int((risk["risk_class"] == 2).sum()),
                "adverse_first_or_conflict_rows": int((risk["risk_class"] == 1).sum()),
                "timeout_rows": int((risk["risk_class"] == 0).sum()),
                "order_ambiguous_rows": int(risk["order_ambiguous"].sum()),
                "required_for_competing_risk": [
                    "frozen side-local stop_atr_by_side", "side-local multiclass feature selection/HPO",
                    "purged 13h-or-greater inner folds", "do not replace explicit meaningful event with __mfe_ge_1_5atr__",
                ],
            },
        }

    # February is a warm-up source only: the *auxiliary* label end must
    # precede the first March decision boundary.  This is intentionally less
    # restrictive than the residual model's 24-hour native target: a head
    # trained only on the 12-hour path outcome may legally retain 31,632 rows
    # per side, whereas the residual's own target retained 31,056.
    march_start = pd.Timestamp("2025-03-01T00:00:00Z")
    warmup_join = warmup.loc[:, [*IDENTITY, "native_label_resolution_utc"]].merge(
        source.loc[:, [*IDENTITY, "__label_end_ts__"]], on=list(IDENTITY), how="left", validate="one_to_one"
    )
    warmup_report: dict[str, Any] = {}
    for side in ("long", "short"):
        rows = warmup_join.loc[warmup_join["side_name"].eq(side)]
        legal = rows["__label_end_ts__"].lt(march_start)
        native = rows["native_label_resolution_utc"].lt(march_start)
        warmup_report[side] = {
            "february_rows": int(len(rows)),
            "source_label_rows_joined": int(rows["__label_end_ts__"].notna().sum()),
            "auxiliary_rows_resolved_before_march": int(legal.sum()),
            "native_residual_rows_resolved_before_march": int(native.sum()),
            "legal_for_march_auxiliary_head_fit": bool(legal.sum() == 31_632),
            "not_equivalent_to_residual_target_purge": bool(int(legal.sum()) > int(native.sum())),
            "rule": "only rows whose 12h auxiliary label end is strictly before 2025-03-01T00:00:00Z may fit March heads",
        }

    # The exact historic policy input is a useful source-parity check. It
    # carries raw ATR and barrier only, not the five outcomes or a CatBoost
    # class, so successful parity does not make CatBoost trainable.
    policy = _utc(
        pd.read_parquet(args.policy_inputs, columns=["candidate_id", "side_name", "__ts__", "__path_auxiliary_atr_fraction__", "__barrier_pct__"]),
        ("__ts__",),
    )
    policy_join = merged.merge(policy, on=list(IDENTITY), how="left", suffixes=("", "__policy"), validate="one_to_one")
    atr_delta = np.abs(
        pd.to_numeric(policy_join["__path_auxiliary_atr_fraction__"], errors="coerce")
        - pd.to_numeric(policy_join["__path_auxiliary_atr_fraction____policy"], errors="coerce")
    )
    policy_parity = {
        "rows_joined": int(policy_join["__barrier_pct__"].notna().sum()),
        "all_strict_rows_joined": bool(policy_join["__barrier_pct__"].notna().all()),
        "max_abs_atr_delta": float(np.nanmax(atr_delta)),
        "atr_exact": bool(np.allclose(atr_delta, 0.0, rtol=0.0, atol=0.0, equal_nan=False)),
        "limitation": "policy input contains only ATR/barrier lineage, not five head outcomes or a cost-aware CatBoost archetype target",
    }
    residual_contracts = _residual_feature_contracts(args.residual_shards)
    existing_packb = ROOT / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8"
    existing_catboost = ROOT / "data_perp/artifacts/execution_ev_catboost_oof_20260725_v1/catboost_oof.parquet"
    target_mismatch = [
        "No mismatch found for the five v6 auxiliary target columns: exact historical source joins all strict rows and its 12h label end matches the residual execution label end.",
        "The source labels cannot substitute for a CatBoost target: they contain no cost-aware path_archetype/path_shape_archetype, no deployed policy geometry sweep result, and no historical CatBoost class-support gate.",
        "The existing auxiliary bundles and CatBoost OOF artifacts are 2026 May-July / different immutable identity populations; their feature selection, HPO and geometry provenance cannot be reused as historical March-Apr evidence.",
    ]
    auxiliary_data_ready = bool(join_ok and auxiliary_resolution_ok and native_resolution_is_later and all(item["legal_for_march_auxiliary_head_fit"] for item in warmup_report.values()))
    gate = {
        "schema": "febapr2025_historical_path_heads_readiness_gate_v1",
        "supersedes": "febapr2025_historical_path_heads_readiness_20260727_v1 (which incorrectly applied the residual's 24h target-purge count to 12h auxiliary labels)",
        "scope": "140682 strict March-April 2025 residual OOF rows, evaluated per side; no model training performed",
        "residual_population": {
            "path": str(args.residual), "sha256": _sha256(args.residual), "strict_oof_rows": int(len(strict)), "february_warmup_rows": int(len(warmup)),
            "rows_by_side": strict["side_name"].value_counts().sort_index().astype(int).to_dict(),
        },
        "auxiliary_target_source": {
            "directory": str(args.label_dir), "schema": "path_auxiliary_targets_v6_supportive_future_paths_12h", "label_end": "decision + 12h", "target_definitions": _expected_auxiliary_contract(),
            "all_strict_rows_exactly_joined": join_ok, "label_resolution_matches_execution_label_end": auxiliary_resolution_ok,
            "residual_native_label_resolution_is_decision_plus_24h": native_resolution_is_later,
            "per_side": side_reports,
        },
        "february_warmup_for_march": warmup_report,
        "deployed_policy_atr_lineage": policy_parity,
        "residual_feature_provenance_only": {
            "per_side_fold": residual_contracts,
            "status": "residual-model-only; it is not a feature-selection/HPO/geometry contract for CatBoost or any auxiliary role",
        },
        "historical_head_model_provenance": {
            "five_auxiliary_head_feature_selection_hpo_materialized": False,
            "catboost_costaware_archetype_labels_materialized": False,
            "catboost_side_local_feature_selection_hpo_geometry_materialized": False,
            "existing_nonhistorical_auxiliary_bundle": str(existing_packb),
            "existing_nonhistorical_catboost_oof": str(existing_catboost),
            "required_before_training": [
                "materialize frozen historical pre-entry feature context on exact candidate_id identity", "run and record feature selection and HPO separately for every auxiliary role and side", "materialize cost-aware historical path archetype labels using the frozen deployed geometry and label-end purge", "run CatBoost feature selection, class-support/balance, HPO and geometry sweep separately per side", "retain the meaningful-MFE classifier and its competing-risk variant as separate side-local families"],
        },
        "target_mismatch_flags": target_mismatch,
        "auxiliary_label_data_ready": auxiliary_data_ready,
        "five_auxiliary_heads_train_ready": False,
        "catboost_train_ready": False,
        "overall_status": "BLOCKED_PENDING_HISTORICAL_PREENTRY_FEATURE_AND_CATBOOST_GEOMETRY_MATERIALIZATION",
    }
    args.output_dir.mkdir(parents=True)
    gate_path = args.output_dir / "readiness_gate.json"
    support_path = args.output_dir / "auxiliary_role_support_by_side.json"
    _write_json(gate_path, gate)
    _write_json(support_path, side_reports)
    return {"gate": gate_path, "role_support": support_path}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--policy-inputs", type=Path, default=DEFAULT_POLICY_INPUTS)
    parser.add_argument("--residual-shards", type=Path, default=DEFAULT_RESIDUAL_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    print(json.dumps({key: str(value) for key, value in run(parser().parse_args()).items()}, indent=2))
