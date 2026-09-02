#!/usr/bin/env python3
"""Materialise Apr-2023--Dec-2024 exact-policy Wait10 training labels.

This is an all-candidate OOF training panel, never a reconstructed trading
book.  It joins the held-calendar-block base/residual stack to candidate-keyed
OOF regime-transition context and exact 720x1m policy paths.  Every shard must
reproduce its sealed enter-now control before Wait10 labels are accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_febapr_current_policy_wait10_action import (
    identity_digest,
    safe,
    sha256,
    simulate_path_batch,
    write_json,
)

STACK_ROOT = (
    ROOT
    / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4"
)
CONTEXT_ROOT = (
    ROOT
    / "data_perp/artifacts/reconstructed_2023apr_2024_candidate_oof_regime_transition_20260730_v1"
)
POLICY_PATH = (
    ROOT
    / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
    / "production_staging/best_policy_params.json"
)
OUT = (
    ROOT
    / "data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1"
)
SEGMENTS = (
    {
        "name": "2023apr_dec",
        "path_root": ROOT
        / "data_perp/artifacts/failure_2022_2023_pf_exact1m_paths_20260730_v1",
        "target_root": ROOT
        / "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2",
        "label_root": ROOT
        / "data_perp/artifacts/failure_2022_2023_pf_exact1m_policy_labels_20260730_v1",
    },
    {
        "name": "2024",
        "path_root": ROOT
        / "data_perp/artifacts/failure_2024_exact1m_paths_20260730_v2",
        "target_root": ROOT
        / "data_perp/artifacts/failure_2024_exact1m_label_inputs_20260730_v2",
        "label_root": ROOT
        / "data_perp/artifacts/failure_2024_exact1m_policy_labels_20260730_v2",
    },
)

SCHEMA = "historical_oof_2023apr_2024_current_policy_wait10_action_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SCORE_FEATURES = (
    "score_base_alpha",
    "score_residual_alpha",
    "score_base_expected_ev",
    "score_residual_expected_ev",
    "score_residual_delta_alpha",
)
REGIME_FEATURES = (
    "regime_state_ood_score",
    "regime_state_entropy",
    "regime_state_margin",
    "regime_state_uncertainty",
)
TRANSITION_FEATURES = (
    "transition_active_probability",
    "transition_state_p__stable",
    "transition_state_p__approach",
    "transition_state_p__immediate_lead",
    "transition_state_p__transition",
    "transition_state_p__acceleration",
    "transition_state_p__early_destination",
    "transition_state_p__settled_destination",
    "transition_state_ood_score",
    "transition_state_entropy",
    "transition_state_margin",
    "transition_state_uncertainty",
)
MODEL_FEATURES = (*SCORE_FEATURES, *REGIME_FEATURES, *TRANSITION_FEATURES)
PROVENANCE_FIELDS = (
    "stack_lineage",
    "residual_fold",
    "residual_is_oof",
    "regime_source_utc",
    "regime_fold_id",
    "regime_train_end_utc",
    "regime_available_utc",
    "transition_source_utc",
    "transition_fold_id",
    "transition_train_end_utc",
    "transition_available_utc",
)


class HistoricalContractError(RuntimeError):
    pass


def raw_policy_archetype(values: pd.Series) -> pd.Series:
    """Undo the persisted normalisation before the canonical resolver reruns it."""

    text = values.astype(str)
    prefix = "policy_archetype_"
    return text.where(
        ~text.str.startswith(prefix),
        text.str.slice(start=len(prefix)),
    )


def completed_batch(
    data_path: Path, manifest_path: Path, source_hash: str
) -> bool:
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(
        manifest.get("schema") == SCHEMA
        and manifest.get("source_identity_sha256") == source_hash
        and manifest.get("output_sha256") == sha256(data_path)
    )


def verify_sources(args: argparse.Namespace) -> dict[str, str]:
    provenance: dict[str, str] = {}
    stack_manifest = json.loads((args.stack_root / "manifest.json").read_text())
    if (
        stack_manifest.get("schema")
        != "historical_base_residual_stack_calendar_block_oof_v1"
        or sha256(args.stack_root / "oof_scores.parquet")
        != stack_manifest.get("outputs_sha256", {}).get("oof_scores.parquet")
    ):
        raise HistoricalContractError("historical OOF stack does not verify")
    provenance["stack_manifest_sha256"] = sha256(args.stack_root / "manifest.json")
    provenance["stack_sha256"] = sha256(args.stack_root / "oof_scores.parquet")

    context_manifest = json.loads((args.context_root / "manifest.json").read_text())
    if (
        context_manifest.get("schema") != "candidate_oof_regime_transition_adapter_v1"
        or sha256(args.context_root / "candidate_oof_regime_transition.parquet")
        != context_manifest.get("outputs", {}).get(
            "candidate_oof_regime_transition.parquet"
        )
    ):
        raise HistoricalContractError("candidate OOF regime-transition context does not verify")
    provenance["context_manifest_sha256"] = sha256(
        args.context_root / "manifest.json"
    )
    provenance["context_sha256"] = sha256(
        args.context_root / "candidate_oof_regime_transition.parquet"
    )

    for segment in args.segments:
        name = str(segment["name"])
        path_root = Path(segment["path_root"])
        target_root = Path(segment["target_root"])
        label_root = Path(segment["label_root"])
        path_manifest = json.loads((path_root / "manifest.json").read_text())
        target_manifest = json.loads((target_root / "manifest.json").read_text())
        label_manifest = json.loads((label_root / "manifest.json").read_text())
        if path_manifest.get("schema") != "execution_entry_timing_1m_paths_v1":
            raise HistoricalContractError(f"{name} path schema mismatch")
        target_record = target_manifest.get("outputs", {}).get("path_targets", {})
        if (
            target_manifest.get("schema") != "historical_backcast_exact1m_label_inputs_v1"
            or sha256(target_root / "path_targets.parquet")
            != target_record.get("sha256")
        ):
            raise HistoricalContractError(f"{name} path-target hash mismatch")
        label_record = label_manifest.get("output", {})
        if (
            label_manifest.get("schema") != "execution_ev_deployed_policy_1m_labels_v1"
            or sha256(label_root / "execution_policy_labels.parquet")
            != label_record.get("sha256")
        ):
            raise HistoricalContractError(f"{name} current-policy label hash mismatch")
        provenance[f"{name}_path_manifest_sha256"] = sha256(
            path_root / "manifest.json"
        )
        provenance[f"{name}_paths_sha256"] = sha256(path_root / "paths.parquet")
        provenance[f"{name}_target_manifest_sha256"] = sha256(
            target_root / "manifest.json"
        )
        provenance[f"{name}_targets_sha256"] = target_record["sha256"]
        provenance[f"{name}_label_manifest_sha256"] = sha256(
            label_root / "manifest.json"
        )
        provenance[f"{name}_labels_sha256"] = label_record["sha256"]
    return provenance


def merge_candidate_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
    *,
    source: str,
) -> pd.DataFrame:
    selected = right.loc[:, ["candidate_id", *columns]].copy()
    if selected["candidate_id"].duplicated().any():
        raise HistoricalContractError(f"{source} candidate IDs are duplicated")
    result = left.merge(selected, on="candidate_id", how="left", validate="one_to_one")
    if result.loc[:, list(columns)].isna().all(axis=1).any():
        raise HistoricalContractError(f"{source} coverage is incomplete")
    return result


def load_contract(args: argparse.Namespace) -> pd.DataFrame:
    context = pd.read_parquet(
        args.context_root / "candidate_oof_regime_transition.parquet",
        columns=[*IDENTITY, *REGIME_FEATURES, *TRANSITION_FEATURES, *PROVENANCE_FIELDS[3:]],
    )
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True)
    context["side_name"] = context["side_name"].astype(str).str.lower()
    if (
        len(context) != 293_828
        or context.duplicated(list(IDENTITY), keep=False).any()
    ):
        raise HistoricalContractError("unexpected candidate OOF context population")

    stack = pd.read_parquet(
        args.stack_root / "oof_scores.parquet",
        columns=[
            *IDENTITY,
            *SCORE_FEATURES,
            *PROVENANCE_FIELDS[:3],
        ],
    )
    result = merge_candidate_columns(
        context,
        stack,
        [*SCORE_FEATURES, *PROVENANCE_FIELDS[:3]],
        source="historical OOF stack",
    )
    if not result["residual_is_oof"].eq(True).all():
        raise HistoricalContractError("historical residual scores are not all OOF")

    targets: list[pd.DataFrame] = []
    labels: list[pd.DataFrame] = []
    label_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "policy_archetype",
        "execution_geometry_key",
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
    ]
    for segment in args.segments:
        targets.append(
            pd.read_parquet(
                Path(segment["target_root"]) / "path_targets.parquet",
                columns=[
                    *IDENTITY,
                    "__barrier_pct__",
                    "__path_auxiliary_atr_fraction__",
                ],
            )
        )
        labels.append(
            pd.read_parquet(
                Path(segment["label_root"]) / "execution_policy_labels.parquet",
                columns=label_columns,
            )
        )
    target = pd.concat(targets, ignore_index=True)
    label = pd.concat(labels, ignore_index=True)
    result = merge_candidate_columns(
        result,
        target,
        ["__barrier_pct__", "__path_auxiliary_atr_fraction__"],
        source="exact policy targets",
    )
    result = merge_candidate_columns(
        result,
        label,
        [name for name in label_columns if name not in IDENTITY],
        source="current-policy controls",
    )
    result["__decision_ts__"] = pd.to_datetime(
        result["execution_decision_utc"], utc=True
    )
    result["execution_label_end_utc"] = pd.to_datetime(
        result["execution_label_end_utc"], utc=True
    )
    result["execution_label_available_at"] = pd.to_datetime(
        result["execution_label_available_at"], utc=True
    )
    # Historical controls persist the resolver's normalized archetype label.
    # `_resolved_geometry` expects the raw source token and normalizes it once.
    # Passing the persisted value unchanged would add a second
    # `policy_archetype_` prefix and silently fall back to side-parent geometry.
    result["policy_archetype"] = raw_policy_archetype(
        result["policy_archetype"]
    )
    for field in (
        "regime_source_utc",
        "regime_train_end_utc",
        "regime_available_utc",
        "transition_source_utc",
        "transition_train_end_utc",
        "transition_available_utc",
    ):
        result[field] = pd.to_datetime(result[field], utc=True)
    if not result["__decision_ts__"].eq(
        result["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise HistoricalContractError("decision time is not signal plus one hour")
    if not result["execution_label_end_utc"].eq(
        result["__decision_ts__"] + pd.Timedelta(hours=12)
    ).all():
        raise HistoricalContractError("control labels do not retain the exact 12h deadline")
    for prefix in ("regime", "transition"):
        if not result[f"{prefix}_available_utc"].le(result["__decision_ts__"]).all():
            raise HistoricalContractError(f"{prefix} context is not available by decision")
        if not result[f"{prefix}_train_end_utc"].lt(result["__decision_ts__"]).all():
            raise HistoricalContractError(f"{prefix} training reaches the decision")
    if not np.isfinite(
        result.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=float)
    ).all():
        raise HistoricalContractError("OOF action features are non-finite")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.with_name(f".{args.output.name}.work")
    work.mkdir(parents=True, exist_ok=True)
    (work / "batches").mkdir(exist_ok=True)
    provenance = verify_sources(args)
    contract = load_contract(args)
    contract_by_id = contract.set_index("candidate_id", drop=False)
    candidate_ids = set(contract_by_id.index.astype(str))
    policy = json.loads(args.policy_path.read_text())
    batch_files: list[Path] = []
    parity_files: list[Path] = []

    for segment in args.segments:
        segment_name = str(segment["name"])
        path_file = pq.ParquetFile(Path(segment["path_root"]) / "paths.parquet")
        for row_group in range(path_file.num_row_groups):
            path_rows = path_file.read_row_group(row_group).to_pandas()
            path_rows = path_rows.loc[
                path_rows["candidate_id"].astype(str).isin(candidate_ids)
            ].reset_index(drop=True)
            if path_rows.empty:
                continue
            source_hash = hashlib_identity(path_rows)
            stem = f"{segment_name}_{row_group:04d}"
            data_path = work / "batches" / f"{stem}.parquet"
            parity_path = work / "batches" / f"{stem}.parity.csv"
            manifest_path = work / "batches" / f"{stem}.manifest.json"
            if not completed_batch(data_path, manifest_path, source_hash):
                ids = path_rows["candidate_id"].astype(str)
                local_contract = contract_by_id.loc[ids].reset_index(drop=True)
                labels, parity = simulate_path_batch(path_rows, local_contract, policy)
                temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
                labels.to_parquet(temporary, index=False, compression="zstd")
                os.replace(temporary, data_path)
                parity.to_csv(parity_path, index=False)
                write_json(
                    manifest_path,
                    {
                        "schema": SCHEMA,
                        "segment": segment_name,
                        "row_group": row_group,
                        "rows": int(len(labels)),
                        "source_identity_sha256": source_hash,
                        "output_sha256": sha256(data_path),
                        "parity_sha256": sha256(parity_path),
                    },
                )
            batch_files.append(data_path)
            parity_files.append(parity_path)

    labels = pd.concat(
        [pd.read_parquet(path) for path in batch_files], ignore_index=True
    ).sort_values(["execution_decision_utc", "candidate_id"], kind="stable")
    features = contract.loc[:, [*IDENTITY, *MODEL_FEATURES]].copy()
    provenance_frame = contract.loc[
        :, [*IDENTITY, *PROVENANCE_FIELDS]
    ].copy()
    if (
        len(labels) != 293_828
        or labels.duplicated(list(IDENTITY), keep=False).any()
        or identity_digest(labels) != identity_digest(contract)
    ):
        raise HistoricalContractError("final historical Wait10 identity coverage changed")
    parity = pd.concat([pd.read_csv(path) for path in parity_files], ignore_index=True)
    parity_summary = (
        parity.groupby("field", sort=True)
        .agg(
            rows=("rows", "sum"),
            mismatch_rows=("mismatch_rows", "sum"),
            max_abs_delta=("max_abs_delta", "max"),
        )
        .reset_index()
    )
    if parity_summary["mismatch_rows"].sum() != 0:
        raise HistoricalContractError("full historical current-policy parity failed")

    final = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        labels.to_parquet(final / "action_labels.parquet", index=False, compression="zstd")
        features.to_parquet(final / "preentry_features.parquet", index=False, compression="zstd")
        provenance_frame.to_parquet(
            final / "oof_provenance.parquet", index=False, compression="zstd"
        )
        parity_summary.to_csv(final / "control_parity.csv", index=False)
        roles = {
            "schema": SCHEMA,
            "model_inputs": list(MODEL_FEATURES),
            "score_inputs": list(SCORE_FEATURES),
            "regime_inputs": list(REGIME_FEATURES),
            "transition_inputs": list(TRANSITION_FEATURES),
            "provenance_only": list(PROVENANCE_FIELDS),
            "target_only": [
                "enter_now_gross",
                "enter_now_cost",
                "enter_now_net",
                "wait10_gross",
                "wait10_cost",
                "wait10_net",
                "wait_delta",
                "wait_better",
                "execution_label_end_utc",
            ],
            "explicitly_excluded": [
                "__reconstructed_soft_alpha_12h__",
                "execution_future_path",
                "__barrier_pct__",
                "__path_auxiliary_atr_fraction__",
                "policy_archetype",
                "regime_state_id",
                "transition_state_id",
                "regime_state_p__0",
                "regime_state_p__1",
                "regime_state_p__2",
                "calendar_month",
            ],
        }
        write_json(final / "feature_roles.json", roles)
        outputs = {
            name: sha256(final / name)
            for name in (
                "action_labels.parquet",
                "preentry_features.parquet",
                "oof_provenance.parquet",
                "control_parity.csv",
                "feature_roles.json",
            )
        }
        by_month_side = (
            labels.groupby(["candidate_month", "side_name"], sort=True)
            .agg(
                rows=("candidate_id", "size"),
                wait_better_rate=("wait_better", "mean"),
                mean_wait_delta=("wait_delta", "mean"),
            )
            .reset_index()
            .to_dict("records")
        )
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_HISTORICAL_OOF_TRAINING_LEDGER_NO_BOOK_RECONSTRUCTION",
            "rows": int(len(labels)),
            "identity_sha256": identity_digest(labels),
            "rows_by_month_side": by_month_side,
            "feature_count": len(MODEL_FEATURES),
            "contract": {
                "population": "exact candidate-keyed Apr-2023--Dec-2024 OOF regime-transition intersection; no global-book weights are created",
                "scores": "held-calendar-block OOF residual stack; historical frozen base remains diagnostic lineage",
                "features": "candidate-keyed OOF regime/transition probabilities and summaries available before decision; component-unstable regime p0/p1/p2 excluded",
                "action": "current-policy enter-now versus Wait10 with exact original deadline, barrier, side-archetype geometry and once-only costs",
                "archetype_normalization": "persisted policy_archetype_ prefix is removed before the canonical resolver adds it once; exact geometry-key and outcome parity are mandatory",
                "parity": "every enter-now row must exactly reproduce the sealed historical current-policy control",
                "use": "all-candidate action learning and transfer diagnosis only; nonpromotable without untouched frozen-book evaluation",
            },
            "input_provenance": {
                **provenance,
                "policy_sha256": sha256(args.policy_path),
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(final / "manifest.json", manifest)
        (final / "manifest.sha256").write_text(
            f"{sha256(final / 'manifest.json')}  manifest.json\n"
        )
        os.replace(final, args.output)
    except Exception:
        shutil.rmtree(final, ignore_errors=True)
        raise
    shutil.rmtree(work)
    return manifest


def hashlib_identity(frame: pd.DataFrame) -> str:
    normalized = frame.loc[:, list(IDENTITY)].copy()
    normalized["__ts__"] = pd.to_datetime(normalized["__ts__"], utc=True)
    normalized["side_name"] = normalized["side_name"].astype(str).str.lower()
    return identity_digest(normalized)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stack-root", type=Path, default=STACK_ROOT)
    result.add_argument("--context-root", type=Path, default=CONTEXT_ROOT)
    result.add_argument("--policy-path", type=Path, default=POLICY_PATH)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    args.segments = SEGMENTS
    print(json.dumps(safe(run(args)), indent=2))


if __name__ == "__main__":
    main()
