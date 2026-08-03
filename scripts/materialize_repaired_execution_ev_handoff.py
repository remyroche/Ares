#!/usr/bin/env python3
"""Add repaired OOF event/risk/magnitude features to the exact EV handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
ALPHA_CANDIDATE_CONTEXT_COLUMNS = (
    "base_oof_score",
    "base_candidate_rank_pct_timestamp_side",
    "base_candidate_group_rows",
    "base_cutoff_score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_score_z_timestamp_side",
    "base_signal_zscore_within_archetype",
    "base_rank_decile",
)
ALPHA_CANDIDATE_RAW_SCORE_COLUMNS = frozenset({"base_oof_score"})
ALPHA_CANDIDATE_CUTOFF_CONTEXT_COLUMNS = frozenset(
    {"base_margin_to_cutoff", "base_margin_to_cutoff_z"}
)
ALPHA_CANDIDATE_TIMESTAMP_RELATIVE_COLUMNS = frozenset(
    {"base_candidate_rank_pct_timestamp_side", "base_score_z_timestamp_side"}
)
ALPHA_CANDIDATE_ARCHETYPE_Z_COLUMNS = frozenset(
    {"base_signal_zscore_within_archetype"}
)
ALPHA_CANDIDATE_DECILE_GROUP_COLUMNS = frozenset(
    {"base_rank_decile", "base_candidate_group_rows"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_unique(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    if frame.loc[:, ID_COLUMNS].isna().any().any():
        raise ValueError(f"{path} contains null identities")
    if frame.duplicated(list(ID_COLUMNS)).any():
        raise ValueError(f"{path} violates one-to-one identity")
    return frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--literal-oof", type=Path, required=True)
    parser.add_argument("--clean-oof", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--execution-labels", type=Path)
    parser.add_argument("--execution-label-manifest", type=Path)
    parser.add_argument("--representation-context", type=Path)
    parser.add_argument("--representation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joined = pd.read_parquet(args.joined)
    label_manifest = None
    if (args.execution_labels is None) != (args.execution_label_manifest is None):
        raise ValueError(
            "--execution-labels and --execution-label-manifest must be supplied together"
        )
    if args.execution_labels is not None:
        replacement_columns = [
            "execution_decision_utc",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
            "execution_label_end_utc",
            "execution_exit_reason",
            "execution_exit_hour",
            "execution_mfe_return_12h",
            "execution_mae_return_12h",
            "execution_label_available_at",
        ]
        replacement = _read_unique(
            args.execution_labels,
            [*ID_COLUMNS, *replacement_columns],
        ).rename(
            columns={
                "execution_label_available_at": "execution_labels_available_at"
            }
        )
        drop_columns = [
            *replacement_columns[:-1],
            "execution_labels_available_at",
            "execution_labels_label_resolution_available_at",
        ]
        joined = joined.drop(
            columns=[name for name in drop_columns if name in joined.columns]
        ).merge(replacement, on=list(ID_COLUMNS), how="inner", validate="one_to_one")
        joined["execution_labels_label_resolution_available_at"] = joined[
            "execution_labels_available_at"
        ]
        label_manifest = json.loads(
            args.execution_label_manifest.read_text(encoding="utf-8")
        )
    representation_columns: list[str] = []
    representation_manifest = None
    if (args.representation_context is None) != (
        args.representation_manifest is None
    ):
        raise ValueError(
            "--representation-context and --representation-manifest must be supplied together"
        )
    if args.representation_context is not None:
        representation_manifest = json.loads(
            args.representation_manifest.read_text(encoding="utf-8")
        )
        representation_columns = [
            "gmm_representation_available",
            *representation_manifest["representation"]["generated_features"],
        ]
        context_columns = [
            name
            for name in ALPHA_CANDIDATE_CONTEXT_COLUMNS
            if name not in representation_columns
        ]
        representation = _read_unique(
            args.representation_context,
            [*ID_COLUMNS, *context_columns, *representation_columns],
        )
        representation.loc[:, context_columns] = representation.loc[
            :, context_columns
        ].apply(pd.to_numeric, errors="coerce")
        if representation.loc[:, context_columns].isna().any().any():
            missing = representation.loc[:, context_columns].isna().sum()
            raise ValueError(f"alpha candidate context contains nulls:\n{missing}")
        representation["gmm_representation_available"] = pd.to_numeric(
            representation["gmm_representation_available"], errors="coerce"
        ).fillna(0.0)
        generated = [
            name
            for name in representation_columns
            if name != "gmm_representation_available"
        ]
        representation.loc[:, generated] = (
            representation.loc[:, generated]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        joined = joined.merge(
            representation,
            on=list(ID_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        for suffix in ("dae", "gmm_posterior", "gmm_geometry", "risk_summary"):
            alias = f"representation_available__{suffix}"
            joined[alias] = joined["gmm_representation_available"].astype("float64")
            representation_columns.append(alias)
        representation_columns.extend(context_columns)
    literal = _read_unique(
        args.literal_oof,
        [*ID_COLUMNS, "config_base_residual_raw"],
    ).rename(columns={"config_base_residual_raw": "oof_literal_reach_probability"})
    clean = _read_unique(
        args.clean_oof,
        [
            *ID_COLUMNS,
            "catboost_hard_ensemble_platt",
            "catboost_competing_p_favorable",
            "catboost_competing_net_probability",
            "catboost_conditional_quality",
        ],
    ).rename(
        columns={
            "catboost_hard_ensemble_platt": "oof_clean_favorable_probability",
            "catboost_competing_p_favorable": "oof_competing_favorable_probability",
            "catboost_conditional_quality": "oof_conditional_path_quality",
        }
    )
    clean["oof_competing_adverse_probability"] = np.clip(
        1.0
        + clean["oof_competing_favorable_probability"]
        - 2.0 * clean["catboost_competing_net_probability"],
        0.0,
        1.0,
    )
    clean["oof_competing_timeout_probability"] = np.clip(
        1.0
        - clean["oof_competing_favorable_probability"]
        - clean["oof_competing_adverse_probability"],
        0.0,
        1.0,
    )
    clean = clean.drop(columns=["catboost_competing_net_probability"])
    targets = _read_unique(
        args.targets,
        [*ID_COLUMNS, "__path_auxiliary_atr_fraction__"],
    ).rename(columns={"__path_auxiliary_atr_fraction__": "oof_entry_atr_fraction"})

    original_rows = len(joined)
    for supplement in (literal, clean, targets):
        joined = joined.merge(
            supplement,
            on=list(ID_COLUMNS),
            how="left",
            validate="one_to_one",
        )
    if len(joined) != original_rows:
        raise AssertionError("supplement merge changed the exact handoff row count")
    added = [
        "oof_literal_reach_probability",
        "oof_clean_favorable_probability",
        "oof_competing_favorable_probability",
        "oof_competing_adverse_probability",
        "oof_competing_timeout_probability",
        "oof_conditional_path_quality",
        "oof_entry_atr_fraction",
    ]
    if joined[added].isna().any().any():
        missing = joined[added].isna().sum()
        raise ValueError(f"repaired OOF inputs do not cover the exact handoff:\n{missing}")
    for column in added:
        joined[column] = pd.to_numeric(joined[column], errors="raise").astype("float64")
    for column in added[:-1]:
        if not joined[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} is not a probability/quality in [0,1]")
    if (joined["oof_entry_atr_fraction"] <= 0.0).any():
        raise ValueError("entry ATR fraction must be positive")

    peak = joined["pred_peak_MFE_12h_ATR"].clip(lower=0.0)
    atr = joined["oof_entry_atr_fraction"]
    literal_p = joined["oof_literal_reach_probability"]
    clean_p = joined["oof_clean_favorable_probability"]
    quality = joined["oof_conditional_path_quality"]
    adverse_p = joined["oof_competing_adverse_probability"]
    joined["oof_literal_expected_peak_mfe_atr"] = literal_p * peak
    joined["oof_clean_expected_peak_mfe_atr"] = clean_p * peak
    joined["oof_literal_expected_peak_return"] = literal_p * peak * atr
    joined["oof_clean_expected_peak_return"] = clean_p * peak * atr
    joined["oof_quality_adjusted_expected_peak_return"] = (
        literal_p * quality * peak * atr
    )
    joined["oof_competing_risk_adjusted_peak_return"] = (
        literal_p * (1.0 - adverse_p) * peak * atr
    )
    joined["oof_literal_peak_net_cost_margin"] = (
        joined["oof_literal_expected_peak_return"] - 0.01
    )
    joined["oof_clean_peak_net_cost_margin"] = (
        joined["oof_clean_expected_peak_return"] - 0.01
    )

    payload = json.loads(args.provenance.read_text(encoding="utf-8"))
    available_at_col = "execution_decision_utc"
    source_base = (
        "strict side-local outer-OOF repaired meaningful-MFE ablation; exact identity merge"
    )
    declarations = {
        "oof_literal_reach_probability": (
            "literal_reach_probability",
            source_base + "; base-to-residual literal reach classifier",
        ),
        "oof_clean_favorable_probability": (
            "clean_event_probability",
            source_base + "; clean favorable-before-adverse classifier",
        ),
        "oof_competing_favorable_probability": (
            "competing_risk_probability",
            source_base + "; competing-risk favorable probability",
        ),
        "oof_competing_adverse_probability": (
            "competing_risk_probability",
            source_base + "; reconstructed competing-risk adverse probability",
        ),
        "oof_competing_timeout_probability": (
            "competing_risk_probability",
            source_base + "; reconstructed competing-risk timeout probability",
        ),
        "oof_conditional_path_quality": (
            "conditional_path_quality",
            source_base + "; conditional path-quality prediction",
        ),
        "oof_entry_atr_fraction": (
            "probability_magnitude_economics",
            "causal entry-time ATR fraction from auxiliary target materializer",
        ),
    }
    economics_columns = [
        "oof_literal_expected_peak_mfe_atr",
        "oof_clean_expected_peak_mfe_atr",
        "oof_literal_expected_peak_return",
        "oof_clean_expected_peak_return",
        "oof_quality_adjusted_expected_peak_return",
        "oof_competing_risk_adjusted_peak_return",
        "oof_literal_peak_net_cost_margin",
        "oof_clean_peak_net_cost_margin",
    ]
    for column in economics_columns:
        declarations[column] = (
            "probability_magnitude_economics",
            "deterministic pre-entry composition of repaired OOF probability, "
            "OOF conditional peak magnitude, causal ATR, quality/risk where named, "
            "and the known 1% cost contract for margin columns",
        )
    for column, (family, source) in declarations.items():
        payload["features"][column] = {
            "family": family,
            "source": source,
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": available_at_col,
            "model_input": True,
        }
    def representation_family(column: str) -> str:
        if column in ALPHA_CANDIDATE_RAW_SCORE_COLUMNS:
            return "alpha_candidate_raw_score"
        if column in ALPHA_CANDIDATE_CUTOFF_CONTEXT_COLUMNS:
            return "alpha_candidate_cutoff_context"
        if column in ALPHA_CANDIDATE_TIMESTAMP_RELATIVE_COLUMNS:
            return "alpha_candidate_timestamp_relative"
        if column in ALPHA_CANDIDATE_ARCHETYPE_Z_COLUMNS:
            return "alpha_candidate_archetype_z"
        if column in ALPHA_CANDIDATE_DECILE_GROUP_COLUMNS:
            return "alpha_candidate_decile_group"
        if column == "base_cutoff_score":
            return "alpha_candidate_cutoff_score_control"
        if column in ALPHA_CANDIDATE_CONTEXT_COLUMNS:
            raise ValueError(f"Unassigned alpha candidate context column: {column}")
        if column.startswith(("dae_b16_", "representation_available__dae")):
            return "representation_dae"
        if column.startswith(
            ("gmm_cluster_posterior_", "representation_available__gmm_posterior")
        ):
            return "representation_gmm_posterior"
        if column.startswith(
            (
                "gmm_dist_center_",
                "gmm_mahal_",
                "representation_available__gmm_geometry",
            )
        ):
            return "representation_gmm_geometry"
        return "representation_risk_summary"

    for column in representation_columns:
        payload["features"][column] = {
            "family": representation_family(column),
            "source": (
                "strict OOF alpha candidate geometry from the frozen base stream"
                if column in ALPHA_CANDIDATE_CONTEXT_COLUMNS
                else
                "outcome-free frozen side-local pre-March AE/GMM representation; "
                "missing generated values filled with zero and accompanied by "
                "gmm_representation_available"
            ),
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": available_at_col,
            "model_input": True,
        }
    source_artifacts = payload["handoff"]["source_artifacts"]
    source_artifacts["repaired_literal_reach_oof"] = {
        "path": str(args.literal_oof),
        "sha256": _sha256(args.literal_oof),
    }
    source_artifacts["repaired_clean_and_competing_risk_oof"] = {
        "path": str(args.clean_oof),
        "sha256": _sha256(args.clean_oof),
    }
    source_artifacts["causal_entry_atr"] = {
        "path": str(args.targets),
        "sha256": _sha256(args.targets),
    }
    payload["handoff"]["row_count"] = len(joined)
    if label_manifest is not None:
        exit_contract = label_manifest["exit_policy_contract"]
        payload["targets"]["execution_net_ev_12h"].update(
            {
                "horizon_hours": float(exit_contract["horizon_minutes"]) / 60.0,
                "exit_policy_contract": exit_contract,
                "label_end_time_col": "execution_label_end_utc",
                "source": "replacement_execution_labels_12h",
            }
        )
        source_artifacts["replacement_execution_labels_12h"] = {
            "path": str(args.execution_labels),
            "sha256": _sha256(args.execution_labels),
            "manifest": str(args.execution_label_manifest),
            "manifest_sha256": _sha256(args.execution_label_manifest),
        }
    if representation_manifest is not None:
        source_artifacts["frozen_ae_gmm_representation"] = {
            "path": str(args.representation_context),
            "sha256": _sha256(args.representation_context),
            "manifest": str(args.representation_manifest),
            "manifest_sha256": _sha256(args.representation_manifest),
            "selection_exception": representation_manifest["ae_gmm"][
                "representation_selection_exception"
            ],
        }
    payload["materializer"] = {
        "schema": "repaired_execution_ev_handoff_v1",
        "literal_oof_sha256": _sha256(args.literal_oof),
        "clean_oof_sha256": _sha256(args.clean_oof),
        "targets_sha256": _sha256(args.targets),
        "cost_contract_return": 0.01,
        "probability_magnitude_rule": "P(event) * conditional peak ATR * entry ATR fraction",
        "execution_label_manifest": (
            str(args.execution_label_manifest)
            if args.execution_label_manifest is not None
            else None
        ),
    }
    joined_path = args.output_dir / "joined.parquet"
    provenance_path = args.output_dir / "joined.provenance.json"
    joined.to_parquet(joined_path, index=False, compression="zstd")
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return joined_path, provenance_path


def main() -> None:
    joined, provenance = run(_parser().parse_args())
    print(f"joined: {joined}")
    print(f"provenance: {provenance}")


if __name__ == "__main__":
    main()
