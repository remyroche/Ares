#!/usr/bin/env python3
"""Enrich reliability v2 with chronological supports and pre-exit capture."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_canonical_execution_reliability_input as v2


SOURCE = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v2"
SUPPORTS = ROOT / "data_perp/artifacts/canonical_repaired_full_base_chronological_supports_20260730_v2"
CAPTURE = ROOT / "data_perp/artifacts/canonical_pre_exit_capture_labels_20260730_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v3"
IDENTITY = ("candidate_id", "side_name")
EXPECTED_ROWS = 110_730
CAPTURE_COLUMNS = (
    "pre_exit_mfe_return",
    "pre_exit_mfe_atr",
    "target_pre_exit_meaningful_mfe",
    "target_pre_exit_economic_opportunity",
    "pre_exit_path_policy_parity",
    "target_pre_exit_capture_valid",
    "target_pre_exit_capture_net_positive",
    "target_pre_exit_capture_ratio",
    "target_pre_exit_economic_capture_ratio",
    "target_pre_exit_capture_shortfall_return",
    "target_pre_exit_uncaptured_net_opportunity_return",
)
SUPPORT_PROVENANCE = (
    "support_fold",
    "support_validation_start_utc",
    "support_validation_end_utc",
    "support_train_decision_max_utc",
    "support_train_label_end_max_utc",
    "support_train_rows",
)


class ReliabilityV3Error(RuntimeError):
    pass


def verify(root: Path, schema: str) -> dict[str, Any]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not manifest.is_file() or not seal.is_file():
        raise ReliabilityV3Error(f"sealed source missing: {root}")
    if v2.sha256(manifest) != seal.read_text().split()[0]:
        raise ReliabilityV3Error(f"source seal mismatch: {root}")
    payload = json.loads(manifest.read_text())
    if payload.get("schema") != schema:
        raise ReliabilityV3Error(f"source schema mismatch: {root}")
    for name, expected in payload.get("outputs_sha256", {}).items():
        path = root / name
        if not path.is_file() or v2.sha256(path) != expected:
            raise ReliabilityV3Error(f"source output mismatch: {path}")
    return payload


def replace_march_supports(
    panel: pd.DataFrame,
    supports: pd.DataFrame,
    support_columns: Sequence[str],
) -> pd.DataFrame:
    result = panel.copy()
    for column in ("__ts__", "execution_decision_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
        supports[column] = pd.to_datetime(supports[column], utc=True, errors="raise")
    if len(supports) != 41_472 or supports.duplicated(list(IDENTITY)).any():
        raise ReliabilityV3Error("chronological support identity contract failed")
    march = result.model_development_eligible.astype(bool)
    expected = result.loc[
        march, [*IDENTITY, "__symbol__", "__ts__", "execution_decision_utc"]
    ]
    joined = expected.merge(
        supports.loc[
            :,
            [
                *IDENTITY,
                "__symbol__",
                "__ts__",
                "execution_decision_utc",
                *support_columns,
                *SUPPORT_PROVENANCE,
            ],
        ],
        on=[*IDENTITY, "__symbol__", "__ts__", "execution_decision_utc"],
        how="left",
        validate="one_to_one",
    )
    if len(joined) != int(march.sum()) or joined[list(support_columns)].isna().any().any():
        raise ReliabilityV3Error("chronological supports do not cover March exactly")
    key = pd.MultiIndex.from_frame(result.loc[:, list(IDENTITY)])
    joined_key = pd.MultiIndex.from_frame(joined.loc[:, list(IDENTITY)])
    positions = key.get_indexer(joined_key)
    if np.any(positions < 0):
        raise ReliabilityV3Error("chronological support key lookup failed")
    for column in support_columns:
        result.loc[positions, column] = joined[column].to_numpy(float)
    for column in SUPPORT_PROVENANCE:
        result[column] = pd.NA
        result.loc[positions, column] = joined[column].to_numpy()
    result["support_score_role"] = np.where(
        march,
        "strict_chronological_side_local_oof",
        "frozen_forward_side_local",
    )
    result["support_scores_are_chronological_oof"] = march
    result["support_scores_are_frozen_forward"] = ~march
    for column in support_columns:
        if result[column].isna().any() or not np.isfinite(result[column].to_numpy(float)).all():
            raise ReliabilityV3Error(f"support replacement produced invalid {column}")
    return result


def join_capture(panel: pd.DataFrame, capture: pd.DataFrame) -> pd.DataFrame:
    selected = capture.loc[
        :,
        [
            *IDENTITY,
            "__symbol__",
            "__ts__",
            "execution_decision_utc",
            "execution_label_end_utc",
            *CAPTURE_COLUMNS,
        ],
    ].copy()
    for column in ("__ts__", "execution_decision_utc", "execution_label_end_utc"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="raise")
        selected[column] = pd.to_datetime(selected[column], utc=True, errors="raise")
    if len(selected) != EXPECTED_ROWS or selected.duplicated(list(IDENTITY)).any():
        raise ReliabilityV3Error("capture identity contract failed")
    result = panel.merge(
        selected,
        on=[
            *IDENTITY,
            "__symbol__",
            "__ts__",
            "execution_decision_utc",
            "execution_label_end_utc",
        ],
        how="left",
        validate="one_to_one",
    )
    if result[list(CAPTURE_COLUMNS[:7])].isna().any().any():
        raise ReliabilityV3Error("capture labels do not cover every reliability row")
    invalid = ~result.target_pre_exit_capture_valid.astype(bool)
    conditional = [
        "target_pre_exit_economic_capture_ratio",
        "target_pre_exit_uncaptured_net_opportunity_return",
    ]
    if result.loc[invalid, conditional].notna().any().any():
        raise ReliabilityV3Error("invalid capture rows received conditional targets")
    return result


def updated_roles(old: Mapping[str, Any]) -> dict[str, Any]:
    roles = dict(old)
    target_only = list(roles["target_only_never_features"])
    for column in CAPTURE_COLUMNS:
        if column not in target_only:
            target_only.append(column)
    roles["target_only_never_features"] = target_only
    roles["support_provenance"] = list(SUPPORT_PROVENANCE)
    roles["capture_target_contract"] = {
        "event_heads": [
            "target_pre_exit_meaningful_mfe",
            "target_pre_exit_economic_opportunity",
            "target_pre_exit_capture_net_positive",
        ],
        "capture_validity_mask": "target_pre_exit_capture_valid",
        "conditional_magnitudes": [
            "target_pre_exit_capture_ratio",
            "target_pre_exit_economic_capture_ratio",
            "target_pre_exit_capture_shortfall_return",
            "target_pre_exit_uncaptured_net_opportunity_return",
        ],
    }
    roles["explicitly_unavailable"] = [
        item
        for item in roles.get("explicitly_unavailable", [])
        if "pre-exit capture" not in str(item)
    ]
    roles["capture_limitations"] = [
        "1,301 path-policy parity failures are excluded from capture training.",
        "Capture targets are labels only and can never be model features.",
    ]
    if set(roles["default_ev_inputs"]).intersection(CAPTURE_COLUMNS):
        raise ReliabilityV3Error("capture label escaped into default EV inputs")
    return roles


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_manifest = verify(
        args.source, "canonical_execution_reliability_input_v2"
    )
    support_manifest = verify(
        args.supports, "canonical_repaired_full_base_chronological_supports_v2"
    )
    capture_manifest = verify(
        args.capture, "canonical_pre_exit_capture_labels_v2"
    )
    panel = pd.read_parquet(args.source / "panel.parquet")
    old_roles = json.loads((args.source / "feature_roles.json").read_text())
    support_columns = list(old_roles["repaired_full_base_support_sidecars"])
    supports = pd.read_parquet(args.supports / "support_sidecars.parquet")
    panel = replace_march_supports(panel, supports, support_columns)
    capture = pd.read_parquet(args.capture / "labels.parquet")
    panel = join_capture(panel, capture)
    roles = updated_roles(old_roles)
    if len(panel) != EXPECTED_ROWS or panel.duplicated(list(IDENTITY)).any():
        raise ReliabilityV3Error("final v3 identity contract failed")
    if panel.loc[:, roles["default_ev_inputs"]].isna().any().any():
        raise ReliabilityV3Error("default v3 input contains missing values")

    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        panel.to_parquet(stage / "panel.parquet", index=False, compression="zstd")
        v2.write_json(stage / "feature_roles.json", roles)
        capture_support = pd.DataFrame(
            [
                {
                    "metric": "path_policy_parity_rows",
                    "rows": int(panel.pre_exit_path_policy_parity.sum()),
                },
                {
                    "metric": "pre_exit_economic_opportunity_rows",
                    "rows": int(panel.target_pre_exit_economic_opportunity.sum()),
                },
                {
                    "metric": "capture_valid_rows",
                    "rows": int(panel.target_pre_exit_capture_valid.sum()),
                },
                {
                    "metric": "capture_valid_net_positive_rows",
                    "rows": int(
                        panel.loc[
                            panel.target_pre_exit_capture_valid.astype(bool),
                            "target_pre_exit_capture_net_positive",
                        ].sum()
                    ),
                },
            ]
        )
        capture_support.to_csv(stage / "capture_support.csv", index=False)
        outputs = {
            path.name: v2.sha256(path)
            for path in stage.iterdir()
            if path.is_file()
        }
        manifest = {
            "schema": "canonical_execution_reliability_input_v3",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_INPUT_CHRONOLOGICAL_SUPPORTS_PRE_EXIT_CAPTURE_NO_PROMOTION",
            "promotion_eligible": False,
            "rows": len(panel),
            "march_chronological_support_rows": int(
                panel.support_scores_are_chronological_oof.sum()
            ),
            "april_frozen_forward_support_rows": int(
                panel.support_scores_are_frozen_forward.sum()
            ),
            "capture_valid_rows": int(panel.target_pre_exit_capture_valid.sum()),
            "input_sha256": {
                "source_manifest": v2.sha256(args.source / "manifest.json"),
                "source_panel": source_manifest["outputs_sha256"]["panel.parquet"],
                "support_manifest": v2.sha256(args.supports / "manifest.json"),
                "support_sidecars": support_manifest["outputs_sha256"][
                    "support_sidecars.parquet"
                ],
                "capture_manifest": v2.sha256(args.capture / "manifest.json"),
                "capture_labels": capture_manifest["outputs_sha256"]["labels.parquet"],
            },
            "feature_contract": roles,
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": v2.sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "April remains reused diagnostic evidence and is never promotion evidence.",
                "Frozen support configuration selection remains historical static-OOF research evidence; per-row March support predictions are now chronological.",
                "Capture path-policy parity failures are excluded, never imputed.",
            ],
        }
        v2.write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            v2.sha256(stage / "manifest.json") + "  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--supports", type=Path, default=SUPPORTS)
    command.add_argument("--capture", type=Path, default=CAPTURE)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(v2.safe(run(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
