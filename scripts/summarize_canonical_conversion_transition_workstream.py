#!/usr/bin/env python3
"""Summarize conversion feature, target and frozen-tail attribution evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        LABEL_SOURCE,
        sha256,
    )
except ModuleNotFoundError:
    from run_canonical_economic_conversion_transition_head_ablation import (
        LABEL_SOURCE,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTION_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_economic_conversion_contribution_labels_20260729_v1"
)
FEATURE_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_economic_conversion_transition_feature_group_ablation_20260729_v1"
)
TARGET_SOURCE = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_economic_conversion_transition_target_ablation_20260729_v1"
)
ATTRIBUTION_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_base_conversion_prediction_attribution_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_conversion_transition_workstream_summary_20260729_v1"
)
SCHEMA = "canonical_conversion_transition_workstream_summary_v1"


def _artifact_hashes(root: Path, files: tuple[str, ...]) -> dict[str, str]:
    manifest = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    paths = (manifest, sidecar, *(root / name for name in files))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"summary source is incomplete: {missing}")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest):
        raise ValueError(f"summary source manifest checksum fails: {root}")
    return {str(path): sha256(path) for path in paths}


def _soft_label_equivalence(
    base: pd.DataFrame, contribution: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "cohort_anchor_utc",
        "side_name",
        "frozen_base_score_decile",
        "horizon_hours",
    ]
    joined = base.merge(
        contribution,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    records: list[dict[str, Any]] = []
    for horizon, group in joined.groupby("horizon_hours", sort=True):
        complete = (
            group["before_global_hour_complete_flag"].astype(bool)
            & group["after_global_hour_complete_flag"].astype(bool)
        )
        left = group.loc[
            complete, "delta_opportunity_probability_0bps"
        ].to_numpy(float)
        right = group.loc[complete, "delta_net_positive_rate"].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        records.append(
            {
                "horizon_hours": int(horizon),
                "complete_finite_rows": int(finite.sum()),
                "max_absolute_difference": float(
                    np.max(np.abs(left[finite] - right[finite]))
                ),
                "exact_equal": bool(
                    np.array_equal(left[finite], right[finite])
                ),
                "interpretation": "soft net-positive rate duplicates existing opportunity_0bps on complete windows",
            }
        )
    return pd.DataFrame.from_records(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    hashes: dict[str, str] = {}
    hashes.update(
        _artifact_hashes(
            Path(args.base_label_source), ("cohort_transition_labels.parquet",)
        )
    )
    hashes.update(
        _artifact_hashes(
            Path(args.contribution_source), ("cohort_contribution_labels.parquet",)
        )
    )
    hashes.update(
        _artifact_hashes(
            Path(args.feature_source),
            (
                "feature_group_gates.parquet",
                "feature_group_advancement_gates.parquet",
            ),
        )
    )
    hashes.update(
        _artifact_hashes(
            Path(args.target_source),
            ("target_gates.parquet", "period_target_metrics.parquet"),
        )
    )
    hashes.update(
        _artifact_hashes(
            Path(args.attribution_source),
            (
                "march_april_fixed_bin_attribution.parquet",
                "high_low_daily_block_bootstrap.parquet",
                "monthly_base_tail_attribution.parquet",
            ),
        )
    )
    base = pd.read_parquet(
        Path(args.base_label_source) / "cohort_transition_labels.parquet",
        columns=[
            "cohort_anchor_utc",
            "side_name",
            "frozen_base_score_decile",
            "horizon_hours",
            "before_global_hour_complete_flag",
            "after_global_hour_complete_flag",
            "delta_opportunity_probability_0bps",
        ],
    )
    contribution = pd.read_parquet(
        Path(args.contribution_source) / "cohort_contribution_labels.parquet",
        columns=[
            "cohort_anchor_utc",
            "side_name",
            "frozen_base_score_decile",
            "horizon_hours",
            "delta_net_positive_rate",
        ],
    )
    equivalence = _soft_label_equivalence(base, contribution)
    feature_advancement = pd.read_parquet(
        Path(args.feature_source) / "feature_group_advancement_gates.parquet"
    )
    target_gates = pd.read_parquet(
        Path(args.target_source) / "target_gates.parquet"
    )
    attribution = pd.read_parquet(
        Path(args.attribution_source)
        / "march_april_fixed_bin_attribution.parquet"
    )
    high_low = pd.read_parquet(
        Path(args.attribution_source) / "high_low_daily_block_bootstrap.parquet"
    )
    top10 = pd.read_parquet(
        Path(args.attribution_source) / "monthly_base_tail_attribution.parquet"
    )
    top10 = top10.loc[top10["fraction"].eq(0.10)].copy()

    high_low["ci_excludes_zero"] = (
        high_low["ci95_low_bps"].gt(0.0)
        | high_low["ci95_high_bps"].lt(0.0)
    )
    genuine_target_passes = target_gates.loc[
        target_gates["passes_predeclared_component_gate"]
        & ~target_gates["target_arm"].eq("B2_soft_net_positive_rate")
    ].copy()
    decision = {
        "feature_groups_advancing": int(
            feature_advancement[
                "advances_to_frozen_ordering_diagnostic"
            ].sum()
        ),
        "genuine_target_arms_passing": genuine_target_passes[
            "target_arm"
        ].tolist(),
        "duplicate_soft_label_excluded": bool(equivalence["exact_equal"].all()),
        "attribution_heads_with_nonzero_high_low_ci": int(
            high_low["ci_excludes_zero"].sum()
        ),
        "maximum_absolute_predicted_state_composition_effect_bps": float(
            attribution["predicted_state_composition_effect_bps"].abs().max()
        ),
        "march_april_actual_top10_change_bps": float(
            attribution["actual_net_change_bps"].iloc[0]
        ),
        "attach_to_admission": False,
        "portfolio_replay_authorized": False,
        "reason": "unconditional upside labels are learnable, but no feature group repairs opportunity and direct-net stability together and OOF predicted states do not explain or significantly stratify frozen-tail economics",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    frames = {
        "soft_label_equivalence.parquet": equivalence,
        "feature_group_advancement.parquet": feature_advancement,
        "target_component_gates.parquet": target_gates,
        "frozen_tail_fixed_bin_attribution.parquet": attribution,
        "frozen_tail_high_low_uncertainty.parquet": high_low,
        "frozen_tail_top10_metrics.parquet": top10,
    }
    for name, frame in frames.items():
        frame.to_parquet(temporary / name, index=False, compression="zstd")
    (temporary / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETE_DIAGNOSTIC_NO_ADMISSION_NO_REPLAY",
        "source_artifacts_sha256": hashes,
        "decision": decision,
        "outputs_sha256": {
            path.name: sha256(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {"output": str(output), **decision}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--contribution-source", type=Path, default=CONTRIBUTION_SOURCE)
    result.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    result.add_argument("--target-source", type=Path, default=TARGET_SOURCE)
    result.add_argument("--attribution-source", type=Path, default=ATTRIBUTION_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
