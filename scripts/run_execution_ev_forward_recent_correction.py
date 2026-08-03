#!/usr/bin/env python3
"""Extend the frozen execution-EV winner through a causal recent-EV correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ExecutionEVModelAblationConfig,
    ExecutionEVTargetSpec,
    apply_execution_ev_causal_recent_ev_correction,
    load_execution_ev_model_ablation_bundle,
)

ID = ("__ts__", "__symbol__", "side_name", "candidate_id")
RAW_SCORE = "catboost__residual__without_hpo__all_features"
CORRECTED_SCORE = f"{RAW_SCORE}__recent_ev_catboost_predicted_archetype"
HISTORICAL_CORRECTED_SCORE = (
    "catboost__residual__without_hpo__all_features"
    "__recent_ev_catboost_predicted_archetype"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(payload: dict[str, object]) -> ExecutionEVModelAblationConfig:
    values = dict(payload)
    target = values.get("target_spec")
    if isinstance(target, dict):
        values["target_spec"] = ExecutionEVTargetSpec(**target)
    for name in ("target_modes", "additional_input_families", "feature_arms"):
        if name in values:
            values[name] = tuple(values[name])
    return ExecutionEVModelAblationConfig(**values)


def _top10(frame: pd.DataFrame, score_col: str) -> dict[str, float | int]:
    local = frame.loc[
        np.isfinite(pd.to_numeric(frame[score_col], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["execution_net_ev_12h"], errors="coerce"))
    ]
    if local.empty:
        return {"eligible_rows": 0, "selected_rows": 0, "mean_net_ev_bps": np.nan}
    selected_rows = max(1, int(np.ceil(0.10 * len(local))))
    selected = local.nlargest(selected_rows, score_col)
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    return {
        "eligible_rows": int(len(local)),
        "selected_rows": int(len(selected)),
        "mean_net_ev_bps": float(10_000.0 * net.mean()),
        "positive_rate": float((net > 0.0).mean()),
    }


def run(
    *,
    historical_path: Path,
    historical_handoff_path: Path,
    forward_path: Path,
    bundle_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    bundle = load_execution_ev_model_ablation_bundle(bundle_path)
    config = _config(bundle.config)
    historical = pd.read_parquet(historical_path)
    raw_flag = f"{RAW_SCORE}__is_oof"
    if raw_flag not in historical or not historical[raw_flag].astype(bool).any():
        raise ValueError("historical input lacks winner outer-OOF rows")
    historical = historical.loc[historical[raw_flag].astype(bool)].copy()
    historical["evaluation_origin"] = "historical_outer_oof"
    historical["promotion_eligible"] = True

    forward = pd.read_parquet(forward_path)
    if forward["is_oof"].astype(bool).any():
        raise ValueError("forward input must remain non-OOF")
    if forward["promotion_eligible"].astype(bool).any():
        raise ValueError("forward input must remain non-promotable")
    forward = forward.copy()
    forward[RAW_SCORE] = pd.to_numeric(
        forward["frozen_winner_raw_ev"], errors="raise"
    )
    fold_col = "execution_ev_model_ablation_oof_fold"
    if fold_col not in historical:
        raise ValueError(f"historical input lacks fold column {fold_col!r}")
    forward[fold_col] = (
        int(pd.to_numeric(historical[fold_col], errors="raise").max()) + 1
    )
    forward["evaluation_origin"] = "frozen_final_fit_forward_oos"
    overlap = historical.loc[:, ID].merge(forward.loc[:, ID], on=list(ID), how="inner")
    if len(overlap):
        raise ValueError(f"forward input overlaps {len(overlap)} OOF identities")

    required = {
        *ID,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_net_ev_12h",
        "catboost_archetype",
        RAW_SCORE,
        "evaluation_origin",
        "promotion_eligible",
        fold_col,
    }
    missing_historical = sorted(required - set(historical.columns))
    missing_forward = sorted(required - set(forward.columns))
    if missing_historical or missing_forward:
        raise ValueError(
            "recent-correction inputs missing columns: "
            f"historical={missing_historical}, forward={missing_forward}"
        )
    frame = pd.concat(
        [historical.loc[:, sorted(required)], forward.loc[:, sorted(required)]],
        ignore_index=True,
    ).sort_values(
        ["execution_decision_utc", "__symbol__", "side_name", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)
    corrected, correction_report = apply_execution_ev_causal_recent_ev_correction(
        frame,
        pd.to_numeric(frame[RAW_SCORE], errors="raise").to_numpy(float),
        pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float),
        bundle.provenance,
        route="catboost_predicted_archetype",
        config=config,
    )
    frame[CORRECTED_SCORE] = corrected
    forward_mask = frame["evaluation_origin"].eq("frozen_final_fit_forward_oos")
    frame[f"{CORRECTED_SCORE}__is_oof"] = np.isfinite(corrected) & ~forward_mask
    frame[f"{CORRECTED_SCORE}__is_forward_oos"] = (
        np.isfinite(corrected) & forward_mask
    )
    frame[f"{CORRECTED_SCORE}__is_evaluation"] = np.isfinite(corrected)

    historical_parity = {"available": False}
    if HISTORICAL_CORRECTED_SCORE in historical.columns:
        expected = historical.set_index(list(ID))[HISTORICAL_CORRECTED_SCORE]
        observed = frame.loc[~forward_mask].set_index(list(ID))[CORRECTED_SCORE]
        delta = np.abs(
            pd.to_numeric(expected.loc[observed.index], errors="raise").to_numpy(float)
            - pd.to_numeric(observed, errors="raise").to_numpy(float)
        )
        historical_parity = {
            "available": True,
            "rows": int(len(delta)),
            "max_abs_delta": float(np.max(delta)) if len(delta) else 0.0,
        }
        if len(delta) and not np.allclose(delta, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "causal correction failed historical parity: "
                f"max_abs_delta={float(np.max(delta))}"
            )

    metrics: dict[str, object] = {
        "pooled_global_top10": _top10(frame, CORRECTED_SCORE),
        "by_origin": {
            str(origin): _top10(group, CORRECTED_SCORE)
            for origin, group in frame.groupby("evaluation_origin", sort=True)
        },
        "by_month": {
            str(month): _top10(group, CORRECTED_SCORE)
            for month, group in frame.groupby(
                pd.to_datetime(frame["execution_decision_utc"], utc=True).dt.strftime(
                    "%Y-%m"
                ),
                sort=True,
            )
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "mapped_oof_and_forward.parquet"
    frame.to_parquet(output, index=False, compression="zstd")
    label_columns = [
        *ID,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_exit_reason",
        "execution_exit_hour",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
    ]
    historical_handoff = pd.read_parquet(
        historical_handoff_path, columns=label_columns
    )
    historical_keys = historical.loc[:, ID]
    historical_labels = historical_keys.merge(
        historical_handoff,
        on=list(ID),
        how="left",
        validate="one_to_one",
    )
    if historical_labels[label_columns[4:]].isna().any().any():
        raise ValueError("historical handoff does not cover every OOF identity")
    missing_forward_labels = sorted(set(label_columns) - set(forward.columns))
    if missing_forward_labels:
        raise ValueError(
            "forward table is missing policy replay labels: "
            + ", ".join(missing_forward_labels)
        )
    combined_handoff = pd.concat(
        [historical_labels.loc[:, label_columns], forward.loc[:, label_columns]],
        ignore_index=True,
    )
    if combined_handoff.duplicated(list(ID)).any():
        raise ValueError("combined policy handoff identity is not one-to-one")
    combined_handoff_path = output_dir / "combined_policy_handoff.parquet"
    combined_handoff.to_parquet(
        combined_handoff_path, index=False, compression="zstd"
    )
    manifest = {
        "schema": "execution_ev_forward_recent_correction_v1",
        "promotion_eligible": False,
        "selection_basis": "one pooled global top decile across timestamps and sides",
        "historical_parity": historical_parity,
        "correction": correction_report,
        "metrics": metrics,
        "config": asdict(config),
        "sources": {
            "historical": {
                "path": str(historical_path),
                "sha256": _sha256(historical_path),
            },
            "historical_handoff": {
                "path": str(historical_handoff_path),
                "sha256": _sha256(historical_handoff_path),
            },
            "forward": {"path": str(forward_path), "sha256": _sha256(forward_path)},
            "bundle": {"path": str(bundle_path), "sha256": _sha256(bundle_path)},
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "combined_policy_handoff": str(combined_handoff_path),
            "combined_policy_handoff_sha256": _sha256(combined_handoff_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--historical-handoff", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, manifest = run(
        historical_path=args.historical,
        historical_handoff_path=args.historical_handoff,
        forward_path=args.forward,
        bundle_path=args.bundle,
        output_dir=args.output_dir,
    )
    print(f"mapped: {output}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
