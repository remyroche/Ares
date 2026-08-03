#!/usr/bin/env python3
"""Evaluate causal global and side-to-global maps for exact event scores.

The input is the immutable June OOF calibration ledger followed by the frozen
June-trained July predictions.  Each daily map uses only rows whose exact
12-hour outcome resolved before that day's snapshot and within the preceding
21 days.  Selection is one pooled global top-k over the whole July evaluation
period; no timestamp, day, side, asset, or archetype quota is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_meaningful_mfe_exact_grid_june_calibration_oof import (  # noqa: E402
    PREDECLARED_SCORES,
    SCHEMA as LEDGER_SCHEMA,
)
from scripts.run_execution_ev_recent_mapping_ablation import (  # noqa: E402
    causal_mappings,
)
from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (  # noqa: E402
    sha256,
)
from scripts.run_meaningful_mfe_exact_grid_reset import (  # noqa: E402
    IDENTITY,
    stable_top,
)


SCHEMA = "meaningful_mfe_exact_grid_causal_mapping_v1"
FORWARD_PARTITION = "june_to_july_frozen_forward_oos"
MAP_COLUMNS = {
    "raw_common_eligible": "_raw_score",
    "causal_global": "causal_recent_isotonic_ev",
    "side_calibrated_to_global": "causal_recent_side_isotonic_ev",
}
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_ledger(
    ledger_path: Path, ledger_manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != LEDGER_SCHEMA:
        raise ValueError("unexpected calibration-ledger schema")
    expected = manifest.get("outputs", {}).get("ledger", {}).get("sha256")
    if not expected or sha256(ledger_path) != expected:
        raise ValueError("calibration-ledger hash mismatch")
    ledger = pd.read_parquet(ledger_path)
    missing = sorted(
        {
            *IDENTITY,
            *PREDECLARED_SCORES,
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_net_ev_12h",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "source_partition",
            "is_oof",
            "prediction_available_at",
            "score_recipe_hash",
        }.difference(ledger.columns)
    )
    if missing:
        raise ValueError(f"calibration ledger lacks {missing}")
    if ledger.duplicated(list(IDENTITY)).any():
        raise ValueError("calibration ledger contains duplicate identities")
    for column in (
        "execution_decision_utc",
        "execution_label_end_utc",
        "prediction_available_at",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    if not bool(
        ledger["prediction_available_at"].le(
            ledger["execution_decision_utc"]
        ).all()
    ):
        raise ValueError("ledger contains unavailable predictions")
    gross = pd.to_numeric(
        ledger["execution_gross_ev_12h"], errors="raise"
    ).to_numpy(float)
    cost = pd.to_numeric(
        ledger["execution_cost_return"], errors="raise"
    ).to_numpy(float)
    net = pd.to_numeric(
        ledger["execution_net_ev_12h"], errors="raise"
    ).to_numpy(float)
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise ValueError("ledger violates exact gross-cost-net accounting")
    recipes = ledger["score_recipe_hash"].astype(str).unique()
    if len(recipes) != 1:
        raise ValueError("calibration and forward rows do not share one recipe")
    return ledger.reset_index(drop=True), manifest


def _cvar5(values: pd.Series) -> float:
    local = pd.to_numeric(values, errors="raise").sort_values()
    count = max(1, int(math.ceil(len(local) * 0.05)))
    return float(local.iloc[:count].mean())


def tail_row(
    population: pd.DataFrame,
    *,
    score_column: str,
    score_name: str,
    arm: str,
    fraction: float,
    eligibility: str,
    common_eligible_rows: int,
) -> dict[str, Any]:
    if population.empty:
        return {
            "score_name": score_name,
            "arm": arm,
            "eligibility": eligibility,
            "fraction": fraction,
            "common_eligible_rows": common_eligible_rows,
            "admitted_rows": 0,
            "selected_rows": 0,
            "net_ev_bps": np.nan,
            "gross_ev_bps": np.nan,
            "cost_bps": np.nan,
            "positive_net_rate": np.nan,
            "cvar5_bps": np.nan,
            "long_share": np.nan,
            "asset_count": 0,
        }
    selected = stable_top(population, score_column, fraction)
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    gross = pd.to_numeric(
        selected["execution_gross_ev_12h"], errors="raise"
    )
    cost = pd.to_numeric(selected["execution_cost_return"], errors="raise")
    if not np.allclose(
        gross.to_numpy(float) - cost.to_numpy(float),
        net.to_numpy(float),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("selected tail violates exact accounting")
    return {
        "score_name": score_name,
        "arm": arm,
        "eligibility": eligibility,
        "fraction": fraction,
        "common_eligible_rows": common_eligible_rows,
        "admitted_rows": len(population),
        "selected_rows": len(selected),
        "selected_fraction_of_common": len(selected) / common_eligible_rows,
        "net_ev_bps": float(net.mean() * 1e4),
        "gross_ev_bps": float(gross.mean() * 1e4),
        "cost_bps": float(cost.mean() * 1e4),
        "positive_net_rate": float((net > 0.0).mean()),
        "cvar5_bps": float(_cvar5(net) * 1e4),
        "long_share": float(
            selected["side_name"].astype(str).eq("long").mean()
        ),
        "asset_count": int(selected["__symbol__"].nunique()),
    }


def evaluate_score(
    ledger: pd.DataFrame,
    *,
    score_name: str,
    window_days: int,
    min_reference_rows: int,
    side_support_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped, audit = causal_mappings(
        ledger,
        score_col=score_name,
        window_days=window_days,
        min_reference_rows=min_reference_rows,
        side_support_target=side_support_target,
    )
    mapped["_raw_score"] = pd.to_numeric(
        mapped[score_name], errors="raise"
    )
    forward = mapped.loc[
        mapped["source_partition"].astype(str).eq(FORWARD_PARTITION)
    ].copy()
    common = (
        np.isfinite(forward["_raw_score"])
        & np.isfinite(forward["causal_recent_isotonic_ev"])
        & np.isfinite(forward["causal_recent_side_isotonic_ev"])
    )
    forward["mapped_eligible"] = common
    eligible = forward.loc[common].copy()
    if eligible.empty:
        raise ValueError(f"{score_name} has no commonly mapped July rows")
    rows: list[dict[str, Any]] = []
    for arm, score_column in MAP_COLUMNS.items():
        for fraction in FRACTIONS:
            rows.append(
                tail_row(
                    eligible,
                    score_column=score_column,
                    score_name=score_name,
                    arm=arm,
                    fraction=fraction,
                    eligibility="common_21d_history",
                    common_eligible_rows=len(eligible),
                )
            )
    for arm, score_column in (
        ("causal_global_positive_admission", "causal_recent_isotonic_ev"),
        (
            "side_to_global_positive_admission",
            "causal_recent_side_isotonic_ev",
        ),
    ):
        admitted = eligible.loc[eligible[score_column].gt(0.0)].copy()
        for fraction in FRACTIONS:
            rows.append(
                tail_row(
                    admitted,
                    score_column=score_column,
                    score_name=score_name,
                    arm=arm,
                    fraction=fraction,
                    eligibility="mapped_ev_gt_zero_after_21d_history",
                    common_eligible_rows=len(eligible),
                )
            )
    coverage = (
        forward.assign(
            decision_day=forward["execution_decision_utc"].dt.floor("D")
        )
        .groupby("decision_day", sort=True)
        .agg(
            evaluation_rows=("candidate_id", "size"),
            mapped_rows=("mapped_eligible", "sum"),
        )
        .reset_index()
    )
    coverage["score_name"] = score_name
    coverage["mapped_fraction"] = (
        coverage["mapped_rows"] / coverage["evaluation_rows"]
    )
    audit_frame = pd.DataFrame(audit)
    audit_frame["score_name"] = score_name
    keep = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "side_name",
        "any_touch",
        "clean_first",
        "positive_net",
        "timeout",
        "mapped_eligible",
        "_raw_score",
        "causal_recent_percentile",
        "causal_recent_robust_z",
        "causal_recent_isotonic_ev",
        "causal_recent_side_isotonic_ev",
    ]
    # side_name is already part of IDENTITY; preserve deterministic order once.
    keep = list(dict.fromkeys(keep))
    long_predictions = forward[keep].copy()
    long_predictions.insert(0, "score_name", score_name)
    return (
        long_predictions,
        pd.DataFrame(rows),
        coverage,
        audit_frame,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    ledger, ledger_manifest = load_ledger(
        args.ledger, args.ledger_manifest
    )
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for score_name in PREDECLARED_SCORES:
        prediction, metric, score_coverage, audit = evaluate_score(
            ledger,
            score_name=score_name,
            window_days=args.window_days,
            min_reference_rows=args.min_reference_rows,
            side_support_target=args.side_support_target,
        )
        predictions.append(prediction)
        metrics.append(metric)
        coverage.append(score_coverage)
        audits.append(audit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Any] = {}
    for name, frame in (
        ("mapped_predictions", pd.concat(predictions, ignore_index=True)),
        ("tail_metrics", pd.concat(metrics, ignore_index=True)),
        ("daily_coverage", pd.concat(coverage, ignore_index=True)),
        ("mapping_audit", pd.concat(audits, ignore_index=True)),
    ):
        path = args.output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {
            "path": path,
            "rows": len(frame),
            "sha256": sha256(path),
        }
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETED_CAUSAL_21D_MAPPING_DIAGNOSTIC_NO_PROMOTION",
        "promotion_eligible": False,
        "contract": {
            "scores": list(PREDECLARED_SCORES),
            "window_days": args.window_days,
            "min_reference_rows": args.min_reference_rows,
            "side_support_target": args.side_support_target,
            "selection": (
                "one pooled global top 1/5/10/20 over the complete July "
                "evaluation; deterministic full-identity ties; no quotas"
            ),
            "common_eligibility": (
                "raw/global/side arms use identical rows with both maps available"
            ),
            "positive_admission": (
                "mapped EV > 0 is separate and has no claimed raw-score equivalent"
            ),
            "reference": (
                "exact outcomes resolved before daily snapshot and no earlier "
                "than snapshot minus 21 days"
            ),
            "cost": "exact net is reused; no second cost subtraction",
        },
        "lineage": {
            "ledger": {
                "path": args.ledger,
                "sha256": sha256(args.ledger),
                "manifest": args.ledger_manifest,
                "manifest_sha256": sha256(args.ledger_manifest),
                "recipe_hash": ledger["score_recipe_hash"].iloc[0],
            },
            "ledger_manifest_schema": ledger_manifest["schema"],
        },
        "runner": {
            "path": Path(__file__).resolve(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": outputs,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    root = Path(
        "data_perp/artifacts/"
        "meaningful_mfe_exact_grid_june_calibration_oof_20260730_v1"
    )
    value.add_argument(
        "--ledger",
        type=Path,
        default=root / "calibration_ledger.parquet",
    )
    value.add_argument(
        "--ledger-manifest",
        type=Path,
        default=root / "manifest.json",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "meaningful_mfe_exact_grid_causal_mapping_20260730_v1"
        ),
    )
    value.add_argument("--window-days", type=int, default=21)
    value.add_argument("--min-reference-rows", type=int, default=500)
    value.add_argument("--side-support-target", type=float, default=500.0)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
