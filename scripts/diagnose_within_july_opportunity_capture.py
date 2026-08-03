#!/usr/bin/env python3
"""Decompose strict within-July ranking into opportunity, capture, regret and cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "within_july_opportunity_capture_diagnosis_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDES = ("long", "short")
PRIMARY_MODE = "forward_expanding"
SCORES = {
    "within_july_model": "prediction",
    "frozen_alpha": "existing_alpha_ev",
}


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _top_decile(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    count = max(1, int(np.ceil(0.10 * len(frame))))
    return frame.sort_values(
        [score, "__ts__", "candidate_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(count)


def economic_components(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize components whose means reconcile net = MFE - regret - cost."""

    gross = frame["execution_gross_ev_12h"].to_numpy(dtype=float)
    cost = frame["execution_cost_return"].to_numpy(dtype=float)
    net = frame["execution_net_ev_12h"].to_numpy(dtype=float)
    mfe = frame["execution_mfe_return_12h"].to_numpy(dtype=float)
    mae = frame["execution_mae_return_12h"].to_numpy(dtype=float)
    mfe_to_gross_gap = mfe - gross
    exit_regret_proxy = np.maximum(mfe - np.maximum(gross, 0.0), 0.0)
    opportunity_net_of_cost = mfe - cost
    positive_opportunity = opportunity_net_of_cost > 0.0
    capture_ratio = np.divide(
        np.maximum(gross, 0.0),
        mfe,
        out=np.full(len(frame), np.nan, dtype=float),
        where=mfe > 1e-8,
    )
    return {
        "rows": int(len(frame)),
        "mean_path_mfe_bps": float(mfe.mean() * 10_000.0),
        "mean_path_mae_bps": float(mae.mean() * 10_000.0),
        "mean_gross_bps": float(gross.mean() * 10_000.0),
        "mean_cost_bps": float(cost.mean() * 10_000.0),
        "mean_net_bps": float(net.mean() * 10_000.0),
        "mean_mfe_to_gross_gap_bps": float(
            mfe_to_gross_gap.mean() * 10_000.0
        ),
        "mean_hindsight_exit_regret_proxy_bps": float(
            exit_regret_proxy.mean() * 10_000.0
        ),
        "mean_path_opportunity_net_of_cost_bps": float(
            opportunity_net_of_cost.mean() * 10_000.0
        ),
        "path_opportunity_positive_rate": float(positive_opportunity.mean()),
        "path_opportunity_net_150bps_rate": float(
            (opportunity_net_of_cost >= 0.015).mean()
        ),
        "gross_positive_rate": float((gross > 0.0).mean()),
        "net_positive_rate": float((net > 0.0).mean()),
        "loss_worse_100bps_rate": float((net <= -0.01).mean()),
        "mae_above_150bps_rate": float((mae >= 0.015).mean()),
        "favorable_first_rate": float(frame["favorable_first"].mean()),
        "adverse_first_rate": float(frame["adverse_first"].mean()),
        "timeout_rate": float(frame["timeout"].mean()),
        "gross_capture_ratio_of_means": (
            float(gross.mean() / mfe.mean()) if mfe.mean() > 1e-8 else np.nan
        ),
        "median_row_gross_capture_ratio": (
            float(np.nanmedian(np.clip(capture_ratio, -2.0, 2.0)))
            if np.isfinite(capture_ratio).any()
            else np.nan
        ),
        "mean_exit_hour": float(frame["execution_exit_hour"].mean()),
    }


def selection_metrics(
    frame: pd.DataFrame,
    *,
    score_name: str,
    score_column: str,
    evaluation: str,
    scope: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = _top_decile(frame, score_column)
    population = economic_components(frame)
    metrics = economic_components(selected)
    metrics.update(
        {
            "evaluation": evaluation,
            "scope": scope,
            "score_name": score_name,
            "score_column": score_column,
            "population_rows": int(len(frame)),
            "top_k_fraction": 0.10,
            "lift_net_bps_vs_population": (
                metrics["mean_net_bps"] - population["mean_net_bps"]
            ),
            "lift_mfe_bps_vs_population": (
                metrics["mean_path_mfe_bps"] - population["mean_path_mfe_bps"]
            ),
            "lift_mfe_to_gross_gap_bps_vs_population": (
                metrics["mean_mfe_to_gross_gap_bps"]
                - population["mean_mfe_to_gross_gap_bps"]
            ),
            "lift_cost_bps_vs_population": (
                metrics["mean_cost_bps"] - population["mean_cost_bps"]
            ),
        }
    )
    return metrics, selected


def compare_selections(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    evaluation: str,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = list(IDENTITY)
    challenger_ids = pd.MultiIndex.from_frame(challenger[identity])
    baseline_ids = pd.MultiIndex.from_frame(baseline[identity])
    added = challenger.loc[~challenger_ids.isin(baseline_ids)]
    dropped = baseline.loc[~baseline_ids.isin(challenger_ids)]
    challenger_components = economic_components(challenger)
    baseline_components = economic_components(baseline)
    delta_mfe = (
        challenger_components["mean_path_mfe_bps"]
        - baseline_components["mean_path_mfe_bps"]
    )
    delta_regret = (
        challenger_components["mean_mfe_to_gross_gap_bps"]
        - baseline_components["mean_mfe_to_gross_gap_bps"]
    )
    delta_cost = (
        challenger_components["mean_cost_bps"]
        - baseline_components["mean_cost_bps"]
    )
    delta_net = (
        challenger_components["mean_net_bps"]
        - baseline_components["mean_net_bps"]
    )
    summary = {
        "evaluation": evaluation,
        "scope": scope,
        "challenger": "within_july_model",
        "baseline": "frozen_alpha",
        "selected_rows": int(len(challenger)),
        "overlap_rows": int(challenger_ids.isin(baseline_ids).sum()),
        "added_rows": int(len(added)),
        "dropped_rows": int(len(dropped)),
        "challenger_net_bps": challenger_components["mean_net_bps"],
        "baseline_net_bps": baseline_components["mean_net_bps"],
        "delta_net_bps": delta_net,
        "delta_mfe_bps": delta_mfe,
        "delta_mfe_to_gross_gap_bps": delta_regret,
        "delta_cost_bps": delta_cost,
        "reconstructed_delta_net_bps": delta_mfe - delta_regret - delta_cost,
        "reconciliation_error_bps": (
            delta_net - (delta_mfe - delta_regret - delta_cost)
        ),
    }
    replacement = []
    for replacement_role, sample in (("added", added), ("dropped", dropped)):
        replacement.append(
            {
                "evaluation": evaluation,
                "scope": scope,
                "replacement_role": replacement_role,
                **economic_components(sample),
            }
        )
    return summary, replacement


def prepare_joined(
    predictions: pd.DataFrame,
    policy: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    grid_name: str,
) -> pd.DataFrame:
    primary = predictions.loc[
        predictions["mode"].eq(PRIMARY_MODE)
        & predictions["is_valid_forward_oos"].astype(bool)
    ].copy()
    if not len(primary):
        raise ValueError("no valid forward-expanding within-July predictions")
    if primary.duplicated(list(IDENTITY)).any():
        raise ValueError("forward-expanding evaluation rows overlap across folds")
    policy_columns = [
        *IDENTITY,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_reason",
        "execution_exit_hour",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
    ]
    if policy.duplicated(list(IDENTITY)).any():
        raise ValueError("policy labels contain duplicate identities")
    grid = grid.loc[grid["grid_name"].eq(grid_name) & grid["label_valid"]].copy()
    grid_columns = [
        *IDENTITY,
        "execution_net_ev_12h",
        "favorable_first",
        "adverse_first",
        "timeout",
    ]
    if grid.duplicated(list(IDENTITY)).any():
        raise ValueError("meaningful-MFE grid contains duplicate identities")
    prediction_net = primary["execution_net_ev_12h"].copy()
    primary = primary.drop(columns=["execution_net_ev_12h"])
    joined = primary.merge(
        policy.loc[:, policy_columns],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    ).merge(
        grid.loc[:, grid_columns],
        on=list(IDENTITY),
        how="inner",
        suffixes=("", "_grid"),
        validate="one_to_one",
    )
    max_grid_net_delta = float(
        np.max(
            np.abs(
                joined["execution_net_ev_12h"].to_numpy(dtype=float)
                - joined["execution_net_ev_12h_grid"].to_numpy(dtype=float)
            )
        )
    )
    if max_grid_net_delta > 1e-7:
        raise ValueError(f"grid/policy net mismatch: {max_grid_net_delta}")
    joined = joined.drop(columns=["execution_net_ev_12h_grid"])
    expected = primary.loc[:, [*IDENTITY]].copy()
    expected["prediction_net"] = prediction_net.to_numpy(dtype=float)
    reconciled = joined.merge(
        expected,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    max_net_delta = float(
        np.max(
            np.abs(
                reconciled["execution_net_ev_12h"].to_numpy(dtype=float)
                - reconciled["prediction_net"].to_numpy(dtype=float)
            )
        )
    )
    if max_net_delta > 1e-7:
        raise ValueError(f"prediction/policy net mismatch: {max_net_delta}")
    accounting_delta = (
        joined["execution_gross_ev_12h"].to_numpy(dtype=float)
        - joined["execution_cost_return"].to_numpy(dtype=float)
        - joined["execution_net_ev_12h"].to_numpy(dtype=float)
    )
    max_accounting_delta = float(np.max(np.abs(accounting_delta)))
    if max_accounting_delta > 1e-7:
        raise ValueError(f"gross-cost-net mismatch: {max_accounting_delta}")
    joined.attrs["prediction_rows"] = int(len(primary))
    joined.attrs["joined_rows"] = int(len(joined))
    joined.attrs["coverage"] = float(len(joined) / len(primary))
    joined.attrs["max_net_delta"] = max_net_delta
    joined.attrs["max_accounting_delta"] = max_accounting_delta
    joined.attrs["max_grid_net_delta"] = max_grid_net_delta
    return joined


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    predictions = pd.read_parquet(args.predictions)
    policy = pd.read_parquet(args.policy_labels)
    grid = pd.read_parquet(args.label_grid)
    joined = prepare_joined(
        predictions,
        policy,
        grid,
        grid_name=args.grid_name,
    )
    evaluation_frames = {"aggregate": joined}
    for fold_id, sample in joined.groupby("fold_id", sort=True):
        evaluation_frames[str(fold_id)] = sample.copy()
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    for evaluation, evaluation_frame in evaluation_frames.items():
        for scope in ("pooled_global", "side_long", "side_short"):
            sample = (
                evaluation_frame
                if scope == "pooled_global"
                else evaluation_frame.loc[
                    evaluation_frame["side_name"].eq(scope.removeprefix("side_"))
                ]
            )
            selected: dict[str, pd.DataFrame] = {}
            for score_name, score_column in SCORES.items():
                metrics, selected[score_name] = selection_metrics(
                    sample,
                    score_name=score_name,
                    score_column=score_column,
                    evaluation=evaluation,
                    scope=scope,
                )
                metric_rows.append(metrics)
                exits = (
                    selected[score_name]["execution_exit_reason"]
                    .value_counts(normalize=True)
                    .rename_axis("execution_exit_reason")
                    .reset_index(name="rate")
                )
                for row in exits.to_dict("records"):
                    exit_rows.append(
                        {
                            "evaluation": evaluation,
                            "scope": scope,
                            "score_name": score_name,
                            **row,
                        }
                    )
            comparison, replacement = compare_selections(
                selected["within_july_model"],
                selected["frozen_alpha"],
                evaluation=evaluation,
                scope=scope,
            )
            comparison_rows.append(comparison)
            replacement_rows.extend(replacement)
    args.output_dir.mkdir(parents=True)
    paths = {
        "joined": args.output_dir / "within_july_opportunity_capture_rows.parquet",
        "metrics": args.output_dir / "selection_component_metrics.csv",
        "comparisons": args.output_dir / "model_vs_alpha_decomposition.csv",
        "replacements": args.output_dir / "added_dropped_components.csv",
        "exit_reasons": args.output_dir / "selected_exit_reason_rates.csv",
    }
    joined.to_parquet(paths["joined"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(comparison_rows).to_csv(paths["comparisons"], index=False)
    pd.DataFrame(replacement_rows).to_csv(paths["replacements"], index=False)
    pd.DataFrame(exit_rows).to_csv(paths["exit_reasons"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_diagnostic_not_promotion_evidence",
        "contract": {
            "prediction_evidence": "valid forward-expanding within-July OOS only",
            "primary_selection": "one pooled global top10 across timestamps and sides",
            "score_orientation": "higher is better for model and frozen alpha",
            "path_mfe": "diagnostic favorable path excursion, not a guaranteed executable exit",
            "exit_regret": "path MFE return minus exact-policy realized gross return",
            "identity": list(IDENTITY),
            "supporting_barrier_grid": args.grid_name,
        },
        "coverage": {
            "prediction_rows": joined.attrs["prediction_rows"],
            "joined_rows": joined.attrs["joined_rows"],
            "coverage": joined.attrs["coverage"],
            "max_prediction_policy_net_delta": joined.attrs["max_net_delta"],
            "max_gross_cost_net_delta": joined.attrs["max_accounting_delta"],
            "max_grid_policy_net_delta": joined.attrs["max_grid_net_delta"],
        },
        "inputs": {
            "predictions": {
                "path": str(args.predictions),
                "sha256": _sha256(args.predictions),
            },
            "policy_labels": {
                "path": str(args.policy_labels),
                "sha256": _sha256(args.policy_labels),
            },
            "label_grid": {
                "path": str(args.label_grid),
                "sha256": _sha256(args.label_grid),
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "execution_ev_exact_policy_within_july_learnability_20260727_v1/"
            "within_july_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--policy-labels",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
            "execution_ev_policy_labels.parquet"
        ),
    )
    parser.add_argument(
        "--label-grid",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "meaningful_mfe_label_grid.parquet"
        ),
    )
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
