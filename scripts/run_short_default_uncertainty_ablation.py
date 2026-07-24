#!/usr/bin/env python3
"""Test train-OOF-selected uncertainty context on the V11 short-default overlay."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts import run_meta_residual_event_balanced_error_overlay as v11


GROUP = ("short", "short_default_clean_path")
SHORT_BREAKOUT = ("short", "short_breakout_precision")
TOP10 = 0.90
RISK_COLUMNS = (
    "ensemble_risk_mean",
    "ensemble_risk_std",
    "neighbor_shrunken_adverse_rate",
    "neighbor_weighted_ev_mean",
    "neighbor_weighted_ev_std",
    "neighbor_effective_count",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _percentile(values: np.ndarray, reference: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    finite = np.sort(reference[np.isfinite(reference)])
    if not len(finite):
        result = np.full(len(values), 0.5, dtype=np.float32)
    else:
        result = (
            np.searchsorted(finite, values, side="right") / float(len(finite))
        ).astype(np.float32)
    return 1.0 - result if reverse else result


def _add_uncertainty_components(
    train: pd.DataFrame, score: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    score = score.copy()
    orientation = {
        "ensemble_risk_mean": False,
        "ensemble_risk_std": False,
        "neighbor_shrunken_adverse_rate": False,
        "neighbor_weighted_ev_mean": True,
        "neighbor_weighted_ev_std": False,
        "neighbor_effective_count": True,
    }
    for name, reverse in orientation.items():
        reference = pd.to_numeric(train[name], errors="coerce").to_numpy(np.float32)
        output = f"uncertainty__{name}"
        train[output] = _percentile(reference, reference, reverse=reverse)
        values = pd.to_numeric(score[name], errors="coerce").to_numpy(np.float32)
        score[output] = _percentile(values, reference, reverse=reverse)
    return train, score


def _weight_templates() -> dict[str, np.ndarray]:
    # Order matches RISK_COLUMNS. These are deliberately few and interpretable;
    # the OOF search tunes only the family and penalty geometry.
    return {
        "requested_core_equal": np.array([0, 1, 1, 0, 1, 0], np.float32) / 3,
        "disagreement_neighbor": np.array([0, 1, 2, 0, 1, 0], np.float32) / 4,
        "risk_disagreement_neighbor": np.array([1, 1, 2, 0, 1, 0], np.float32) / 5,
        "risk_ev_context": np.array([1, 1, 1, 1, 1, 0], np.float32) / 5,
        "support_aware": np.array([1, 1, 2, 1, 1, 1], np.float32) / 7,
        "neighbor_only": np.array([0, 0, 2, 1, 1, 1], np.float32) / 5,
    }


def _uncertainty(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    columns = [f"uncertainty__{name}" for name in RISK_COLUMNS]
    matrix = frame[columns].to_numpy(np.float32, copy=False)
    return np.nan_to_num(matrix, nan=0.5) @ weights


def _adjust_rank(
    rank: np.ndarray, uncertainty: np.ndarray, threshold: float, alpha: float
) -> np.ndarray:
    result = rank.astype(np.float32, copy=True)
    intensity = np.clip(
        (uncertainty - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    result -= np.float32(alpha) * intensity
    return np.clip(result, 0.0, 1.0)


def _metrics(frame: pd.DataFrame, rank: np.ndarray) -> dict[str, float]:
    selected = rank >= TOP10
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(np.float32)
    clean = pd.to_numeric(frame["clean_exec"], errors="coerce").to_numpy(np.float32)
    month = frame["__ts__"].dt.strftime("%Y-%m").to_numpy()
    monthly = [float(np.nanmean(ev[selected & (month == value)])) for value in np.unique(month) if (selected & (month == value)).any()]
    return {
        "selected_rows": int(selected.sum()),
        "mean_ev": float(np.nanmean(ev[selected])) if selected.any() else np.nan,
        "sum_ev": float(np.nansum(ev[selected])),
        "positive_ev_rate": float(np.nanmean(ev[selected] > 0)) if selected.any() else np.nan,
        "clean_precision": float(np.nanmean(clean[selected])) if selected.any() else np.nan,
        "mean_month_ev": float(np.mean(monthly)) if monthly else np.nan,
        "std_month_ev": float(np.std(monthly)) if monthly else np.nan,
        "worst_month_ev": float(np.min(monthly)) if monthly else np.nan,
    }


def _accepted(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame = pd.read_csv(path)
    return {
        (str(row.side_name), str(row.archetype_policy_key)): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _merge_diagnostics(rows: pd.DataFrame, diagnostics: pd.DataFrame, stage: str) -> pd.DataFrame:
    keys = ["__ts__", "side_name", "archetype_policy_key"]
    diagnostic_columns = [*RISK_COLUMNS, "neighbor_distance_percentile"]
    local = diagnostics.loc[
        diagnostics["stage"].eq(stage), keys + diagnostic_columns
    ].drop_duplicates(keys, keep="last")
    result = rows.merge(local, on=keys, how="left", validate="many_to_one")
    return result


def _short_breakout_inversion(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, local in diagnostics.loc[
        diagnostics["side_name"].eq(SHORT_BREAKOUT[0])
        & diagnostics["archetype_policy_key"].eq(SHORT_BREAKOUT[1])
    ].groupby("stage", observed=True):
        y = local[v11.TARGET].to_numpy(np.int8)
        dirty = (1 - local["clean_exec"].to_numpy(np.int8)).astype(np.int8)
        negative_ev = local["ev_after_1pct"].le(0.0).to_numpy(np.int8)
        risk = local["ensemble_risk_mean"].to_numpy(np.float32)
        if len(np.unique(y)) < 2:
            continue
        for name, score in (("risk", risk), ("inverted_risk", 1.0 - risk)):
            cutoff = float(np.quantile(score, 0.90))
            selected = score >= cutoff
            rows.append(
                {
                    "stage": stage,
                    "score": name,
                    "rows": len(local),
                    "adverse_prevalence": float(y.mean()),
                    "roc_auc": float(roc_auc_score(y, score)),
                    "average_precision": float(average_precision_score(y, score)),
                    "dirty_path_auc": float(roc_auc_score(dirty, score))
                    if len(np.unique(dirty)) > 1 else np.nan,
                    "negative_ev_auc": float(roc_auc_score(negative_ev, score))
                    if len(np.unique(negative_ev)) > 1 else np.nan,
                    "top10_adverse_rate": float(y[selected].mean()),
                    "top10_lift": float(y[selected].mean() / max(y.mean(), 1e-8)),
                    "top10_mean_ev": float(local.loc[selected, "ev_after_1pct"].mean()),
                    "top10_clean_rate": float(local.loc[selected, "clean_exec"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _predictability_labels(train: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    result: list[pd.DataFrame] = []
    for stage, frame in (("train_oof", train), ("eval_oos", score)):
        elevated_model = frame["uncertainty__ensemble_risk_mean"].ge(0.75)
        elevated_neighbor = frame["uncertainty__neighbor_shrunken_adverse_rate"].ge(0.75)
        stable = frame["uncertainty__ensemble_risk_std"].lt(0.75)
        supported = frame["uncertainty__neighbor_effective_count"].lt(0.75)
        recurring = frame["neighbor_distance_percentile"].lt(0.90)
        distinguishable = elevated_model & elevated_neighbor & stable & supported & recurring
        adverse = frame[v11.TARGET].astype(bool)
        part = frame.loc[:, ["__ts__", "side_name", "archetype_policy_key", v11.TARGET]].copy()
        part["stage"] = stage
        part["historically_distinguishable"] = distinguishable.to_numpy(np.int8)
        part["predictable_adverse"] = (adverse & distinguishable).to_numpy(np.int8)
        part["ambiguous_or_exogenous_adverse"] = (adverse & ~distinguishable).to_numpy(np.int8)
        result.append(part)
    return pd.concat(result, ignore_index=True, copy=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_parquet(args.diagnostics)
    diagnostics["__ts__"] = pd.to_datetime(diagnostics["__ts__"], utc=True)
    train = pd.read_parquet(args.v11_dir / "train_oof_predictions.parquet")
    evaluated = pd.read_parquet(args.v11_dir / "oos_predictions.parquet")
    for frame in (train, evaluated):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    train = _merge_diagnostics(train, diagnostics, "train_oof")
    evaluated = _merge_diagnostics(evaluated, diagnostics, "eval_oos")
    accepted = _accepted(args.v11_dir / "accepted_local_overlays.csv")
    train_rank, _ = v11._apply_selected_overlays(train, accepted, "parent_rank_v9")
    train["v11_rank"] = train_rank
    evaluated["v11_rank"] = evaluated["parent_rank_v9_residual_error_overlay"].to_numpy(np.float32)

    local_train = train.loc[
        train["side_name"].eq(GROUP[0]) & train["archetype_policy_key"].eq(GROUP[1])
    ].dropna(subset=list(RISK_COLUMNS))
    local_eval = evaluated.loc[
        evaluated["side_name"].eq(GROUP[0]) & evaluated["archetype_policy_key"].eq(GROUP[1])
    ].dropna(subset=list(RISK_COLUMNS))
    local_train, local_eval = _add_uncertainty_components(local_train, local_eval)

    base_train = _metrics(local_train, local_train["v11_rank"].to_numpy(np.float32))
    non_march = ~local_train["__ts__"].dt.strftime("%Y-%m").eq("2026-03")
    base_non_march = _metrics(
        local_train.loc[non_march], local_train.loc[non_march, "v11_rank"].to_numpy(np.float32)
    )
    search_rows: list[dict[str, Any]] = []
    for family, weights in _weight_templates().items():
        uncertainty = _uncertainty(local_train, weights)
        for threshold, alpha in itertools.product((0.65, 0.75, 0.85), (0.01, 0.02, 0.04, 0.06)):
            rank = _adjust_rank(local_train["v11_rank"].to_numpy(np.float32), uncertainty, threshold, alpha)
            metrics = _metrics(local_train, rank)
            non_march_metrics = _metrics(local_train.loc[non_march], rank[non_march.to_numpy()])
            activity = metrics["selected_rows"] / max(base_train["selected_rows"], 1)
            promotable = (
                activity >= 0.95
                and metrics["mean_ev"] > base_train["mean_ev"]
                and metrics["sum_ev"] >= base_train["sum_ev"]
                and metrics["clean_precision"] >= base_train["clean_precision"]
                and non_march_metrics["mean_ev"] >= base_non_march["mean_ev"]
                and non_march_metrics["sum_ev"] >= base_non_march["sum_ev"]
            )
            objective = (
                metrics["mean_month_ev"]
                - 0.5 * metrics["std_month_ev"]
                + 0.25 * metrics["worst_month_ev"]
                + 0.25 * (metrics["mean_ev"] - base_train["mean_ev"])
            )
            search_rows.append({
                "family": family, "threshold": threshold, "alpha": alpha,
                "activity_ratio": activity, "promotable": promotable,
                "objective": objective,
                "non_march_mean_ev_delta": non_march_metrics["mean_ev"] - base_non_march["mean_ev"],
                "non_march_sum_ev_delta": non_march_metrics["sum_ev"] - base_non_march["sum_ev"],
                **metrics,
            })
    search = pd.DataFrame(search_rows).sort_values(
        ["promotable", "objective"], ascending=[False, False], kind="stable"
    )
    search.to_csv(args.output / "train_oof_uncertainty_search.csv", index=False)
    base_eval = _metrics(local_eval, local_eval["v11_rank"].to_numpy(np.float32))
    sensitivity_rows: list[dict[str, Any]] = []
    for row in search.loc[search["promotable"]].itertuples(index=False):
        candidate_uncertainty = _uncertainty(
            local_eval, _weight_templates()[str(row.family)]
        )
        candidate_rank = _adjust_rank(
            local_eval["v11_rank"].to_numpy(np.float32),
            candidate_uncertainty,
            float(row.threshold),
            float(row.alpha),
        )
        candidate_metrics = _metrics(local_eval, candidate_rank)
        sensitivity_rows.append(
            {
                "family": row.family,
                "threshold": row.threshold,
                "alpha": row.alpha,
                "train_oof_objective": row.objective,
                "eval_activity_ratio": candidate_metrics["selected_rows"]
                / max(base_eval["selected_rows"], 1),
                "eval_mean_ev_delta": candidate_metrics["mean_ev"]
                - base_eval["mean_ev"],
                "eval_sum_ev_delta": candidate_metrics["sum_ev"]
                - base_eval["sum_ev"],
                "eval_clean_precision_delta": candidate_metrics["clean_precision"]
                - base_eval["clean_precision"],
                **{f"eval_{key}": value for key, value in candidate_metrics.items()},
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output / "oos_train_promotable_sensitivity.csv", index=False
    )
    best = search.iloc[0].to_dict()
    weights = _weight_templates()[str(best["family"])]
    eval_uncertainty = _uncertainty(local_eval, weights)

    evaluated["short_default_uncertainty_score"] = np.float32(0.0)
    evaluated["v11_short_default_uncertainty_rank"] = evaluated["v11_rank"].to_numpy(np.float32)
    evaluated.loc[local_eval.index, "short_default_uncertainty_score"] = eval_uncertainty
    evaluated.loc[local_eval.index, "v11_short_default_uncertainty_rank"] = _adjust_rank(
        local_eval["v11_rank"].to_numpy(np.float32), eval_uncertainty,
        float(best["threshold"]), float(best["alpha"]),
    )
    evaluated.to_parquet(args.output / "oos_predictions_diagnostic.parquet", index=False, compression="zstd")

    metric_rows: list[dict[str, Any]] = []
    for scope, frame in (("global", evaluated), ("short_default", evaluated.loc[evaluated["side_name"].eq(GROUP[0]) & evaluated["archetype_policy_key"].eq(GROUP[1])])):
        for selector, column in (("v11", "v11_rank"), ("v11_plus_uncertainty", "v11_short_default_uncertainty_rank")):
            metric_rows.append({"scope": scope, "selector": selector, **_metrics(frame, frame[column].to_numpy(np.float32))})
            for month, month_frame in frame.groupby(frame["__ts__"].dt.strftime("%Y-%m"), observed=True):
                metric_rows.append({"scope": f"{scope}::{month}", "selector": selector, **_metrics(month_frame, month_frame[column].to_numpy(np.float32))})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "oos_replication_metrics.csv", index=False)

    quintile_frame = local_eval.copy()
    quintile_frame["uncertainty_score"] = eval_uncertainty
    quintile_frame["uncertainty_quintile"] = pd.qcut(
        quintile_frame["uncertainty_score"], 5, labels=False, duplicates="drop"
    )
    quintiles = quintile_frame.groupby("uncertainty_quintile", observed=True).agg(
        rows=("ev_after_1pct", "size"), mean_ev=("ev_after_1pct", "mean"),
        clean_precision=("clean_exec", "mean"), adverse_rate=(v11.TARGET, "mean"),
        uncertainty_mean=("uncertainty_score", "mean"),
    ).reset_index()
    quintiles.to_csv(args.output / "oos_uncertainty_quintiles.csv", index=False)

    inversion = _short_breakout_inversion(diagnostics)
    inversion.to_csv(args.output / "short_breakout_score_inversion.csv", index=False)
    labels = _predictability_labels(local_train, local_eval)
    labels.to_parquet(args.output / "predictable_vs_ambiguous_adverse.parquet", index=False, compression="zstd")

    manifest = {
        "schema": "short_default_uncertainty_ablation_v1",
        "status": "diagnostic_only_not_activated",
        "selected_train_oof_configuration": best,
        "selected_weights": dict(zip(RISK_COLUMNS, weights.tolist(), strict=True)),
        "train_oof_rows": len(local_train), "eval_oos_rows": len(local_eval),
        "evaluation_period": [str(evaluated["__ts__"].min()), str(evaluated["__ts__"].max())],
        "evaluation_status": "OOS replication; April-June has prior research exposure and is not a new untouched test.",
        "leakage_contract": "Weights, component percentiles, threshold, and alpha are selected from chronological train OOF diagnostics only. April-June receives frozen transforms and parameters. Two-day purging is inherited from the distinguishability artifact.",
    }
    (args.output / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print("\nShort-breakout inversion\n", inversion.to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=Path("data_perp/reports/residual_distinguishability_20260713_v4_weighted_neighbors/state_distinguishability_predictions.parquet"))
    parser.add_argument("--v11-dir", type=Path, default=Path("data_perp/reports/meta_residual_event_balanced_error_overlay_20260713_v11_predicted_damage"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/short_default_uncertainty_ablation_20260713_v1"))
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
