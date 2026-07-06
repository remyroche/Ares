#!/usr/bin/env python3
"""Optimize and materialize economically constrained TBM targets."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.economic_target_optimizer import (
    EconomicTargetSpec,
    append_economic_target_columns,
    build_economic_target,
    candidate_specs,
    economic_target_column_names,
)

from scripts.run_label_quality_proxy_diagnostics import (
    TOP_FRACS,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _proxy_score,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _safe_std,
    _selection_metrics,
    _spearman,
)


DEFAULT_LABELS_DIR = Path(
    "data_perp/artifacts/"
    "20260702_184500_single_head_monthly_walkforward_"
    "bidirectional_sideaware_policy_net_labels/labels"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/economic_target_optimization")


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        out.append(float(text))
    return out


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _sample_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    sample_frac: float,
    random_seed: int,
) -> pd.DataFrame:
    out = frame
    if 0.0 < float(sample_frac) < 1.0:
        out = out.sample(frac=float(sample_frac), random_state=int(random_seed))
    if int(max_rows) > 0 and len(out) > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=int(random_seed))
    if out is not frame:
        out = out.sort_values(["__ts__", "__symbol__"], kind="mergesort")
    return out.reset_index(drop=True)


def _attach_feature_store(
    frame: pd.DataFrame,
    *,
    feature_dir: Path | None,
    feature_list_csv: Path | None,
    max_feature_store_features: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_dir is None or feature_list_csv is None:
        return frame, {"enabled": False, "reason": "not_requested"}
    selected = _read_feature_list(
        feature_list_csv,
        max_features=int(max_feature_store_features) if int(max_feature_store_features) > 0 else None,
    )
    matrix, diagnostics = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected,
    )
    if matrix.empty:
        return frame, diagnostics
    out = frame.copy()
    new_columns: dict[str, pd.Series] = {}
    for column in matrix.columns:
        if column in out.columns:
            left = pd.to_numeric(out[column], errors="coerce")
            right = pd.to_numeric(matrix[column], errors="coerce")
            out[column] = left.where(left.notna(), right).astype(np.float32)
        else:
            new_columns[column] = pd.to_numeric(matrix[column], errors="coerce").astype(
                np.float32
            )
    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
    diagnostics = dict(diagnostics)
    diagnostics["attached_features"] = int(len(matrix.columns))
    return out, diagnostics


def _build_eval_metrics(frame: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    metrics = _path_metrics(frame)
    econ_net = pd.to_numeric(target["__u_econ_net__"], errors="coerce")
    metrics["u_policy_net"] = econ_net
    metrics["ret_net"] = econ_net
    metrics["econ_adjusted_net"] = pd.to_numeric(
        target["__u_econ_adjusted_net__"], errors="coerce"
    )
    metrics["econ_feasible"] = pd.to_numeric(target["__econ_feasible__"], errors="coerce")
    return metrics


def _target_for_helpers(target: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": pd.to_numeric(target["__y_econ_soft__"], errors="coerce"),
            "target_hard": pd.to_numeric(target["__y_econ_bin__"], errors="coerce"),
        },
        index=target.index,
    )


def _aggregate_proxy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "proxy_months": 0,
            "proxy_top10_mean_net": float("nan"),
            "proxy_top10_q10_net": float("nan"),
            "proxy_top10_hit_net": float("nan"),
            "proxy_top10_delta_mean": float("nan"),
            "proxy_top10_ic_soft": float("nan"),
            "proxy_top10_ic_net": float("nan"),
        }
    frame = pd.DataFrame(rows)
    top10 = frame[np.isclose(pd.to_numeric(frame["top_frac"], errors="coerce"), 0.10)]
    if top10.empty:
        top10 = frame
    return {
        "proxy_months": int(top10["period"].nunique()),
        "proxy_top10_mean_net": _safe_mean(top10["mean_u"]),
        "proxy_top10_q10_net": _safe_mean(top10["q10_u"]),
        "proxy_top10_hit_net": _safe_mean(top10["hit_u"]),
        "proxy_top10_delta_mean": _safe_mean(top10["delta_mean_u_vs_period"]),
        "proxy_top10_ic_soft": _safe_mean(top10["proxy_ic_soft"]),
        "proxy_top10_ic_net": _safe_mean(top10["proxy_ic_u"]),
    }


def _proxy_oos_rows_for_spec(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    spec_name: str,
    min_train_rows: int,
    min_valid_rows: int,
) -> list[dict[str, Any]]:
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    if not features:
        return rows
    periods = frame["__ts__"].dt.to_period("M").astype(str)
    for month in months[1:]:
        train_mask = periods < month
        valid_mask = periods == month
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        score, diag = _proxy_score(
            train,
            valid,
            features,
            target.loc[train_mask, "target_soft"],
        )
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
        valid_reset = valid.reset_index(drop=True)
        score = score.reset_index(drop=True)
        period_mean = _safe_mean(valid_metrics["u_policy_net"])
        period_hit = _safe_mean(valid_metrics["u_policy_net"] > 0.0)
        period_q10 = _safe_quantile(valid_metrics["u_policy_net"], 0.10)
        for frac in TOP_FRACS:
            row = _selection_metrics(
                frame=valid_reset,
                metrics=valid_metrics,
                target=valid_target,
                score=score,
                arm=spec_name,
                selector="economic_target_feature_proxy_oos",
                period=str(month),
                top_frac=float(frac),
            )
            row.update(
                {
                    "period_baseline_mean_u": period_mean,
                    "period_baseline_hit_u": period_hit,
                    "period_baseline_q10_u": period_q10,
                    "delta_mean_u_vs_period": (
                        float(row["mean_u"] - period_mean)
                        if math.isfinite(float(row["mean_u"])) and math.isfinite(period_mean)
                        else float("nan")
                    ),
                    "delta_hit_u_vs_period": (
                        float(row["hit_u"] - period_hit)
                        if math.isfinite(float(row["hit_u"])) and math.isfinite(period_hit)
                        else float("nan")
                    ),
                    "delta_q10_u_vs_period": (
                        float(row["q10_u"] - period_q10)
                        if math.isfinite(float(row["q10_u"])) and math.isfinite(period_q10)
                        else float("nan")
                    ),
                    "proxy_ic_soft": _spearman(score, valid_target["target_soft"]),
                    "proxy_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "proxy_features": ",".join(diag.get("proxy_features", [])),
                    "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                    "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
                }
            )
            rows.append(row)
    return rows


def _oracle_rows_for_spec(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    spec_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frac in TOP_FRACS:
        rows.append(
            _selection_metrics(
                frame=frame,
                metrics=metrics,
                target=target,
                score=target["target_soft"],
                arm=spec_name,
                selector="target_oracle",
                period="all",
                top_frac=float(frac),
            )
        )
    return rows


def _score_candidate(row: dict[str, Any], args: argparse.Namespace) -> float:
    hard_rate = float(row.get("hard_rate", float("nan")))
    feasible_rate = float(row.get("feasible_rate", float("nan")))
    soft_std = float(row.get("soft_std", float("nan")))
    proxy_mean = float(row.get("proxy_top10_mean_net", float("nan")))
    proxy_delta = float(row.get("proxy_top10_delta_mean", float("nan")))
    proxy_q10 = float(row.get("proxy_top10_q10_net", float("nan")))
    proxy_hit = float(row.get("proxy_top10_hit_net", float("nan")))
    proxy_ic_net = float(row.get("proxy_top10_ic_net", float("nan")))
    proxy_ic = float(row.get("proxy_top10_ic_soft", float("nan")))
    oracle_mean = float(row.get("oracle_top10_mean_net", float("nan")))
    feature_ic = float(row.get("feature_top_abs_ic", float("nan")))

    if not math.isfinite(hard_rate) or not math.isfinite(feasible_rate):
        return float("-inf")
    if hard_rate < float(args.min_hard_rate) or hard_rate > float(args.max_hard_rate):
        return float("-inf")
    if feasible_rate < float(args.min_feasible_rate):
        return float("-inf")
    if not math.isfinite(soft_std) or soft_std < float(args.min_soft_std):
        return float("-inf")

    proxy_learnable = (
        math.isfinite(proxy_ic)
        and proxy_ic >= float(args.min_proxy_ic_soft)
        and (not math.isfinite(proxy_delta) or proxy_delta >= float(args.min_proxy_delta))
    )
    if bool(args.require_proxy_positive_net):
        if not math.isfinite(proxy_mean) or proxy_mean < float(args.min_proxy_mean_net):
            return float("-inf")
        if not math.isfinite(proxy_ic_net) or proxy_ic_net < float(args.min_proxy_ic_net):
            return float("-inf")
        if math.isfinite(proxy_hit) and proxy_hit < float(args.min_proxy_hit_net):
            return float("-inf")
        if math.isfinite(proxy_q10) and proxy_q10 < float(args.min_proxy_q10_net):
            return float("-inf")
        if math.isfinite(proxy_delta) and proxy_delta < float(args.min_proxy_delta):
            return float("-inf")
    if math.isfinite(proxy_mean) and proxy_mean > 0.0:
        economic_mean = proxy_mean
    elif (
        not bool(args.require_proxy_positive_net)
        and proxy_learnable
        and math.isfinite(oracle_mean)
        and oracle_mean > 0.0
    ):
        economic_mean = 0.50 * oracle_mean + 0.50 * max(proxy_delta, 0.0)
    else:
        proxy_months = int(row.get("proxy_months", 0) or 0)
        economic_mean = (
            oracle_mean
            if proxy_months <= 0 and math.isfinite(oracle_mean)
            else float("nan")
        )
    if not math.isfinite(economic_mean) or economic_mean <= 0.0:
        return float("-inf")

    q10_penalty = (
        max(0.0, -proxy_q10)
        if bool(args.require_proxy_positive_net) and math.isfinite(proxy_q10)
        else 0.0
    )
    delta_bonus = 25.0 * max(proxy_delta, 0.0) if math.isfinite(proxy_delta) else 0.0
    ic_bonus = 2.0 * max(proxy_ic, 0.0) if math.isfinite(proxy_ic) else 0.0
    feature_bonus = 1.0 * max(feature_ic, 0.0) if math.isfinite(feature_ic) else 0.0
    balance = 1.0 - min(abs(hard_rate - 0.15) / 0.15, 1.0)
    return (
        100.0 * economic_mean
        + delta_bonus
        + ic_bonus
        + feature_bonus
        + 0.25 * feasible_rate
        + 0.25 * balance
        - 20.0 * q10_penalty
    )


def _evaluate_spec(
    *,
    frame: pd.DataFrame,
    spec: EconomicTargetSpec,
    features: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_target, target_summary = build_economic_target(frame, spec)
    target = _target_for_helpers(raw_target)
    metrics = _build_eval_metrics(frame, raw_target)
    oracle = _oracle_rows_for_spec(
        frame=frame,
        metrics=metrics,
        target=target,
        spec_name=spec.name,
    )
    oracle_top10 = next(
        (row for row in oracle if np.isclose(float(row.get("top_frac", 0.0)), 0.10)),
        oracle[0] if oracle else {},
    )
    proxy_rows = _proxy_oos_rows_for_spec(
        frame=frame,
        metrics=metrics,
        target=target,
        features=features,
        spec_name=spec.name,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
    )
    proxy_summary = _aggregate_proxy_rows(proxy_rows)
    soft = target["target_soft"]
    net = metrics["u_policy_net"]
    hard = target["target_hard"]
    feature_ic = pd.Series(dtype=float)
    if features:
        top_abs: list[float] = []
        for feature in features:
            ic = _spearman(frame[feature], soft)
            if math.isfinite(ic):
                top_abs.append(abs(ic))
        feature_ic = pd.Series(top_abs, dtype=float).sort_values(ascending=False)

    row = {
        "candidate": spec.name,
        **spec.to_dict(),
        "rows": int(len(frame)),
        "finite_soft_frac": float(soft.notna().mean()) if len(soft) else float("nan"),
        "soft_mean": _safe_mean(soft),
        "soft_std": _safe_std(soft),
        "soft_p10": _safe_quantile(soft, 0.10),
        "soft_p90": _safe_quantile(soft, 0.90),
        "hard_rate": _safe_mean(hard),
        "feasible_rate": float(target_summary.get("feasible_rate", float("nan"))),
        "mean_net_utility": _safe_mean(net),
        "median_net_utility": _safe_quantile(net, 0.50),
        "p90_net_utility": _safe_quantile(net, 0.90),
        "ic_soft_vs_net": _spearman(soft, net),
        "ic_soft_vs_adjusted_net": _spearman(soft, metrics["econ_adjusted_net"]),
        "oracle_top10_mean_net": float(oracle_top10.get("mean_u", float("nan"))),
        "oracle_top10_q10_net": float(oracle_top10.get("q10_u", float("nan"))),
        "oracle_top10_hit_net": float(oracle_top10.get("hit_u", float("nan"))),
        "feature_count": int(len(features)),
        "feature_top_abs_ic": float(feature_ic.iloc[0]) if len(feature_ic) else float("nan"),
        "feature_mean_top_abs_ic": float(feature_ic.head(8).mean()) if len(feature_ic) else float("nan"),
        **proxy_summary,
    }
    row["objective"] = _score_candidate(row, args)
    return row, proxy_rows


def _candidate_grid(args: argparse.Namespace) -> list[EconomicTargetSpec]:
    specs = candidate_specs(
        utility_sources=_parse_str_list(args.utility_sources),
        margins=_parse_float_list(args.margins),
        vol_sources=_parse_str_list(args.vol_sources),
        costs=_parse_float_list(args.costs),
        sl_buffer=float(args.sl_buffer),
        temperatures=_parse_float_list(args.temperatures),
        mae_penalties=_parse_float_list(args.mae_penalties),
        timeout_penalties=_parse_float_list(args.timeout_penalties),
    )
    max_candidates = int(args.max_candidates)
    if max_candidates > 0 and len(specs) > max_candidates:
        rng = np.random.default_rng(int(args.random_seed))
        keep = np.sort(rng.choice(np.arange(len(specs)), size=max_candidates, replace=False))
        specs = [specs[int(i)] for i in keep]
    return specs


def _read_manifest(labels_dir: Path) -> dict[str, Any]:
    path = labels_dir / "labels_manifest.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"datasets": {}}


def _dataset_files(labels_dir: Path, manifest: dict[str, Any]) -> list[tuple[str, Path, dict[str, Any]]]:
    datasets = manifest.get("datasets", {}) if isinstance(manifest, dict) else {}
    rows: list[tuple[str, Path, dict[str, Any]]] = []
    if isinstance(datasets, dict) and datasets:
        for name, meta in datasets.items():
            if not isinstance(meta, dict):
                continue
            file_name = str(meta.get("file") or "")
            if not file_name.endswith(".parquet"):
                continue
            path = labels_dir / file_name
            if path.exists():
                rows.append((str(name), path, dict(meta)))
    if rows:
        return rows
    return [(path.stem, path, {"file": path.name}) for path in sorted(labels_dir.glob("*.parquet"))]


def _materialize_selected(
    *,
    labels_dir: Path,
    output_labels_dir: Path,
    spec: EconomicTargetSpec,
    optimizer_summary: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    labels_dir = labels_dir.resolve()
    output_labels_dir = output_labels_dir.resolve()
    if labels_dir == output_labels_dir and not overwrite:
        raise RuntimeError(
            "Refusing to materialize economic targets in-place without --overwrite-labels."
        )
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(labels_dir)
    out_manifest = dict(manifest)
    out_manifest["economic_target_optimization"] = optimizer_summary
    out_manifest["economic_target_selected_spec"] = spec.to_dict()
    out_manifest["economic_target_columns"] = economic_target_column_names()
    out_manifest["source_labels_dir"] = str(labels_dir)
    out_manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out_manifest.setdefault("datasets", {})

    materialized: list[dict[str, Any]] = []
    for dataset_name, source_path, meta in _dataset_files(labels_dir, manifest):
        df = pd.read_parquet(source_path)
        out_df, target_summary = append_economic_target_columns(df, spec, copy=True)
        output_path = output_labels_dir / source_path.name
        out_df.to_parquet(output_path, index=False)
        columns = list(meta.get("columns", []))
        for column in economic_target_column_names():
            if column not in columns:
                columns.append(column)
        out_meta = dict(meta)
        out_meta["file"] = source_path.name
        out_meta["rows"] = int(len(out_df))
        out_meta["columns"] = columns
        out_manifest["datasets"][dataset_name] = out_meta
        materialized.append(
            {
                "dataset": dataset_name,
                "source_file": str(source_path),
                "output_file": str(output_path),
                "rows": int(len(out_df)),
                "target_summary": target_summary,
            }
        )

    with (output_labels_dir / "labels_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(out_manifest), handle, indent=2, sort_keys=True)
    with (output_labels_dir / "economic_target_materialization_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_json_safe({"datasets": materialized, "selected_spec": spec.to_dict()}), handle, indent=2, sort_keys=True)
    return {"output_labels_dir": str(output_labels_dir), "datasets": materialized}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-labels-dir", type=Path, default=None)
    parser.add_argument("--overwrite-labels", action="store_true")
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--feature-list-csv", type=Path, default=None)
    parser.add_argument("--max-feature-store-features", type=int, default=240)
    parser.add_argument("--max-feature-columns", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--costs", default="0.01")
    parser.add_argument("--margins", default="0.0025,0.005,0.0075,0.01,0.015,0.02")
    parser.add_argument("--sl-buffer", type=float, default=1.2)
    parser.add_argument("--utility-sources", default="y_ret,conservative,policy_net,path_adjusted")
    parser.add_argument("--vol-sources", default="barrier,sl,max_sl_barrier")
    parser.add_argument("--temperatures", default="0.5,0.75")
    parser.add_argument("--mae-penalties", default="0,0.25")
    parser.add_argument("--timeout-penalties", default="0,0.5")
    parser.add_argument("--max-candidates", type=int, default=360)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--min-valid-rows", type=int, default=250)
    parser.add_argument("--min-hard-rate", type=float, default=0.005)
    parser.add_argument("--max-hard-rate", type=float, default=0.60)
    parser.add_argument("--min-feasible-rate", type=float, default=0.10)
    parser.add_argument("--min-soft-std", type=float, default=0.02)
    parser.add_argument("--min-proxy-ic-soft", type=float, default=0.02)
    parser.add_argument("--min-proxy-delta", type=float, default=0.0)
    parser.add_argument("--min-proxy-mean-net", type=float, default=0.0)
    parser.add_argument("--min-proxy-ic-net", type=float, default=0.0)
    parser.add_argument("--min-proxy-hit-net", type=float, default=0.0)
    parser.add_argument("--min-proxy-q10-net", type=float, default=float("-inf"))
    parser.add_argument("--require-proxy-positive-net", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_labels(args.labels_dir)
    labels = _sample_frame(
        labels,
        max_rows=int(args.max_rows),
        sample_frac=float(args.sample_frac),
        random_seed=int(args.random_seed),
    )
    labels, feature_store_diag = _attach_feature_store(
        labels,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=int(args.max_feature_store_features),
    )
    features = _feature_columns(labels)
    if int(args.max_feature_columns) > 0:
        features = features[: int(args.max_feature_columns)]
    specs = _candidate_grid(args)
    if not specs:
        raise RuntimeError("Candidate grid is empty.")

    candidate_rows: list[dict[str, Any]] = []
    proxy_rows_all: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        row, proxy_rows = _evaluate_spec(
            frame=labels,
            spec=spec,
            features=features,
            args=args,
        )
        row["candidate_index"] = int(idx)
        candidate_rows.append(row)
        proxy_rows_all.extend(proxy_rows)
        if idx == 1 or idx % 25 == 0 or idx == len(specs):
            print(
                json.dumps(
                    {
                        "progress": f"{idx}/{len(specs)}",
                        "candidate": spec.name,
                        "objective": row.get("objective"),
                        "hard_rate": row.get("hard_rate"),
                        "feasible_rate": row.get("feasible_rate"),
                    },
                    sort_keys=True,
                )
            )

    candidate_path = args.output_dir / "candidate_summary.csv"
    proxy_path = args.output_dir / "proxy_oos_rows.csv"
    selected_path = args.output_dir / "selected_target_config.json"
    summary_path = args.output_dir / "economic_target_optimization_summary.json"

    summary_frame = pd.DataFrame(candidate_rows)
    summary_frame = summary_frame.sort_values("objective", ascending=False, na_position="last")
    summary_frame.to_csv(candidate_path, index=False)
    pd.DataFrame(proxy_rows_all).to_csv(proxy_path, index=False)
    finite_objective = summary_frame[np.isfinite(pd.to_numeric(summary_frame["objective"], errors="coerce"))]
    if finite_objective.empty:
        raise RuntimeError(
            f"No candidate passed economic target gates. Inspect {candidate_path}."
        )
    selected_row = finite_objective.iloc[0].to_dict()
    selected_spec = EconomicTargetSpec(
        name=str(selected_row["candidate"]),
        utility_source=str(selected_row["utility_source"]),
        cost=float(selected_row["cost"]),
        margin=float(selected_row["margin"]),
        sl_buffer=float(selected_row["sl_buffer"]),
        vol_source=str(selected_row["vol_source"]),
        temperature=float(selected_row["temperature"]),
        clip_abs=float(selected_row["clip_abs"]),
        mae_penalty=float(selected_row["mae_penalty"]),
        timeout_penalty=float(selected_row["timeout_penalty"]),
        min_vol=float(selected_row["min_vol"]),
    )

    optimizer_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_dir": str(args.labels_dir),
        "rows_evaluated": int(len(labels)),
        "candidate_count": int(len(specs)),
        "feature_count": int(len(features)),
        "features": features,
        "feature_store": feature_store_diag,
        "horizon_hours": [3, 5, 7],
        "cost_contract": {
            "round_trip_cost_fraction": _parse_float_list(args.costs),
            "margin_plus_cost_gt_sl_buffer_times_sl": float(args.sl_buffer),
        },
        "selection_gates": {
            "min_hard_rate": float(args.min_hard_rate),
            "max_hard_rate": float(args.max_hard_rate),
            "min_feasible_rate": float(args.min_feasible_rate),
            "min_soft_std": float(args.min_soft_std),
            "min_proxy_ic_soft": float(args.min_proxy_ic_soft),
            "min_proxy_delta": float(args.min_proxy_delta),
            "require_proxy_positive_net": bool(args.require_proxy_positive_net),
            "min_proxy_mean_net": float(args.min_proxy_mean_net),
            "min_proxy_ic_net": float(args.min_proxy_ic_net),
            "min_proxy_hit_net": float(args.min_proxy_hit_net),
            "min_proxy_q10_net": float(args.min_proxy_q10_net),
        },
        "selected": selected_row,
        "selected_spec": selected_spec.to_dict(),
        "candidate_summary_csv": str(candidate_path),
        "proxy_oos_csv": str(proxy_path),
    }

    materialized = None
    if args.output_labels_dir is not None:
        materialized = _materialize_selected(
            labels_dir=args.labels_dir,
            output_labels_dir=args.output_labels_dir,
            spec=selected_spec,
            optimizer_summary=optimizer_summary,
            overwrite=bool(args.overwrite_labels),
        )
        optimizer_summary["materialized"] = materialized

    with selected_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(selected_spec.to_dict()), handle, indent=2, sort_keys=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(optimizer_summary), handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            _json_safe(
                {
                    "selected_spec": selected_spec.to_dict(),
                    "selected_objective": selected_row.get("objective"),
                    "candidate_summary_csv": str(candidate_path),
                    "summary_json": str(summary_path),
                    "materialized": materialized,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
