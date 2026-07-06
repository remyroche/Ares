#!/usr/bin/env python3
"""No-training dual-proxy selector for clean first-touch label candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_first_touch_label_ablation import (  # noqa: E402
    DEFAULT_CHALLENGER_PREFIX,
    DEFAULT_METRIC_PREFIX,
    DEFAULT_OUTPUT_DIR as CLEAN_ABLATION_OUTPUT_DIR,
    DEFAULT_PRIMARY_PREFIX,
    _build_arms,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _proxy_score,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
    _spearman,
)
from scripts.run_materialized_label_column_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_PATH,
    _execution_metrics,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/clean_first_touch_dual_proxy_selector_stage166_v1")
DEFAULT_UTILITY_ARMS = (
    "B_stage164_primary_support",
    "C_support_clean_rank_blend",
    "C_timeout_mae_penalized",
    "C_clean_rank_net",
    "D_low_mae_clean",
    "D_low_mae_rank_blend",
)
DEFAULT_RISK_ARMS = (
    "B_stage164_challenger_support",
    "C_timeout_mae_penalized",
    "C_clean_rank_net",
    "D_low_mae_clean",
    "D_low_mae_rank_blend",
    "D_support_low_mae",
    "E_tail_veto_low_mae_clean",
    "E_tail_veto_support_lowmae",
    "E_tail_veto_rank_margin",
    "F_fullpath_veto_support",
    "F_fullpath_rank_margin",
)
DEFAULT_GATE_FRACS = (0.03, 0.05, 0.10, 0.20)
DEFAULT_TOP_FRACS = (0.01, 0.03)


def _effective_n(values: Any) -> float:
    counts = pd.Series(values, dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return float(1.0 / denom) if denom > 0.0 else 0.0


def _rank_top(score: pd.Series, k: int) -> np.ndarray:
    values = _safe_numeric(score)
    valid = values.notna().to_numpy()
    if not bool(valid.any()) or k <= 0:
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(int(k), int(len(valid_idx)))
    order = np.argsort(-values.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _selection_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_idx: np.ndarray,
    utility_arm: str,
    risk_arm: str,
    period: str,
    gate_frac: float,
    top_frac: float,
    utility_proxy_ic_soft: float,
    risk_proxy_ic_soft: float,
    utility_proxy_ic_net: float,
    risk_proxy_ic_net: float,
) -> dict[str, Any]:
    selected = np.asarray(selected_idx, dtype=np.int64)
    selected_metrics = metrics.iloc[selected] if len(selected) else metrics.iloc[:0]
    selected_frame = frame.iloc[selected] if len(selected) else frame.iloc[:0]
    side = _safe_numeric(selected_metrics.get("side")).fillna(1.0)
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    mae = selected_metrics.get("first_touch_mae_to_sl", pd.Series(dtype=float))
    full_path_mae = _safe_numeric(
        selected_frame.get("__first_touch_full_path_mae_to_sl__", mae),
    ).reindex(selected_frame.index)
    net = selected_metrics.get("first_touch_net", pd.Series(dtype=float))
    return {
        "utility_arm": str(utility_arm),
        "risk_arm": str(risk_arm),
        "selector": "risk_gate_then_utility_proxy",
        "period": str(period),
        "gate_frac": float(gate_frac),
        "top_frac": float(top_frac),
        "rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selected_long_rows": int((side > 0.0).sum()),
        "selected_short_rows": int((side < 0.0).sum()),
        "mean_first_touch_net": _safe_mean(net),
        "q10_first_touch_net": _safe_quantile(net, 0.10),
        "hit_first_touch_net": _safe_mean(net > 0.0),
        "clean_first_touch_exec_rate": _safe_mean(selected_metrics.get("clean_first_touch_exec")),
        "first_touch_timeout_rate": _safe_mean(selected_metrics.get("first_touch_timeout").astype(float)),
        "bad_first_touch_mae_to_sl_rate": _safe_mean(mae >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(mae, 0.90),
        "bad_full_path_mae_3r_rate": _safe_mean(full_path_mae >= 3.0),
        "p90_full_path_mae_to_sl": _safe_quantile(full_path_mae, 0.90),
        "mean_barrier": _safe_mean(selected_metrics.get("barrier")),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics.get("barrier") > 0.025),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "utility_proxy_ic_soft": utility_proxy_ic_soft,
        "risk_proxy_ic_soft": risk_proxy_ic_soft,
        "utility_proxy_ic_first_touch_net": utility_proxy_ic_net,
        "risk_proxy_ic_first_touch_net": risk_proxy_ic_net,
    }


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    keys = ["utility_arm", "risk_arm", "gate_frac", "top_frac"]
    for key, group in monthly.groupby(keys, dropna=False, observed=True):
        utility_arm, risk_arm, gate_frac, top_frac = key
        net = pd.to_numeric(group["mean_first_touch_net"], errors="coerce")
        clean = pd.to_numeric(group["clean_first_touch_exec_rate"], errors="coerce")
        bad = pd.to_numeric(group["bad_first_touch_mae_to_sl_rate"], errors="coerce")
        p90 = pd.to_numeric(group["p90_first_touch_mae_to_sl"], errors="coerce")
        full_bad = pd.to_numeric(group.get("bad_full_path_mae_3r_rate"), errors="coerce")
        full_p90 = pd.to_numeric(group.get("p90_full_path_mae_to_sl"), errors="coerce")
        rows.append(
            {
                "utility_arm": utility_arm,
                "risk_arm": risk_arm,
                "gate_frac": float(gate_frac),
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((net > 0.0).sum()),
                "mean_first_touch_net": _safe_mean(net),
                "worst_month_first_touch_net": _safe_quantile(net, 0.0),
                "q10_first_touch_net": _safe_mean(group["q10_first_touch_net"]),
                "hit_first_touch_net": _safe_mean(group["hit_first_touch_net"]),
                "clean_first_touch_exec_rate": _safe_mean(clean),
                "first_touch_timeout_rate": _safe_mean(group["first_touch_timeout_rate"]),
                "bad_first_touch_mae_to_sl_rate": _safe_mean(bad),
                "p90_first_touch_mae_to_sl": _safe_mean(p90),
                "bad_full_path_mae_3r_rate": _safe_mean(full_bad),
                "p90_full_path_mae_to_sl": _safe_mean(full_p90),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "utility_proxy_ic_soft": _safe_mean(group["utility_proxy_ic_soft"]),
                "risk_proxy_ic_soft": _safe_mean(group["risk_proxy_ic_soft"]),
                "utility_proxy_ic_first_touch_net": _safe_mean(group["utility_proxy_ic_first_touch_net"]),
                "risk_proxy_ic_first_touch_net": _safe_mean(group["risk_proxy_ic_first_touch_net"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["proxy_gate_pass"] = (
        (out["months"] >= 3)
        & (out["positive_months"] >= 3)
        & (out["mean_first_touch_net"] > 0.0025)
        & (out["worst_month_first_touch_net"] >= 0.0)
        & (out["clean_first_touch_exec_rate"] >= 0.65)
        & (out["bad_first_touch_mae_to_sl_rate"] <= 0.25)
        & (out["p90_first_touch_mae_to_sl"] <= 2.0)
    )
    out["proxy_gate_score"] = (
        100.0 * out["mean_first_touch_net"]
        + 50.0 * out["worst_month_first_touch_net"].clip(upper=out["mean_first_touch_net"])
        + 0.75 * out["clean_first_touch_exec_rate"]
        - 0.80 * out["bad_first_touch_mae_to_sl_rate"]
        - 0.08 * (out["p90_first_touch_mae_to_sl"] - 1.0).clip(lower=0.0)
    )
    return out.sort_values(["proxy_gate_pass", "proxy_gate_score"], ascending=[False, False])


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "clean_first_touch_dual_proxy_selector.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Clean First-Touch Dual-Proxy Selector",
        "",
        "Scope: pre-training proxy test. Prior-month feature IC scores are learned separately for utility and risk labels, then risk gates the next month before utility ranking.",
        f"Labels: `{manifest['labels_path']}`",
        "",
        "## Gate Leaders",
        "",
        table(
            aggregate,
            [
                "utility_arm",
                "risk_arm",
                "gate_frac",
                "top_frac",
                "proxy_gate_pass",
                "proxy_gate_score",
                "positive_months",
                "mean_first_touch_net",
                "worst_month_first_touch_net",
                "clean_first_touch_exec_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "bad_full_path_mae_3r_rate",
                "p90_full_path_mae_to_sl",
                "top_symbol_share",
            ],
            limit=30,
        ),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_selector(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    metric_prefix: str,
    primary_prefix: str,
    challenger_prefix: str,
    utility_arms: tuple[str, ...],
    risk_arms: tuple[str, ...],
    gate_fracs: tuple[float, ...],
    top_fracs: tuple[float, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [
                frame.drop(columns=[col for col in feature_matrix.columns if col in frame.columns]),
                feature_matrix.astype(np.float32, copy=False),
            ],
            axis=1,
        )
    features = _feature_columns(frame)
    metrics = _execution_metrics(frame, metric_prefix)
    arms = {
        arm.name: arm
        for arm in _build_arms(
            frame=frame,
            metrics=metrics,
            primary_prefix=primary_prefix,
            challenger_prefix=challenger_prefix,
        )
    }
    missing = [arm for arm in tuple(utility_arms) + tuple(risk_arms) if arm not in arms]
    if missing:
        raise ValueError(f"Unknown arms: {sorted(set(missing))}")

    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    score_cache: dict[tuple[str, str], tuple[pd.Series, dict[str, Any]]] = {}
    monthly_rows: list[dict[str, Any]] = []
    month_ser = frame["__ts__"].dt.to_period("M").astype(str)
    scored_arms = tuple(dict.fromkeys(tuple(utility_arms) + tuple(risk_arms)))
    for month in months[1:]:
        train_mask = month_ser < month
        valid_mask = month_ser == month
        if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_reset = valid.reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        n_valid = int(len(valid_reset))
        for arm_name in scored_arms:
            arm = arms[arm_name]
            score, diag = _proxy_score(
                train,
                valid,
                features,
                arm.target.loc[train_mask, "target_soft"],
            )
            score_cache[(month, arm_name)] = (score.reset_index(drop=True), diag)
        for risk_arm in risk_arms:
            risk_score, risk_diag = score_cache[(month, risk_arm)]
            risk_valid_target = arms[risk_arm].target.loc[valid_mask].reset_index(drop=True)
            for gate_frac in gate_fracs:
                gate_idx = _rank_top(risk_score, int(math.ceil(float(gate_frac) * n_valid)))
                if not len(gate_idx):
                    continue
                for utility_arm in utility_arms:
                    utility_score, utility_diag = score_cache[(month, utility_arm)]
                    utility_valid_target = arms[utility_arm].target.loc[valid_mask].reset_index(drop=True)
                    for top_frac in top_fracs:
                        top_k = min(len(gate_idx), max(1, int(math.ceil(float(top_frac) * n_valid))))
                        gated_scores = utility_score.iloc[gate_idx].reset_index(drop=True)
                        local_idx = _rank_top(gated_scores, top_k)
                        selected_idx = gate_idx[local_idx] if len(local_idx) else np.array([], dtype=np.int64)
                        monthly_rows.append(
                            _selection_row(
                                frame=valid_reset,
                                metrics=valid_metrics,
                                selected_idx=selected_idx,
                                utility_arm=utility_arm,
                                risk_arm=risk_arm,
                                period=month,
                                gate_frac=gate_frac,
                                top_frac=top_frac,
                                utility_proxy_ic_soft=_spearman(
                                    utility_score,
                                    utility_valid_target["target_soft"],
                                ),
                                risk_proxy_ic_soft=_spearman(
                                    risk_score,
                                    risk_valid_target["target_soft"],
                                ),
                                utility_proxy_ic_net=_spearman(
                                    utility_score,
                                    valid_metrics["first_touch_net"],
                                ),
                                risk_proxy_ic_net=_spearman(
                                    risk_score,
                                    valid_metrics["first_touch_net"],
                                ),
                            )
                        )

    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)
    paths = {
        "monthly": output_dir / "clean_first_touch_dual_proxy_selector_monthly.csv",
        "aggregate": output_dir / "clean_first_touch_dual_proxy_selector_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    manifest = {
        "scope": "no_training_clean_first_touch_dual_proxy_selector",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "metric_prefix": str(metric_prefix),
        "utility_arms": list(utility_arms),
        "risk_arms": list(risk_arms),
        "gate_fracs": list(gate_fracs),
        "top_fracs": list(top_fracs),
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "source_ablation_script": str(CLEAN_ABLATION_OUTPUT_DIR),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _csv_tuple(value: str, *, cast: type = str) -> tuple[Any, ...]:
    if not value:
        return tuple()
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=160)
    parser.add_argument("--metric-prefix", default=DEFAULT_METRIC_PREFIX)
    parser.add_argument("--primary-prefix", default=DEFAULT_PRIMARY_PREFIX)
    parser.add_argument("--challenger-prefix", default=DEFAULT_CHALLENGER_PREFIX)
    parser.add_argument("--utility-arms", default=",".join(DEFAULT_UTILITY_ARMS))
    parser.add_argument("--risk-arms", default=",".join(DEFAULT_RISK_ARMS))
    parser.add_argument("--gate-fracs", default=",".join(str(v) for v in DEFAULT_GATE_FRACS))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_selector(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        metric_prefix=str(args.metric_prefix),
        primary_prefix=str(args.primary_prefix),
        challenger_prefix=str(args.challenger_prefix),
        utility_arms=_csv_tuple(str(args.utility_arms), cast=str),
        risk_arms=_csv_tuple(str(args.risk_arms), cast=str),
        gate_fracs=_csv_tuple(str(args.gate_fracs), cast=float),
        top_fracs=_csv_tuple(str(args.top_fracs), cast=float),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
