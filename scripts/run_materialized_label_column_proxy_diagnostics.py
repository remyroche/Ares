#!/usr/bin/env python3
"""No-training proxy diagnostics for materialized label target columns."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    TOP_FRACS,
    _decile_diagnostics,
    _feature_columns,
    _feature_ic,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _proxy_score,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
    _safe_std,
    _spearman,
)


DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_170000_first_touch_two_head_stage164_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage164_materialized_label_column_proxy_diagnostics_v1")
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260629_050000")
DEFAULT_FEATURE_LIST_CSV = Path("data_perp/artifacts/20260629_050000_lgbm_mda/quality_reports/base_model_feature_importance.csv")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    soft_col: str
    hard_col: str
    metric_prefix: str


def _load_manifest(labels_path: Path) -> dict[str, Any]:
    manifest_path = labels_path / "labels_manifest.json" if labels_path.is_dir() else labels_path.parent / "labels_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _discover_specs(frame: pd.DataFrame, labels_path: Path, recipe_manifest_key: str) -> list[TargetSpec]:
    manifest = _load_manifest(labels_path)
    recipe = manifest.get(recipe_manifest_key)
    specs: list[TargetSpec] = []
    roles = []
    if isinstance(recipe, dict):
        primary_prefix = str(recipe.get("primary_column_prefix") or "")
        challenger_prefix = str(recipe.get("balanced_column_prefix") or "")
        if primary_prefix:
            roles.append(("primary", primary_prefix))
        if challenger_prefix:
            roles.append(("challenger", challenger_prefix))
    if not roles:
        roles = [
            ("primary", "__stage164_primary"),
            ("challenger", "__stage164_challenger"),
        ]
    for role, prefix in roles:
        for head in ("utility", "support"):
            soft = f"{prefix}_{head}_target_soft__"
            hard = f"{prefix}_{head}_target_hard__"
            if soft in frame.columns and hard in frame.columns:
                specs.append(TargetSpec(name=f"{role}_{head}", soft_col=soft, hard_col=hard, metric_prefix=prefix))
    return specs


def _parse_target_specs(raw_specs: list[str]) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for raw in raw_specs:
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError("--target-spec must be name:soft_col:hard_col:metric_prefix")
        specs.append(TargetSpec(name=parts[0], soft_col=parts[1], hard_col=parts[2], metric_prefix=parts[3]))
    return specs


def _execution_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    metrics = _path_metrics(frame)
    net_col = f"{prefix}_first_touch_net__"
    clean_col = f"{prefix}_clean_first_touch_exec__"
    timeout_col = f"{prefix}_first_touch_timeout__"
    mae_col = f"{prefix}_first_touch_mae_to_sl__"
    missing = [col for col in (net_col, clean_col, timeout_col, mae_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing execution metric columns for {prefix}: {missing}")
    first_touch_net = _safe_numeric(frame[net_col]).reindex(frame.index)
    clean_exec = _safe_numeric(frame[clean_col]).reindex(frame.index).fillna(0.0)
    timeout = _safe_numeric(frame[timeout_col]).reindex(frame.index).fillna(0.0)
    mae_to_sl = _safe_numeric(frame[mae_col]).reindex(frame.index)
    metrics["u_policy_net"] = first_touch_net
    metrics["ret_net"] = first_touch_net
    metrics["first_touch_net"] = first_touch_net
    metrics["clean_first_touch_exec"] = clean_exec
    metrics["first_touch_timeout"] = timeout > 0.5
    metrics["is_timeout"] = timeout > 0.5
    metrics["first_touch_mae_to_sl"] = mae_to_sl
    metrics["mae_norm"] = mae_to_sl.fillna(metrics["mae_norm"])
    metrics.attrs["utility_source"] = net_col
    return metrics


def _target_from_spec(frame: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    missing = [col for col in (spec.soft_col, spec.hard_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing target columns for {spec.name}: {missing}")
    soft = _safe_numeric(frame[spec.soft_col]).clip(0.0, 1.0)
    hard = _safe_numeric(frame[spec.hard_col]).clip(0.0, 1.0)
    return pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=frame.index)


def _effective_n(values: Any) -> float:
    counts = pd.Series(values, dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return float(1.0 / denom) if denom > 0.0 else 0.0


def _selection_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
) -> dict[str, Any]:
    idx = _rank_top_indices(score, top_frac)
    selected_metrics = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    selected_frame = frame.iloc[idx] if len(idx) else frame.iloc[:0]
    selected_target = target.iloc[idx] if len(idx) else target.iloc[:0]
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    side = _safe_numeric(selected_metrics.get("side")).fillna(1.0)
    long_rows = int((side > 0.0).sum())
    short_rows = int((side < 0.0).sum())
    first_touch_net = selected_metrics["first_touch_net"]
    mae_to_sl = selected_metrics["first_touch_mae_to_sl"]
    row = {
        "arm": str(arm),
        "selector": str(selector),
        "period": str(period),
        "top_frac": float(top_frac),
        "rows": int(len(frame)),
        "selected_rows": int(len(idx)),
        "selected_long_rows": long_rows,
        "selected_short_rows": short_rows,
        "selected_long_share": float(long_rows / len(idx)) if len(idx) else 0.0,
        "selected_short_share": float(short_rows / len(idx)) if len(idx) else 0.0,
        "target_top_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_top_hard_rate": _safe_mean(selected_target.get("target_hard")),
        "mean_first_touch_net": _safe_mean(first_touch_net),
        "q10_first_touch_net": _safe_quantile(first_touch_net, 0.10),
        "hit_first_touch_net": _safe_mean(first_touch_net > 0.0),
        "clean_first_touch_exec_rate": _safe_mean(selected_metrics["clean_first_touch_exec"]),
        "first_touch_timeout_rate": _safe_mean(selected_metrics["first_touch_timeout"].astype(float)),
        "bad_first_touch_mae_to_sl_rate": _safe_mean(mae_to_sl >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(mae_to_sl, 0.90),
        "mean_barrier": _safe_mean(selected_metrics["barrier"]),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] > 0.025),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
    }
    return row


def _summarise_target(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    spec: TargetSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    soft = target["target_soft"]
    hard = target["target_hard"]
    ic = _feature_ic(frame, features, soft)
    top_ic = ic.head(8)
    summary = {
        "arm": spec.name,
        "soft_col": spec.soft_col,
        "hard_col": spec.hard_col,
        "metric_prefix": spec.metric_prefix,
        "rows": int(len(frame)),
        "finite_soft_frac": float(soft.notna().mean()) if len(soft) else float("nan"),
        "soft_mean": _safe_mean(soft),
        "soft_std": _safe_std(soft),
        "soft_p10": _safe_quantile(soft, 0.10),
        "soft_p50": _safe_quantile(soft, 0.50),
        "soft_p90": _safe_quantile(soft, 0.90),
        "soft_low_sat_rate": _safe_mean(soft <= 0.05),
        "soft_high_sat_rate": _safe_mean(soft >= 0.95),
        "hard_rate": _safe_mean(hard),
        "ic_soft_vs_first_touch_net": _spearman(soft, metrics["first_touch_net"]),
        "ic_soft_vs_clean_exec": _spearman(soft, metrics["clean_first_touch_exec"]),
        "ic_soft_vs_timeout": _spearman(soft, metrics["first_touch_timeout"].astype(float)),
        "ic_soft_vs_mae_to_sl": _spearman(soft, metrics["first_touch_mae_to_sl"]),
        "feature_count": int(len(features)),
        "feature_top_abs_ic": float(top_ic["abs_ic"].iloc[0]) if len(top_ic) else float("nan"),
        "feature_mean_top_abs_ic": float(top_ic["abs_ic"].mean()) if len(top_ic) else float("nan"),
        "feature_n_abs_ic_ge_002": int((ic["abs_ic"] >= 0.02).sum()) if not ic.empty else 0,
        "feature_n_abs_ic_ge_005": int((ic["abs_ic"] >= 0.05).sum()) if not ic.empty else 0,
        "feature_top_names": ",".join(top_ic["feature"].astype(str).tolist()) if len(top_ic) else "",
    }
    summary.update(_decile_diagnostics(soft, metrics["first_touch_net"]))
    if not ic.empty:
        ic = ic.head(25).copy()
        ic.insert(0, "arm", spec.name)
    return summary, ic


def _proxy_oos_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    spec: TargetSpec,
) -> list[dict[str, Any]]:
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for month in months[1:]:
        month_ser = frame["__ts__"].dt.to_period("M").astype(str)
        train_mask = month_ser < month
        valid_mask = month_ser == month
        if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
        score, diag = _proxy_score(train, valid, features, target.loc[train_mask, "target_soft"])
        score = score.reset_index(drop=True)
        valid_reset = valid.reset_index(drop=True)
        period_mean = _safe_mean(valid_metrics["first_touch_net"])
        period_clean = _safe_mean(valid_metrics["clean_first_touch_exec"])
        period_bad = _safe_mean(valid_metrics["first_touch_mae_to_sl"] >= 1.0)
        for frac in TOP_FRACS:
            row = _selection_metrics(
                frame=valid_reset,
                metrics=valid_metrics,
                target=valid_target,
                score=score,
                arm=spec.name,
                selector="feature_ic_proxy_oos",
                period=month,
                top_frac=frac,
            )
            row.update(
                {
                    "period_baseline_mean_first_touch_net": period_mean,
                    "period_baseline_clean_first_touch_exec_rate": period_clean,
                    "period_baseline_bad_first_touch_mae_to_sl_rate": period_bad,
                    "delta_mean_first_touch_net_vs_period": row["mean_first_touch_net"] - period_mean,
                    "delta_clean_first_touch_exec_rate_vs_period": row["clean_first_touch_exec_rate"] - period_clean,
                    "delta_bad_first_touch_mae_to_sl_rate_vs_period": row["bad_first_touch_mae_to_sl_rate"] - period_bad,
                    "proxy_ic_soft": _spearman(score, valid_target["target_soft"]),
                    "proxy_ic_first_touch_net": _spearman(score, valid_metrics["first_touch_net"]),
                    "proxy_features": ",".join(diag.get("proxy_features", [])),
                    "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                    "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
                }
            )
            rows.append(row)
    return rows


def _aggregate_proxy(proxy: pd.DataFrame) -> pd.DataFrame:
    if proxy.empty:
        return proxy
    rows: list[dict[str, Any]] = []
    for (arm, frac), group in proxy.groupby(["arm", "top_frac"], dropna=False, observed=True):
        net = pd.to_numeric(group["mean_first_touch_net"], errors="coerce")
        rows.append(
            {
                "arm": arm,
                "top_frac": float(frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((net > 0.0).sum()),
                "mean_first_touch_net": _safe_mean(net),
                "worst_month_first_touch_net": _safe_quantile(net, 0.0),
                "hit_first_touch_net": _safe_mean(group["hit_first_touch_net"]),
                "q10_first_touch_net": _safe_mean(group["q10_first_touch_net"]),
                "clean_first_touch_exec_rate": _safe_mean(group["clean_first_touch_exec_rate"]),
                "first_touch_timeout_rate": _safe_mean(group["first_touch_timeout_rate"]),
                "bad_first_touch_mae_to_sl_rate": _safe_mean(group["bad_first_touch_mae_to_sl_rate"]),
                "p90_first_touch_mae_to_sl": _safe_mean(group["p90_first_touch_mae_to_sl"]),
                "delta_mean_first_touch_net_vs_period": _safe_mean(group["delta_mean_first_touch_net_vs_period"]),
                "delta_clean_first_touch_exec_rate_vs_period": _safe_mean(group["delta_clean_first_touch_exec_rate_vs_period"]),
                "delta_bad_first_touch_mae_to_sl_rate_vs_period": _safe_mean(group["delta_bad_first_touch_mae_to_sl_rate_vs_period"]),
                "proxy_ic_soft": _safe_mean(group["proxy_ic_soft"]),
                "proxy_ic_first_touch_net": _safe_mean(group["proxy_ic_first_touch_net"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["top_frac", "mean_first_touch_net"], ascending=[True, False])


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    oracle: pd.DataFrame,
    proxy_agg: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "materialized_label_column_proxy_diagnostics.md"

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
        "# Materialized Label Column Proxy Diagnostics",
        "",
        "Scope: no LightGBM, Optuna, or policy-geometry training. Feature proxies are prior-month univariate IC ranks.",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature store: `{manifest['feature_store'].get('feature_dir', '')}`",
        "",
        "## Label Shape And Feature Association",
        "",
        table(
            summary.sort_values(["feature_mean_top_abs_ic", "ic_soft_vs_first_touch_net"], ascending=[False, False]),
            [
                "arm",
                "soft_mean",
                "soft_std",
                "hard_rate",
                "ic_soft_vs_first_touch_net",
                "ic_soft_vs_clean_exec",
                "ic_soft_vs_mae_to_sl",
                "decile_spearman_u",
                "decile_violations_u",
                "feature_mean_top_abs_ic",
                "feature_n_abs_ic_ge_002",
                "feature_n_abs_ic_ge_005",
                "feature_top_names",
            ],
        ),
        "",
        "## Oracle Label Sort Top 10%",
        "",
        table(
            oracle[oracle["top_frac"].eq(0.10)].sort_values(
                ["mean_first_touch_net", "clean_first_touch_exec_rate"],
                ascending=[False, False],
            ),
            [
                "arm",
                "selected_rows",
                "mean_first_touch_net",
                "q10_first_touch_net",
                "hit_first_touch_net",
                "clean_first_touch_exec_rate",
                "first_touch_timeout_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "top_symbol_share",
            ],
        ),
        "",
        "## Feature-IC Proxy OOS Top 10%",
        "",
        table(
            proxy_agg[proxy_agg["top_frac"].eq(0.10)].sort_values(
                ["mean_first_touch_net", "worst_month_first_touch_net"],
                ascending=[False, False],
            ),
            [
                "arm",
                "months",
                "positive_months",
                "mean_first_touch_net",
                "worst_month_first_touch_net",
                "delta_mean_first_touch_net_vs_period",
                "clean_first_touch_exec_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "proxy_ic_soft",
                "proxy_ic_first_touch_net",
                "top_symbol_share",
            ],
        ),
        "",
        "## Feature-IC Proxy OOS Top 1%",
        "",
        table(
            proxy_agg[proxy_agg["top_frac"].eq(0.01)].sort_values(
                ["mean_first_touch_net", "worst_month_first_touch_net"],
                ascending=[False, False],
            ),
            [
                "arm",
                "months",
                "positive_months",
                "mean_first_touch_net",
                "worst_month_first_touch_net",
                "delta_mean_first_touch_net_vs_period",
                "clean_first_touch_exec_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "proxy_ic_soft",
                "proxy_ic_first_touch_net",
                "top_symbol_share",
            ],
        ),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Oracle selection: `{manifest['outputs']['oracle_selection']}`",
        f"- Feature IC: `{manifest['outputs']['feature_ic']}`",
        f"- Proxy OOS monthly: `{manifest['outputs']['proxy_oos_monthly']}`",
        f"- Proxy OOS aggregate: `{manifest['outputs']['proxy_oos_aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostics(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    recipe_manifest_key: str,
    target_specs: list[TargetSpec] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    specs = target_specs or _discover_specs(frame, labels_path, recipe_manifest_key)
    if not specs:
        raise ValueError("No materialized target specs found")
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

    summary_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    feature_ic_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []

    for spec in specs:
        metrics = _execution_metrics(frame, spec.metric_prefix)
        target = _target_from_spec(frame, spec)
        summary, feature_ic = _summarise_target(
            frame=frame,
            metrics=metrics,
            target=target,
            features=features,
            spec=spec,
        )
        summary_rows.append(summary)
        if not feature_ic.empty:
            feature_ic_rows.extend(feature_ic.to_dict("records"))
        for frac in TOP_FRACS:
            oracle_rows.append(
                _selection_metrics(
                    frame=frame,
                    metrics=metrics,
                    target=target,
                    score=target["target_soft"],
                    arm=spec.name,
                    selector="oracle_label_sort",
                    period="all",
                    top_frac=frac,
                )
            )
        proxy_rows.extend(
            _proxy_oos_rows(
                frame=frame,
                metrics=metrics,
                target=target,
                features=features,
                spec=spec,
            )
        )

    summary = pd.DataFrame(summary_rows)
    oracle = pd.DataFrame(oracle_rows)
    feature_ic = pd.DataFrame(feature_ic_rows)
    proxy = pd.DataFrame(proxy_rows)
    proxy_agg = _aggregate_proxy(proxy)

    paths = {
        "summary": output_dir / "materialized_label_column_summary.csv",
        "oracle_selection": output_dir / "materialized_label_column_oracle_selection.csv",
        "feature_ic": output_dir / "materialized_label_column_feature_ic_top25.csv",
        "proxy_oos_monthly": output_dir / "materialized_label_column_proxy_oos_monthly.csv",
        "proxy_oos_aggregate": output_dir / "materialized_label_column_proxy_oos_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    oracle.to_csv(paths["oracle_selection"], index=False)
    feature_ic.to_csv(paths["feature_ic"], index=False)
    proxy.to_csv(paths["proxy_oos_monthly"], index=False)
    proxy_agg.to_csv(paths["proxy_oos_aggregate"], index=False)

    manifest = {
        "scope": "no_training_materialized_label_column_proxy_diagnostics",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "recipe_manifest_key": str(recipe_manifest_key),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "target_specs": [spec.__dict__ for spec in specs],
        "features": features,
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "top_fracs": list(TOP_FRACS),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=summary,
        oracle=oracle,
        proxy_agg=proxy_agg,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=160)
    parser.add_argument("--recipe-manifest-key", default="stage164_two_head_label_recipe")
    parser.add_argument(
        "--target-spec",
        action="append",
        default=[],
        help="Optional name:soft_col:hard_col:metric_prefix. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_specs = _parse_target_specs(list(args.target_spec)) if args.target_spec else None
    manifest = run_diagnostics(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        recipe_manifest_key=str(args.recipe_manifest_key),
        target_specs=target_specs,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
