#!/usr/bin/env python3
"""No-training ablation for clean first-touch utility label designs."""

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
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _spearman,
)
from scripts.run_materialized_label_column_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_PATH,
    _aggregate_proxy,
    _execution_metrics,
    _proxy_oos_rows,
    _selection_metrics,
    _summarise_target,
    _target_from_spec,
    TargetSpec,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/clean_first_touch_label_ablation_stage165_v1")
DEFAULT_METRIC_PREFIX = "__stage164_primary"
DEFAULT_PRIMARY_PREFIX = "__stage164_primary"
DEFAULT_CHALLENGER_PREFIX = "__stage164_challenger"


@dataclass(frozen=True)
class AblationArm:
    name: str
    description: str
    target: pd.DataFrame


def _sigmoid(values: Any, scale: float = 1.0) -> pd.Series:
    series = _safe_numeric(values).astype(float)
    scaled = np.clip(series.to_numpy(dtype=np.float64) / float(scale), -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-scaled)), index=series.index)


def _timestamp_rank(values: pd.Series, timestamps: pd.Series) -> pd.Series:
    ranked = values.groupby(timestamps, dropna=False).rank(method="average", pct=True)
    fallback = values.rank(method="average", pct=True)
    return ranked.fillna(fallback).fillna(0.0).clip(0.0, 1.0)


def _target(soft: pd.Series, hard: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(soft).clip(0.0, 1.0),
            "target_hard": _safe_numeric(hard).fillna(0.0).clip(0.0, 1.0),
        },
        index=soft.index,
    )


def _column_target(frame: pd.DataFrame, *, name: str, soft_col: str, hard_col: str, prefix: str) -> AblationArm:
    spec = TargetSpec(name=name, soft_col=soft_col, hard_col=hard_col, metric_prefix=prefix)
    return AblationArm(
        name=name,
        description=f"materialized baseline: {soft_col}",
        target=_target_from_spec(frame, spec),
    )


def _build_arms(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    primary_prefix: str,
    challenger_prefix: str,
) -> list[AblationArm]:
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    net = _safe_numeric(metrics["first_touch_net"]).fillna(-0.05)
    clean = _safe_numeric(metrics["clean_first_touch_exec"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(metrics["first_touch_timeout"].astype(float)).fillna(0.0).clip(0.0, 1.0)
    mae = _safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    full_path_mae = _safe_numeric(frame.get("__first_touch_full_path_mae_to_sl__", mae)).fillna(mae).clip(lower=0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.025).clip(lower=0.0)

    primary_support = _safe_numeric(frame[f"{primary_prefix}_support_target_soft__"]).fillna(0.0).clip(0.0, 1.0)
    challenger_support = _safe_numeric(frame[f"{challenger_prefix}_support_target_soft__"]).fillna(0.0).clip(0.0, 1.0)

    net_sig = _sigmoid(net - 0.0015, scale=0.004)
    mae_good = _sigmoid(0.85 - mae, scale=0.25)
    barrier_good = _sigmoid(0.025 - barrier, scale=0.006)
    no_timeout = 1.0 - timeout
    clean_rank_net = _timestamp_rank(net.where(clean >= 0.5, -1.0), ts)
    low_mae_rank = _timestamp_rank((-mae).where(clean >= 0.5, -10.0), ts)
    pathsafe_rank = _timestamp_rank((net - 0.004 * mae).where(clean >= 0.5, -1.0), ts)
    full_path_low_mae_rank = _timestamp_rank((-full_path_mae).where(clean >= 0.5, -100.0), ts)
    tail_veto_rank = _timestamp_rank(
        (
            net
            - 0.0060 * (mae - 0.45).clip(lower=0.0)
            - 0.0020 * (full_path_mae - 3.00).clip(lower=0.0)
            - 0.0100 * timeout
            - 0.0015 * (barrier > 0.025).astype(float)
        ).where(clean >= 0.5, -1.0),
        ts,
    )
    clean_local_rank = (clean * clean_rank_net).clip(0.0, 1.0)
    support_blend = (0.70 * primary_support + 0.30 * challenger_support).clip(0.0, 1.0)
    strict_mae_good = _sigmoid(0.45 - mae, scale=0.12)
    tail_mae_good = _sigmoid(0.65 - mae, scale=0.18)
    full_path_good = _sigmoid(3.00 - full_path_mae, scale=0.80)
    penalized_raw = (
        net
        - 0.0100 * timeout
        - 0.0060 * (mae - 0.80).clip(lower=0.0)
        - 0.0015 * (full_path_mae - 3.00).clip(lower=0.0)
        - 0.0040 * (1.0 - clean)
        - 0.0015 * (barrier - 0.025).clip(lower=0.0) / 0.010
    )

    arms = [
        _column_target(
            frame,
            name="B_stage164_primary_utility",
            soft_col=f"{primary_prefix}_utility_target_soft__",
            hard_col=f"{primary_prefix}_utility_target_hard__",
            prefix=primary_prefix,
        ),
        _column_target(
            frame,
            name="B_stage164_primary_support",
            soft_col=f"{primary_prefix}_support_target_soft__",
            hard_col=f"{primary_prefix}_support_target_hard__",
            prefix=primary_prefix,
        ),
        _column_target(
            frame,
            name="B_stage164_challenger_support",
            soft_col=f"{challenger_prefix}_support_target_soft__",
            hard_col=f"{challenger_prefix}_support_target_hard__",
            prefix=primary_prefix,
        ),
        AblationArm(
            name="C_clean_net_sigmoid",
            description="first-touch net utility, zeroed unless clean first-touch execution",
            target=_target(clean * net_sig, (clean >= 0.5) & (net > 0.0)),
        ),
        AblationArm(
            name="C_support_times_clean_net",
            description="Stage164 support times clean-gated net utility",
            target=_target(primary_support * clean * net_sig, (primary_support >= 0.50) & (clean >= 0.5) & (net > 0.0)),
        ),
        AblationArm(
            name="C_clean_rank_net",
            description="timestamp-local net rank with dirty rows capped at zero",
            target=_target(clean_local_rank, (clean >= 0.5) & (clean_rank_net >= 0.80) & (net > 0.0)),
        ),
        AblationArm(
            name="C_clean_mae_net_blend",
            description="clean first-touch blend of net utility and low-adverse path",
            target=_target(clean * ((0.65 * net_sig) + (0.25 * mae_good) + (0.10 * barrier_good)), (clean >= 0.5) & (net > 0.0) & (mae <= 1.0)),
        ),
        AblationArm(
            name="C_timeout_mae_penalized",
            description="net utility with explicit timeout, adverse path, and dirty-exec penalties",
            target=_target(_sigmoid(penalized_raw, scale=0.004), (clean >= 0.5) & (net > 0.0) & (mae <= 1.0) & (timeout < 0.5)),
        ),
        AblationArm(
            name="C_support_clean_rank_blend",
            description="support target blended with clean timestamp-local net rank",
            target=_target((0.55 * support_blend) + (0.45 * clean_local_rank), (support_blend >= 0.45) & (clean >= 0.5) & (net > 0.0)),
        ),
        AblationArm(
            name="C_conservative_clean_support",
            description="support target gated by clean/no-timeout and low first-touch MAE",
            target=_target(support_blend * clean * no_timeout * mae_good, (support_blend >= 0.35) & (clean >= 0.5) & (mae <= 0.85)),
        ),
        AblationArm(
            name="D_low_mae_clean",
            description="clean first-touch target dominated by low first-touch MAE",
            target=_target(clean * no_timeout * _sigmoid(0.55 - mae, scale=0.15), (clean >= 0.5) & (net > 0.0) & (mae <= 0.55)),
        ),
        AblationArm(
            name="D_low_mae_net_blend",
            description="low first-touch MAE target with a smaller positive-net term",
            target=_target(
                clean * no_timeout * ((0.65 * _sigmoid(0.55 - mae, scale=0.15)) + (0.35 * net_sig)),
                (clean >= 0.5) & (net > 0.0) & (mae <= 0.65),
            ),
        ),
        AblationArm(
            name="D_support_low_mae",
            description="Stage164 support target multiplied by a strict low-MAE envelope",
            target=_target(
                support_blend * clean * no_timeout * _sigmoid(0.60 - mae, scale=0.18),
                (support_blend >= 0.25) & (clean >= 0.5) & (net > 0.0) & (mae <= 0.70),
            ),
        ),
        AblationArm(
            name="D_low_mae_rank_blend",
            description="timestamp-local blend of low first-touch MAE rank and clean net rank",
            target=_target(
                clean * no_timeout * ((0.70 * low_mae_rank) + (0.30 * clean_rank_net)),
                (clean >= 0.5) & (net > 0.0) & (low_mae_rank >= 0.75),
            ),
        ),
        AblationArm(
            name="D_pathsafe_rank_blend",
            description="timestamp-local rank of net minus adverse path, gated by clean execution",
            target=_target(
                clean * no_timeout * ((0.70 * pathsafe_rank) + (0.30 * support_blend)),
                (clean >= 0.5) & (net > 0.0) & (mae <= 1.0) & (pathsafe_rank >= 0.70),
            ),
        ),
        AblationArm(
            name="E_tail_veto_low_mae_clean",
            description="strict low first-touch MAE veto with positive clean net utility",
            target=_target(
                clean * no_timeout * strict_mae_good * ((0.65 * net_sig) + (0.35 * barrier_good)),
                (clean >= 0.5) & (net > 0.0) & (mae <= 0.45) & (timeout < 0.5),
            ),
        ),
        AblationArm(
            name="E_tail_veto_support_lowmae",
            description="Stage164 support target inside a stricter low-MAE/no-timeout envelope",
            target=_target(
                support_blend * clean * no_timeout * ((0.75 * tail_mae_good) + (0.25 * barrier_good)),
                (support_blend >= 0.25) & (clean >= 0.5) & (net > 0.0) & (mae <= 0.65) & (timeout < 0.5),
            ),
        ),
        AblationArm(
            name="E_tail_veto_rank_margin",
            description="timestamp-local rank of net after explicit MAE, timeout, and wide-barrier penalties",
            target=_target(
                clean
                * no_timeout
                * ((0.55 * tail_veto_rank) + (0.30 * low_mae_rank) + (0.15 * barrier_good)),
                (clean >= 0.5) & (net > 0.0) & (mae <= 0.70) & (tail_veto_rank >= 0.80) & (timeout < 0.5),
            ),
        ),
        AblationArm(
            name="F_fullpath_veto_support",
            description="Stage164 support inside first-touch and full-path adverse excursion limits",
            target=_target(
                support_blend
                * clean
                * no_timeout
                * ((0.45 * tail_mae_good) + (0.35 * full_path_good) + (0.20 * barrier_good)),
                (support_blend >= 0.25)
                & (clean >= 0.5)
                & (net > 0.0)
                & (mae <= 0.75)
                & (full_path_mae <= 3.00)
                & (timeout < 0.5),
            ),
        ),
        AblationArm(
            name="F_fullpath_rank_margin",
            description="timestamp-local rank of clean net after first-touch and full-path MAE penalties",
            target=_target(
                clean
                * no_timeout
                * (
                    (0.45 * tail_veto_rank)
                    + (0.35 * full_path_low_mae_rank)
                    + (0.20 * clean_rank_net)
                ),
                (clean >= 0.5)
                & (net > 0.0)
                & (mae <= 0.85)
                & (full_path_mae <= 3.50)
                & (full_path_low_mae_rank >= 0.70)
                & (timeout < 0.5),
            ),
        ),
    ]
    return arms


def _decision_rows(proxy_agg: pd.DataFrame) -> pd.DataFrame:
    if proxy_agg.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in proxy_agg.iterrows():
        frac = float(row["top_frac"])
        if frac not in {0.01, 0.03, 0.10}:
            continue
        months = int(row.get("months", 0))
        positive = int(row.get("positive_months", 0))
        mean_net = float(row.get("mean_first_touch_net", float("nan")))
        worst_net = float(row.get("worst_month_first_touch_net", float("nan")))
        clean = float(row.get("clean_first_touch_exec_rate", float("nan")))
        bad = float(row.get("bad_first_touch_mae_to_sl_rate", float("nan")))
        p90 = float(row.get("p90_first_touch_mae_to_sl", float("nan")))
        pass_proxy = (
            months >= 3
            and positive >= 3
            and math.isfinite(mean_net)
            and mean_net > 0.0025
            and math.isfinite(worst_net)
            and worst_net >= 0.0
            and math.isfinite(clean)
            and clean >= 0.65
            and math.isfinite(bad)
            and bad <= 0.25
            and math.isfinite(p90)
            and p90 <= 2.00
        )
        score = (
            100.0 * mean_net
            + 50.0 * min(worst_net, mean_net)
            + 0.75 * clean
            - 0.80 * bad
            - 0.08 * max(0.0, p90 - 1.0)
        )
        rows.append(
            {
                "arm": row["arm"],
                "top_frac": frac,
                "months": months,
                "positive_months": positive,
                "mean_first_touch_net": mean_net,
                "worst_month_first_touch_net": worst_net,
                "clean_first_touch_exec_rate": clean,
                "bad_first_touch_mae_to_sl_rate": bad,
                "p90_first_touch_mae_to_sl": p90,
                "proxy_gate_pass": bool(pass_proxy),
                "proxy_gate_score": float(score),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["proxy_gate_pass", "proxy_gate_score"], ascending=[False, False])


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    oracle: pd.DataFrame,
    proxy_agg: pd.DataFrame,
    decisions: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "clean_first_touch_label_ablation.md"

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
        "# Clean First-Touch Label Ablation",
        "",
        "Scope: pre-training label QA only. No LightGBM, Optuna, or policy geometry is fitted.",
        "Feature proxies use prior months to rank the next month.",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature store: `{manifest['feature_store'].get('feature_dir', '')}`",
        "",
        "## Proxy Gate Leaders",
        "",
        table(
            decisions,
            [
                "arm",
                "top_frac",
                "proxy_gate_pass",
                "proxy_gate_score",
                "positive_months",
                "mean_first_touch_net",
                "worst_month_first_touch_net",
                "clean_first_touch_exec_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
            ],
            limit=20,
        ),
        "",
        "## Label Shape",
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
                "ic_soft_vs_timeout",
                "ic_soft_vs_mae_to_sl",
                "feature_mean_top_abs_ic",
                "feature_n_abs_ic_ge_002",
                "feature_n_abs_ic_ge_005",
            ],
        ),
        "",
        "## Oracle Top 10%",
        "",
        table(
            oracle[oracle["top_frac"].eq(0.10)].sort_values(
                ["clean_first_touch_exec_rate", "mean_first_touch_net"],
                ascending=[False, False],
            ),
            [
                "arm",
                "selected_rows",
                "mean_first_touch_net",
                "clean_first_touch_exec_rate",
                "first_touch_timeout_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "top_symbol_share",
            ],
        ),
        "",
        "## Feature Proxy OOS Top 1%",
        "",
        table(
            proxy_agg[proxy_agg["top_frac"].eq(0.01)].sort_values(
                ["clean_first_touch_exec_rate", "mean_first_touch_net"],
                ascending=[False, False],
            ),
            [
                "arm",
                "months",
                "positive_months",
                "mean_first_touch_net",
                "worst_month_first_touch_net",
                "clean_first_touch_exec_rate",
                "first_touch_timeout_rate",
                "bad_first_touch_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "proxy_ic_soft",
                "proxy_ic_first_touch_net",
                "top_symbol_share",
            ],
        ),
        "",
        "## Feature Proxy OOS Top 3%",
        "",
        table(
            proxy_agg[proxy_agg["top_frac"].eq(0.03)].sort_values(
                ["clean_first_touch_exec_rate", "mean_first_touch_net"],
                ascending=[False, False],
            ),
            [
                "arm",
                "months",
                "positive_months",
                "mean_first_touch_net",
                "worst_month_first_touch_net",
                "clean_first_touch_exec_rate",
                "first_touch_timeout_rate",
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
        f"- Oracle: `{manifest['outputs']['oracle']}`",
        f"- Proxy monthly: `{manifest['outputs']['proxy_monthly']}`",
        f"- Proxy aggregate: `{manifest['outputs']['proxy_aggregate']}`",
        f"- Decisions: `{manifest['outputs']['decisions']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    metric_prefix: str,
    primary_prefix: str,
    challenger_prefix: str,
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
    arms = _build_arms(
        frame=frame,
        metrics=metrics,
        primary_prefix=primary_prefix,
        challenger_prefix=challenger_prefix,
    )

    summary_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    feature_ic_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    for arm in arms:
        spec = TargetSpec(
            name=arm.name,
            soft_col=f"{arm.name}:in_memory_soft",
            hard_col=f"{arm.name}:in_memory_hard",
            metric_prefix=metric_prefix,
        )
        summary, feature_ic = _summarise_target(
            frame=frame,
            metrics=metrics,
            target=arm.target,
            features=features,
            spec=spec,
        )
        summary["description"] = arm.description
        summary_rows.append(summary)
        if not feature_ic.empty:
            feature_ic_rows.extend(feature_ic.to_dict("records"))
        for frac in TOP_FRACS:
            oracle_rows.append(
                _selection_metrics(
                    frame=frame,
                    metrics=metrics,
                    target=arm.target,
                    score=arm.target["target_soft"],
                    arm=arm.name,
                    selector="oracle_label_sort",
                    period="all",
                    top_frac=frac,
                )
            )
        proxy_rows.extend(
            _proxy_oos_rows(
                frame=frame,
                metrics=metrics,
                target=arm.target,
                features=features,
                spec=spec,
            )
        )

    summary = pd.DataFrame(summary_rows)
    oracle = pd.DataFrame(oracle_rows)
    feature_ic = pd.DataFrame(feature_ic_rows)
    proxy = pd.DataFrame(proxy_rows)
    proxy_agg = _aggregate_proxy(proxy)
    decisions = _decision_rows(proxy_agg)

    paths = {
        "summary": output_dir / "clean_first_touch_label_ablation_summary.csv",
        "oracle": output_dir / "clean_first_touch_label_ablation_oracle.csv",
        "feature_ic": output_dir / "clean_first_touch_label_ablation_feature_ic_top25.csv",
        "proxy_monthly": output_dir / "clean_first_touch_label_ablation_proxy_monthly.csv",
        "proxy_aggregate": output_dir / "clean_first_touch_label_ablation_proxy_aggregate.csv",
        "decisions": output_dir / "clean_first_touch_label_ablation_decisions.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    oracle.to_csv(paths["oracle"], index=False)
    feature_ic.to_csv(paths["feature_ic"], index=False)
    proxy.to_csv(paths["proxy_monthly"], index=False)
    proxy_agg.to_csv(paths["proxy_aggregate"], index=False)
    decisions.to_csv(paths["decisions"], index=False)

    manifest = {
        "scope": "no_training_clean_first_touch_label_ablation",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "metric_prefix": str(metric_prefix),
        "primary_prefix": str(primary_prefix),
        "challenger_prefix": str(challenger_prefix),
        "arm_descriptions": {arm.name: arm.description for arm in arms},
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "top_fracs": list(TOP_FRACS),
        "decision_gate": {
            "months": ">=3",
            "positive_months": ">=3",
            "mean_first_touch_net": ">0.0025",
            "worst_month_first_touch_net": ">=0",
            "clean_first_touch_exec_rate": ">=0.65",
            "bad_first_touch_mae_to_sl_rate": "<=0.25",
            "p90_first_touch_mae_to_sl": "<=2.00",
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=summary,
        oracle=oracle,
        proxy_agg=proxy_agg,
        decisions=decisions,
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
    parser.add_argument("--metric-prefix", default=DEFAULT_METRIC_PREFIX)
    parser.add_argument("--primary-prefix", default=DEFAULT_PRIMARY_PREFIX)
    parser.add_argument("--challenger-prefix", default=DEFAULT_CHALLENGER_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        metric_prefix=str(args.metric_prefix),
        primary_prefix=str(args.primary_prefix),
        challenger_prefix=str(args.challenger_prefix),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
