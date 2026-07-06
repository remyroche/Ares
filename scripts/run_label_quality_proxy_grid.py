#!/usr/bin/env python3
"""Grid-search risk-adjusted policy-net labels with proxy tests only.

This is a pre-training label QA tool. It does not fit LightGBM, Optuna, or
policy geometry. The grid is intentionally cheap: rank by each candidate label,
filter by economic envelope, then run an out-of-time univariate feature proxy on
the shortlist.
"""

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

from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    TOP_FRACS,
    _aggregate_proxy,
    _decile_diagnostics,
    _feature_columns,
    _feature_ic,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _proxy_oos_rows,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _safe_std,
    _selection_metrics,
    _sigmoid,
    _spearman,
    _weekly_selection_rows,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_quality_proxy_grid_s13_v1")
DEFAULT_MAX_PROXY_CANDIDATES = 28


@dataclass(frozen=True)
class GridSpec:
    arm: str
    family: str
    mae_penalty: float
    mae_free: float
    time_penalty: float
    barrier_penalty: float
    barrier_free: float
    margin: float
    temperature: float


def _candidate_specs() -> list[GridSpec]:
    specs: list[GridSpec] = []
    specs.append(
        GridSpec(
            arm="baseline_s10_soft_temp012",
            family="raw_policy_net",
            mae_penalty=0.0,
            mae_free=0.0,
            time_penalty=0.0,
            barrier_penalty=0.0,
            barrier_free=0.0,
            margin=0.0,
            temperature=0.012,
        )
    )
    for mae_penalty in (0.0025, 0.0040, 0.0060):
        for mae_free in (0.50, 0.75, 1.00):
            for time_penalty in (0.0, 0.00075, 0.00150):
                for barrier_penalty in (0.0, 0.20, 0.35):
                    for margin in (0.0, 0.0015, 0.0030):
                        specs.append(
                            GridSpec(
                                arm=(
                                    "s13"
                                    f"_mae{mae_penalty:g}"
                                    f"_free{mae_free:g}"
                                    f"_time{time_penalty:g}"
                                    f"_bar{barrier_penalty:g}"
                                    "_bf0.018"
                                    f"_m{margin:g}"
                                    "_t0.008"
                                ),
                                family="risk_adjusted_policy_net",
                                mae_penalty=mae_penalty,
                                mae_free=mae_free,
                                time_penalty=time_penalty,
                                barrier_penalty=barrier_penalty,
                                barrier_free=0.018,
                                margin=margin,
                                temperature=0.008,
                            )
                        )
    return specs


def _build_target(metrics: pd.DataFrame, spec: GridSpec) -> tuple[pd.DataFrame, pd.Series]:
    u = metrics["u_policy_net"].fillna(-0.02)
    if spec.family == "raw_policy_net":
        risk_u = u - spec.margin
    else:
        risk_u = (
            u
            - spec.margin
            - spec.mae_penalty * (metrics["mae_norm"] - spec.mae_free).clip(lower=0.0)
            - spec.time_penalty * np.log1p(metrics["bars_to_mfe"].clip(lower=0.0))
            - spec.barrier_penalty * (metrics["barrier"] - spec.barrier_free).clip(lower=0.0)
        )
    soft = pd.Series(_sigmoid(risk_u / max(spec.temperature, 1e-12)), index=metrics.index).clip(0.0, 1.0)
    hard = (risk_u > 0.0).fillna(False).astype(float)
    return pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=metrics.index), risk_u


def _summarise_target(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    risk_u: pd.Series,
    features: list[str],
    spec: GridSpec,
) -> dict[str, Any]:
    soft = target["target_soft"]
    hard = target["target_hard"]
    ic = _feature_ic(frame, features, soft)
    top_ic = ic.head(8)
    row = {
        "arm": spec.arm,
        "family": spec.family,
        "mae_penalty": spec.mae_penalty,
        "mae_free": spec.mae_free,
        "time_penalty": spec.time_penalty,
        "barrier_penalty": spec.barrier_penalty,
        "barrier_free": spec.barrier_free,
        "margin": spec.margin,
        "temperature": spec.temperature,
        "rows": int(len(frame)),
        "soft_mean": _safe_mean(soft),
        "soft_std": _safe_std(soft),
        "soft_low_sat_rate": _safe_mean(soft <= 0.05),
        "soft_high_sat_rate": _safe_mean(soft >= 0.95),
        "hard_rate": _safe_mean(hard),
        "risk_u_mean": _safe_mean(risk_u),
        "risk_u_q10": _safe_quantile(risk_u, 0.10),
        "ic_soft_vs_u": _spearman(soft, metrics["u_policy_net"]),
        "ic_soft_vs_mae_norm": _spearman(soft, metrics["mae_norm"]),
        "ic_soft_vs_barrier": _spearman(soft, metrics["barrier"]),
        "ic_soft_vs_bars_to_mfe": _spearman(soft, metrics["bars_to_mfe"]),
        "feature_mean_top_abs_ic": float(top_ic["abs_ic"].mean()) if len(top_ic) else float("nan"),
        "feature_top_abs_ic": float(top_ic["abs_ic"].iloc[0]) if len(top_ic) else float("nan"),
        "feature_n_abs_ic_ge_002": int((ic["abs_ic"] >= 0.02).sum()) if not ic.empty else 0,
        "feature_n_abs_ic_ge_005": int((ic["abs_ic"] >= 0.05).sum()) if not ic.empty else 0,
        "feature_top_names": ",".join(top_ic["feature"].astype(str).tolist()) if len(top_ic) else "",
    }
    row.update(_decile_diagnostics(soft, metrics["u_policy_net"]))
    return row


def _oracle_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    spec: GridSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frac in TOP_FRACS:
        row = _selection_metrics(
            frame=frame,
            metrics=metrics,
            target=target,
            score=target["target_soft"],
            arm=spec.arm,
            selector="oracle_label_sort",
            period="all",
            top_frac=frac,
        )
        row.update(
            {
                "family": spec.family,
                "mae_penalty": spec.mae_penalty,
                "mae_free": spec.mae_free,
                "time_penalty": spec.time_penalty,
                "barrier_penalty": spec.barrier_penalty,
                "barrier_free": spec.barrier_free,
                "margin": spec.margin,
                "temperature": spec.temperature,
            }
        )
        rows.append(row)
    return rows


def _weekly_aggregate(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return weekly
    return (
        weekly.groupby("arm", observed=True)
        .agg(
            weeks=("period", "nunique"),
            positive_weeks=("mean_u", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            q25_week_mean_u=("mean_u", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.25))),
            worst_week_mean_u=("mean_u", lambda s: float(pd.to_numeric(s, errors="coerce").min())),
        )
        .reset_index()
    )


def _score_shortlist(
    *,
    summary: pd.DataFrame,
    oracle: pd.DataFrame,
    weekly_agg: pd.DataFrame,
    max_proxy_candidates: int,
) -> pd.DataFrame:
    top30 = oracle[oracle["top_frac"].eq(0.30)].copy()
    top10 = oracle[oracle["top_frac"].eq(0.10)].copy()
    merged = top30.merge(
        top10[
            [
                "arm",
                "mean_u",
                "q10_u",
                "hit_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
            ]
        ].rename(
            columns={
                "mean_u": "top10_mean_u",
                "q10_u": "top10_q10_u",
                "hit_u": "top10_hit_u",
                "bad_mae_1r_rate": "top10_bad_mae_1r_rate",
                "wide_barrier_25bps_rate": "top10_wide_barrier_25bps_rate",
            }
        ),
        on="arm",
        how="left",
    )
    merged = merged.merge(
        summary[
            [
                "arm",
                "feature_mean_top_abs_ic",
                "feature_n_abs_ic_ge_002",
                "ic_soft_vs_u",
                "soft_std",
                "hard_rate",
            ]
        ],
        on="arm",
        how="left",
    )
    merged = merged.merge(weekly_agg, on="arm", how="left")
    merged["oracle_pass"] = (
        (merged["mean_u"] >= 0.0060)
        & (merged["top10_mean_u"] >= 0.0200)
        & (merged["bad_mae_1r_rate"] <= 0.40)
        & (merged["wide_barrier_25bps_rate"] <= 0.04)
        & (merged["q10_u"] >= -0.0115)
        & (merged["q25_week_mean_u"] >= 0.0030)
        & (merged["feature_mean_top_abs_ic"] >= 0.08)
        & (merged["hard_rate"].between(0.05, 0.35))
    )
    merged["pre_proxy_score"] = (
        250.0 * pd.to_numeric(merged["mean_u"], errors="coerce").fillna(-1.0)
        + 80.0 * pd.to_numeric(merged["top10_mean_u"], errors="coerce").fillna(-1.0)
        + 40.0 * pd.to_numeric(merged["q25_week_mean_u"], errors="coerce").fillna(-1.0)
        + 2.0 * pd.to_numeric(merged["feature_mean_top_abs_ic"], errors="coerce").fillna(0.0)
        - 0.70 * pd.to_numeric(merged["bad_mae_1r_rate"], errors="coerce").fillna(1.0)
        - 1.00 * pd.to_numeric(merged["wide_barrier_25bps_rate"], errors="coerce").fillna(1.0)
    )
    passed = merged[merged["oracle_pass"]].copy()
    if passed.empty:
        passed = merged.copy()
    baseline = merged[merged["arm"].eq("baseline_s10_soft_temp012")].copy()
    ranked = passed.sort_values(
        ["pre_proxy_score", "mean_u", "feature_mean_top_abs_ic"],
        ascending=[False, False, False],
    ).head(max_proxy_candidates)
    out = pd.concat([baseline, ranked], ignore_index=True)
    out = out.drop_duplicates(subset=["arm"], keep="first")
    return out.sort_values("pre_proxy_score", ascending=False)


def _write_markdown(
    *,
    output_dir: Path,
    shortlisted: pd.DataFrame,
    proxy_agg: pd.DataFrame,
    oracle: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_quality_proxy_grid.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    def sorted_proxy(frac: float) -> pd.DataFrame:
        return proxy_agg[proxy_agg["top_frac"].eq(frac)].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )

    def proxy_section(title: str, frac: float) -> list[str]:
        return [
            f"## {title}",
            "",
            table(
                sorted_proxy(frac),
                [
                    "arm",
                    "months",
                    "positive_months",
                    "mean_u",
                    "worst_month_mean_u",
                    "hit_u",
                    "q10_u",
                    "delta_mean_u_vs_period",
                    "delta_hit_u_vs_period",
                    "proxy_ic_u",
                    "bad_mae_1r_rate",
                    "wide_barrier_25bps_rate",
                    "top_symbol_share",
                ],
                limit=30,
            ),
            "",
        ]

    oracle_top30 = oracle[oracle["top_frac"].eq(0.30)].sort_values(
        ["mean_u", "bad_mae_1r_rate"],
        ascending=[False, True],
    )
    lines = [
        "# Label Quality Proxy Grid",
        "",
        "Scope: S13-style risk-adjusted policy-net label grid. No model training or policy geometry optimization.",
        "",
        "Shortlisting requires oracle economics, bad-MAE/barrier envelope, weekly stability, and feature association before OOT proxy testing.",
        "",
        "## Shortlist Before Proxy",
        "",
        table(
            shortlisted,
            [
                "arm",
                "pre_proxy_score",
                "oracle_pass",
                "mean_u",
                "q10_u",
                "top10_mean_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "q25_week_mean_u",
                "feature_mean_top_abs_ic",
                "hard_rate",
            ],
            limit=30,
        ),
        "",
        *proxy_section("Proxy OOS Top 30%", 0.30),
        *proxy_section("Proxy OOS Top 10%", 0.10),
        *proxy_section("Proxy OOS Top 5%", 0.05),
        *proxy_section("Proxy OOS Top 3%", 0.03),
        *proxy_section("Proxy OOS Top 1%", 0.01),
        "## Best Oracle Top 30%",
        "",
        table(
            oracle_top30,
            [
                "arm",
                "mean_u",
                "q10_u",
                "hit_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "mean_bars_to_mfe",
                "mae_penalty",
                "mae_free",
                "time_penalty",
                "barrier_penalty",
                "margin",
            ],
            limit=30,
        ),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Oracle metrics: `{manifest['outputs']['oracle']}`",
        f"- Weekly metrics: `{manifest['outputs']['weekly']}`",
        f"- Shortlist: `{manifest['outputs']['shortlist']}`",
        f"- Proxy OOS monthly: `{manifest['outputs']['proxy_monthly']}`",
        f"- Proxy OOS aggregate: `{manifest['outputs']['proxy_aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_grid(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    max_proxy_candidates: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(
        feature_list_csv,
        max_features=max_feature_store_features,
    )
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    features = _feature_columns(frame)

    specs = _candidate_specs()
    target_cache: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []

    for spec in specs:
        target, risk_u = _build_target(metrics, spec)
        target_cache[spec.arm] = target
        summary_rows.append(
            _summarise_target(
                frame=frame,
                metrics=metrics,
                target=target,
                risk_u=risk_u,
                features=features,
                spec=spec,
            )
        )
        oracle_rows.extend(_oracle_rows(frame=frame, metrics=metrics, target=target, spec=spec))
        weekly = _weekly_selection_rows(
            frame=frame,
            metrics=metrics,
            target=target,
            score=target["target_soft"],
            arm=spec.arm,
            selector="oracle_label_sort",
            top_frac=0.30,
        )
        for row in weekly:
            row.update(
                {
                    "family": spec.family,
                    "mae_penalty": spec.mae_penalty,
                    "mae_free": spec.mae_free,
                    "time_penalty": spec.time_penalty,
                    "barrier_penalty": spec.barrier_penalty,
                    "barrier_free": spec.barrier_free,
                    "margin": spec.margin,
                    "temperature": spec.temperature,
                }
            )
        weekly_rows.extend(weekly)

    summary = pd.DataFrame(summary_rows)
    oracle = pd.DataFrame(oracle_rows)
    weekly = pd.DataFrame(weekly_rows)
    weekly_agg = _weekly_aggregate(weekly)
    shortlist = _score_shortlist(
        summary=summary,
        oracle=oracle,
        weekly_agg=weekly_agg,
        max_proxy_candidates=max_proxy_candidates,
    )

    proxy_rows: list[dict[str, Any]] = []
    spec_by_arm = {spec.arm: spec for spec in specs}
    for arm in shortlist["arm"].astype(str).tolist():
        target = target_cache[arm]
        spec = spec_by_arm[arm]
        rows = _proxy_oos_rows(
            frame=frame,
            metrics=metrics,
            target=target,
            features=features,
            arm=arm,
        )
        for row in rows:
            row.update(
                {
                    "family": spec.family,
                    "mae_penalty": spec.mae_penalty,
                    "mae_free": spec.mae_free,
                    "time_penalty": spec.time_penalty,
                    "barrier_penalty": spec.barrier_penalty,
                    "barrier_free": spec.barrier_free,
                    "margin": spec.margin,
                    "temperature": spec.temperature,
                }
            )
        proxy_rows.extend(rows)
    proxy_monthly = pd.DataFrame(proxy_rows)
    proxy_agg = _aggregate_proxy(proxy_monthly)

    paths = {
        "summary": output_dir / "grid_label_summary.csv",
        "oracle": output_dir / "grid_oracle_selection_metrics.csv",
        "weekly": output_dir / "grid_weekly_oracle_selection_metrics.csv",
        "weekly_aggregate": output_dir / "grid_weekly_oracle_aggregate.csv",
        "shortlist": output_dir / "grid_shortlist_pre_proxy.csv",
        "proxy_monthly": output_dir / "grid_feature_ic_proxy_oos_monthly.csv",
        "proxy_aggregate": output_dir / "grid_feature_ic_proxy_oos_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    oracle.to_csv(paths["oracle"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    weekly_agg.to_csv(paths["weekly_aggregate"], index=False)
    shortlist.to_csv(paths["shortlist"], index=False)
    proxy_monthly.to_csv(paths["proxy_monthly"], index=False)
    proxy_agg.to_csv(paths["proxy_aggregate"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "grid_size": int(len(specs)),
        "shortlist_size": int(len(shortlist)),
        "proxy_candidate_limit": int(max_proxy_candidates),
        "features": features,
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "top_fracs": list(TOP_FRACS),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        shortlisted=shortlist,
        proxy_agg=proxy_agg,
        oracle=oracle,
        summary=summary,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--max-proxy-candidates", type=int, default=DEFAULT_MAX_PROXY_CANDIDATES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_grid(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        max_proxy_candidates=int(args.max_proxy_candidates),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
