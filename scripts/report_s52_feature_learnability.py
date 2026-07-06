#!/usr/bin/env python3
"""Diagnose whether S52 path-ordered labels are learnable from current features.

The S52 ranker smoke showed that the materialized target has positive top-k
structure, while the trained ranker does not recover it. This report evaluates
each pre-entry feature as a univariate selector, by month and side, using the
same top-k path metrics as the Gate 3 HPO/ranker reports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_ROUND_TRIP_COST,
    _prepare_folds,
    _top_metrics,
)
from scripts.run_s52_ranker_smoke import (  # noqa: E402
    DEFAULT_BEST_CONFIG,
    DEFAULT_MONTHS,
    _materialized_soft_label,
)


DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260705_s52_bidirectional_first_touch_sidegeom_tp125_lsl075_ssl050_fast16_bar50_cost100bps_labels/"
    "labels"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_feature_learnability_20260705_v1")
TOP_SORT_METRIC = "top10_mean_first_touch_net"
EXCLUDE_STORE_FEATURE_PREFIXES = ("__",)
EXCLUDE_STORE_FEATURE_NAMES = {"timestamp", "ts", "symbol"}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_mean(values: pd.Series | list[float]) -> float:
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")


def _safe_min(values: pd.Series | list[float]) -> float:
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.min()) if len(s) else float("nan")


def _safe_corr(a: pd.Series, b: pd.Series, *, method: str = "spearman") -> float:
    aa = pd.to_numeric(a.reset_index(drop=True), errors="coerce")
    bb = pd.to_numeric(b.reset_index(drop=True), errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 20:
        return float("nan")
    if aa.loc[mask].nunique(dropna=True) < 3 or bb.loc[mask].nunique(dropna=True) < 3:
        return float("nan")
    return _safe_float(aa.loc[mask].corr(bb.loc[mask], method=method))


def _sample_for_prefilter(
    *series: pd.Series,
    max_rows: int,
) -> tuple[pd.Series, ...]:
    if not series:
        return ()
    n = len(series[0])
    if max_rows <= 0 or n <= max_rows:
        return tuple(s.reset_index(drop=True) for s in series)
    step = max(1, int(math.ceil(n / float(max_rows))))
    idx = np.arange(0, n, step, dtype=np.int64)[:max_rows]
    return tuple(s.reset_index(drop=True).iloc[idx].reset_index(drop=True) for s in series)


def _feature_family(name: str) -> str:
    n = str(name).lower()
    tokens = (
        ("gmm", "state_gmm"),
        ("cluster", "state_cluster"),
        ("reconstruction", "state_ae"),
        ("mahal", "state_mahal"),
        ("entropy", "state_entropy"),
        ("funding", "funding"),
        ("open_interest", "open_interest"),
        ("oi_", "open_interest"),
        ("spread", "microstructure"),
        ("basis", "basis"),
        ("vol", "volatility"),
        ("atr", "volatility"),
        ("bb", "bollinger"),
        ("ema", "trend"),
        ("rsi", "momentum"),
        ("ret", "returns"),
        ("return", "returns"),
        ("volume", "volume"),
        ("corr", "cross_asset"),
        ("beta", "cross_asset"),
        ("market", "market"),
        ("btc", "cross_asset"),
        ("eth", "cross_asset"),
    )
    for token, family in tokens:
        if token in n:
            return family
    return "other"


def _store_schema_features(feature_dir: Path) -> list[str]:
    files = sorted(feature_dir.glob("symbol=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No symbol=*.parquet feature files found under {feature_dir}")
    names: list[str]
    try:
        import pyarrow.parquet as pq

        names = [str(v) for v in pq.read_schema(files[0]).names]
    except Exception:
        names = [str(v) for v in pd.read_parquet(files[0]).columns]
    out: list[str] = []
    for name in names:
        lower = name.lower()
        if lower in EXCLUDE_STORE_FEATURE_NAMES:
            continue
        if any(name.startswith(prefix) for prefix in EXCLUDE_STORE_FEATURE_PREFIXES):
            continue
        out.append(name)
    if not out:
        raise ValueError(f"No usable feature columns found in {files[0]}")
    return out


def _feature_list_for_scope(
    *,
    feature_scope: str,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    max_features: int | None,
) -> Path:
    if feature_scope == "selected":
        return feature_list_csv
    if feature_scope != "all-store":
        raise ValueError(f"unknown feature scope: {feature_scope}")
    features = _store_schema_features(feature_dir)
    if max_features is not None and int(max_features) > 0:
        features = features[: int(max_features)]
    path = output_dir / "s52_all_store_feature_list.csv"
    pd.DataFrame({"feature": features}).to_csv(path, index=False)
    return path


def _subset(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> pd.DataFrame:
    return frame.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)


def _segment_masks(metrics: pd.DataFrame) -> dict[str, np.ndarray]:
    side = pd.to_numeric(metrics["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    return {
        "all": np.ones(len(metrics), dtype=bool),
        "long": side > 0.0,
        "short": side < 0.0,
    }


def _score_metrics(
    *,
    score: pd.Series,
    label: pd.DataFrame,
    metrics: pd.DataFrame,
    round_trip_cost: float,
) -> dict[str, float]:
    return _top_metrics(
        score=score.reset_index(drop=True),
        label=label.reset_index(drop=True),
        metrics=metrics.reset_index(drop=True),
        round_trip_cost=float(round_trip_cost),
    )


def _choose_polarity(
    *,
    score: pd.Series,
    label: pd.DataFrame,
    metrics: pd.DataFrame,
    round_trip_cost: float,
) -> tuple[str, dict[str, float]]:
    pos = _score_metrics(score=score, label=label, metrics=metrics, round_trip_cost=round_trip_cost)
    neg = _score_metrics(score=-score, label=label, metrics=metrics, round_trip_cost=round_trip_cost)
    pos_key = _safe_float(pos.get(TOP_SORT_METRIC))
    neg_key = _safe_float(neg.get(TOP_SORT_METRIC))
    if neg_key > pos_key:
        return "negative", neg
    if pos_key > neg_key:
        return "positive", pos
    pos_ev = _safe_float(pos.get("top10_ev_weighted_first_touch_precision"))
    neg_ev = _safe_float(neg.get("top10_ev_weighted_first_touch_precision"))
    return ("negative", neg) if neg_ev > pos_ev else ("positive", pos)


def _oracle_rows(
    *,
    fold: dict[str, Any],
    label: pd.DataFrame,
    round_trip_cost: float,
    min_rows: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metrics = fold["valid_metrics"].reset_index(drop=True)
    for segment, mask in _segment_masks(metrics).items():
        if int(mask.sum()) < int(min_rows):
            continue
        seg_label = _subset(label, mask)
        seg_metrics = _subset(metrics, mask)
        scores = {
            "oracle_target_soft": pd.to_numeric(seg_label["target_soft"], errors="coerce"),
            "oracle_first_touch_net": pd.to_numeric(seg_metrics["first_touch_net"], errors="coerce"),
            "oracle_low_full_path_mae": -pd.to_numeric(
                seg_metrics.get("first_touch_full_path_mae_norm", seg_metrics["mae_norm"]),
                errors="coerce",
            ),
        }
        for name, score in scores.items():
            row: dict[str, Any] = {
                "month": fold["month"],
                "segment": segment,
                "selector": name,
                "rows": int(mask.sum()),
            }
            row.update(_score_metrics(score=score, label=seg_label, metrics=seg_metrics, round_trip_cost=round_trip_cost))
            out.append(row)
    return out


def _feature_rows(
    *,
    fold: dict[str, Any],
    label: pd.DataFrame,
    round_trip_cost: float,
    min_rows: int,
    candidate_features_per_segment: int,
    prefilter_sample_rows: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    x_valid = fold["x_valid"].reset_index(drop=True)
    metrics = fold["valid_metrics"].reset_index(drop=True)
    target = pd.to_numeric(label["target_soft"], errors="coerce").reset_index(drop=True)
    first_touch_net = pd.to_numeric(metrics["first_touch_net"], errors="coerce").reset_index(drop=True)
    first_good = pd.to_numeric(label["first_pass_good"], errors="coerce").reset_index(drop=True)
    full_path_mae = pd.to_numeric(
        metrics.get("first_touch_full_path_mae_norm", metrics["mae_norm"]),
        errors="coerce",
    ).reset_index(drop=True)

    for segment, mask in _segment_masks(metrics).items():
        if int(mask.sum()) < int(min_rows):
            continue
        seg_x = _subset(x_valid, mask)
        seg_label = _subset(label, mask)
        seg_metrics = _subset(metrics, mask)
        seg_target = target.loc[mask].reset_index(drop=True)
        seg_ft_net = first_touch_net.loc[mask].reset_index(drop=True)
        seg_first_good = first_good.loc[mask].reset_index(drop=True)
        seg_full_path_mae = full_path_mae.loc[mask].reset_index(drop=True)
        feature_candidates = _candidate_features(
            seg_x=seg_x,
            seg_target=seg_target,
            seg_ft_net=seg_ft_net,
            seg_first_good=seg_first_good,
            seg_full_path_mae=seg_full_path_mae,
            min_rows=min_rows,
            candidate_features_per_segment=candidate_features_per_segment,
            prefilter_sample_rows=prefilter_sample_rows,
        )
        print(
            f"[feature-learnability] month={fold['month']} segment={segment} "
            f"rows={len(seg_x)} features={len(feature_candidates)}/{len(seg_x.columns)}",
            flush=True,
        )
        for feature in feature_candidates:
            score = pd.to_numeric(seg_x[feature], errors="coerce")
            finite = score.replace([np.inf, -np.inf], np.nan).notna()
            if int(finite.sum()) < int(min_rows) or score.loc[finite].nunique(dropna=True) < 3:
                continue
            polarity, metrics_out = _choose_polarity(
                score=score,
                label=seg_label,
                metrics=seg_metrics,
                round_trip_cost=round_trip_cost,
            )
            row: dict[str, Any] = {
                "month": fold["month"],
                "segment": segment,
                "feature": feature,
                "feature_family": _feature_family(feature),
                "polarity": polarity,
                "rows": int(len(score)),
                "finite_rate": float(finite.mean()),
                "spearman_target_soft": _safe_corr(score, seg_target, method="spearman"),
                "spearman_first_touch_net": _safe_corr(score, seg_ft_net, method="spearman"),
                "spearman_first_pass_good": _safe_corr(score, seg_first_good, method="spearman"),
                "spearman_full_path_mae": _safe_corr(score, seg_full_path_mae, method="spearman"),
            }
            row.update(metrics_out)
            out.append(row)
    return out


def _candidate_features(
    *,
    seg_x: pd.DataFrame,
    seg_target: pd.Series,
    seg_ft_net: pd.Series,
    seg_first_good: pd.Series,
    seg_full_path_mae: pd.Series,
    min_rows: int,
    candidate_features_per_segment: int,
    prefilter_sample_rows: int,
) -> list[str]:
    cols = list(seg_x.columns)
    if candidate_features_per_segment <= 0 or len(cols) <= candidate_features_per_segment:
        return cols
    target_s, ft_s, good_s, mae_s = _sample_for_prefilter(
        seg_target,
        seg_ft_net,
        seg_first_good,
        seg_full_path_mae,
        max_rows=prefilter_sample_rows,
    )
    rows: list[tuple[float, str]] = []
    for feature in cols:
        score = pd.to_numeric(seg_x[feature], errors="coerce")
        (score_s,) = _sample_for_prefilter(score, max_rows=prefilter_sample_rows)
        finite = score_s.replace([np.inf, -np.inf], np.nan).notna()
        if int(finite.sum()) < min(100, int(min_rows)) or score_s.loc[finite].nunique(dropna=True) < 3:
            continue
        corrs = [
            abs(_safe_corr(score_s, target_s, method="spearman")),
            abs(_safe_corr(score_s, ft_s, method="spearman")),
            abs(_safe_corr(score_s, good_s, method="spearman")),
            abs(_safe_corr(score_s, mae_s, method="spearman")),
        ]
        best = max((v for v in corrs if math.isfinite(v)), default=float("nan"))
        if math.isfinite(best):
            rows.append((best, feature))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [feature for _score, feature in rows[: int(candidate_features_per_segment)]]


def _aggregate_features(feature_rows: pd.DataFrame) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame()
    group_cols = ["segment", "feature", "feature_family"]
    metric_cols = [
        "top10_mean_first_touch_net",
        "top20_mean_first_touch_net",
        "top30_mean_first_touch_net",
        "top10_ev_weighted_first_touch_precision",
        "top20_ev_weighted_first_touch_precision",
        "top30_ev_weighted_first_touch_precision",
        "top10_clean_precision",
        "top10_first_pass_good_rate",
        "top10_first_touch_bad_mae_1r_rate",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_mfe_1r_before_mae_1r_rate",
        "top10_mae_1r_before_mfe_1r_rate",
        "top10_timeout_rate",
        "spearman_target_soft",
        "spearman_first_touch_net",
        "spearman_first_pass_good",
        "spearman_full_path_mae",
        "finite_rate",
    ]
    rows: list[dict[str, Any]] = []
    for keys, part in feature_rows.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(part["month"].nunique())
        row["dominant_polarity"] = str(part["polarity"].mode(dropna=True).iloc[0]) if len(part["polarity"].mode()) else ""
        for col in metric_cols:
            if col not in part.columns:
                continue
            row[f"mean_{col}"] = _safe_mean(part[col])
            row[f"min_{col}"] = _safe_min(part[col])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [
        "mean_top10_mean_first_touch_net",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_first_pass_good_rate",
    ]
    return out.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    oracle_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# S52 Feature Learnability Report")
    lines.append("")
    lines.append(f"Rows: `{manifest.get('rows')}`")
    lines.append(f"Symbols: `{manifest.get('symbols')}`")
    lines.append(f"Fold months: `{', '.join(map(str, manifest.get('fold_months', [])))}`")
    lines.append(f"Feature count: `{manifest.get('features')}`")
    lines.append("")
    lines.append("## Oracle Baselines")
    lines.append("")
    keep = [
        "month",
        "segment",
        "selector",
        "top10_mean_first_touch_net",
        "top10_ev_weighted_first_touch_precision",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_timeout_rate",
    ]
    if not oracle_df.empty:
        lines.append(oracle_df[[c for c in keep if c in oracle_df.columns]].round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Best Univariate Features")
    lines.append("")
    keep = [
        "segment",
        "feature",
        "feature_family",
        "dominant_polarity",
        "mean_top10_mean_first_touch_net",
        "min_top10_mean_first_touch_net",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_spearman_target_soft",
        "mean_spearman_first_touch_net",
    ]
    if not summary_df.empty:
        for segment in ("all", "long", "short"):
            part = summary_df[summary_df["segment"].eq(segment)].head(15)
            lines.append(f"### {segment.title()}")
            lines.append("")
            lines.append(part[[c for c in keep if c in part.columns]].round(6).to_markdown(index=False))
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        lines.append(
            "- Best univariate selector: "
            f"`{best.get('feature')}` / `{best.get('segment')}` with "
            f"mean top10 first-touch net `{_safe_float(best.get('mean_top10_mean_first_touch_net')):.6f}`."
        )
        strong = summary_df[pd.to_numeric(summary_df.get("mean_top10_mean_first_touch_net"), errors="coerce") > 0.0]
        lines.append(f"- Positive mean top10 first-touch-net univariate selectors: `{len(strong)}`.")
    lines.append(
        "- If oracle target-soft is positive but univariate features are weak, the next repair is feature expansion/"
        "selection rather than another stricter label blend."
    )
    (output_dir / "s52_feature_learnability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    best_config_path: Path,
    output_dir: Path,
    months: list[str],
    round_trip_cost: float,
    feature_scope: str,
    max_features: int | None,
    candidate_features_per_segment: int,
    prefilter_sample_rows: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    min_segment_rows: int,
    seed: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scoped_feature_list_csv = _feature_list_for_scope(
        feature_scope=feature_scope,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        output_dir=output_dir,
        max_features=max_features,
    )
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=scoped_feature_list_csv,
        months=months,
        spread_baseline_path=None,
        spread_rank_column="p75_spread_bps",
        target_symbol_count=None,
        max_feature_store_features=None,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
        seed=seed,
    )
    # Load config to make the manifest explicit; materialized S52 labels are used
    # for the actual diagnostic so stale HPO config values cannot drive the target.
    config_payload = json.loads(best_config_path.read_text(encoding="utf-8"))
    oracle_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    if candidate_features_per_segment <= 0 and feature_scope == "all-store":
        candidate_features_per_segment = 160
    for fold in folds:
        label = _materialized_soft_label(fold["valid_frame"], fold["valid_metrics"]).reset_index(drop=True)
        oracle_rows.extend(
            _oracle_rows(
                fold=fold,
                label=label,
                round_trip_cost=round_trip_cost,
                min_rows=min_segment_rows,
            )
        )
        feature_rows.extend(
            _feature_rows(
                fold=fold,
                label=label,
                round_trip_cost=round_trip_cost,
                min_rows=min_segment_rows,
                candidate_features_per_segment=int(candidate_features_per_segment),
                prefilter_sample_rows=int(prefilter_sample_rows),
            )
        )
    oracle_df = pd.DataFrame(oracle_rows)
    feature_df = pd.DataFrame(feature_rows)
    summary_df = _aggregate_features(feature_df)
    paths = {
        "manifest": output_dir / "manifest.json",
        "oracle": output_dir / "s52_feature_learnability_oracle.csv",
        "feature_folds": output_dir / "s52_feature_learnability_feature_folds.csv",
        "feature_summary": output_dir / "s52_feature_learnability_feature_summary.csv",
        "report": output_dir / "s52_feature_learnability_report.md",
    }
    oracle_df.to_csv(paths["oracle"], index=False)
    feature_df.to_csv(paths["feature_folds"], index=False)
    summary_df.to_csv(paths["feature_summary"], index=False)
    manifest_out = {
        **{k: str(v) for k, v in manifest.items()},
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "scoped_feature_list_csv": str(scoped_feature_list_csv),
        "feature_scope": str(feature_scope),
        "max_features": max_features,
        "candidate_features_per_segment": int(candidate_features_per_segment),
        "prefilter_sample_rows": int(prefilter_sample_rows),
        "best_config_path": str(best_config_path),
        "round_trip_cost": float(round_trip_cost),
        "include_ae_gmm_state_features": bool(include_ae_gmm_state_features),
        "ae_gmm_state_feature_max_train_rows": int(ae_gmm_state_feature_max_train_rows),
        "ae_gmm_state_feature_max_iter": int(ae_gmm_state_feature_max_iter),
        "min_segment_rows": int(min_segment_rows),
        "config_label_name": config_payload.get("label_name"),
    }
    paths["manifest"].write_text(json.dumps(manifest_out, indent=2, default=str) + "\n", encoding="utf-8")
    _write_report(output_dir=output_dir, manifest=manifest, oracle_df=oracle_df, summary_df=summary_df)
    return {k: str(v) for k, v in paths.items()}


def _parse_months(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--best-config-path", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--feature-scope", choices=("selected", "all-store"), default="selected")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument(
        "--candidate-features-per-segment",
        type=int,
        default=0,
        help="Run full top-k metrics only on the best cheap-correlation candidates per fold/side segment; "
        "0 means all selected features, or 160 for all-store.",
    )
    parser.add_argument("--prefilter-sample-rows", type=int, default=20000)
    parser.add_argument("--include-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=30_000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=32)
    parser.add_argument("--min-segment-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_report(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        best_config_path=args.best_config_path,
        output_dir=args.output_dir,
        months=_parse_months(args.months),
        round_trip_cost=float(args.round_trip_cost),
        feature_scope=str(args.feature_scope),
        max_features=args.max_features,
        candidate_features_per_segment=int(args.candidate_features_per_segment),
        prefilter_sample_rows=int(args.prefilter_sample_rows),
        include_ae_gmm_state_features=bool(args.include_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        min_segment_rows=int(args.min_segment_rows),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
