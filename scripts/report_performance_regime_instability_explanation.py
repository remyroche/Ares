#!/usr/bin/env python3
"""Summarise how much strategy instability is explained by performance regimes."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            if columns is not None:
                return pd.read_parquet(path)
            raise
    frame = pd.read_csv(path)
    if columns is not None:
        return frame.loc[:, [c for c in columns if c in frame.columns]]
    return frame


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _strategy_from_scope(scope_root: Path) -> str:
    name = scope_root.name
    if name.startswith("head_"):
        return name.replace("head_", "", 1)
    return "global"


def _iter_scope_roots(run_root: Path) -> list[tuple[str, Path]]:
    head_roots = sorted(path for path in run_root.glob("head_*") if path.is_dir())
    if head_roots:
        return [(_strategy_from_scope(path), path) for path in head_roots]
    return [(_strategy_from_scope(run_root), run_root)]


def _iter_fold_roots(scope_root: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in sorted(scope_root.glob("fold_*")):
        if not path.is_dir():
            continue
        try:
            fold_id = int(path.name.rsplit("_", 1)[1])
        except Exception:
            fold_id = len(out) + 1
        out.append((fold_id, path))
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan
    if weights is None:
        return float(np.nanmean(x[ok]))
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok &= np.isfinite(w) & (w >= 0.0)
    if not ok.any():
        return np.nan
    return float(np.average(x[ok], weights=np.maximum(w[ok], 1e-12)))


def _parse_sequence_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist() if str(v)]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("("):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed if str(v)]
        except Exception:
            pass
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


def _family(feature: str) -> str:
    text = str(feature)
    if "__" in text:
        return text.split("__", 1)[0]
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def _summarise_labels(fold_root: Path, strategy: str) -> dict[str, float | int | str]:
    labels = _read_frame(
        fold_root / "labels" / "strategy_bad_good_labels.parquet",
        columns=[
            "strategy",
            "timestamp",
            "strategy_performance",
            "ewma_performance",
            "bad_label",
            "composite_bad_pressure",
        ],
    )
    if labels.empty:
        return {}
    if "strategy" in labels.columns:
        rows = labels.loc[labels["strategy"].astype(str) == str(strategy)]
        if rows.empty:
            rows = labels
    else:
        rows = labels
    perf = pd.to_numeric(rows.get("strategy_performance"), errors="coerce")
    ewma = pd.to_numeric(rows.get("ewma_performance"), errors="coerce")
    bad = pd.to_numeric(rows.get("bad_label"), errors="coerce")
    pressure = pd.to_numeric(rows.get("composite_bad_pressure"), errors="coerce")
    return {
        "label_timestamp_count": int(len(rows)),
        "mean_strategy_performance": float(perf.mean()) if len(perf) else np.nan,
        "negative_performance_share": float(perf.lt(0.0).mean()) if len(perf) else np.nan,
        "ewma_performance_std": float(ewma.std(ddof=0)) if len(ewma) else np.nan,
        "bad_label_mean": float(bad.mean()) if len(bad) else np.nan,
        "bad_label_std": float(bad.std(ddof=0)) if len(bad) else np.nan,
        "bad_label_ge_075_share": float(bad.ge(0.75).mean()) if len(bad) else np.nan,
        "composite_bad_pressure_share": float(pressure.gt(0.0).mean()) if len(pressure) else np.nan,
        "composite_bad_pressure_mean": float(pressure.mean()) if len(pressure) else np.nan,
    }


def _summarise_first_stage(fold_root: Path, strategy: str) -> dict[str, float | int | str]:
    metrics = _read_frame(fold_root / "evaluation" / "first_stage_oof_metrics.parquet")
    if metrics.empty:
        return {}
    rows = metrics.copy()
    if "strategy" in rows.columns:
        rows = rows.loc[rows["strategy"].astype(str) == str(strategy)]
    if "direction" in rows.columns:
        rows = rows.loc[rows["direction"].astype(str) == "bad"]
    if rows.empty:
        return {}
    weights = pd.to_numeric(rows.get("n_valid"), errors="coerce") if "n_valid" in rows else None
    r2 = pd.to_numeric(rows.get("oof_weighted_r2"), errors="coerce")
    explained = r2.clip(lower=0.0, upper=1.0)
    top_features = Counter()
    for value in rows.get("top_features", pd.Series(dtype=object)).dropna():
        for feature in _parse_sequence_cell(value):
            top_features[feature] += 1
    return {
        "first_stage_model_rows": int(len(rows)),
        "first_stage_mean_oof_r2": _weighted_mean(r2, weights),
        "explained_instability_share": _weighted_mean(explained, weights),
        "unexplained_instability_share": float(1.0 - _weighted_mean(explained, weights))
        if np.isfinite(_weighted_mean(explained, weights))
        else np.nan,
        "first_stage_mean_oof_brier": _weighted_mean(
            pd.to_numeric(rows.get("oof_weighted_brier"), errors="coerce"),
            weights,
        ),
        "first_stage_median_prediction_std": float(
            pd.to_numeric(rows.get("prediction_std"), errors="coerce").median()
        ),
        "first_stage_over_regularised_share": float(
            rows.get("over_regularised_flag", pd.Series(False, index=rows.index)).astype(bool).mean()
        ),
        "first_stage_effective_leaves_used": _weighted_mean(
            pd.to_numeric(rows.get("effective_leaves_used"), errors="coerce"),
            weights,
        ),
        "first_stage_top_features": ", ".join(name for name, _ in top_features.most_common(12)),
    }


def _leaf_feature_rows(fold_root: Path, strategy: str, fold_id: int) -> list[dict[str, Any]]:
    leaves = _read_frame(fold_root / "leaves" / "pruned_leaves.parquet")
    if leaves.empty:
        return []
    rows = leaves.copy()
    if "strategy" in rows.columns:
        rows = rows.loc[rows["strategy"].astype(str) == str(strategy)]
    if "direction" in rows.columns:
        rows = rows.loc[rows["direction"].astype(str) == "bad"]
    if rows.empty or "split_path_features" not in rows.columns:
        return []
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in rows.iterrows():
        leaf_score = float(pd.to_numeric(pd.Series([row.get("contribution_share", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        leaf_score *= max(
            float(pd.to_numeric(pd.Series([row.get("positive_label_edge", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
            1e-12,
        )
        features = _parse_sequence_cell(row.get("split_path_features"))
        for feature in features:
            item = scores[feature]
            item["leaf_count"] += 1.0
            item["weighted_score"] += leaf_score
            item["active_timestamp_count"] += float(
                pd.to_numeric(pd.Series([row.get("active_timestamp_count", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
            )
            item["contribution_share"] += float(
                pd.to_numeric(pd.Series([row.get("contribution_share", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
            )
    return [
        {
            "strategy": strategy,
            "fold": int(fold_id),
            "feature": feature,
            "feature_family": _family(feature),
            **values,
        }
        for feature, values in scores.items()
    ]


def _interaction_rows(fold_root: Path, strategy: str, fold_id: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kind, rel in [
        ("pair", "interactions/leaf_guided_pairs.parquet"),
        ("triple", "interactions/leaf_guided_triples.parquet"),
    ]:
        frame = _read_frame(fold_root / rel)
        if frame.empty:
            continue
        rows = frame.copy()
        if "strategy" in rows.columns:
            rows = rows.loc[rows["strategy"].astype(str) == str(strategy)]
        if "direction" in rows.columns:
            rows = rows.loc[rows["direction"].astype(str) == "bad"]
        for _, row in rows.iterrows():
            features = [str(row.get("feature_i", "")), str(row.get("feature_j", ""))]
            if kind == "triple":
                features.append(str(row.get("feature_k", "")))
            out.append(
                {
                    "strategy": strategy,
                    "fold": int(fold_id),
                    "kind": kind,
                    "features": " × ".join(feature for feature in features if feature),
                    "candidate_score": float(
                        pd.to_numeric(pd.Series([row.get("candidate_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
                    ),
                    "fold_frequency": int(
                        pd.to_numeric(pd.Series([row.get("fold_frequency", 0)]), errors="coerce").fillna(0).iloc[0]
                    ),
                    "source_leaf_count": int(
                        pd.to_numeric(pd.Series([row.get("source_leaf_count", 0)]), errors="coerce").fillna(0).iloc[0]
                    ),
                }
            )
    return out


def _family_coverage_rows(fold_root: Path, strategy: str, fold_id: int) -> list[dict[str, Any]]:
    frame = _read_frame(fold_root / "features" / "feature_family_coverage.parquet")
    if frame.empty:
        return []
    rows = []
    for _, row in frame.iterrows():
        requested = float(row.get("requested_feature_count", 0.0) or 0.0)
        missing = float(row.get("missing_feature_count", 0.0) or 0.0)
        rows.append(
            {
                "strategy": strategy,
                "fold": int(fold_id),
                "family": str(row.get("family", "")),
                "requested_feature_count": int(requested),
                "available_feature_count": int(float(row.get("available_feature_count", 0.0) or 0.0)),
                "missing_feature_count": int(missing),
                "missing_share": float(missing / requested) if requested > 0 else 0.0,
            }
        )
    return rows


def build_report(run_root: Path, output_dir: Path, *, top_n: int = 30) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for strategy, scope_root in _iter_scope_roots(run_root):
        for fold_id, fold_root in _iter_fold_roots(scope_root):
            row: dict[str, Any] = {
                "strategy": strategy,
                "fold": int(fold_id),
                "fold_root": str(fold_root),
            }
            row.update(_summarise_labels(fold_root, strategy))
            row.update(_summarise_first_stage(fold_root, strategy))
            fold_rows.append(row)
            leaf_rows.extend(_leaf_feature_rows(fold_root, strategy, fold_id))
            interaction_rows.extend(_interaction_rows(fold_root, strategy, fold_id))
            coverage_rows.extend(_family_coverage_rows(fold_root, strategy, fold_id))

    fold_frame = pd.DataFrame(fold_rows)
    if not fold_frame.empty:
        weight = pd.to_numeric(fold_frame.get("label_timestamp_count"), errors="coerce").fillna(1.0)
        strategy_rows = []
        for strategy, group in fold_frame.groupby("strategy", sort=True):
            group_weight = weight.reindex(group.index).fillna(1.0)
            explained = _weighted_mean(group["explained_instability_share"], group_weight)
            unexplained = float(1.0 - explained) if np.isfinite(explained) else np.nan
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "fold_count": int(len(group)),
                    "timestamp_count": int(pd.to_numeric(group.get("label_timestamp_count"), errors="coerce").sum()),
                    "mean_strategy_performance": _weighted_mean(group.get("mean_strategy_performance"), group_weight),
                    "negative_performance_share": _weighted_mean(group.get("negative_performance_share"), group_weight),
                    "bad_label_mean": _weighted_mean(group.get("bad_label_mean"), group_weight),
                    "bad_label_std": _weighted_mean(group.get("bad_label_std"), group_weight),
                    "bad_label_ge_075_share": _weighted_mean(group.get("bad_label_ge_075_share"), group_weight),
                    "composite_bad_pressure_share": _weighted_mean(group.get("composite_bad_pressure_share"), group_weight),
                    "first_stage_mean_oof_r2": _weighted_mean(group.get("first_stage_mean_oof_r2"), group_weight),
                    "explained_instability_share": explained,
                    "unexplained_instability_share": unexplained,
                    "first_stage_mean_oof_brier": _weighted_mean(group.get("first_stage_mean_oof_brier"), group_weight),
                    "first_stage_median_prediction_std": float(
                        pd.to_numeric(group.get("first_stage_median_prediction_std"), errors="coerce").median()
                    ),
                    "first_stage_over_regularised_share": _weighted_mean(group.get("first_stage_over_regularised_share"), group_weight),
                    "first_stage_effective_leaves_used": _weighted_mean(group.get("first_stage_effective_leaves_used"), group_weight),
                }
            )
        strategy_frame = pd.DataFrame(strategy_rows).sort_values("strategy", kind="mergesort")
    else:
        strategy_frame = pd.DataFrame()

    leaf_frame = pd.DataFrame(leaf_rows)
    if not leaf_frame.empty:
        leaf_summary = (
            leaf_frame.groupby(["strategy", "feature", "feature_family"], as_index=False)
            .agg(
                leaf_count=("leaf_count", "sum"),
                weighted_score=("weighted_score", "sum"),
                active_timestamp_count=("active_timestamp_count", "sum"),
                contribution_share=("contribution_share", "sum"),
                fold_count=("fold", "nunique"),
            )
            .sort_values(["strategy", "weighted_score"], ascending=[True, False], kind="mergesort")
            .groupby("strategy", group_keys=False)
            .head(int(top_n))
            .reset_index(drop=True)
        )
    else:
        leaf_summary = pd.DataFrame()

    interaction_frame = pd.DataFrame(interaction_rows)
    if not interaction_frame.empty:
        interaction_summary = (
            interaction_frame.groupby(["strategy", "kind", "features"], as_index=False)
            .agg(
                candidate_score=("candidate_score", "sum"),
                fold_count=("fold", "nunique"),
                source_leaf_count=("source_leaf_count", "sum"),
            )
            .sort_values(["strategy", "candidate_score"], ascending=[True, False], kind="mergesort")
            .groupby("strategy", group_keys=False)
            .head(int(top_n))
            .reset_index(drop=True)
        )
    else:
        interaction_summary = pd.DataFrame()

    coverage_frame = pd.DataFrame(coverage_rows)
    if not coverage_frame.empty:
        coverage_summary = (
            coverage_frame.groupby(["strategy", "family"], as_index=False)
            .agg(
                requested_feature_count=("requested_feature_count", "mean"),
                available_feature_count=("available_feature_count", "mean"),
                missing_feature_count=("missing_feature_count", "mean"),
                missing_share=("missing_share", "mean"),
                fold_count=("fold", "nunique"),
            )
            .sort_values(["strategy", "missing_share"], ascending=[True, False], kind="mergesort")
        )
    else:
        coverage_summary = pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fold": output_dir / "instability_explanation_by_fold.csv",
        "strategy": output_dir / "instability_explanation_by_strategy.csv",
        "leaf_features": output_dir / "instability_top_leaf_features.csv",
        "interactions": output_dir / "instability_top_interactions.csv",
        "coverage": output_dir / "instability_feature_family_coverage.csv",
    }
    fold_frame.to_csv(outputs["fold"], index=False)
    strategy_frame.to_csv(outputs["strategy"], index=False)
    leaf_summary.to_csv(outputs["leaf_features"], index=False)
    interaction_summary.to_csv(outputs["interactions"], index=False)
    coverage_summary.to_csv(outputs["coverage"], index=False)

    report_path = output_dir / "instability_explanation_report.md"
    with report_path.open("w") as fh:
        fh.write("# Performance-Regime Instability Explanation\n\n")
        fh.write(f"Run root: `{run_root}`\n\n")
        fh.write("## Strategy Summary\n\n")
        if strategy_frame.empty:
            fh.write("No strategy rows found.\n\n")
        else:
            cols = [
                "strategy",
                "fold_count",
                "timestamp_count",
                "explained_instability_share",
                "unexplained_instability_share",
                "bad_label_ge_075_share",
                "composite_bad_pressure_share",
                "first_stage_mean_oof_brier",
                "first_stage_median_prediction_std",
            ]
            fh.write(strategy_frame.loc[:, [c for c in cols if c in strategy_frame.columns]].to_markdown(index=False))
            fh.write("\n\n")
        fh.write("## Top Bad-Regime Leaf Features\n\n")
        if leaf_summary.empty:
            fh.write("No retained bad-regime leaf features found.\n\n")
        else:
            cols = ["strategy", "feature", "feature_family", "weighted_score", "leaf_count", "fold_count"]
            fh.write(leaf_summary.loc[:, [c for c in cols if c in leaf_summary.columns]].head(40).to_markdown(index=False))
            fh.write("\n\n")
        fh.write("## Top Leaf-Guided Interactions\n\n")
        if interaction_summary.empty:
            fh.write("No leaf-guided interactions found.\n\n")
        else:
            cols = ["strategy", "kind", "features", "candidate_score", "fold_count", "source_leaf_count"]
            fh.write(interaction_summary.loc[:, [c for c in cols if c in interaction_summary.columns]].head(40).to_markdown(index=False))
            fh.write("\n\n")
        fh.write("## Interpretation\n\n")
        fh.write(
            "Explained instability share is the clipped OOF weighted R2 of the fold-local "
            "bad-performance LightGBM soft-label model. Unexplained share is the residual. "
            "This measures learnability of poor-performance states from timestamp-level "
            "market/regime features, not deployable PnL improvement.\n"
        )

    summary = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "strategy_count": int(strategy_frame.shape[0]),
        "fold_rows": int(fold_frame.shape[0]),
        "leaf_feature_rows": int(leaf_summary.shape[0]),
        "interaction_rows": int(interaction_summary.shape[0]),
        "outputs": {key: str(value) for key, value in outputs.items()} | {"report": str(report_path)},
    }
    with (output_dir / "instability_explanation_summary.json").open("w") as fh:
        json.dump(_json_safe(summary), fh, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_report(args.run_root, args.output_dir, top_n=int(args.top_n))
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
