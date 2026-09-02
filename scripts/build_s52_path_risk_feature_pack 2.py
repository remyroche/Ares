#!/usr/bin/env python3
"""Build S52 feature packs that include path-risk learnability features.

The opportunity-oriented feature list improved top-k precision, but the selected
rows still have high full-path bad-MAE. This builder reuses the all-store
univariate diagnostics and produces a compact feature pack that explicitly
mixes:

- opportunity features by side,
- low full-path bad-MAE features by side,
- clean path-order features by side,
- the previous learnability feature list.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOURCE_DIR = Path("data_perp/reports/s52_feature_learnability_allstore_prefilter_noae_20260705_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_path_risk_feature_pack_20260705_v1")


def _safe_num(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _norm_rank(series: pd.Series, *, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    return numeric.rank(method="average", pct=True, ascending=ascending).fillna(0.0)


def _read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    col = "feature" if "feature" in frame.columns else frame.columns[0]
    return [str(v) for v in frame[col].dropna().tolist()]


def _score_rows(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    net = _safe_num(frame["mean_top10_mean_first_touch_net"], 0.0)
    evw = _safe_num(frame["mean_top10_ev_weighted_first_touch_precision"], 0.0)
    first_good = _safe_num(frame["mean_top10_first_pass_good_rate"], 0.0)
    full_bad = _safe_num(frame["mean_top10_first_touch_full_path_bad_mae_1r_rate"], 1.0)
    mae_before = _safe_num(frame["mean_top10_mae_1r_before_mfe_1r_rate"], 1.0)
    mfe_before = _safe_num(frame["mean_top10_mfe_1r_before_mae_1r_rate"], 0.0)
    timeout = _safe_num(frame["mean_top10_timeout_rate"], 1.0)

    frame["opportunity_score"] = (
        2.0 * _norm_rank(net, ascending=True)
        + 1.0 * _norm_rank(evw, ascending=True)
        + 0.8 * _norm_rank(first_good, ascending=True)
        + 0.5 * _norm_rank(mfe_before, ascending=True)
        - 0.6 * _norm_rank(full_bad, ascending=False)
    )
    frame["path_clean_score"] = (
        2.5 * _norm_rank(full_bad, ascending=False)
        + 1.2 * _norm_rank(mae_before, ascending=False)
        + 0.9 * _norm_rank(mfe_before, ascending=True)
        + 0.6 * _norm_rank(timeout, ascending=False)
        + 0.4 * _norm_rank(evw, ascending=True)
    )
    frame["balanced_score"] = (
        1.5 * frame["opportunity_score"]
        + 1.2 * frame["path_clean_score"]
        + 25.0 * net.clip(lower=-0.005, upper=0.005)
    )
    frame["risk_reason"] = np.select(
        [
            (full_bad.le(0.50) & net.gt(0.0)),
            full_bad.le(0.50),
            net.gt(0.0),
        ],
        ["positive_net_low_bad_mae", "low_bad_mae", "positive_net"],
        default="ranked_tradeoff",
    )
    return frame


def _select_pack(
    scored: pd.DataFrame,
    *,
    previous_features: list[str],
    top_per_segment_objective: int,
    top_per_segment_path: int,
    top_per_segment_balanced: int,
    max_total_features: int,
) -> pd.DataFrame:
    selected_rows: list[pd.DataFrame] = []
    for segment in ("long", "short", "all"):
        part = scored[scored["segment"].eq(segment)].copy()
        if part.empty:
            continue
        selected_rows.append(
            part.sort_values("opportunity_score", ascending=False)
            .head(int(top_per_segment_objective))
            .assign(selection_reason=f"{segment}_opportunity")
        )
        selected_rows.append(
            part.sort_values("path_clean_score", ascending=False)
            .head(int(top_per_segment_path))
            .assign(selection_reason=f"{segment}_path_clean")
        )
        selected_rows.append(
            part.sort_values("balanced_score", ascending=False)
            .head(int(top_per_segment_balanced))
            .assign(selection_reason=f"{segment}_balanced")
        )
    if selected_rows:
        selected = pd.concat(selected_rows, ignore_index=True)
    else:
        selected = scored.iloc[:0].copy()
        selected["selection_reason"] = []

    previous = pd.DataFrame({"feature": previous_features})
    previous["selection_reason"] = "previous_learnability"
    previous["segment"] = "previous"
    selected = pd.concat([selected, previous], ignore_index=True, sort=False)

    selected["feature"] = selected["feature"].astype(str)
    rank_map: dict[str, int] = {}
    ordered_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        feature = str(row["feature"])
        if feature in rank_map:
            continue
        rank_map[feature] = len(rank_map) + 1
        ordered_rows.append(row.to_dict())
        if len(rank_map) >= int(max_total_features):
            break
    return pd.DataFrame(ordered_rows)


def _write_report(output_dir: Path, selected: pd.DataFrame, scored: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(frame: pd.DataFrame, cols: list[str], n: int = 20) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.6f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "segment",
        "feature",
        "selection_reason",
        "risk_reason",
        "mean_top10_mean_first_touch_net",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "opportunity_score",
        "path_clean_score",
        "balanced_score",
    ]
    lines = [
        "# S52 Path-Risk Feature Pack",
        "",
        f"Source summary: `{manifest['source_summary']}`",
        f"Features selected: `{manifest['selected_feature_count']}`",
        "",
        "## Selected Features",
        "",
        fmt(selected, cols, n=80),
        "",
        "## Best Short Path-Clean Features",
        "",
        fmt(
            scored[scored["segment"].eq("short")].sort_values("path_clean_score", ascending=False),
            cols,
            n=25,
        ),
        "",
        "## Best Long Path-Clean Features",
        "",
        fmt(
            scored[scored["segment"].eq("long")].sort_values("path_clean_score", ascending=False),
            cols,
            n=25,
        ),
        "",
    ]
    output_dir.joinpath("s52_path_risk_feature_pack.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    source_dir: Path,
    output_dir: Path,
    top_per_segment_objective: int,
    top_per_segment_path: int,
    top_per_segment_balanced: int,
    max_total_features: int,
) -> dict[str, str]:
    summary_path = source_dir / "s52_feature_learnability_feature_summary.csv"
    previous_path = source_dir / "s52_learnability_ranker_feature_list_top360.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    required = {
        "segment",
        "feature",
        "mean_top10_mean_first_touch_net",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_timeout_rate",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary missing required columns: {missing}")

    scored = _score_rows(summary)
    previous_features = _read_feature_list(previous_path)
    selected = _select_pack(
        scored,
        previous_features=previous_features,
        top_per_segment_objective=int(top_per_segment_objective),
        top_per_segment_path=int(top_per_segment_path),
        top_per_segment_balanced=int(top_per_segment_balanced),
        max_total_features=int(max_total_features),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_list_path = output_dir / "s52_path_risk_ranker_feature_list.csv"
    selected_path = output_dir / "s52_path_risk_feature_selection.csv"
    scored_path = output_dir / "s52_path_risk_feature_scores.csv"
    manifest_path = output_dir / "manifest.json"
    selected[["feature"]].drop_duplicates().to_csv(feature_list_path, index=False)
    selected.to_csv(selected_path, index=False)
    scored.sort_values("balanced_score", ascending=False).to_csv(scored_path, index=False)
    manifest = {
        "source_dir": str(source_dir),
        "source_summary": str(summary_path),
        "previous_feature_list": str(previous_path),
        "output_dir": str(output_dir),
        "selected_feature_count": int(selected["feature"].nunique()),
        "top_per_segment_objective": int(top_per_segment_objective),
        "top_per_segment_path": int(top_per_segment_path),
        "top_per_segment_balanced": int(top_per_segment_balanced),
        "max_total_features": int(max_total_features),
        "outputs": {
            "feature_list": str(feature_list_path),
            "selected": str(selected_path),
            "scores": str(scored_path),
            "report": str(output_dir / "s52_path_risk_feature_pack.md"),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir, selected, scored, manifest)
    print(f"wrote {feature_list_path}")
    print(selected[["segment", "feature", "selection_reason", "risk_reason"]].head(30).to_string(index=False))
    return {k: str(v) for k, v in manifest["outputs"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-per-segment-objective", type=int, default=60)
    parser.add_argument("--top-per-segment-path", type=int, default=80)
    parser.add_argument("--top-per-segment-balanced", type=int, default=80)
    parser.add_argument("--max-total-features", type=int, default=420)
    args = parser.parse_args()
    run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        top_per_segment_objective=int(args.top_per_segment_objective),
        top_per_segment_path=int(args.top_per_segment_path),
        top_per_segment_balanced=int(args.top_per_segment_balanced),
        max_total_features=int(args.max_total_features),
    )


if __name__ == "__main__":
    main()
