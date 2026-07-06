#!/usr/bin/env python3
"""Diagnose why v17 source-quality labels are hard to learn OOS.

This replays the same fixed month-forward ExtraTrees smoke model used by
``run_source_quality_label_walkforward_ablation.py`` and evaluates three
rankings for each label candidate:

1. ``target_oracle``: rank directly by the label target. This is diagnostic
   and target-side only; it is not a deployable model.
2. ``model_score``: rank by the fitted month-forward feature model.
3. ``inverted_model_score``: rank by the negative model score, to detect
   systematic sign inversions.

The output distinguishes bad labels from labels that are economically aligned
but not learnable from the current feature/model setup.
"""

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

from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _make_targets,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (
    DEFAULT_ABLATIONS,
    DEFAULT_MANIFEST,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_TOP_FRACS,
    DEFAULT_MONTHS,
    DEFAULT_SEEDS,
    VANILLA_LABEL_ARM,
    VANILLA_NAME,
    _load_joined_frame,
    _load_manifest_specs,
    _parse_csv,
    _parse_float_csv,
    _parse_int_csv,
    _source_feature_columns,
    _target_for_spec,
    _training_mask_for_spec,
    _weight_series,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/learnability_gap")


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _rank_score_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    ablation: str,
    selector: str,
    period: str,
    top_fracs: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for top_frac in top_fracs:
        row = _selection_metrics(
            frame=frame,
            metrics=metrics,
            target=target,
            score=score,
            arm=ablation,
            selector=selector,
            period=period,
            top_frac=top_frac,
        )
        row.update(
            {
                "ablation": ablation,
                "ranker": selector,
                "score_ic_u": _spearman(score, metrics["u_policy_net"]),
                "score_ic_label": _spearman(score, target["target_soft"]),
                "target_ic_u": _spearman(target["target_soft"], metrics["u_policy_net"]),
            }
        )
        rows.append(row)
    return rows


def _classify(month_group: pd.DataFrame) -> str:
    target_ic = _safe_mean(month_group.loc[month_group["ranker"].eq("target_oracle"), "score_ic_u"])
    model_ic = _safe_mean(month_group.loc[month_group["ranker"].eq("model_score"), "score_ic_u"])
    inverted_ic = _safe_mean(month_group.loc[month_group["ranker"].eq("inverted_model_score"), "score_ic_u"])
    oracle_mean_u = _safe_mean(month_group.loc[month_group["ranker"].eq("target_oracle"), "mean_u"])
    model_mean_u = _safe_mean(month_group.loc[month_group["ranker"].eq("model_score"), "mean_u"])
    inverted_mean_u = _safe_mean(month_group.loc[month_group["ranker"].eq("inverted_model_score"), "mean_u"])
    if math.isfinite(target_ic) and target_ic <= 0.0 and math.isfinite(oracle_mean_u) and oracle_mean_u <= 0.0:
        return "label_not_economically_aligned"
    if math.isfinite(target_ic) and target_ic > 0.0 and math.isfinite(model_ic) and model_ic < 0.0:
        if math.isfinite(inverted_ic) and inverted_ic > 0.0 and inverted_mean_u > model_mean_u:
            return "model_sign_inversion_or_feature_anti_proxy"
        return "feature_model_learnability_failure"
    if math.isfinite(target_ic) and target_ic > 0.0 and math.isfinite(model_ic) and model_ic >= 0.0:
        if math.isfinite(model_mean_u) and math.isfinite(oracle_mean_u) and model_mean_u < 0.5 * oracle_mean_u:
            return "weak_capture_of_good_label"
        return "learnable_candidate"
    return "insufficient_signal"


def _summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(["ablation", "period", "top_frac"], dropna=False, observed=True):
        ablation, period, top_frac = key
        row: dict[str, Any] = {
            "ablation": ablation,
            "period": period,
            "top_frac": float(top_frac),
            "diagnosis": _classify(group),
        }
        for ranker in ["target_oracle", "model_score", "inverted_model_score"]:
            sub = group[group["ranker"].eq(ranker)]
            prefix = ranker
            row[f"{prefix}_mean_u"] = _safe_mean(sub["mean_u"])
            row[f"{prefix}_hit_u"] = _safe_mean(sub["hit_u"])
            row[f"{prefix}_bad_mae_1r_rate"] = _safe_mean(sub["bad_mae_1r_rate"])
            row[f"{prefix}_timeout_rate"] = _safe_mean(sub["timeout_rate"])
            row[f"{prefix}_wide_barrier_25bps_rate"] = _safe_mean(sub["wide_barrier_25bps_rate"])
            row[f"{prefix}_score_ic_u"] = _safe_mean(sub["score_ic_u"])
            row[f"{prefix}_score_ic_label"] = _safe_mean(sub["score_ic_label"])
        row["model_vs_oracle_mean_u_gap"] = row["model_score_mean_u"] - row["target_oracle_mean_u"]
        row["inverted_vs_model_mean_u_delta"] = row["inverted_model_score_mean_u"] - row["model_score_mean_u"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["top_frac", "period", "ablation"],
        ascending=[True, True, True],
        na_position="last",
    )


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in summary.groupby(["ablation", "top_frac"], dropna=False, observed=True):
        ablation, top_frac = key
        rows.append(
            {
                "ablation": ablation,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "target_oracle_positive_months": int((_safe_numeric(group["target_oracle_mean_u"]) > 0.0).sum()),
                "model_positive_months": int((_safe_numeric(group["model_score_mean_u"]) > 0.0).sum()),
                "inverted_positive_months": int((_safe_numeric(group["inverted_model_score_mean_u"]) > 0.0).sum()),
                "target_ic_positive_months": int((_safe_numeric(group["target_oracle_score_ic_u"]) > 0.0).sum()),
                "model_ic_positive_months": int((_safe_numeric(group["model_score_score_ic_u"]) > 0.0).sum()),
                "inverted_ic_positive_months": int((_safe_numeric(group["inverted_model_score_score_ic_u"]) > 0.0).sum()),
                "mean_target_oracle_u": _safe_mean(group["target_oracle_mean_u"]),
                "mean_model_u": _safe_mean(group["model_score_mean_u"]),
                "mean_inverted_u": _safe_mean(group["inverted_model_score_mean_u"]),
                "mean_model_vs_oracle_gap": _safe_mean(group["model_vs_oracle_mean_u_gap"]),
                "mean_inverted_vs_model_delta": _safe_mean(group["inverted_vs_model_mean_u_delta"]),
                "mean_target_ic_u": _safe_mean(group["target_oracle_score_ic_u"]),
                "mean_model_ic_u": _safe_mean(group["model_score_score_ic_u"]),
                "mean_inverted_ic_u": _safe_mean(group["inverted_model_score_score_ic_u"]),
                "dominant_diagnosis": str(group["diagnosis"].mode().iloc[0]) if len(group["diagnosis"].mode()) else "",
                "recommendation": _recommend(group),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_frac", "mean_model_u"],
        ascending=[True, False],
        na_position="last",
    )


def _recommend(group: pd.DataFrame) -> str:
    months = int(group["period"].nunique())
    target_positive = int((_safe_numeric(group["target_oracle_mean_u"]) > 0.0).sum())
    model_positive = int((_safe_numeric(group["model_score_mean_u"]) > 0.0).sum())
    inverted_better = int((_safe_numeric(group["inverted_vs_model_mean_u_delta"]) > 0.0).sum())
    model_ic_positive = int((_safe_numeric(group["model_score_score_ic_u"]) > 0.0).sum())
    mean_gap = _safe_mean(group["model_vs_oracle_mean_u_gap"])
    if months < 3:
        return "incomplete_evidence"
    if target_positive < 2:
        return "rework_label_objective"
    if model_positive == 0 and inverted_better >= 2:
        return "check_sign_or_feature_anti_proxy"
    if model_ic_positive == 0 and mean_gap < 0.0:
        return "feature_model_learnability_failure"
    if model_positive >= 2 and mean_gap > -0.005:
        return "candidate_for_larger_ablation"
    return "diagnostic_only"


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_report(output_dir: Path, summary: pd.DataFrame, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_quality_label_learnability_gap.md"
    top10 = summary[summary["top_frac"].eq(0.10)].copy()
    lines = [
        "# Source Quality Label Learnability Gap",
        "",
        "Scope: diagnostic target-oracle versus month-forward model comparison. `target_oracle` uses realized labels and is not deployable.",
        "",
        "## Aggregate",
        "",
        _table(
            aggregate,
            [
                "recommendation",
                "ablation",
                "top_frac",
                "months",
                "target_oracle_positive_months",
                "model_positive_months",
                "inverted_positive_months",
                "mean_target_oracle_u",
                "mean_model_u",
                "mean_inverted_u",
                "mean_model_vs_oracle_gap",
                "mean_target_ic_u",
                "mean_model_ic_u",
                "mean_inverted_ic_u",
                "dominant_diagnosis",
            ],
            limit=120,
        ),
        "",
        "## Top 10% Monthly Detail",
        "",
        _table(
            top10,
            [
                "diagnosis",
                "period",
                "ablation",
                "target_oracle_mean_u",
                "model_score_mean_u",
                "inverted_model_score_mean_u",
                "model_vs_oracle_mean_u_gap",
                "inverted_vs_model_mean_u_delta",
                "target_oracle_score_ic_u",
                "model_score_score_ic_u",
                "inverted_model_score_score_ic_u",
                "model_score_score_ic_label",
                "model_score_bad_mae_1r_rate",
                "model_score_timeout_rate",
                "model_score_wide_barrier_25bps_rate",
            ],
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Monthly ranker metrics: `{manifest['outputs']['monthly_ranker_metrics']}`",
        f"- Learnability summary: `{manifest['outputs']['learnability_summary']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    manifest_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    ablations: list[str],
    months: list[str],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _load_manifest_specs(manifest_path, ablations)
    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    vanilla_targets = _make_targets(frame, metrics)
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    for month in months:
        valid_mask = month_period.eq(month)
        valid_frame = frame.loc[valid_mask].reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].reset_index(drop=True)
        if int(valid_mask.sum()) < int(min_valid_rows):
            continue
        for spec in specs:
            train_mask = _training_mask_for_spec(
                spec=spec,
                frame=frame,
                month=month,
                train_lookback_months=train_lookback_months,
            )
            if int(train_mask.sum()) < int(min_train_rows):
                diag_rows.append(
                    {
                        "ablation": spec.name,
                        "period": month,
                        "skipped": True,
                        "reason": "too_few_train_rows",
                        "train_rows": int(train_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                    }
                )
                continue
            features = list(base_features)
            if spec.add_source_features:
                features = list(dict.fromkeys(features + source_features))
            target = _target_for_spec(spec=spec, frame=frame, vanilla_targets=vanilla_targets)
            valid_target = target.loc[valid_mask].reset_index(drop=True)
            x_train, x_valid = _month_model_frame(
                frame,
                train_mask=train_mask,
                valid_mask=valid_mask,
                features=features,
            )
            target_train = target.loc[train_mask]
            if spec.is_vanilla:
                weights = _safe_numeric(frame.loc[train_mask, "__w__"] if "__w__" in frame.columns else 1.0).fillna(1.0)
            else:
                weights = _weight_series(frame, spec.sample_weight_column).loc[train_mask]
            pred_matrix = np.vstack(
                [
                    _fit_predict(
                        x_train=x_train,
                        y_train=target_train["target_soft"],
                        w_train=weights,
                        x_valid=x_valid,
                        seed=seed,
                    )
                    for seed in seeds
                ]
            )
            model_score = pd.Series(np.mean(pred_matrix, axis=0).astype(np.float32), index=valid_frame.index)
            target_score = valid_target["target_soft"].reset_index(drop=True)
            rankers = {
                "target_oracle": target_score,
                "model_score": model_score,
                "inverted_model_score": -model_score,
            }
            for ranker, score in rankers.items():
                rows.extend(
                    _rank_score_metrics(
                        frame=valid_frame,
                        metrics=valid_metrics,
                        target=valid_target,
                        score=score,
                        ablation=spec.name,
                        selector=ranker,
                        period=month,
                        top_fracs=top_fracs,
                    )
                )
            diag_rows.append(
                {
                    "ablation": spec.name,
                    "period": month,
                    "skipped": False,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "model_feature_count": int(len(features)),
                    "target_train_mean": _safe_mean(target_train["target_soft"]),
                    "target_valid_mean": _safe_mean(valid_target["target_soft"]),
                    "target_valid_ic_u": _spearman(valid_target["target_soft"], valid_metrics["u_policy_net"]),
                    "model_valid_ic_u": _spearman(model_score, valid_metrics["u_policy_net"]),
                    "model_valid_ic_label": _spearman(model_score, valid_target["target_soft"]),
                }
            )

    monthly = pd.DataFrame(rows)
    summary = _summarize(monthly)
    aggregate = _aggregate(summary)
    diagnostics = pd.DataFrame(diag_rows)
    paths = {
        "monthly_ranker_metrics": output_dir / "source_quality_label_learnability_ranker_metrics.csv",
        "learnability_summary": output_dir / "source_quality_label_learnability_summary.csv",
        "aggregate": output_dir / "source_quality_label_learnability_aggregate.csv",
        "diagnostics": output_dir / "source_quality_label_learnability_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly_ranker_metrics"], index=False)
    summary.to_csv(paths["learnability_summary"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "source_quality_label_learnability_gap_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "label_ablation_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "months": list(months),
        "ablations": [spec.name for spec in specs],
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(seed) for seed in seeds],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, summary, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--label-ablation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--ablations", type=str, default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        manifest_path=args.label_ablation_manifest,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        ablations=_parse_csv(args.ablations, DEFAULT_ABLATIONS),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
