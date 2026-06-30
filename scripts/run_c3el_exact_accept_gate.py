#!/usr/bin/env python3
"""Evaluate an exact-label accept/reject gate for C3el fallback cuts.

This is a no-replay diagnostic. It joins exact cloned-state labels for candidate
size-cut actions to deployable action/state features and optional C3el score
features, then evaluates a conservative leave-one-day gate. The gate defaults
to no-op when the training slice cannot find a positive, sufficiently precise
selection rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


KEYS = ["timestamp", "strategy_id", "action_value"]
ALLOWED_SCORE_FEATURES = {"p_intervene", "pred_action_delta_J"}
LEAK_TERMS = (
    "delta_",
    "base_",
    "action_",
    "candidate_",
    "baseline_",
    "direct_",
    "net_pnl",
    "gross_pnl",
    "cost_pnl",
    "full_j",
    "immediate_j",
    "turnover",
    "trade_count",
    "full_sl",
    "timeout",
    "is_baseline_action",
    "exact_positive",
    "label",
    "target",
    "future",
    "pnl",
)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_labels(paths: list[Path], *, default_action_value: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_frame(path).copy()
        frame["source_path"] = str(path)
        parts.append(frame)
    if not parts:
        raise ValueError("At least one label file is required")
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "action_value" in out.columns:
        action_value = pd.to_numeric(out["action_value"], errors="coerce")
    elif "multiplier" in out.columns:
        action_value = pd.to_numeric(out["multiplier"], errors="coerce")
    else:
        action_value = pd.Series(np.nan, index=out.index)
    out["action_value"] = action_value.fillna(float(default_action_value)).round(6)
    out["delta_full_J"] = pd.to_numeric(out["delta_full_J"], errors="coerce").fillna(0.0)
    if "delta_immediate_J" in out.columns:
        out["delta_immediate_J"] = pd.to_numeric(out["delta_immediate_J"], errors="coerce").fillna(0.0)
    else:
        out["delta_immediate_J"] = 0.0
    out = out.drop_duplicates(KEYS, keep="last")
    out["exact_positive_e50"] = out["delta_full_J"].gt(50.0)
    out["exact_positive_full"] = out["delta_full_J"].gt(0.0)
    out["day"] = out["timestamp"].dt.floor("D")
    return out.reset_index(drop=True)


def _normalise_action_features(path: Path, *, default_action_value: float) -> pd.DataFrame:
    out = _read_frame(path).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "multiplier" not in out.columns:
        out["multiplier"] = default_action_value
    out["action_value"] = pd.to_numeric(out["multiplier"], errors="coerce").fillna(float(default_action_value)).round(6)
    return out


def _normalise_scores(paths: list[Path], *, default_action_value: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_frame(path).copy()
        frame["score_source_path"] = str(path)
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=[*KEYS, *ALLOWED_SCORE_FEATURES])
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "action_value" in out.columns:
        action_value = pd.to_numeric(out["action_value"], errors="coerce")
    elif "multiplier" in out.columns:
        action_value = pd.to_numeric(out["multiplier"], errors="coerce")
    else:
        action_value = pd.Series(np.nan, index=out.index)
    out["action_value"] = action_value.fillna(float(default_action_value)).round(6)
    keep = [*KEYS, *[col for col in ALLOWED_SCORE_FEATURES if col in out.columns]]
    out = out[keep].drop_duplicates(KEYS, keep="last")
    return out


def _is_deployable_feature(col: str) -> bool:
    if col in ALLOWED_SCORE_FEATURES:
        return True
    lower = col.lower()
    if lower.endswith("_label"):
        return False
    return not any(term in lower for term in LEAK_TERMS)


def _feature_columns(frame: pd.DataFrame, *, max_features: int) -> list[str]:
    excluded = {
        "timestamp",
        "strategy_id",
        "action_value",
        "day",
        "source_path",
        "score_source_path",
        "delta_full_J",
        "delta_immediate_J",
        "exact_positive_e50",
        "exact_positive_full",
        "feature_row_matched",
    }
    candidates: list[str] = []
    for col in frame.columns:
        if col in excluded or not _is_deployable_feature(str(col)):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() >= 8 and vals.nunique(dropna=True) > 1:
            candidates.append(str(col))
    y = frame["exact_positive_e50"].astype(int)
    scored: list[tuple[float, str]] = []
    for col in candidates:
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr = vals.corr(y, method="spearman")
        scored.append((0.0 if pd.isna(corr) else abs(float(corr)), col))
    return [col for _, col in sorted(scored, reverse=True)[: int(max_features)]]


def _prepare_matrix(frame: pd.DataFrame, features: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame(index=frame.index)
    for col in features:
        x[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else np.nan
    x = x.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.fillna(medians).fillna(0.0).astype(np.float32), medians.astype(float)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = train["exact_positive_e50"].astype(int).to_numpy()
    if len(features) == 0 or len(np.unique(y)) < 2:
        base_rate = float(np.mean(y)) if len(y) else 0.0
        return np.full(len(train), base_rate), np.full(len(test), base_rate)
    x_train, medians = _prepare_matrix(train, features)
    x_test, _ = _prepare_matrix(test, features, medians)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=3,
        min_samples_leaf=max(2, int(np.ceil(0.08 * len(train)))),
        random_state=int(seed),
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    model.fit(x_train, y)
    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_test)[:, 1]


def _summarise(frame: pd.DataFrame, selected: pd.Series, *, name: str, threshold: float | None) -> dict[str, Any]:
    selected = selected.fillna(False).astype(bool)
    kept = frame.loc[selected]
    rejected = frame.loc[~selected]
    return {
        "selection": name,
        "threshold": None if threshold is None else float(threshold),
        "keep_count": int(selected.sum()),
        "reject_count": int((~selected).sum()),
        "positive_e50_rate": float(kept["exact_positive_e50"].mean()) if len(kept) else 0.0,
        "positive_full_rate": float(kept["exact_positive_full"].mean()) if len(kept) else 0.0,
        "delta_full_J_sum": float(pd.to_numeric(kept["delta_full_J"], errors="coerce").fillna(0.0).sum()) if len(kept) else 0.0,
        "delta_full_J_worst": float(pd.to_numeric(kept["delta_full_J"], errors="coerce").fillna(0.0).min()) if len(kept) else 0.0,
        "delta_immediate_J_sum": float(pd.to_numeric(kept["delta_immediate_J"], errors="coerce").fillna(0.0).sum()) if len(kept) else 0.0,
        "rejected_negative_delta_full_J_sum": float(
            pd.to_numeric(rejected["delta_full_J"], errors="coerce").fillna(0.0).clip(upper=0.0).sum()
        )
        if len(rejected)
        else 0.0,
        "rejected_positive_delta_full_J_sum": float(
            pd.to_numeric(rejected["delta_full_J"], errors="coerce").fillna(0.0).clip(lower=0.0).sum()
        )
        if len(rejected)
        else 0.0,
    }


def _choose_threshold(
    train: pd.DataFrame,
    pred: np.ndarray,
    *,
    thresholds: list[float],
    min_keep: int,
    min_precision: float,
) -> tuple[float | None, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pred_series = pd.Series(pred, index=train.index)
    for threshold in thresholds:
        selected = pred_series.ge(float(threshold))
        row = _summarise(train, selected, name="train_threshold", threshold=threshold)
        row["objective"] = (
            float(row["delta_full_J_sum"])
            + 250.0 * float(row["positive_e50_rate"])
            + min(float(row["delta_full_J_worst"]), 0.0)
        )
        row["valid"] = (
            int(row["keep_count"]) >= int(min_keep)
            and float(row["positive_e50_rate"]) >= float(min_precision)
            and float(row["delta_full_J_sum"]) > 0.0
        )
        rows.append(row)
    trials = pd.DataFrame(rows).sort_values(["valid", "objective", "delta_full_J_sum"], ascending=[False, False, False])
    valid = trials.loc[trials["valid"].astype(bool)]
    if valid.empty:
        return None, trials
    return float(valid.iloc[0]["threshold"]), trials


def _join_frames(labels: pd.DataFrame, features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    feature_payload = features.copy()
    feature_payload["_feature_row_marker"] = True
    feature_cols = [col for col in feature_payload.columns if col not in labels.columns or col in KEYS or col == "_feature_row_marker"]
    joined = labels.merge(feature_payload[feature_cols], on=KEYS, how="left")
    joined["feature_row_matched"] = joined["_feature_row_marker"].eq(True)
    joined = joined.drop(columns=["_feature_row_marker"], errors="ignore")
    if not scores.empty:
        joined = joined.merge(scores, on=KEYS, how="left", suffixes=("", "_score"))
    return joined


def run_accept_gate(
    *,
    label_paths: list[Path],
    action_features_path: Path,
    score_paths: list[Path],
    out_dir: Path,
    default_action_value: float,
    max_features: int,
    thresholds: list[float],
    min_keep: int,
    min_precision: float,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _normalise_labels(label_paths, default_action_value=default_action_value)
    features = _normalise_action_features(action_features_path, default_action_value=default_action_value)
    scores = _normalise_scores(score_paths, default_action_value=default_action_value)
    frame = _join_frames(labels, features, scores)
    feature_cols = _feature_columns(frame, max_features=max_features)
    frame.to_csv(out_dir / "accept_gate_joined_rows.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "selected_features.csv", index=False)

    no_filter = _summarise(frame, pd.Series(True, index=frame.index), name="no_filter", threshold=None)
    loo_rows: list[dict[str, Any]] = []
    threshold_parts: list[pd.DataFrame] = []
    for day in sorted(frame["day"].dropna().unique()):
        train = frame.loc[frame["day"].ne(day)].copy()
        test = frame.loc[frame["day"].eq(day)].copy()
        if train.empty or test.empty:
            continue
        train_pred, test_pred = _fit_predict(train, test, feature_cols, seed=seed)
        threshold, trials = _choose_threshold(
            train,
            train_pred,
            thresholds=thresholds,
            min_keep=min_keep,
            min_precision=min_precision,
        )
        trials["heldout_day"] = str(pd.Timestamp(day).date())
        threshold_parts.append(trials)
        selected = pd.Series(False, index=test.index) if threshold is None else pd.Series(test_pred, index=test.index).ge(threshold)
        row = _summarise(test, selected, name="loo_gate", threshold=threshold)
        row["heldout_day"] = str(pd.Timestamp(day).date())
        row["train_selected_threshold"] = None if threshold is None else float(threshold)
        row["test_rows"] = int(len(test))
        loo_rows.append(row)
    loo = pd.DataFrame(loo_rows)
    threshold_trials = pd.concat(threshold_parts, ignore_index=True, sort=False) if threshold_parts else pd.DataFrame()
    loo.to_csv(out_dir / "leave_one_day_gate_validation.csv", index=False)
    threshold_trials.to_csv(out_dir / "threshold_trials.csv", index=False)
    summary = {
        "generated_by": "run_c3el_exact_accept_gate",
        "label_paths": [str(path) for path in label_paths],
        "action_features_path": str(action_features_path),
        "score_paths": [str(path) for path in score_paths],
        "rows": int(len(frame)),
        "feature_rows_matched": int(frame["feature_row_matched"].sum()),
        "features": int(len(feature_cols)),
        "min_keep": int(min_keep),
        "min_precision": float(min_precision),
        "no_filter": no_filter,
    }
    if not loo.empty:
        summary["loo"] = {
            "heldout_days": int(len(loo)),
            "total_keep_count": int(pd.to_numeric(loo["keep_count"], errors="coerce").fillna(0).sum()),
            "total_delta_full_J": float(pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0).sum()),
            "positive_day_share": float(pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0).gt(0.0).mean()),
            "worst_day_delta_full_J": float(pd.to_numeric(loo["delta_full_J_sum"], errors="coerce").fillna(0.0).min()),
        }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    lines = [
        "# C3el exact-label accept gate",
        "",
        "This is a no-replay diagnostic. It trains a shallow exact-label accept/reject gate and validates it leave-one-day-out.",
        "",
        f"Rows: `{summary['rows']}`",
        f"Feature rows matched: `{summary['feature_rows_matched']}`",
        f"Features selected: `{summary['features']}`",
        "",
        "## No Filter",
        "",
        pd.DataFrame([no_filter]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Leave-One-Day",
        "",
    ]
    if loo.empty:
        lines.append("No validation rows.")
    else:
        lines.append(pd.DataFrame([summary["loo"]]).to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append(
            loo[
                [
                    "heldout_day",
                    "test_rows",
                    "train_selected_threshold",
                    "keep_count",
                    "positive_e50_rate",
                    "delta_full_J_sum",
                    "delta_full_J_worst",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--score-files", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--default-action-value", type=float, default=0.0)
    parser.add_argument("--max-features", type=int, default=24)
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument("--min-precision", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    summary = run_accept_gate(
        label_paths=list(args.labels),
        action_features_path=args.action_features,
        score_paths=list(args.score_files),
        out_dir=args.out_dir,
        default_action_value=float(args.default_action_value),
        max_features=int(args.max_features),
        thresholds=thresholds,
        min_keep=int(args.min_keep),
        min_precision=float(args.min_precision),
        seed=int(args.seed),
    )
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
