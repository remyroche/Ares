#!/usr/bin/env python3
"""Diagnose whether high-value zero-cut interventions are learnable.

This is intentionally diagnostic rather than a deployable policy arm. The
existing sparse size-action policy misses many exact-state oracle-positive
groups, especially multiplier=0.00 cuts. Before wiring another gate into the
portfolio replay, this script tests whether those cuts are separable from
dangerous false cuts with fold-local feature selection and hard-negative
weighting.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_PANEL = Path(
    "data_perp/reports/exact_state_size_action_learning_20260626_cached_panel_8fold_train720_eval120.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/zero_cut_oracle_classifier_diagnostic")
MATERIAL_DELTA_EPS = 1e-6

LABEL_COLUMNS = {
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "best_immediate_gain",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "best_nonbaseline_multiplier",
    "best_capacity_gain",
    "best_capacity_gain_per_notional",
    "best_immediate_gain_per_notional",
    "group_affected_notional",
    "group_can_bind",
    "y_intervene",
}

KEY_COLUMNS = {"timestamp", "strategy_id", "fold_id", "split", "multiplier"}


def _load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Panel not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"timestamp", "strategy_id", "fold_id", "split", "multiplier", "action_binds", "delta_full_J"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Panel is missing required columns: {missing}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce").astype("Int64")
    frame["multiplier"] = pd.to_numeric(frame["multiplier"], errors="coerce").fillna(1.0).astype(float)
    return frame.loc[frame["timestamp"].notna() & frame["fold_id"].notna()].copy()


def _zero_action_rows(panel: pd.DataFrame) -> pd.DataFrame:
    rows = panel.loc[np.isclose(panel["multiplier"].astype(float), 0.0)].copy()
    rows["action_binds"] = pd.to_numeric(rows.get("action_binds"), errors="coerce").fillna(0.0)
    rows = rows.loc[rows["action_binds"] > 0.0].copy()
    if rows.empty:
        return rows
    rows["zero_delta_full_J"] = pd.to_numeric(rows.get("delta_full_J"), errors="coerce").fillna(0.0)
    rows["zero_delta_immediate_J"] = pd.to_numeric(rows.get("delta_immediate_J"), errors="coerce").fillna(0.0)
    rows["zero_positive"] = rows["zero_delta_full_J"] > MATERIAL_DELTA_EPS
    rows["zero_large_positive"] = rows["zero_delta_full_J"] > 50.0
    rows["zero_false"] = rows["zero_delta_full_J"] <= MATERIAL_DELTA_EPS
    rows["strategy_code"] = pd.factorize(rows["strategy_id"].astype(str), sort=True)[0].astype(float)
    return rows.sort_values(["fold_id", "split", "timestamp", "strategy_id"])


def _apply_target_mode(rows: pd.DataFrame, *, target_mode: str, material_gain: float) -> pd.DataFrame:
    out = rows.copy()
    delta = pd.to_numeric(out["zero_delta_full_J"], errors="coerce").fillna(0.0)
    mode = str(target_mode)
    if mode == "positive_vs_false":
        out["target_positive"] = delta > MATERIAL_DELTA_EPS
        out["target_trainable"] = True
    elif mode == "high_value_vs_dangerous":
        out["target_positive"] = delta >= float(material_gain)
        out["target_trainable"] = (delta >= float(material_gain)) | (delta <= -float(material_gain))
    elif mode == "high_value_vs_nonpositive":
        out["target_positive"] = delta >= float(material_gain)
        out["target_trainable"] = (delta >= float(material_gain)) | (delta <= MATERIAL_DELTA_EPS)
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return out


def _feature_columns(rows: pd.DataFrame, max_features: int | None = None) -> list[str]:
    exclude = set(LABEL_COLUMNS) | set(KEY_COLUMNS)
    numeric_cols: list[str] = []
    for col in rows.columns:
        if col in exclude or col.startswith("zero_"):
            continue
        if pd.api.types.is_bool_dtype(rows[col]) or pd.api.types.is_numeric_dtype(rows[col]):
            vals = pd.to_numeric(rows[col], errors="coerce")
            if vals.replace([np.inf, -np.inf], np.nan).notna().sum() > 0 and vals.nunique(dropna=True) > 1:
                numeric_cols.append(col)
    # Keep the diagnostic compact and aligned with the objective: portfolio,
    # strategy opportunity and action-impact fields first.
    priority_prefixes = (
        "projected_",
        "strategy_",
        "timestamp_",
        "remaining_",
        "open_",
        "notional_exiting_",
        "positions_exiting_",
        "cooldown_",
        "side_",
        "symbol_",
        "unrealized_",
        "wallet",
    )
    numeric_cols = sorted(
        numeric_cols,
        key=lambda c: (
            0 if c == "strategy_code" else 1 if c.startswith(priority_prefixes) else 2,
            c,
        ),
    )
    if max_features and max_features > 0:
        return numeric_cols[: int(max_features)]
    return numeric_cols


def _matrix(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = rows[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.fillna(med).astype(np.float32)


def _sample_weights(train: pd.DataFrame, *, negative_harm_weight: float, positive_gain_weight: float) -> np.ndarray:
    delta = pd.to_numeric(train["zero_delta_full_J"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = train["target_positive"].astype(bool).to_numpy()
    pos_scale = np.nanmedian(np.abs(delta[y])) if np.any(y) else 1.0
    neg_scale = np.nanmedian(np.abs(delta[~y])) if np.any(~y) else 1.0
    pos_scale = max(float(pos_scale), 1.0)
    neg_scale = max(float(neg_scale), 1.0)
    weights = np.ones(len(train), dtype=float)
    weights[y] *= 1.0 + float(positive_gain_weight) * np.clip(delta[y] / pos_scale, 0.0, 10.0)
    weights[~y] *= 1.0 + float(negative_harm_weight) * np.clip(np.abs(np.minimum(delta[~y], 0.0)) / neg_scale, 0.0, 20.0)
    ambiguous = np.abs(delta) < 10.0
    weights[ambiguous] *= 0.50
    return weights


def _fit_lgbm(train: pd.DataFrame, features: list[str], seed: int, weights: np.ndarray) -> Any:
    from lightgbm import LGBMClassifier

    y = train["target_positive"].astype(int).to_numpy()
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=max(20, int(0.03 * len(train))),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=1,
        verbosity=-1,
        scale_pos_weight=float(neg / pos),
    )
    model.fit(_matrix(train, features), y, sample_weight=weights)
    return model


def _select_features_fold(train: pd.DataFrame, features: list[str], seed: int, weights: np.ndarray, top_k: int) -> list[str]:
    if len(features) <= top_k:
        return features
    model = _fit_lgbm(train, features, seed, weights)
    gains = np.asarray(getattr(model, "feature_importances_", np.zeros(len(features))), dtype=float)
    order = np.argsort(-gains)
    selected = [features[i] for i in order[: int(top_k)] if gains[i] > 0]
    if len(selected) < min(10, len(features)):
        selected = [features[i] for i in order[: int(top_k)]]
    return selected


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _top_fraction_rows(eval_rows: pd.DataFrame, score_col: str, fractions: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_positive = int(eval_rows["target_positive"].sum())
    total_gain = float(eval_rows.loc[eval_rows["target_positive"], "zero_delta_full_J"].sum())
    total_false_loss = float(eval_rows.loc[~eval_rows["target_positive"], "zero_delta_full_J"].clip(upper=0.0).sum())
    for frac in fractions:
        n_top = max(1, int(np.ceil(len(eval_rows) * float(frac))))
        chosen = eval_rows.nlargest(min(n_top, len(eval_rows)), score_col).copy()
        positives = chosen["target_positive"].astype(bool)
        false = ~positives
        false_delta = pd.to_numeric(chosen.loc[false, "zero_delta_full_J"], errors="coerce").fillna(0.0)
        pos_delta = pd.to_numeric(chosen.loc[positives, "zero_delta_full_J"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "top_fraction": float(frac),
                "selected_groups": int(len(chosen)),
                "selected_positive_groups": int(positives.sum()),
                "selected_false_groups": int(false.sum()),
                "precision": float(positives.mean()) if len(chosen) else 0.0,
                "recall": float(positives.sum() / max(total_positive, 1)),
                "selected_positive_gain": float(pos_delta.sum()),
                "selected_false_delta_sum": float(false_delta.sum()),
                "selected_net_delta_sum": float(pos_delta.sum() + false_delta.sum()),
                "gain_capture": float(pos_delta.sum() / max(total_gain, 1.0)),
                "total_positive_groups": total_positive,
                "total_positive_gain": total_gain,
                "total_false_delta_sum": total_false_loss,
            }
        )
    return pd.DataFrame(rows)


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    panel = _load_panel(args.panel)
    rows = _apply_target_mode(_zero_action_rows(panel), target_mode=args.target_mode, material_gain=args.material_gain)
    if rows.empty:
        raise SystemExit("No binding multiplier=0 action rows found.")
    features = _feature_columns(rows, max_features=args.initial_feature_cap)
    if len(features) < 5:
        raise SystemExit(f"Too few usable features: {features}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for fold_id in sorted(rows["fold_id"].dropna().unique()):
        fold = int(fold_id)
        train = rows.loc[(rows["fold_id"].astype(int) == fold) & rows["split"].eq("train")].copy()
        eval_rows = rows.loc[(rows["fold_id"].astype(int) == fold) & rows["split"].eq("eval")].copy()
        train = train.loc[train["target_trainable"].eq(True) & train["target_positive"].notna()].copy()
        eval_rows = eval_rows.loc[eval_rows["target_positive"].notna()].copy()
        if len(train) < 30 or len(eval_rows) < 10 or train["target_positive"].nunique() < 2:
            continue
        weights = _sample_weights(
            train,
            negative_harm_weight=args.negative_harm_weight,
            positive_gain_weight=args.positive_gain_weight,
        )
        selected_features = _select_features_fold(train, features, args.seed + fold, weights, args.selected_feature_count)
        model = _fit_lgbm(train, selected_features, args.seed + 1000 + fold, weights)
        eval_scored = eval_rows.copy()
        eval_scored["zero_cut_score"] = model.predict_proba(_matrix(eval_scored, selected_features))[:, 1]
        y_eval = eval_scored["target_positive"].astype(int).to_numpy()
        score = eval_scored["zero_cut_score"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "fold_id": fold,
                "train_groups": int(len(train)),
                "eval_groups": int(len(eval_scored)),
                "train_positive_groups": int(train["target_positive"].sum()),
                "eval_positive_groups": int(eval_scored["target_positive"].sum()),
                "eval_any_positive_groups": int(eval_scored["zero_positive"].sum()),
                "feature_count": int(len(selected_features)),
                "auc": _safe_auc(y_eval, score),
                "ap": _safe_ap(y_eval, score),
                "eval_positive_gain": float(eval_scored.loc[eval_scored["target_positive"], "zero_delta_full_J"].sum()),
                "eval_any_positive_gain": float(eval_scored.loc[eval_scored["zero_positive"], "zero_delta_full_J"].sum()),
                "eval_false_delta_sum": float(eval_scored.loc[~eval_scored["target_positive"], "zero_delta_full_J"].clip(upper=0.0).sum()),
            }
        )
        for rank, feature in enumerate(selected_features, start=1):
            feature_rows.append({"fold_id": fold, "rank": rank, "feature": feature})
        scored_frames.append(eval_scored)

    if not scored_frames:
        raise SystemExit("No folds could be scored.")

    scored = pd.concat(scored_frames, ignore_index=True)
    fold_summary = pd.DataFrame(fold_rows)
    feature_summary = pd.DataFrame(feature_rows)
    top_rows = []
    for fold_id, fold_eval in scored.groupby("fold_id", sort=True):
        part = _top_fraction_rows(fold_eval, "zero_cut_score", args.top_fractions)
        part.insert(0, "fold_id", int(fold_id))
        top_rows.append(part)
    top_summary = pd.concat(top_rows, ignore_index=True)
    pooled_top = _top_fraction_rows(scored, "zero_cut_score", args.top_fractions)
    pooled_top.insert(0, "fold_id", "pooled")

    scored.to_csv(out_dir / "zero_cut_oracle_oof_scores.csv", index=False)
    fold_summary.to_csv(out_dir / "zero_cut_oracle_fold_metrics.csv", index=False)
    feature_summary.to_csv(out_dir / "zero_cut_oracle_selected_features.csv", index=False)
    pd.concat([top_summary, pooled_top], ignore_index=True).to_csv(out_dir / "zero_cut_oracle_top_fraction_metrics.csv", index=False)

    manifest = {
        "generated_by": "diagnose_zero_cut_oracle_classifier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": str(args.panel),
        "output_dir": str(out_dir),
        "initial_feature_cap": int(args.initial_feature_cap),
        "selected_feature_count": int(args.selected_feature_count),
        "target_mode": str(args.target_mode),
        "material_gain": float(args.material_gain),
        "negative_harm_weight": float(args.negative_harm_weight),
        "positive_gain_weight": float(args.positive_gain_weight),
        "seed": int(args.seed),
        "folds_scored": int(fold_summary["fold_id"].nunique()),
        "pooled_auc": _safe_auc(scored["target_positive"].astype(int).to_numpy(), scored["zero_cut_score"].to_numpy(dtype=float)),
        "pooled_ap": _safe_ap(scored["target_positive"].astype(int).to_numpy(), scored["zero_cut_score"].to_numpy(dtype=float)),
        "outputs": {
            "scores": str(out_dir / "zero_cut_oracle_oof_scores.csv"),
            "fold_metrics": str(out_dir / "zero_cut_oracle_fold_metrics.csv"),
            "selected_features": str(out_dir / "zero_cut_oracle_selected_features.csv"),
            "top_fraction_metrics": str(out_dir / "zero_cut_oracle_top_fraction_metrics.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Zero-Cut Oracle Classifier Diagnostic",
        "",
        f"Panel: `{args.panel}`",
        f"Pooled AUC: `{manifest['pooled_auc']:.4f}`",
        f"Pooled AP: `{manifest['pooled_ap']:.4f}`",
        "",
        "## Fold Metrics",
        "",
        fold_summary.to_markdown(index=False),
        "",
        "## Pooled Top-Fraction Metrics",
        "",
        pooled_top.to_markdown(index=False),
        "",
    ]
    (out_dir / "zero_cut_oracle_classifier_report.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-feature-cap", type=int, default=96)
    parser.add_argument("--selected-feature-count", type=int, default=48)
    parser.add_argument("--negative-harm-weight", type=float, default=0.50)
    parser.add_argument("--positive-gain-weight", type=float, default=0.25)
    parser.add_argument(
        "--target-mode",
        choices=["positive_vs_false", "high_value_vs_dangerous", "high_value_vs_nonpositive"],
        default="positive_vs_false",
    )
    parser.add_argument("--material-gain", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--top-fractions", type=float, nargs="*", default=[0.01, 0.025, 0.05, 0.075, 0.10, 0.15])
    return parser.parse_args()


def main() -> None:
    manifest = run_diagnostic(parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
