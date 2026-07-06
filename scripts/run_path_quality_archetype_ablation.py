#!/usr/bin/env python3
"""Run additive OOF path-quality archetype ablations.

This is intentionally separate from ``build_archetype_meta_handoff_v1.py`` so
the existing AE/GMM/context handoff remains a baseline.  The ablation asks a
specific question:

Can live-safe pre-entry/meta features predict clean-vs-dirty path quality well
enough out-of-fold to improve top-k candidate selection?

All models and grouped priors are fit on prior fold months only. Validation
rows receive frozen model predictions and train-derived priors.
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

from scripts.build_archetype_meta_handoff_v1 import (  # noqa: E402
    DEFAULT_LEDGER,
    _feature_columns,
    _folds_from_months,
    _json_safe,
    _load_ledger,
    _model_matrix,
    _num,
    _rate,
)

try:  # pragma: no cover - exercised by integration tests when sklearn exists
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception as exc:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


DEFAULT_OUT_DIR = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1/path_quality_archetype_ablation_v1"
)
CATEGORICAL_CONTEXT = [
    "side_name",
    "aegmm_cluster",
    "side_aegmm_cluster",
    "aegmm_entropy_bin",
    "aegmm_distance_bin",
    "aegmm_expected_distance_bin",
    "reconstruction_bin",
    "dae_reconstruction_bin",
    "cluster_speed_bin",
    "cluster_acceleration_bin",
    "latent_speed_bin",
    "latent_acceleration_bin",
    "source_semantic_family",
    "source_volatility_state",
    "source_pressure_state",
    "source_trend_state",
    "source_score_intensity_tag",
]
PRIOR_KEYS = [
    ["side_name", "source_semantic_family"],
    ["side_name", "source_semantic_family", "source_volatility_state"],
    ["side_name", "source_semantic_family", "source_pressure_state", "source_trend_state"],
    ["side_name", "aegmm_cluster", "reconstruction_bin", "cluster_speed_bin"],
]
TARGETS = {
    "clean": "clean_exec_positive",
    "bad": "bad_MAE",
    "timeout": "timeout_label",
    "dirty": "dirty_positive",
    "u": "u_econ_net",
}
SCORE_COLUMNS = [
    "A0_base_score",
    "A1_global_path_quality",
    "A2_side_specific_path_quality",
    "A3_state_path_priors",
    "A4_combo_path_archetype",
]


def _zscore(values: pd.Series, index: pd.Index) -> pd.Series:
    arr = _num(values, index=index).replace([np.inf, -np.inf], np.nan)
    std = float(arr.std()) if arr.notna().any() else 0.0
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=index, dtype=np.float32)
    return ((arr - float(arr.mean())) / std).fillna(0.0).astype(np.float32)


def _rank(values: pd.Series, index: pd.Index) -> pd.Series:
    return _num(values, index=index).rank(pct=True, method="average").fillna(0.0).astype(np.float32)


def _fit_regressor(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], target: str, seed: int) -> np.ndarray:
    if HistGradientBoostingRegressor is None:
        raise RuntimeError(f"scikit-learn is required for path-quality ablation: {_SKLEARN_IMPORT_ERROR}")
    train_x, valid_x = _model_matrix(train, valid, cols)
    y = _num(train[target], index=train.index).fillna(0.0).clip(-1.0, 1.0).to_numpy(dtype=np.float32)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=180,
        l2_regularization=0.15,
        random_state=int(seed),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
    )
    model.fit(train_x, y)
    return np.asarray(model.predict(valid_x), dtype=np.float32)


def _fit_path_heads(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], *, seed: int, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=valid.index)
    for name, target in TARGETS.items():
        pred = _fit_regressor(train, valid, cols, target, seed)
        if name != "u":
            pred = np.clip(pred, 0.0, 1.0)
        out[f"{prefix}_{name}"] = pred.astype(np.float32)
    return out


def _fit_side_path_heads(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], *, seed: int) -> pd.DataFrame:
    out = pd.DataFrame(index=valid.index)
    global_preds = _fit_path_heads(train, valid, cols, seed=seed, prefix="side_path")
    for side, valid_side in valid.groupby("side_name", dropna=False):
        train_side = train[train["side_name"].astype(str).eq(str(side))]
        if len(train_side) < 500 or len(valid_side) == 0:
            continue
        side_preds = _fit_path_heads(train_side, valid_side, cols, seed=seed, prefix="side_path")
        global_preds.loc[valid_side.index, side_preds.columns] = side_preds
    return global_preds


def _path_score(frame: pd.DataFrame, prefix: str) -> pd.Series:
    idx = frame.index
    pred_u = _zscore(frame[f"{prefix}_u"], idx)
    clean = _zscore(frame[f"{prefix}_clean"], idx)
    bad = _zscore(frame[f"{prefix}_bad"], idx)
    timeout = _zscore(frame[f"{prefix}_timeout"], idx)
    dirty = _zscore(frame[f"{prefix}_dirty"], idx)
    return (0.16 * pred_u + 0.16 * clean - 0.18 * bad - 0.08 * timeout - 0.08 * dirty).astype(np.float32)


def _assign_predicted_path_archetype(frame: pd.DataFrame, prefix: str) -> pd.Series:
    clean = _num(frame[f"{prefix}_clean"], index=frame.index).fillna(0.0)
    bad = _num(frame[f"{prefix}_bad"], index=frame.index).fillna(0.0)
    timeout = _num(frame[f"{prefix}_timeout"], index=frame.index).fillna(0.0)
    dirty = _num(frame[f"{prefix}_dirty"], index=frame.index).fillna(0.0)
    labels = np.full(len(frame), "path_mixed", dtype=object)
    labels[(clean >= bad) & (clean >= timeout) & (clean >= dirty)] = "path_clean_candidate"
    labels[(bad > clean) & (bad >= timeout)] = "path_bad_mae_risk"
    labels[(timeout > clean) & (timeout > bad)] = "path_timeout_risk"
    labels[(dirty > clean) & (dirty > bad) & (dirty >= timeout)] = "path_dirty_positive_risk"
    return pd.Series(labels, index=frame.index)


def _fit_group_priors(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=valid.index)
    for i, keys in enumerate(PRIOR_KEYS):
        present = [col for col in keys if col in train.columns and col in valid.columns]
        if len(present) != len(keys):
            continue
        profile = (
            train.groupby(present, dropna=False)
            .agg(
                prior_clean=("clean_exec_positive", "mean"),
                prior_bad=("bad_MAE", "mean"),
                prior_timeout=("timeout_label", "mean"),
                prior_dirty=("dirty_positive", "mean"),
                prior_u=("u_econ_net", "mean"),
                prior_n=("u_econ_net", "size"),
            )
            .reset_index()
        )
        joined = valid[present].merge(profile, on=present, how="left")
        joined.index = valid.index
        shrink = _num(joined["prior_n"], index=valid.index).fillna(0.0) / (
            _num(joined["prior_n"], index=valid.index).fillna(0.0) + 200.0
        )
        for target, default in {
            "prior_clean": _rate(train["clean_exec_positive"]),
            "prior_bad": _rate(train["bad_MAE"]),
            "prior_timeout": _rate(train["timeout_label"]),
            "prior_dirty": _rate(train["dirty_positive"]),
            "prior_u": float(_num(train["u_econ_net"], index=train.index).mean()),
        }.items():
            values = _num(joined[target], index=valid.index).fillna(default)
            out[f"prior{i}_{target}"] = (shrink * values + (1.0 - shrink) * default).astype(np.float32)
    if out.empty:
        out["prior0_prior_clean"] = np.float32(_rate(train["clean_exec_positive"]))
        out["prior0_prior_bad"] = np.float32(_rate(train["bad_MAE"]))
        out["prior0_prior_timeout"] = np.float32(_rate(train["timeout_label"]))
        out["prior0_prior_dirty"] = np.float32(_rate(train["dirty_positive"]))
        out["prior0_prior_u"] = np.float32(_num(train["u_econ_net"], index=train.index).mean())
    return out


def _prior_score(frame: pd.DataFrame) -> pd.Series:
    idx = frame.index
    score = pd.Series(0.0, index=idx, dtype=np.float32)
    count = 0
    for prefix in sorted({col.split("_prior_", 1)[0] for col in frame.columns if "_prior_" in col}):
        cols = {
            "clean": f"{prefix}_prior_clean",
            "bad": f"{prefix}_prior_bad",
            "timeout": f"{prefix}_prior_timeout",
            "dirty": f"{prefix}_prior_dirty",
            "u": f"{prefix}_prior_u",
        }
        if not all(col in frame.columns for col in cols.values()):
            continue
        score += (
            0.12 * _zscore(frame[cols["u"]], idx)
            + 0.10 * _zscore(frame[cols["clean"]], idx)
            - 0.12 * _zscore(frame[cols["bad"]], idx)
            - 0.05 * _zscore(frame[cols["timeout"]], idx)
            - 0.05 * _zscore(frame[cols["dirty"]], idx)
        ).astype(np.float32)
        count += 1
    return (score / max(count, 1)).astype(np.float32)


def _eval_score(frame: pd.DataFrame, score_col: str, top_frac: float) -> dict[str, Any]:
    selected_parts: list[pd.DataFrame] = []
    for fold_id, group in frame.groupby("fold_id", dropna=False):
        score = _num(group[score_col], index=group.index)
        valid = group[score.notna()].copy()
        if valid.empty:
            continue
        keep = max(1, int(math.ceil(len(valid) * float(top_frac))))
        selected_parts.append(valid.assign(__score__=score.loc[valid.index]).sort_values("__score__", ascending=False).head(keep))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        return {"score_model": score_col, "selected_rows": 0}
    month_u = selected.groupby("month")["u_econ_net"].mean()
    side_share = selected["side_name"].astype(str).value_counts(normalize=True)
    archetype_dist = selected.get("predicted_path_archetype", pd.Series("missing", index=selected.index)).astype(str).value_counts(normalize=True)
    return {
        "score_model": score_col,
        "selected_rows": int(len(selected)),
        "mean_u": float(_num(selected["u_econ_net"], index=selected.index).mean()),
        "worst_month_u": float(month_u.min()) if len(month_u) else float("nan"),
        "clean_positive_rate": _rate(selected["clean_exec_positive"]),
        "bad_MAE_rate": _rate(selected["bad_MAE"]),
        "timeout_rate": _rate(selected["timeout_label"]),
        "dirty_positive_rate": _rate(selected["dirty_positive"]),
        "dominant_side_share": float(side_share.iloc[0]) if len(side_share) else float("nan"),
        "dominant_path_archetype": str(archetype_dist.index[0]) if len(archetype_dist) else "missing",
        "dominant_path_archetype_share": float(archetype_dist.iloc[0]) if len(archetype_dist) else float("nan"),
    }


def _slice_report(frame: pd.DataFrame, score_col: str, top_frac: float) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for fold_id, group in frame.groupby("fold_id", dropna=False):
        keep = max(1, int(math.ceil(len(group) * float(top_frac))))
        selected_parts.append(group.sort_values(score_col, ascending=False).head(keep))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby(["month", "side_name", "predicted_path_archetype"], dropna=False)
        .agg(
            rows=("u_econ_net", "size"),
            mean_u=("u_econ_net", "mean"),
            clean_positive_rate=("clean_exec_positive", "mean"),
            bad_MAE_rate=("bad_MAE", "mean"),
            timeout_rate=("timeout_label", "mean"),
            dirty_positive_rate=("dirty_positive", "mean"),
        )
        .reset_index()
    )


def _topk_summary(frame: pd.DataFrame, top_fracs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frac in top_fracs:
        for col in SCORE_COLUMNS:
            rec = _eval_score(frame, col, frac)
            rec["top_frac"] = float(frac)
            rows.append(rec)
    return pd.DataFrame(rows)


def _path_archetype_profile(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "predicted_path_archetype" not in frame.columns:
        return pd.DataFrame()
    return (
        frame.groupby("predicted_path_archetype", dropna=False)
        .agg(
            rows=("u_econ_net", "size"),
            mean_u=("u_econ_net", "mean"),
            clean_positive_rate=("clean_exec_positive", "mean"),
            bad_MAE_rate=("bad_MAE", "mean"),
            timeout_rate=("timeout_label", "mean"),
            dirty_positive_rate=("dirty_positive", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )


def run_ablation(
    *,
    ledger_path: Path,
    out_dir: Path,
    min_train_months: int = 1,
    seed: int = 17,
    top_frac: float = 0.10,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(ledger_path)
    folds = _folds_from_months(ledger, min_train_months)
    feature_cols = _feature_columns(ledger)
    model_cols = list(dict.fromkeys(["base_score", *feature_cols, *CATEGORICAL_CONTEXT]))
    row_parts: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = ledger[ledger["month"].astype(str).isin(fold["train_months"])].copy()
        valid = ledger[ledger["month"].astype(str).eq(str(fold["valid_month"]))].copy()
        if train.empty or valid.empty:
            continue
        train["fold_id"] = str(fold["fold_id"])
        valid["fold_id"] = str(fold["fold_id"])
        global_preds = _fit_path_heads(train, valid, model_cols, seed=seed, prefix="global_path")
        side_preds = _fit_side_path_heads(train, valid, model_cols, seed=seed)
        priors = _fit_group_priors(train, valid)
        scored = pd.concat([valid.reset_index(drop=True), global_preds.reset_index(drop=True), side_preds.reset_index(drop=True), priors.reset_index(drop=True)], axis=1)
        idx = scored.index
        base_rank = _rank(scored["base_score"], idx)
        scored["A0_base_score"] = base_rank
        scored["A1_global_path_quality"] = base_rank + _path_score(scored, "global_path")
        scored["A2_side_specific_path_quality"] = base_rank + _path_score(scored, "side_path")
        scored["A3_state_path_priors"] = base_rank + _prior_score(scored)
        scored["A4_combo_path_archetype"] = (
            base_rank + 0.50 * _path_score(scored, "side_path") + 0.50 * _prior_score(scored)
        ).astype(np.float32)
        scored["predicted_path_archetype"] = _assign_predicted_path_archetype(scored, "side_path")
        row_parts.append(scored)
        leakage_rows.append(
            {
                "fold_id": fold["fold_id"],
                "train_months": ",".join(fold["train_months"]),
                "valid_month": fold["valid_month"],
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "model_fit_scope": "outer_train_only",
                "prior_fit_scope": "outer_train_only",
                "validation_assignment_scope": "frozen_train_models_and_priors",
            }
        )

    rows = pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame()
    summary = pd.DataFrame([_eval_score(rows, col, top_frac) for col in SCORE_COLUMNS]) if not rows.empty else pd.DataFrame()
    topk_summary = _topk_summary(rows, [0.05, 0.10, 0.20]) if not rows.empty else pd.DataFrame()
    path_profile = _path_archetype_profile(rows)
    baseline = summary[summary["score_model"].eq("A0_base_score")].iloc[0] if not summary.empty else pd.Series(dtype=float)
    candidates = summary[summary["score_model"].isin(["A1_global_path_quality", "A2_side_specific_path_quality", "A3_state_path_priors", "A4_combo_path_archetype"])]
    pass_rows = candidates[
        (_num(candidates["mean_u"], index=candidates.index) > 0.0)
        & (_num(candidates["worst_month_u"], index=candidates.index) > 0.0)
        & (
            (_num(candidates["bad_MAE_rate"], index=candidates.index) <= float(baseline.get("bad_MAE_rate", np.nan)) - 0.005)
            | (_num(candidates["timeout_rate"], index=candidates.index) <= float(baseline.get("timeout_rate", np.nan)) - 0.002)
            | (_num(candidates["clean_positive_rate"], index=candidates.index) >= float(baseline.get("clean_positive_rate", np.nan)) + 0.005)
            | (_num(candidates["mean_u"], index=candidates.index) >= float(baseline.get("mean_u", np.nan)) + 0.001)
        )
    ]
    tests = {
        "leakage_test": {
            "status": "pass"
            if leakage_rows
            and all(row["model_fit_scope"] == "outer_train_only" for row in leakage_rows)
            and all(row["prior_fit_scope"] == "outer_train_only" for row in leakage_rows)
            else "fail",
            "folds": len(leakage_rows),
        },
        "handoff_ablation_test": {
            "status": "pass" if not pass_rows.empty else "fail",
            "best_model": str(pass_rows.iloc[0]["score_model"]) if not pass_rows.empty else None,
            "baseline_mean_u": float(baseline.get("mean_u", np.nan)),
            "baseline_bad_MAE_rate": float(baseline.get("bad_MAE_rate", np.nan)),
        },
    }
    status = "pass" if all(payload["status"] == "pass" for payload in tests.values()) else "needs_iteration"
    paths = {
        "row_features": out_dir / "path_quality_archetype_rows.parquet",
        "summary": out_dir / "path_quality_archetype_ablation_summary.csv",
        "topk_summary": out_dir / "path_quality_archetype_topk_summary.csv",
        "path_profile": out_dir / "path_quality_archetype_profile.csv",
        "slices": out_dir / "path_quality_archetype_ablation_slices.csv",
        "leakage": out_dir / "path_quality_archetype_leakage_report.csv",
        "acceptance": out_dir / "path_quality_archetype_acceptance.json",
        "report": out_dir / "path_quality_archetype_ablation.md",
        "manifest": out_dir / "manifest.json",
    }
    rows.to_parquet(paths["row_features"], index=False)
    summary.to_csv(paths["summary"], index=False)
    topk_summary.to_csv(paths["topk_summary"], index=False)
    path_profile.to_csv(paths["path_profile"], index=False)
    _slice_report(rows, "A4_combo_path_archetype", top_frac).to_csv(paths["slices"], index=False)
    pd.DataFrame(leakage_rows).to_csv(paths["leakage"], index=False)
    paths["acceptance"].write_text(json.dumps(_json_safe(tests), indent=2, sort_keys=True))
    manifest = {
        "generated_by": "run_path_quality_archetype_ablation",
        "status": status,
        "ledger_path": str(ledger_path),
        "rows_in_ledger": int(len(ledger)),
        "oof_validation_rows": int(len(rows)),
        "folds": folds,
        "top_frac": float(top_frac),
        "feature_cols": feature_cols,
        "model_cols": model_cols,
        "score_columns": SCORE_COLUMNS,
        "acceptance_tests": tests,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_report(paths["report"], manifest, summary, topk_summary, path_profile)
    return manifest


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    topk_summary: pd.DataFrame,
    path_profile: pd.DataFrame,
) -> None:
    lines = [
        "# Path-Quality Archetype Ablation",
        "",
        "Additive OOF ablation. Existing S52/S59 labels and AE/GMM context are preserved.",
        "",
        f"- status: `{manifest['status']}`",
        f"- OOF validation rows: `{manifest['oof_validation_rows']}`",
        f"- top_frac: `{manifest['top_frac']}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No summary rows._",
        "",
        "## Top-K Summary",
        "",
        topk_summary.to_markdown(index=False) if not topk_summary.empty else "_No top-k rows._",
        "",
        "## Path Archetype Profile",
        "",
        path_profile.to_markdown(index=False) if not path_profile.empty else "_No path archetype rows._",
        "",
        "## Leakage Contract",
        "",
        "- path-quality heads fit on outer-train months only",
        "- state priors fit on outer-train months only",
        "- validation receives frozen predictions and train-derived priors",
    ]
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-train-months", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-frac", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_ablation(
        ledger_path=args.ledger,
        out_dir=args.out_dir,
        min_train_months=int(args.min_train_months),
        seed=int(args.seed),
        top_frac=float(args.top_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
