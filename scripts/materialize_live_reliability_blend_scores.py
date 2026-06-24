#!/usr/bin/env python3
"""Score live ledgers with a distilled reliability-blend model.

The original reliability-blend experiment exported OOF component/blend scores but
did not persist deployable q_fail/new_period models.  This script trains a
head-specific distillation model:

    live-available model-state features -> selected reliability blend score

on historical OOF rows, then scores post-boundary live ledgers.  It is not a
calibrated probability model; the output is a ranking score intended for the
native simple-policy replay.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_meta_recent_failures import lgb  # noqa: E402
from scripts.run_blend_native_simple_policy_replay import _load_default_variants  # noqa: E402
from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import (  # noqa: E402
    STRATEGY_IDS,
    _file_sha256,
    _json_safe,
)


HEAD_BY_STRATEGY_ID = {strategy_id: head for head, strategy_id in STRATEGY_IDS.items()}
STRATEGY_BY_HEAD = dict(STRATEGY_IDS)
DEFAULT_COMPONENT_SCORES = Path(
    "data_perp/reports/reliability_blend_optuna_20260623_full"
    "/reliability_blend_component_scores.parquet"
)
DEFAULT_META_OOF_DIR = Path("data_perp/artifacts/meta_featureselect_recentguard_20260622_0119/meta_oof")
DEFAULT_CONFIG = Path("config/reliability_blend_default_configs.json")


OOF_TO_LIVE_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("anchor_score", "__component_anchor_score__", "__live_anchor_score__"),
    ("anchor_rank_timestamp", "__component_anchor_rank__", "__live_rank__"),
    ("oof_lgbm_prob", "oof_lgbm_prob", "calibrated_score"),
    ("oof_meta_clf", "oof_meta_clf", "meta_pred"),
    ("oof_base_clf", "oof_base_clf", "base_pred"),
    ("oof_p_move", "oof_p_move", "raw_prediction_score"),
    ("oof_rank_pct", "oof_rank_pct", "policy_rank_pct"),
    ("oof_prob_uncertainty", "oof_prob_uncertainty", "prob_uncertainty"),
    ("oof_entropy", "oof_entropy", "uncertainty_score"),
    ("oof_rare_leaf_fraction", "oof_rare_leaf_fraction", "meta_lgbm_rare_leaf_fraction"),
    ("oof_leaf_count_p10", "oof_leaf_count_p10", "meta_lgbm_leaf_count_p10"),
    ("oof_leaf_count_min", "oof_leaf_count_min", "meta_lgbm_leaf_count_min"),
    ("oof_leaf_weight_p10", "oof_leaf_weight_p10", "meta_lgbm_leaf_weight_p10"),
    ("oof_leaf_depth_mean", "oof_leaf_depth_mean", "meta_lgbm_leaf_count_p10"),
    ("oof_contrib_top1_abs_share", "oof_contrib_top1_abs_share", "meta_lgbm_contrib_top1_abs_share"),
    ("oof_contrib_top3_abs_share", "oof_contrib_top3_abs_share", "meta_lgbm_contrib_top3_abs_share"),
    ("oof_contrib_entropy", "oof_contrib_entropy", "meta_lgbm_contrib_entropy"),
    ("oof_contrib_balance", "oof_contrib_balance", "meta_lgbm_contrib_balance"),
    ("oof_num_material_contrib_features", "oof_num_material_contrib_features", "meta_lgbm_num_material_contrib_features"),
    ("oof_feature_drift_psi_core", "oof_feature_drift_psi_core", "meta_lgbm_feature_drift_psi_core"),
    ("oof_feature_drift_ks_core", "oof_feature_drift_ks_core", "meta_lgbm_feature_drift_ks_core"),
    ("oof_feature_drift_cov_shift", "oof_feature_drift_cov_shift", "meta_lgbm_feature_drift_cov_shift"),
    ("oof_regime_centroid_similarity_train", "oof_regime_centroid_similarity_train", "meta_lgbm_regime_centroid_similarity_train"),
)


def _head_from_oof_path(path: Path) -> str | None:
    stem = path.name
    if not stem.startswith("meta_oof_") or not stem.endswith("_tbm_clf.parquet"):
        return None
    strategy_id = stem[len("meta_oof_") : -len("_tbm_clf.parquet")]
    return HEAD_BY_STRATEGY_ID.get(strategy_id)


def _rank01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True)


def _read_ledgers(paths: list[Path], *, start: str | None, end: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        df["_ledger_path"] = str(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No ledger paths supplied.")
    out = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    ts_col = "signal_bar_ts" if "signal_bar_ts" in out.columns else "timestamp"
    out["timestamp"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(HEAD_BY_STRATEGY_ID)
    out = out[out["head"].notna()].copy()
    out = out.dropna(subset=["timestamp", "symbol", "head"])
    out = out.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    out = out.drop_duplicates(["timestamp", "strategy_id", "symbol"], keep="last")
    if out.empty:
        raise RuntimeError("No live ledger target rows after filtering.")
    return out.reset_index(drop=True)


def _timestamp_features(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    score = pd.to_numeric(frame[score_col], errors="coerce")
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    grouped = score.groupby(ts)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    out["score_minus_ts_mean"] = score - mean
    out["score_ts_z"] = (score - mean) / std
    out["score_ts_rank"] = score.groupby(ts).rank(method="average", pct=True)
    out["timestamp_row_count_log"] = np.log1p(ts.groupby(ts).transform("size").astype(float))
    return out


def _build_train_features(comp: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    frame = comp.merge(
        oof,
        on=["timestamp", "symbol"],
        how="inner",
        suffixes=("", "_oof"),
    )
    feats = pd.DataFrame(index=frame.index)
    for name, oof_col, _live_col in OOF_TO_LIVE_FEATURES:
        if oof_col == "__component_anchor_score__":
            feats[name] = pd.to_numeric(frame["anchor_score"], errors="coerce")
        elif oof_col == "__component_anchor_rank__":
            feats[name] = pd.to_numeric(frame["anchor_rank_timestamp"], errors="coerce")
        elif oof_col in frame.columns:
            feats[name] = pd.to_numeric(frame[oof_col], errors="coerce")
    feats = pd.concat([feats, _timestamp_features(frame, "anchor_score")], axis=1)
    feats["target"] = pd.to_numeric(frame["__target_blend_score__"], errors="coerce")
    feats["timestamp"] = frame["timestamp"].to_numpy()
    return feats


def _build_live_features(live: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=live.index)
    for name, _oof_col, live_col in OOF_TO_LIVE_FEATURES:
        if live_col == "__live_anchor_score__":
            source = "calibrated_score"
        elif live_col == "__live_rank__":
            source = "policy_rank_pct" if "policy_rank_pct" in live.columns else "batch_rank_pct"
        else:
            source = live_col
        if source in live.columns:
            feats[name] = pd.to_numeric(live[source], errors="coerce")
    feats = pd.concat([feats, _timestamp_features(live, "calibrated_score")], axis=1)
    return feats


def _prepare_matrices(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    *,
    selected_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, float]]:
    if selected_cols is None:
        selected_cols = [
            c
            for c in x_train.columns
            if c not in {"target", "timestamp"}
            and pd.to_numeric(x_train[c], errors="coerce").notna().mean() >= 0.05
        ]
    medians: dict[str, float] = {}
    train_parts: list[pd.Series] = []
    valid_parts: list[pd.Series] = []
    for col in selected_cols:
        tr = pd.to_numeric(x_train[col], errors="coerce") if col in x_train.columns else pd.Series(np.nan, index=x_train.index)
        va = pd.to_numeric(x_valid[col], errors="coerce") if col in x_valid.columns else pd.Series(np.nan, index=x_valid.index)
        med = float(tr.replace([np.inf, -np.inf], np.nan).median())
        if not np.isfinite(med):
            med = 0.0
        medians[col] = med
        train_parts.append(tr.replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32").rename(col))
        valid_parts.append(va.replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32").rename(col))
    return pd.concat(train_parts, axis=1), pd.concat(valid_parts, axis=1), selected_cols, medians


def _fit_model(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    min_child = max(50, int(math.ceil(0.025 * len(y))))
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.035,
        max_depth=3,
        num_leaves=8,
        min_child_samples=min_child,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(x, y.astype("float32"))
    return model


def _sample_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    idx = np.linspace(0, len(frame) - 1, max_rows).round().astype(int)
    return frame.iloc[np.unique(idx)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="append", type=Path, required=True)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--meta-oof-dir", type=Path, default=DEFAULT_META_OOF_DIR)
    parser.add_argument("--blend-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/reliability_blend_live_scores_20260624"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-train-rows", type=int, default=90000)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = _load_default_variants(args.blend_config)
    components = pd.read_parquet(args.component_scores)
    components["timestamp"] = pd.to_datetime(components["timestamp"], utc=True, errors="coerce")
    live = _read_ledgers(args.ledger, start=args.start, end=args.end)
    live_features = _build_live_features(live)

    score_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    oof_paths = {head: path for path in args.meta_oof_dir.glob("meta_oof_*_tbm_clf.parquet") if (head := _head_from_oof_path(path))}
    for head, strategy_id in STRATEGY_BY_HEAD.items():
        variant = variants.get(head)
        target_col = f"blend_{variant}_score" if variant else ""
        if not variant or target_col not in components.columns:
            diag_rows.append({"head": head, "status": "missing_variant", "variant": variant})
            continue
        if head not in oof_paths:
            diag_rows.append({"head": head, "status": "missing_oof_path", "variant": variant})
            continue
        comp = components.loc[components["head"].astype(str).eq(head)].copy()
        comp["__target_blend_score__"] = pd.to_numeric(comp[target_col], errors="coerce")
        oof = pd.read_parquet(oof_paths[head])
        oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True, errors="coerce")
        train_frame = _build_train_features(comp, oof).dropna(subset=["target", "timestamp"])
        train_frame = train_frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if len(train_frame) < 1000:
            diag_rows.append({"head": head, "status": "insufficient_train_rows", "rows": int(len(train_frame))})
            continue
        split = int(len(train_frame) * 0.80)
        split = max(500, min(split, len(train_frame) - 500))
        train_part = _sample_rows(train_frame.iloc[:split].copy(), int(args.max_train_rows))
        valid_part = train_frame.iloc[split:].copy()
        x_tr, x_va, keep_cols, medians = _prepare_matrices(train_part, valid_part)
        y_tr = pd.to_numeric(train_part["target"], errors="coerce").to_numpy(dtype=np.float32)
        y_va = pd.to_numeric(valid_part["target"], errors="coerce").to_numpy(dtype=np.float32)
        eval_model = _fit_model(x_tr, y_tr, seed=int(args.seed))
        pred_va = eval_model.predict(x_va).astype(np.float32)
        rho = spearmanr(y_va, pred_va, nan_policy="omit").correlation
        diag = {
            "head": head,
            "strategy_id": strategy_id,
            "status": "ok",
            "variant": variant,
            "target_col": target_col,
            "train_rows_full": int(len(train_frame)),
            "train_rows_eval": int(len(train_part)),
            "valid_rows_eval": int(len(valid_part)),
            "feature_count": int(len(keep_cols)),
            "valid_spearman": float(rho) if np.isfinite(rho) else np.nan,
            "valid_mae": float(mean_absolute_error(y_va, pred_va)),
            "valid_rmse": float(math.sqrt(mean_squared_error(y_va, pred_va))),
            "target_min": float(np.nanmin(train_frame["target"])),
            "target_max": float(np.nanmax(train_frame["target"])),
        }
        full_train = _sample_rows(train_frame, int(args.max_train_rows))
        x_full, _x_dummy, keep_cols, medians = _prepare_matrices(full_train, full_train, selected_cols=keep_cols)
        y_full = pd.to_numeric(full_train["target"], errors="coerce").to_numpy(dtype=np.float32)
        final_model = _fit_model(x_full, y_full, seed=int(args.seed) + 100)
        models[head] = {"model": final_model, "features": keep_cols, "medians": medians, "diagnostics": diag}

        live_mask = live["head"].astype(str).eq(head)
        if live_mask.any():
            live_x_raw = live_features.loc[live_mask].copy()
            _x_unused, live_x, _cols, _med = _prepare_matrices(full_train.loc[:, keep_cols], live_x_raw, selected_cols=keep_cols)
            pred_live = final_model.predict(live_x).astype(np.float32)
            pred_live = np.clip(pred_live, diag["target_min"], diag["target_max"])
            frame = live.loc[live_mask, ["timestamp", "symbol", "head", "strategy_id"]].copy()
            frame["anchor_score"] = pd.to_numeric(live.loc[live_mask, "calibrated_score"], errors="coerce").to_numpy(dtype=np.float32)
            frame["reliability_blend_score"] = pred_live
            frame["score_source"] = "distilled_reliability_blend_oof_target"
            frame["blend_variant"] = variant
            score_frames.append(frame)
            diag["live_rows_scored"] = int(len(frame))
            diag["live_score_min"] = float(np.nanmin(pred_live)) if len(pred_live) else np.nan
            diag["live_score_max"] = float(np.nanmax(pred_live)) if len(pred_live) else np.nan
        else:
            diag["live_rows_scored"] = 0
        diag_rows.append(diag)

    if not score_frames:
        raise RuntimeError("No live reliability blend scores were produced.")
    scores = pd.concat(score_frames, axis=0, ignore_index=True)
    scores = scores.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    score_path = args.output_dir / "live_reliability_blend_scores.parquet"
    scores.to_parquet(score_path, index=False)
    diag_table = pd.DataFrame(diag_rows)
    diag_table.to_csv(args.output_dir / "live_reliability_blend_score_diagnostics.csv", index=False)
    joblib.dump(models, args.output_dir / "distilled_reliability_blend_models.joblib")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "materialize_live_reliability_blend_scores",
        "score_path": str(score_path),
        "component_scores": str(args.component_scores),
        "component_scores_sha256": _file_sha256(args.component_scores),
        "blend_config": str(args.blend_config),
        "blend_config_sha256": _file_sha256(args.blend_config),
        "meta_oof_dir": str(args.meta_oof_dir),
        "ledger_paths": [str(p) for p in args.ledger],
        "ledger_sha256": {str(p): _file_sha256(p) for p in args.ledger if p.exists()},
        "start": args.start,
        "end": args.end,
        "rows": int(len(scores)),
        "timestamp_min": pd.to_datetime(scores["timestamp"], utc=True).min().isoformat(),
        "timestamp_max": pd.to_datetime(scores["timestamp"], utc=True).max().isoformat(),
        "diagnostics": diag_rows,
        "feature_mapping": [
            {"name": name, "oof_col": oof_col, "live_col": live_col}
            for name, oof_col, live_col in OOF_TO_LIVE_FEATURES
        ],
    }
    (args.output_dir / "live_reliability_blend_score_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n"
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
