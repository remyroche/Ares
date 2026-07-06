#!/usr/bin/env python3
"""S52 two-head selector smoke.

This tests the next Gate 3 repair after post-hoc blends and gates failed:
train a dedicated opportunity head and a dedicated path-clean head, then combine
their OOF predictions. The path-clean head is trained on the actual failure
zone: first-touch positive opportunities whose full path remains acceptable.
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

try:
    from lightgbm import LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False

from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_ROUND_TRIP_COST,
    LabelConfig,
    SideParams,
    _json_safe,
    _prepare_folds,
    _score_fold,
    _summarize_trial,
)
from scripts.run_s52_ranker_smoke import (  # noqa: E402
    DEFAULT_MONTHS,
    _cap_indices,
    _materialized_soft_label,
    _scored_ledger,
)


DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260705_s52_bidirectional_first_touch_sidegeom_tp125_lsl075_ssl050_fast16_bar50_cost100bps_labels/"
    "labels"
)
DEFAULT_FEATURE_LIST = Path(
    "data_perp/reports/s52_path_risk_feature_pack_20260705_v1/"
    "s52_path_risk_ranker_feature_list.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_two_head_selector_pathrisk_features_noae_20260705_v1")
DEFAULT_CLEAN_WEIGHTS = "0,0.25,0.5,0.75,1.0,1.5,2.0"


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return sorted(set(out))


def _neutral_config(name: str) -> LabelConfig:
    side = SideParams(
        min_net_edge=0.0,
        temperature=1.0,
        mae_cap_r=0.0,
        hard_mae_cap_r=0.0,
        mae_penalty=0.0,
        mfe_min_r=0.0,
        mfe_bonus=0.0,
        mfe_mae_ratio_min=0.0,
        time_to_mfe_max_bars=0.0,
        exit_bars_min=0.0,
        exit_bars_max=0.0,
        timeout_penalty=0.0,
        late_penalty=0.0,
        dirty_positive_cap=0.0,
        timeout_cap=0.0,
        bad_mae_cap=0.0,
        post_win_mfe_min_r=0.0,
        post_win_mfe_bonus=0.0,
        first_pass_target_r=0.0,
        first_pass_bad_r=0.0,
        first_pass_reward=0.0,
        first_pass_penalty=0.0,
        adverse_pre_mfe_cap_r=0.0,
        adverse_pre_mfe_penalty=0.0,
        underwater_bars_cap=0.0,
        underwater_penalty=0.0,
        ordered_clean_floor=0.0,
        ordered_dirty_cap=0.0,
    )
    return LabelConfig(name=name, family="s52_two_head", long=side, short=side)


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def _path_clean_target(label: pd.DataFrame, metrics: pd.DataFrame, *, full_path_mae_cap: float) -> pd.Series:
    first_touch_net = _num(metrics, "first_touch_net")
    full_path_mae = _num(metrics, "first_touch_full_path_mae_norm", default=99.0)
    mfe_before = _num(metrics, "mfe_1r_before_mae_1r")
    mae_before = _num(metrics, "mae_1r_before_mfe_1r")
    timeout = _num(metrics, "is_timeout")
    first_good = _num(label, "first_pass_good")
    clean = (
        first_good.gt(0.5)
        & first_touch_net.gt(0.0)
        & full_path_mae.le(float(full_path_mae_cap))
        & mfe_before.gt(0.5)
        & mae_before.le(0.5)
        & timeout.le(0.5)
    )
    return clean.astype(np.float32)


def _sample_weight(label: pd.DataFrame, metrics: pd.DataFrame, clean: pd.Series) -> pd.Series:
    first_touch_net = _num(metrics, "first_touch_net")
    full_path_mae = _num(metrics, "first_touch_full_path_mae_norm", default=99.0)
    first_bad = _num(label, "first_pass_bad").gt(0.5)
    positive = first_touch_net.gt(0.0)
    dirty_positive = positive & first_bad
    w = pd.Series(1.0, index=metrics.index, dtype=np.float32)
    w += 2.5 * clean.astype(float)
    w += 2.0 * dirty_positive.astype(float)
    w += 1.5 * (positive & full_path_mae.ge(1.0)).astype(float)
    edge = first_touch_net.abs().clip(upper=0.02) / 0.02
    w += edge.fillna(0.0)
    return (w / max(float(w.mean()), 1e-12)).clip(0.1, 6.0).astype(np.float32)


def _fit_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    x_valid: pd.DataFrame,
    *,
    seed: int,
    objective: str = "regression",
) -> np.ndarray:
    if not _LIGHTGBM_AVAILABLE or LGBMRegressor is None:
        raise RuntimeError("lightgbm is not available")
    model = LGBMRegressor(
        objective=objective,
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=45,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        x_train.reset_index(drop=True),
        pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        sample_weight=pd.to_numeric(weights, errors="coerce").fillna(1.0).to_numpy(dtype=np.float32),
    )
    return model.predict(x_valid.reset_index(drop=True)).astype(np.float32)


def _zscore(values: pd.Series, group: pd.Series | None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if group is None:
        std = numeric.std(ddof=0)
        if not math.isfinite(float(std)) or float(std) <= 1e-12:
            return pd.Series(0.0, index=numeric.index)
        return ((numeric - numeric.mean()) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grouped = numeric.groupby(group, observed=True, dropna=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_heads_for_fold(
    fold: dict[str, Any],
    *,
    max_train_rows: int,
    full_path_mae_cap: float,
    side_specific: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_label_full = _materialized_soft_label(fold["train_frame"], fold["train_metrics"])
    valid_label = _materialized_soft_label(fold["valid_frame"], fold["valid_metrics"]).reset_index(drop=True)
    idx = _cap_indices(int(fold["train_rows"]), int(max_train_rows), seed=int(seed))
    x_train_full = fold["x_train"].iloc[idx].reset_index(drop=True)
    train_metrics = fold["train_metrics"].iloc[idx].reset_index(drop=True)
    train_label = train_label_full.iloc[idx].reset_index(drop=True)
    x_valid = fold["x_valid"].reset_index(drop=True)
    valid_metrics = fold["valid_metrics"].reset_index(drop=True)

    clean_train = _path_clean_target(train_label, train_metrics, full_path_mae_cap=float(full_path_mae_cap))
    weights = _sample_weight(train_label, train_metrics, clean_train)
    opp_y = pd.to_numeric(train_label["target_soft"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    opp_pred = np.full(len(x_valid), np.nan, dtype=np.float32)
    clean_pred = np.full(len(x_valid), np.nan, dtype=np.float32)
    if not side_specific:
        opp_pred = _fit_regressor(x_train_full, opp_y, weights, x_valid, seed=int(seed) + 1)
        clean_pred = _fit_regressor(x_train_full, clean_train, weights, x_valid, seed=int(seed) + 2)
        return opp_pred, clean_pred, valid_label

    train_side = _num(train_metrics, "side", 1.0)
    valid_side = _num(valid_metrics, "side", 1.0)
    for offset, train_mask, valid_mask in (
        (0, train_side.ge(0.0), valid_side.ge(0.0)),
        (1, train_side.lt(0.0), valid_side.lt(0.0)),
    ):
        train_idx = np.flatnonzero(train_mask.to_numpy(dtype=bool))
        valid_idx = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
        if len(train_idx) < 500 or len(valid_idx) == 0:
            continue
        opp_pred[valid_idx] = _fit_regressor(
            x_train_full.iloc[train_idx].reset_index(drop=True),
            opp_y.iloc[train_idx].reset_index(drop=True),
            weights.iloc[train_idx].reset_index(drop=True),
            x_valid.iloc[valid_idx].reset_index(drop=True),
            seed=int(seed) + 11 + offset,
        )
        clean_pred[valid_idx] = _fit_regressor(
            x_train_full.iloc[train_idx].reset_index(drop=True),
            clean_train.iloc[train_idx].reset_index(drop=True),
            weights.iloc[train_idx].reset_index(drop=True),
            x_valid.iloc[valid_idx].reset_index(drop=True),
            seed=int(seed) + 21 + offset,
        )
    for arr in (opp_pred, clean_pred):
        if np.isnan(arr).any():
            fill = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
            arr[:] = np.where(np.isfinite(arr), arr, fill).astype(np.float32)
    return opp_pred, clean_pred, valid_label


def _combine_scores(
    opp_pred: np.ndarray,
    clean_pred: np.ndarray,
    valid_metrics: pd.DataFrame,
    *,
    clean_weight: float,
) -> pd.Series:
    side = np.where(_num(valid_metrics, "side", 1.0).to_numpy(dtype=float) < 0.0, "short", "long")
    group = pd.Series(side)
    opp_z = _zscore(pd.Series(opp_pred), group)
    clean_z = _zscore(pd.Series(clean_pred), group)
    return opp_z + float(clean_weight) * clean_z


def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(frame: pd.DataFrame, cols: list[str], n: int = 30) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.6f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_pass_bad_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_p90_first_touch_full_path_mae_norm",
        "mean_top10_timeout_rate",
        "mean_long_top10_mean_first_touch_net",
        "mean_short_top10_mean_first_touch_net",
    ]
    lines = [
        "# S52 Two-Head Selector Smoke",
        "",
        "Opportunity and path-clean heads are trained OOF and combined over a clean-weight grid.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Features: `{manifest['features']}`",
        f"Full-path MAE cap: `{manifest['full_path_mae_cap']}`",
        "",
        "## Top Rows",
        "",
        fmt(summary.sort_values("objective", ascending=False), cols, n=40),
        "",
        "## Fold Rows For Top Variants",
        "",
        fmt(folds[folds["variant"].isin(summary.sort_values("objective", ascending=False)["variant"].head(8))], ["variant", "month"] + cols[2:], n=200),
        "",
    ]
    output_dir.joinpath("s52_two_head_selector.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    max_train_rows: int,
    round_trip_cost: float,
    clean_weights: list[float],
    full_path_mae_cap: float,
    include_ae_gmm_state_features: bool,
    seed: int,
) -> dict[str, str]:
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        months=months,
        spread_baseline_path=None,
        spread_rank_column="p75_spread_bps",
        target_symbol_count=None,
        max_feature_store_features=None,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=30_000,
        ae_gmm_state_feature_max_iter=32,
        seed=int(seed),
    )
    fold_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    config = _neutral_config("s52_two_head")
    for side_specific in (False, True):
        mode = "side_specific" if side_specific else "shared"
        for fold_i, fold in enumerate(folds):
            opp_pred, clean_pred, valid_label = _fit_heads_for_fold(
                fold,
                max_train_rows=int(max_train_rows),
                full_path_mae_cap=float(full_path_mae_cap),
                side_specific=bool(side_specific),
                seed=int(seed) + 100 * fold_i + (1000 if side_specific else 0),
            )
            for clean_weight in clean_weights:
                variant = f"two_head_{mode}_cleanw{int(round(float(clean_weight) * 100)):03d}"
                score = _combine_scores(
                    opp_pred,
                    clean_pred,
                    fold["valid_metrics"].reset_index(drop=True),
                    clean_weight=float(clean_weight),
                )
                row = _score_fold(
                    score,
                    valid_label,
                    fold["valid_metrics"].reset_index(drop=True),
                    str(fold["month"]),
                    round_trip_cost=float(round_trip_cost),
                )
                row.update(
                    {
                        "variant": variant,
                        "stage": variant,
                        "trial_number": 0,
                        "label_name": variant,
                        "family": "s52_two_head",
                        "side_specific": bool(side_specific),
                        "clean_weight": float(clean_weight),
                        "full_path_mae_cap": float(full_path_mae_cap),
                    }
                )
                fold_rows.append(row)
                ledgers.append(
                    _scored_ledger(
                        variant=variant,
                        fold=fold,
                        score=score.to_numpy(dtype=np.float32),
                        valid_label=valid_label,
                    )
                )

    summary_rows: list[dict[str, Any]] = []
    for variant, part in pd.DataFrame(fold_rows).groupby("variant", observed=True, dropna=False):
        summary = _summarize_trial(
            str(variant),
            0,
            config,
            part.to_dict(orient="records"),
            objective_mode="precision_topk",
        )
        summary["variant"] = str(variant)
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    ledger_df = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "s52_two_head_summary.csv"
    folds_path = output_dir / "s52_two_head_folds.csv"
    ledger_path = output_dir / "s52_two_head_scored_ledger.parquet"
    manifest_path = output_dir / "manifest.json"
    summary_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)
    ledger_df.to_parquet(ledger_path, index=False)
    manifest_out = {
        **{k: str(v) for k, v in manifest.items()},
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "months": list(months),
        "max_train_rows": int(max_train_rows),
        "round_trip_cost": float(round_trip_cost),
        "clean_weights": [float(v) for v in clean_weights],
        "full_path_mae_cap": float(full_path_mae_cap),
        "include_ae_gmm_state_features": bool(include_ae_gmm_state_features),
        "outputs": {
            "summary": str(summary_path),
            "folds": str(folds_path),
            "scored_ledger": str(ledger_path),
            "report": str(output_dir / "s52_two_head_selector.md"),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest_out), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir, summary_df, folds_df, manifest_out)
    print(f"wrote {summary_path}")
    cols = [
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_p90_first_touch_full_path_mae_norm",
        "mean_top10_timeout_rate",
    ]
    print(summary_df[cols].head(12).to_string(index=False))
    return {k: str(v) for k, v in manifest_out["outputs"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--max-train-rows", type=int, default=125_000)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--clean-weights", default=DEFAULT_CLEAN_WEIGHTS)
    parser.add_argument("--full-path-mae-cap", type=float, default=1.0)
    parser.add_argument("--include-ae-gmm-state-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months),
        max_train_rows=int(args.max_train_rows),
        round_trip_cost=float(args.round_trip_cost),
        clean_weights=_parse_floats(args.clean_weights),
        full_path_mae_cap=float(args.full_path_mae_cap),
        include_ae_gmm_state_features=bool(args.include_ae_gmm_state_features),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
