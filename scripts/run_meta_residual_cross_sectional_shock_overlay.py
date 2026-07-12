#!/usr/bin/env python3
"""Walk-forward market-shock risk overlay for the residual-meta champion.

The challenger is intentionally narrow: only market-wide pre-entry lifecycle
features are used, models are local to side x base archetype, and a single
overlay coefficient is selected on March burn-in before April-June OOS.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    EVAL_MONTHS,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
)

CHAMPION = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"
ARM = f"{CHAMPION}_cross_sectional_shock"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
MARKET_FEATURES = [
    "mkt_median_oi_chg_1h_rz",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_1h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_pct_oi_drawdown_24h_lt_minus5pct",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_median_oi_recovery_fraction_24h",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_pct_price_down_oi_down_1h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_up_oi_down_4h",
    "mkt_median_long_flush_intensity_4h",
    "mkt_median_short_cover_intensity_1h",
    "market_breadth_recovery_from_24h_min",
    "market_breadth_drawdown_from_6h_max",
    "market_pct_recovering_from_24h_low",
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_24h",
    "market_pc1_variance_share_chg_4h",
    "market_downside_pairwise_corr_24h",
    "market_downside_corr_minus_unconditional_corr_24h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
]
ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


@dataclass
class _RiskModel:
    model: Any
    medians: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    prediction_mean: float
    prediction_std: float
    support_timestamps: int

    def predict_z(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = frame.reindex(columns=MARKET_FEATURES).to_numpy(
            dtype=np.float32, copy=True
        )
        values = np.where(np.isfinite(values), values, self.medians)
        values = np.clip(values, self.clip_low, self.clip_high)
        pred = np.asarray(self.model.predict(values), dtype=np.float32)
        z = np.clip(
            (pred - np.float32(self.prediction_mean))
            / np.float32(max(self.prediction_std, 1e-3)),
            -3.0,
            3.0,
        ).astype(np.float32, copy=False)
        return pred, z


@dataclass
class _RiskState:
    local_models: dict[tuple[str, str], _RiskModel]
    side_models: dict[str, _RiskModel]
    train_end: str

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = np.zeros(len(frame), dtype=np.float32)
        z = np.zeros(len(frame), dtype=np.float32)
        local = np.zeros(len(frame), dtype=np.int8)
        side = frame["side_name"].astype(str).str.lower()
        arch = frame["archetype_policy_key"].astype(str)
        groups = pd.DataFrame({"side": side, "arch": arch}, index=frame.index)
        for (side_key, arch_key), idx in groups.groupby(
            ["side", "arch"], sort=False
        ).groups.items():
            model = self.local_models.get((str(side_key), str(arch_key)))
            is_local = model is not None
            if model is None:
                model = self.side_models.get(str(side_key))
            if model is None:
                continue
            positions = frame.index.get_indexer(idx)
            p, zz = model.predict_z(frame.loc[idx])
            pred[positions] = p
            z[positions] = zz
            local[positions] = np.int8(is_local)
        return pred, z, local


def _month_side_rank(frame: pd.DataFrame, score_col: str) -> np.ndarray:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    month = ts.dt.to_period("M").astype(str)
    score = pd.to_numeric(frame[score_col], errors="coerce")
    key = month.astype(str) + "|" + frame["side_name"].astype(str).str.lower()
    return (
        score.groupby(key, sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def _fit_one(group: pd.DataFrame, seed: int, max_depth: int) -> _RiskModel | None:
    if len(group) < 2_000:
        return None
    top20 = group["training_rank_pct"].ge(0.80)
    work = group.loc[top20]
    if len(work) < 1_000:
        return None
    aggregations: dict[str, str] = {name: "median" for name in MARKET_FEATURES}
    aggregations.update({"risk_target": "mean", "training_rank_pct": "mean"})
    hourly = work.groupby("__ts__", sort=True).agg(aggregations)
    hourly["support"] = work.groupby("__ts__", sort=True).size().astype(np.float32)
    if len(hourly) < 300:
        return None
    values = hourly[MARKET_FEATURES].to_numpy(dtype=np.float32, copy=True)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0)
    values = np.where(np.isfinite(values), values, medians)
    low = np.nanpercentile(values, 0.5, axis=0).astype(np.float32)
    high = np.nanpercentile(values, 99.5, axis=0).astype(np.float32)
    values = np.clip(values, low, high)
    target = hourly["risk_target"].to_numpy(dtype=np.float32)
    weights = np.sqrt(hourly["support"].to_numpy(dtype=np.float32)) * (
        1.0 + hourly["training_rank_pct"].ge(0.90).to_numpy(dtype=np.float32)
    )
    params = {
        "objective": "huber",
        "alpha": 0.90,
        "learning_rate": 0.025,
        "num_leaves": 4 if max_depth <= 2 else 8,
        "max_depth": max_depth,
        "min_data_in_leaf": 80,
        "min_gain_to_split": 0.01,
        "lambda_l1": 0.10,
        "lambda_l2": 8.0,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "feature_fraction": 0.75,
        "seed": seed,
        "num_threads": 2,
        "verbosity": -1,
        "force_col_wise": True,
    }
    dataset = lgb.Dataset(values, label=target, weight=weights, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=180)
    fitted = np.asarray(model.predict(values), dtype=np.float32)
    return _RiskModel(
        model=model,
        medians=medians,
        clip_low=low,
        clip_high=high,
        prediction_mean=float(np.mean(fitted)),
        prediction_std=float(max(np.std(fitted), 1e-3)),
        support_timestamps=int(len(hourly)),
    )


def _fit_state(train: pd.DataFrame, seed: int, max_depth: int) -> _RiskState:
    side_models: dict[str, _RiskModel] = {}
    local_models: dict[tuple[str, str], _RiskModel] = {}
    for side, idx in train.groupby(
        train["side_name"].astype(str).str.lower(), sort=True
    ).groups.items():
        model = _fit_one(train.loc[idx], seed + len(side_models) * 101, max_depth)
        if model is not None:
            side_models[str(side)] = model
    groups = train.groupby(
        [
            train["side_name"].astype(str).str.lower(),
            train["archetype_policy_key"].astype(str),
        ],
        sort=True,
    ).groups
    for (side, arch), idx in groups.items():
        model = _fit_one(train.loc[idx], seed + len(local_models) * 37 + 11, max_depth)
        if model is not None:
            local_models[(str(side), str(arch))] = model
    return _RiskState(
        local_models=local_models,
        side_models=side_models,
        train_end=str(pd.to_datetime(train["__ts__"], utc=True).max()),
    )


def _reconstruct_march_champion(root: Path) -> pd.DataFrame:
    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    residual = pd.read_parquet(
        root
        / "cache"
        / "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline.parquet"
    )
    burnin = _merge_residual_features(burnin, residual)
    state = joblib.load(root / CHAMPION / "residual_overlay_state.joblib")
    safe = burnin.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in burnin.columns
        ],
        errors="ignore",
    )
    burnin["score_champion"] = state.transform(
        safe,
        pd.to_numeric(burnin["score_alternative"], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float32),
    )
    return burnin


def _cdf_rank(prior: pd.DataFrame, current: pd.DataFrame, score_col: str) -> np.ndarray:
    output = np.zeros(len(current), dtype=np.float32)
    side = current["side_name"].astype(str).str.lower().to_numpy()
    prior_side = prior["side_name"].astype(str).str.lower().to_numpy()
    query = current[score_col].to_numpy(dtype=np.float32)
    reference = prior[score_col].to_numpy(dtype=np.float32)
    for side_key in np.unique(side):
        values = np.sort(reference[(prior_side == side_key) & np.isfinite(reference)])
        mask = side == side_key
        if len(values) == 0:
            output[mask] = 0.5
        else:
            output[mask] = np.searchsorted(values, query[mask], side="right") / len(
                values
            )
    return output


def _selected_metrics(frame: pd.DataFrame, rank_col: str) -> dict[str, Any]:
    selected = frame.loc[frame[rank_col].ge(0.90)].copy()
    selected["week"] = (
        pd.to_datetime(selected["__ts__"], utc=True).dt.to_period("W-SUN").dt.start_time
    )
    weekly = selected.groupby("week", sort=True)["ev_after_1pct"].mean()
    return {
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
        ),
        "clean_exec_precision": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
        "full_path_bad_mae_rate": float(
            pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(selected["timeout"], errors="coerce").mean()
        ),
        "worst_week_ev": float(weekly.min()),
        "positive_weeks": int(weekly.gt(0.0).sum()),
        "weeks": int(len(weekly)),
    }


def main() -> None:
    root = DEFAULT_OUT_DIR
    max_depth = max(2, min(3, int(os.environ.get("EPM_META_SHOCK_MAX_DEPTH", "2"))))
    fine_alpha = os.environ.get("EPM_META_SHOCK_FINE_ALPHA", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    alpha_grid = (
        (0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02)
        if fine_alpha
        else ALPHAS
    )
    arm = ARM if max_depth == 2 else f"{ARM}_depth{max_depth}"
    if fine_alpha:
        arm = f"{arm}_finealpha"
    arm_dir = root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    compact_path = root / "cache" / "compact_reference_with_lifecycle.parquet"
    requested = list(
        dict.fromkeys(
            [
                *KEYS,
                "score_meta_base_soft_label",
                "clean_exec",
                "ev_after_1pct",
                *MARKET_FEATURES,
            ]
        )
    )
    data = pd.read_parquet(compact_path, columns=requested)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data.sort_values("__ts__", kind="stable").reset_index(drop=True)
    data["training_rank_pct"] = _month_side_rank(data, "score_meta_base_soft_label")
    score = pd.to_numeric(data["score_meta_base_soft_label"], errors="coerce").fillna(
        0.5
    )
    clean = pd.to_numeric(data["clean_exec"], errors="coerce").fillna(0.0)
    data["risk_target"] = (score - clean).clip(lower=0.0).astype(np.float32)

    champion = pd.read_parquet(
        root
        / f"historical_rank_oos_{CHAMPION}"
        / "oos_predictions_historical_rank.parquet"
    )
    champion["__ts__"] = pd.to_datetime(champion["__ts__"], utc=True, errors="coerce")
    march = _reconstruct_march_champion(root)
    march["__ts__"] = pd.to_datetime(march["__ts__"], utc=True, errors="coerce")
    march = march.rename(columns={"score_champion": "score_champion"})

    scored_months: list[pd.DataFrame] = []
    states: dict[str, _RiskState] = {}
    for fold_idx, month in enumerate(("2026-03", *EVAL_MONTHS)):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data.loc[data["__ts__"].lt(start)]
        valid_features = data.loc[
            data["__ts__"].ge(start) & data["__ts__"].lt(end), KEYS + MARKET_FEATURES
        ]
        state = _fit_state(train, 20260711 + fold_idx * 1009, max_depth)
        risk, risk_z, local = state.predict(valid_features)
        pred = valid_features[KEYS].copy()
        pred["shock_risk_pred"] = risk
        pred["shock_risk_z"] = risk_z
        pred["shock_risk_local_model"] = local
        pred["calendar_month"] = month
        scored_months.append(pred)
        states[month] = state
        print(
            json.dumps(
                {
                    "event": "shock_fold_complete",
                    "month": month,
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(pred)),
                    "side_models": len(state.side_models),
                    "local_models": len(state.local_models),
                }
            ),
            flush=True,
        )

    risk_scores = pd.concat(scored_months, ignore_index=True)
    march = march.merge(
        risk_scores[risk_scores["calendar_month"].eq("2026-03")],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    champion = champion.merge(
        risk_scores[risk_scores["calendar_month"].isin(EVAL_MONTHS)].drop(
            columns="calendar_month"
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    search_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        score_adjusted = np.clip(
            march["score_champion"].to_numpy(dtype=np.float32)
            - np.float32(alpha)
            * march["shock_risk_z"].fillna(0.0).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        )
        probe = march.assign(score_adjusted=score_adjusted)
        rank = _month_side_rank(probe, "score_adjusted")
        selected = probe.loc[rank >= 0.90]
        search_rows.append(
            {
                "alpha": alpha,
                "selected_rows": int(len(selected)),
                "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
                "clean_exec_precision": float(selected["clean_exec"].mean()),
            }
        )
    search = pd.DataFrame(search_rows)
    search["objective"] = (
        search["mean_ev_after_1pct"] + 0.002 * search["clean_exec_precision"]
    )
    best = search.sort_values(
        ["objective", "alpha"], ascending=[False, True], kind="stable"
    ).iloc[0]
    alpha = float(best["alpha"])
    search.to_csv(arm_dir / "march_alpha_search.csv", index=False)

    march["score_adjusted"] = np.clip(
        march["score_champion"].to_numpy(dtype=np.float32)
        - np.float32(alpha)
        * march["shock_risk_z"].fillna(0.0).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    champion["score_adjusted"] = np.clip(
        champion["score_alternative"].to_numpy(dtype=np.float32)
        - np.float32(alpha)
        * champion["shock_risk_z"].fillna(0.0).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    calibrator = _fit_platt(march["score_adjusted"], march["clean_exec"])
    champion["hit_prob_adjusted"] = _calibrate(calibrator, champion["score_adjusted"])
    prior = march[KEYS + ["score_adjusted"]].copy()
    ranked: list[pd.DataFrame] = []
    for month in EVAL_MONTHS:
        current = champion.loc[champion["calendar_month"].eq(month)].copy()
        current["historical_rank_adjusted"] = _cdf_rank(
            prior, current, "score_adjusted"
        )
        ranked.append(current)
        prior = pd.concat(
            [prior, current[KEYS + ["score_adjusted"]]], ignore_index=True
        )
    output = pd.concat(ranked, ignore_index=True)
    output.to_parquet(
        arm_dir / "oos_predictions_historical_rank.parquet",
        index=False,
        compression="zstd",
    )
    joblib.dump(
        {"states": states, "alpha": alpha, "calibrator": calibrator},
        arm_dir / "shock_overlay_state.joblib",
        compress=3,
    )

    baseline = _selected_metrics(output, "historical_rank_alternative")
    challenger = _selected_metrics(output, "historical_rank_adjusted")
    metrics = pd.DataFrame(
        [
            {"selector": CHAMPION, **baseline},
            {"selector": arm, **challenger},
        ]
    )
    metrics.to_csv(arm_dir / "top10_metrics.csv", index=False)
    event = output.loc[
        pd.to_datetime(output["__ts__"], utc=True)
        .dt.floor("D")
        .eq(pd.Timestamp("2026-06-30", tz="UTC"))
        & output["side_name"].eq("long")
        & output["archetype_policy_key"].eq("long_mixed_wideslow_tentative")
    ]
    event_rows: list[dict[str, Any]] = []
    for selector, rank_col in (
        (CHAMPION, "historical_rank_alternative"),
        (arm, "historical_rank_adjusted"),
    ):
        selected = event.loc[event[rank_col].ge(0.90)]
        event_rows.append(
            {
                "selector": selector,
                "selected_rows": int(len(selected)),
                "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
                "clean_exec_precision": float(selected["clean_exec"].mean()),
                "full_path_bad_mae_rate": float(
                    selected["full_path_bad_mae_1r"].mean()
                ),
                "mean_shock_risk_z": float(selected["shock_risk_z"].mean()),
            }
        )
    event_table = pd.DataFrame(event_rows)
    event_table.to_csv(arm_dir / "june30_long_mixed_event.csv", index=False)
    manifest = {
        "schema": "meta_residual_cross_sectional_shock_overlay_v1",
        "arm": arm,
        "parent": CHAMPION,
        "max_depth": max_depth,
        "alpha_grid": list(alpha_grid),
        "market_features": MARKET_FEATURES,
        "selected_alpha": alpha,
        "alpha_selected_on": "2026-03 burn-in only",
        "walkforward_folds": [
            "train<March->March",
            "train<April->April",
            "train<May->May",
            "train<June->June",
        ],
        "baseline": baseline,
        "challenger": challenger,
        "june30": event_rows,
        "current_model_overwritten": False,
        "leakage_contract": (
            "Each shallow side x archetype shock model uses prior rows only. The overlay alpha and hit "
            "calibrator are selected on March; April-June historical ranks use prior adjusted scores only."
        ),
    }
    _write_json(arm_dir / "manifest.json", manifest)
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
