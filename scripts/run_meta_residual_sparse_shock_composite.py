#!/usr/bin/env python3
"""Sparse train-percentile shock composite over the residual-meta champion."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_residual_cross_sectional_shock_overlay import (  # noqa: E402
    CHAMPION,
    KEYS,
    _cdf_rank,
    _reconstruct_march_champion,
    _selected_metrics,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    EVAL_MONTHS,
    _calibrate,
    _fit_platt,
)

ARM = f"{CHAMPION}_sparse_shock_composite"
COMPONENTS = {
    "mkt_median_oi_chg_1h_rz": -1.0,
    "mkt_median_oi_chg_4h_rz": -1.0,
    "mkt_median_oi_drawdown_from_peak_24h": -1.0,
    "mkt_pct_oi_chg_4h_rz_lt_minus2": 1.0,
    "mkt_oi_flush_breadth_accel_1h": 1.0,
    "mkt_systemic_deleveraging_score": 1.0,
    "mkt_pct_price_up_oi_down_1h": 1.0,
}
THRESHOLDS = (0.90, 0.95, 0.975, 0.99, 0.995)
ALPHAS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)
TUNE_MONTHS = tuple(
    str(value) for value in pd.period_range("2025-04", "2026-03", freq="M")
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class ShockCompositeState:
    references: dict[str, np.ndarray]
    archetype_multipliers: dict[str, float]
    train_end: str

    def transform_raw(self, frame: pd.DataFrame) -> np.ndarray:
        percentiles: list[np.ndarray] = []
        for name, direction in COMPONENTS.items():
            query = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                dtype=np.float32
            )
            query = np.float32(direction) * query
            reference = self.references[name]
            finite = np.isfinite(query)
            pct = np.full(len(frame), 0.5, dtype=np.float32)
            pct[finite] = np.searchsorted(reference, query[finite], side="right") / max(
                len(reference), 1
            )
            percentiles.append(pct)
        matrix = np.column_stack(percentiles).astype(np.float32, copy=False)
        flush = matrix[:, :6].mean(axis=1)
        rebound = matrix[:, 6]
        return np.sqrt(np.clip(flush * rebound, 0.0, 1.0)).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        composite = self.transform_raw(frame)
        side = frame["side_name"].astype(str).str.lower().to_numpy()
        arch = frame["archetype_policy_key"].astype(str).to_numpy()
        multiplier = np.fromiter(
            (
                self.archetype_multipliers.get(
                    f"{s}||{a}", self.archetype_multipliers.get(f"{s}||*", 1.0)
                )
                for s, a in zip(side, arch)
            ),
            dtype=np.float32,
            count=len(frame),
        )
        return np.clip(composite * multiplier, 0.0, 1.0).astype(np.float32, copy=False)


def _rank(frame: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(frame["__ts__"], utc=True)
    key = (
        ts.dt.to_period("M").astype(str)
        + "|"
        + frame["side_name"].astype(str).str.lower()
    )
    return (
        pd.to_numeric(frame["score_meta_base_soft_label"], errors="coerce")
        .groupby(key, sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def _fit_state(train: pd.DataFrame) -> ShockCompositeState:
    references: dict[str, np.ndarray] = {}
    timestamp_state = train.groupby("__ts__", sort=True)[list(COMPONENTS)].median()
    for name, direction in COMPONENTS.items():
        values = np.float32(direction) * pd.to_numeric(
            timestamp_state[name], errors="coerce"
        ).to_numpy(dtype=np.float32)
        values = np.sort(values[np.isfinite(values)])
        references[name] = values.astype(np.float32, copy=False)
    provisional = ShockCompositeState(references, {}, str(train["__ts__"].max()))
    composite = provisional.transform_raw(train)
    top20 = train["training_rank_pct"].to_numpy(dtype=np.float32) >= 0.80
    target = train["risk_target"].to_numpy(dtype=np.float32)
    side = train["side_name"].astype(str).str.lower().to_numpy()
    arch = train["archetype_policy_key"].astype(str).to_numpy()
    multipliers: dict[str, float] = {}
    for side_key in np.unique(side):
        parent = top20 & (side == side_key)
        base = float(np.mean(target[parent])) if parent.any() else 0.1
        high = parent & (composite >= 0.95)
        high_mean = float(np.mean(target[high])) if high.any() else base
        support = int(high.sum())
        confidence = support / (support + 500.0)
        ratio = np.clip(high_mean / max(base, 1e-3), 0.5, 2.0)
        multipliers[f"{side_key}||*"] = float(
            np.clip(1.0 + confidence * (ratio - 1.0), 0.5, 2.0)
        )
        for arch_key in np.unique(arch[side == side_key]):
            local = parent & (arch == arch_key)
            local_high = local & (composite >= 0.95)
            local_base = float(np.mean(target[local])) if local.any() else base
            local_high_mean = (
                float(np.mean(target[local_high])) if local_high.any() else local_base
            )
            local_support = int(local_high.sum())
            local_conf = local_support / (local_support + 500.0)
            local_ratio = np.clip(local_high_mean / max(local_base, 1e-3), 0.5, 2.0)
            multipliers[f"{side_key}||{arch_key}"] = float(
                np.clip(1.0 + local_conf * (local_ratio - 1.0), 0.5, 2.0)
            )
    provisional.archetype_multipliers = multipliers
    return provisional


def _risk_intensity(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip((score - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0).astype(
        np.float32
    )


def main() -> None:
    root = DEFAULT_OUT_DIR
    arm_dir = root / ARM
    arm_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        *KEYS,
        "score_meta_base_soft_label",
        "clean_exec",
        "ev_after_1pct",
        *COMPONENTS,
    ]
    data = pd.read_parquet(
        root / "cache" / "compact_reference_with_lifecycle.parquet", columns=columns
    )
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data.sort_values("__ts__", kind="stable").reset_index(drop=True)
    data["training_rank_pct"] = _rank(data)
    raw_score = pd.to_numeric(
        data["score_meta_base_soft_label"], errors="coerce"
    ).fillna(0.5)
    clean = pd.to_numeric(data["clean_exec"], errors="coerce").fillna(0.0)
    data["risk_target"] = (raw_score - clean).clip(lower=0.0).astype(np.float32)

    month_scores: list[pd.DataFrame] = []
    states: dict[str, ShockCompositeState] = {}
    for month in (*TUNE_MONTHS, *EVAL_MONTHS):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data.loc[data["__ts__"].lt(start)]
        valid = data.loc[
            data["__ts__"].ge(start) & data["__ts__"].lt(end), KEYS + list(COMPONENTS)
        ]
        state = _fit_state(train)
        part = valid[KEYS].copy()
        part["shock_composite_raw"] = state.transform_raw(valid)
        part["shock_composite_local"] = state.transform(valid)
        part["calendar_month"] = month
        month_scores.append(part)
        states[month] = state
    shock = pd.concat(month_scores, ignore_index=True)

    tuning = data.loc[
        pd.to_datetime(data["__ts__"], utc=True)
        .dt.to_period("M")
        .astype(str)
        .isin(TUNE_MONTHS),
        KEYS + ["score_meta_base_soft_label", "ev_after_1pct", "clean_exec"],
    ].merge(
        shock[shock["calendar_month"].isin(TUNE_MONTHS)].drop(columns="calendar_month"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    tuning["calendar_month"] = (
        pd.to_datetime(tuning["__ts__"], utc=True).dt.to_period("M").astype(str)
    )

    march = _reconstruct_march_champion(root)
    march["__ts__"] = pd.to_datetime(march["__ts__"], utc=True)
    march = march.merge(
        shock[shock["calendar_month"].eq("2026-03")],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    champion = pd.read_parquet(
        root
        / f"historical_rank_oos_{CHAMPION}"
        / "oos_predictions_historical_rank.parquet"
    )
    champion["__ts__"] = pd.to_datetime(champion["__ts__"], utc=True)
    champion = champion.merge(
        shock[shock["calendar_month"].isin(EVAL_MONTHS)].drop(columns="calendar_month"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    search_rows: list[dict[str, Any]] = []
    best_by_side: dict[str, dict[str, float]] = {}
    for side in ("long", "short"):
        side_frame = tuning.loc[tuning["side_name"].eq(side)].copy()
        for variant in ("raw", "local"):
            feature = f"shock_composite_{variant}"
            for threshold in THRESHOLDS:
                intensity = _risk_intensity(
                    side_frame[feature].fillna(0.0).to_numpy(dtype=np.float32),
                    threshold,
                )
                for alpha in ALPHAS:
                    side_frame["score_probe"] = np.clip(
                        side_frame["score_meta_base_soft_label"].to_numpy(
                            dtype=np.float32
                        )
                        - np.float32(alpha) * intensity,
                        0.0,
                        1.0,
                    )
                    monthly_ev: list[float] = []
                    monthly_clean: list[float] = []
                    selected_rows = 0
                    for _, month_frame in side_frame.groupby(
                        "calendar_month", sort=True
                    ):
                        rank = (
                            month_frame["score_probe"]
                            .rank(method="average", pct=True)
                            .to_numpy(dtype=np.float32)
                        )
                        selected = month_frame.loc[rank >= 0.90]
                        monthly_ev.append(float(selected["ev_after_1pct"].mean()))
                        monthly_clean.append(float(selected["clean_exec"].mean()))
                        selected_rows += len(selected)
                    values = np.asarray(monthly_ev, dtype=np.float64)
                    mean_ev = float(np.nanmean(values))
                    std_ev = float(np.nanstd(values))
                    worst_ev = float(np.nanmin(values))
                    clean_rate = float(np.nanmean(monthly_clean))
                    search_rows.append(
                        {
                            "side_name": side,
                            "variant": variant,
                            "threshold": threshold,
                            "alpha": alpha,
                            "selected_rows": int(selected_rows),
                            "months": int(len(values)),
                            "mean_month_ev": mean_ev,
                            "std_month_ev": std_ev,
                            "worst_month_ev": worst_ev,
                            "clean_exec_precision": clean_rate,
                            "objective": mean_ev
                            - 0.5 * std_ev
                            + 0.25 * worst_ev
                            + 0.002 * clean_rate,
                        }
                    )
        side_search = pd.DataFrame(
            [row for row in search_rows if row["side_name"] == side]
        )
        best = side_search.sort_values(
            ["objective", "alpha", "threshold"],
            ascending=[False, True, False],
            kind="stable",
        ).iloc[0]
        best_by_side[side] = {
            "variant": str(best["variant"]),
            "threshold": float(best["threshold"]),
            "alpha": float(best["alpha"]),
        }
    pd.DataFrame(search_rows).to_csv(
        arm_dir / "historical_side_search.csv", index=False
    )

    for frame, score_col in (
        (march, "score_champion"),
        (champion, "score_alternative"),
    ):
        adjusted = frame[score_col].to_numpy(dtype=np.float32, copy=True)
        for side, params in best_by_side.items():
            mask = frame["side_name"].eq(side).to_numpy()
            intensity = _risk_intensity(
                frame.loc[mask, f"shock_composite_{params['variant']}"]
                .fillna(0.0)
                .to_numpy(dtype=np.float32),
                params["threshold"],
            )
            adjusted[mask] -= np.float32(params["alpha"]) * intensity
        frame["score_adjusted"] = np.clip(adjusted, 0.0, 1.0)
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
        {"states": states, "best_by_side": best_by_side, "calibrator": calibrator},
        arm_dir / "shock_composite_state.joblib",
        compress=3,
    )

    baseline = _selected_metrics(output, "historical_rank_alternative")
    challenger = _selected_metrics(output, "historical_rank_adjusted")
    pd.DataFrame(
        [
            {"selector": CHAMPION, **baseline},
            {"selector": ARM, **challenger},
        ]
    ).to_csv(arm_dir / "top10_metrics.csv", index=False)
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
        (ARM, "historical_rank_adjusted"),
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
                "mean_shock_composite_raw": float(
                    selected["shock_composite_raw"].mean()
                ),
                "mean_shock_composite_local": float(
                    selected["shock_composite_local"].mean()
                ),
            }
        )
    pd.DataFrame(event_rows).to_csv(
        arm_dir / "june30_long_mixed_event.csv", index=False
    )
    manifest = {
        "schema": "meta_residual_sparse_shock_composite_v1",
        "arm": ARM,
        "parent": CHAMPION,
        "components": COMPONENTS,
        "selected_side_parameters": best_by_side,
        "selection_months": list(TUNE_MONTHS),
        "selection_objective": "mean_month_ev - 0.5*std_month_ev + 0.25*worst_month_ev + 0.002*clean_precision",
        "baseline": baseline,
        "challenger": challenger,
        "june30": event_rows,
        "current_model_overwritten": False,
        "leakage_contract": (
            "Percentile references and side x archetype support multipliers use prior rows only. "
            "Thresholds/alphas are selected on monthly walk-forward states through March; calibration "
            "is fitted on March burn-in; April-June ranks use "
            "prior adjusted scores only."
        ),
    }
    (arm_dir / "manifest.json").write_text(
        json.dumps(_safe_json(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_safe_json(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
