#!/usr/bin/env python3
"""OOS event-block and matched-control enrichment for residual semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    KEY_COLUMNS,
    _merge_residual_features,
)

ARM = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"
SEMANTIC_CACHE = "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline.parquet"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    weight = (
        pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    )
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    return (
        float(np.average(value[valid], weights=weight[valid]))
        if valid.any()
        else np.nan
    )


def _quantile_bin(values: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    rank = numeric.rank(method="average", pct=True).fillna(0.5)
    return np.minimum((rank * bins).astype(int), bins - 1).astype(str)


def _matched_delta(
    frame: pd.DataFrame, state_col: str, threshold: float
) -> tuple[float, int]:
    work = frame.copy()
    work["__state_high"] = pd.to_numeric(work[state_col], errors="coerce").ge(threshold)
    work["__rank_bin"] = _quantile_bin(work["historical_rank_current_reference"], 10)
    work["__score_bin"] = _quantile_bin(work["score_current_reference"], 10)
    work["__shock_bin"] = _quantile_bin(work.get("shock_12h"), 5)
    work["__vol_bin"] = _quantile_bin(work.get("rv_rel_universe"), 5)
    strata = [
        "side_name",
        "archetype_policy_key",
        "__rank_bin",
        "__score_bin",
        "__shock_bin",
        "__vol_bin",
    ]
    rows: list[tuple[float, float]] = []
    for _, group in work.groupby(strata, dropna=False, sort=False):
        high = group[group["__state_high"]]
        control = group[~group["__state_high"]]
        if len(high) < 2 or len(control) < 2:
            continue
        rows.append(
            (
                float(high["hit_surprise"].mean() - control["hit_surprise"].mean()),
                float(min(len(high), len(control))),
            )
        )
    if not rows:
        return np.nan, 0
    values = np.asarray([row[0] for row in rows], dtype=np.float64)
    weights = np.asarray([row[1] for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights)), int(weights.sum())


def _block_bootstrap_ci(
    frame: pd.DataFrame,
    state_col: str,
    *,
    draws: int = 1_000,
    seed: int = 20260711,
) -> tuple[float, float]:
    daily = (
        frame.groupby("date", sort=True)
        .apply(
            lambda group: pd.Series(
                {
                    "weighted_sum": float(
                        np.sum(
                            pd.to_numeric(group[state_col], errors="coerce").fillna(0.0)
                            * pd.to_numeric(
                                group["hit_surprise"], errors="coerce"
                            ).fillna(0.0)
                        )
                    ),
                    "weight": float(
                        pd.to_numeric(group[state_col], errors="coerce")
                        .fillna(0.0)
                        .sum()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    daily = daily[daily["weight"].gt(0.0)]
    if len(daily) < 3:
        return np.nan, np.nan
    sums = daily["weighted_sum"].to_numpy(dtype=np.float64)
    weights = daily["weight"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        idx = rng.integers(0, len(daily), size=len(daily))
        estimates[draw] = sums[idx].sum() / max(weights[idx].sum(), 1e-12)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    source_cols = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score_meta_base_soft_label",
        "clean_exec",
        "ev_after_1pct",
        "full_path_bad_mae_1r",
        "timeout",
        "shock_12h",
        "rv_rel_universe",
    ]
    source = pd.read_parquet(
        root / "cache" / "compact_reference_with_lifecycle.parquet",
        columns=source_cols,
    )
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source = source[
        source["__ts__"].ge(pd.Timestamp("2026-04-01", tz="UTC"))
        & source["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))
    ].copy()
    semantic = pd.read_parquet(root / "cache" / SEMANTIC_CACHE)
    semantic["__ts__"] = pd.to_datetime(semantic["__ts__"], utc=True, errors="coerce")
    data = _merge_residual_features(source, semantic)
    ranked = pd.read_parquet(
        root / f"historical_rank_oos_{ARM}" / "oos_predictions_historical_rank.parquet",
        columns=[
            *KEY_COLUMNS,
            "historical_rank_current_reference",
            "score_current_reference",
            "hit_prob_current_reference",
            "calendar_month",
        ],
    )
    ranked["__ts__"] = pd.to_datetime(ranked["__ts__"], utc=True, errors="coerce")
    keys = [
        name for name in KEY_COLUMNS if name in data.columns and name in ranked.columns
    ]
    data = data.merge(ranked, on=keys, how="inner", validate="one_to_one")
    data["date"] = data["__ts__"].dt.floor("D")
    data["hit_surprise"] = pd.to_numeric(
        data["clean_exec"], errors="coerce"
    ) - pd.to_numeric(data["hit_prob_current_reference"], errors="coerce")
    data["negative_surprise"] = (-data["hit_surprise"]).clip(lower=0.0)
    data["positive_surprise"] = data["hit_surprise"].clip(lower=0.0)
    grouped = data.groupby(["side_name", "archetype_policy_key"], dropna=False)
    data["negative_tail"] = data["hit_surprise"].le(
        grouped["hit_surprise"].transform(lambda values: values.quantile(0.10))
    )
    data["positive_tail"] = data["hit_surprise"].ge(
        grouped["hit_surprise"].transform(lambda values: values.quantile(0.90))
    )
    semantic_cols = [
        name for name in data.columns if name.startswith("meta_resid_arch_prob__")
    ]
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for semantic_idx, state_col in enumerate(semantic_cols):
        semantic_name = state_col.removeprefix("meta_resid_arch_prob__")
        weights = (
            pd.to_numeric(data[state_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
        support = float(weights.sum())
        if support <= 0.0:
            continue
        negative_rate = _weighted_mean(data["negative_tail"].astype(float), weights)
        positive_rate = _weighted_mean(data["positive_tail"].astype(float), weights)
        global_negative = float(data["negative_tail"].mean())
        global_positive = float(data["positive_tail"].mean())
        daily_mass = data.assign(__weight=weights).groupby("date")["__weight"].sum()
        tail_mass_negative = (
            data.assign(__weight=weights * data["negative_tail"].astype(float))
            .groupby("date")["__weight"]
            .sum()
        )
        tail_mass_positive = (
            data.assign(__weight=weights * data["positive_tail"].astype(float))
            .groupby("date")["__weight"]
            .sum()
        )
        dominant_negative = negative_rate >= positive_rate
        tail_mass = tail_mass_negative if dominant_negative else tail_mass_positive
        nonzero_events = tail_mass[tail_mass.gt(0.25)]
        largest_event_share = float(
            nonzero_events.max() / max(nonzero_events.sum(), 1e-12)
        )
        policy_mass = (
            data.assign(__weight=weights)
            .groupby(["side_name", "archetype_policy_key"], dropna=False)["__weight"]
            .sum()
        )
        largest_policy_share = float(policy_mass.max() / max(policy_mass.sum(), 1e-12))
        ci_low, ci_high = _block_bootstrap_ci(
            data,
            state_col,
            seed=20260711 + semantic_idx * 101,
        )
        threshold = float(weights.quantile(0.90))
        matched_delta, matched_rows = _matched_delta(data, state_col, threshold)
        month_effects = (
            data.assign(__weight=weights)
            .groupby("calendar_month")
            .apply(
                lambda group: _weighted_mean(group["hit_surprise"], group["__weight"]),
                include_groups=False,
            )
        )
        mean_surprise = _weighted_mean(data["hit_surprise"], weights)
        expected_sign = int(np.sign(mean_surprise))
        recurrence = (
            float((np.sign(month_effects) == expected_sign).mean())
            if expected_sign
            else 0.0
        )
        direction_ci_pass = bool(ci_low > 0.0 or ci_high < 0.0)
        matched_sign_pass = (
            bool(np.sign(matched_delta) == expected_sign)
            if np.isfinite(matched_delta)
            else False
        )
        enrichment = max(
            negative_rate / max(global_negative, 1e-12),
            positive_rate / max(global_positive, 1e-12),
        )
        passes = bool(
            enrichment >= 1.5
            and len(nonzero_events) >= 3
            and largest_event_share <= 0.50
            and recurrence >= (2.0 / 3.0)
            and direction_ci_pass
            and matched_sign_pass
        )
        rows.append(
            {
                "semantic": semantic_name,
                "posterior_weighted_rows": support,
                "mean_signed_hit_surprise": mean_surprise,
                "mean_negative_surprise": _weighted_mean(
                    data["negative_surprise"], weights
                ),
                "mean_positive_surprise": _weighted_mean(
                    data["positive_surprise"], weights
                ),
                "mean_ev_after_1pct": _weighted_mean(data["ev_after_1pct"], weights),
                "negative_tail_rate": negative_rate,
                "positive_tail_rate": positive_rate,
                "negative_enrichment_ratio": negative_rate
                / max(global_negative, 1e-12),
                "positive_enrichment_ratio": positive_rate
                / max(global_positive, 1e-12),
                "dominant_tail": "negative" if dominant_negative else "positive",
                "distinct_event_days": int(len(nonzero_events)),
                "largest_event_share": largest_event_share,
                "largest_policy_archetype_share": largest_policy_share,
                "block_bootstrap_ci025": ci_low,
                "block_bootstrap_ci975": ci_high,
                "fold_direction_recurrence": recurrence,
                "matched_high_state_surprise_delta": matched_delta,
                "matched_rows": matched_rows,
                "enrichment_pass": passes,
            }
        )
        for date, mass in nonzero_events.items():
            event_rows.append(
                {
                    "semantic": semantic_name,
                    "date": date,
                    "dominant_tail": "negative" if dominant_negative else "positive",
                    "posterior_tail_mass": float(mass),
                    "all_state_mass": float(daily_mass.get(date, 0.0)),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["enrichment_pass", "posterior_weighted_rows"], ascending=[False, False]
    )
    events = pd.DataFrame(event_rows)
    result.to_csv(report_dir / "stage8_semantic_enrichment.csv", index=False)
    events.to_csv(report_dir / "stage8_semantic_event_support.csv", index=False)
    manifest = {
        "schema": "meta_residual_semantic_enrichment_v1",
        "oos_rows": int(len(data)),
        "semantics_evaluated": int(len(result)),
        "semantics_passing": int(result["enrichment_pass"].sum()) if len(result) else 0,
        "passed_semantics": result.loc[result["enrichment_pass"], "semantic"].tolist(),
        "matched_controls": "side x base archetype x rank decile x score decile x shock quintile x volatility quintile",
        "bootstrap_unit": "UTC day",
        "leakage_contract": "All semantic probabilities are monthly train-only OOS transforms; realized outcomes are used only for this report.",
    }
    (report_dir / "stage8_semantic_enrichment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
