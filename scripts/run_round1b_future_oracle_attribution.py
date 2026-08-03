#!/usr/bin/env python3
"""Round-1B D: exact-H12 future-information family attribution.

This intentionally non-promotable diagnostic answers *which resolved-path
mechanisms carry the future-oracle advantage*.  It is not an execution model:
every inference field is post-entry, and models are trained only on rows whose
labels were available before the held-out meta-OOS period.

The evaluation book is one pooled global ranking over exactly the Round-1B
75,200 candidate IDs (August--November 2024).  No threshold, side quota,
portfolio rule, or execution-policy change is applied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as error:  # pragma: no cover
    raise RuntimeError("lightgbm is required for future-oracle attribution") from error


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"
PACK = ART / "root_cause_exact_h12_execution_target_pack_20260801_v5"
ROUND = ART / "sequential_funnel_round1_g0_tau025_20260801_v6"
DEFAULT_OUTPUT = ART / "sequential_funnel_round1b_future_oracle_attribution_20260801_v1"
TEST_START = pd.Timestamp("2024-08-01", tz="UTC")
TEST_END = pd.Timestamp("2024-12-01", tz="UTC")
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
SEED = 20260801

# Families are mutually exclusive by construction.  Do not add direct gross,
# net, endpoint-return, or conditional-exact-PnL fields: those would turn an
# intentional future-path diagnostic into an outcome-label shortcut.
FAMILIES: dict[str, tuple[str, ...]] = {
    "reachability": (
        "__meaningful_mfe_reached_12h__", "__mfe_ge_0_5atr__", "__mfe_ge_1atr__",
        "__mfe_ge_1_5atr__", "__mfe_ge_2atr__", "__mfe_ge_3atr__", "__mfe_ge_4atr__",
        "competing_risk_event", "competing_risk_class", "adverse_first", "timeout",
        "clean_economic_favorable_first",
    ),
    "mfe_magnitude": (
        "__peak_mfe_atr_12h__", "__peak_mfe_atr_clip_6__", "__peak_mfe_atr_clip_8__",
        "__log1p_peak_mfe_12h_atr__", "__mfe_integral_atr_hours_12h__",
        "__favorable_path_integral_atr__", "__peak_mfe_return_12h__",
    ),
    "adverse_mae": (
        "__mae_before_meaningful_mfe_atr_12h__", "__mae_until_horizon_if_no_1_5atr__",
        "__pre_mfe_mae_event__", "__pre_mfe_underwater_bars__", "__pre_mfe_underwater_fraction__",
        "__adverse_trough_atr__", "__adverse_trough_bar__", "__adverse_trough_recovery_fraction__",
        "__adverse_trough_recovered_50pct__", "__adverse_trough_recovered_80pct__",
        "__adverse_trough_recovered_100pct__", "__bars_from_adverse_trough_to_full_recovery__",
        "__time_from_adverse_trough_to_full_recovery_hours__", "__hits_minus_1_0atr_before_plus_1_5atr__",
        "__hits_minus_0_5atr_before_plus_1_5atr__", "__bars_below_entry_before_1_5atr__",
        "__fraction_bars_below_entry_before_1_5atr__", "__trough_before_1_5atr_mfe__",
        "__meaningful_mfe_before_mae_0_25atr__", "__meaningful_mfe_before_mae_0_5atr__",
        "__meaningful_mfe_before_mae_0_75atr__", "__meaningful_mfe_before_mae_1atr__",
        "__meaningful_mfe_before_mae_1_5atr__",
    ),
    "timing": (
        "__time_to_first_meaningful_mfe_hours_12h__", "__time_to_peak_mfe_hours_12h__",
        "__time_to_50pct_peak_mfe_hours_12h__", "__time_to_80pct_peak_mfe_hours_12h__",
        "__bars_to_peak_mfe__", "__bars_to_50pct_peak_mfe__", "__bars_to_80pct_peak_mfe__",
        "__bars_to_1atr__", "__bars_to_1_5atr__", "__bars_to_2atr__",
        "first_favorable_minute", "first_adverse_minute", "first_event_minute",
    ),
    "persistence_giveback": (
        "__peak_mfe_bars_above_50pct_12h__", "__peak_mfe_fraction_above_50pct_12h__",
        "__peak_mfe_bars_above_80pct_12h__", "__peak_mfe_fraction_above_80pct_12h__",
        "__mfe_ratio_to_peak_at_2h_12h__", "__mfe_ratio_to_peak_at_4h_12h__",
        "__mfe_ratio_to_peak_at_8h_12h__", "__mfe_2h_over_mfe_12h__",
        "__mfe_4h_over_mfe_12h__", "__mfe_8h_over_mfe_12h__",
        "__mfe_persistence_path_efficiency_12h__", "__peak_mfe_within_1h_12h__",
        "__peak_mfe_within_2h_12h__", "__peak_mfe_within_4h_12h__", "__peak_mfe_within_8h_12h__",
    ),
    "future_path_quality": (
        "__mfe_mae_path_efficiency_12h__", "__mfe_integral_path_efficiency_12h__",
        "__mfe_timing_path_efficiency_12h__", "__path_efficiency_12h__",
        "__path_efficiency_to_1_5atr__", "__path_efficiency_to_2atr__",
        "__path_efficiency_to_80pct_peak__", "__path_efficiency_to_90pct_peak__",
        "__path_efficiency_to_first_meaningful_mfe__", "__future_slope_2h_atr_per_hour__",
        "__future_slope_4h_atr_per_hour__", "__future_slope_8h_atr_per_hour__",
        "__future_slope_12h_atr_per_hour__",
    ),
}
FORBIDDEN_TOKENS = ("net", "gross", "endpoint_signed", "conditional_exact", "execution_exact")


def _stable_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(values.astype(str).unique())).encode()).hexdigest()


def _check_feature_contract(frame: pd.DataFrame) -> dict[str, list[str]]:
    missing = {name: sorted(set(cols).difference(frame.columns)) for name, cols in FAMILIES.items()}
    missing = {name: cols for name, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"supportive-label pack missing declared attribution fields: {missing}")
    all_features = [column for cols in FAMILIES.values() for column in cols]
    if len(all_features) != len(set(all_features)):
        raise ValueError("attribution feature families must be disjoint")
    bad = [column for column in all_features if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    if bad:
        raise ValueError(f"declared future family includes forbidden direct-outcome field(s): {bad}")
    return {name: list(columns) for name, columns in FAMILIES.items()}


def _matrix(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numerically encode features using only categories seen during training."""
    x_train = pd.DataFrame(index=train.index)
    x_test = pd.DataFrame(index=test.index)
    for column in features:
        if pd.api.types.is_numeric_dtype(train[column]):
            x_train[column] = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            x_test[column] = pd.to_numeric(test[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            train_values = train[column].astype("string").fillna("__MISSING__")
            categories = pd.Index(train_values.unique())
            lookup = {value: code for code, value in enumerate(categories)}
            x_train[column] = train_values.map(lookup).astype(float)
            x_test[column] = test[column].astype("string").fillna("__MISSING__").map(lookup).fillna(-1.0).astype(float)
    return x_train, x_test


def _model() -> Any:
    return lgb.LGBMRegressor(
        objective="huber", alpha=0.90, n_estimators=320, learning_rate=0.035,
        num_leaves=31, max_depth=6, min_child_samples=140, colsample_bytree=0.90,
        subsample=0.90, reg_lambda=8.0, reg_alpha=0.10, random_state=SEED,
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _pooled_metrics(frame: pd.DataFrame, score: str, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        n = int(np.ceil(len(frame) * fraction))
        selected = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort").head(n)
        rows.append({
            "arm": arm, "top_fraction": fraction, "population_rows": len(frame), "selected_rows": len(selected),
            "gross_bps_per_trade": float(selected["gross_bps"].mean()),
            "net_bps_per_trade": float(selected["net_bps"].mean()),
            "cost_bps_per_trade": float((selected["gross_bps"] - selected["net_bps"]).mean()),
        })
    return rows


def _top10_attribution(frame: pd.DataFrame, score: str, arm: str) -> list[dict[str, Any]]:
    n = int(np.ceil(len(frame) * 0.10))
    selected = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort").head(n).copy()
    selected["month"] = pd.to_datetime(selected["decision_ts"], utc=True).dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for slice_kind, cols in (("side", ["side"]), ("month", ["month"])):
        for key, part in selected.groupby(cols, observed=True):
            value = key if isinstance(key, str) else key[0]
            rows.append({
                "arm": arm, "top_fraction": 0.10, "slice_kind": slice_kind, "slice_value": str(value),
                "selected_rows": len(part), "gross_bps_per_trade": float(part["gross_bps"].mean()),
                "net_bps_per_trade": float(part["net_bps"].mean()),
            })
    return rows


def _fit_arm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], arm: str) -> tuple[pd.Series, list[dict[str, Any]]]:
    prediction = pd.Series(index=test.index, dtype=float)
    lineage: list[dict[str, Any]] = []
    for side in ("long", "short"):
        fit = train.loc[train["side"].eq(side)].copy()
        evaluate = test.loc[test["side"].eq(side)].copy()
        if fit.empty or evaluate.empty:
            raise ValueError(f"{arm}/{side}: empty train or test partition")
        x_fit, x_eval = _matrix(fit, evaluate, features)
        estimator = _model()
        estimator.fit(x_fit, fit["gross_bps"].to_numpy(float))
        prediction.loc[evaluate.index] = estimator.predict(x_eval)
        lineage.append({
            "arm": arm, "side": side, "features": len(features), "feature_names": features,
            "train_rows": len(fit), "test_rows": len(evaluate),
            "train_max_label_available_ts": str(pd.to_datetime(fit["label_available_ts"], utc=True).max()),
            "test_start": str(TEST_START), "target": "execution_exact_h12_gross_bps",
            "model": "LightGBM Huber; fixed non-HPO diagnostic parameters", "seed": SEED,
        })
    if prediction.isna().any():
        raise ValueError(f"{arm}: prediction coverage incomplete")
    return prediction, lineage


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    primary = pd.read_parquet(PACK / "primary_labels.parquet")
    support = pd.read_parquet(PACK / "supportive_labels.parquet")
    fields = _check_feature_contract(support)
    primary = primary.rename(columns={
        "execution_exact_h12_gross_bps": "gross_bps", "execution_exact_h12_net_bps": "net_bps",
    })
    keep = ["candidate_id", "symbol", "side", "decision_ts", "label_available_ts", "gross_bps", "net_bps"]
    data = primary.loc[:, keep].merge(support.loc[:, ["candidate_id", *[c for xs in fields.values() for c in xs]]], on="candidate_id", how="inner", validate="one_to_one")
    if len(data) != len(primary) or data["candidate_id"].nunique() != len(data):
        raise ValueError("primary/supportive exact-H12 identity join is incomplete")

    round_predictions = pd.read_parquet(ROUND / "base_meta_stack_predictions.parquet")
    round_reference = round_predictions.loc[
        round_predictions["target_arm"].eq("T1_exact_net_huber") & round_predictions["model_variant"].eq("base_plus_meta"),
        ["candidate_id", "__ts__", "__decision_ts__"],
    ].copy()
    round_ids = round_reference["candidate_id"]
    if round_ids.nunique() != 75_200:
        raise ValueError("unexpected Round-1B population")
    test = data.loc[data["candidate_id"].isin(set(round_ids))].copy()
    train = data.loc[pd.to_datetime(data["label_available_ts"], utc=True).lt(TEST_START)].copy()
    test_decision = pd.to_datetime(test["decision_ts"], utc=True)
    if len(test) != len(round_ids) or set(test["candidate_id"]) != set(round_ids):
        raise ValueError("test candidate IDs are not exactly the Round-1B universe")
    # The source fold is defined on raw feature cutoffs.  Decision/entry are
    # after candle close and can therefore include exactly 2024-12-01 00:00.
    # Verify this convention against the frozen Round-1 artifact rather than
    # silently dropping that last decision timestamp.
    round_cutoff = pd.to_datetime(round_reference["__ts__"], utc=True)
    if not round_cutoff.ge(TEST_START).all() or not round_cutoff.lt(TEST_END).all():
        raise ValueError("Round-1B raw feature cutoffs outside sealed meta-OOS period")
    expected_decisions = round_reference.set_index("candidate_id")["__decision_ts__"]
    if not test_decision.eq(pd.to_datetime(test["candidate_id"].map(expected_decisions), utc=True)).all():
        raise ValueError("primary target decision timestamps do not match Round-1 candidate contract")
    if not pd.to_datetime(train["label_available_ts"], utc=True).lt(TEST_START).all():
        raise ValueError("training labels violate meta-OOS boundary")

    all_features = [column for columns in fields.values() for column in columns]
    arm_features: dict[str, list[str]] = {"full_future_oracle": all_features}
    for family, columns in fields.items():
        arm_features[f"family_only__{family}"] = columns
        arm_features[f"leave_one_out__{family}"] = [column for column in all_features if column not in set(columns)]

    scored = test.loc[:, ["candidate_id", "symbol", "side", "decision_ts", "gross_bps", "net_bps"]].copy()
    metrics: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for arm, feature_names in arm_features.items():
        score, arm_lineage = _fit_arm(train, test, feature_names, arm)
        column = f"score__{arm}"
        scored[column] = score.to_numpy(float)
        metrics.extend(_pooled_metrics(scored, column, arm))
        attribution.extend(_top10_attribution(scored, column, arm))
        lineage.extend(arm_lineage)

    metric_frame = pd.DataFrame(metrics)
    full = metric_frame.loc[metric_frame["arm"].eq("full_future_oracle")].set_index("top_fraction")
    delta_rows: list[dict[str, Any]] = []
    for family in fields:
        for fraction in TOP_FRACTIONS:
            family_only = metric_frame.loc[(metric_frame["arm"].eq(f"family_only__{family}")) & metric_frame["top_fraction"].eq(fraction)].iloc[0]
            leave_out = metric_frame.loc[(metric_frame["arm"].eq(f"leave_one_out__{family}")) & metric_frame["top_fraction"].eq(fraction)].iloc[0]
            reference = full.loc[fraction]
            delta_rows.append({
                "family": family, "top_fraction": fraction,
                "family_only_gross_bps": float(family_only.gross_bps_per_trade),
                "family_only_net_bps": float(family_only.net_bps_per_trade),
                "family_only_minus_full_gross_bps": float(family_only.gross_bps_per_trade - reference.gross_bps_per_trade),
                "family_only_minus_full_net_bps": float(family_only.net_bps_per_trade - reference.net_bps_per_trade),
                "lofo_gross_bps": float(leave_out.gross_bps_per_trade),
                "lofo_net_bps": float(leave_out.net_bps_per_trade),
                "full_minus_lofo_gross_bps": float(reference.gross_bps_per_trade - leave_out.gross_bps_per_trade),
                "full_minus_lofo_net_bps": float(reference.net_bps_per_trade - leave_out.net_bps_per_trade),
            })

    output.mkdir(parents=True)
    scored.to_parquet(output / "oracle_attribution_predictions.parquet", index=False, compression="zstd")
    metric_frame.to_parquet(output / "pooled_global_metrics.parquet", index=False, compression="zstd")
    metric_frame.to_csv(output / "pooled_global_metrics.csv", index=False)
    pd.DataFrame(attribution).to_parquet(output / "top10_side_month_attribution.parquet", index=False, compression="zstd")
    pd.DataFrame(delta_rows).to_parquet(output / "family_ablation_deltas.parquet", index=False, compression="zstd")
    pd.DataFrame(delta_rows).to_csv(output / "family_ablation_deltas.csv", index=False)
    (output / "model_lineage.json").write_text(json.dumps(lineage, indent=2, default=str) + "\n")
    contract = {
        "status": "COMPLETED_HINDSIGHT_DIAGNOSTIC_ONLY_NOT_PROMOTION_ELIGIBLE",
        "candidate_count": len(test), "candidate_id_sha256_sorted": _stable_hash(test["candidate_id"]),
        "train_rows": len(train), "train_label_boundary": "label_available_ts < 2024-08-01T00:00:00Z",
        "test_period": "2024-08-01T00:00:00Z <= raw feature cutoff < 2024-12-01T00:00:00Z; decision timestamp follows source candle close",
        "target": "exact H12 execution-adjusted gross bps", "selection": "pooled global top-k with candidate_id ascending ties",
        "families": fields,
        "future_market_confirmation": "UNAVAILABLE: no exact-ID aligned distinct future market-confirmation materialisation in v5 support pack",
        "leakage_guard": "direct execution gross/net, endpoint signed return and conditional exact-PnL fields rejected from inputs; remaining inputs are intentionally future-resolved path labels only",
        "lineage": "per-side LightGBM Huber, fixed diagnostic parameters, trained only pre-meta-OOS",
    }
    (output / "run_manifest.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


if __name__ == "__main__":
    run()
