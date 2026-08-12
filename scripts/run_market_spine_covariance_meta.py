#!/usr/bin/env python3
"""Frozen market-spine covariance residual-ranking ablation.

This runner is intentionally a diagnostic.  It constructs one robust hourly
market spine from declared decision-time fields, freezes the cluster artifact
before each fold's calibration month, and compares the existing causal context
against the raw spine, its factors, and its covariance-break features.  Every
meta arm receives the direct strict-OOF R3 base outputs.  A base-only control
is evaluated on exactly the same candidate rows and global-tail contract.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.market_spine_cluster_covariance import (  # noqa: E402
    MarketSpineClusterCovarianceConfig,
    aggregate_hourly_market_spine,
    fit_market_spine_cluster_model,
    transform_market_spine_cluster_covariance,
)


SCHEMA = "market_spine_covariance_meta_v2_base_control"
INPUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/market_spine_covariance_meta_20260805_v2_base_control"
LABEL_DELAY = pd.Timedelta(hours=13)
TAILS = (0.01, 0.05, 0.10)
SEED = 20260805

# Exact target-free source list: 14 context mechanisms plus the sealed,
# causal soft-state surface.  The R3 simplex is a legal base-prediction input,
# never a target-derived feature.
CONTEXT_FIELDS = (
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)
SOFT_STATE_FIELDS = (
    "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition",
    "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
)
SOURCE_SPINE_FIELDS = (*CONTEXT_FIELDS, *SOFT_STATE_FIELDS)
EXISTING_CAUSAL_META_FIELDS = (
    *(f"regime_z__{field}" for field in CONTEXT_FIELDS), *SOFT_STATE_FIELDS,
)
BASE_OUTPUT_FIELDS = (
    "p_clear", "p_adverse", "p_weak", "prequential_base_expected_net_bps",
    "base_score_clear_minus_half_adverse",
)
FORBIDDEN_MODEL_FIELDS = (
    "base_raw",
    "soft_regime_prior_residual", "target__", "residual", "regime_relative",
)


@dataclass(frozen=True)
class ContinuousFold:
    name: str
    family: str
    train_start: str
    calibration_start: str
    test_start: str
    test_end: str

    @property
    def calibration_end(self) -> pd.Timestamp:
        return _utc(self.test_start) - pd.Timedelta(hours=1)


# Calibration is a complete calendar month immediately before the test.  Each
# cluster artifact is fit no later than this explicit calibration end.
FOLDS = (
    ContinuousFold("primary_2023_09_10", "primary", "2023-07-01", "2023-08-01", "2023-09-01", "2023-11-01"),
    ContinuousFold("primary_2023_11_12", "primary", "2023-07-01", "2023-10-01", "2023-11-01", "2024-01-01"),
    ContinuousFold("primary_2024_01_02", "primary", "2023-07-01", "2023-12-01", "2024-01-01", "2024-03-01"),
    ContinuousFold("transport_2024_07_08", "transport", "2024-05-01", "2024-06-01", "2024-07-01", "2024-09-01"),
    ContinuousFold("transport_2024_09_10", "transport", "2024-05-01", "2024-08-01", "2024-09-01", "2024-11-01"),
    ContinuousFold("transport_2024_11_partial", "transport", "2024-05-01", "2024-10-01", "2024-11-01", "2024-12-01"),
)

# Same out-of-sample months as ``FOLDS``, but the later transport folds use
# every preceding compatible TP6/SL4 resolved month.  The March--April gap is
# retained as an explicit elapsed-time gap in the hourly spine; it is never
# silently compressed into adjacent observations.
LONG_HISTORY_FOLDS = (
    *FOLDS[:3],
    ContinuousFold("transport_long_2024_07_08", "transport_long", "2023-07-01", "2024-06-01", "2024-07-01", "2024-09-01"),
    ContinuousFold("transport_long_2024_09_10", "transport_long", "2023-07-01", "2024-08-01", "2024-09-01", "2024-11-01"),
    ContinuousFold("transport_long_2024_11_partial", "transport_long", "2023-07-01", "2024-10-01", "2024-11-01", "2024-12-01"),
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") if not isinstance(value, pd.Timestamp) or value.tzinfo is None else value.tz_convert("UTC")


def residual_grade(residual_bps: Sequence[float]) -> np.ndarray:
    """The declared fixed five-level ordinal residual target."""
    value = np.asarray(residual_bps, dtype=float)
    return np.select(
        (value <= -150.0, value <= -50.0, value <= 50.0, value <= 150.0),
        (0, 1, 2, 3), default=4,
    ).astype(np.int32)


def _required_columns() -> list[str]:
    return [
        "candidate_id", "__ts__", "side_name", "net_bps", "gross_bps",
        "shared_regime_contract_complete", *BASE_OUTPUT_FIELDS[:-1],
        "state_reference_cutoff_utc", *SOURCE_SPINE_FIELDS, *EXISTING_CAUSAL_META_FIELDS,
    ]


def load_ledger(path: Path = INPUT) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(_required_columns()))).copy()
    missing = set(_required_columns()).difference(frame.columns)
    if missing:
        raise ValueError(f"ledger is missing declared source/meta fields: {sorted(missing)}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate identities must be unique")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["label_available_ts"] = frame["__ts__"] + LABEL_DELAY
    frame["state_reference_cutoff_utc"] = pd.to_datetime(frame["state_reference_cutoff_utc"], utc=True, errors="raise")
    if (frame["state_reference_cutoff_utc"] > frame["__ts__"]).any():
        raise ValueError("soft-state reference cutoff looks ahead of a candidate")
    frame = frame.loc[frame["shared_regime_contract_complete"].fillna(False).astype(bool)].copy()
    numeric = list(dict.fromkeys([
        "net_bps", "gross_bps", *BASE_OUTPUT_FIELDS[:-1],
        *SOURCE_SPINE_FIELDS, *EXISTING_CAUSAL_META_FIELDS,
    ]))
    frame.loc[:, numeric] = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    frame["base_score_clear_minus_half_adverse"] = (
        frame["p_clear"] - .5 * frame["p_adverse"]
    )
    complete = np.isfinite(frame.loc[:, numeric].to_numpy(float)).all(axis=1)
    frame = frame.loc[complete].copy()
    if not np.allclose(frame["gross_bps"] - frame["net_bps"], 100.0, atol=.02):
        raise ValueError("fixed 100-bps cost contract failed")
    frame["realized_residual_bps"] = frame["net_bps"] - frame["prequential_base_expected_net_bps"]
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _assert_feature_contract(fields: Iterable[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(map(str, fields)))
    forbidden = [field for field in result if any(token in field.lower() for token in FORBIDDEN_MODEL_FIELDS)]
    if forbidden:
        raise ValueError(f"outcome/base/residual/simplex field entered a model arm: {forbidden}")
    return result


def _build_fold_feature_panel(data: pd.DataFrame, fold: ContinuousFold) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build per-fold causal features, fitting memberships through calibration."""
    calibration_end = fold.calibration_end
    # The later transport is intentionally independent: neither its cluster
    # memberships nor rolling state may bridge the March--April data gap.
    candidate_history = data.loc[
        data["__ts__"].between(_utc(fold.train_start), _utc(fold.test_end), inclusive="left")
    ].copy()
    spine = aggregate_hourly_market_spine(candidate_history, SOURCE_SPINE_FIELDS, timestamp_col="__ts__")
    # Materialise absent hours explicitly.  Rolling windows are elapsed-hour
    # contracts, not observations-in-a-sparse-index contracts; this prevents
    # an unavailable interval from being silently adjacent to the next era.
    spine = spine.reindex(pd.date_range(spine.index.min(), spine.index.max(), freq="h", tz="UTC"))
    spine.index.name = "timestamp"
    all_median_columns = tuple(f"mspine__{field}__median" for field in SOURCE_SPINE_FIELDS)
    training_spine = spine.loc[:calibration_end, list(all_median_columns)]
    # Constant state fields have no causal innovation/covariance semantics.
    # Retain them in the raw-spine control, but exclude them from the frozen
    # cluster representation and disclose the fold-local exclusion.
    median_columns = tuple(
        field for field in all_median_columns
        if training_spine[field].dropna().nunique() > 1
    )
    excluded_constant_columns = sorted(set(all_median_columns).difference(median_columns))
    if len(median_columns) < 2:
        raise ValueError(f"insufficient varying spine fields for {fold.name}")
    config = MarketSpineClusterCovarianceConfig()
    model = fit_market_spine_cluster_model(spine, calibration_end, config, cluster_columns=median_columns)
    if model.training_end > calibration_end:
        raise ValueError("cluster membership was fit after the declared calibration end")
    transformed = transform_market_spine_cluster_covariance(spine, model)
    # The original source fields already exist on each candidate row.  Join
    # only genuinely new factor/break outputs; joining their identical hourly
    # medians would create ambiguous duplicate state columns.
    hourly = pd.concat([transformed.factors, transformed.features], axis=1)
    panel = candidate_history.merge(hourly, left_on=candidate_history["__ts__"].dt.floor("h"), right_index=True, how="left", validate="many_to_one")
    # merge with a Series key creates an implementation-named key only on old
    # pandas; retain just declared columns below so it cannot become a feature.
    factor_fields = tuple(transformed.factors.columns)
    break_fields = tuple(transformed.features.columns)
    arms = {
        "M0_base_plus_existing_causal_meta": _assert_feature_contract((*BASE_OUTPUT_FIELDS, *EXISTING_CAUSAL_META_FIELDS)),
        "M1_base_plus_raw_spine_sources": _assert_feature_contract((*BASE_OUTPUT_FIELDS, *SOURCE_SPINE_FIELDS)),
        "M2_base_plus_spine_factors": _assert_feature_contract((*BASE_OUTPUT_FIELDS, *SOURCE_SPINE_FIELDS, *factor_fields)),
        "M3_base_plus_spine_factor_breaks": _assert_feature_contract((*BASE_OUTPUT_FIELDS, *SOURCE_SPINE_FIELDS, *factor_fields, *break_fields)),
    }
    audit = {
        "cluster_training_end": model.training_end.isoformat(),
        "calibration_end": calibration_end.isoformat(),
        "excluded_constant_spine_columns": excluded_constant_columns,
        "memberships": {name: list(members) for name, members in model.memberships.items()},
        "source_spine_fields": list(SOURCE_SPINE_FIELDS),
        "factor_fields": list(factor_fields),
        "break_fields": list(break_fields),
    }
    return panel, arms, audit


def _ranker_arrays(frame: pd.DataFrame, fields: Sequence[str], label: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = frame.loc[:, ["__ts__", *fields]].copy()
    work["__query__"] = work["__ts__"].dt.floor("h")
    # LambdaRank is meaningful only for contemporaneous choice sets.  Singleton
    # hours contain no pairwise information and are excluded from fitting.
    work["__row__"] = np.arange(len(work))
    work = work.loc[work.groupby("__query__", observed=True)["__query__"].transform("size").ge(2)]
    work = work.sort_values(["__query__", "__row__"], kind="stable")
    order = work["__row__"].to_numpy(dtype=int)
    groups = work.groupby("__query__", sort=False, observed=True).size().to_numpy(dtype=np.int32)
    if len(groups) < 2:
        raise ValueError("insufficient timestamp groups for LambdaRank")
    return work.loc[:, fields].replace([np.inf, -np.inf], np.nan), label[order], groups


def fit_side_ranker(train: pd.DataFrame, fields: Sequence[str]):
    labels = residual_grade(train["realized_residual_bps"].to_numpy(float))
    x, y, groups = _ranker_arrays(train, fields, labels)
    if len(x) < 1_000:
        raise ValueError("insufficient resolved rows for side-local LambdaRank")
    if any(x[field].notna().sum() == 0 for field in fields):
        raise ValueError("a declared model feature is entirely unavailable in a training fold")
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1, 2, 3, 4],
        n_estimators=250, learning_rate=.035, num_leaves=24, min_child_samples=200,
        colsample_bytree=.8, subsample=.8, subsample_freq=1, reg_lambda=12.,
        lambdarank_truncation_level=10, random_state=SEED, n_jobs=1, verbosity=-1,
    )
    model.fit(x, y, group=groups)
    return model, {"train_rows": int(len(x)), "query_groups": int(len(groups))}


def fit_residual_calibration(raw_score: Sequence[float], residual_bps: Sequence[float]) -> IsotonicRegression:
    raw, target = np.asarray(raw_score, dtype=float), np.asarray(residual_bps, dtype=float)
    good = np.isfinite(raw) & np.isfinite(target)
    if int(good.sum()) < 500:
        raise ValueError("insufficient resolved calibration rows for rank-score-to-residual map")
    return IsotonicRegression(out_of_bounds="clip", y_min=-1_000., y_max=1_000.).fit(raw[good], target[good])


def _metric_rows(test: pd.DataFrame, *, fold: ContinuousFold, arm: str, score: np.ndarray) -> list[dict[str, Any]]:
    scored = test.copy(); scored["selection_score"] = score; scored["month"] = scored["__ts__"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for period, window in [("fold", scored), *((month, q) for month, q in scored.groupby("month", sort=True, observed=True))]:
        for side, side_rows in [("pooled", window), *((name, q) for name, q in window.groupby("side_name", sort=True, observed=True))]:
            for tail in TAILS:
                count = max(1, int(np.ceil(len(side_rows) * tail)))
                chosen = side_rows.sort_values(["selection_score", "candidate_id"], ascending=[False, True], kind="stable").head(count)
                rows.append({
                    "fold": fold.name, "fold_family": fold.family, "arm": arm, "period": period,
                    "side": side, "tail": tail, "rows": int(len(side_rows)), "tail_rows": int(len(chosen)),
                    "tail_coverage": float(len(chosen) / len(side_rows)) if len(side_rows) else np.nan,
                    "net_bps": float(chosen["net_bps"].mean()), "gross_bps": float(chosen["gross_bps"].mean()),
                    "selection_score": float(chosen["selection_score"].mean()),
                    "rank_ic": float(side_rows["selection_score"].rank().corr(side_rows["net_bps"].rank())),
                })
    return rows


def run(*, input_path: Path = INPUT, out: Path = DEFAULT_OUT, folds: Sequence[ContinuousFold] = FOLDS) -> Path:
    """Execute every declared fold; no fitting occurs on import."""
    data = load_ledger(input_path)
    out.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    importance: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    fold_audits: dict[str, Any] = {}
    for fold in folds:
        panel, arms, audit = _build_fold_feature_panel(data, fold)
        fold_audits[fold.name] = audit
        train_start, calibration_start, test_start, test_end = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        train = panel.loc[panel["__ts__"].between(train_start, calibration_start, inclusive="left") & panel["label_available_ts"].lt(calibration_start)].copy()
        calibration = panel.loc[panel["__ts__"].between(calibration_start, test_start, inclusive="left") & panel["label_available_ts"].lt(test_start)].copy()
        test = panel.loc[panel["__ts__"].between(test_start, test_end, inclusive="left")].copy()
        if train.empty or calibration.empty or test.empty:
            raise ValueError(f"empty strict continuous split: {fold.name}")
        # Exact broad-base control, globally ranked on the same test candidates.
        metrics.extend(_metric_rows(
            test, fold=fold, arm="B0_direct_r3_pclear_minus_half_padverse_global",
            score=test["base_score_clear_minus_half_adverse"].to_numpy(float),
        ))
        base_prediction = test.loc[:, ["candidate_id", "__ts__", "side_name", "net_bps", *BASE_OUTPUT_FIELDS]].copy()
        base_prediction["fold"] = fold.name
        base_prediction["arm"] = "B0_direct_r3_pclear_minus_half_padverse_global"
        base_prediction["predicted_residual_bps"] = np.nan
        base_prediction["selection_score"] = base_prediction["base_score_clear_minus_half_adverse"]
        prediction_parts.append(base_prediction)
        for arm, fields in arms.items():
            for field in fields:
                coverage.append({"fold": fold.name, "arm": arm, "feature": field, "train_coverage": float(train[field].notna().mean()), "calibration_coverage": float(calibration[field].notna().mean()), "test_coverage": float(test[field].notna().mean())})
            unavailable = [field for field in fields if train[field].notna().sum() == 0]
            # This is a pre-test, outcome-free availability screen.  A
            # singleton/constant family has no defined internal covariance
            # geometry; excluding its undefined fields is preferable to
            # treating NaN as a market-break signal or discarding the full arm.
            effective_fields = tuple(field for field in fields if field not in unavailable)
            if not effective_fields:
                fold_audits[fold.name].setdefault("arms", {})[arm] = {
                    "status": "SKIPPED_NO_CAUSALLY_AVAILABLE_FEATURES",
                    "unavailable_train_features": unavailable,
                }
                continue
            predicted = np.empty(len(test), dtype=float)
            details: dict[str, Any] = {}
            for side in ("long", "short"):
                train_side = train.loc[train["side_name"].eq(side)].copy()
                cal_side = calibration.loc[calibration["side_name"].eq(side)].copy()
                test_side = test.loc[test["side_name"].eq(side)].copy()
                if test_side.empty:
                    continue
                model, detail = fit_side_ranker(train_side, effective_fields)
                gains = model.booster_.feature_importance(importance_type="gain")
                for field, gain in zip(effective_fields, gains, strict=True):
                    importance.append({
                        "fold": fold.name, "fold_family": fold.family, "arm": arm,
                        "side": side, "feature": field, "gain": float(gain),
                        "feature_count_effective": int(len(effective_fields)),
                    })
                raw_calibration = np.asarray(model.predict(cal_side.loc[:, effective_fields].replace([np.inf, -np.inf], np.nan)), dtype=float)
                calibrator = fit_residual_calibration(raw_calibration, cal_side["realized_residual_bps"])
                position = test.index.get_indexer(test_side.index)
                raw_test = np.asarray(model.predict(test_side.loc[:, effective_fields].replace([np.inf, -np.inf], np.nan)), dtype=float)
                predicted[position] = calibrator.predict(raw_test)
                details[side] = detail | {"calibration_rows": int(len(cal_side)), "feature_count_effective": int(len(effective_fields))}
            if unavailable:
                details["availability_screen"] = {
                    "status": "PARTIAL_CAUSAL_WARMUP_EXCLUDED",
                    "unavailable_train_features": unavailable,
                    "effective_features": list(effective_fields),
                }
            expected_net = test["prequential_base_expected_net_bps"].to_numpy(float) + predicted
            metrics.extend(_metric_rows(test, fold=fold, arm=arm, score=expected_net))
            prediction = test.loc[:, ["candidate_id", "__ts__", "side_name", "net_bps", *BASE_OUTPUT_FIELDS]].copy()
            prediction["fold"], prediction["arm"] = fold.name, arm
            prediction["predicted_residual_bps"], prediction["selection_score"] = predicted, expected_net
            prediction_parts.append(prediction)
            fold_audits[fold.name].setdefault("arms", {})[arm] = details
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    pd.DataFrame(coverage).to_parquet(out / "coverage.parquet", index=False)
    pd.DataFrame(importance).to_parquet(out / "feature_importance.parquet", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION", "input": str(input_path),
        "source_spine_fields": list(SOURCE_SPINE_FIELDS), "existing_causal_meta_fields": list(EXISTING_CAUSAL_META_FIELDS),
        "base_output_fields": list(BASE_OUTPUT_FIELDS),
        "arms": {
            "B0_direct_r3_pclear_minus_half_padverse_global": "direct strict-OOF R3 base score, globally ranked",
            "M0_base_plus_existing_causal_meta": "B0 base outputs plus existing causal meta inputs",
            "M1_base_plus_raw_spine_sources": "B0 base outputs plus 21 pure causal raw sources",
            "M2_base_plus_spine_factors": "M1 plus frozen cluster factors",
            "M3_base_plus_spine_factor_breaks": "M2 plus frozen covariance-break features",
        },
        "folds": [asdict(fold) | {"calibration_end": fold.calibration_end.isoformat()} for fold in folds],
        "training": "strictly resolved labels; side-local native LightGBM LambdaRank; timestamp x side query groups; fixed residual grades <=-150,-150..-50,-50..50,50..150,>150",
        "calibration": "side-local isotonic map fit on the preceding resolved calibration month: raw rank score -> realized residual bps",
        "reconstruction": "prequential_base_expected_net_bps + calibrated_predicted_residual_bps; B0 is ranked only by p_clear - 0.5*p_adverse",
        "cluster_audits": fold_audits,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--long-history", action="store_true", help="use all compatible resolved TP6/SL4 history before each later transport fold")
    parser.add_argument("--transport-only", action="store_true", help="run only the three later transport folds for focused attribution")
    args = parser.parse_args()
    selected_folds = LONG_HISTORY_FOLDS if args.long_history else FOLDS
    if args.transport_only:
        selected_folds = selected_folds[3:]
    print(run(input_path=args.input, out=args.out, folds=selected_folds))


if __name__ == "__main__":
    main()
