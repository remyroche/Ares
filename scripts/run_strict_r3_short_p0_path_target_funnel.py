#!/usr/bin/env python3
"""Strict chronological P0 path-target funnel for the short base.

This producer tests whether the short P0/F90 conversion failure is primarily
an opportunity-label problem or a policy-conversion problem.  It deliberately
uses *only* the frozen 41-field M4 base contract.  Rich exact-one-minute path
quantities are supervised labels, never inference fields.

Targets
-------
T0  Stored strict-OOF M4 control (six-state policy-margin ordinal).
T1  Cost-clear opportunity magnitude ordinal from exact H12 MFE.
T2  Fast cost-clear ordinal, with velocity cut points fitted per training fold.
T3  Adverse-adjusted 3h opportunity, lambda = 0.25 / 0.50 / 1.00.
T4  min(opportunity grade, train-fitted convertibility grade).
T5  Conditional low-policy-regret diagnostic where MFE_H12 > 200 bps.

For every held calendar month models see only rows whose exact H12 label was
available before the month began.  The score-to-policy-net map is fitted only
on expanding chronological OOF predictions from that same training fold.
Invalid/incomplete outcomes are scored as live candidates but have no target,
do not train a model, and never reserve pseudo-capacity in outcome metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA = "strict_r3_short_p0_path_target_funnel_v1"
SIDE = "short"
POLICY_CLIP_BPS = 500.0
MIN_TRAIN_ROWS = 500
MIN_OOF_ROWS = 240
OOF_SPLITS = 3
P80 = 0.80
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
CORE_SOURCE = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
)
T1_EDGES = (100.0, 150.0, 250.0, 400.0, 600.0)
T5_REGRET_EDGES = (50.0, 100.0, 200.0, 400.0)
T3_LAMBDAS = (0.25, 0.50, 1.00)
ERAS = (
    ("2024", pd.Timestamp("2024-05-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    ("2025", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
    ("2026", pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp | pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, utc=True, errors="raise")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _finite(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.isclose(
        pd.to_numeric(left, errors="coerce").to_numpy(float),
        pd.to_numeric(right, errors="coerce").to_numpy(float),
        rtol=0.0, atol=2e-4, equal_nan=True,
    ).all())


def _deduplicate_population_frame(result: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    """Validate cumulative copies, retaining both shared and unique rows."""
    duplicate = result["candidate_id"].duplicated(keep=False)
    if not duplicate.any():
        return result.copy()
    repeated = result.loc[duplicate].sort_values(["candidate_id", "__decision_ts__"], kind="stable")
    # A cumulative immutable source is admissible only when every retained
    # target-free frozen field is identical across copies.
    for candidate_id, block in repeated.groupby("candidate_id", sort=False):
        for column in (*CORE_SOURCE[1:], *fields):
            values = block[column]
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                first = pd.Series(np.repeat(values.iloc[0], len(values)), index=values.index)
                equal = _numeric_equal(values, first)
            else:
                equal = bool((values.eq(values.iloc[0]) | (values.isna() & pd.isna(values.iloc[0]))).all())
            if not equal:
                raise AssertionError(f"cumulative P0 sources disagree on {column} for {candidate_id}")
    # Earlier artifacts contain duplicate warm-up rows; later artifacts also
    # introduce genuinely new months.  Keep one identical copy of each shared
    # identity *and* every unique identity.  This must not become a de facto
    # historical-support filter.
    unique = result.loc[~duplicate]
    canonical = repeated.drop_duplicates("candidate_id", keep="first")
    return pd.concat([unique, canonical], ignore_index=True)


def _stable_unique_population(roots: Sequence[Path], fields: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    columns = [*CORE_SOURCE, *fields]
    for root in roots:
        population = root / "short_p0_top1_hourly_population.parquet"
        manifest = root / "run_manifest.json"
        if not population.exists() or not manifest.exists():
            raise FileNotFoundError(f"not an immutable P0 source: {root}")
        frame = pd.read_parquet(population, columns=columns)
        for column in ("__ts__", "__decision_ts__", "policy_label_available_at"):
            frame[column] = _utc(frame[column])
        if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
            raise ValueError(f"non-short P0 source: {root}")
        pieces.append(frame)
        hashes[str(root.resolve())] = _sha256(manifest)
    result = pd.concat(pieces, ignore_index=True)
    result = _deduplicate_population_frame(result, fields)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("P0 source identity deduplication failed")
    if not result["__decision_ts__"].eq(result["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("P0 target-free entry convention is not signal close + one hour")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), hashes


def _load_fields(root: Path) -> tuple[str, ...]:
    path = root / "feature_contract.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    fields = tuple(payload.get("base", ()))
    if len(fields) != 41 or len(set(fields)) != 41:
        raise AssertionError("target funnel requires the frozen 41-field M4 base contract")
    return fields


def _load_rich_labels(root: Path) -> tuple[pd.DataFrame, str]:
    manifest = root / "run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    columns = [
        *IDENTITY, "__label_available_at__", "rich_path_label_valid", "rich_path_target_invalid",
        "mfe_3h_bps", "mfe_12h_bps", "reached_100bps", "time_to_100bps_minutes",
        "mae_before_100bps_atr", "mae_before_mfe_3h_bps", "policy_stop_out",
        "policy_regret_bps", "policy_net_bps", "policy_gross_bps", "policy_exit_reason",
    ]
    parts = sorted(root.glob("parts/month=*/side=short.parquet"))
    if not parts:
        raise FileNotFoundError(f"no rich exact-path parts beneath {root}")
    frame = pd.concat([pd.read_parquet(part, columns=columns) for part in parts], ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = _utc(frame[column])
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError("rich exact-path labels are not a unique short identity population")
    return frame, _sha256(manifest)


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["rich_path_label_valid"].fillna(False).astype(bool)
        & ~frame["rich_path_target_invalid"].fillna(True).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["policy_net_bps"]).notna()
        & frame["__label_available_at__"].notna()
    )


def _load_m4(roots: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    cols = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "arm", "held_month",
        "expected_net_bps", "raw_meta_score", "train_p80_expected_bps",
        "policy_path_valid",
    ]
    for root in roots:
        path = root / "short_absolute_conversion_oof_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=cols)
        frame = frame.loc[frame["arm"].astype(str).eq("M4")].copy()
        frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
        pieces.append(frame)
        hashes[str(root.resolve())] = _sha256(path)
    result = pd.concat(pieces, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("stored M4 OOF predictions overlap in candidate IDs")
    return result, hashes


def _matrix(frame: pd.DataFrame, fields: tuple[str, ...], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, list(fields)].apply(_finite)
    if medians is None:
        medians = values.median().fillna(0.0)
    return values.fillna(medians).fillna(0.0).astype(np.float32), medians


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: Literal["ordinal", "regression"]
    description: str
    lambda_: float | None = None
    diagnostic_only: bool = False


SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("T1_cost_clear_magnitude", "ordinal", "Six-grade MFE-H12 opportunity magnitude above fixed cost."),
    TargetSpec("T2_fast_cost_clear", "ordinal", "Train-quantile gross-opportunity velocity after clearing +100 bps."),
    *(TargetSpec(f"T3_adverse_adjusted_mfe3h_l{str(value).replace('.', '')}", "regression", "MFE-3h minus train-fixed lambda times prior adverse move.", lambda_=value) for value in T3_LAMBDAS),
    TargetSpec("T4_min_opportunity_convertibility", "ordinal", "min(fixed MFE opportunity grade, train-fitted clean-conversion grade)."),
    TargetSpec("T5_conditional_low_regret", "ordinal", "Conditional low-policy-regret quality given MFE-H12 > 200 bps.", diagnostic_only=True),
)


def _t1_grade(frame: pd.DataFrame) -> np.ndarray:
    return np.digitize(_finite(frame["mfe_12h_bps"]).to_numpy(float), T1_EDGES, right=False).astype(int)


def _fit_target_params(frame: pd.DataFrame, spec: TargetSpec) -> dict[str, float | list[float]]:
    if spec.name == "T2_fast_cost_clear":
        reach = frame["reached_100bps"].fillna(False).astype(bool).to_numpy()
        speed = np.divide(
            _finite(frame["mfe_12h_bps"]).to_numpy(float),
            np.maximum(_finite(frame["time_to_100bps_minutes"]).to_numpy(float), 1.0),
            out=np.full(len(frame), np.nan), where=reach,
        )
        values = speed[np.isfinite(speed)]
        if len(values) < 80 or np.unique(values).size < 5:
            raise ValueError("insufficient fast-clear support to fit train-only target cuts")
        edges = np.quantile(values, (0.20, 0.40, 0.60, 0.80)).astype(float)
        if np.unique(edges).size < len(edges):
            raise ValueError("fast-clear train quantiles are degenerate")
        return {"speed_edges": edges.tolist()}
    if spec.name == "T4_min_opportunity_convertibility":
        reach = frame["reached_100bps"].fillna(False).astype(bool)
        local = frame.loc[reach].copy()
        if len(local) < 80:
            raise ValueError("insufficient cost-clear support to fit convertibility cuts")
        time = _finite(local["time_to_100bps_minutes"]).dropna().to_numpy(float)
        mae = _finite(local["mae_before_100bps_atr"]).dropna().to_numpy(float)
        mfe = _finite(local["mfe_12h_bps"]).dropna().to_numpy(float)
        if min(len(time), len(mae), len(mfe)) < 80:
            raise ValueError("incomplete convertibility labels")
        return {
            "fast_time": float(np.quantile(time, .33)),
            "slow_time": float(np.quantile(time, .67)),
            "clean_mae": float(np.quantile(mae, .33)),
            "substantial_mae": float(np.quantile(mae, .67)),
            "severe_mae": float(np.quantile(mae, .80)),
            "continuation_mfe": float(np.quantile(mfe, .75)),
        }
    return {}


def _domain(frame: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    if spec.name == "T5_conditional_low_regret":
        return frame.loc[_finite(frame["mfe_12h_bps"]).gt(200.0)].copy()
    return frame.copy()


def _target_values(frame: pd.DataFrame, spec: TargetSpec, params: dict[str, float | list[float]]) -> np.ndarray:
    if spec.name == "T1_cost_clear_magnitude":
        return _t1_grade(frame)
    if spec.name == "T2_fast_cost_clear":
        reach = frame["reached_100bps"].fillna(False).astype(bool).to_numpy()
        speed = np.divide(
            _finite(frame["mfe_12h_bps"]).to_numpy(float),
            np.maximum(_finite(frame["time_to_100bps_minutes"]).to_numpy(float), 1.0),
            out=np.full(len(frame), np.nan), where=reach,
        )
        result = np.zeros(len(frame), dtype=int)
        result[reach] = np.digitize(speed[reach], np.asarray(params["speed_edges"], dtype=float), right=False).astype(int) + 1
        return result
    if spec.name.startswith("T3_adverse_adjusted_mfe3h"):
        assert spec.lambda_ is not None
        return np.clip(
            _finite(frame["mfe_3h_bps"]).to_numpy(float) - spec.lambda_ * _finite(frame["mae_before_mfe_3h_bps"]).to_numpy(float),
            -POLICY_CLIP_BPS, POLICY_CLIP_BPS,
        )
    if spec.name == "T4_min_opportunity_convertibility":
        opportunity = _t1_grade(frame)
        reach = frame["reached_100bps"].fillna(False).astype(bool).to_numpy()
        stop = frame["policy_stop_out"].fillna(False).astype(bool).to_numpy()
        time = _finite(frame["time_to_100bps_minutes"]).to_numpy(float)
        mae = _finite(frame["mae_before_100bps_atr"]).to_numpy(float)
        mfe = _finite(frame["mfe_12h_bps"]).to_numpy(float)
        conversion = np.zeros(len(frame), dtype=int)
        usable = reach & ~stop & np.isfinite(time) & np.isfinite(mae)
        conversion[usable] = 3
        conversion[usable & (time >= float(params["slow_time"]))] = 1
        conversion[usable & (mae >= float(params["substantial_mae"]))] = 2
        severe = reach & (stop | (np.isfinite(mae) & (mae >= float(params["severe_mae"]))))
        conversion[severe] = 0
        fast_clean = usable & (time <= float(params["fast_time"])) & (mae <= float(params["clean_mae"]))
        conversion[fast_clean] = 4
        conversion[fast_clean & (mfe >= float(params["continuation_mfe"]))] = 5
        return np.minimum(opportunity, conversion)
    if spec.name == "T5_conditional_low_regret":
        regret = _finite(frame["policy_regret_bps"]).to_numpy(float)
        return (4 - np.digitize(regret, T5_REGRET_EDGES, right=False)).astype(int)
    raise ValueError(spec.name)


def _model(spec: TargetSpec, *, seed: int):
    common = dict(
        n_estimators=160, learning_rate=.035, max_depth=3, num_leaves=15,
        min_child_samples=35, subsample=.85, colsample_bytree=.85,
        reg_lambda=4.0, reg_alpha=.10, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    if spec.kind == "regression":
        return LGBMRegressor(objective="huber", alpha=.90, **common)
    return LGBMClassifier(objective="multiclass", num_class=6 if spec.name != "T5_conditional_low_regret" else 5, class_weight="balanced", **common)


def _raw_prediction(model, x: pd.DataFrame, spec: TargetSpec) -> np.ndarray:
    if spec.kind == "regression":
        return np.asarray(model.predict(x), dtype=float)
    probability = np.asarray(model.predict_proba(x), dtype=float)
    values = np.arange(probability.shape[1], dtype=float)
    return probability @ values


def _chronological_oof_raw(train: pd.DataFrame, fields: tuple[str, ...], spec: TargetSpec, *, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    local = _domain(train, spec).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    n = len(local)
    boundaries = np.linspace(0, n, OOF_SPLITS + 2, dtype=int)
    rows: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    for fold in range(OOF_SPLITS):
        fit_end, valid_end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        if fit_end < max(160, MIN_TRAIN_ROWS // 3) or valid_end <= fit_end:
            continue
        fit, valid = local.iloc[:fit_end], local.iloc[fit_end:valid_end]
        params = _fit_target_params(fit, spec)
        y = _target_values(fit, spec, params)
        if np.unique(y).size < 2:
            continue
        x_fit, medians = _matrix(fit, fields)
        x_valid, _ = _matrix(valid, fields, medians)
        estimator = _model(spec, seed=seed + fold)
        estimator.fit(x_fit, y)
        rows.append(np.arange(fit_end, valid_end, dtype=int))
        raw_parts.append(_raw_prediction(estimator, x_valid, spec))
    if not raw_parts:
        raise ValueError("insufficient chronological support for OOF target calibration")
    return np.concatenate(rows), np.concatenate(raw_parts), local


def _fit_calibrator(train: pd.DataFrame, fields: tuple[str, ...], spec: TargetSpec, *, seed: int) -> tuple[IsotonicRegression, float, np.ndarray, int]:
    indices, raw, local = _chronological_oof_raw(train, fields, spec, seed=seed)
    observed = _finite(local.iloc[indices]["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    if len(raw) < MIN_OOF_ROWS or np.unique(raw).size < 4:
        raise ValueError("insufficient OOF diversity for policy-net calibration")
    rho = float(pd.Series(raw).corr(pd.Series(observed), method="spearman"))
    calibrator = IsotonicRegression(
        increasing=bool(np.nan_to_num(rho, nan=0.0) >= 0.0), out_of_bounds="clip",
        y_min=-POLICY_CLIP_BPS, y_max=POLICY_CLIP_BPS,
    )
    calibrator.fit(raw, observed)
    return calibrator, rho, raw, int(len(raw))


def _m4_style_train_p80(calibrator: IsotonicRegression, oof_raw: np.ndarray) -> float:
    """Match the stored M4 producer: calibrate the raw OOF p80 cut point."""
    raw_cut = float(np.quantile(np.asarray(oof_raw, dtype=float), P80))
    return float(calibrator.predict(np.asarray([raw_cut], dtype=float))[0])


def _fit_predict(train: pd.DataFrame, held: pd.DataFrame, fields: tuple[str, ...], spec: TargetSpec, *, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    domain = _domain(train, spec)
    if len(domain) < MIN_TRAIN_ROWS:
        raise ValueError(f"insufficient target-domain training rows: {len(domain)}")
    params = _fit_target_params(domain, spec)
    y = _target_values(domain, spec, params)
    if np.unique(y).size < 2:
        raise ValueError("target has fewer than two training classes")
    calibrator, rho, oof_raw, oof_rows = _fit_calibrator(domain, fields, spec, seed=seed)
    x_train, medians = _matrix(domain, fields)
    x_held, _ = _matrix(held, fields, medians)
    model = _model(spec, seed=seed + 100)
    model.fit(x_train, y)
    # Every downstream metric uses labelled boolean masks.  Reset the held
    # index at the producer boundary so a month inherited from the cumulative
    # P0 frame cannot ever align a mask to another fold's source index.
    output = held.copy().reset_index(drop=True)
    raw = _raw_prediction(model, x_held, spec)
    output["raw_score"] = raw.astype(np.float32)
    output["expected_policy_net_bps"] = calibrator.predict(raw).astype(np.float32)
    # This is intentionally *not* percentile(expected OOF bps).  M4 first
    # takes the p80 of its raw model score, then sends that point through its
    # isotonic map.  The distinction matters when an OOF map has flat bins.
    output["train_p80_expected_policy_net_bps"] = _m4_style_train_p80(calibrator, oof_raw)
    # Held target values are for post-outcome diagnostic metrics only.  For
    # T2/T4 their cuts are the frozen outer-training cuts, never held cuts.
    target = np.full(len(output), np.nan, dtype=float)
    target_domain_mask = _domain(output, spec).index
    if len(target_domain_mask):
        target[output.index.get_indexer(target_domain_mask)] = _target_values(output.loc[target_domain_mask], spec, params)
    output["diagnostic_target_value"] = target.astype(np.float32)
    return output, {
        "train_rows": int(len(train)), "target_domain_train_rows": int(len(domain)),
        "feature_count": int(len(fields)), "oof_calibration_rows": int(oof_rows),
        "oof_raw_policy_net_spearman": rho, "train_p80_expected_policy_net_bps": _m4_style_train_p80(calibrator, oof_raw),
        "target_parameters": params,
    }


def _selection_metrics(frame: pd.DataFrame, *, condition: pd.Series, evaluation_mask: pd.Series | None = None) -> dict[str, float]:
    active = condition if evaluation_mask is None else (condition & evaluation_mask)
    selected = frame.loc[active].copy()
    valid = selected.loc[_valid_label(selected)].copy()
    y = _finite(valid["policy_net_bps"]).to_numpy(float)
    return {
        "scored_candidates": float(len(selected)),
        "outcome_known_candidates": float(len(valid)),
        "outcome_coverage": float(len(valid) / len(selected)) if len(selected) else float("nan"),
        "trades": float(len(valid)),
        "net_bps_per_trade": float(np.mean(y)) if len(y) else float("nan"),
        "total_net_bps": float(np.sum(y)) if len(y) else float("nan"),
        "positive_rate": float(np.mean(y > 0.0)) if len(y) else float("nan"),
    }


def _metric_rows(prediction: pd.DataFrame, *, arm: str, month: pd.Timestamp, diagnostic_only: bool) -> list[dict[str, Any]]:
    score = _finite(prediction["expected_policy_net_bps"])
    valid_mask = _valid_label(prediction)
    # T5 asks a deliberately conditional question: can causal state predict
    # low regret *when a real opportunity exists*?  Its fit already limits to
    # MFE-H12 > 200.  The evaluation must apply the identical oracle label
    # condition and must never be confused with a deployable full-population
    # admission result.
    if diagnostic_only:
        valid_mask = valid_mask & _finite(prediction["mfe_12h_bps"]).gt(200.0)
    valid = prediction.loc[valid_mask].copy()
    base: dict[str, Any] = {
        "arm": arm, "held_month": month.strftime("%Y-%m"), "diagnostic_only": diagnostic_only,
        "held_scored_candidates": int(len(prediction)), "held_outcome_known": int(len(valid)),
        "held_outcome_coverage": float(len(valid) / len(prediction)) if len(prediction) else float("nan"),
    }
    if len(valid):
        y = _finite(valid["policy_net_bps"])
        s = _finite(valid["expected_policy_net_bps"])
        base["score_policy_net_spearman"] = float(s.corr(y, method="spearman"))
        target = _finite(valid["diagnostic_target_value"])
        base["score_target_spearman"] = float(s[target.notna()].corr(target[target.notna()], method="spearman")) if target.notna().sum() > 4 else float("nan")
        for label, threshold in (("net_gt0", 0.0), ("net_gt100", 100.0)):
            truth = (y > threshold).astype(int)
            if truth.nunique() > 1:
                base[f"auc_{label}"] = float(roc_auc_score(truth, s))
                base[f"prauc_{label}"] = float(average_precision_score(truth, s))
            else:
                base[f"auc_{label}"] = float("nan")
                base[f"prauc_{label}"] = float("nan")
    else:
        for field in ("score_policy_net_spearman", "score_target_spearman", "auc_net_gt0", "prauc_net_gt0", "auc_net_gt100", "prauc_net_gt100"):
            base[field] = float("nan")
    p80 = float(prediction["train_p80_expected_policy_net_bps"].iat[0])
    return [
        {**base, "selection": "all_scored_outcome_known", "causal_selection": False, **_selection_metrics(prediction, condition=pd.Series(True, index=prediction.index), evaluation_mask=valid_mask if diagnostic_only else None)},
        {**base, "selection": "conditional_train_p80_expected_bps" if diagnostic_only else "causal_train_p80_expected_bps", "causal_selection": False if diagnostic_only else True, **_selection_metrics(prediction, condition=score.ge(p80), evaluation_mask=valid_mask if diagnostic_only else None)},
    ]


def _era_of(series: pd.Series) -> pd.Series:
    result = pd.Series(index=series.index, dtype="string")
    stamp = _utc(series)
    for era, start, end in ERAS:
        result.loc[stamp.ge(start) & stamp.lt(end)] = era
    return result


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, selection, diagnostic, era), block in monthly.groupby(["arm", "selection", "diagnostic_only", "era"], dropna=False, sort=True):
        trades = float(block["trades"].sum())
        total = float(block["total_net_bps"].sum(min_count=1))
        rows.append({
            "arm": arm, "selection": selection, "diagnostic_only": bool(diagnostic), "era": era,
            "months": int(block["held_month"].nunique()), "trades": trades,
            "net_bps_per_trade": total / trades if trades else float("nan"), "total_net_bps": total,
            "positive_months": int((block["net_bps_per_trade"] > 0.0).sum()),
            "worst_month_net_bps_per_trade": float(block["net_bps_per_trade"].min()),
            "mean_outcome_coverage": float(block["outcome_coverage"].mean()),
            "mean_score_policy_net_spearman": float(block["score_policy_net_spearman"].mean()),
            "mean_auc_net_gt0": float(block["auc_net_gt0"].mean()),
            "mean_auc_net_gt100": float(block["auc_net_gt100"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["selection", "era", "net_bps_per_trade"], ascending=[True, True, False], kind="stable")


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for record in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{float(value):.3f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *body])


def _report(out: Path, *, aggregate: pd.DataFrame, fold: pd.DataFrame, manifest: dict[str, Any]) -> None:
    p80 = aggregate.loc[aggregate["selection"].eq("causal_train_p80_expected_bps")].copy()
    p80 = p80.loc[:, ["arm", "era", "months", "trades", "net_bps_per_trade", "total_net_bps", "positive_months", "worst_month_net_bps_per_trade", "mean_outcome_coverage", "mean_score_policy_net_spearman"]]
    conditional = aggregate.loc[aggregate["selection"].eq("conditional_train_p80_expected_bps")].copy()
    conditional = conditional.loc[:, ["arm", "era", "months", "trades", "net_bps_per_trade", "total_net_bps", "positive_months", "worst_month_net_bps_per_trade", "mean_outcome_coverage", "mean_score_policy_net_spearman"]]
    lines = [
        "# Short P0 exact-path target funnel",
        "",
        "This is strict chronological OOF research using the frozen 41-field M4 input contract. Rich one-minute path quantities are labels only; no new inference feature, score gate, policy geometry, or live stack is changed.",
        "",
        "## Causal train-p80 policy-net admission",
        "",
        _markdown(p80),
        "",
        "## Conditional conversion diagnostic (not deployable)",
        "",
        _markdown(conditional),
        "",
        "## Fold coverage",
        "",
        _markdown(fold.loc[:, [column for column in ("held_month", "arm", "status", "train_rows", "target_domain_train_rows", "held_rows", "oof_calibration_rows", "oof_raw_policy_net_spearman") if column in fold]]),
        "",
        "## Interpretation",
        "",
        "- T0 is the stored historical M4 control; it may have shorter 2025 support than the new exact-path arms.",
        "- T1–T4 score the full target-free held P0 population. Invalid/unresolved outcomes are not fitted and are shown only through outcome coverage.",
        "- T5 conditions on realised MFE > 200 bps in both fitting and evaluation. Its `conditional_train_p80_expected_bps` result is an oracle-condition diagnostic, not a deployable admission model.",
        "- An arm advances only if its causal p80 policy-net result improves cross-era portability without a severe 2024 failure. No automatic promotion occurs here.",
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
    ]
    (out / "SHORT_P0_PATH_TARGET_FUNNEL_REPORT.md").write_text("\n".join(lines) + "\n")


def repair_metric_scope(*, source_artifact: Path, rich_labels_root: Path, out: Path) -> Path:
    """Supersede a sealed fit only when a pure reporting scope needs repair.

    This is deliberately narrow: it copies the original model scores and
    fold audit after hashing them, then rebuilds only the outcome metric tables
    from the immutable exact-path labels.  It cannot tune, refit, rescore, or
    change any admission threshold.
    """
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    prediction_path = source_artifact / "path_target_oof_predictions.parquet"
    source_manifest_path = source_artifact / "run_manifest.json"
    fold_path = source_artifact / "fold_audit.parquet"
    if not prediction_path.exists() or not source_manifest_path.exists() or not fold_path.exists():
        raise FileNotFoundError("metric repair requires a sealed target-funnel source artifact")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema") != SCHEMA or source_manifest.get("status") != "complete":
        raise AssertionError("metric repair source is not a completed strict target-funnel artifact")
    prediction = pd.read_parquet(prediction_path)
    if prediction.duplicated(["arm", "candidate_id"]).any():
        raise AssertionError("source prediction identities are not unique by arm")
    labels, label_hash = _load_rich_labels(rich_labels_root)
    label_fields = labels.loc[:, [*IDENTITY, "mfe_12h_bps"]]
    # Use an explicit name: v3 predates the persisted MFE label context, so
    # pandas would otherwise keep it unsuffixed when joining the label table.
    label_fields = label_fields.rename(columns={"mfe_12h_bps": "__repair_mfe_12h_bps"})
    merged = prediction.merge(label_fields, on=list(IDENTITY), how="left", validate="many_to_one")
    if len(merged) != len(prediction):
        raise AssertionError("metric repair changed prediction identities")
    if "mfe_12h_bps" in merged:
        existing = _finite(merged["mfe_12h_bps"])
        repaired = _finite(merged["__repair_mfe_12h_bps"])
        both = existing.notna() & repaired.notna()
        if both.any() and not _numeric_equal(existing.loc[both], repaired.loc[both]):
            raise AssertionError("source prediction MFE label differs from immutable rich-label source")
        merged["mfe_12h_bps"] = existing.where(existing.notna(), repaired)
    else:
        merged["mfe_12h_bps"] = _finite(merged["__repair_mfe_12h_bps"])
    merged = merged.drop(columns="__repair_mfe_12h_bps")
    # Non-T0 source rows already carry label validity.  A label rejoin may not
    # revise it; failure would mean the source and current label contracts are
    # not the same artifact family.
    source_valid = merged["rich_path_label_valid"].fillna(False).astype(bool)
    joined_valid = merged["mfe_12h_bps"].notna()
    if (source_valid & ~joined_valid).any():
        raise AssertionError("metric repair cannot recover MFE for a source-valid exact path")
    metric_rows: list[dict[str, Any]] = []
    for (arm, held_month), block in merged.groupby(["arm", pd.to_datetime(merged["__decision_ts__"], utc=True).dt.to_period("M")], sort=True):
        month = pd.Timestamp(held_month.start_time, tz="UTC")
        diagnostic = str(arm) == "T5_conditional_low_regret"
        metric_rows.extend(_metric_rows(block.copy(), arm=str(arm), month=month, diagnostic_only=diagnostic))
    monthly = pd.DataFrame(metric_rows)
    monthly["era"] = _era_of(pd.to_datetime(monthly["held_month"] + "-01", utc=True))
    aggregate = _aggregate(monthly)
    fold = pd.read_parquet(fold_path)
    manifest = dict(source_manifest)
    manifest["status"] = "complete"
    manifest["metrics_repair"] = {
        "reason": "T5 conditional low-regret evaluation now conditions on the same realised MFE_H12 > 200 bps domain as its training target.",
        "source_artifact": str(source_artifact.resolve()),
        "source_prediction_sha256": _sha256(prediction_path),
        "source_fold_audit_sha256": _sha256(fold_path),
        "rich_label_manifest_sha256": label_hash,
        "guarantee": "scores, thresholds, models, feature contract, candidate identities, policy labels, and fold audit are copied from the sealed source; only metric aggregation is recomputed.",
    }
    out.mkdir(parents=True)
    merged.to_parquet(out / "path_target_oof_predictions.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    aggregate.to_parquet(out / "cross_era_metrics.parquet", index=False, compression="zstd")
    fold.to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    (out / "feature_contract.json").write_text(json.dumps(manifest["feature_contract"], indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, aggregate=aggregate, fold=fold, manifest=manifest)
    return out


def run(*, population_roots: Sequence[Path], m4_roots: Sequence[Path], rich_labels_root: Path, out: Path, start: pd.Timestamp, end: pd.Timestamp, seed: int) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    fields = _load_fields(population_roots[0])
    for root in population_roots[1:]:
        if _load_fields(root) != fields:
            raise AssertionError("frozen 41-field M4 contracts differ across source artifacts")
    population, population_hashes = _stable_unique_population(population_roots, fields)
    labels, label_hash = _load_rich_labels(rich_labels_root)
    merged = population.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(merged) != len(population):
        raise AssertionError("rich-label join changed P0 candidate identities")
    overlap = _finite(merged["p0_canonical_net_bps"]).notna() & _finite(merged["policy_net_bps"]).notna()
    if overlap.any() and not _numeric_equal(merged.loc[overlap, "p0_canonical_net_bps"], merged.loc[overlap, "policy_net_bps"]):
        raise AssertionError("rich exact policy labels disagree with source canonical policy net")
    valid = _valid_label(merged)
    if valid.any() and not merged.loc[valid, "__label_available_at__"].eq(merged.loc[valid, "__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise AssertionError("path labels are not available exactly at decision + 12 hours")
    if merged.loc[~valid, ["mfe_3h_bps", "mfe_12h_bps", "policy_net_bps"]].notna().any().any():
        raise AssertionError("invalid paths carry supervised exact-path targets")
    m4, m4_hashes = _load_m4(m4_roots)
    m4 = m4.merge(labels, on=["candidate_id", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    m4["raw_score"] = _finite(m4["raw_meta_score"])
    m4["expected_policy_net_bps"] = _finite(m4["expected_net_bps"])
    m4["train_p80_expected_policy_net_bps"] = _finite(m4["train_p80_expected_bps"])
    m4["diagnostic_target_value"] = np.nan

    out.mkdir(parents=True)
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    months = pd.date_range(start.normalize().replace(day=1), end.normalize().replace(day=1), freq="MS", inclusive="left")

    # T0 is retained exactly as stored.  Its selection threshold is not
    # recomputed, ensuring a genuinely matched historical control.
    for month in months:
        held_t0 = m4.loc[m4["__decision_ts__"].ge(month) & m4["__decision_ts__"].lt(month + pd.offsets.MonthBegin(1))].copy()
        if held_t0.empty:
            continue
        held_t0["arm"] = "T0_frozen_M4_control"
        held_t0["diagnostic_only"] = False
        held_t0["mfe_12h_bps"] = np.nan
        all_predictions.append(held_t0.loc[:, [*IDENTITY, "arm", "raw_score", "expected_policy_net_bps", "train_p80_expected_policy_net_bps", "diagnostic_target_value", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "policy_net_bps", "mfe_12h_bps", "__label_available_at__"]])
        metric_rows.extend(_metric_rows(held_t0, arm="T0_frozen_M4_control", month=month, diagnostic_only=False))
        folds.append({"held_month": month.strftime("%Y-%m"), "arm": "T0_frozen_M4_control", "status": "stored_control", "held_rows": int(len(held_t0))})

    for outer_idx, month in enumerate(months):
        next_month = month + pd.offsets.MonthBegin(1)
        held = merged.loc[merged["__decision_ts__"].ge(month) & merged["__decision_ts__"].lt(next_month)].copy()
        train = merged.loc[
            merged["__decision_ts__"].lt(month)
            & merged["__label_available_at__"].lt(month)
            & valid
        ].copy()
        if held.empty:
            continue
        for spec_idx, spec in enumerate(SPECS):
            try:
                prediction, audit = _fit_predict(train, held, fields, spec, seed=seed + outer_idx * 1000 + spec_idx * 31)
                prediction["arm"] = spec.name
                prediction["diagnostic_only"] = spec.diagnostic_only
                all_predictions.append(prediction.loc[:, [*IDENTITY, "arm", "raw_score", "expected_policy_net_bps", "train_p80_expected_policy_net_bps", "diagnostic_target_value", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", "policy_net_bps", "mfe_12h_bps", "__label_available_at__"]])
                metric_rows.extend(_metric_rows(prediction, arm=spec.name, month=month, diagnostic_only=spec.diagnostic_only))
                folds.append({"held_month": month.strftime("%Y-%m"), "arm": spec.name, "status": "complete", "held_rows": int(len(held)), **audit})
            except ValueError as error:
                folds.append({"held_month": month.strftime("%Y-%m"), "arm": spec.name, "status": "skipped", "reason": str(error), "held_rows": int(len(held)), "train_rows": int(len(train)), "target_domain_train_rows": int(len(_domain(train, spec)))})

    if not all_predictions:
        raise RuntimeError("strict path-target funnel produced no output")
    predictions = pd.concat(all_predictions, ignore_index=True)
    monthly = pd.DataFrame(metric_rows)
    monthly["era"] = _era_of(pd.to_datetime(monthly["held_month"] + "-01", utc=True))
    aggregate = _aggregate(monthly)
    folds_frame = pd.DataFrame(folds)
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "scope": "short P0 rank-1 hourly population; target-only comparison; long and live stacks untouched",
        "entry": "frozen signal-close + one-hour exact decision-minute entry", "horizon": "exact post-decision H12 one-minute path", "cost": "100 bps applied once in policy outcome",
        "training": "expanding monthly strict chronology; fit rows require exact label availability before held-month start",
        "calibration": "same-fold expanding chronological OOF raw score -> isotonic policy-net bps; held outcomes never calibrate held scores",
        "selection": "exact M4 convention: calibrate the train-only p80 raw OOF score, then admit held expected bps at/above that mapped cut; held top-k is not computed or used",
        "invalidity": "invalid/incomplete H12 rows are scored for population coverage only; excluded from supervised fitting and outcome metrics",
        "feature_contract": {"count": len(fields), "fields": list(fields), "forbidden": "all rich exact-path columns are supervised labels only"},
        "targets": [{"name": spec.name, "kind": spec.kind, "description": spec.description, "lambda": spec.lambda_, "diagnostic_only": spec.diagnostic_only} for spec in SPECS],
        "sources": {"population_manifest_hashes": population_hashes, "m4_prediction_hashes": m4_hashes, "rich_label_manifest_sha256": label_hash},
    }
    predictions.to_parquet(out / "path_target_oof_predictions.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    aggregate.to_parquet(out / "cross_era_metrics.parquet", index=False, compression="zstd")
    folds_frame.to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    (out / "feature_contract.json").write_text(json.dumps(manifest["feature_contract"], indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, aggregate=aggregate, fold=folds_frame, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, action="append")
    parser.add_argument("--m4-root", type=Path, action="append")
    parser.add_argument("--rich-label-root", type=Path, required=True)
    parser.add_argument("--repair-from", type=Path, help="sealed target-funnel artifact whose scores are copied while only metric scope is repaired")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.repair_from is not None:
        print(repair_metric_scope(source_artifact=args.repair_from.resolve(), rich_labels_root=args.rich_label_root.resolve(), out=args.out.resolve()))
        return
    if not args.population_root or not args.m4_root:
        parser.error("--population-root and --m4-root are required unless --repair-from is supplied")
    start, end = _utc(args.start), _utc(args.end)
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp) or end <= start:
        raise ValueError("end must follow start")
    print(run(
        population_roots=[path.resolve() for path in args.population_root],
        m4_roots=[path.resolve() for path in args.m4_root],
        rich_labels_root=args.rich_label_root.resolve(), out=args.out.resolve(), start=start, end=end, seed=int(args.seed),
    ))


if __name__ == "__main__":
    main()
