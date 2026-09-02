#!/usr/bin/env python3
"""Sequential strict-OOF supportive-label funnel for the long strict-R3 stack.

This is intentionally the first, *label-quality* stage of the requested path
archetype study.  It answers a falsifiable preliminary question before any
downstream stack integration: do causal predictions of an outcome-clustered
H12 path contain more policy-economic information than directly predicted path
quantities?

P1 outcomes are materialised in a separate target-only sidecar.  For each
three-month outer holdout, every PCA/scaler/clustering prototype, class-policy
prior, causal model, direct-target model and direct score-to-policy map is fit
only on labels resolved before that block.  OOS rows are merely assigned to
frozen prototypes and scored from decision-time fields.  The runner never
adds a future path label to inference features or modifies live artifacts.

Stage 1 arms
------------
* D1 direct MFE / MAE / time / retention / efficiency controls;
* P1 Ward prototype labels, K=4/6/8;
* P3 GMM soft-membership labels, K=4/6/8;
* causal-only, causal-plus-base, and causal-plus-OOF-stack predictors.

K-medoids, HDBSCAN and DTW require unavailable optional libraries and are
written as explicit skipped families rather than silently substituted.  The
next stage may run only if one P1 arm clears the direct-control gate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_long_supportive_label_funnel_stage1_v1"
SIDE = "long"
SEED = 1729
MAX_TRAIN_ROWS = 180_000
MAX_CLUSTER_ROWS = 60_000
# Exact Ward linkage is quadratic in the prototype sample.  The path-label
# study therefore fits its reusable Ward prototypes on a fixed equal-month
# subset of the already bounded, train-only P1 sample; all train/held rows are
# still assigned to those frozen prototypes afterwards.  This is a compute
# proxy, not a change to labels, chronology, or held evaluation.
MAX_WARD_FIT_ROWS = 8_000
MIN_TRAIN_ROWS = 20_000
MIN_CLASS_ROWS = 250
EMBARGO = pd.Timedelta(hours=12)
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_raw15m_strictfull_prior28_20260812_v1/prequential_stack_ledger.parquet"
)
DEFAULT_LABELS = ROOT / "data_perp/artifacts/strict_r3_long_supportive_path_labels_2024_2026_20260823_v5"

CAUSAL_BASE_START = 23
CAUSAL_BASE_END = 143
STACK_FIELDS = (
    "prequential_p_adverse",
    "prequential_p_weak",
    "prequential_p_clear",
    "prequential_base_score",
    "prequential_base_rank42",
    "prequential_base_anchor_bps",
    "prequential_consensus_rank",
    "prequential_residual_rank",
    "prequential_upstream",
)
DIRECT_TARGETS = (
    ("direct_peak_mfe", "supportive_peak_mfe_atr_h12", 1.0),
    ("direct_pre_mfe_mae", "supportive_mae_before_meaningful_atr_h12", -1.0),
    ("direct_time_to_meaningful", "supportive_time_to_meaningful_mfe_h12", -1.0),
    ("direct_final_return", "supportive_final_return_atr_h12", 1.0),
    ("direct_efficiency", "supportive_path_efficiency_h12", 1.0),
)
P1_K_VALUES = (4, 6, 8)
TAILS = (0.01, 0.02, 0.05, 0.10)
# The P1 clustering representation is deliberately compact.  The label
# sidecar retains additional audit diagnostics (cost/ATR ratios, high-threshold
# reach flags, raw path probes), but those can become numerically dominant in
# near-flat ATR states and are not the requested path-shape supervision.
P1_FUTURE_FIELDS = (
    *(f"path_arch_mfe_{h}h_r" for h in ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0", "12.0")),
    *(f"path_arch_mae_{h}h_r" for h in ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0", "12.0")),
    "path_arch_time_to_025r_h", "path_arch_time_to_05r_h", "path_arch_time_to_1r_h",
    "path_arch_time_to_tp_h", "path_arch_time_to_trailing_h", "path_arch_time_to_stop_h",
    "path_arch_time_to_first_meaningful_mfe_h", "path_arch_time_to_90pct_peak_mfe_h",
    "path_arch_mfe_before_mae", "path_arch_mae_before_mfe", "path_arch_peak_mfe_r",
    "path_arch_efficiency", "path_arch_reversal_count", "path_arch_final_return_r",
    "path_arch_final_to_peak", "path_arch_bars_to_meaningful_mfe",
    "path_arch_bars_to_80pct_peak", "path_arch_bars_to_90pct_peak",
    "path_arch_mfe_before_stop_r", "path_arch_mae_before_meaningful_mfe_r",
    "path_arch_path_efficiency_to_meaningful_mfe", "path_arch_path_efficiency_to_90pct_peak",
    "path_arch_peak_retention_ratio", "path_arch_fraction_bars_above_50pct_peak",
    *(f"path_arch_close_return_r_{h}h" for h in ("1", "2", "4", "8", "12")),
    *(f"path_arch_cumulative_variation_r_{h}h" for h in ("1", "2", "4", "8", "12")),
)


@dataclass(frozen=True)
class OuterFold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    cohort: str


FOLDS: tuple[OuterFold, ...] = (
    OuterFold("dev_2025_q2", pd.Timestamp("2025-04-01T00:00:00Z"), pd.Timestamp("2025-07-01T00:00:00Z"), "development"),
    OuterFold("dev_2025_q3", pd.Timestamp("2025-07-01T00:00:00Z"), pd.Timestamp("2025-10-01T00:00:00Z"), "development"),
    OuterFold("holdout_2025_q4", pd.Timestamp("2025-10-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z"), "holdout"),
    OuterFold("oos_2026_q1", pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:00:00Z"), "portability"),
    OuterFold("oos_2026_q2", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z"), "portability"),
    OuterFold("oos_2026_jul", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z"), "portability"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 8:
        return np.nan
    a = pd.Series(left[valid]).rank(method="average").to_numpy(float)
    b = pd.Series(right[valid]).rank(method="average").to_numpy(float)
    if a.std(ddof=0) <= 1e-12 or b.std(ddof=0) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _ledger_fields(ledger: Path) -> tuple[str, ...]:
    names = pq.ParquetFile(ledger).schema.names
    fields = tuple(names[CAUSAL_BASE_START:CAUSAL_BASE_END])
    if len(fields) != 120 or len(set(fields)) != 120:
        raise AssertionError(f"expected frozen 120-field base contract, received {len(fields)} fields")
    return fields


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    yield from pd.date_range(start.normalize().replace(day=1), end.normalize().replace(day=1), freq="MS", inclusive="left")


def _read_labels(root: Path, *, start: pd.Timestamp, end: pd.Timestamp, columns: Sequence[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    requested = list(dict.fromkeys([*IDENTITY, *columns]))
    # Candidate sidecars are partitioned by signal timestamp, while all outer
    # folds are defined by the executable decision timestamp.  The first
    # decision hour of a fold is therefore emitted from the preceding
    # calendar month's signal partition.  Read that one-hour causal boundary
    # explicitly, then filter back to the requested decision interval.
    signal_start = start - pd.Timedelta(hours=1)
    requested_first_month = start.normalize().replace(day=1)
    for month in _month_starts(signal_start, end):
        path = root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
        if not path.exists():
            # The earliest compatible ledger begins at the requested start,
            # so its preceding signal-month can legitimately be absent.  A
            # missing requested-month partition is always a hard failure.
            if month < requested_first_month:
                continue
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path, columns=requested))
    result = pd.concat(parts, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "supportive_label_available_ts"):
        if column in result:
            result[column] = _utc(result[column])
    result = result.loc[
        result["__decision_ts__"].ge(start) & result["__decision_ts__"].lt(end)
    ].copy()
    if result["candidate_id"].duplicated().any():
        raise AssertionError("supportive label sidecar has duplicate candidate IDs")
    return result


def _read_population(
    ledger: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
) -> pd.DataFrame:
    cols = [
        *IDENTITY,
        "r3_label_available_ts", "policy_path_valid", "policy_label_available_ts",
        "policy_gross_bps", "policy_net_bps", "h12_tp6_sl4_net_bps",
        "base_contract_complete", "base_feature_available_fraction",
        *fields, *STACK_FIELDS,
    ]
    result = pd.read_parquet(
        ledger, columns=list(dict.fromkeys(cols)),
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    ).copy()
    for column in ("__ts__", "__decision_ts__", "r3_label_available_ts", "policy_label_available_ts"):
        result[column] = _utc(result[column])
    if result["candidate_id"].duplicated().any():
        raise AssertionError("prequential ledger has duplicate candidate IDs in a requested interval")
    if not result["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError("supportive funnel received non-long rows")
    return result


def _joined_population(
    ledger: Path,
    labels_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fields: Sequence[str],
    p1_fields: Sequence[str],
) -> pd.DataFrame:
    label_columns = [
        "supportive_label_available_ts", "supportive_path_valid", "supportive_target_invalid",
        *p1_fields, *(column for _, column, _ in DIRECT_TARGETS),
        "policy_net_bps", "policy_gross_bps", "h12_tp6_sl4_net_bps",
    ]
    population = _read_population(ledger, start=start, end=end, fields=fields)
    labels = _read_labels(labels_root, start=start, end=end, columns=label_columns)
    identity_check = labels.loc[:, list(IDENTITY)].rename(
        columns={column: f"supportive_{column}" for column in IDENTITY if column != "candidate_id"}
    )
    population = population.merge(identity_check, on="candidate_id", how="left", validate="one_to_one")
    for column in IDENTITY[1:]:
        paired = f"supportive_{column}"
        if population[paired].isna().any() or not population[column].eq(population[paired]).all():
            raise AssertionError(f"candidate identity mismatch between ledger and supportive labels: {column}")
    population = population.drop(columns=[f"supportive_{column}" for column in IDENTITY[1:]])
    suffix = labels.drop(columns=[column for column in IDENTITY if column != "candidate_id"])
    labels_policy = suffix.pop("policy_net_bps").rename("supportive_policy_net_bps")
    labels_gross = suffix.pop("policy_gross_bps").rename("supportive_policy_gross_bps")
    labels_tp6 = suffix.pop("h12_tp6_sl4_net_bps").rename("supportive_h12_tp6_sl4_net_bps")
    suffix = pd.concat([suffix, labels_policy, labels_gross, labels_tp6], axis=1)
    result = population.merge(suffix, on="candidate_id", how="left", validate="one_to_one")
    if result["supportive_path_valid"].isna().any():
        raise AssertionError("candidate-to-supportive-label identity coverage failed")
    # The policy outcome has two source representations.  Exact equality is a
    # hard guard against accidental label geometry mixing in this experiment.
    left, right = _finite(result["policy_net_bps"]), _finite(result["supportive_policy_net_bps"])
    both = left.notna() & right.notna()
    if both.any() and not np.isclose(left[both], right[both], rtol=0.0, atol=2e-4).all():
        raise AssertionError("supportive sidecar policy net differs from the prequential ledger")
    return result


def _train_path_eligible(frame: pd.DataFrame, *, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rows eligible to fit a future-path target, independently of policy labels."""
    valid = (
        frame["supportive_path_valid"].fillna(False).astype(bool)
        & ~frame["supportive_target_invalid"].fillna(True).astype(bool)
        & _finite(frame["base_feature_available_fraction"]).ge(0.90)
        & frame["base_contract_complete"].fillna(False).astype(bool)
    )
    if cutoff is not None:
        valid &= frame["supportive_label_available_ts"].lt(cutoff - EMBARGO)
        valid &= frame["r3_label_available_ts"].lt(cutoff - EMBARGO)
    return frame.loc[valid].copy()


def _score_eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time scoring population; it must never require a future label."""
    valid = (
        _finite(frame["base_feature_available_fraction"]).ge(0.90)
        & frame["base_contract_complete"].fillna(False).astype(bool)
    )
    return frame.loc[valid].copy()


def _sample_month_balanced(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    local = frame.copy()
    local["__month__"] = local["__decision_ts__"].dt.to_period("M").astype(str)
    groups = list(local.groupby("__month__", sort=True, observed=True))
    each = max(1, cap // len(groups))
    chunks: list[pd.DataFrame] = []
    for index, (_, group) in enumerate(groups):
        if len(group) <= each:
            chunks.append(group)
        else:
            # Candidate identity makes the sample deterministic without
            # allowing future outcomes to influence membership.
            key = pd.util.hash_pandas_object(group["candidate_id"], index=False).to_numpy(np.uint64)
            order = np.argsort(key ^ np.uint64(seed + index), kind="stable")[:each]
            chunks.append(group.iloc[order])
    result = pd.concat(chunks, ignore_index=True)
    if len(result) < cap:
        remainder = local.loc[~local["candidate_id"].isin(result["candidate_id"])].copy()
        key = pd.util.hash_pandas_object(remainder["candidate_id"], index=False).to_numpy(np.uint64)
        result = pd.concat([result, remainder.iloc[np.argsort(key, kind="stable")[: cap - len(result)]]], ignore_index=True)
    return result.drop(columns="__month__", errors="ignore")


def _matrix(frame: pd.DataFrame, fields: Sequence[str], *, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    numeric = frame.loc[:, list(fields)].apply(_finite)
    if medians is None:
        medians = numeric.median(axis=0).fillna(0.0)
    matrix = numeric.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
    return matrix, medians


def _p1_matrix(frame: pd.DataFrame, fields: Sequence[str], *, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    return _matrix(frame, fields, medians=medians)


def _fit_p1_transform(train: pd.DataFrame, p1_fields: Sequence[str]) -> tuple[RobustScaler, PCA, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    sample = _sample_month_balanced(train, MAX_CLUSTER_ROWS, seed=SEED)
    raw, medians = _p1_matrix(sample, p1_fields)
    # Fit clipping bounds only on the train-only P1 sample.  This keeps rare
    # zero-ATR path ratios from defining an archetype while preserving their
    # uncapped values in the target sidecar for audits.
    lower = np.nanquantile(raw, 0.005, axis=0).astype(np.float32)
    upper = np.nanquantile(raw, 0.995, axis=0).astype(np.float32)
    raw = np.clip(raw, lower, upper)
    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    scaled = scaler.fit_transform(raw)
    components = min(16, scaled.shape[1], max(2, scaled.shape[0] - 1))
    pca = PCA(n_components=components, random_state=SEED, svd_solver="randomized")
    sample_pca = pca.fit_transform(scaled).astype(np.float32)
    return scaler, pca, medians, sample_pca, lower, upper


def _transform_p1(frame: pd.DataFrame, fields: Sequence[str], scaler: RobustScaler, pca: PCA, medians: pd.Series, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values, _ = _p1_matrix(frame, fields, medians=medians)
    values = np.clip(values, lower, upper)
    return pca.transform(scaler.transform(values)).astype(np.float32)


def _prototype_assign(values: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    squared = ((values[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    return squared.argmin(axis=1).astype(np.int16)


def _fit_ward(train_pca: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray]:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(train_pca).astype(np.int16)
    prototypes = np.vstack([np.median(train_pca[labels == index], axis=0) for index in range(k)]).astype(np.float32)
    return labels, prototypes


def _fit_gmm(train_pca: np.ndarray, *, k: int) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    model = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, random_state=SEED, max_iter=200, n_init=2)
    model.fit(train_pca)
    probabilities = model.predict_proba(train_pca).astype(np.float32)
    labels = probabilities.argmax(axis=1).astype(np.int16)
    return model, labels, probabilities


def _model_classifier(*, classes: int, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=classes, n_estimators=220, learning_rate=0.035,
        max_depth=4, num_leaves=31, min_child_samples=180, subsample=0.85,
        colsample_bytree=0.80, reg_lambda=6.0, reg_alpha=0.05,
        class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _model_regressor(*, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="huber", alpha=0.90, n_estimators=220, learning_rate=0.035,
        max_depth=4, num_leaves=31, min_child_samples=180, subsample=0.85,
        colsample_bytree=0.80, reg_lambda=6.0, reg_alpha=0.05,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _class_policy_priors(train: pd.DataFrame, labels: np.ndarray, *, k: int) -> np.ndarray:
    actual = _finite(train["policy_net_bps"]).to_numpy(float)
    policy_valid = np.isfinite(actual)
    if not policy_valid.any():
        raise AssertionError("path classifier has no resolved train-only policy outcomes for its value map")
    global_mean = float(np.mean(actual[policy_valid]))
    values = np.full(k, global_mean, dtype=np.float64)
    for index in range(k):
        mask = (labels == index) & policy_valid
        support = int(mask.sum())
        if support:
            # A fixed 250-row prior prevents tiny archetypes from manufacturing
            # extreme expected EV and remains entirely train-only.
            values[index] = (actual[mask].sum() + MIN_CLASS_ROWS * global_mean) / (support + MIN_CLASS_ROWS)
    return values


def _align_probabilities(model: LGBMClassifier, probability: np.ndarray, *, k: int) -> np.ndarray:
    output = np.zeros((len(probability), k), dtype=np.float32)
    for column, label in enumerate(model.classes_.astype(int)):
        if 0 <= label < k:
            output[:, label] = probability[:, column]
    missing = output.sum(axis=1) <= 0.0
    if missing.any():
        output[missing] = 1.0 / k
    return output / output.sum(axis=1, keepdims=True)


def _chronological_direct_oof(
    train: pd.DataFrame,
    fields: Sequence[str],
    target: str,
    direction: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train-only chronological OOF scores for a direct target map."""
    ordered = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    raw = np.full(len(ordered), np.nan, dtype=float)
    policy = _finite(ordered["policy_net_bps"]).to_numpy(float)
    boundaries = np.linspace(0, len(ordered), 5, dtype=int)
    for fold in range(3):
        fit_end, valid_end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        if fit_end < MIN_TRAIN_ROWS // 3 or valid_end <= fit_end:
            continue
        fit, valid = ordered.iloc[:fit_end], ordered.iloc[fit_end:valid_end]
        target_fit = _finite(fit[target]).to_numpy(float)
        usable = np.isfinite(target_fit)
        if usable.sum() < 1_000:
            continue
        x_fit, medians = _matrix(fit.loc[usable], fields)
        x_valid, _ = _matrix(valid, fields, medians=medians)
        model = _model_regressor(seed=seed + fold)
        model.fit(x_fit, target_fit[usable])
        raw[fit_end:valid_end] = direction * model.predict(x_valid)
    return raw, policy


def _direct_score(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    target: str,
    direction: float,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    oof, oof_policy = _chronological_direct_oof(train, fields, target, direction, seed=seed)
    usable = np.isfinite(oof) & np.isfinite(oof_policy)
    if usable.sum() < 2_000 or np.unique(oof[usable]).size < 10:
        return np.full(len(held), np.nan), {"direct_map_rows": int(usable.sum()), "status": "insufficient_oof"}
    mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
    mapper.fit(oof[usable], oof_policy[usable])
    target_fit = _finite(train[target]).to_numpy(float)
    valid_fit = np.isfinite(target_fit)
    x_fit, medians = _matrix(train.loc[valid_fit], fields)
    x_held, _ = _matrix(held, fields, medians=medians)
    model = _model_regressor(seed=seed + 100)
    model.fit(x_fit, target_fit[valid_fit])
    return mapper.predict(direction * model.predict(x_held)), {"direct_map_rows": int(usable.sum()), "status": "ok"}


def _quality_metrics(
    *,
    fold: OuterFold,
    arm: str,
    feature_mode: str,
    score: np.ndarray,
    held: pd.DataFrame,
    label: np.ndarray | None = None,
    probability: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    actual = _finite(held["policy_net_bps"]).to_numpy(float)
    base = _finite(held["prequential_upstream"]).to_numpy(float)
    rows: list[dict[str, Any]] = []
    common = {
        "fold": fold.name, "cohort": fold.cohort, "start": fold.start, "end_exclusive": fold.end,
        "arm": arm, "feature_mode": feature_mode, "held_rows": int(len(held)),
        "score_policy_spearman": _spearman(score, actual),
        "score_policy_residual_spearman": _spearman(score, actual - base),
    }
    if probability is not None and label is not None:
        valid = np.isfinite(probability).all(axis=1) & np.isfinite(actual) & held["supportive_path_valid"].fillna(False).to_numpy(bool)
        classes = np.unique(label)
        if valid.sum() and len(classes) > 1:
            try:
                common["label_ovr_auc"] = float(roc_auc_score(label[valid], probability[valid], multi_class="ovr", average="macro"))
            except ValueError:
                common["label_ovr_auc"] = np.nan
            truth = (actual >= 50.0).astype(int)
            common["policy_ge50_ap"] = float(average_precision_score(truth[valid], score[valid])) if len(np.unique(truth[valid])) > 1 else np.nan
    for tail in TAILS:
        valid = np.isfinite(score) & np.isfinite(actual)
        count = max(1, int(np.ceil(tail * valid.sum()))) if valid.any() else 0
        order = np.argsort(score[valid], kind="stable")[-count:]
        selected = actual[valid][order] if valid.sum() else np.array([], dtype=float)
        rows.append({
            **common, "metric": f"top_{tail:.0%}_net_ev_bps", "tail": tail, "selected_rows": int(len(selected)),
            "value": float(selected.mean()) if len(selected) else np.nan,
            "net_sum_bps": float(selected.sum()) if len(selected) else np.nan,
            "policy_ge50_fraction": float((selected >= 50.0).mean()) if len(selected) else np.nan,
        })
    rows.append({**common, "metric": "global_score_policy_spearman", "tail": np.nan, "selected_rows": int(np.isfinite(score).sum()), "value": common["score_policy_spearman"], "net_sum_bps": np.nan, "policy_ge50_fraction": np.nan})
    rows.append({**common, "metric": "global_score_policy_residual_spearman", "tail": np.nan, "selected_rows": int(np.isfinite(score).sum()), "value": common["score_policy_residual_spearman"], "net_sum_bps": np.nan, "policy_ge50_fraction": np.nan})
    return rows


def _prototype_table(*, fold: OuterFold, family: str, k: int, labels: np.ndarray, train: pd.DataFrame, p1_fields: Sequence[str], priors: np.ndarray) -> pd.DataFrame:
    records = []
    for index in range(k):
        subset = train.loc[labels == index]
        record: dict[str, Any] = {
            "fold": fold.name, "family": family, "k": k, "cluster": index,
            "support": int(len(subset)), "support_fraction": float(len(subset) / max(len(train), 1)),
            "train_policy_net_bps_shrunk": float(priors[index]),
        }
        for column in (
            "supportive_peak_mfe_atr_h12", "supportive_mae_before_meaningful_atr_h12",
            "supportive_time_to_meaningful_mfe_h12", "supportive_final_return_atr_h12",
            "supportive_path_efficiency_h12", "policy_net_bps",
        ):
            record[f"mean_{column}"] = float(_finite(subset[column]).mean())
        records.append(record)
    return pd.DataFrame(records)


def _run_cluster_arm(
    *,
    family: str,
    k: int,
    train: pd.DataFrame,
    held: pd.DataFrame,
    p1_fields: Sequence[str],
    predictor_fields: Sequence[str],
    fold: OuterFold,
    feature_mode: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    scaler, pca, medians, sample_pca, lower, upper = _fit_p1_transform(train, p1_fields)
    # fit transforms on a bounded, deterministic, train-only sample, then
    # assign all train/held rows to its frozen representation.
    train_pca = _transform_p1(train, p1_fields, scaler, pca, medians, lower, upper)
    held_pca = _transform_p1(held, p1_fields, scaler, pca, medians, lower, upper)
    if family == "ward":
        if len(sample_pca) > MAX_WARD_FIT_ROWS:
            ward_indices = np.linspace(0, len(sample_pca) - 1, MAX_WARD_FIT_ROWS, dtype=int)
            ward_sample = sample_pca[ward_indices]
        else:
            ward_sample = sample_pca
        sample_labels, prototypes = _fit_ward(ward_sample, k=k)
        labels = _prototype_assign(train_pca, prototypes)
        held_realised_label = _prototype_assign(held_pca, prototypes)
        train_membership = np.eye(k, dtype=np.float32)[labels]
    elif family == "gmm":
        gmm, _, _ = _fit_gmm(sample_pca, k=k)
        train_membership = gmm.predict_proba(train_pca).astype(np.float32)
        labels = train_membership.argmax(axis=1).astype(np.int16)
        held_realised_label = gmm.predict(held_pca).astype(np.int16)
    else:
        raise ValueError(family)
    priors = _class_policy_priors(train, labels, k=k)
    train_x, x_medians = _matrix(train, predictor_fields)
    held_x, _ = _matrix(held, predictor_fields, medians=x_medians)
    classifier = _model_classifier(classes=k, seed=SEED + k)
    classifier.fit(train_x, labels)
    probability = _align_probabilities(classifier, classifier.predict_proba(held_x), k=k)
    score = probability @ priors
    arm = f"P_{family}_k{k}"
    metrics = _quality_metrics(fold=fold, arm=arm, feature_mode=feature_mode, score=score, held=held, label=held_realised_label, probability=probability)
    prediction = pd.DataFrame({
        "candidate_id": held["candidate_id"].to_numpy(), "__decision_ts__": held["__decision_ts__"].to_numpy(),
        "fold": fold.name, "cohort": fold.cohort, "arm": arm, "feature_mode": feature_mode,
        "predicted_policy_net_bps": score.astype(np.float32), "realised_policy_net_bps": _finite(held["policy_net_bps"]).to_numpy(np.float32),
        "realised_cluster": held_realised_label.astype(np.int16), "predicted_cluster": probability.argmax(axis=1).astype(np.int16),
        "path_entropy": (-np.clip(probability, 1e-9, 1.0) * np.log(np.clip(probability, 1e-9, 1.0))).sum(axis=1).astype(np.float32),
        "path_max_probability": probability.max(axis=1).astype(np.float32),
    })
    for index in range(k):
        prediction[f"path_p_{index:02d}"] = probability[:, index]
    prototypes = _prototype_table(fold=fold, family=family, k=k, labels=labels, train=train, p1_fields=p1_fields, priors=priors)
    return metrics, prediction, prototypes


def _baseline_metrics(fold: OuterFold, held: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    score = _finite(held["prequential_upstream"]).to_numpy(float)
    metrics = _quality_metrics(fold=fold, arm="B0_prequential_upstream", feature_mode="frozen_stack", score=score, held=held)
    predictions = pd.DataFrame({
        "candidate_id": held["candidate_id"].to_numpy(), "__decision_ts__": held["__decision_ts__"].to_numpy(),
        "fold": fold.name, "cohort": fold.cohort, "arm": "B0_prequential_upstream", "feature_mode": "frozen_stack",
        "predicted_policy_net_bps": score.astype(np.float32), "realised_policy_net_bps": _finite(held["policy_net_bps"]).to_numpy(np.float32),
    })
    return metrics, predictions


def _run_direct_controls(
    *, train: pd.DataFrame, held: pd.DataFrame, predictor_fields: Sequence[str], fold: OuterFold, feature_mode: str
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for index, (name, target, direction) in enumerate(DIRECT_TARGETS):
        score, extra = _direct_score(train, held, predictor_fields, target, direction, seed=SEED + 1000 + index * 31)
        metrics.extend(_quality_metrics(fold=fold, arm=f"D_{name}", feature_mode=feature_mode, score=score, held=held))
        predictions.append(pd.DataFrame({
            "candidate_id": held["candidate_id"].to_numpy(), "__decision_ts__": held["__decision_ts__"].to_numpy(),
            "fold": fold.name, "cohort": fold.cohort, "arm": f"D_{name}", "feature_mode": feature_mode,
            "predicted_policy_net_bps": score.astype(np.float32), "realised_policy_net_bps": _finite(held["policy_net_bps"]).to_numpy(np.float32),
            "direct_map_rows": extra["direct_map_rows"], "status": extra["status"],
        }))
    return metrics, predictions


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    tail = metrics.loc[metrics["metric"].isin(("top_1%_net_ev_bps", "top_5%_net_ev_bps", "global_score_policy_residual_spearman"))].copy()
    if tail.empty:
        return tail
    return tail.groupby(["arm", "feature_mode", "cohort", "metric"], as_index=False).agg(
        mean_value=("value", "mean"), median_value=("value", "median"), worst_value=("value", "min"), folds=("fold", "nunique"),
    )


def run(*, ledger: Path, labels_root: Path, out: Path, max_train_rows: int = MAX_TRAIN_ROWS) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    labels_manifest = labels_root / "run_manifest.json"
    if not labels_manifest.exists():
        raise FileNotFoundError(labels_manifest)
    source_fields = _ledger_fields(ledger)
    first_part = next(iter(sorted(labels_root.glob("parts/month=*/side=long.parquet"))), None)
    if first_part is None:
        raise FileNotFoundError("no supportive-label parts")
    label_names = pq.ParquetFile(first_part).schema.names
    p1_fields = tuple(column for column in P1_FUTURE_FIELDS if column in label_names)
    if len(p1_fields) < 20:
        raise AssertionError(f"insufficient P1 fields: {len(p1_fields)}")
    all_metrics: list[dict[str, Any]] = []
    all_prototypes: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    prediction_root = out / "stage1_oof_prediction_parts"
    prediction_root.mkdir(parents=True, exist_ok=False)
    history_start = pd.Timestamp("2024-01-01T00:00:00Z")
    for fold_index, fold in enumerate(FOLDS):
        train_raw = _joined_population(ledger, labels_root, start=history_start, end=fold.start, fields=source_fields, p1_fields=p1_fields)
        held_raw = _joined_population(ledger, labels_root, start=fold.start, end=fold.end, fields=source_fields, p1_fields=p1_fields)
        # Fit path targets on every complete observed H12 path; map to policy
        # net only through the resolved subset inside each train-only map.
        # Score every point-in-time feature-complete held candidate, with
        # outcomes joined solely for evaluation after predictions are formed.
        train = _train_path_eligible(train_raw, cutoff=fold.start)
        held = _score_eligible(held_raw)
        # Eligibility functions make independent copies.  The full raw panels are
        # hundreds of fields wide and no longer needed after this point.
        del train_raw, held_raw
        gc.collect()
        if len(train) < MIN_TRAIN_ROWS or len(held) < 5_000:
            fold_audit.append({"fold": fold.name, "status": "insufficient_support", "train_rows": len(train), "held_rows": len(held)})
            continue
        train = _sample_month_balanced(train, max_train_rows, seed=SEED + fold_index)
        fold_audit.append({"fold": fold.name, "status": "ok", "train_rows": len(train), "held_rows": len(held), "train_label_cutoff": fold.start, "embargo_hours": 12})
        fold_prediction_dir = prediction_root / f"fold={fold_index:02d}_{fold.name}"
        fold_prediction_dir.mkdir(parents=True, exist_ok=False)
        prediction_part = 0
        metrics, predictions = _baseline_metrics(fold, held)
        all_metrics.extend(metrics)
        predictions.to_parquet(fold_prediction_dir / f"part={prediction_part:03d}.parquet", index=False, compression="zstd")
        prediction_part += 1
        del predictions
        feature_modes = {
            "causal120": source_fields,
            "causal120_plus_base": (*source_fields, "prequential_p_adverse", "prequential_p_weak", "prequential_p_clear", "prequential_base_score", "prequential_base_rank42"),
            "causal120_plus_oof_stack": (*source_fields, *STACK_FIELDS),
        }
        for mode, fields in feature_modes.items():
            direct_metrics, direct_predictions = _run_direct_controls(train=train, held=held, predictor_fields=fields, fold=fold, feature_mode=mode)
            all_metrics.extend(direct_metrics)
            for prediction in direct_predictions:
                prediction.to_parquet(fold_prediction_dir / f"part={prediction_part:03d}.parquet", index=False, compression="zstd")
                prediction_part += 1
            del direct_predictions
            gc.collect()
            for family in ("ward", "gmm"):
                for k in P1_K_VALUES:
                    metrics, prediction, prototypes = _run_cluster_arm(
                        family=family, k=k, train=train, held=held, p1_fields=p1_fields,
                        predictor_fields=fields, fold=fold, feature_mode=mode,
                    )
                    all_metrics.extend(metrics)
                    prediction.to_parquet(fold_prediction_dir / f"part={prediction_part:03d}.parquet", index=False, compression="zstd")
                    prediction_part += 1
                    all_prototypes.append(prototypes)
                    del prediction, prototypes
                    gc.collect()
        del train, held
        gc.collect()
        print(json.dumps(fold_audit[-1], default=str, sort_keys=True), flush=True)
    metrics_frame = pd.DataFrame(all_metrics)
    prototypes_frame = pd.concat(all_prototypes, ignore_index=True) if all_prototypes else pd.DataFrame()
    metrics_frame.to_parquet(out / "stage1_metrics.parquet", index=False, compression="zstd")
    (out / "stage1_oof_predictions_manifest.json").write_text(json.dumps({
        "format": "partitioned_parquet", "root": str(prediction_root),
        "parts": [str(item.relative_to(prediction_root)) for item in sorted(prediction_root.rglob("*.parquet"))],
    }, indent=2) + "\n")
    prototypes_frame.to_parquet(out / "stage1_train_only_prototypes.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_audit).to_parquet(out / "stage1_fold_audit.parquet", index=False, compression="zstd")
    _aggregate(metrics_frame).to_parquet(out / "stage1_summary.parquet", index=False, compression="zstd")
    skipped = pd.DataFrame([
        {"family": "P2_kmedoids", "status": "not_run_stage1", "reason": "optional sklearn-extra absent; run only if P1 clears direct-control gate"},
        {"family": "P4_hdbscan", "status": "not_run_stage1", "reason": "optional hdbscan absent; run only if P1 clears direct-control gate"},
        {"family": "P5_dtw_kmedoids", "status": "not_run_stage1", "reason": "optional tslearn absent and P2 shape data is not yet warranted"},
    ])
    skipped.to_parquet(out / "stage1_deferred_families.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline long-only supportive-label research; no inference/live mutation",
        "side": SIDE,
        "ledger": str(ledger.resolve()), "ledger_sha256": _sha256(ledger),
        "labels_root": str(labels_root.resolve()), "labels_manifest_sha256": _sha256(labels_manifest),
        "outer_folds": [{"name": fold.name, "start": str(fold.start), "end_exclusive": str(fold.end), "cohort": fold.cohort} for fold in FOLDS],
        "purge_embargo": "H12 path labels and R3 labels strictly before fold start minus 12h embargo; policy-net maps use only finite resolved policy outcomes",
        "causal_features": list(source_fields),
        "p1_future_labels": list(p1_fields),
        "direct_future_labels": list(DIRECT_TARGETS),
        "families": ["D1_direct", "P1_Ward", "P3_GMM"],
        "frozen_cluster_contract": "scaler/PCA/clustering and outcome priors fit only on sampled train rows; held rows assigned/predicted from frozen state",
        "inference_contract": "all feature-complete held candidates are scored before any outcome join; only causal classifier probabilities and train-derived expected-policy summaries could be downstream inputs; raw P1 and direct target values are prohibited",
        "deferred": skipped.to_dict("records"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS)
    args = parser.parse_args()
    print(run(ledger=args.ledger.resolve(), labels_root=args.labels_root.resolve(), out=args.out.resolve(), max_train_rows=args.max_train_rows))


if __name__ == "__main__":
    main()
