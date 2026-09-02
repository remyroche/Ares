#!/usr/bin/env python3
"""Strict-prequential short P0 opportunity × conversion research challenger.

The runner deliberately factorises the two questions that the prior short
target funnel showed should not be collapsed:

* O: ``P(MFE_H12 > 200 bps | causal state)``;
* C: low-regret conversion quality conditional on that opportunity occurring.

Every target-free P0 rank-1 candidate receives both O and C predictions.  The
realised opportunity label limits *training* for C only; it is never an
inference route, candidate filter, feature, or admission condition.

For every held calendar month the outer model uses only labels available before
the month starts.  The final mapper is fitted only on inner chronological OOF
O/C predictions from that same outer training population.  This prevents both
same-fold base stacking and a future-MFE admission leak.

This is a research-only short-side challenger.  It does not alter the live
long-only stack, short execution, or any canonical policy bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


SIDE = "short"
SCHEMA = "strict_r3_short_p0_two_stage_opportunity_conversion_v1"
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
CORE_SOURCE = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
)
MFE_OPPORTUNITY_BPS = 200.0
POLICY_CLIP_BPS = 500.0
MIN_OUTER_TRAIN_ROWS = 800
MIN_C_POSITIVES = 500
MIN_MAPPER_OOF_ROWS = 1000
MIN_MAPPER_MONTHS = 3
INNER_SPLITS = 3
P80 = 0.80
MDA_START = pd.Timestamp("2024-05-01T00:00:00Z")
MDA_END = pd.Timestamp("2025-01-01T00:00:00Z")
MDA_SEED = 1729
T5_REGRET_EDGES = np.asarray((50.0, 100.0, 200.0, 400.0), dtype=float)
ORDINAL_OPP_EDGES = np.asarray((0.0, 100.0, 200.0, 400.0, 600.0), dtype=float)

DEFAULT_POPULATION_ROOTS = (
    Path("data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2024_maydec_20260821_v1"),
    Path("data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2025_h1_20260821_v1"),
    Path("data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2026_janjul_20260821_v1"),
)
DEFAULT_RICH_LABELS = Path("data_perp/artifacts/strict_r3_short_p0_rich_path_labels_apr2024_jul2026_20260821_v1")
DEFAULT_T5_ARTIFACT = Path("data_perp/artifacts/strict_r3_short_p0_path_target_funnel_2024may_2026jul_20260821_v4")
DEFAULT_F90_SELECTION = Path("data_perp/artifacts/strict_r3_short_policy_conversion_p1_k32_chronological_mda_20260820_v1/selected_features.json")
DEFAULT_FEATURE_PANELS = (
    Path("data_perp/artifacts/strict_r3_short_features_full2024_20260820_v1/canonical120_features.parquet"),
    Path("data_perp/artifacts/strict_r3_short_p0_f90_targetfree_2025_aug2026_20260821_v2/canonical120_features.parquet"),
    Path("data_perp/artifacts/strict_r3_short_p0_f90_features_late_july2026_20260821_v1/canonical120_features.parquet"),
)


@dataclass(frozen=True)
class OpportunityContract:
    name: str
    fields: tuple[str, ...]
    kind: str = "binary"


@dataclass(frozen=True)
class EmpiricalCDF:
    values: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        if len(self.values) == 0:
            return np.full(len(values), 0.5, dtype=float)
        return np.searchsorted(self.values, values, side="right").astype(float) / float(len(self.values))


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


def _finite(values: pd.Series | np.ndarray | Iterable[float]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _numeric_equal(left: pd.Series, right: pd.Series, *, atol: float = 2e-4) -> bool:
    return bool(np.isclose(
        pd.to_numeric(left, errors="coerce").to_numpy(float),
        pd.to_numeric(right, errors="coerce").to_numpy(float),
        rtol=0.0, atol=atol, equal_nan=True,
    ).all())


def _deduplicate_population_frame(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """Permit cumulative immutable source copies but reject value disagreement."""
    repeated_mask = frame["candidate_id"].duplicated(keep=False)
    if not repeated_mask.any():
        return frame.copy()
    repeated = frame.loc[repeated_mask].sort_values(["candidate_id", "__decision_ts__"], kind="stable")
    group = repeated.groupby("candidate_id", sort=False)
    # The three immutable sources have substantial cumulative overlap.  Validate
    # it column-wise rather than with a Python loop over identities: the latter
    # performs roughly one million tiny allocations without adding protection.
    for column in (*CORE_SOURCE[1:], *fields):
        values = repeated[column]
        first = group[column].transform("first")
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            equal = np.isclose(
                pd.to_numeric(values, errors="coerce").to_numpy(float),
                pd.to_numeric(first, errors="coerce").to_numpy(float),
                rtol=0.0, atol=2e-4, equal_nan=True,
            )
        else:
            equal = (values.eq(first) | (values.isna() & first.isna())).to_numpy(bool)
        if not bool(np.all(equal)):
            candidate_id = str(repeated.loc[~equal, "candidate_id"].iloc[0])
            raise AssertionError(f"cumulative P0 sources disagree on {column} for {candidate_id}")
    return pd.concat(
        [frame.loc[~repeated_mask], repeated.drop_duplicates("candidate_id", keep="first")],
        ignore_index=True,
    )


def _load_m4_fields(root: Path) -> tuple[str, ...]:
    path = root / "feature_contract.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    fields = tuple(payload.get("base", ()))
    if len(fields) != 41 or len(set(fields)) != 41:
        raise AssertionError("the matched M4 control must retain its frozen 41-field contract")
    return fields


def _load_population(roots: Sequence[Path], m4_fields: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    columns = [*CORE_SOURCE, *m4_fields]
    for root in roots:
        population = root / "short_p0_top1_hourly_population.parquet"
        manifest = root / "run_manifest.json"
        if not population.exists() or not manifest.exists():
            raise FileNotFoundError(f"not an immutable P0 source: {root}")
        frame = pd.read_parquet(population, columns=columns)
        for column in ("__ts__", "__decision_ts__", "policy_label_available_at"):
            frame[column] = _utc(frame[column])
        if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
            raise AssertionError(f"non-short candidate source: {root}")
        pieces.append(frame)
        hashes[str(root.resolve())] = _sha256(manifest)
    result = _deduplicate_population_frame(pd.concat(pieces, ignore_index=True), m4_fields)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("candidate deduplication failed")
    if not result["__decision_ts__"].eq(result["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("P0 decision convention is not signal close + one hour")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), hashes


def _load_rich_labels(root: Path) -> tuple[pd.DataFrame, str]:
    manifest = root / "run_manifest.json"
    columns = [
        *IDENTITY, "__label_available_at__", "rich_path_label_valid", "rich_path_target_invalid",
        "mfe_12h_bps", "policy_regret_bps", "policy_net_bps", "policy_gross_bps",
        "policy_exit_reason", "policy_stop_out",
    ]
    parts = sorted(root.glob("parts/month=*/side=short.parquet"))
    if not manifest.exists() or not parts:
        raise FileNotFoundError(f"missing rich short path labels beneath {root}")
    frame = pd.concat([pd.read_parquet(part, columns=columns) for part in parts], ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = _utc(frame[column])
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError("rich labels must be unique short candidate identities")
    return frame, _sha256(manifest)


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["rich_path_label_valid"].fillna(False).astype(bool)
        & ~frame["rich_path_target_invalid"].fillna(True).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["policy_net_bps"]).notna()
        & frame["__label_available_at__"].notna()
    )


def _read_feature_panel(panel: Path, candidate_ids: set[str], fields: tuple[str, ...]) -> pd.DataFrame:
    if not panel.exists():
        raise FileNotFoundError(panel)
    available = set(ds.dataset(panel, format="parquet").schema.names)
    required = {"candidate_id", "__ts__", "__symbol__", "side_name", *fields}
    missing = sorted(required - available)
    if missing:
        raise AssertionError(f"feature panel lacks required F115 fields: {missing}")
    table = ds.dataset(panel, format="parquet").to_table(
        columns=["candidate_id", "__ts__", "__symbol__", "side_name", *fields],
        filter=pc.field("candidate_id").isin(sorted(candidate_ids)),
    )
    result = table.to_pandas()
    result["__ts__"] = _utc(result["__ts__"])
    if not result["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError(f"non-short feature rows in {panel}")
    return result


def _load_f115_features(population: pd.DataFrame, fields: tuple[str, ...], panels: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read only rank-1 candidate rows from the target-free F115 panels."""
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    ids = set(population["candidate_id"].astype(str))
    for panel in panels:
        part = _read_feature_panel(panel, ids, fields)
        parts.append(part)
        hashes[str(panel.resolve())] = _sha256(panel)
    values = pd.concat(parts, ignore_index=True)
    duplicate = values["candidate_id"].duplicated(keep=False)
    if duplicate.any():
        repeated = values.loc[duplicate].sort_values("candidate_id", kind="stable")
        for candidate_id, block in repeated.groupby("candidate_id", sort=False):
            for field in fields:
                if not _numeric_equal(block[field], pd.Series(np.repeat(block[field].iloc[0], len(block)))):
                    raise AssertionError(f"feature panels disagree for {candidate_id}/{field}")
        values = pd.concat([values.loc[~duplicate], repeated.drop_duplicates("candidate_id", keep="first")], ignore_index=True)
    if values["candidate_id"].duplicated().any():
        raise AssertionError("feature panel candidate identity is non-unique")
    missing_ids = ids - set(values["candidate_id"].astype(str))
    if missing_ids:
        sample = sorted(missing_ids)[:8]
        raise AssertionError(f"F115 feature source misses {len(missing_ids)} P0 candidates; e.g. {sample}")
    return values.loc[:, ["candidate_id", *fields]], hashes


def _load_control_predictions(roots: Sequence[Path]) -> pd.DataFrame:
    """Load the frozen same-model P0 anchor (M0) and M4 control.

    The controls are immutable outputs of the prior absolute-conversion funnel.
    Keeping both here prevents the two-stage challenger from being evaluated
    only against M4, which would hide whether it adds value beyond the native
    P0 policy anchor.
    """
    parts: list[pd.DataFrame] = []
    cols = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "arm", "held_month",
        "expected_net_bps", "raw_meta_score", "train_p80_expected_bps", "policy_path_valid",
    ]
    for root in roots:
        path = root / "short_absolute_conversion_oof_predictions.parquet"
        frame = pd.read_parquet(path, columns=cols)
        frame = frame.loc[frame["arm"].astype(str).isin(("M0", "M4"))].copy()
        frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(["arm", "candidate_id"]).any():
        raise AssertionError("stored control overlaps candidate identities within arm")
    return out


def _load_feature_selection(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = json.loads(path.read_text())
    f90 = tuple(payload["feature_sets"]["90"])
    f115 = tuple(payload["feature_sets"]["115"])
    if len(f90) != 90 or len(f115) != 115 or not set(f90).issubset(f115):
        raise AssertionError("invalid frozen short F90/F115 selection contract")
    return f90, f115


def _matrix(frame: pd.DataFrame, fields: tuple[str, ...], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, list(fields)].apply(_finite).astype(np.float32)
    if medians is None:
        medians = values.median(axis=0, skipna=True).fillna(0.0).astype(np.float32)
    return values.fillna(medians).astype(np.float32), medians


def _opportunity_label(frame: pd.DataFrame) -> np.ndarray:
    return _finite(frame["mfe_12h_bps"]).gt(MFE_OPPORTUNITY_BPS).astype(np.int8).to_numpy()


def _conversion_grade(frame: pd.DataFrame) -> np.ndarray:
    regret = _finite(frame["policy_regret_bps"]).to_numpy(float)
    return (4 - np.digitize(regret, T5_REGRET_EDGES, right=False)).astype(np.int8)


def _ordinal_opportunity_grade(frame: pd.DataFrame) -> np.ndarray:
    return np.digitize(_finite(frame["mfe_12h_bps"]).to_numpy(float), ORDINAL_OPP_EDGES, right=False).astype(np.int8)


def _binary_model(*, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=0.035, max_depth=3, num_leaves=15,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0,
        reg_alpha=0.10, class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _ordinal_model(*, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=6, n_estimators=180, learning_rate=0.035,
        max_depth=3, num_leaves=15, min_child_samples=40, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=4.0, reg_alpha=0.10, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _predict_opportunity(model: LGBMClassifier, x: pd.DataFrame, *, kind: str) -> np.ndarray:
    probability = np.asarray(model.predict_proba(x), dtype=float)
    if kind == "binary":
        return probability[:, 1]
    # Grades 0, 1, 2 correspond to MFE <= 200 bps; grades 3+ imply >200 bps.
    return probability[:, 3:].sum(axis=1)


def _predict_conversion(model: LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(model.predict_proba(x), dtype=float)
    return probability @ np.arange(probability.shape[1], dtype=float)


def _empirical_cdf(values: np.ndarray) -> EmpiricalCDF:
    local = np.asarray(values, dtype=float)
    return EmpiricalCDF(np.sort(local[np.isfinite(local)]))


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 5 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return 0.0
    value = pd.Series(left).corr(pd.Series(right), method="spearman")
    return 0.0 if not np.isfinite(value) else float(value)


def _fit_isotonic(x: np.ndarray, y: np.ndarray, *, low: float, high: float) -> tuple[IsotonicRegression, bool, float]:
    rho = _safe_spearman(x, y)
    estimator = IsotonicRegression(
        increasing=bool(rho >= 0.0), out_of_bounds="clip", y_min=low, y_max=high,
    )
    estimator.fit(x, y)
    return estimator, bool(rho >= 0.0), rho


def _month_count(frame: pd.DataFrame) -> int:
    return int(frame["__decision_ts__"].dt.strftime("%Y-%m").nunique())


def _inner_oof(
    train: pd.DataFrame,
    *,
    opportunity_fields: tuple[str, ...],
    conversion_fields: tuple[str, ...],
    opportunity_kind: str,
    seed: int,
) -> pd.DataFrame:
    """Create joint OOF O/C predictions with C trained only on prior O=1 rows."""
    local = train.loc[_valid_label(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(local) < MIN_OUTER_TRAIN_ROWS:
        raise ValueError("insufficient valid rows for inner chronological OOF")
    boundaries = np.linspace(0, len(local), INNER_SPLITS + 2, dtype=int)
    parts: list[pd.DataFrame] = []
    for fold in range(INNER_SPLITS):
        fit_end, valid_end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        if fit_end < MIN_OUTER_TRAIN_ROWS or valid_end <= fit_end:
            continue
        fit = local.iloc[:fit_end].copy()
        valid = local.iloc[fit_end:valid_end].copy()
        y_opportunity = _opportunity_label(fit)
        if np.unique(y_opportunity).size < 2:
            continue
        x_fit_o, med_o = _matrix(fit, opportunity_fields)
        x_valid_o, _ = _matrix(valid, opportunity_fields, med_o)
        o_model = _binary_model(seed=seed + fold) if opportunity_kind == "binary" else _ordinal_model(seed=seed + fold)
        o_model.fit(x_fit_o, y_opportunity if opportunity_kind == "binary" else _ordinal_opportunity_grade(fit))
        opportunity_raw = _predict_opportunity(o_model, x_valid_o, kind=opportunity_kind)

        c_fit = fit.loc[_opportunity_label(fit).astype(bool)].copy()
        if len(c_fit) < MIN_C_POSITIVES or _month_count(c_fit) < MIN_MAPPER_MONTHS:
            continue
        y_conversion = _conversion_grade(c_fit)
        if np.unique(y_conversion).size < 2:
            continue
        x_fit_c, med_c = _matrix(c_fit, conversion_fields)
        x_valid_c, _ = _matrix(valid, conversion_fields, med_c)
        c_model = _ordinal_model(seed=seed + 100 + fold)
        c_model.fit(x_fit_c, y_conversion)
        part = valid.loc[:, [*IDENTITY, "__label_available_at__", "mfe_12h_bps", "policy_net_bps", "policy_regret_bps"]].copy()
        part["opp_oof_raw"] = opportunity_raw.astype(np.float32)
        part["conversion_oof_raw"] = _predict_conversion(c_model, x_valid_c).astype(np.float32)
        part["inner_fold"] = fold
        part["c_fit_opportunity_rows"] = len(c_fit)
        parts.append(part)
    if not parts:
        raise ValueError("no joint inner OOF slices with enough conditional conversion support")
    oof = pd.concat(parts, ignore_index=True)
    if len(oof) < MIN_MAPPER_OOF_ROWS or _month_count(oof) < MIN_MAPPER_MONTHS:
        raise ValueError("insufficient joint OOF rows or calendar-month support for final mapper")
    if oof["candidate_id"].duplicated().any():
        raise AssertionError("inner OOF prediction identity is not unique")
    return oof


@dataclass
class CombinerBundle:
    opportunity_calibrator: IsotonicRegression
    opportunity_calibration_increasing: bool
    conversion_mu1: IsotonicRegression
    conversion_mu1_increasing: bool
    no_opportunity_mu0: float
    conversion_cdf: EmpiricalCDF
    product_mapper: IsotonicRegression
    product_mapper_increasing: bool
    o_cdf: EmpiricalCDF
    k2_table: np.ndarray
    k2_global_mean: float
    o_only_mapper: IsotonicRegression
    o_only_mapper_increasing: bool
    thresholds: dict[str, float]
    oof_rows: int
    oof_months: int


def _fit_combiner_bundle(oof: pd.DataFrame) -> CombinerBundle:
    y = _finite(oof["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    opp = _opportunity_label(oof)
    o_calibrator, o_inc, _ = _fit_isotonic(
        oof["opp_oof_raw"].to_numpy(float), opp.astype(float), low=0.0, high=1.0,
    )
    p_o = np.asarray(o_calibrator.predict(oof["opp_oof_raw"].to_numpy(float)), dtype=float)
    c_raw = oof["conversion_oof_raw"].to_numpy(float)
    cdf = _empirical_cdf(c_raw)
    positive = opp.astype(bool)
    if positive.sum() < MIN_C_POSITIVES:
        raise ValueError("insufficient opportunity-positive OOF rows for conditional conversion map")
    mu1, mu1_inc, _ = _fit_isotonic(c_raw[positive], y[positive], low=-POLICY_CLIP_BPS, high=POLICY_CLIP_BPS)
    global_mean = float(np.mean(y))
    negatives = ~positive
    # Strong shrinkage protects a small/noisy O=0 economic estimate.  Its prior
    # is the entire OOF policy distribution, not a later or held outcome set.
    mu0 = float((y[negatives].sum() + 500.0 * global_mean) / (negatives.sum() + 500.0))
    mu1_oof = np.asarray(mu1.predict(c_raw), dtype=float)
    k0 = p_o * mu1_oof + (1.0 - p_o) * mu0

    product = p_o * cdf.transform(c_raw)
    product_mapper, product_inc, _ = _fit_isotonic(product, y, low=-POLICY_CLIP_BPS, high=POLICY_CLIP_BPS)
    k1 = np.asarray(product_mapper.predict(product), dtype=float)

    o_cdf = _empirical_cdf(p_o)
    o_bin = np.minimum((o_cdf.transform(p_o) * 5.0).astype(int), 4)
    c_bin = np.minimum((cdf.transform(c_raw) * 5.0).astype(int), 4)
    table = np.full((5, 5), global_mean, dtype=float)
    for ob in range(5):
        own = y[o_bin == ob]
        own_mean = float((own.sum() + 100.0 * global_mean) / (len(own) + 100.0))
        for cb in range(5):
            cell = y[(o_bin == ob) & (c_bin == cb)]
            table[ob, cb] = float((cell.sum() + 50.0 * own_mean) / (len(cell) + 50.0))
    k2 = table[o_bin, c_bin]

    o_only_mapper, o_only_inc, _ = _fit_isotonic(p_o, y, low=-POLICY_CLIP_BPS, high=POLICY_CLIP_BPS)
    o_only = np.asarray(o_only_mapper.predict(p_o), dtype=float)
    thresholds = {
        "O_only": float(np.quantile(o_only, P80)),
        "K0_analytic_mixture": float(np.quantile(k0, P80)),
        "K1_product_isotonic": float(np.quantile(k1, P80)),
        "K2_hierarchical_2d": float(np.quantile(k2, P80)),
    }
    return CombinerBundle(
        opportunity_calibrator=o_calibrator,
        opportunity_calibration_increasing=o_inc,
        conversion_mu1=mu1,
        conversion_mu1_increasing=mu1_inc,
        no_opportunity_mu0=mu0,
        conversion_cdf=cdf,
        product_mapper=product_mapper,
        product_mapper_increasing=product_inc,
        o_cdf=o_cdf,
        k2_table=table,
        k2_global_mean=global_mean,
        o_only_mapper=o_only_mapper,
        o_only_mapper_increasing=o_only_inc,
        thresholds=thresholds,
        oof_rows=len(oof), oof_months=_month_count(oof),
    )


def _apply_combiner(bundle: CombinerBundle, raw_o: np.ndarray, raw_c: np.ndarray) -> pd.DataFrame:
    p_o = np.asarray(bundle.opportunity_calibrator.predict(raw_o), dtype=float)
    cdf = bundle.conversion_cdf.transform(raw_c)
    mu1 = np.asarray(bundle.conversion_mu1.predict(raw_c), dtype=float)
    product = p_o * cdf
    ob = np.minimum((bundle.o_cdf.transform(p_o) * 5.0).astype(int), 4)
    cb = np.minimum((cdf * 5.0).astype(int), 4)
    return pd.DataFrame({
        "opportunity_probability": p_o.astype(np.float32),
        "conversion_cdf": cdf.astype(np.float32),
        "O_only_expected_policy_net_bps": bundle.o_only_mapper.predict(p_o).astype(np.float32),
        "K0_analytic_mixture_expected_policy_net_bps": (p_o * mu1 + (1.0 - p_o) * bundle.no_opportunity_mu0).astype(np.float32),
        "K1_product_isotonic_expected_policy_net_bps": bundle.product_mapper.predict(product).astype(np.float32),
        "K2_hierarchical_2d_expected_policy_net_bps": bundle.k2_table[ob, cb].astype(np.float32),
        "opportunity_bin": ob.astype(np.int8), "conversion_bin": cb.astype(np.int8),
    })


def _fit_outer_predict(
    train: pd.DataFrame,
    held: pd.DataFrame,
    *,
    opportunity_contract: OpportunityContract,
    conversion_fields: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    valid_train = train.loc[_valid_label(train)].copy()
    if len(valid_train) < MIN_OUTER_TRAIN_ROWS:
        raise ValueError("insufficient strict-prequential valid outer training rows")
    oof = _inner_oof(
        valid_train, opportunity_fields=opportunity_contract.fields,
        conversion_fields=conversion_fields, opportunity_kind=opportunity_contract.kind, seed=seed,
    )
    bundle = _fit_combiner_bundle(oof)

    y_opportunity = _opportunity_label(valid_train)
    x_train_o, med_o = _matrix(valid_train, opportunity_contract.fields)
    x_held_o, _ = _matrix(held, opportunity_contract.fields, med_o)
    o_model = _binary_model(seed=seed + 1000) if opportunity_contract.kind == "binary" else _ordinal_model(seed=seed + 1000)
    o_model.fit(x_train_o, y_opportunity if opportunity_contract.kind == "binary" else _ordinal_opportunity_grade(valid_train))
    raw_o = _predict_opportunity(o_model, x_held_o, kind=opportunity_contract.kind)

    c_train = valid_train.loc[y_opportunity.astype(bool)].copy()
    if len(c_train) < MIN_C_POSITIVES or _month_count(c_train) < MIN_MAPPER_MONTHS:
        raise ValueError("insufficient strict-prequential O=1 rows for final conversion head")
    y_c = _conversion_grade(c_train)
    if np.unique(y_c).size < 2:
        raise ValueError("conditional conversion target has fewer than two classes")
    x_train_c, med_c = _matrix(c_train, conversion_fields)
    x_held_c, _ = _matrix(held, conversion_fields, med_c)
    c_model = _ordinal_model(seed=seed + 2000)
    c_model.fit(x_train_c, y_c)
    raw_c = _predict_conversion(c_model, x_held_c)

    output = held.loc[:, [*IDENTITY, "__label_available_at__", "mfe_12h_bps", "policy_net_bps", "policy_regret_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid"]].copy().reset_index(drop=True)
    output["opportunity_raw_score"] = raw_o.astype(np.float32)
    output["conversion_raw_score"] = raw_c.astype(np.float32)
    output = pd.concat([output, _apply_combiner(bundle, raw_o, raw_c)], axis=1)
    for arm, threshold in bundle.thresholds.items():
        output[f"{arm}_train_p80_expected_policy_net_bps"] = np.float32(threshold)
    # Attach the OOF scores for audit and exact causal mapper lineage.
    oof = oof.copy()
    oof["opportunity_probability"] = bundle.opportunity_calibrator.predict(oof["opp_oof_raw"].to_numpy(float)).astype(np.float32)
    oof["conversion_cdf"] = bundle.conversion_cdf.transform(oof["conversion_oof_raw"].to_numpy(float)).astype(np.float32)
    audit = {
        "outer_train_rows": len(valid_train), "outer_c_train_rows": len(c_train),
        "inner_oof_rows": bundle.oof_rows, "inner_oof_months": bundle.oof_months,
        "opportunity_kind": opportunity_contract.kind, "opportunity_feature_count": len(opportunity_contract.fields),
        "conversion_feature_count": len(conversion_fields), "mu0_no_opportunity_bps": bundle.no_opportunity_mu0,
        "o_calibration_increasing": bundle.opportunity_calibration_increasing,
        "c_mu1_increasing": bundle.conversion_mu1_increasing,
        "product_map_increasing": bundle.product_mapper_increasing,
        "o_only_map_increasing": bundle.o_only_mapper_increasing,
        "thresholds": bundle.thresholds,
    }
    return output, oof, audit


def _safe_binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(y) < 5 or np.unique(y).size < 2:
        return {key: float("nan") for key in ("auc", "prauc", "brier", "logloss", "calibration_slope", "calibration_intercept")}
    probability = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    slope, intercept = np.polyfit(probability, y.astype(float), 1) if np.unique(probability).size >= 2 else (float("nan"), float("nan"))
    return {
        "auc": float(roc_auc_score(y, probability)),
        "prauc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "logloss": float(log_loss(y, probability, labels=[0, 1])),
        "calibration_slope": float(slope), "calibration_intercept": float(intercept),
    }


def _opportunity_metric_rows(prediction: pd.DataFrame, *, arm: str, month: pd.Timestamp) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = prediction.loc[_valid_label(prediction)].copy()
    y = _opportunity_label(valid)
    score = valid["opportunity_probability"].to_numpy(float)
    base = {
        "arm": arm, "held_month": month.strftime("%Y-%m"), "valid_rows": len(valid),
        "opportunity_prevalence": float(np.mean(y)) if len(y) else float("nan"),
        **_safe_binary_metrics(y, score),
    }
    deciles: list[dict[str, Any]] = []
    if len(valid):
        order = np.argsort(score, kind="stable")
        rank = np.empty(len(valid), dtype=float)
        rank[order] = (np.arange(len(valid), dtype=float) + 1.0) / float(len(valid))
        for decile in range(10):
            member = (rank > decile / 10.0) & (rank <= (decile + 1) / 10.0)
            deciles.append({
                "arm": arm, "held_month": month.strftime("%Y-%m"), "decile": decile + 1,
                "rows": int(member.sum()), "opportunity_prevalence": float(np.mean(y[member])) if member.any() else float("nan"),
            })
        prevalence = float(np.mean(y))
        for fraction in (0.10, 0.20, 0.30):
            member = rank > 1.0 - fraction
            prefix = f"top{int(fraction * 100)}"
            top = float(np.mean(y[member])) if member.any() else float("nan")
            base[f"{prefix}_opportunity_prevalence"] = top
            base[f"{prefix}_opportunity_lift"] = top / prevalence if prevalence > 0 else float("nan")
    else:
        for fraction in (10, 20, 30):
            base[f"top{fraction}_opportunity_prevalence"] = float("nan")
            base[f"top{fraction}_opportunity_lift"] = float("nan")
    return base, deciles


def _economic_metric_rows(prediction: pd.DataFrame, *, arm: str, month: pd.Timestamp, expected_column: str, threshold_column: str, selection: str = "causal_train_p80") -> dict[str, Any]:
    threshold = float(_finite(prediction[threshold_column]).iloc[0])
    selected = prediction.loc[_finite(prediction[expected_column]).ge(threshold)].copy()
    valid = selected.loc[_valid_label(selected)].copy()
    net = _finite(valid["policy_net_bps"]).to_numpy(float)
    return {
        "arm": arm, "held_month": month.strftime("%Y-%m"), "selection": selection,
        "threshold_bps": threshold, "scored_candidates": len(selected), "outcome_known_candidates": len(valid),
        "outcome_coverage": float(len(valid) / len(selected)) if len(selected) else float("nan"),
        "trades": len(valid), "net_bps_per_trade": float(np.mean(net)) if len(net) else float("nan"),
        "total_net_bps": float(np.sum(net)) if len(net) else float("nan"),
        "positive_rate": float(np.mean(net > 0.0)) if len(net) else float("nan"),
        "cvar10_bps": float(np.mean(np.sort(net)[:max(1, math.ceil(len(net) * 0.10))])) if len(net) else float("nan"),
    }


def _aggregate_opportunity(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, block in monthly.groupby("arm", sort=False):
        for era, part in block.assign(era=pd.to_datetime(block["held_month"] + "-01", utc=True).dt.year).groupby("era", sort=True):
            rows.append({
                "arm": arm, "era": int(era), "months": int(part["held_month"].nunique()), "valid_rows": int(part["valid_rows"].sum()),
                **{column: float(part[column].mean()) for column in ("opportunity_prevalence", "auc", "prauc", "brier", "logloss", "calibration_slope", "calibration_intercept", "top10_opportunity_lift", "top20_opportunity_lift", "top30_opportunity_lift")},
                "positive_lift_months": int((part["top20_opportunity_lift"] > 1.0).sum()),
            })
    return pd.DataFrame(rows)


def _aggregate_economics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, block in monthly.groupby("arm", sort=False):
        for era, part in block.assign(era=pd.to_datetime(block["held_month"] + "-01", utc=True).dt.year).groupby("era", sort=True):
            trade_weight = np.maximum(pd.to_numeric(part["trades"], errors="coerce").fillna(0).to_numpy(float), 0.0)
            values = pd.to_numeric(part["net_bps_per_trade"], errors="coerce").to_numpy(float)
            usable = np.isfinite(values) & (trade_weight > 0)
            rows.append({
                "arm": arm, "era": int(era), "months": int(part["held_month"].nunique()),
                "trades": int(trade_weight.sum()),
                "net_bps_per_trade": float(np.average(values[usable], weights=trade_weight[usable])) if usable.any() else float("nan"),
                "total_net_bps": float(pd.to_numeric(part["total_net_bps"], errors="coerce").sum()),
                "positive_months": int((part["net_bps_per_trade"] > 0.0).sum()),
                "worst_month_net_bps_per_trade": float(pd.to_numeric(part["net_bps_per_trade"], errors="coerce").min()),
                "mean_outcome_coverage": float(pd.to_numeric(part["outcome_coverage"], errors="coerce").mean()),
                "mean_cvar10_bps": float(pd.to_numeric(part["cvar10_bps"], errors="coerce").mean()),
            })
    return pd.DataFrame(rows)


def _mda_objective(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) < 20 or np.unique(y).size < 2:
        return float("nan")
    pr_auc = float(average_precision_score(y, score))
    prevalence = float(np.mean(y))
    ranks = pd.Series(score).rank(method="first", pct=True).to_numpy(float)
    top = float(np.mean(y[ranks > .80]))
    lift = top / prevalence if prevalence else 0.0
    brier = float(brier_score_loss(y, np.clip(score, 1e-6, 1.0 - 1e-6)))
    brier_skill = 1.0 - brier / max(prevalence * (1.0 - prevalence), 1e-6)
    return pr_auc / max(prevalence, 1e-6) + lift + brier_skill


def _chronological_mda(frame: pd.DataFrame, fields: tuple[str, ...], *, seed: int) -> pd.DataFrame:
    """Target-specific time-ordered permutation MDA for opp200.

    The model is fit once per chronological validation slice.  Each candidate
    field is then permuted only inside that held slice, avoiding a broad
    re-fitting search while measuring incremental opportunity discrimination.
    """
    local = frame.loc[_valid_label(frame)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(local) < MIN_OUTER_TRAIN_ROWS:
        raise ValueError("insufficient development rows for chronological opportunity MDA")
    boundaries = np.linspace(0, len(local), INNER_SPLITS + 2, dtype=int)
    deltas: dict[str, list[float]] = {field: [] for field in fields}
    records: list[dict[str, Any]] = []
    for fold in range(INNER_SPLITS):
        fit_end, valid_end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        if fit_end < MIN_OUTER_TRAIN_ROWS or valid_end <= fit_end:
            continue
        fit, valid = local.iloc[:fit_end], local.iloc[fit_end:valid_end]
        y_fit, y_valid = _opportunity_label(fit), _opportunity_label(valid)
        if np.unique(y_fit).size < 2 or np.unique(y_valid).size < 2:
            continue
        x_fit, medians = _matrix(fit, fields)
        x_valid, _ = _matrix(valid, fields, medians)
        model = _binary_model(seed=seed + fold)
        model.fit(x_fit, y_fit)
        baseline = _mda_objective(y_valid, _predict_opportunity(model, x_valid, kind="binary"))
        rng = np.random.default_rng(seed + 100 + fold)
        for field in fields:
            permuted = x_valid.copy()
            permuted[field] = rng.permutation(permuted[field].to_numpy())
            degraded = _mda_objective(y_valid, _predict_opportunity(model, permuted, kind="binary"))
            delta = baseline - degraded
            deltas[field].append(delta)
            records.append({"feature": field, "fold": fold, "baseline_objective": baseline, "permuted_objective": degraded, "mda_delta": delta})
    result = pd.DataFrame({
        "feature": list(fields),
        "mda_mean": [float(np.nanmean(deltas[field])) if deltas[field] else float("nan") for field in fields],
        "mda_min": [float(np.nanmin(deltas[field])) if deltas[field] else float("nan") for field in fields],
        "mda_positive_folds": [int(np.sum(np.asarray(deltas[field]) > 0.0)) for field in fields],
    })
    result = result.sort_values(["mda_mean", "mda_min", "feature"], ascending=[False, False, True], kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result, pd.DataFrame(records)


def _decomposition(frame: pd.DataFrame, m4: pd.DataFrame, t5_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    m4_join = m4.copy()
    t5 = pd.read_parquet(t5_path / "path_target_oof_predictions.parquet")
    t5 = t5.loc[t5["arm"].astype(str).eq("T5_conditional_low_regret")].copy()
    t5["__decision_ts__"] = _utc(t5["__decision_ts__"])
    t5_valid = _valid_label(t5) & _finite(t5["mfe_12h_bps"]).gt(MFE_OPPORTUNITY_BPS)
    for era in (2024, 2025, 2026):
        part = frame.loc[(frame["__decision_ts__"].dt.year == era) & _valid_label(frame)].copy()
        opp = _opportunity_label(part).astype(bool)
        row: dict[str, Any] = {
            "era": era, "all_p0_valid_rows": len(part), "opportunity_prevalence_all_p0": float(np.mean(opp)) if len(part) else float("nan"),
            "policy_net_opportunity_bps": float(_finite(part.loc[opp, "policy_net_bps"]).mean()) if opp.any() else float("nan"),
            "policy_net_no_opportunity_bps": float(_finite(part.loc[~opp, "policy_net_bps"]).mean()) if (~opp).any() else float("nan"),
        }
        m4_part = m4_join.loc[(m4_join["__decision_ts__"].dt.year == era) & _valid_label(m4_join)].copy()
        if len(m4_part):
            admitted = _finite(m4_part["expected_net_bps"]).ge(_finite(m4_part["train_p80_expected_bps"]))
            row["m4_admitted_valid_rows"] = int(admitted.sum())
            row["opportunity_prevalence_m4_admitted"] = float(_opportunity_label(m4_part.loc[admitted]).mean()) if admitted.any() else float("nan")
        t5_part = t5.loc[(t5["__decision_ts__"].dt.year == era) & t5_valid].copy()
        if len(t5_part):
            selected = _finite(t5_part["expected_policy_net_bps"]).ge(_finite(t5_part["train_p80_expected_policy_net_bps"]))
            row["t5_conditional_selected_rows"] = int(selected.sum())
            row["t5_selected_policy_net_bps_given_opportunity"] = float(_finite(t5_part.loc[selected, "policy_net_bps"]).mean()) if selected.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _control_metric_rows(controls: pd.DataFrame, *, month: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control_arm, label in (("M0", "M0_P0_anchor_control"), ("M4", "M4_stored_control")):
        held = controls.loc[
            controls["arm"].astype(str).eq(control_arm)
            & controls["__decision_ts__"].ge(month)
            & controls["__decision_ts__"].lt(month + pd.offsets.MonthBegin(1))
        ].copy()
        if held.empty:
            continue
        held["control_expected_policy_net_bps"] = _finite(held["expected_net_bps"])
        held["control_train_p80_expected_policy_net_bps"] = _finite(held["train_p80_expected_bps"])
        rows.append(_economic_metric_rows(
            held, arm=label, month=month,
            expected_column="control_expected_policy_net_bps",
            threshold_column="control_train_p80_expected_policy_net_bps",
        ))
    return rows


def _write_report(out: Path, *, decomposition: pd.DataFrame, opportunity_era: pd.DataFrame, economics_era: pd.DataFrame, fold_audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
        if columns is not None:
            frame = frame.loc[:, [column for column in columns if column in frame]]
        if not len(frame):
            return "_No supported rows._"
        # ``tabulate`` is optional in the minimal Ares training runtime.  A
        # completed immutable result bundle must not be marked failed merely
        # because its human-readable companion cannot import that dependency.
        # The fallback deliberately retains all numeric cells as plain text;
        # parquet remains the authoritative machine-readable artifact.
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            header = " | ".join(map(str, frame.columns))
            separator = " | ".join(["---"] * len(frame.columns))
            rows = [" | ".join(map(str, row)) for row in frame.itertuples(index=False, name=None)]
            return "\n".join([header, separator, *rows])

    text = [
        "# Short P0 strict-OOF opportunity × conversion challenger",
        "",
        "Research-only. Every target-free P0 winner receives O and C scores; realised MFE is used only to train C and to score outcomes.",
        "",
        "## Exact-path decomposition",
        "",
        table(decomposition),
        "",
        "## Opportunity discrimination by era",
        "",
        table(opportunity_era, ["arm", "era", "months", "valid_rows", "opportunity_prevalence", "auc", "prauc", "brier", "logloss", "top10_opportunity_lift", "top20_opportunity_lift", "top30_opportunity_lift", "positive_lift_months"]),
        "",
        "## Causal p80 economic result by era",
        "",
        table(economics_era, ["arm", "era", "months", "trades", "net_bps_per_trade", "total_net_bps", "positive_months", "worst_month_net_bps_per_trade", "mean_cvar10_bps"]),
        "",
        "## Contract",
        "",
        "- Outer rows require `label_available_at < held_month_start`.",
        "- O and C mapper inputs are inner chronological OOF predictions; C fits only prior `MFE_H12 > 200 bps` rows but scores all held candidates.",
        "- K0 is analytic mixture, K1 is product-to-isotonic, K2 is a 5×5 hierarchical O/C map.",
        "- Selection is each mapper's train-only OOF p80 expected-bps threshold; no held top-k is used.",
        "- Invalid exact paths are excluded only after scoring, and never become failures.",
        "- MDA selections use the declared development window.  An MDA arm is not independent evidence for any held period overlapping that window; independent MDA evidence begins after its window ends.",
        "",
        "## Fold coverage",
        "",
        table(fold_audit, ["arm", "held_month", "status", "outer_train_rows", "outer_c_train_rows", "inner_oof_rows", "inner_oof_months", "reason"]),
        "",
        "## Manifest summary",
        "",
        "```json",
        json.dumps({key: manifest[key] for key in ("schema", "side", "training", "entry", "policy", "feature_selection", "minimum_support")}, indent=2),
        "```",
        "",
    ]
    (out / "SHORT_P0_TWO_STAGE_OPPORTUNITY_CONVERSION_REPORT.md").write_text("\n".join(text))


def run(
    *,
    population_roots: Sequence[Path],
    rich_labels_root: Path,
    t5_artifact: Path,
    feature_selection: Path,
    feature_panels: Sequence[Path],
    out: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    f90, f115 = _load_feature_selection(feature_selection)
    m4_fields = _load_m4_fields(population_roots[0])
    for root in population_roots[1:]:
        if _load_m4_fields(root) != m4_fields:
            raise AssertionError("M4 contracts disagree between matched P0 sources")
    population, population_hashes = _load_population(population_roots, m4_fields)
    rich_labels, label_hash = _load_rich_labels(rich_labels_root)
    features, feature_hashes = _load_f115_features(population, f115, feature_panels)
    shared_f115 = tuple(field for field in f115 if field in population.columns)
    check_features = population.loc[:, ["candidate_id", *shared_f115]].merge(
        features.loc[:, ["candidate_id", *shared_f115]], on="candidate_id", how="left",
        suffixes=("_p0", "_f115"), validate="one_to_one",
    )
    for field in shared_f115:
        if not _numeric_equal(check_features[f"{field}_p0"], check_features[f"{field}_f115"]):
            raise AssertionError(f"P0 M4 source and target-free F115 source disagree for {field}")
    new_f115 = tuple(field for field in f115 if field not in population.columns)
    frame = population.merge(features.loc[:, ["candidate_id", *new_f115]], on="candidate_id", how="left", validate="one_to_one")
    # The frozen P0 source owns exact policy-path validity; rich labels supply
    # target values and their own target-validity state.
    frame = frame.merge(rich_labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(frame) != len(population) or frame["candidate_id"].duplicated().any():
        raise AssertionError("feature/label joins changed target-free P0 identity")
    if not frame["__label_available_at__"].dropna().eq(frame.loc[frame["__label_available_at__"].notna(), "__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise AssertionError("exact rich labels must resolve at decision plus 12 hours")
    if frame.loc[~_valid_label(frame), ["mfe_12h_bps", "policy_net_bps", "policy_regret_bps"]].notna().any().any():
        raise AssertionError("invalid exact paths carry supervised outcome values")
    feature_coverage = pd.DataFrame({
        "feature": list(f115), "finite_fraction": [float(_finite(frame[field]).notna().mean()) for field in f115],
    })
    feature_coverage["meets_p0_candidate_90pct_gate"] = feature_coverage["finite_fraction"].ge(.90)
    # F90/F115 are deliberately retained as exact historical-contract controls:
    # their in-fold median imputer is causal and therefore makes them evaluable.
    # MDA, however, must not select a field that fails the current rank-1
    # candidate coverage contract.  This exposes rather than hides a mismatch
    # between the original global 115-field selection gate and the P0 stream.
    mda_pool = tuple(feature_coverage.loc[feature_coverage["meets_p0_candidate_90pct_gate"], "feature"])
    if len(mda_pool) < 90:
        raise AssertionError("fewer than 90 current P0 candidate-coverage-valid F115 fields")

    mda_population = frame.loc[frame["__decision_ts__"].ge(MDA_START) & frame["__decision_ts__"].lt(MDA_END)].copy()
    mda, mda_folds = _chronological_mda(mda_population, mda_pool, seed=MDA_SEED)
    mda_fields = {
        "O_MDA30_binary": tuple(mda.head(30)["feature"]),
        "O_MDA60_binary": tuple(mda.head(60)["feature"]),
        "O_MDA90_binary": tuple(mda.head(90)["feature"]),
    }
    contracts = (
        OpportunityContract("O_F41_binary", m4_fields),
        OpportunityContract("O_F90_binary", f90),
        OpportunityContract("O_F115_binary", f115),
        *(OpportunityContract(name, fields) for name, fields in mda_fields.items()),
        OpportunityContract("O_F90_ordinal_secondary", f90, kind="ordinal"),
    )
    controls = _load_control_predictions(population_roots).merge(
        frame.loc[:, ["candidate_id", "mfe_12h_bps", "policy_net_bps", "rich_path_label_valid", "rich_path_target_invalid", "__label_available_at__"]],
        on="candidate_id", how="left", validate="many_to_one",
    )
    m4_for_decomposition = controls.loc[controls["arm"].astype(str).eq("M4")].copy()
    decomposition = _decomposition(frame, m4_for_decomposition, t5_artifact)

    output_rows: list[pd.DataFrame] = []
    oof_rows: list[pd.DataFrame] = []
    opportunity_monthly: list[dict[str, Any]] = []
    opportunity_deciles: list[dict[str, Any]] = []
    economics_monthly: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    months = pd.date_range(start.normalize().replace(day=1), end.normalize().replace(day=1), freq="MS", inclusive="left")
    for month_index, month in enumerate(months):
        next_month = month + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(next_month)].copy()
        train = frame.loc[
            frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid_label(frame)
        ].copy()
        if held.empty:
            continue
        economics_monthly.extend(_control_metric_rows(controls, month=month))
        for contract_index, contract in enumerate(contracts):
            try:
                prediction, oof, audit = _fit_outer_predict(
                    train, held, opportunity_contract=contract, conversion_fields=m4_fields,
                    seed=seed + month_index * 1000 + contract_index * 53,
                )
                prediction["arm"] = contract.name
                prediction["held_month"] = month.strftime("%Y-%m")
                oof["arm"] = contract.name
                oof["held_month"] = month.strftime("%Y-%m")
                output_rows.append(prediction)
                oof_rows.append(oof)
                opp_metrics, deciles = _opportunity_metric_rows(prediction, arm=contract.name, month=month)
                opportunity_monthly.append(opp_metrics)
                opportunity_deciles.extend(deciles)
                if contract.kind == "binary":
                    for arm in ("O_only", "K0_analytic_mixture", "K1_product_isotonic", "K2_hierarchical_2d"):
                        economics_monthly.append(_economic_metric_rows(
                            prediction, arm=f"{contract.name}__{arm}", month=month,
                            expected_column=f"{arm}_expected_policy_net_bps",
                            threshold_column=f"{arm}_train_p80_expected_policy_net_bps",
                        ))
                # C is deliberately diagnostic: restrict reporting to realised
                # opportunity paths while never applying this condition to the
                # held population prediction or any deployable combiner.
                c_diag = prediction.loc[_valid_label(prediction) & _finite(prediction["mfe_12h_bps"]).gt(MFE_OPPORTUNITY_BPS)].copy()
                if len(c_diag):
                    threshold = float(np.quantile(c_diag["conversion_raw_score"].to_numpy(float), P80))
                    c_diag["C_diagnostic_expected_policy_net_bps"] = c_diag["conversion_raw_score"]
                    c_diag["C_diagnostic_train_p80_expected_policy_net_bps"] = threshold
                    economics_monthly.append(_economic_metric_rows(
                        c_diag, arm=f"{contract.name}__C_oracle_diagnostic", month=month,
                        expected_column="C_diagnostic_expected_policy_net_bps",
                        threshold_column="C_diagnostic_train_p80_expected_policy_net_bps",
                        selection="oracle_opportunity_diagnostic_only",
                    ))
                fold_rows.append({"arm": contract.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "held_rows": len(held), **audit})
            except ValueError as error:
                fold_rows.append({"arm": contract.name, "held_month": month.strftime("%Y-%m"), "status": "skipped", "held_rows": len(held), "outer_train_rows": len(train), "outer_c_train_rows": int(_opportunity_label(train).sum()) if len(train) else 0, "reason": str(error)})

    if not output_rows:
        raise RuntimeError("two-stage challenger did not produce any strict-OOS predictions")
    output = pd.concat(output_rows, ignore_index=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    opportunity_monthly_frame = pd.DataFrame(opportunity_monthly)
    opportunity_decile_frame = pd.DataFrame(opportunity_deciles)
    economics_monthly_frame = pd.DataFrame(economics_monthly)
    opportunity_era = _aggregate_opportunity(opportunity_monthly_frame)
    economics_era = _aggregate_economics(economics_monthly_frame)
    fold_audit = pd.DataFrame(fold_rows)
    out.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "scope": "strict-OOF short P0 opportunity × conditional conversion research; no short live authority",
        "entry": "frozen signal close plus one hour, exact decision-minute entry",
        "policy": "short SL 3 ATR; trailing activation 0.5 ATR; giveback 0.25 ATR; H12 timeout; 100 bps once",
        "training": "outer monthly strict-prequential fit rows require label_available_at < held month start; inner chronological OOF supplies every mapper input",
        "minimum_support": {"conditional_C_positive_rows": MIN_C_POSITIVES, "mapper_joint_oof_rows": MIN_MAPPER_OOF_ROWS, "mapper_calendar_months": MIN_MAPPER_MONTHS, "failure": "fail closed / skipped fold"},
        "opportunity": {"label": "MFE_H12 > 200 bps", "inference": "scores every target-free P0 winner", "contracts": {contract.name: {"kind": contract.kind, "fields": list(contract.fields)} for contract in contracts}},
        "conversion": {"label": "frozen T5 low-regret ordinal", "fit_domain": "prior valid O=1 rows only", "inference": "scores every held P0 winner", "fields": list(m4_fields)},
        "combiners": {"K0": "p(O)*mu1(C)+(1-p(O))*shrunk_mu0", "K1": "p(O)*CDF(C) then OOF isotonic policy-net map", "K2": "OOF 5x5 O/C hierarchical shrinkage policy-net map"},
        "selection": "each expected-net arm uses its own outer-training inner-OOF p80 expected-bps threshold; no held top-k percentile is computed",
        "controls": {
            "M0": "frozen same-model P0 policy anchor only",
            "M4": "stored absolute-conversion M4 control",
        },
        "feature_selection": {"f90_f115": str(feature_selection), "mda_window": [MDA_START.isoformat(), MDA_END.isoformat()], "mda_objective": "PR-AUC lift + top20 opportunity lift + Brier skill across chronological folds", "mda_pool": list(mda_pool), "legacy_contract_missingness": feature_coverage.loc[~feature_coverage["meets_p0_candidate_90pct_gate"], "feature"].tolist()},
        "invalidity": "invalid/incomplete exact paths are scored for population coverage but excluded from targets, maps, and outcome metrics",
        "sources": {"population_manifest_hashes": population_hashes, "rich_label_manifest_sha256": label_hash, "feature_panel_sha256": feature_hashes, "t5_artifact": str(t5_artifact), "t5_manifest_sha256": _sha256(t5_artifact / "run_manifest.json")},
    }
    decomposition.to_parquet(out / "decomposition_by_era.parquet", index=False, compression="zstd")
    feature_coverage.to_parquet(out / "opportunity_feature_coverage.parquet", index=False, compression="zstd")
    mda.to_parquet(out / "opportunity_chronological_mda.parquet", index=False, compression="zstd")
    mda_folds.to_parquet(out / "opportunity_chronological_mda_folds.parquet", index=False, compression="zstd")
    output.to_parquet(out / "two_stage_outer_oof_predictions.parquet", index=False, compression="zstd")
    oof.to_parquet(out / "two_stage_inner_oof_ledger.parquet", index=False, compression="zstd")
    opportunity_monthly_frame.to_parquet(out / "opportunity_monthly_metrics.parquet", index=False, compression="zstd")
    opportunity_decile_frame.to_parquet(out / "opportunity_score_deciles.parquet", index=False, compression="zstd")
    opportunity_era.to_parquet(out / "opportunity_era_metrics.parquet", index=False, compression="zstd")
    economics_monthly_frame.to_parquet(out / "combined_monthly_metrics.parquet", index=False, compression="zstd")
    economics_era.to_parquet(out / "combined_era_metrics.parquet", index=False, compression="zstd")
    fold_audit.to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(out, decomposition=decomposition, opportunity_era=opportunity_era, economics_era=economics_era, fold_audit=fold_audit, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--population-root", type=Path, action="append")
    parser.add_argument("--rich-labels", type=Path, default=DEFAULT_RICH_LABELS)
    parser.add_argument("--t5-artifact", type=Path, default=DEFAULT_T5_ARTIFACT)
    parser.add_argument("--feature-selection", type=Path, default=DEFAULT_F90_SELECTION)
    parser.add_argument("--feature-panel", type=Path, action="append")
    args = parser.parse_args()
    run(
        population_roots=tuple(args.population_root or DEFAULT_POPULATION_ROOTS), rich_labels_root=args.rich_labels,
        t5_artifact=args.t5_artifact, feature_selection=args.feature_selection,
        feature_panels=tuple(args.feature_panel or DEFAULT_FEATURE_PANELS), out=args.out,
        start=_utc(args.start), end=_utc(args.end), seed=args.seed,
    )


if __name__ == "__main__":
    main()
