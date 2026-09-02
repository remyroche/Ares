#!/usr/bin/env python3
"""Round 1: strict-prequential short P0 → O → C → K0 definition sweep.

This runner freezes the architecture at exactly three supervised components:

    P0 target-free winner → O opportunity probability → C conditional T5 → K0

It changes only the *common* opportunity definition.  For every candidate
definition ``MFE_h > x`` the O target, C training population, and both K0
conditional means use the same event.  No consensus, risk, residual, trust, or
post-K0 mapper is fit or scored.

All rich path columns are supervised labels only.  Models score every held
target-free P0 candidate.  Exact policy outcomes remain H12-resolved, including
for 3h and 6h opportunity definitions; that conservative availability choice
keeps the C/K0 policy target and all training outcomes aligned.
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


ROOT = Path(__file__).resolve().parents[1]
SIDE = "short"
SCHEMA = "strict_r3_short_p0_oc_k0_round1_v1"
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
CORE_SOURCE = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
)
DEFAULT_POPULATION_ROOTS = (
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2024_maydec_20260821_v1",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2025_h1_20260821_v1",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2026_janjul_20260821_v1",
)
DEFAULT_RICH_LABELS = ROOT / "data_perp/artifacts/strict_r3_short_p0_rich_path_labels_apr2024_jul2026_20260821_v1"
DEFAULT_FEATURE_SELECTION = ROOT / "data_perp/artifacts/strict_r3_short_policy_conversion_p1_k32_chronological_mda_20260820_v1/selected_features.json"
DEFAULT_MDA_RANKING = ROOT / "data_perp/artifacts/strict_r3_short_p0_two_stage_opportunity_conversion_2024may_2026jul_20260821_v2/opportunity_chronological_mda.parquet"
DEFAULT_FEATURE_PANELS = (
    ROOT / "data_perp/artifacts/strict_r3_short_features_full2024_20260820_v1/canonical120_features.parquet",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_f90_targetfree_2025_aug2026_20260821_v2/canonical120_features.parquet",
    ROOT / "data_perp/artifacts/strict_r3_short_p0_f90_features_late_july2026_20260821_v1/canonical120_features.parquet",
)
POLICY_CLIP_BPS = 500.0
T5_REGRET_EDGES = np.asarray((50.0, 100.0, 200.0, 400.0), dtype=float)
P80 = .80
MIN_OUTER_TRAIN_ROWS = 800
MIN_C_POSITIVES = 500
MIN_MAPPER_OOF_ROWS = 1_000
MIN_MAPPER_MONTHS = 3
INNER_SPLITS = 3


@dataclass(frozen=True)
class OpportunitySpec:
    name: str
    horizon_hours: int
    threshold_bps: float

    @property
    def label_field(self) -> str:
        return f"mfe_{self.horizon_hours}h_bps"

    @property
    def description(self) -> str:
        return f"{self.label_field} > {self.threshold_bps:g} bps"


# The prescribed threshold sweep plus the predeclared shorter-horizon probes;
# duplicate H12 controls are intentionally represented only once.
ROUND1_SPECS = (
    OpportunitySpec("O150_H12", 12, 150.0),
    OpportunitySpec("O175_H12", 12, 175.0),
    OpportunitySpec("O200_H12_control", 12, 200.0),
    OpportunitySpec("O225_H12", 12, 225.0),
    OpportunitySpec("O250_H12", 12, 250.0),
    OpportunitySpec("O300_H12", 12, 300.0),
    OpportunitySpec("O200_H3", 3, 200.0),
    OpportunitySpec("O200_H6", 6, 200.0),
    OpportunitySpec("O250_H6", 6, 250.0),
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


def _finite(values: pd.Series | np.ndarray | Iterable[float]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _numeric_equal(left: pd.Series, right: pd.Series, *, atol: float = 2e-4) -> bool:
    return bool(np.isclose(
        _finite(left).to_numpy(float), _finite(right).to_numpy(float), rtol=0.0, atol=atol, equal_nan=True,
    ).all())


def _deduplicate_population(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    repeated_mask = frame["candidate_id"].duplicated(keep=False)
    if not repeated_mask.any():
        return frame.copy()
    repeated = frame.loc[repeated_mask].sort_values(["candidate_id", "__decision_ts__"], kind="stable")
    group = repeated.groupby("candidate_id", sort=False)
    for column in (*CORE_SOURCE[1:], *fields):
        values = repeated[column]
        first = group[column].transform("first")
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            equal = np.isclose(_finite(values).to_numpy(float), _finite(first).to_numpy(float), rtol=0.0, atol=2e-4, equal_nan=True)
        else:
            equal = (values.eq(first) | (values.isna() & first.isna())).to_numpy(bool)
        if not bool(np.all(equal)):
            candidate = str(repeated.loc[~equal, "candidate_id"].iloc[0])
            raise AssertionError(f"cumulative target-free P0 sources disagree for {candidate}/{column}")
    return pd.concat([frame.loc[~repeated_mask], repeated.drop_duplicates("candidate_id", keep="first")], ignore_index=True)


def _load_m4_fields(root: Path) -> tuple[str, ...]:
    payload = json.loads((root / "feature_contract.json").read_text())
    fields = tuple(payload.get("base", ()))
    if len(fields) != 41 or len(set(fields)) != 41:
        raise AssertionError("Round 1 requires the frozen 41-field T5 conversion contract")
    return fields


def _load_population(roots: Sequence[Path], m4_fields: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for root in roots:
        population = root / "short_p0_top1_hourly_population.parquet"
        manifest = root / "run_manifest.json"
        if not population.exists() or not manifest.exists():
            raise FileNotFoundError(f"not an immutable short P0 source: {root}")
        local = pd.read_parquet(population, columns=[*CORE_SOURCE, *m4_fields])
        for column in ("__ts__", "__decision_ts__", "policy_label_available_at"):
            local[column] = _utc(local[column])
        if not local["side_name"].astype(str).str.lower().eq(SIDE).all():
            raise AssertionError(f"non-short P0 source: {root}")
        pieces.append(local)
        hashes[str(root.resolve())] = _sha256(manifest)
    output = _deduplicate_population(pd.concat(pieces, ignore_index=True), m4_fields)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("P0 candidate identity is non-unique")
    if not output["__decision_ts__"].eq(output["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("P0 decision convention is not signal close plus one hour")
    return output.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), hashes


def _load_f115_selection(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    fields = tuple(payload["feature_sets"]["115"])
    if len(fields) != 115 or len(set(fields)) != 115:
        raise AssertionError("invalid frozen F115 source selection")
    return fields


def _load_mda90_fields(path: Path, f115: tuple[str, ...]) -> tuple[str, ...]:
    ranking = pd.read_parquet(path)
    fields = tuple(ranking.sort_values("rank", kind="stable").head(90)["feature"].astype(str))
    if len(fields) != 90 or len(set(fields)) != 90 or not set(fields).issubset(f115):
        raise AssertionError("MDA90 ranking is not a 90-field subset of the frozen F115 pool")
    return fields


def _read_feature_panel(panel: Path, candidate_ids: set[str], fields: tuple[str, ...]) -> pd.DataFrame:
    available = set(ds.dataset(panel, format="parquet").schema.names)
    required = {"candidate_id", "__ts__", "__symbol__", "side_name", *fields}
    missing = sorted(required - available)
    if missing:
        raise AssertionError(f"feature panel lacks frozen F115 fields: {missing}")
    table = ds.dataset(panel, format="parquet").to_table(
        columns=["candidate_id", "__ts__", "__symbol__", "side_name", *fields],
        filter=pc.field("candidate_id").isin(sorted(candidate_ids)),
    )
    output = table.to_pandas()
    output["__ts__"] = _utc(output["__ts__"])
    if not output["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError(f"non-short feature rows in {panel}")
    return output


def _load_features(population: pd.DataFrame, f115: tuple[str, ...], panels: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    ids = set(population["candidate_id"].astype(str))
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for panel in panels:
        if not panel.exists():
            raise FileNotFoundError(panel)
        pieces.append(_read_feature_panel(panel, ids, f115))
        hashes[str(panel.resolve())] = _sha256(panel)
    values = pd.concat(pieces, ignore_index=True)
    repeated = values["candidate_id"].duplicated(keep=False)
    if repeated.any():
        duplicate = values.loc[repeated].sort_values("candidate_id", kind="stable")
        for field in f115:
            first = duplicate.groupby("candidate_id", sort=False)[field].transform("first")
            if not _numeric_equal(duplicate[field], first):
                candidate = str(duplicate.loc[~np.isclose(_finite(duplicate[field]).to_numpy(float), _finite(first).to_numpy(float), rtol=0.0, atol=2e-4, equal_nan=True), "candidate_id"].iloc[0])
                raise AssertionError(f"feature panels disagree for {candidate}/{field}")
        values = pd.concat([values.loc[~repeated], duplicate.drop_duplicates("candidate_id", keep="first")], ignore_index=True)
    missing = ids - set(values["candidate_id"].astype(str))
    if missing:
        raise AssertionError(f"target-free feature source misses {len(missing)} P0 candidates")
    return values.loc[:, ["candidate_id", *f115]], hashes


def _load_rich_labels(root: Path) -> tuple[pd.DataFrame, str]:
    manifest = root / "run_manifest.json"
    columns = [
        *IDENTITY, "__label_available_at__", "rich_path_label_valid", "rich_path_target_invalid",
        "mfe_3h_bps", "mfe_6h_bps", "mfe_12h_bps", "policy_regret_bps", "policy_net_bps",
        "policy_gross_bps", "policy_exit_reason", "policy_stop_out",
    ]
    parts = sorted(root.glob("parts/month=*/side=short.parquet"))
    if not parts or not manifest.exists():
        raise FileNotFoundError(f"missing short rich labels under {root}")
    frame = pd.concat([pd.read_parquet(part, columns=columns) for part in parts], ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = _utc(frame[column])
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError("rich label identities are not unique short rows")
    return frame, _sha256(manifest)


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["rich_path_label_valid"].fillna(False).astype(bool)
        & ~frame["rich_path_target_invalid"].fillna(True).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & _finite(frame["policy_net_bps"]).notna()
        & frame["__label_available_at__"].notna()
    )


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    values = frame.loc[:, list(fields)].apply(_finite).astype(np.float32)
    if medians is None:
        medians = values.median(axis=0, skipna=True).fillna(0.0).astype(np.float32)
    return values.fillna(medians).fillna(0.0).astype(np.float32), medians


def _binary_model(seed: int) -> LGBMClassifier:
    # Frozen current O geometry for Round 1.  Later rounds alone may HPO it.
    return LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.035, max_depth=3, num_leaves=15,
        min_child_samples=40, subsample=.85, colsample_bytree=.85, reg_lambda=4.0,
        reg_alpha=.10, class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _ordinal_model(seed: int) -> LGBMClassifier:
    # Frozen current conditional T5 geometry for Round 1.
    return LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=180, learning_rate=.035,
        max_depth=3, num_leaves=15, min_child_samples=40, subsample=.85,
        colsample_bytree=.85, reg_lambda=4.0, reg_alpha=.10, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _event(frame: pd.DataFrame, spec: OpportunitySpec) -> np.ndarray:
    return _finite(frame[spec.label_field]).gt(spec.threshold_bps).astype(np.int8).to_numpy()


def _conversion_grade(frame: pd.DataFrame) -> np.ndarray:
    regret = _finite(frame["policy_regret_bps"]).to_numpy(float)
    return (4 - np.digitize(regret, T5_REGRET_EDGES, right=False)).astype(np.int8)


def _cdf(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    return ordered


def _cdf_transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not len(reference):
        return np.full(len(values), .5, dtype=float)
    return np.searchsorted(reference, values, side="right").astype(float) / float(len(reference))


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 5 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return 0.0
    value = pd.Series(left).corr(pd.Series(right), method="spearman")
    return 0.0 if not np.isfinite(value) else float(value)


def _fit_isotonic(x: np.ndarray, y: np.ndarray, low: float, high: float) -> tuple[IsotonicRegression, bool]:
    increasing = _safe_spearman(x, y) >= 0.0
    fitted = IsotonicRegression(increasing=increasing, out_of_bounds="clip", y_min=low, y_max=high).fit(x, y)
    return fitted, increasing


@dataclass
class K0Bundle:
    opportunity_calibrator: IsotonicRegression
    opportunity_increasing: bool
    mu1: IsotonicRegression
    mu1_increasing: bool
    mu0: float
    threshold: float
    oof_rows: int
    oof_months: int


def _month_count(frame: pd.DataFrame) -> int:
    return int(frame["__decision_ts__"].dt.strftime("%Y-%m").nunique())


def _inner_oof(
    train: pd.DataFrame,
    *,
    spec: OpportunitySpec,
    o_fields: tuple[str, ...],
    c_fields: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Joint chronological OOF ledger with a label-availability purge.

    Every inner O/C model sees only labels resolved before the first decision
    timestamp of its validation slice.  This is deliberately stricter than a
    simple row-order split for a 12-hour exact-policy label.
    """
    local = train.loc[_valid_label(train)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    boundaries = np.linspace(0, len(local), INNER_SPLITS + 2, dtype=int)
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for fold in range(INNER_SPLITS):
        valid_start, valid_end = int(boundaries[fold + 1]), int(boundaries[fold + 2])
        if valid_end <= valid_start:
            continue
        valid = local.iloc[valid_start:valid_end].copy()
        decision_start = valid["__decision_ts__"].min()
        fit = local.loc[local["__label_available_at__"].lt(decision_start)].copy()
        c_fit = fit.loc[_event(fit, spec).astype(bool)].copy()
        row = {
            "inner_fold": fold, "validation_start": decision_start.isoformat(), "validation_rows": int(len(valid)),
            "fit_rows": int(len(fit)), "c_fit_rows": int(len(c_fit)),
            "fit_max_label_available_at": fit["__label_available_at__"].max().isoformat() if len(fit) else None,
        }
        if len(fit) < MIN_OUTER_TRAIN_ROWS or len(c_fit) < MIN_C_POSITIVES or _month_count(c_fit) < MIN_MAPPER_MONTHS:
            row["status"] = "skipped_insufficient_support"
            audit.append(row)
            continue
        if not fit["__label_available_at__"].lt(decision_start).all():
            raise AssertionError("inner OOF fit contains unresolved label at validation decision")
        y_o = _event(fit, spec)
        if np.unique(y_o).size < 2:
            row["status"] = "skipped_single_o_class"
            audit.append(row)
            continue
        y_c = _conversion_grade(c_fit)
        if np.unique(y_c).size < 2:
            row["status"] = "skipped_single_c_class"
            audit.append(row)
            continue
        x_fit_o, med_o = _matrix(fit, o_fields)
        x_valid_o, _ = _matrix(valid, o_fields, med_o)
        o_model = _binary_model(seed + fold)
        o_model.fit(x_fit_o, y_o)
        raw_o = np.asarray(o_model.predict_proba(x_valid_o)[:, 1], dtype=float)
        x_fit_c, med_c = _matrix(c_fit, c_fields)
        x_valid_c, _ = _matrix(valid, c_fields, med_c)
        c_model = _ordinal_model(seed + 100 + fold)
        c_model.fit(x_fit_c, y_c)
        raw_c = np.asarray(c_model.predict_proba(x_valid_c) @ np.arange(5, dtype=float), dtype=float)
        part = valid.loc[:, [*IDENTITY, "__label_available_at__", spec.label_field, "policy_net_bps", "policy_regret_bps"]].copy()
        part["opp_oof_raw"] = raw_o.astype(np.float32)
        part["conversion_oof_raw"] = raw_c.astype(np.float32)
        part["inner_fold"] = fold
        part["fit_rows"] = len(fit)
        part["c_fit_rows"] = len(c_fit)
        parts.append(part)
        row["status"] = "complete"
        audit.append(row)
    if not parts:
        raise ValueError("no purged inner OOF slices with valid O/C support")
    output = pd.concat(parts, ignore_index=True)
    if len(output) < MIN_MAPPER_OOF_ROWS or _month_count(output) < MIN_MAPPER_MONTHS:
        raise ValueError("insufficient purged inner OOF support for K0 calibration")
    if output["candidate_id"].duplicated().any():
        raise AssertionError("inner OOF candidate identity is not unique")
    return output, audit


def _fit_k0(oof: pd.DataFrame, spec: OpportunitySpec) -> K0Bundle:
    y = _finite(oof["policy_net_bps"]).clip(-POLICY_CLIP_BPS, POLICY_CLIP_BPS).to_numpy(float)
    event = _event(oof, spec).astype(bool)
    calibrator, o_increasing = _fit_isotonic(oof["opp_oof_raw"].to_numpy(float), event.astype(float), 0.0, 1.0)
    p_o = np.asarray(calibrator.predict(oof["opp_oof_raw"].to_numpy(float)), dtype=float)
    c_raw = oof["conversion_oof_raw"].to_numpy(float)
    if int(event.sum()) < MIN_C_POSITIVES:
        raise ValueError("insufficient event-positive OOF rows for K0 mu1")
    mu1, mu1_increasing = _fit_isotonic(c_raw[event], y[event], -POLICY_CLIP_BPS, POLICY_CLIP_BPS)
    global_mean = float(np.mean(y))
    negative = ~event
    mu0 = float((y[negative].sum() + 500.0 * global_mean) / (negative.sum() + 500.0))
    k0 = p_o * np.asarray(mu1.predict(c_raw), dtype=float) + (1.0 - p_o) * mu0
    return K0Bundle(
        opportunity_calibrator=calibrator, opportunity_increasing=o_increasing,
        mu1=mu1, mu1_increasing=mu1_increasing, mu0=mu0,
        threshold=float(np.quantile(k0, P80)), oof_rows=len(oof), oof_months=_month_count(oof),
    )


def _apply_k0(bundle: K0Bundle, raw_o: np.ndarray, raw_c: np.ndarray) -> pd.DataFrame:
    probability = np.asarray(bundle.opportunity_calibrator.predict(raw_o), dtype=float)
    expected = probability * np.asarray(bundle.mu1.predict(raw_c), dtype=float) + (1.0 - probability) * bundle.mu0
    return pd.DataFrame({
        "opportunity_probability": probability.astype(np.float32),
        "conversion_score": np.asarray(raw_c, dtype=np.float32),
        "K0_expected_policy_net_bps": expected.astype(np.float32),
        "K0_train_p80_expected_policy_net_bps": np.full(len(expected), bundle.threshold, dtype=np.float32),
    })


def _fit_outer_predict(
    train: pd.DataFrame,
    held: pd.DataFrame,
    *,
    spec: OpportunitySpec,
    o_fields: tuple[str, ...],
    c_fields: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    valid_train = train.loc[_valid_label(train)].copy()
    if len(valid_train) < MIN_OUTER_TRAIN_ROWS:
        raise ValueError("insufficient strict-prequential outer O/C training rows")
    inner, inner_audit = _inner_oof(valid_train, spec=spec, o_fields=o_fields, c_fields=c_fields, seed=seed)
    bundle = _fit_k0(inner, spec)
    y_o = _event(valid_train, spec)
    c_train = valid_train.loc[y_o.astype(bool)].copy()
    if len(c_train) < MIN_C_POSITIVES or _month_count(c_train) < MIN_MAPPER_MONTHS:
        raise ValueError("insufficient strict-prequential event-positive C training support")
    y_c = _conversion_grade(c_train)
    if np.unique(y_o).size < 2 or np.unique(y_c).size < 2:
        raise ValueError("outer O or C target lacks class support")
    x_o, med_o = _matrix(valid_train, o_fields)
    x_held_o, _ = _matrix(held, o_fields, med_o)
    o_model = _binary_model(seed + 1_000)
    o_model.fit(x_o, y_o)
    raw_o = np.asarray(o_model.predict_proba(x_held_o)[:, 1], dtype=float)
    x_c, med_c = _matrix(c_train, c_fields)
    x_held_c, _ = _matrix(held, c_fields, med_c)
    c_model = _ordinal_model(seed + 2_000)
    c_model.fit(x_c, y_c)
    raw_c = np.asarray(c_model.predict_proba(x_held_c) @ np.arange(5, dtype=float), dtype=float)
    output = held.loc[:, [*IDENTITY, "__label_available_at__", spec.label_field, "policy_net_bps", "policy_regret_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid"]].copy().reset_index(drop=True)
    output["opportunity_raw_score"] = raw_o.astype(np.float32)
    output = pd.concat([output, _apply_k0(bundle, raw_o, raw_c)], axis=1)
    inner = inner.copy()
    inner["opportunity_probability"] = bundle.opportunity_calibrator.predict(inner["opp_oof_raw"].to_numpy(float)).astype(np.float32)
    audit = {
        "outer_train_rows": int(len(valid_train)), "outer_c_train_rows": int(len(c_train)),
        "outer_event_prevalence": float(y_o.mean()), "inner_oof_rows": bundle.oof_rows,
        "inner_oof_months": bundle.oof_months, "k0_mu0_bps": bundle.mu0,
        "k0_threshold_bps": bundle.threshold, "o_calibration_increasing": bundle.opportunity_increasing,
        "mu1_calibration_increasing": bundle.mu1_increasing,
    }
    return output, inner, audit, inner_audit


def _probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if len(y) < 5 or np.unique(y).size < 2:
        return {key: float("nan") for key in ("auc", "prauc", "brier", "logloss", "calibration_slope", "calibration_intercept")}
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    slope, intercept = np.polyfit(p, y.astype(float), 1) if np.unique(p).size >= 2 else (float("nan"), float("nan"))
    return {
        "auc": float(roc_auc_score(y, p)), "prauc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_slope": float(slope), "calibration_intercept": float(intercept),
    }


def _o_metrics(prediction: pd.DataFrame, spec: OpportunitySpec, month: pd.Timestamp) -> dict[str, Any]:
    valid = prediction.loc[_valid_label(prediction)].copy()
    y = _event(valid, spec)
    probability = _finite(valid["opportunity_probability"]).to_numpy(float)
    row: dict[str, Any] = {
        "arm": spec.name, "held_month": month.strftime("%Y-%m"), "valid_rows": int(len(valid)),
        "opportunity_prevalence": float(y.mean()) if len(y) else float("nan"),
        **_probability_metrics(y, probability),
    }
    if len(valid):
        order = np.argsort(probability, kind="stable")
        rank = np.empty(len(valid), dtype=float)
        rank[order] = (np.arange(len(valid), dtype=float) + 1.0) / len(valid)
        base = float(y.mean())
        for fraction in (.10, .20, .30):
            selected = rank > 1.0 - fraction
            precision = float(y[selected].mean()) if selected.any() else float("nan")
            tag = int(fraction * 100)
            row[f"precision_top{tag}"] = precision
            row[f"lift_top{tag}"] = precision / base if base > 0 else float("nan")
    return row


def _cvar(values: np.ndarray, fraction: float = .10) -> float:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(finite):
        return float("nan")
    return float(finite[:max(1, int(math.ceil(len(finite) * fraction)))].mean())


def _k0_metrics(prediction: pd.DataFrame, spec: OpportunitySpec, month: pd.Timestamp) -> dict[str, Any]:
    threshold = float(_finite(prediction["K0_train_p80_expected_policy_net_bps"]).iloc[0])
    selected = prediction.loc[_finite(prediction["K0_expected_policy_net_bps"]).ge(threshold)].copy()
    valid = selected.loc[_valid_label(selected)].copy()
    net = _finite(valid["policy_net_bps"]).to_numpy(float)
    return {
        "arm": spec.name, "held_month": month.strftime("%Y-%m"), "threshold_bps": threshold,
        "scored_candidates": int(len(prediction)), "selected_candidates": int(len(selected)),
        "outcome_known_candidates": int(len(valid)), "outcome_coverage": float(len(valid) / len(selected)) if len(selected) else float("nan"),
        "net_bps_per_trade": float(net.mean()) if len(net) else float("nan"), "total_net_bps": float(net.sum()) if len(net) else 0.0,
        "cvar10_bps": _cvar(net), "fraction_lt_neg200": float(np.mean(net < -200.0)) if len(net) else float("nan"),
        "fraction_lt_neg400": float(np.mean(net < -400.0)) if len(net) else float("nan"),
        "positive_fraction": float(np.mean(net > 0.0)) if len(net) else float("nan"),
    }


def _aggregate_o(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, group in monthly.groupby("arm", sort=True):
        for era in ("2024", "2025", "2026"):
            local = group.loc[group["held_month"].str.startswith(era)]
            if local.empty:
                continue
            row = {"arm": arm, "era": era, "months": int(len(local))}
            for column in ("valid_rows", "opportunity_prevalence", "auc", "prauc", "brier", "logloss", "precision_top10", "precision_top20", "precision_top30", "lift_top10", "lift_top20", "lift_top30"):
                row[column] = float(local[column].mean()) if column in local else float("nan")
            row["positive_lift20_months"] = int((local["lift_top20"] > 1.0).sum()) if "lift_top20" in local else 0
            row["positive_lift30_months"] = int((local["lift_top30"] > 1.0).sum()) if "lift_top30" in local else 0
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_k0(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, group in monthly.groupby("arm", sort=True):
        for era in ("2024", "2025", "2026"):
            local = group.loc[group["held_month"].str.startswith(era)]
            if local.empty:
                continue
            row = {"arm": arm, "era": era, "months": int(len(local))}
            for column in ("scored_candidates", "selected_candidates", "outcome_known_candidates", "total_net_bps"):
                row[column] = float(local[column].sum())
            # Per-trade aggregates are weighted by known selected outcomes.
            known = local["outcome_known_candidates"].to_numpy(float)
            weight = known / known.sum() if known.sum() else np.zeros(len(local))
            for column in ("net_bps_per_trade", "cvar10_bps", "fraction_lt_neg200", "fraction_lt_neg400", "positive_fraction", "outcome_coverage"):
                row[column] = float(np.nansum(local[column].to_numpy(float) * weight)) if column in local else float("nan")
            row["positive_months"] = int((local["net_bps_per_trade"] > 0.0).sum())
            row["worst_month_net_bps_per_trade"] = float(local["net_bps_per_trade"].min())
            rows.append(row)
    return pd.DataFrame(rows)


def _ranking(k0_era: pd.DataFrame) -> pd.DataFrame:
    control = k0_era.loc[(k0_era["arm"] == "O200_H12_control") & k0_era["era"].isin(("2025", "2026"))].set_index("era")
    rows: list[dict[str, Any]] = []
    for arm, group in k0_era.loc[k0_era["era"].isin(("2025", "2026"))].groupby("arm", sort=True):
        years = group.set_index("era")
        if not {"2025", "2026"}.issubset(years.index):
            continue
        candidate = years.loc[["2025", "2026"]]
        base = control.loc[["2025", "2026"]]
        participation = candidate["outcome_known_candidates"].sum() / max(base["outcome_known_candidates"].sum(), 1.0)
        row = {
            "arm": arm,
            "mean_net_bps_per_trade": float(np.average(candidate["net_bps_per_trade"], weights=np.maximum(candidate["outcome_known_candidates"], 1.0))),
            "total_net_bps": float(candidate["total_net_bps"].sum()),
            "participation_vs_control": float(participation),
            "worst_era_net_bps_per_trade": float(candidate["net_bps_per_trade"].min()),
            "worst_month_net_bps_per_trade": float(candidate["worst_month_net_bps_per_trade"].min()),
            "mean_cvar10_bps": float(np.average(candidate["cvar10_bps"], weights=np.maximum(candidate["outcome_known_candidates"], 1.0))),
            "both_eras_ge90": bool((candidate["net_bps_per_trade"] >= 90.0).all()),
            "participation_ge70pct": bool(participation >= .70),
        }
        row["advances_round1"] = bool(row["both_eras_ge90"] and row["participation_ge70pct"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["advances_round1", "mean_net_bps_per_trade", "total_net_bps"], ascending=[False, False, False], kind="stable")


def _table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    local = frame.loc[:, [column for column in columns if column in frame]]
    if local.empty:
        return "_No supported rows._"
    try:
        return local.to_markdown(index=False)
    except ImportError:
        return "\n".join([" | ".join(local.columns), " | ".join(["---"] * len(local.columns)), *[" | ".join(map(str, row)) for row in local.itertuples(index=False, name=None)]])


def _report(out: Path, *, o_era: pd.DataFrame, k0_era: pd.DataFrame, ranking: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Short P0 → O → C → K0 Round 1: opportunity definition sweep",
        "",
        "Research-only. The architecture is fixed to P0 → O → C → K0. Each arm conditions O, T5 C, and K0 on the same declared MFE horizon/threshold event.",
        "",
        "## Round-1 preliminary ranking (2025–2026)",
        "",
        _table(ranking, ["arm", "mean_net_bps_per_trade", "total_net_bps", "participation_vs_control", "worst_era_net_bps_per_trade", "worst_month_net_bps_per_trade", "mean_cvar10_bps", "both_eras_ge90", "participation_ge70pct", "advances_round1"]),
        "",
        "## O opportunity diagnostics by era",
        "",
        _table(o_era, ["arm", "era", "months", "valid_rows", "opportunity_prevalence", "auc", "prauc", "brier", "precision_top20", "lift_top20", "precision_top30", "lift_top30", "positive_lift20_months"]),
        "",
        "## K0 causal admission economics by era",
        "",
        _table(k0_era, ["arm", "era", "months", "outcome_known_candidates", "net_bps_per_trade", "total_net_bps", "cvar10_bps", "fraction_lt_neg200", "fraction_lt_neg400", "positive_months", "worst_month_net_bps_per_trade"]),
        "",
        "## Contract",
        "",
        "- Exact short policy outcome: decision at signal close +1h, exact policy path, H12 availability, cost recorded once.",
        "- Outer fit: `label_available_at < held_month_start`.",
        "- Inner OOF: each O/C fit uses `label_available_at < inner_validation_start`; this purges the 12h outcome horizon rather than relying on row order.",
        "- Rich MFE labels are training/evaluation labels only. Every target-free P0 candidate is scored; invalid paths are excluded after scoring.",
        "- K0 uses only `p(O) * mu1(C) + (1-p(O)) * shrunk_mu0`; selection is each outer fold's inner-OOF p80 expected-policy-EV threshold. No held top-k is used for admission.",
        "- 2024 feature/MDA development use is explicitly non-untouched. 2025–2026 remain model-selection evidence, not a new untouched promotion period.",
        "",
        "## Fold audit",
        "",
        _table(folds, ["arm", "held_month", "status", "held_rows", "outer_train_rows", "outer_c_train_rows", "inner_oof_rows", "inner_oof_months", "reason"]),
        "",
        "```json",
        json.dumps({key: manifest[key] for key in ("schema", "side", "scope", "opportunity_specs", "features", "policy", "causality")}, indent=2),
        "```",
        "",
    ]
    (out / "SHORT_P0_OC_K0_ROUND1_REPORT.md").write_text("\n".join(lines))


def run(
    *,
    population_roots: Sequence[Path], rich_labels_root: Path, feature_selection: Path,
    mda_ranking: Path, feature_panels: Sequence[Path], out: Path,
    start: pd.Timestamp, end: pd.Timestamp, seed: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    m4_fields = _load_m4_fields(population_roots[0])
    if any(_load_m4_fields(root) != m4_fields for root in population_roots[1:]):
        raise AssertionError("T5 conversion feature contracts disagree across P0 sources")
    f115 = _load_f115_selection(feature_selection)
    o_fields = _load_mda90_fields(mda_ranking, f115)
    population, population_hashes = _load_population(population_roots, m4_fields)
    feature_values, feature_hashes = _load_features(population, f115, feature_panels)
    shared = tuple(field for field in f115 if field in population.columns)
    check = population.loc[:, ["candidate_id", *shared]].merge(feature_values.loc[:, ["candidate_id", *shared]], on="candidate_id", how="left", suffixes=("_p0", "_panel"), validate="one_to_one")
    for field in shared:
        if not _numeric_equal(check[f"{field}_p0"], check[f"{field}_panel"]):
            raise AssertionError(f"target-free feature panel/P0 disagreement for {field}")
    new = tuple(field for field in f115 if field not in population.columns)
    frame = population.merge(feature_values.loc[:, ["candidate_id", *new]], on="candidate_id", how="left", validate="one_to_one")
    labels, label_hash = _load_rich_labels(rich_labels_root)
    frame = frame.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(frame) != len(population) or frame["candidate_id"].duplicated().any():
        raise AssertionError("target-free candidate identity changed while joining labels")
    valid = _valid_label(frame)
    for spec in ROUND1_SPECS:
        if frame.loc[valid, spec.label_field].isna().any():
            raise AssertionError(f"valid exact-policy labels lack {spec.label_field}")
    rows: list[pd.DataFrame] = []
    inner_rows: list[pd.DataFrame] = []
    o_metrics: list[dict[str, Any]] = []
    k0_metrics: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    months = pd.date_range(start.normalize().replace(day=1), end.normalize().replace(day=1), freq="MS", inclusive="left")
    for spec_index, spec in enumerate(ROUND1_SPECS):
        for month_index, month in enumerate(months):
            stop = month + pd.offsets.MonthBegin(1)
            held = frame.loc[frame["__decision_ts__"].ge(month) & frame["__decision_ts__"].lt(stop)].copy()
            if held.empty:
                continue
            train = frame.loc[
                frame["__decision_ts__"].lt(month) & frame["__label_available_at__"].lt(month) & _valid_label(frame)
            ].copy()
            try:
                prediction, inner, audit, inner_audit = _fit_outer_predict(
                    train, held, spec=spec, o_fields=o_fields, c_fields=m4_fields,
                    seed=seed + spec_index * 10_000 + month_index * 53,
                )
                prediction["arm"] = spec.name
                prediction["held_month"] = month.strftime("%Y-%m")
                rows.append(prediction)
                inner["arm"] = spec.name
                inner["held_month"] = month.strftime("%Y-%m")
                inner_rows.append(inner)
                o_metrics.append(_o_metrics(prediction, spec, month))
                k0_metrics.append(_k0_metrics(prediction, spec, month))
                fold_rows.append({"arm": spec.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "held_rows": len(held), **audit})
                for inner_row in inner_audit:
                    fold_rows.append({"arm": spec.name, "held_month": month.strftime("%Y-%m"), "status": f"inner_{inner_row.pop('status')}", "held_rows": len(held), **inner_row})
            except ValueError as error:
                fold_rows.append({"arm": spec.name, "held_month": month.strftime("%Y-%m"), "status": "skipped", "held_rows": len(held), "outer_train_rows": len(train), "outer_c_train_rows": int(_event(train, spec).sum()) if len(train) else 0, "reason": str(error)})
    if not rows:
        raise RuntimeError("Round 1 did not produce any strict-OOS O/C/K0 predictions")
    prediction_frame = pd.concat(rows, ignore_index=True)
    inner_frame = pd.concat(inner_rows, ignore_index=True)
    o_monthly = pd.DataFrame(o_metrics)
    k0_monthly = pd.DataFrame(k0_metrics)
    folds = pd.DataFrame(fold_rows)
    o_era = _aggregate_o(o_monthly)
    k0_era = _aggregate_k0(k0_monthly)
    ranking = _ranking(k0_era)
    out.mkdir(parents=True)
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "scope": "Round 1 research-only P0→O→C→K0 common opportunity definition sweep; no live/canonical change",
        "opportunity_specs": [{"name": spec.name, "horizon_hours": spec.horizon_hours, "threshold_bps": spec.threshold_bps, "event": spec.description} for spec in ROUND1_SPECS],
        "features": {"O": {"contract": "frozen MDA90", "fields": list(o_fields)}, "C": {"contract": "frozen current T5 M4", "fields": list(m4_fields)}},
        "policy": "short exact policy outcome; H12 availability; costs embedded once",
        "causality": {"outer_fit": "label_available_at < held month start", "inner_fit": "label_available_at < inner validation start", "labels": "exact rich MFE labels are supervised-only", "inference": "score all target-free P0 candidates", "admission": "outer-train inner-OOF p80 K0 expected-policy-EV"},
        "sources": {"population_manifest_hashes": population_hashes, "rich_labels_manifest_sha256": label_hash, "feature_panels_sha256": feature_hashes, "feature_selection_sha256": _sha256(feature_selection), "mda_ranking_sha256": _sha256(mda_ranking)},
        "selection_gate": {"both_2025_2026_net_bps_per_trade_ge": 90.0, "participation_vs_control_ge": .70, "control": "O200_H12_control"},
    }
    prediction_frame.to_parquet(out / "round1_outer_oof_predictions.parquet", index=False, compression="zstd")
    inner_frame.to_parquet(out / "round1_inner_oof_ledger.parquet", index=False, compression="zstd")
    o_monthly.to_parquet(out / "round1_o_monthly_metrics.parquet", index=False, compression="zstd")
    o_era.to_parquet(out / "round1_o_era_metrics.parquet", index=False, compression="zstd")
    k0_monthly.to_parquet(out / "round1_k0_monthly_metrics.parquet", index=False, compression="zstd")
    k0_era.to_parquet(out / "round1_k0_era_metrics.parquet", index=False, compression="zstd")
    ranking.to_parquet(out / "round1_ranking.parquet", index=False, compression="zstd")
    folds.to_parquet(out / "round1_fold_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _report(out, o_era=o_era, k0_era=k0_era, ranking=ranking, folds=folds, manifest=manifest)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--population-root", type=Path, action="append")
    parser.add_argument("--rich-labels", type=Path, default=DEFAULT_RICH_LABELS)
    parser.add_argument("--feature-selection", type=Path, default=DEFAULT_FEATURE_SELECTION)
    parser.add_argument("--mda-ranking", type=Path, default=DEFAULT_MDA_RANKING)
    parser.add_argument("--feature-panel", type=Path, action="append")
    args = parser.parse_args()
    print(run(
        population_roots=tuple(args.population_root or DEFAULT_POPULATION_ROOTS), rich_labels_root=args.rich_labels,
        feature_selection=args.feature_selection, mda_ranking=args.mda_ranking,
        feature_panels=tuple(args.feature_panel or DEFAULT_FEATURE_PANELS), out=args.out,
        start=_utc(args.start), end=_utc(args.end), seed=args.seed,
    ))


if __name__ == "__main__":
    main()
