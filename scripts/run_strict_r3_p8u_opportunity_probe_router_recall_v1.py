#!/usr/bin/env python3
"""Strict fold-local data-driven opportunity probes for P8U Router recall.

This is deliberately an *offline Router-stage research program*.  It never
alters or scores downstream Base/Meta/MC1/admission/portfolio models.  A
target-free broad candidate universe is fixed before policy outcomes are
joined.  All geometry discovery, feature preprocessing, specialist fitting,
ranking references, and configuration choices are train/inner-fold local.

Development uses 2024--2025 chronological nested folds.  A single fully
frozen specification is then fitted before 2026 and evaluated unchanged on
2026.  The output is evidence for a possible Router rescue architecture, not
a promotion or a live-stack artifact.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

try:  # Diagnostic only; its output cannot become a canonical category system.
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent diagnostic.
    hdbscan = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.opportunity_probe_contract import (  # noqa: E402
    OPPORTUNITY_PROBE_COVERAGE_QUALIFIED_FEATURE_KEYS,
    OPPORTUNITY_PROBE_DISCOVERY_FEATURE_KEYS,
    OPPORTUNITY_PROBE_PREDICTIVE_FEATURE_KEYS,
)
from extreme_price_movements.archetype_recovery import (  # noqa: E402
    StructuralQualification,
    archetypal_memberships,
    farthest_point_archetypes,
    matched_signature_correlation,
    qualification_metrics,
    structural_signatures,
)


# Keep the offline P8U runner independent from the monolithic runtime config.
# The three keys remain the public contract names consumed by the JSON spec.
_P8U_FEATURE_CONTRACTS = {
    "OPPORTUNITY_PROBE_DISCOVERY_FEATURE_KEYS": OPPORTUNITY_PROBE_DISCOVERY_FEATURE_KEYS,
    "OPPORTUNITY_PROBE_PREDICTIVE_FEATURE_KEYS": OPPORTUNITY_PROBE_PREDICTIVE_FEATURE_KEYS,
    "OPPORTUNITY_PROBE_COVERAGE_QUALIFIED_FEATURE_KEYS": OPPORTUNITY_PROBE_COVERAGE_QUALIFIED_FEATURE_KEYS,
}


SCHEMA = "strict_r3_p8u_opportunity_probe_router_recall_v2"
ARCHETYPE_RECOVERY_SCHEMA = "strict_r3_p8u_archetype_recovery_v1"
MAX_LABEL_HORIZON_HOURS = 12
IDENTITY_COLUMNS = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
LABEL_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_label_available_ts", "policy_cost_bps", "policy_exit_reason",
    "label_source_complete_1m_path",
)
FORBIDDEN_PROBE_INPUT_TOKENS = (
    "router_", "prediction_", "label_", "policy_", "outcome_", "future_",
    "candidate_id", "__symbol__", "side_name", "__decision_ts__", "timestamp", "net_bps", "gross_bps",
)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_fields(fields: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, fields)).encode()).hexdigest()


def _deep_merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_config(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load an immutable config or a hash-bound research overlay."""
    raw = json.loads(config_path.read_text())
    parent = raw.pop("extends", None)
    if not parent:
        return raw
    parent_path = config_path.parent / str(parent["path"])
    expected = str(parent.get("sha256", ""))
    actual = _sha256_file(parent_path)
    if not expected or actual != expected:
        raise ValueError(f"parent config hash mismatch for {parent_path}: expected {expected}, got {actual}")
    return _deep_merge_config(_load_config(parent_path), raw)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _write_parquet_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _month_starts(start: object, end: object) -> list[pd.Timestamp]:
    return list(pd.date_range(_utc(start).normalize().replace(day=1), _utc(end).normalize().replace(day=1), freq="MS", tz="UTC"))


def _month_path(root: Path, month: pd.Timestamp, filename: str) -> Path:
    return root / f"month={month:%Y-%m}" / filename


def _source_for(month: pd.Timestamp, sources: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for source in sources:
        if _utc(source["start"]) <= month <= _utc(source["end"]):
            return source
    raise KeyError(f"no source declared for {month:%Y-%m}")


def _assert_causal_feature_contract(fields: Sequence[str]) -> tuple[str, ...]:
    result = tuple(map(str, fields))
    if not result or len(result) != len(set(result)):
        raise AssertionError("probe feature contract must be a non-empty unique list")
    illegal = [field for field in result if any(token in field.lower() for token in FORBIDDEN_PROBE_INPUT_TOKENS)]
    if illegal:
        raise AssertionError(f"probe contract contains forbidden target/model identity fields: {illegal}")
    return result


def _read_panel(
    config: dict[str, Any], fields: Sequence[str], sidecar_fields: Sequence[str], *, start: object | None = None, end: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Read only target-free features/router scores; labels are deliberately separate."""
    fields = tuple(fields)
    sidecar_fields = tuple(sidecar_fields)
    base_fields = tuple(field for field in fields if field not in set(sidecar_fields))
    unexpected = set(sidecar_fields).difference(fields)
    if unexpected:
        raise AssertionError(f"sidecar contract is not a subset of predictive contract: {sorted(unexpected)}")
    feature_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    period_start = config["research_period"][0] if start is None else start
    period_end = config["research_period"][1] if end is None else end
    for month in _month_starts(period_start, period_end):
        feature_source = _source_for(month, config["feature_sources"])
        score_source = _source_for(month, config["router_score_sources"])
        feature_path = _month_path(ROOT / feature_source["root"], month, "causal_feature_universe.parquet")
        score_root = ROOT / score_source["root"]
        score_path = score_root / "target_free_scores" / f"month={month:%Y-%m}.parquet"
        if not score_path.exists():
            score_path = _month_path(score_root, month, "raw_oos_predictions.parquet")
        if not feature_path.exists() or not score_path.exists():
            raise FileNotFoundError(f"missing declared monthly source: {feature_path} / {score_path}")
        available_features = set(pq.ParquetFile(feature_path).schema.names)
        missing = sorted(set(IDENTITY_COLUMNS).union(base_fields).difference(available_features))
        if missing:
            raise AssertionError(f"{month:%Y-%m} feature source lacks contract fields: {missing}")
        feature = pd.read_parquet(feature_path, columns=[*IDENTITY_COLUMNS, *base_fields])
        sidecar_path = ROOT / config["probe_feature_sidecar_root"] / f"month={month:%Y-%m}" / "causal_probe_intraday_features.parquet"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"missing declared causal probe sidecar: {sidecar_path}")
        sidecar_available = set(pq.ParquetFile(sidecar_path).schema.names)
        sidecar_missing = set((*IDENTITY_COLUMNS, *sidecar_fields)).difference(sidecar_available)
        if sidecar_missing:
            raise AssertionError(f"{sidecar_path} lacks sidecar contract fields: {sorted(sidecar_missing)}")
        sidecar = pd.read_parquet(sidecar_path, columns=[*IDENTITY_COLUMNS, *sidecar_fields])
        if sidecar["candidate_id"].duplicated().any():
            raise AssertionError(f"duplicate candidate identity in causal probe sidecar {month:%Y-%m}")
        identity_check = feature[list(IDENTITY_COLUMNS)].merge(
            sidecar[list(IDENTITY_COLUMNS)], on="candidate_id", how="left", suffixes=("_source", "_sidecar"), validate="one_to_one",
        )
        for field in IDENTITY_COLUMNS[1:]:
            left, right = identity_check[f"{field}_source"], identity_check[f"{field}_sidecar"]
            if field == "__decision_ts__":
                left, right = pd.to_datetime(left, utc=True), pd.to_datetime(right, utc=True)
            if not left.eq(right).fillna(False).all():
                raise AssertionError(f"{month:%Y-%m}: causal probe sidecar changed target-free identity field {field}")
        feature = feature.merge(sidecar[["candidate_id", *sidecar_fields]], on="candidate_id", how="left", validate="one_to_one")
        if feature[list(sidecar_fields)].isna().all(axis=None):
            raise AssertionError(f"{month:%Y-%m}: causal probe sidecar joined no values")
        score_probe = pd.read_parquet(score_path)
        rank_column = next((name for name in ("router_primary_rank", "router_primary_only_rank", "router_full_ae_rank") if name in score_probe.columns), None)
        if rank_column is None:
            raise AssertionError(f"{score_path} lacks an accepted Router rank column")
        score = score_probe[["candidate_id", rank_column]].rename(columns={rank_column: "router_rank"})
        feature["__decision_ts__"] = pd.to_datetime(feature["__decision_ts__"], utc=True)
        feature["side_name"] = feature["side_name"].astype(str).str.lower()
        feature.drop(feature.index[feature["side_name"] != str(config["side"]).lower()], inplace=True)
        for frame in (feature, score):
            if frame["candidate_id"].duplicated().any():
                raise AssertionError(f"duplicate candidate identity in target-free source {month:%Y-%m}")
        merged = feature.merge(score[["candidate_id", "router_rank"]], on="candidate_id", how="inner", validate="one_to_one")
        if len(merged) != len(feature):
            raise AssertionError(f"Router score join lost target-free candidates in {month:%Y-%m}")
        feature_rows.append(merged)
        audit.append({
            "month": str(month.date()), "feature_root": feature_source["root"], "router_root": score_source["root"],
            "rows_feature": int(len(feature)), "rows_router": int(len(score)), "rows_joined": int(len(merged)),
            "feature_path": str(feature_path.relative_to(ROOT)), "score_path": str(score_path.relative_to(ROOT)),
            "sidecar_path": str(sidecar_path.relative_to(ROOT)),
            "sidecar_field_coverage": {field: float(feature[field].notna().mean()) for field in sidecar_fields},
        })
    target_free = pd.concat(feature_rows, ignore_index=True)
    if target_free["candidate_id"].duplicated().any():
        raise AssertionError("candidate IDs must be unique across target-free monthly sources")
    # The second returned frame is only an explicit source-audit view; no label is
    # joined until _attach_outcomes is called below.
    return target_free, target_free[list(IDENTITY_COLUMNS) + ["router_rank"]].copy(), audit


def _load_labels(
    path: str,
    *,
    decision_start: object | None = None,
    decision_end: object | None = None,
) -> pd.DataFrame:
    """Load only the label time range represented by the sealed candidate panel.

    The policy ledger is intentionally append-only and can be materially larger
    than a bounded outer-fold experiment.  Filtering on its persisted decision
    timestamp changes neither label identity nor value; it only avoids holding
    unrelated future/history labels alongside the target-free feature panel.
    """
    resolved = ROOT / path
    filters: list[tuple[str, str, object]] = []
    if decision_start is not None:
        filters.append(("__decision_ts__", ">=", _utc(decision_start).to_pydatetime()))
    if decision_end is not None:
        filters.append(("__decision_ts__", "<", _utc(decision_end).to_pydatetime()))
    labels = pd.read_parquet(
        resolved,
        columns=list(LABEL_COLUMNS),
        filters=filters or None,
    )
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("policy label source contains duplicate candidate IDs")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    labels["policy_net_bps"] = pd.to_numeric(labels["policy_net_bps"], errors="coerce")
    labels["policy_path_valid"] = labels["policy_path_valid"].fillna(False).astype(bool)
    return labels


def _attach_outcomes(target_free: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach labels in place after sealing the target-free candidate universe.

    A pandas ``merge`` makes a second complete copy of the 100+ field feature
    panel before appending a handful of label columns.  That is unnecessary
    and can exhaust memory on the full development history.  The immutable
    target-free identity artifact has already been persisted at this point, so
    index-aligned label assignment gives the identical one-to-one left-join
    semantics without duplicating the causal feature substrate.
    """
    if target_free["candidate_id"].duplicated().any() or labels["candidate_id"].duplicated().any():
        raise AssertionError("outcome attachment requires unique candidate identities")
    label_index = labels.set_index("candidate_id", drop=False)
    target_free["label_join_status"] = np.where(
        target_free["candidate_id"].isin(label_index.index), "both", "left_only",
    )
    for column in labels.columns:
        if column == "candidate_id":
            continue
        target_free[column] = target_free["candidate_id"].map(label_index[column])
    attached = target_free
    attached["label_valid"] = (
        attached["policy_path_valid"].eq(True)
        & attached["policy_label_available_ts"].notna()
        & attached["policy_net_bps"].notna()
    )
    atr_bps = (
        pd.to_numeric(attached["probe_atr_bps_14h"], errors="coerce")
        if "probe_atr_bps_14h" in attached
        else pd.Series(np.nan, index=attached.index, dtype=float)
    )
    attached["probe_target_valid"] = (
        attached["label_valid"] & np.isfinite(atr_bps) & atr_bps.gt(1e-6)
    )
    attached["policy_net_atr"] = (
        pd.to_numeric(attached["policy_net_bps"], errors="coerce") / atr_bps
    ).replace([np.inf, -np.inf], np.nan)
    return attached


def _eligible_labels(frame: pd.DataFrame, cutoff: object) -> pd.DataFrame:
    cutoff_ts = _utc(cutoff)
    return frame.loc[frame["probe_target_valid"] & (frame["policy_label_available_ts"] < cutoff_ts)].copy()


def _coverage_qualified_fields(
    frame: pd.DataFrame, candidate_fields: Sequence[str], *, minimum_coverage: float,
    fold: str, stage: str,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Freeze a target-free, prior-only feature contract with no sparse fields.

    Availability is deliberately calculated on the target-free population before
    any label join, target construction, Router result, or held data is used.
    Missing values are still robustly imputed for occasional source outages, but
    a feature that is structurally sparse is not permitted into the fold.
    """
    fields = tuple(candidate_fields)
    available = frame.reindex(columns=list(fields)).notna().mean(axis=0)
    audit = pd.DataFrame({
        "fold": fold,
        "stage": stage,
        "field": fields,
        "coverage": [float(available.get(field, 0.0)) for field in fields],
    })
    audit["minimum_coverage"] = float(minimum_coverage)
    audit["kept"] = audit["coverage"] >= float(minimum_coverage)
    retained = tuple(audit.loc[audit["kept"], "field"].tolist())
    if len(retained) < 4:
        raise RuntimeError(
            f"{fold}/{stage}: coverage gate retained only {len(retained)} fields at {minimum_coverage:.0%}"
        )
    return retained, audit


def _sample_indices(n: int, maximum: int, seed: int) -> np.ndarray:
    if n <= maximum:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=maximum, replace=False))


@dataclass
class RobustPreprocessor:
    fields: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    percentile_grid: np.ndarray | None = None
    percentile_levels: np.ndarray | None = None

    @classmethod
    def fit(cls, frame: pd.DataFrame, candidate_fields: Sequence[str], cfg: dict[str, Any], seed: int) -> "RobustPreprocessor":
        selected = _sample_indices(len(frame), int(cfg["max_fit_rows"]), seed)
        raw = frame.iloc[selected][list(candidate_fields)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        lo = np.nanquantile(raw, float(cfg["winsor_lower"]), axis=0)
        hi = np.nanquantile(raw, float(cfg["winsor_upper"]), axis=0)
        median = np.nanmedian(raw, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        raw = np.clip(np.where(np.isfinite(raw), raw, median), lo, hi)
        q25, q75 = np.quantile(raw, (0.25, 0.75), axis=0)
        scale = q75 - q25
        # A pure IQR scale is unstable for zero-inflated structural fields: a
        # near-zero IQR can coexist with a meaningful causal activation tail,
        # amplifying that tail by orders of magnitude and collapsing all
        # subsequent geometry into one pseudo-dimension.  The recovery config
        # explicitly enables a train-only central-span fallback.  Sparse fields
        # with no meaningful span remain excluded as near-constant.
        fallback_quantiles = cfg.get("zero_iqr_fallback_quantiles")
        if fallback_quantiles is not None:
            lower, upper = map(float, fallback_quantiles)
            qlo, qhi = np.quantile(raw, (lower, upper), axis=0)
            fallback = 0.5 * (qhi - qlo)
            meaningful = np.maximum(np.maximum(np.abs(qlo), np.abs(qhi)), 1e-12)
            scale = np.where(scale <= float(cfg.get("zero_iqr_relative_floor", 1e-5)) * meaningful, fallback, scale)
        keep = np.isfinite(scale) & (scale > 1e-8) & np.isfinite(lo) & np.isfinite(hi)
        if keep.sum() < 4:
            raise ValueError("too few non-constant discovery fields after fold-local preprocessing")
        fields = tuple(np.asarray(candidate_fields)[keep].tolist())
        lo, hi, median, scale = lo[keep], hi[keep], median[keep], scale[keep]
        normalized_raw = np.clip(np.where(np.isfinite(raw[:, keep]), raw[:, keep], median), lo, hi)
        percentile_grid: np.ndarray | None = None
        percentile_levels: np.ndarray | None = None
        if str(cfg.get("discovery_scaling", "robust_iqr")) == "percentile":
            percentile_levels = np.linspace(
                float(cfg.get("percentile_lower", .001)), float(cfg.get("percentile_upper", .999)),
                int(cfg.get("percentile_grid_size", 257)), dtype=np.float64,
            )
            percentile_grid = np.quantile(normalized_raw, percentile_levels, axis=0)
            normalized = np.empty_like(normalized_raw, dtype=np.float32)
            for column in range(normalized_raw.shape[1]):
                normalized[:, column] = (
                    2.0 * np.interp(normalized_raw[:, column], percentile_grid[:, column], percentile_levels) - 1.0
                )
        else:
            normalized = (normalized_raw - median) / scale
        # Deterministic high-correlation collapse; retain the first declared
        # representative, so no outcome or Router score can affect the choice.
        corr = np.abs(np.corrcoef(normalized, rowvar=False))
        retained: list[int] = []
        threshold = float(cfg["correlation_collapse_abs"])
        for index in range(len(fields)):
            if not retained or np.all(corr[index, retained] < threshold):
                retained.append(index)
        take = np.asarray(retained, dtype=int)
        return cls(
            tuple(np.asarray(fields)[take]), lo[take], hi[take], median[take], scale[take],
            None if percentile_grid is None else percentile_grid[:, take], percentile_levels,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.loc[:, list(self.fields)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, self.median)
        raw = np.clip(raw, self.lo, self.hi)
        if self.percentile_grid is not None and self.percentile_levels is not None:
            transformed = np.empty_like(raw, dtype=np.float32)
            for column in range(raw.shape[1]):
                transformed[:, column] = (
                    2.0 * np.interp(raw[:, column], self.percentile_grid[:, column], self.percentile_levels) - 1.0
                )
            return transformed
        return ((raw - self.median) / self.scale).astype(np.float32)

    def subset(self, positions: Sequence[int]) -> "RobustPreprocessor":
        """Return an immutable selected-field view of this train-only state."""
        take = np.asarray(positions, dtype=int)
        return RobustPreprocessor(
            tuple(np.asarray(self.fields)[take].tolist()), self.lo[take], self.hi[take],
            self.median[take], self.scale[take],
            None if self.percentile_grid is None else self.percentile_grid[:, take], self.percentile_levels,
        )


@dataclass
class CategoryModel:
    algorithm: str
    covariance: str | None
    k: int
    preprocessor: RobustPreprocessor
    pca: PCA | None
    nmf_shift: np.ndarray | None
    archetype_temperature: float | None
    model: Any
    centers: np.ndarray
    seed: int

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(frame)
        if self.pca is not None:
            matrix = self.pca.transform(matrix)
        if self.nmf_shift is not None:
            matrix = np.maximum(matrix + self.nmf_shift, 0.0)
        return matrix.astype(np.float32)

    def membership(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(frame)
        if self.algorithm == "gmm":
            return self.model.predict_proba(matrix).astype(np.float32)
        if self.algorithm == "archetypal":
            membership, _ = archetypal_memberships(
                matrix, self.centers, temperature=self.archetype_temperature,
            )
            return membership
        raw = self.model.transform(matrix)
        total = raw.sum(axis=1, keepdims=True)
        return (raw / np.maximum(total, 1e-8)).astype(np.float32)


def _fit_category_model(
    frame: pd.DataFrame, fields: Sequence[str], preprocess_cfg: dict[str, Any], *, algorithm: str,
    covariance: str | None, k: int, seed: int,
    preprocessor: RobustPreprocessor | None = None, matrix_all: np.ndarray | None = None,
    archetype_temperature_multiplier: float = 1.0,
) -> CategoryModel:
    if (preprocessor is None) != (matrix_all is None):
        raise ValueError("preprocessor and matrix_all must be supplied together")
    if preprocessor is None:
        preprocessor = RobustPreprocessor.fit(frame, fields, preprocess_cfg, seed)
        matrix_all = preprocessor.transform(frame)
    assert matrix_all is not None
    sample = matrix_all[_sample_indices(len(matrix_all), int(preprocess_cfg["max_discovery_rows"]), seed + 1)]
    pca: PCA | None = None
    nmf_shift: np.ndarray | None = None
    archetype_temperature: float | None = None
    if algorithm == "gmm":
        pca = PCA(n_components=float(preprocess_cfg["pca_explained_variance"]), svd_solver="full", random_state=seed)
        sample_fit = pca.fit_transform(sample)
        model = GaussianMixture(
            n_components=k, covariance_type=str(covariance), random_state=seed,
            reg_covar=float(preprocess_cfg.get("gmm_reg_covar", 1e-3)),
            max_iter=int(preprocess_cfg.get("gmm_max_iter", 80)), n_init=int(preprocess_cfg.get("gmm_n_init", 3)),
            init_params="kmeans",
        ).fit(sample_fit)
        centers = model.means_.astype(np.float32)
    elif algorithm == "nmf":
        nmf_shift = np.maximum(-np.nanmin(sample, axis=0), 0.0) + 1e-5
        sample_fit = np.maximum(sample + nmf_shift, 0.0)
        model = NMF(n_components=k, init="nndsvda", max_iter=int(preprocess_cfg.get("nmf_max_iter", 300)), random_state=seed, alpha_W=0.02, alpha_H=0.02, l1_ratio=0.10)
        weights = model.fit_transform(sample_fit)
        centers = model.components_.astype(np.float32)
    elif algorithm == "archetypal":
        # Practical archetypal-analysis approximation.  A PCA-whitened
        # MiniBatchKMeans fit finds broad structural basins; each centroid is
        # pushed modestly away from the population centre and snapped to an
        # observed row.  Rows are then soft convex combinations of these
        # observed, structurally extreme representatives.  This avoids the
        # single-outlier attraction of raw farthest-point seeding while
        # retaining an interpretable archetype-vector contract.
        pca = PCA(n_components=float(preprocess_cfg["pca_explained_variance"]), whiten=True, svd_solver="full", random_state=seed)
        sample_fit = pca.fit_transform(sample)
        model = MiniBatchKMeans(
            n_clusters=k, random_state=seed, batch_size=min(2048, len(sample_fit)),
            n_init=int(preprocess_cfg.get("archetypal_n_init", 5)), max_iter=int(preprocess_cfg.get("archetypal_max_iter", 200)),
        ).fit(sample_fit)
        centre = sample_fit.mean(axis=0, keepdims=True)
        outward = centre + float(preprocess_cfg.get("archetypal_outward_scale", 1.25)) * (model.cluster_centers_ - centre)
        selected: list[int] = []
        for target in outward:
            order = np.argsort(np.einsum("ij,ij->i", sample_fit - target, sample_fit - target), kind="stable")
            selected.append(next((int(index) for index in order if int(index) not in selected), int(order[0])))
        centers = sample_fit[np.asarray(selected, dtype=int)].astype(np.float32)
        _, inferred_temperature = archetypal_memberships(sample_fit, centers)
        archetype_temperature = float(inferred_temperature * float(archetype_temperature_multiplier))
    else:  # pragma: no cover - guarded by configuration.
        raise ValueError(f"unsupported category algorithm {algorithm}")
    return CategoryModel(
        algorithm, covariance, k, preprocessor, pca, nmf_shift,
        archetype_temperature, model, centers, seed,
    )


def _membership_stability(model: CategoryModel, frame: pd.DataFrame, *, seed: int, maximum: int) -> float:
    """Bootstrap/refit agreement, matched by Hungarian centroid similarity."""
    if len(frame) < max(1000, model.k * 40):
        return float("nan")
    sample = frame.iloc[_sample_indices(len(frame), maximum, seed)]
    try:
        peer = _fit_category_model(
            sample, model.preprocessor.fields,
            {"winsor_lower": 0.005, "winsor_upper": 0.995, "correlation_collapse_abs": 0.985,
             "pca_explained_variance": 0.9, "max_fit_rows": min(len(sample), 12_000), "max_discovery_rows": min(len(sample), 5_000), "gmm_max_iter": 50, "nmf_max_iter": 80},
            algorithm=model.algorithm, covariance=model.covariance, k=model.k, seed=seed + 31,
        )
        left, right = model.membership(sample), peer.membership(sample)
        # Match memberships, not raw coordinate names.  Pearson is sufficient
        # here because it is a stability diagnostic, never a score-time field.
        cross = np.corrcoef(left.T, right.T)[: model.k, model.k :]
        rows, cols = linear_sum_assignment(-np.nan_to_num(cross, nan=-1.0))
        return float(np.nanmean(cross[rows, cols]))
    except Exception:
        return float("nan")


class ArchetypeDiscoveryFailure(RuntimeError):
    """Raised before probe fitting when no causal structural decomposition qualifies."""

    def __init__(self, fold: str, metrics: pd.DataFrame, definitions: pd.DataFrame) -> None:
        super().__init__(f"{fold}: archetype layer failed structural qualification; probes were not trained")
        self.fold = fold
        self.metrics = metrics
        self.definitions = definitions


def _temporal_structural_stability(
    frame: pd.DataFrame, preprocessor: RobustPreprocessor, preprocess_cfg: dict[str, Any],
    *, algorithm: str, covariance: str | None, k: int, seed: int, maximum: int,
    archetype_temperature_multiplier: float = 1.0,
) -> tuple[float, int]:
    """Independent adjacent-window refits, matched by original-feature signatures.

    Both preprocessing and the unsupervised decomposition are independently
    fitted on their own chronological window.  Signatures are subsequently
    expressed in a common *target-free* reference scale solely for matching.
    This avoids giving either side a shared scaling contract that could hide a
    regime-sensitive structural definition, while keeping cosine signatures
    comparable in original field order.
    """
    ordered = np.asarray(
        np.argsort(pd.to_datetime(frame["__decision_ts__"], utc=True).astype("int64"), kind="stable"),
        dtype=np.int64,
    )
    midpoint = len(ordered) // 2
    if midpoint < max(200, k * 20) or len(ordered) - midpoint < max(200, k * 20):
        return float("nan"), 0
    left, right = ordered[:midpoint], ordered[midpoint:]
    left = left[_sample_indices(len(left), min(maximum, len(left)), seed + 17)]
    right = right[_sample_indices(len(right), min(maximum, len(right)), seed + 19)]
    left_frame, right_frame = frame.iloc[left], frame.iloc[right]
    # ``preprocessor`` is comparison-only.  Each fitted model below builds its
    # own winsorisation / percentile contract from its own adjacent window.
    left_comparison_matrix = preprocessor.transform(left_frame)
    right_comparison_matrix = preprocessor.transform(right_frame)
    try:
        first = _fit_category_model(
            left_frame, preprocessor.fields, preprocess_cfg, algorithm=algorithm,
            covariance=covariance, k=k, seed=seed + 101,
            archetype_temperature_multiplier=archetype_temperature_multiplier,
        )
        second = _fit_category_model(
            right_frame, preprocessor.fields, preprocess_cfg, algorithm=algorithm,
            covariance=covariance, k=k, seed=seed + 211,
            archetype_temperature_multiplier=archetype_temperature_multiplier,
        )
        left_membership = first.membership(left_frame)
        right_membership = second.membership(right_frame)
        left_signatures = structural_signatures(left_comparison_matrix, left_membership)
        right_signatures = structural_signatures(right_comparison_matrix, right_membership)
        value, _ = matched_signature_correlation(left_signatures, right_signatures)
        return value, int(min(len(left), len(right)))
    except Exception:
        return float("nan"), 0


def _strict_structural_discovery_candidates(
    train_target_free: pd.DataFrame,
    fields: Sequence[str],
    config: dict[str, Any],
    *, seed: int, fold_name: str,
) -> tuple[list[tuple[dict[str, Any], CategoryModel]], pd.DataFrame, pd.DataFrame]:
    """Select a latent geometry purely from earlier target-free structure.

    This is intentionally independent of ``inner_outcomes``.  An earlier P8U
    version selected K partly from economic and Router separation then allowed a
    degenerate K=2 result through.  The recovery contract may not train a probe
    unless every structural gate passes first.
    """
    discovery = config["discovery"]
    preprocess = config["preprocessing"]
    candidate_k = [int(value) for value in discovery["k_values"] if 4 <= int(value) <= 10]
    if not candidate_k:
        raise ValueError("strict archetype recovery requires K values in [4, 10]")
    frozen_selection = discovery.get("frozen_structural_selection", {})
    frozen_spec = dict(frozen_selection.get("folds", {}).get(fold_name, {}))
    frozen_controls = discovery.get("frozen_structural_controls", {})
    frozen_control_spec = dict(frozen_controls.get("folds", {}).get(fold_name, {}))
    candidates: list[dict[str, Any]] = []

    def append_frozen(specification: dict[str, Any], *, role: str) -> None:
        algorithm = str(specification["algorithm"])
        covariance = specification.get("covariance")
        k = int(specification["k"])
        if algorithm not in {"nmf", "archetypal", "gmm"} or k not in candidate_k:
            raise ValueError(f"{fold_name}: invalid frozen structural {role} {specification!r}")
        if role == "control" and algorithm != "gmm":
            raise ValueError(f"{fold_name}: frozen constrained control must be a GMM")
        candidates.append({
            "algorithm": algorithm,
            "covariance": None if covariance is None else str(covariance),
            "k": k,
            "archetype_temperature_multiplier": float(specification.get("archetype_temperature_multiplier", 1.0)),
            # A target-free control must preserve its sealed seed rather than
            # acquire a fold-local enumeration seed during supervised reuse.
            "seed": int(specification.get("seed", seed + len(candidates) * 101)),
            "frozen_role": role,
        })

    if frozen_spec:
        append_frozen(frozen_spec, role="primary")
        if frozen_control_spec:
            append_frozen(frozen_control_spec, role="control")
    else:
        algorithms = tuple(discovery.get("algorithms", ["nmf", "archetypal", "gmm"]))
        covariances = tuple(discovery.get("gmm_covariances", ["tied", "diag"]))
        for algorithm in algorithms:
            if algorithm == "gmm":
                candidates.extend({
                    "algorithm": algorithm, "covariance": str(covariance), "k": k,
                    "archetype_temperature_multiplier": 1.0, "seed": int(seed + len(candidates) * 101),
                    "frozen_role": "candidate",
                } for covariance in covariances for k in candidate_k)
            elif algorithm in {"nmf", "archetypal"}:
                multipliers = discovery.get("archetypal_temperature_multipliers", [1.0]) if algorithm == "archetypal" else [1.0]
                candidates.extend({
                    "algorithm": algorithm, "covariance": None, "k": k,
                    "archetype_temperature_multiplier": float(multiplier), "seed": int(seed + len(candidates) * 101),
                    "frozen_role": "candidate",
                } for k in candidate_k for multiplier in multipliers)
            else:
                raise ValueError(f"unsupported strict archetype discovery algorithm {algorithm!r}")
    preprocessor = RobustPreprocessor.fit(train_target_free, fields, preprocess, seed)
    evaluation = _sample_indices(
        len(train_target_free), min(int(discovery.get("max_qualification_rows", 50_000)), len(train_target_free)), seed + 7,
    )
    evaluation_frame = train_target_free.iloc[evaluation]
    evaluation_matrix = preprocessor.transform(evaluation_frame)
    fit_indices = _sample_indices(
        len(train_target_free), min(int(preprocess["max_discovery_rows"]), len(train_target_free)), seed + 11,
    )
    fit_frame = train_target_free.iloc[fit_indices]
    fit_matrix = preprocessor.transform(fit_frame)
    qualification_cfg = discovery["qualification"]
    metrics: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], CategoryModel]] = []
    for candidate in candidates:
        algorithm = str(candidate["algorithm"])
        covariance = candidate["covariance"]
        k = int(candidate["k"])
        temperature_multiplier = float(candidate["archetype_temperature_multiplier"])
        candidate_seed = int(candidate["seed"])
        frozen_role = str(candidate["frozen_role"])
        spec = {
            "algorithm": algorithm, "covariance": covariance, "k": k, "seed": candidate_seed,
            "archetype_temperature_multiplier": temperature_multiplier, "frozen_role": frozen_role,
        }
        try:
            model = _fit_category_model(
                fit_frame, fields, preprocess, algorithm=algorithm, covariance=covariance,
                k=k, seed=candidate_seed, preprocessor=preprocessor, matrix_all=fit_matrix,
                archetype_temperature_multiplier=temperature_multiplier,
            )
            membership = model.membership(evaluation_frame)
            qualification: StructuralQualification = qualification_metrics(
                evaluation_matrix, membership,
                max_mass_share=float(qualification_cfg["max_mass_share"]),
                min_ess_fraction=float(qualification_cfg["min_ess_fraction"]),
                max_pairwise_signature_cosine=float(qualification_cfg["max_pairwise_signature_cosine"]),
                min_median_max_membership=float(qualification_cfg["min_median_max_membership"]),
                min_median_second_membership=float(qualification_cfg["min_median_second_membership"]),
                max_median_effective_fraction=float(qualification_cfg["max_median_effective_fraction"]),
            )
            temporal_stability, stability_rows = _temporal_structural_stability(
                train_target_free, preprocessor, preprocess, algorithm=algorithm,
                covariance=covariance, k=k, seed=candidate_seed,
                maximum=int(discovery.get("max_stability_rows", 12_000)),
                archetype_temperature_multiplier=temperature_multiplier,
            )
            temporal_pass = bool(np.isfinite(temporal_stability) and temporal_stability >= float(qualification_cfg["min_temporal_signature_correlation"]))
            qualified = bool(qualification.local_pass and temporal_pass)
            # Purely structural Pareto-style scalar used only to choose among
            # already-qualified candidates.  It deliberately has no economic,
            # Router, label, or probe term.
            structural_score = (
                float(np.nan_to_num(temporal_stability, nan=-1.0))
                + 0.20 * qualification.min_ess_fraction
                + 0.15 * (1.0 - qualification.max_mass_share)
                + 0.15 * (1.0 - qualification.max_pairwise_signature_cosine)
                + 0.10 * qualification.median_max_membership
                - 0.02 * float(k)
                - 0.02 * math.log1p(max(qualification.reconstruction_mse, 0.0))
            )
            signatures = structural_signatures(evaluation_matrix, membership)
            mass = membership.sum(axis=0)
            ess = mass ** 2 / np.maximum(np.square(membership).sum(axis=0), 1e-12)
            record = {
                "fold": fold_name, **spec,
                "structural_selection_source": "frozen_target_free_artifact" if frozen_spec else "current_target_free_matrix",
                "status": "qualified" if qualified else "rejected_structural",
                "selected_inner": False,
                "structural_selection_score": float(structural_score),
                "stability": float(temporal_stability), "temporal_signature_correlation": float(temporal_stability),
                "stability_rows_per_window": stability_rows,
                "min_ess": float(qualification.min_ess), "mean_ess": float(np.mean(ess)),
                "min_ess_fraction": float(qualification.min_ess_fraction),
                "min_ess_meets_5pct": bool(qualification.min_ess_meets_5pct),
                "max_category_mass_share": float(qualification.max_mass_share),
                "max_pairwise_signature_cosine": float(qualification.max_pairwise_signature_cosine),
                "median_max_membership": float(qualification.median_max_membership),
                "median_second_membership": float(qualification.median_second_membership),
                "median_membership_entropy": float(qualification.median_membership_entropy),
                "median_effective_archetype_count": float(qualification.median_effective_archetype_count),
                "reconstruction_mse": float(qualification.reconstruction_mse),
                "gate_anti_collapse": qualification.anti_collapse_pass,
                "gate_support": qualification.support_pass,
                "gate_membership_sparsity": qualification.sparsity_pass,
                "gate_membership_overlap": qualification.overlap_pass,
                "gate_structural_distinctness": qualification.distinctness_pass,
                "gate_temporal_stability": temporal_pass,
                "preprocessor_fields": list(preprocessor.fields),
                "qualification_population_rows": int(len(evaluation)),
            }
            metrics.append(record)
            fitted.append((spec, model))
            order = np.argsort(signatures, axis=1)
            for slot in range(k):
                definitions.append({
                    "fold": fold_name, **spec, "category": slot,
                    "discovery_ess": float(ess[slot]),
                    "discovery_mass_share": float(mass[slot] / max(mass.sum(), 1e-12)),
                    "signature": signatures[slot].astype(float).tolist(),
                    "signature_fields": list(preprocessor.fields),
                    "top_positive_features": [str(preprocessor.fields[index]) for index in order[slot, -min(10, len(preprocessor.fields)):][::-1]],
                    "top_negative_features": [str(preprocessor.fields[index]) for index in order[slot, :min(10, len(preprocessor.fields))]],
                })
        except Exception as exc:
            metrics.append({"fold": fold_name, **spec, "status": "failed", "selected_inner": False, "error": repr(exc)})
    metric_frame = pd.DataFrame(metrics)
    qualified = metric_frame.loc[metric_frame["status"].eq("qualified")].copy()
    definitions_frame = pd.DataFrame(definitions)
    # A predeclared GMM exists solely as C4, never as a substitute primary.
    # If the frozen primary itself cannot requalify, halt before probe fitting.
    primary_candidates = qualified.loc[qualified["frozen_role"].ne("control")].copy()
    if primary_candidates.empty:
        raise ArchetypeDiscoveryFailure(fold_name, metric_frame, definitions_frame)
    if frozen_spec:
        expected_primary = primary_candidates.loc[primary_candidates["frozen_role"].eq("primary")]
        if expected_primary.empty:
            raise ArchetypeDiscoveryFailure(fold_name, metric_frame, definitions_frame)
        winner = expected_primary.iloc[0]
    else:
        best_score = float(primary_candidates["structural_selection_score"].max())
    # Prefer the smallest representation on a genuine structural plateau.
        near = primary_candidates.loc[primary_candidates["structural_selection_score"] >= best_score - float(discovery.get("structural_selection_tolerance", 0.05))]
        winner = near.sort_values(
            ["k", "temporal_signature_correlation", "max_category_mass_share", "algorithm"],
            ascending=[True, False, True, True], kind="stable",
        ).iloc[0]
    mask = (
        metric_frame["algorithm"].eq(winner["algorithm"])
        & metric_frame["k"].eq(winner["k"])
        & metric_frame["seed"].eq(winner["seed"])
        & metric_frame["covariance"].fillna("none").eq(str(winner["covariance"] if pd.notna(winner["covariance"]) else "none"))
    )
    metric_frame.loc[mask, "selected_inner"] = True
    return fitted, metric_frame, definitions_frame


ATR_UTILITY_BINS = np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
ATR_UTILITY_CLASS_VALUES = np.asarray([-1.5, -0.5, 0.25, 0.75, 1.5, 2.5], dtype=np.float32)


def _atr_utility(frame: pd.DataFrame) -> np.ndarray:
    """Policy-net utility in decision-time ATR units, clipped for robustness."""
    utility = pd.to_numeric(frame["policy_net_atr"], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(utility).all():
        raise ValueError("ATR-normalised probe target contains non-finite values")
    # The utility tails are real but they should not dominate a shallow
    # reliability probe.  Evaluation remains exact policy-net bps.
    return np.clip(utility, -4.0, 4.0).astype(np.float32)


def _target_values(frame: pd.DataFrame, family: str) -> np.ndarray:
    utility = _atr_utility(frame)
    if family == "atr_utility":
        return utility
    if family == "atr_ordinal":
        # right=True implements the declared intervals exactly:
        # <=-1, (-1,0], (0,.5], (.5,1], (1,2], >2.
        return np.digitize(utility, ATR_UTILITY_BINS, right=True).astype(np.int32)
    if family == "atr_timestamp_rank":
        # Explicit ranking ablation: it is retained for comparison only, not
        # the canonical probe formulation.  The rank is based on the same
        # ATR-normalised policy utility, never on a BPS target.
        ranks = pd.Series(utility, index=frame.index).groupby(frame["__decision_ts__"], sort=False).rank(pct=True, method="average")
        return np.minimum((ranks.fillna(0.0).to_numpy() * 6.0).astype(np.int32), 5)
    raise KeyError(f"unknown ATR-normalised target family {family}")


def _timestamp_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.sort_values("__decision_ts__", kind="stable").copy()
    grouped = frame.groupby("__decision_ts__", sort=False).size()
    choices = grouped.index.to_numpy()
    rng = np.random.default_rng(seed)
    chosen: list[Any] = []
    total = 0
    for stamp in rng.permutation(choices):
        count = int(grouped.loc[stamp])
        if total and total + count > maximum:
            continue
        chosen.append(stamp)
        total += count
        if total >= maximum:
            break
    return frame.loc[frame["__decision_ts__"].isin(chosen)].sort_values("__decision_ts__", kind="stable").copy()


@dataclass
class ProbeModel:
    category: int
    feature_set: str
    target_family: str
    model_kind: str
    fields: tuple[str, ...]
    preprocessor: RobustPreprocessor
    model: Any
    reference: np.ndarray
    train_rows: int
    train_queries: int
    seed: int

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.preprocessor.transform(frame)
        if self.model_kind.endswith("ordinal"):
            probability = self.model.predict_proba(matrix)
            classes = np.asarray(getattr(self.model, "classes_", np.arange(probability.shape[1])), dtype=int)
            raw = probability @ ATR_UTILITY_CLASS_VALUES[classes]
        else:
            raw = self.model.predict(matrix)
        raw = np.asarray(raw, dtype=np.float32)
        rank = np.searchsorted(self.reference, raw, side="right") / float(len(self.reference))
        return raw, np.clip(rank, 0.0, 1.0).astype(np.float32)


def _select_category_probe_fields(
    matrix: np.ndarray, target: np.ndarray, membership_weight: np.ndarray, *,
    feature_set: str, cfg: dict[str, Any], seed: int, maximum_override: int | None = None,
) -> np.ndarray:
    """Small, weighted, train-only nonlinear screen for one specialist.

    The screen never sees inner or held rows.  A weighted binned-MI proxy
    matches the probe's soft-membership training mass with bounded O(rows ×
    fields) compute; it avoids the slow kNN estimator that made broad P1
    screening impractical.  It is an input screen, not a score-time feature
    and not an economic selection on held data.
    """
    maximum = int(
        cfg["feature_selection"][f"max_fields_{feature_set.lower()}"]
        if maximum_override is None
        else maximum_override
    )
    if maximum >= matrix.shape[1]:
        return np.arange(matrix.shape[1], dtype=int)
    sample = _sample_indices(len(matrix), int(cfg["feature_selection"]["max_rows"]), seed + 7919)
    raw = np.asarray(matrix[sample], dtype=np.float32)
    values = np.asarray(target, dtype=float)[sample]
    weights = np.maximum(np.asarray(membership_weight, dtype=float)[sample], 1e-8)
    bins = int(cfg["feature_selection"]["mi_bins"])
    try:
        quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        target_edges = np.quantile(values, quantiles)
        target_bin = np.searchsorted(target_edges, values, side="right")
        feature_edges = np.quantile(raw, quantiles, axis=0)
        score = np.empty(raw.shape[1], dtype=float)
        for feature in range(raw.shape[1]):
            source_bin = np.searchsorted(feature_edges[:, feature], raw[:, feature], side="right")
            joint = np.bincount(source_bin * bins + target_bin, weights=weights, minlength=bins * bins).reshape(bins, bins)
            probability = joint / max(joint.sum(), 1e-12)
            marginal_x = probability.sum(axis=1, keepdims=True)
            marginal_y = probability.sum(axis=0, keepdims=True)
            valid = probability > 0.0
            score[feature] = float(np.sum(probability[valid] * np.log(probability[valid] / (marginal_x @ marginal_y)[valid])))
    except Exception:
        # A zero-information screen should remain safe and deterministic rather
        # than silently substitute a held-period outcome statistic.
        score = np.zeros(matrix.shape[1], dtype=float)
    score = np.nan_to_num(score, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    ordered = np.lexsort((np.arange(len(score)), -score))
    return np.sort(ordered[:maximum])


def _fit_probe(
    frame: pd.DataFrame, fields: Sequence[str], category: int, membership: np.ndarray,
    feature_set: str, target_family: str, model_kind: str, cfg: dict[str, Any], preprocess_cfg: dict[str, Any], seed: int,
    *, selection_stage: bool = False, feature_budget: int | None = None,
    model_overrides: dict[str, Any] | None = None,
) -> ProbeModel:
    maximum = int(cfg.get("selection_max_train_rows", cfg["max_train_rows"])) if selection_stage else int(cfg["max_train_rows"])
    sampled = _timestamp_sample(frame, maximum, seed)
    # Membership is aligned with the unsampled frame.  Candidate identity is
    # deterministic and target-free, so reindex explicitly rather than assume
    # a positional relation after chronological sampling.
    member_map = pd.Series(membership[:, category], index=frame["candidate_id"].to_numpy())
    weights = sampled["candidate_id"].map(member_map).fillna(0.0).to_numpy(dtype=np.float32)
    weights = np.maximum(weights, 1e-5)
    preprocessor = RobustPreprocessor.fit(sampled, fields, preprocess_cfg, seed)
    matrix = preprocessor.transform(sampled)
    target = _target_values(sampled, target_family)
    selected_positions = _select_category_probe_fields(
        matrix, target, weights, feature_set=feature_set, cfg=cfg, seed=seed,
        maximum_override=feature_budget,
    )
    preprocessor = preprocessor.subset(selected_positions)
    matrix = matrix[:, selected_positions]
    if len(np.unique(target)) < 2:
        raise ValueError(f"category {category} {target_family}: degenerate training target")
    params = {**cfg["models"], **(model_overrides or {})}
    n_estimators = int(params.get("selection_n_estimators", params["n_estimators"])) if selection_stage else int(params["n_estimators"])
    common = {
        "n_estimators": n_estimators,
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "colsample_bytree": float(params["feature_fraction"]),
        "subsample": float(params["bagging_fraction"]),
        "reg_lambda": float(params["reg_lambda"]),
        "random_state": seed,
        "n_jobs": int(params.get("n_jobs", 4)),
        "verbosity": -1,
    }
    if model_kind == "lgbm_huber":
        model = LGBMRegressor(objective="huber", alpha=float(params.get("huber_alpha", 0.9)), **common)
        model.fit(matrix, target, sample_weight=weights)
        raw_train = model.predict(matrix)
    elif model_kind == "lgbm_ordinal":
        model = LGBMClassifier(objective="multiclass", num_class=6, class_weight="balanced", **common)
        model.fit(matrix, target.astype(int), sample_weight=weights)
        raw_train = model.predict_proba(matrix) @ ATR_UTILITY_CLASS_VALUES[np.asarray(model.classes_, dtype=int)]
    elif model_kind == "catboost_huber":
        model = CatBoostRegressor(
            loss_function=f"Huber:delta={float(params.get('catboost_huber_delta', 1.0))}",
            iterations=n_estimators, learning_rate=float(params["learning_rate"]), depth=int(params["max_depth"]),
            l2_leaf_reg=float(params["reg_lambda"]), random_seed=seed, random_strength=0.5,
            rsm=float(params["feature_fraction"]), bootstrap_type="Bernoulli", subsample=float(params["bagging_fraction"]),
            thread_count=int(params.get("n_jobs", 4)), verbose=False, allow_writing_files=False,
        )
        model.fit(matrix, target, sample_weight=weights, verbose=False)
        raw_train = model.predict(matrix)
    elif model_kind == "catboost_ordinal":
        model = CatBoostClassifier(
            loss_function="MultiClass", iterations=n_estimators, learning_rate=float(params["learning_rate"]),
            depth=int(params["max_depth"]), l2_leaf_reg=float(params["reg_lambda"]), random_seed=seed,
            random_strength=0.5, rsm=float(params["feature_fraction"]), bootstrap_type="Bernoulli",
            subsample=float(params["bagging_fraction"]), thread_count=int(params.get("n_jobs", 4)),
            verbose=False, allow_writing_files=False,
        )
        model.fit(matrix, target.astype(int), sample_weight=weights, verbose=False)
        raw_train = model.predict_proba(matrix) @ ATR_UTILITY_CLASS_VALUES[np.asarray(model.classes_, dtype=int)]
    elif model_kind == "lgbm_ranker":
        if target_family != "atr_timestamp_rank":
            raise ValueError("LambdaRank is an explicit timestamp-rank ablation only")
        group = sampled.groupby("__decision_ts__", sort=False).size().to_numpy(dtype=np.int32)
        model = LGBMRanker(
            objective="lambdarank", metric="ndcg", **common,
        )
        model.fit(matrix, target, group=group, sample_weight=weights)
        raw_train = model.predict(matrix)
    else:
        raise KeyError(f"unsupported probe model kind {model_kind!r}")
    reference = np.sort(np.asarray(raw_train, dtype=np.float32))
    return ProbeModel(category, feature_set, target_family, model_kind, tuple(preprocessor.fields), preprocessor, model, reference, len(sampled), len(grouped := sampled.groupby("__decision_ts__")), seed)


def _fit_discovery_candidates(
    train_target_free: pd.DataFrame,
    inner_outcomes: pd.DataFrame,
    fields: Sequence[str],
    config: dict[str, Any],
    *, seed: int, fold_name: str,
) -> tuple[list[tuple[dict[str, Any], CategoryModel]], pd.DataFrame, pd.DataFrame]:
    """Fit unsupervised candidates and select only from inner-fold diagnostics."""
    discovery = config["discovery"]
    if bool(discovery.get("strict_structural_qualification", False)):
        # ``inner_outcomes`` is deliberately ignored.  This branch selects a
        # structural representation from earlier target-free rows only, then
        # fails before any economic probe is fit when the layer is invalid.
        return _strict_structural_discovery_candidates(
            train_target_free, fields, config, seed=seed, fold_name=fold_name,
        )
    preprocess = config["preprocessing"]
    candidates: list[tuple[str, str | None, int]] = []
    for covariance in discovery["gmm_covariances"]:
        for k in discovery["k_values"]:
            if covariance == "full" and int(k) > int(discovery["gmm_full_max_k"]):
                continue
            candidates.append(("gmm", str(covariance), int(k)))
    candidates.extend(("nmf", None, int(k)) for k in discovery["nmf_k_values"])
    metrics: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], CategoryModel]] = []
    definitions: list[dict[str, Any]] = []
    # Preprocessing is target-free and identical across all discovery
    # candidates in this fold.  Fit/transform it once rather than repeating a
    # costly robust correlation-collapse pass for every K/covariance model.
    shared_preprocessor = RobustPreprocessor.fit(train_target_free, fields, preprocess, seed)
    shared_matrix = shared_preprocessor.transform(train_target_free)
    inner_signature_matrix = shared_preprocessor.transform(inner_outcomes)
    for number, (algorithm, covariance, k) in enumerate(candidates):
        candidate_seed = seed + number * 101
        try:
            model = _fit_category_model(
                train_target_free, fields, preprocess, algorithm=algorithm,
                covariance=covariance, k=k, seed=candidate_seed,
                preprocessor=shared_preprocessor, matrix_all=shared_matrix,
            )
            membership = model.membership(inner_outcomes)
            ess = (membership.sum(axis=0) ** 2) / np.maximum((membership ** 2).sum(axis=0), 1e-8)
            net = pd.to_numeric(inner_outcomes["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
            router = pd.to_numeric(inner_outcomes["router_rank"], errors="coerce").to_numpy(dtype=float)
            means = np.asarray([
                np.average(net, weights=np.maximum(membership[:, slot], 1e-8))
                for slot in range(k)
            ])
            router_means = np.asarray([
                np.average(router, weights=np.maximum(membership[:, slot], 1e-8))
                for slot in range(k)
            ])
            # A structural signature is based only on the causal discovery
            # features, while economic/Router separation is inner-evaluation
            # evidence for selecting a representation.
            signature = np.asarray([
                np.average(inner_signature_matrix, axis=0, weights=np.maximum(membership[:, slot], 1e-8))
                for slot in range(k)
            ])
            if k > 1:
                feature_distinct = float(np.nanmean([
                    np.linalg.norm(signature[a] - signature[b]) / math.sqrt(signature.shape[1])
                    for a in range(k) for b in range(a + 1, k)
                ]))
            else:
                feature_distinct = 0.0
            stability = _membership_stability(model, train_target_free, seed=candidate_seed + 7, maximum=min(20000, len(train_target_free)))
            econ_spread = float(np.nanstd(means))
            router_spread = float(np.nanstd(router_means))
            min_ess = float(np.nanmin(ess))
            # This utility only selects the representation in the development
            # inner fold.  It never becomes a model input or score-time field.
            utility = (
                0.40 * (0.0 if not np.isfinite(stability) else max(0.0, stability))
                + 0.25 * min(1.0, min_ess / max(float(discovery["minimum_category_ess"]), 1.0))
                + 0.20 * min(1.0, feature_distinct / 2.0)
                + 0.10 * min(1.0, econ_spread / 100.0)
                + 0.05 * min(1.0, router_spread / 0.10)
            )
            spec = {"algorithm": algorithm, "covariance": covariance, "k": k, "seed": candidate_seed}
            metrics.append({
                "fold": fold_name, **spec, "status": "ok", "stability": stability,
                "min_ess": min_ess, "mean_ess": float(np.nanmean(ess)),
                "structural_distinctness": feature_distinct, "economic_mean_spread_bps": econ_spread,
                "router_mean_spread": router_spread, "inner_selection_utility": utility,
                "preprocessor_fields": list(model.preprocessor.fields),
            })
            fitted.append((spec, model))
            for slot in range(k):
                definitions.append({
                    "fold": fold_name, **spec, "category": slot, "inner_ess": float(ess[slot]),
                    "inner_weighted_net_bps": float(means[slot]), "inner_weighted_router_rank": float(router_means[slot]),
                    "signature": signature[slot].astype(float).tolist(), "signature_fields": list(model.preprocessor.fields),
                })
        except Exception as exc:  # retain failures as an auditable discovery result.
            metrics.append({"fold": fold_name, "algorithm": algorithm, "covariance": covariance, "k": k, "seed": candidate_seed, "status": "failed", "error": repr(exc)})
    # Density clustering is a deliberately non-canonical diagnostic.  It
    # answers whether the soft GMM/NMF decomposition is forcing a large noise
    # population into arbitrary categories; it never produces score-time
    # memberships or participates in specialist selection.
    if bool(discovery.get("hdbscan_diagnostic", False)):
        if hdbscan is None:
            metrics.append({"fold": fold_name, "algorithm": "hdbscan_diagnostic", "covariance": None, "k": 0, "seed": seed, "status": "skipped", "error": "hdbscan_unavailable"})
        else:
            try:
                diagnostic_sample = shared_matrix[_sample_indices(len(shared_matrix), min(8_000, len(shared_matrix)), seed + 8803)]
                density = hdbscan.HDBSCAN(min_cluster_size=max(100, int(len(diagnostic_sample) * .01)), min_samples=25, prediction_data=False).fit(diagnostic_sample)
                labels = np.asarray(density.labels_, dtype=int)
                clusters = labels[labels >= 0]
                metrics.append({
                    "fold": fold_name, "algorithm": "hdbscan_diagnostic", "covariance": None,
                    "k": int(len(np.unique(clusters))), "seed": seed, "status": "diagnostic",
                    "noise_share": float(np.mean(labels < 0)), "sample_rows": int(len(labels)),
                    "preprocessor_fields": list(shared_preprocessor.fields),
                })
            except Exception as exc:
                metrics.append({"fold": fold_name, "algorithm": "hdbscan_diagnostic", "covariance": None, "k": 0, "seed": seed, "status": "failed", "error": repr(exc)})
    metric_frame = pd.DataFrame(metrics)
    valid = metric_frame.loc[
        (metric_frame["status"] == "ok")
        & (metric_frame["min_ess"] >= float(discovery["minimum_category_ess"]))
        & (metric_frame["stability"].fillna(0.0) >= float(discovery["minimum_stability"]))
    ].copy()
    if valid.empty:
        valid = metric_frame.loc[metric_frame["status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError(f"{fold_name}: every category discovery candidate failed")
    # Smallest Pareto-ish representation within 95% of the best admissible
    # inner utility.  This makes K a stability/complexity decision, not an
    # outcome chase across the later outer holdout.
    best = float(valid["inner_selection_utility"].max())
    near = valid.loc[valid["inner_selection_utility"] >= best * 0.95]
    winner_row = near.sort_values(["k", "inner_selection_utility", "algorithm"], ascending=[True, False, True], kind="stable").iloc[0]
    metric_frame["selected_inner"] = False
    selected_mask = (
        (metric_frame["algorithm"] == winner_row["algorithm"])
        & (metric_frame["covariance"].fillna("none") == str(winner_row["covariance"] if pd.notna(winner_row["covariance"]) else "none"))
        & (metric_frame["k"] == winner_row["k"])
    )
    metric_frame.loc[selected_mask, "selected_inner"] = True
    return fitted, metric_frame, pd.DataFrame(definitions)


def _selected_category_spec(metrics: pd.DataFrame) -> dict[str, Any]:
    selected = metrics.loc[metrics["selected_inner"].fillna(False)]
    if len(selected) != 1:
        raise AssertionError("expected exactly one selected category candidate")
    row = selected.iloc[0]
    return {
        "algorithm": str(row["algorithm"]), "covariance": None if pd.isna(row["covariance"]) else str(row["covariance"]),
        "k": int(row["k"]), "seed": int(row["seed"]),
        "archetype_temperature_multiplier": float(row.get("archetype_temperature_multiplier", 1.0)),
    }


def _category_count_inner_ablation(
    train: pd.DataFrame, inner: pd.DataFrame, fitted_discovery: Sequence[tuple[dict[str, Any], CategoryModel]],
    discovery_metrics: pd.DataFrame, discovery_fields: Sequence[str], config: dict[str, Any], *, fold: str, seed: int,
) -> pd.DataFrame:
    """Report viable K performance using the same bounded probe family.

    This is deliberately an inner-only diagnostic.  It cannot alter a held
    score, and uses one compact P0 Huber probe per category rather than a full
    head/model grid so K=2..12 remains computationally feasible.
    """
    def category_key(algorithm: object, covariance: object, k: object) -> tuple[str, str, int]:
        return str(algorithm), "none" if covariance is None or pd.isna(covariance) else str(covariance), int(k)

    columns = [
        "fold", "k", "algorithm", "covariance", "status", "stability", "min_ess", "error",
        "probe_recall_gt100", "combined_recall_gt100", "router_recall_gt100",
        "probe_positive_economic_mass_recall", "combined_positive_economic_mass_recall", "router_positive_economic_mass_recall",
    ]
    spec_to_model = {
        category_key(spec["algorithm"], spec["covariance"], spec["k"]): model
        for spec, model in fitted_discovery
    }
    viable = discovery_metrics.loc[
        (discovery_metrics["status"].isin(["ok", "qualified"]))
        & (discovery_metrics["min_ess"] >= float(config["discovery"]["minimum_category_ess"]))
        & (discovery_metrics["stability"].fillna(0.0) >= float(config["discovery"]["minimum_stability"]))
    ].copy()
    if viable.empty:
        return pd.DataFrame(columns=columns)
    # One highest-quality structural representation for each K.  The model
    # evaluates its own inner memberships, not the already-selected K.
    selection_column = (
        "structural_selection_score"
        if "structural_selection_score" in viable.columns
        else "inner_selection_utility"
    )
    viable = viable.sort_values(["k", selection_column, "algorithm"], ascending=[True, False, True], kind="stable")
    viable = viable.groupby("k", as_index=False, sort=True).head(1)
    rows: list[dict[str, Any]] = []
    budget = float(config["evaluation"]["selection_primary_budget"])
    rescue_share = .5
    # K diagnostics are intentionally broad.  Use a separate small, fixed
    # screening budget so they characterize the plateau without multiplying
    # the final specialist HPO cost by every K.
    ablation_probe = copy.deepcopy(config["probe"])
    ablation_probe["selection_max_train_rows"] = min(8_000, int(ablation_probe["selection_max_train_rows"]))
    ablation_probe["models"]["selection_n_estimators"] = min(60, int(ablation_probe["models"]["selection_n_estimators"]))
    ablation_probe["feature_selection"]["max_fields_p0"] = min(32, int(ablation_probe["feature_selection"]["max_fields_p0"]))
    ablation_probe["feature_selection"]["max_rows"] = min(6_000, int(ablation_probe["feature_selection"]["max_rows"]))
    for number, candidate in enumerate(viable.itertuples(index=False)):
        key = category_key(candidate.algorithm, candidate.covariance, candidate.k)
        model = spec_to_model.get(key)
        if model is None:
            rows.append({"fold": fold, "k": int(candidate.k), "algorithm": candidate.algorithm, "covariance": candidate.covariance, "status": "missing_model"})
            continue
        try:
            membership_train = model.membership(train)
            membership_inner = model.membership(inner)
            heads: list[ProbeModel] = []
            for category in range(int(candidate.k)):
                heads.append(_fit_probe(
                    train, discovery_fields, category, membership_train, "P0", "atr_utility", "lgbm_huber",
                    ablation_probe, config["preprocessing"], seed + number * 10_000 + category * 37, selection_stage=True,
                ))
            rank = np.column_stack([head.score(inner)[1] for head in heads])
            score = _aggregate_heads(
                np.column_stack([_combine_head_score(rank[:, category], membership_inner[:, category], gamma=1.0, method="rank_times_membership") for category in range(rank.shape[1])]),
                "max",
            )
            probe_mask = _selection_mask(inner, score, None, budget, 0.0)
            combined_mask = _selection_mask(inner, inner["router_rank"].to_numpy(dtype=float), score, budget, rescue_share)
            probe_metrics = _metric_row(inner, probe_mask, fold=fold, split="inner", strategy="probe_only", budget=budget, router_share=0.0, score_name="category_count_probe")
            combined_metrics = _metric_row(inner, combined_mask, fold=fold, split="inner", strategy="router_plus_probe_rescue", budget=budget, router_share=rescue_share, score_name="category_count_probe")
            router_metrics = _metric_row(inner, _selection_mask(inner, inner["router_rank"].to_numpy(dtype=float), None, budget, 1.0), fold=fold, split="inner", strategy="router_only", budget=budget, router_share=1.0, score_name="router_rank")
            rows.append({
                "fold": fold, "k": int(candidate.k), "algorithm": candidate.algorithm, "covariance": candidate.covariance,
                "status": "ok", "stability": float(candidate.stability), "min_ess": float(candidate.min_ess),
                "probe_recall_gt100": probe_metrics["recall_gt_100"],
                "combined_recall_gt100": combined_metrics["recall_gt_100"],
                "router_recall_gt100": router_metrics["recall_gt_100"],
                "probe_positive_economic_mass_recall": probe_metrics["positive_economic_mass_recall"],
                "combined_positive_economic_mass_recall": combined_metrics["positive_economic_mass_recall"],
                "router_positive_economic_mass_recall": router_metrics["positive_economic_mass_recall"],
            })
        except Exception as exc:
            rows.append({"fold": fold, "k": int(candidate.k), "algorithm": candidate.algorithm, "covariance": candidate.covariance, "status": "failed", "error": repr(exc)})
    return pd.DataFrame(rows, columns=columns)


def _activated_membership(membership: np.ndarray, floor: float) -> np.ndarray:
    """Apply a head-local, inner-selected soft-membership activation floor.

    The threshold never changes the structural representation or the training
    population.  It only prevents a specialist from contributing where its own
    soft membership is too weak, then rescales the surviving range to preserve
    the original [0, 1] score contract.  This is intentionally distinct from a
    hard category assignment.
    """
    floor = float(floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"membership activation floor must be in [0, 1): {floor}")
    member = np.clip(np.asarray(membership, dtype=np.float32), 0.0, 1.0)
    if floor <= 0.0:
        return member
    return np.clip((member - floor) / (1.0 - floor), 0.0, 1.0).astype(np.float32)


def _combine_head_score(
    rank: np.ndarray, membership: np.ndarray, *, gamma: float, method: str,
    activation_floor: float = 0.0,
) -> np.ndarray:
    member = _activated_membership(membership, activation_floor)
    rank = np.clip(np.asarray(rank, dtype=np.float32), 0.0, 1.0)
    if method == "rank_times_membership":
        return rank * np.power(member, gamma)
    if method == "minimum":
        return np.minimum(rank, np.power(member, gamma))
    if method == "geometric_mean":
        return np.sqrt(np.maximum(rank * np.power(member, gamma), 0.0))
    raise KeyError(method)


def _aggregate_heads(head_values: np.ndarray, method: str) -> np.ndarray:
    if method == "max":
        return np.max(head_values, axis=1)
    if method == "top2_mean":
        if head_values.shape[1] == 1:
            return head_values[:, 0]
        top = np.partition(head_values, -2, axis=1)[:, -2:]
        return top.mean(axis=1)
    if method == "logsumexp":
        maximum = np.max(head_values, axis=1, keepdims=True)
        return (maximum[:, 0] + np.log(np.exp(head_values - maximum).mean(axis=1))).astype(np.float32)
    raise KeyError(method)


def _take_top_indices(group: pd.DataFrame, score: np.ndarray, count: int, excluded: set[Any] | None = None) -> list[Any]:
    if count <= 0:
        return []
    work = pd.DataFrame({"candidate_id": group["candidate_id"].to_numpy(), "score": score})
    if excluded:
        work = work.loc[~work["candidate_id"].isin(excluded)]
    return work.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(count)["candidate_id"].tolist()


def _selection_mask(frame: pd.DataFrame, router_score: np.ndarray, probe_score: np.ndarray | None, budget: float, router_share: float) -> np.ndarray:
    """Exact timestamp-local total budget; probe is a true rescue outside Router."""
    frame = frame.reset_index(drop=True)
    router = np.asarray(router_score, dtype=float)
    probe = None if probe_score is None else np.asarray(probe_score, dtype=float)
    if len(router) != len(frame) or (probe is not None and len(probe) != len(frame)):
        raise ValueError("score arrays must align one-to-one with frame rows")
    # The former implementation sorted a small DataFrame inside a Python loop
    # for every timestamp, budget, rescue allocation and control.  The three
    # stable global sorts below are mathematically identical: score descending,
    # candidate ID ascending on ties, with a timestamp-local count of
    # ceil(n*budget).  This is a compute-only improvement and retains exact
    # timestamp-local candidate capacity.
    work = pd.DataFrame({
        "__decision_ts__": frame["__decision_ts__"].to_numpy(),
        "candidate_id": frame["candidate_id"].to_numpy(),
        "router_score": router,
    }, index=frame.index)
    group_size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(dtype=int)
    total_count = np.maximum(1, np.ceil(group_size.astype(float) * float(budget)).astype(int))
    if probe is None:
        ordered = work.sort_values(["__decision_ts__", "router_score", "candidate_id"], ascending=[True, False, True], kind="stable")
        rank = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy(dtype=int)
        chosen = rank < total_count[ordered.index.to_numpy(dtype=int)]
        result = np.zeros(len(frame), dtype=bool)
        result[ordered.index.to_numpy(dtype=int)] = chosen
        return result

    router_count = np.minimum(total_count, np.ceil(total_count.astype(float) * float(router_share)).astype(int))
    ordered_router = work.sort_values(["__decision_ts__", "router_score", "candidate_id"], ascending=[True, False, True], kind="stable")
    router_rank = ordered_router.groupby("__decision_ts__", sort=False).cumcount().to_numpy(dtype=int)
    router_selected = router_rank < router_count[ordered_router.index.to_numpy(dtype=int)]
    selected = np.zeros(len(frame), dtype=bool)
    selected[ordered_router.index.to_numpy(dtype=int)] = router_selected

    remaining = total_count - router_count
    if np.any(remaining > 0):
        probe_work = work.loc[~selected].copy()
        probe_work["probe_score"] = probe[probe_work.index.to_numpy(dtype=int)]
        ordered_probe = probe_work.sort_values(["__decision_ts__", "probe_score", "candidate_id"], ascending=[True, False, True], kind="stable")
        probe_rank = ordered_probe.groupby("__decision_ts__", sort=False).cumcount().to_numpy(dtype=int)
        probe_selected = probe_rank < remaining[ordered_probe.index.to_numpy(dtype=int)]
        selected[ordered_probe.index.to_numpy(dtype=int)] = probe_selected
    return selected


def _metric_row(
    frame: pd.DataFrame, selected: np.ndarray, *, fold: str, split: str, strategy: str,
    budget: float, router_share: float, score_name: str,
) -> dict[str, Any]:
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    chosen = selected & valid & np.isfinite(net)
    selected_net = net[chosen]
    row: dict[str, Any] = {
        "fold": fold, "split": split, "strategy": strategy, "score_name": score_name,
        "budget_fraction": float(budget), "router_share": float(router_share),
        "candidate_rows": int(len(frame)), "valid_label_rows": int(valid.sum()),
        "selected_rows_all": int(selected.sum()), "selected_rows_valid": int(chosen.sum()),
        "selected_mean_net_bps": float(np.nanmean(selected_net)) if len(selected_net) else float("nan"),
        "selected_median_net_bps": float(np.nanmedian(selected_net)) if len(selected_net) else float("nan"),
        "selected_p10_net_bps": float(np.nanquantile(selected_net, .10)) if len(selected_net) else float("nan"),
        "selected_cvar10_net_bps": float(np.nanmean(np.sort(selected_net)[:max(1, int(math.ceil(.10 * len(selected_net))))])) if len(selected_net) else float("nan"),
        "selected_positive_mass_bps": float(np.nansum(np.maximum(selected_net, 0.0))),
        "all_positive_mass_bps": float(np.nansum(np.maximum(net[valid], 0.0))),
    }
    row["positive_economic_mass_recall"] = row["selected_positive_mass_bps"] / max(row["all_positive_mass_bps"], 1e-8)
    for threshold in (0.0, 50.0, 100.0, 200.0):
        denominator = valid & (net > threshold)
        row[f"recall_gt_{int(threshold)}"] = float((chosen & (net > threshold)).sum() / max(1, denominator.sum()))
        row[f"selected_hit_gt_{int(threshold)}"] = float((chosen & (net > threshold)).sum() / max(1, chosen.sum()))
    # Candidate-budget recall against realised best candidates at each decision
    # timestamp.  This is explicitly an evaluation calculation, never a model
    # feature or label used at score time.
    for fraction in (.20, .10, .05, .02, .01):
        oracle_ids: set[Any] = set()
        for _, group in frame.loc[valid].groupby("__decision_ts__", sort=False):
            n = max(1, int(math.ceil(len(group) * fraction)))
            oracle_ids.update(group.sort_values(["policy_net_bps", "candidate_id"], ascending=[False, True], kind="stable").head(n)["candidate_id"])
        oracle_mask = frame["candidate_id"].isin(oracle_ids).to_numpy()
        row[f"within_ts_top_{fraction:g}_recall"] = float((chosen & oracle_mask).sum() / max(1, (valid & oracle_mask).sum()))
    return row


def _evaluate_strategies(
    frame: pd.DataFrame, probe_score: np.ndarray, *, fold: str, split: str, cfg: dict[str, Any], score_name: str,
) -> tuple[pd.DataFrame, dict[tuple[float, float], np.ndarray]]:
    result: list[dict[str, Any]] = []
    masks: dict[tuple[float, float], np.ndarray] = {}
    router = frame["router_rank"].to_numpy(dtype=float)
    for budget in cfg["evaluation"]["budget_fractions"]:
        baseline = _selection_mask(frame, router, None, float(budget), 1.0)
        result.append(_metric_row(frame, baseline, fold=fold, split=split, strategy="router_only", budget=float(budget), router_share=1.0, score_name="router_rank"))
        for share in cfg["evaluation"]["rescue_router_shares"]:
            mask = _selection_mask(frame, router, probe_score, float(budget), float(share))
            masks[(float(budget), float(share))] = mask
            result.append(_metric_row(frame, mask, fold=fold, split=split, strategy="router_plus_probe_rescue", budget=float(budget), router_share=float(share), score_name=score_name))
        probe_only = _selection_mask(frame, probe_score, None, float(budget), 0.0)
        result.append(_metric_row(frame, probe_only, fold=fold, split=split, strategy="probe_only", budget=float(budget), router_share=0.0, score_name=score_name))
    return pd.DataFrame(result), masks


def _primary_selection_utility(metrics: pd.DataFrame, cfg: dict[str, Any], router_share: float | None = None) -> float:
    budget = float(cfg["evaluation"]["selection_primary_budget"])
    rows = metrics.loc[
        (metrics["strategy"] == "router_plus_probe_rescue")
        & np.isclose(metrics["budget_fraction"], budget)
    ]
    if router_share is not None:
        rows = rows.loc[np.isclose(rows["router_share"], router_share)]
    if rows.empty:
        return float("-inf")
    # Recall of >50 opportunity is primary, with selected EV/CVaR only a
    # secondary tie breaker.  Units are deliberately bounded.
    return float(np.nanmean(rows["recall_gt_50"]) + .001 * np.nanmean(rows["selected_mean_net_bps"]) + .00025 * np.nanmean(rows["selected_cvar10_net_bps"]))


def _head_selection_utility(frame: pd.DataFrame, score: np.ndarray) -> float:
    """Fast inner-only head screen; full oracle metrics are emitted post-selection."""
    selected = _selection_mask(frame, score, None, .10, 1.0)
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    chosen = selected & valid & np.isfinite(net)
    values = net[chosen]
    recall = float((chosen & (net > 50.0)).sum() / max(1, (valid & (net > 50.0)).sum()))
    mean = float(np.nanmean(values)) if len(values) else -500.0
    cvar = float(np.nanmean(np.sort(values)[:max(1, int(math.ceil(.10 * len(values))))])) if len(values) else -500.0
    return recall + .001 * mean + .00025 * cvar


def _fast_rescue_utility(frame: pd.DataFrame, probe_score: np.ndarray, *, budget: float, router_share: float) -> float:
    """Selection-stage equivalent of the primary objective without report-only oracle loops."""
    mask = _selection_mask(frame, frame["router_rank"].to_numpy(dtype=float), probe_score, budget, router_share)
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    chosen = mask & valid & np.isfinite(net)
    recall = float((chosen & (net > 50.0)).sum() / max(1, (valid & (net > 50.0)).sum()))
    values = net[chosen]
    mean = float(np.nanmean(values)) if len(values) else -500.0
    cvar = float(np.nanmean(np.sort(values)[:max(1, int(math.ceil(.10 * len(values))))])) if len(values) else -500.0
    return recall + .001 * mean + .00025 * cvar


def _month_range(frame: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    return frame.loc[(frame["__decision_ts__"] >= _utc(start)) & (frame["__decision_ts__"] < _utc(end))].copy()


def _head_feature_budgets(cfg: dict[str, Any], feature_set: str, available: int) -> list[int]:
    selection = cfg["feature_selection"]
    default = int(selection[f"max_fields_{feature_set.lower()}"])
    declared = selection.get("per_head_budgets", {}).get(feature_set, [default])
    values = sorted({max(1, min(int(value), int(available))) for value in declared})
    return values or [min(default, int(available))]


def _head_hpo_trials(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a compact, predeclared sequential HPO bank.

    Feature/model form selection happens first.  Only the best few inner forms
    for each head enter this bank, avoiding a factorial sweep over categories,
    feature sets and tree geometry.
    """
    raw = cfg.get("per_head_hpo", {})
    trials = raw.get("trials", [{"id": "baseline", "overrides": {}}])
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for number, item in enumerate(trials):
        identifier = str(item.get("id", f"trial_{number:02d}"))
        if identifier in identifiers:
            raise ValueError(f"duplicate per-head HPO id: {identifier}")
        overrides = dict(item.get("overrides", {}))
        unsupported = set(overrides).difference(cfg["models"])
        if unsupported:
            raise ValueError(f"unsupported per-head HPO override(s): {sorted(unsupported)}")
        identifiers.add(identifier)
        result.append({"id": identifier, "overrides": overrides})
    return result


def _head_activation_floors(cfg: dict[str, Any]) -> list[float]:
    values = sorted({float(value) for value in cfg.get("per_head_activation_floors", [0.0])})
    if not values or values[0] < 0.0 or values[-1] >= 1.0:
        raise ValueError("per-head activation floors must be non-empty values in [0, 1)")
    return values


def _build_head_candidates(
    train: pd.DataFrame, inner: pd.DataFrame, membership_train: np.ndarray, membership_inner: np.ndarray,
    discovery_fields: Sequence[str], predictive_fields: Sequence[str], config: dict[str, Any], *, fold: str, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[ProbeModel]]:
    """Select each specialist by a strict inner-only sequential funnel.

    Stage one chooses the target, model family and category-local MI feature
    budget.  Stage two HPOs only the predeclared number of best forms, then
    chooses a membership activation floor for that head.  The generic C0 is
    deliberately not part of either selection stage and is trained only after
    the specialist contract is fixed.
    """
    selections: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    fitted: list[ProbeModel] = []
    hpo_trials = _head_hpo_trials(config["probe"])
    activation_floors = _head_activation_floors(config["probe"])
    hpo_top_forms = max(1, int(config["probe"].get("per_head_hpo", {}).get("top_feature_model_forms", 1)))
    for category in range(membership_train.shape[1]):
        screen: list[tuple[float, dict[str, Any]]] = []
        local: list[tuple[float, dict[str, Any], ProbeModel]] = []
        sequence = 0
        for feature_set, fields in (("P0", discovery_fields), ("P1", predictive_fields)):
            for model_spec in config["probe"]["model_specs"]:
                target_family, kind = str(model_spec["target_family"]), str(model_spec["model_kind"])
                for feature_budget in _head_feature_budgets(config["probe"], feature_set, len(fields)):
                    trial_seed = seed + category * 10000 + sequence * 31
                    sequence += 1
                    try:
                        model = _fit_probe(
                            train, fields, category, membership_train, feature_set,
                            target_family, kind, config["probe"], config["preprocessing"], trial_seed,
                            selection_stage=True, feature_budget=feature_budget,
                        )
                        _, rank = model.score(inner)
                        score = _combine_head_score(rank, membership_inner[:, category], gamma=1.0, method="rank_times_membership")
                        utility = _head_selection_utility(inner, score)
                        record = {
                            "fold": fold, "category": category, "feature_set": feature_set,
                            "target_family": target_family, "model_kind": kind,
                            "canonical_probe_formulation": bool(model_spec.get("canonical", False)),
                            "seed": trial_seed, "status": "ok", "selection_stage": "feature_model_screen",
                            "feature_budget": int(feature_budget), "model_hpo_id": "screen_baseline",
                            "model_hpo_overrides": {}, "membership_activation_floor": 0.0,
                            "inner_head_utility": utility, "train_rows": model.train_rows,
                            "train_queries": model.train_queries, "fields": list(model.fields),
                        }
                        trials.append(record)
                        # Timestamp LambdaRank remains a diagnostic ablation.
                        if bool(model_spec.get("canonical", False)):
                            screen.append((utility, record))
                    except Exception as exc:
                        trials.append({
                            "fold": fold, "category": category, "feature_set": feature_set,
                            "target_family": target_family, "model_kind": kind,
                            "canonical_probe_formulation": bool(model_spec.get("canonical", False)),
                            "seed": trial_seed, "status": "failed", "selection_stage": "feature_model_screen",
                            "feature_budget": int(feature_budget), "error": repr(exc),
                        })
        if not screen:
            raise RuntimeError(f"{fold}: no probe model fit for category {category}")
        # Keep only the most promising feature/model forms for model HPO.  The
        # sorting order is part of the frozen, reproducible funnel.
        screen.sort(key=lambda item: (-item[0], item[1]["feature_set"], item[1]["target_family"], item[1]["model_kind"], item[1]["feature_budget"]))
        for _, form in screen[:hpo_top_forms]:
            fields = discovery_fields if form["feature_set"] == "P0" else predictive_fields
            for hpo in hpo_trials:
                try:
                    model = _fit_probe(
                        train, fields, category, membership_train, str(form["feature_set"]),
                        str(form["target_family"]), str(form["model_kind"]), config["probe"], config["preprocessing"],
                        int(form["seed"]), selection_stage=True, feature_budget=int(form["feature_budget"]),
                        model_overrides=dict(hpo["overrides"]),
                    )
                    _, rank = model.score(inner)
                    for activation_floor in activation_floors:
                        score = _combine_head_score(
                            rank, membership_inner[:, category], gamma=1.0, method="rank_times_membership",
                            activation_floor=float(activation_floor),
                        )
                        utility = _head_selection_utility(inner, score)
                        record = {
                            **{key: value for key, value in form.items() if key not in {"inner_head_utility", "fields", "train_rows", "train_queries", "selection_stage", "model_hpo_id", "model_hpo_overrides", "membership_activation_floor"}},
                            "status": "ok", "selection_stage": "hpo_and_activation",
                            "model_hpo_id": str(hpo["id"]), "model_hpo_overrides": dict(hpo["overrides"]),
                            "membership_activation_floor": float(activation_floor),
                            "inner_head_utility": utility, "train_rows": model.train_rows,
                            "train_queries": model.train_queries, "fields": list(model.fields),
                        }
                        trials.append(record)
                        local.append((utility, record, model))
                except Exception as exc:
                    trials.append({
                        **form, "status": "failed", "selection_stage": "hpo_and_activation",
                        "model_hpo_id": str(hpo["id"]), "model_hpo_overrides": dict(hpo["overrides"]),
                        "error": repr(exc),
                    })
        if not local:
            raise RuntimeError(f"{fold}: no HPO probe model fit for category {category}")
        # Stable lexical ties make the selected head reproducible.
        local.sort(key=lambda item: (-item[0], item[1]["feature_set"], item[1]["target_family"], item[1]["model_kind"]))
        best_utility, best_record, best_model = local[0]
        best_record = dict(best_record)
        best_record["selected_inner"] = True
        selections.append(best_record)
        fitted.append(best_model)
        for record in trials:
            if record.get("category") == category and record.get("status") == "ok":
                record["selected_inner"] = bool(
                    record["feature_set"] == best_record["feature_set"]
                    and record["target_family"] == best_record["target_family"]
                    and record["model_kind"] == best_record["model_kind"]
                    and record["seed"] == best_record["seed"]
                    and record.get("feature_budget") == best_record.get("feature_budget")
                    and record.get("model_hpo_id") == best_record.get("model_hpo_id")
                    and np.isclose(
                        float(record.get("membership_activation_floor", 0.0)),
                        float(best_record.get("membership_activation_floor", 0.0)),
                    )
                )
    return selections, trials, fitted


def _refit_selected_heads(
    train: pd.DataFrame, membership: np.ndarray, head_specs: Sequence[dict[str, Any]], discovery_fields: Sequence[str],
    predictive_fields: Sequence[str], config: dict[str, Any], *, seed: int,
) -> list[ProbeModel]:
    models: list[ProbeModel] = []
    for spec in head_specs:
        fields = discovery_fields if spec["feature_set"] == "P0" else predictive_fields
        models.append(_fit_probe(
            train, fields, int(spec["category"]), membership, str(spec["feature_set"]),
            str(spec["target_family"]), str(spec["model_kind"]), config["probe"], config["preprocessing"],
            seed + int(spec["category"]) * 1009,
            feature_budget=int(spec.get("feature_budget", len(fields))),
            model_overrides=dict(spec.get("model_hpo_overrides", {})),
        ))
    return models


def _choose_combination(
    inner: pd.DataFrame, head_models: Sequence[ProbeModel], head_specs: Sequence[dict[str, Any]], membership: np.ndarray,
    config: dict[str, Any], *, fold: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ranks = np.column_stack([model.score(inner)[1] for model in head_models]).astype(np.float32)
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    for gamma in config["probe"]["membership_gammas"]:
        for combination in config["probe"]["combination_methods"]:
            heads = np.column_stack([
                _combine_head_score(
                    ranks[:, slot], membership[:, slot], gamma=float(gamma), method=str(combination),
                    activation_floor=float(head_specs[slot].get("membership_activation_floor", 0.0)),
                )
                for slot in range(ranks.shape[1])
            ])
            for aggregation in config["probe"]["aggregation_methods"]:
                score = _aggregate_heads(heads, str(aggregation))
                for router_share in config["evaluation"]["rescue_router_shares"]:
                    utility = _fast_rescue_utility(
                        inner, score, budget=float(config["evaluation"]["selection_primary_budget"]), router_share=float(router_share),
                    )
                    spec = {
                        "membership_gamma": float(gamma), "combination_method": str(combination),
                        "aggregation_method": str(aggregation), "router_share": float(router_share),
                    }
                    trials.append({"fold": fold, **spec, "inner_combination_utility": utility})
                    current = (utility, spec)
                    if best is None or current[0] > best[0] or (np.isclose(current[0], best[0]) and str(current[1]) < str(best[1])):
                        best = current
    if best is None:
        raise AssertionError("no combination trial")
    trial_frame = pd.DataFrame(trials)
    trial_frame["selected_inner"] = (
        np.isclose(trial_frame["inner_combination_utility"], best[0])
        & (trial_frame["membership_gamma"] == best[1]["membership_gamma"])
        & (trial_frame["combination_method"] == best[1]["combination_method"])
        & (trial_frame["aggregation_method"] == best[1]["aggregation_method"])
        & (trial_frame["router_share"] == best[1]["router_share"])
    )
    return best[1], trial_frame


def _score_probe_stack(
    frame: pd.DataFrame, membership: np.ndarray, models: Sequence[ProbeModel],
    head_specs: Sequence[dict[str, Any]], combination: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.column_stack([model.score(frame)[0] for model in models]).astype(np.float32)
    ranks = np.column_stack([model.score(frame)[1] for model in models]).astype(np.float32)
    heads = np.column_stack([
        _combine_head_score(
            ranks[:, slot], membership[:, slot], gamma=float(combination["membership_gamma"]),
            method=str(combination["combination_method"]),
            activation_floor=float(head_specs[slot].get("membership_activation_floor", 0.0)),
        )
        for slot in range(ranks.shape[1])
    ])
    return _aggregate_heads(heads, str(combination["aggregation_method"])), ranks


def _permuted_memberships(membership: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute complete membership vectors, preserving simplex and per-head ESS."""
    return np.asarray(membership, dtype=np.float32)[np.random.default_rng(seed).permutation(len(membership))]


def _control_scores(
    train: pd.DataFrame, held: pd.DataFrame, membership_train: np.ndarray, membership_held: np.ndarray,
    models: Sequence[ProbeModel], selections: Sequence[dict[str, Any]], combination: dict[str, Any],
    discovery_fields: Sequence[str], predictive_fields: Sequence[str], config: dict[str, Any], *, seed: int,
) -> dict[str, np.ndarray]:
    """Matched controls that retrain every learned component they replace.

    C1/C2 intentionally do not reuse real-category specialist models: doing so
    would leave their category-specific supervision intact and understate the
    null.  Their target/model/feature-family selection is frozen from the real
    inner-fold winner, while the membership geometry itself is replaced.
    """
    probe_score, ranks = _score_probe_stack(held, membership_held, models, selections, combination)
    random_train = _permuted_memberships(membership_train, seed=seed + 101)
    random_held = _permuted_memberships(membership_held, seed=seed + 103)
    random_models = _refit_selected_heads(
        train, random_train, selections, discovery_fields, predictive_fields, config, seed=seed + 107,
    )
    random_ranks = np.column_stack([model.score(held)[1] for model in random_models]).astype(np.float32)
    random_heads = np.column_stack([
        _combine_head_score(
            random_ranks[:, slot], random_held[:, slot], gamma=float(combination["membership_gamma"]),
            method=str(combination["combination_method"]),
            activation_floor=float(selections[slot].get("membership_activation_floor", 0.0)),
        )
        for slot in range(random_ranks.shape[1])
    ])
    hard_train = np.eye(membership_train.shape[1], dtype=np.float32)[np.argmax(membership_train, axis=1)]
    hard_held = np.eye(membership_held.shape[1], dtype=np.float32)[np.argmax(membership_held, axis=1)]
    hard_models = _refit_selected_heads(
        train, hard_train, selections, discovery_fields, predictive_fields, config, seed=seed + 109,
    )
    hard_ranks = np.column_stack([model.score(held)[1] for model in hard_models]).astype(np.float32)
    hard_heads = np.column_stack([
        _combine_head_score(
            hard_ranks[:, slot], hard_held[:, slot], gamma=float(combination["membership_gamma"]),
            method=str(combination["combination_method"]),
            activation_floor=float(selections[slot].get("membership_activation_floor", 0.0)),
        )
        for slot in range(hard_ranks.shape[1])
    ])
    # C0: category-free full-universe control with exactly K probe seeds.
    ones = np.ones((len(train), 1), dtype=np.float32)
    generic_ranks: list[np.ndarray] = []
    for offset in range(len(models)):
        generic = _fit_probe(train, predictive_fields, 0, ones, "P1", "atr_utility", "lgbm_huber", config["probe"], config["preprocessing"], seed + offset)
        generic_ranks.append(generic.score(held)[1])
    return {
        "C0_multiseed_full_universe": np.mean(np.column_stack(generic_ranks), axis=1).astype(np.float32),
        "C1_random_membership": _aggregate_heads(random_heads, str(combination["aggregation_method"])),
        "C2_hard_category": _aggregate_heads(hard_heads, str(combination["aggregation_method"])),
        "C3_membership_only": np.max(membership_held, axis=1).astype(np.float32),
        "selected_probe": probe_score,
    }


def _constrained_gmm_control_score(
    train_fit: pd.DataFrame, inner: pd.DataFrame, train_final: pd.DataFrame, held: pd.DataFrame,
    target_free_final: pd.DataFrame, fitted_discovery: Sequence[tuple[dict[str, Any], CategoryModel]],
    discovery_metrics: pd.DataFrame, discovery_fields: Sequence[str], predictive_fields: Sequence[str],
    config: dict[str, Any], *, fold: str, seed: int,
) -> np.ndarray | None:
    """C4: rerun the same specialist flow on the best qualified GMM control."""
    viable_status = ["qualified"] if bool(config["discovery"].get("strict_structural_qualification", False)) else ["ok"]
    candidates = discovery_metrics.loc[
        discovery_metrics["algorithm"].eq("gmm") & discovery_metrics["status"].isin(viable_status)
    ].copy()
    if candidates.empty:
        return None
    score_column = "structural_selection_score" if "structural_selection_score" in candidates.columns else "inner_selection_utility"
    candidate = candidates.sort_values([score_column, "k", "covariance"], ascending=[False, True, True], kind="stable").iloc[0]
    covariance = None if pd.isna(candidate["covariance"]) else str(candidate["covariance"])
    model = next((
        fitted for spec, fitted in fitted_discovery
        if spec["algorithm"] == "gmm" and spec["k"] == int(candidate["k"])
        and spec["seed"] == int(candidate["seed"]) and spec["covariance"] == covariance
    ), None)
    if model is None:
        return None
    try:
        membership_train = model.membership(train_fit)
        membership_inner = model.membership(inner)
        selections, _, selected_models = _build_head_candidates(
            train_fit, inner, membership_train, membership_inner, discovery_fields, predictive_fields,
            config, fold=f"{fold}_C4", seed=seed + 17,
        )
        combination, _ = _choose_combination(inner, selected_models, selections, membership_inner, config, fold=f"{fold}_C4")
        final_model = _fit_category_model(
            target_free_final, discovery_fields, config["preprocessing"], algorithm="gmm",
            covariance=covariance, k=int(candidate["k"]), seed=int(candidate["seed"]),
        )
        final_models = _refit_selected_heads(
            train_final, final_model.membership(train_final), selections, discovery_fields, predictive_fields,
            config, seed=seed + 271,
        )
        return _score_probe_stack(held, final_model.membership(held), final_models, selections, combination)[0]
    except Exception:
        # C4 is a control.  A failed control is recorded by its absence rather
        # than weakening the structurally qualified primary representation.
        return None


def _complementarity(frame: pd.DataFrame, router: np.ndarray, probe: np.ndarray, *, budget: float, fold: str, split: str) -> pd.DataFrame:
    router_mask = _selection_mask(frame, router, None, budget, 1.0)
    probe_mask = _selection_mask(frame, probe, None, budget, 0.0)
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    cohorts = {
        "both": router_mask & probe_mask, "router_only": router_mask & ~probe_mask,
        "probe_only": probe_mask & ~router_mask, "neither": ~router_mask & ~probe_mask,
    }
    opportunities: dict[str, np.ndarray] = {
        "all_valid": valid,
        "net_gt_50": valid & (net > 50.0),
        "net_gt_100": valid & (net > 100.0),
        "net_gt_200": valid & (net > 200.0),
    }
    top5 = np.zeros(len(frame), dtype=bool)
    for _, group in frame.loc[valid].groupby("__decision_ts__", sort=False):
        n = max(1, int(math.ceil(len(group) * .05)))
        top5[group.sort_values(["policy_net_bps", "candidate_id"], ascending=[False, True], kind="stable").index.to_numpy(dtype=int)] = True
    opportunities["within_ts_top_5pct"] = valid & top5
    rows: list[dict[str, Any]] = []
    for opportunity_name, opportunity in opportunities.items():
        total_mass = float(np.nansum(np.maximum(net[opportunity], 0.0)))
        for name, mask in cohorts.items():
            cohort_mask = mask & opportunity
            values = net[cohort_mask]
            mass = float(np.nansum(np.maximum(values, 0.0)))
            rows.append({
                "fold": fold, "split": split, "budget_fraction": float(budget), "opportunity": opportunity_name,
                "cohort": name, "rows": int(cohort_mask.sum()), "opportunity_rows": int(opportunity.sum()),
                "mean_net_bps": float(np.nanmean(values)) if len(values) else float("nan"),
                "median_net_bps": float(np.nanmedian(values)) if len(values) else float("nan"),
                "positive_mass_bps": mass, "opportunity_positive_mass_bps": total_mass,
                "economic_mass_share": mass / max(total_mass, 1e-8),
            })
    return pd.DataFrame(rows)


def _category_blind_spots(frame: pd.DataFrame, membership: np.ndarray, router: np.ndarray, probe: np.ndarray, *, fold: str, split: str) -> pd.DataFrame:
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    router_top = _selection_mask(frame, router, None, .10, 1.0)
    probe_top = _selection_mask(frame, probe, None, .10, 0.0)
    rows: list[dict[str, Any]] = []
    for category in range(membership.shape[1]):
        weights = np.maximum(membership[:, category], 1e-8)
        row: dict[str, Any] = {
            "fold": fold, "split": split, "category": category,
            "ess": float(weights.sum() ** 2 / np.square(weights).sum()),
            "weighted_net_bps": float(np.average(np.where(np.isfinite(net), net, 0.0), weights=weights)),
            "weighted_hit_gt50": float(np.average(((net > 50.0) & valid).astype(float), weights=weights)),
            "router_top10_weight": float(np.average(router_top.astype(float), weights=weights)),
            "probe_top10_weight": float(np.average(probe_top.astype(float), weights=weights)),
            "router_rank_mean": float(np.average(router, weights=weights)),
        }
        for threshold in (50.0, 100.0, 200.0):
            opportunity = valid & (net > threshold)
            denominator = float(weights[opportunity].sum())
            row[f"router_recall_gt{int(threshold)}"] = float(weights[opportunity & router_top].sum() / max(denominator, 1e-12))
            row[f"probe_recall_gt{int(threshold)}"] = float(weights[opportunity & probe_top].sum() / max(denominator, 1e-12))
            row[f"union_recall_gt{int(threshold)}"] = float(weights[opportunity & (router_top | probe_top)].sum() / max(denominator, 1e-12))
        economic = valid & np.isfinite(net)
        total_mass = float(np.sum(weights[economic] * np.maximum(net[economic], 0.0)))
        row["router_positive_economic_mass_recall"] = float(
            np.sum(weights[economic & router_top] * np.maximum(net[economic & router_top], 0.0)) / max(total_mass, 1e-12)
        )
        row["probe_positive_economic_mass_recall"] = float(
            np.sum(weights[economic & probe_top] * np.maximum(net[economic & probe_top], 0.0)) / max(total_mass, 1e-12)
        )
        row["union_positive_economic_mass_recall"] = float(
            np.sum(weights[economic & (router_top | probe_top)] * np.maximum(net[economic & (router_top | probe_top)], 0.0)) / max(total_mass, 1e-12)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_fold(
    panel: pd.DataFrame, config: dict[str, Any], *, fold: str, held_start: object, held_end: object,
    discovery_fields: Sequence[str], predictive_fields: Sequence[str], seed: int, frozen: bool = False,
) -> dict[str, Any]:
    held_start_ts, held_end_ts = _utc(held_start), _utc(held_end)
    inner_start = held_start_ts - pd.DateOffset(months=int(config["inner_months"]))
    train_fit = _eligible_labels(panel, inner_start)
    train_fit = train_fit.loc[train_fit["__decision_ts__"] < inner_start].copy()
    inner = _eligible_labels(panel, held_start_ts)
    inner = inner.loc[(inner["__decision_ts__"] >= inner_start) & (inner["__decision_ts__"] < held_start_ts)].copy()
    train_final = _eligible_labels(panel, held_start_ts)
    train_final = train_final.loc[train_final["__decision_ts__"] < held_start_ts].copy()
    held = _month_range(panel, held_start_ts, held_end_ts).reset_index(drop=True)
    train_history_months = int(train_fit["__decision_ts__"].dt.to_period("M").nunique())
    if len(train_fit) < 5000 or len(inner) < 1000 or len(train_final) < 5000 or held.empty:
        raise RuntimeError(f"{fold}: insufficient strict train/inner/held support {len(train_fit)}/{len(inner)}/{len(train_final)}/{len(held)}")
    if train_history_months < int(config["minimum_train_months"]):
        raise RuntimeError(
            f"{fold}: only {train_history_months} pre-inner training months; "
            f"requires {int(config['minimum_train_months'])}"
        )
    minimum_coverage = float(config["preprocessing"]["min_feature_coverage"])
    target_free_fit_mask = panel["__decision_ts__"] < inner_start
    # Explicit target-free column views: labels, outcomes, Router rank and
    # candidate identity are physically absent from the discovery frames.  This
    # avoids a second full feature-panel copy while retaining a verifiable
    # no-label/no-score structural input contract.
    fitted_discovery_fields, discovery_coverage = _coverage_qualified_fields(
        panel.loc[target_free_fit_mask, list(discovery_fields)], discovery_fields,
        minimum_coverage=minimum_coverage, fold=fold, stage="discovery_fit",
    )
    fitted_predictive_fields, predictive_coverage = _coverage_qualified_fields(
        panel.loc[target_free_fit_mask, list(predictive_fields)], predictive_fields,
        minimum_coverage=minimum_coverage, fold=fold, stage="predictive_fit",
    )
    if not set(fitted_discovery_fields).issubset(fitted_predictive_fields):
        raise AssertionError(f"{fold}: target-free coverage gate broke P0/P1 nesting")
    target_free_fit = panel.loc[
        target_free_fit_mask, ["__decision_ts__", *fitted_discovery_fields]
    ].copy()
    fitted_discovery, discovery_metrics, discovery_definitions = _fit_discovery_candidates(
        target_free_fit, inner, fitted_discovery_fields, config, seed=seed, fold_name=fold,
    )
    del target_free_fit
    gc.collect()
    category_count_probe_ablation = _category_count_inner_ablation(
        train_fit, inner, fitted_discovery, discovery_metrics, fitted_discovery_fields,
        config, fold=fold, seed=seed + 80_000,
    )
    category_spec = _selected_category_spec(discovery_metrics)
    category_model = next(
        model for spec, model in fitted_discovery
        if spec["algorithm"] == category_spec["algorithm"] and spec["covariance"] == category_spec["covariance"]
        and spec["k"] == category_spec["k"] and spec["seed"] == category_spec["seed"]
        and float(spec.get("archetype_temperature_multiplier", 1.0)) == float(category_spec.get("archetype_temperature_multiplier", 1.0))
    )
    membership_train_fit = category_model.membership(train_fit)
    membership_inner = category_model.membership(inner)
    selections, head_trials, selection_models = _build_head_candidates(
        train_fit, inner, membership_train_fit, membership_inner, fitted_discovery_fields, fitted_predictive_fields,
        config, fold=fold, seed=seed + 100000,
    )
    combination, combination_trials = _choose_combination(inner, selection_models, selections, membership_inner, config, fold=fold)
    # Refit the selected unsupervised definition and specialist configurations
    # on the whole outer training history, still strictly before held start.
    target_free_final = panel.loc[
        panel["__decision_ts__"] < held_start_ts, ["__decision_ts__", *fitted_discovery_fields]
    ].copy()
    final_category_model = _fit_category_model(
        target_free_final, fitted_discovery_fields, config["preprocessing"], algorithm=category_spec["algorithm"],
        covariance=category_spec["covariance"], k=category_spec["k"], seed=int(category_spec["seed"]),
        archetype_temperature_multiplier=float(category_spec.get("archetype_temperature_multiplier", 1.0)),
    )
    membership_train_final = final_category_model.membership(train_final)
    membership_held = final_category_model.membership(held)
    final_models = _refit_selected_heads(
        train_final, membership_train_final, selections, fitted_discovery_fields, fitted_predictive_fields, config, seed=seed + 200000,
    )
    probe_score, head_ranks = _score_probe_stack(held, membership_held, final_models, selections, combination)
    metrics, masks = _evaluate_strategies(held, probe_score, fold=fold, split="validation" if frozen else "outer_oof", cfg=config, score_name="probe_score")
    controls = _control_scores(
        train_final, held, membership_train_final, membership_held, final_models, selections, combination,
        fitted_discovery_fields, fitted_predictive_fields, config, seed=seed + 300000,
    )
    gmm_control = None
    if bool(config.get("controls", {}).get("run_constrained_gmm", True)):
        gmm_control = _constrained_gmm_control_score(
            train_fit, inner, train_final, held, target_free_final, fitted_discovery, discovery_metrics,
            fitted_discovery_fields, fitted_predictive_fields, config, fold=fold, seed=seed + 350000,
        )
    if gmm_control is not None:
        controls["C4_constrained_gmm"] = gmm_control
    # C4 deliberately refits its GMM geometry on the exact target-free final
    # population.  Keep this compact structural frame alive until that matched
    # control has completed; releasing it beforehand made the control path
    # unbound while leaving the primary score otherwise valid.
    del target_free_final
    gc.collect()
    control_metrics: list[pd.DataFrame] = []
    for name, score in controls.items():
        if name == "selected_probe":
            continue
        candidate_metrics, _ = _evaluate_strategies(held, score, fold=fold, split="validation" if frozen else "outer_oof", cfg=config, score_name=name)
        candidate_metrics["control"] = name
        control_metrics.append(candidate_metrics)
    primary_budget = float(config["evaluation"]["selection_primary_budget"])
    primary_share = float(combination["router_share"])
    metrics["selected_inner_allocation"] = (
        (metrics["strategy"] == "router_plus_probe_rescue")
        & np.isclose(metrics["budget_fraction"], primary_budget)
        & np.isclose(metrics["router_share"], primary_share)
    )
    complementarity = _complementarity(held, held["router_rank"].to_numpy(dtype=float), probe_score, budget=primary_budget, fold=fold, split="validation" if frozen else "outer_oof")
    blind_spots = _category_blind_spots(held, membership_held, held["router_rank"].to_numpy(dtype=float), probe_score, fold=fold, split="validation" if frozen else "outer_oof")
    output = held[[
        *IDENTITY_COLUMNS, "router_rank", "label_valid", "probe_target_valid", "probe_atr_bps_14h", "policy_net_atr",
        "policy_net_bps", "policy_gross_bps", "policy_label_available_ts",
    ]].copy()
    output["fold"] = fold
    output["split"] = "validation" if frozen else "outer_oof"
    output["probe_score"] = probe_score
    output["category_entropy"] = -np.sum(np.clip(membership_held, 1e-8, 1.0) * np.log(np.clip(membership_held, 1e-8, 1.0)), axis=1)
    output["category_top2_margin"] = np.partition(membership_held, -1, axis=1)[:, -1] - (np.partition(membership_held, -2, axis=1)[:, -2] if membership_held.shape[1] > 1 else 0.0)
    for slot in range(membership_held.shape[1]):
        output[f"membership_{slot:02d}"] = membership_held[:, slot]
        output[f"probe_rank_{slot:02d}"] = head_ranks[:, slot]
    for share in config["evaluation"]["rescue_router_shares"]:
        key = (primary_budget, float(share))
        if key in masks:
            output[f"selected_rescue_b{primary_budget:g}_r{share:g}"] = masks[key]
    final_definitions: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    transformed = final_category_model.preprocessor.transform(held)
    primary_router = _selection_mask(held, held["router_rank"].to_numpy(dtype=float), None, primary_budget, 1.0)
    primary_probe = _selection_mask(held, probe_score, None, primary_budget, 0.0)
    primary_combined = masks[(primary_budget, primary_share)]
    held_valid = held["label_valid"].to_numpy(dtype=bool)
    held_net = pd.to_numeric(held["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    held_utility = pd.to_numeric(held["policy_net_atr"], errors="coerce").to_numpy(dtype=float)
    for slot in range(membership_held.shape[1]):
        weight = np.maximum(membership_held[:, slot], 1e-8)
        signature = np.average(transformed, axis=0, weights=weight).astype(float)
        order = np.argsort(signature)
        positive = order[-min(8, len(order)):][::-1]
        negative = order[:min(8, len(order))]
        qualifying = held_valid & np.isfinite(held_net)
        def weighted_recall(selected: np.ndarray, threshold: float) -> float:
            denominator = weight * (qualifying & (held_net > threshold))
            return float((denominator * selected).sum() / max(denominator.sum(), 1e-8))
        def weighted_rate(mask: np.ndarray) -> float:
            return float(weight[mask].sum() / max(weight[qualifying].sum(), 1e-8))
        utility_valid = qualifying & np.isfinite(held_utility)
        final_definitions.append({
            "fold": fold, "split": "validation" if frozen else "outer_oof", **category_spec, "category": slot,
            "held_ess": float(weight.sum() ** 2 / np.square(weight).sum()),
            "held_mean_membership": float(weight.mean()),
            "membership_p10": float(np.quantile(weight, .10)), "membership_median": float(np.median(weight)),
            "membership_p90": float(np.quantile(weight, .90)),
            "membership_share_gt_05": float(np.mean(weight > .5)), "membership_share_gt_07": float(np.mean(weight > .7)),
            "membership_share_gt_09": float(np.mean(weight > .9)),
            "weighted_mean_net_bps": float(np.average(np.where(np.isfinite(held_net), held_net, 0.0), weights=weight)),
            "net_median_bps": float(np.quantile(held_net[qualifying], .50)) if qualifying.any() else float("nan"),
            "utility_mean_atr": float(np.average(held_utility[utility_valid], weights=weight[utility_valid])) if utility_valid.any() else float("nan"),
            "utility_median_atr": float(np.nanmedian(held_utility[utility_valid])) if utility_valid.any() else float("nan"),
            "utility_positive_rate": weighted_rate(qualifying & (held_utility > 0.0)),
            "utility_gt_05_atr_rate": weighted_rate(qualifying & (held_utility > .5)),
            "utility_gt_1_atr_rate": weighted_rate(qualifying & (held_utility > 1.0)),
            "utility_gt_2_atr_rate": weighted_rate(qualifying & (held_utility > 2.0)),
            "net_positive_rate": weighted_rate(qualifying & (held_net > 0.0)),
            "net_gt50_rate": weighted_rate(qualifying & (held_net > 50.0)),
            "net_gt100_rate": weighted_rate(qualifying & (held_net > 100.0)),
            "net_gt200_rate": weighted_rate(qualifying & (held_net > 200.0)),
            "positive_economic_mass_share": float(
                np.sum(weight[qualifying] * np.maximum(held_net[qualifying], 0.0))
                / max(np.sum(np.maximum(held_net[qualifying], 0.0)), 1e-8)
            ),
            "router_weighted_recall_gt50": weighted_recall(primary_router, 50.0),
            "probe_weighted_recall_gt50": weighted_recall(primary_probe, 50.0),
            "combined_weighted_recall_gt50": weighted_recall(primary_combined, 50.0),
            "router_weighted_recall_gt100": weighted_recall(primary_router, 100.0),
            "probe_weighted_recall_gt100": weighted_recall(primary_probe, 100.0),
            "combined_weighted_recall_gt100": weighted_recall(primary_combined, 100.0),
            "structural_signature": signature.tolist(),
            "signature_fields": list(final_category_model.preprocessor.fields),
            "top_positive_features": [str(final_category_model.preprocessor.fields[index]) for index in positive],
            "top_negative_features": [str(final_category_model.preprocessor.fields[index]) for index in negative],
        })
        representative_positions = np.argsort(-weight)[: min(5, len(held))]
        for rank, position in enumerate(representative_positions, start=1):
            row = held.iloc[int(position)]
            representatives.append({
                "fold": fold, "split": "validation" if frozen else "outer_oof", "category": slot,
                "representative_rank": rank, "membership": float(weight[position]),
                "candidate_id": row["candidate_id"], "__decision_ts__": row["__decision_ts__"],
                "__symbol__": row["__symbol__"], "side_name": row["side_name"],
                "router_rank": float(row["router_rank"]), "probe_score": float(probe_score[position]),
                "policy_net_bps": float(held_net[position]) if np.isfinite(held_net[position]) else float("nan"),
                "label_valid": bool(held_valid[position]),
            })
    package = {
        "schema": SCHEMA, "fold": fold, "held_start": held_start_ts, "held_end": held_end_ts,
        "inner_start": inner_start, "category_spec": category_spec, "combination": combination,
        "category_model": final_category_model, "probe_models": final_models,
        "head_specs": [dict(spec) for spec in selections],
        "discovery_fields": tuple(fitted_discovery_fields), "predictive_fields": tuple(fitted_predictive_fields),
    }
    return {
        "metrics": metrics, "controls": pd.concat(control_metrics, ignore_index=True) if control_metrics else pd.DataFrame(),
        "predictions": output, "discovery_metrics": discovery_metrics,
        "discovery_definitions": discovery_definitions, "head_trials": pd.DataFrame(head_trials),
        "head_selections": pd.DataFrame(selections), "combination_trials": combination_trials,
        "category_definitions": pd.DataFrame(final_definitions), "complementarity": complementarity,
        "category_representatives": pd.DataFrame(representatives), "blind_spots": blind_spots,
        "category_count_probe_ablation": category_count_probe_ablation,
        "feature_coverage": pd.concat([discovery_coverage, predictive_coverage], ignore_index=True), "package": package,
        "fold_summary": {
            "fold": fold, "split": "validation" if frozen else "outer_oof", "held_start": str(held_start_ts),
            "held_end": str(held_end_ts), "inner_start": str(inner_start), "train_fit_rows": len(train_fit),
            "inner_rows": len(inner), "train_final_rows": len(train_final), "held_rows": len(held),
            "held_valid_label_rows": int(held["label_valid"].sum()),
            "pre_inner_training_months": train_history_months,
            "min_feature_coverage": minimum_coverage,
            "discovery_fields_kept": len(fitted_discovery_fields), "predictive_fields_kept": len(fitted_predictive_fields),
            **category_spec, **combination,
        },
    }


def _concat(results: Sequence[dict[str, Any]], key: str) -> pd.DataFrame:
    frames = [item[key] for item in results if isinstance(item.get(key), pd.DataFrame) and not item[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _monthly_metrics(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month, group in predictions.groupby(predictions["__decision_ts__"].dt.to_period("M"), sort=True):
        score = group["probe_score"].to_numpy(dtype=float)
        metrics, _ = _evaluate_strategies(group.reset_index(drop=True), score, fold=str(month), split=str(group["split"].iloc[0]), cfg=config, score_name="probe_score")
        pieces.append(metrics)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _quarterly_metrics(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for quarter, group in predictions.groupby(predictions["__decision_ts__"].dt.to_period("Q"), sort=True):
        score = group["probe_score"].to_numpy(dtype=float)
        metrics, _ = _evaluate_strategies(
            group.reset_index(drop=True), score, fold=str(quarter), split=str(group["split"].iloc[0]),
            cfg=config, score_name="probe_score",
        )
        pieces.append(metrics)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _cross_fold_category_stability(definitions: pd.DataFrame, fold_summary: pd.DataFrame) -> pd.DataFrame:
    """Match adjacent learned categories by structural signature, never outcomes."""
    columns = ["from_fold", "to_fold", "status", "common_fields", "from_category", "to_category", "signature_correlation"]
    if definitions.empty or fold_summary.empty:
        return pd.DataFrame(columns=columns)
    order = fold_summary.sort_values("held_start", kind="stable")["fold"].tolist()
    rows: list[dict[str, Any]] = []
    for prior, current in zip(order, order[1:]):
        left = definitions.loc[definitions["fold"] == prior].sort_values("category", kind="stable")
        right = definitions.loc[definitions["fold"] == current].sort_values("category", kind="stable")
        if left.empty or right.empty:
            continue
        left_vectors = [dict(zip(row.signature_fields, row.structural_signature)) for row in left.itertuples()]
        right_vectors = [dict(zip(row.signature_fields, row.structural_signature)) for row in right.itertuples()]
        common = sorted(set(left_vectors[0]).intersection(right_vectors[0]))
        if len(common) < 4:
            rows.append({"from_fold": prior, "to_fold": current, "status": "insufficient_common_features", "common_fields": len(common)})
            continue
        matrix = np.empty((len(left_vectors), len(right_vectors)), dtype=float)
        for i, left_value in enumerate(left_vectors):
            a = np.asarray([left_value[field] for field in common], dtype=float)
            for j, right_value in enumerate(right_vectors):
                b = np.asarray([right_value[field] for field in common], dtype=float)
                matrix[i, j] = np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
        matched_left, matched_right = linear_sum_assignment(-np.nan_to_num(matrix, nan=-1.0))
        for i, j in zip(matched_left, matched_right):
            rows.append({
                "from_fold": prior, "to_fold": current, "status": "ok", "common_fields": len(common),
                "from_category": int(left.iloc[i]["category"]), "to_category": int(right.iloc[j]["category"]),
                "signature_correlation": float(matrix[i, j]),
            })
    return pd.DataFrame(rows, columns=columns)


def _report_markdown(
    output: Path, manifest: dict[str, Any], metrics: pd.DataFrame, controls: pd.DataFrame,
    monthly: pd.DataFrame, fold_summary: pd.DataFrame, feature_coverage: pd.DataFrame,
    category_definitions: pd.DataFrame, complementarity: pd.DataFrame, category_crossfold_stability: pd.DataFrame,
    category_count_probe_ablation: pd.DataFrame, discovery_metrics: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 30) -> str:
        if frame.empty:
            return "_No rows._\n"
        view = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).copy()
        # ``DataFrame.to_markdown`` depends on the optional ``tabulate``
        # package, which is intentionally not part of the production research
        # environment.  Keep reporting self-contained: a deterministic GFM
        # table is sufficient for the audit and must never invalidate an
        # otherwise completed immutable experiment.
        def cell(value: object) -> str:
            if isinstance(value, (list, tuple, np.ndarray)):
                text = ", ".join(str(item) for item in value)
            elif pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            return text.replace("|", "\\|").replace("\n", "<br>")
        headers = [str(column) for column in view.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in view.itertuples(index=False, name=None))
        return "\n".join(lines) + "\n"
    primary = metrics.loc[(metrics["budget_fraction"] == .10) & (metrics["strategy"].isin(["router_only", "router_plus_probe_rescue"]))]
    text = [
        "# Data-Driven Opportunity Probes — Router Recall Research\n",
        "## Scope and causal boundary\n",
        "This is an offline, long-side Router-stage experiment. Candidate identities and all probe inputs are target-free causal fields. Policy outcomes are joined only after that population is sealed. No downstream Base, Meta, MC1, admission, sizing, portfolio, or live artifact is changed.\n",
        "The canonical probe formulations are shallow LGBM/CatBoost Huber regression or ordinal classification on policy-net utility normalised by the decision-time 14-hour ATR. The six ordinal bins are <=−1, (−1,0], (0,0.5], (0.5,1], (1,2], and >2 ATR. Timestamp LambdaRank is emitted only as an ablation. Every Router-recall/economic result below remains realised policy-net **bps**.\n",
        "Development is nested chronological 2024–2025; 2026 is a frozen validation period. The report is research evidence only.\n",
        "## Fold ledger\n", table(fold_summary, ["fold", "split", "inner_start", "held_start", "held_end", "pre_inner_training_months", "train_fit_rows", "inner_rows", "train_final_rows", "held_rows", "held_valid_label_rows", "algorithm", "covariance", "k", "membership_gamma", "combination_method", "aggregation_method"]),
        "## Target-free feature coverage gate\n",
        "Each fold freezes its usable probe inputs from the population available before the inner-selection cut-off. Fields below 90% source availability are omitted; this is not an imputation criterion.\n",
        table(feature_coverage.groupby(["fold", "stage"], as_index=False).agg(fields=("field", "size"), retained=("kept", "sum"), min_coverage=("coverage", "min"), median_coverage=("coverage", "median")) if not feature_coverage.empty else feature_coverage, ["fold", "stage", "fields", "retained", "min_coverage", "median_coverage"]),
        "## Structural archetype qualification — before any probe training\n",
        "A `qualified` row has passed anti-collapse, support, membership-shape, structural-distinctness, and independent adjacent-window stability gates using target-free training geometry only. `rejected_structural` candidates were never eligible for specialist probes.\n",
        table(discovery_metrics, ["fold", "algorithm", "covariance", "k", "status", "selected_inner", "max_category_mass_share", "min_ess_fraction", "min_ess_meets_5pct", "temporal_signature_correlation", "max_pairwise_signature_cosine", "median_max_membership", "median_second_membership", "median_effective_archetype_count", "reconstruction_mse", "gate_anti_collapse", "gate_support", "gate_membership_sparsity", "gate_membership_overlap", "gate_structural_distinctness", "gate_temporal_stability", "structural_selection_score"], limit=240),
        "## Learned category definitions\n",
        table(category_definitions, ["fold", "split", "category", "held_ess", "held_mean_membership", "membership_share_gt_05", "weighted_mean_net_bps", "router_weighted_recall_gt50", "probe_weighted_recall_gt50", "combined_weighted_recall_gt50", "top_positive_features", "top_negative_features"], limit=60),
        "## Adjacent-fold category signature stability\n", table(category_crossfold_stability, ["from_fold", "to_fold", "from_category", "to_category", "common_fields", "signature_correlation", "status"], limit=120),
        "## Viable category-count inner ablation\n", table(category_count_probe_ablation, ["fold", "k", "algorithm", "covariance", "status", "stability", "min_ess", "router_recall_gt100", "probe_recall_gt100", "combined_recall_gt100", "router_positive_economic_mass_recall", "combined_positive_economic_mass_recall"], limit=120),
        "## Primary 10% Router versus rescue comparison\n", table(primary, ["fold", "split", "strategy", "router_share", "selected_rows_valid", "recall_gt_50", "recall_gt_100", "positive_economic_mass_recall", "selected_mean_net_bps", "selected_cvar10_net_bps"]),
        "## Frozen-validation / monthly evidence\n", table(monthly.loc[(monthly["budget_fraction"] == .10) & (monthly["strategy"] == "router_plus_probe_rescue")], ["fold", "split", "router_share", "selected_rows_valid", "recall_gt_50", "selected_mean_net_bps", "selected_cvar10_net_bps"], limit=60),
        "## Negative controls\n", table(controls.loc[(controls.get("budget_fraction", pd.Series(dtype=float)) == .10)] if not controls.empty else controls, ["fold", "split", "control", "strategy", "router_share", "recall_gt_50", "selected_mean_net_bps", "selected_cvar10_net_bps"], limit=60),
        "## Complementarity at the primary candidate budget\n", table(complementarity.loc[complementarity.get("opportunity", pd.Series(dtype=str)).isin(["net_gt_50", "net_gt_100", "net_gt_200", "within_ts_top_5pct"])] if not complementarity.empty else complementarity, ["fold", "split", "opportunity", "cohort", "rows", "opportunity_rows", "mean_net_bps", "positive_mass_bps", "economic_mass_share"], limit=120),
        "## Interpretation gate\n",
        "A probe architecture can advance only if its matched Router+rescue results improve candidate-budget economic recall without a material stability failure, beat random-membership and membership-only controls, and retain the result on the untouched 2026 block. Otherwise the terminal decision is `DO_NOT_PROMOTE` or `INCONCLUSIVE`; this runner never changes the canonical Router.\n",
        "## Manifest\n",
        "```json\n" + json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n```\n",
    ]
    (output / "DATA_DRIVEN_OPPORTUNITY_PROBES_ROUTER_RECALL_REPORT.md").write_text("\n".join(text))


def _write_archetype_failure_artifacts(
    output: Path, config_path: Path, config: dict[str, Any], failure: ArchetypeDiscoveryFailure,
    *, source_audit: Sequence[dict[str, Any]],
) -> None:
    """Seal an explicit terminal report when structure fails before probes.

    A failed latent representation is a valid research result.  Leaving only a
    stack trace would invite someone to rerun it with weaker gates, so this
    writes the same immutable evidence classes as a completed experiment while
    making it impossible to mistake the result for a probe evaluation.
    """
    _write_parquet_exclusive(failure.metrics, output / "archetype_qualification.parquet")
    _write_parquet_exclusive(failure.definitions, output / "archetype_structural_definitions.parquet")
    correctness = {
        "schema": config["schema_version"],
        "archetype_layer_qualifies": False,
        "probe_training_started": False,
        "economic_labels_used_for_archetype_selection": False,
        "target_free_population_written_before_outcome_join": True,
        "reason": "no discovery candidate passed all structural gates",
    }
    manifest = {
        "schema": config["schema_version"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256_file(config_path),
        "decision": "ARCHETYPE_DISCOVERY_FAILED",
        "failed_fold": failure.fold,
        "source_audit": list(source_audit),
        "archetype_qualification": "archetype_qualification.parquet",
        "archetype_structural_definitions": "archetype_structural_definitions.parquet",
        "canonical_router_changed": False,
    }
    _write_json_exclusive(output / "correctness_report.json", correctness)
    _write_json_exclusive(output / "frozen_validation_decision.json", {"decision": "ARCHETYPE_DISCOVERY_FAILED", "failed_fold": failure.fold})
    _write_json_exclusive(output / "run_manifest.json", manifest)
    lines = [
        "# P8U Archetype Recovery — Structural Qualification Failure\n",
        "## Terminal decision\n\n`ARCHETYPE LAYER FAILS — DO NOT TRAIN PROBES`\n",
        f"The first failing fold is `{failure.fold}`.  The candidate universe and labels were sealed, but no specialist probe was fit.  The canonical Router, Base, Meta, MC1, admission, policy, portfolio, and live artifacts are unchanged.\n",
        "## Gate evidence\n",
    ]
    columns = [
        "algorithm", "covariance", "k", "status", "max_category_mass_share", "min_ess_fraction", "min_ess_meets_5pct",
        "temporal_signature_correlation", "max_pairwise_signature_cosine", "median_max_membership",
        "median_effective_archetype_count", "gate_anti_collapse", "gate_support",
        "gate_membership_sparsity", "gate_membership_overlap", "gate_structural_distinctness", "gate_temporal_stability", "error",
    ]
    available = failure.metrics.loc[:, [field for field in columns if field in failure.metrics.columns]]
    lines.append(available.to_csv(index=False))
    (output / "ARCHETYPE_RECOVERY_REPORT.md").write_text("\n".join(lines))


def _exclusive_day_end(value: object) -> pd.Timestamp:
    stamp = _utc(value)
    return (stamp.normalize() + pd.Timedelta(days=1)) if stamp == stamp.normalize() else stamp


def _run_structural_qualification_only(
    config_path: Path, output: Path, *, only_folds: set[str] | None = None,
) -> None:
    """Seal the target-free archetype gate before any label source is opened.

    This command is intentionally useful on its own: a failed structural layer
    is a terminal research result and must not be obscured by fitting a probe
    simply because policy labels happen to be readily available elsewhere.
    """
    config = _load_config(config_path)
    if config.get("schema_version") != ARCHETYPE_RECOVERY_SCHEMA:
        raise AssertionError("structural-only qualification requires the archetype-recovery schema")
    if output.exists():
        raise FileExistsError(f"immutable experiment output already exists: {output}")
    discovery_key = config["feature_contract_keys"]["discovery"]
    predictive_key = config["feature_contract_keys"]["predictive"]
    sidecar_key = config["probe_feature_sidecar_fields_key"]
    discovery_fields = _assert_causal_feature_contract(_P8U_FEATURE_CONTRACTS[discovery_key])
    predictive_fields = _assert_causal_feature_contract(_P8U_FEATURE_CONTRACTS[predictive_key])
    sidecar_fields = _assert_causal_feature_contract(_P8U_FEATURE_CONTRACTS[sidecar_key])
    selected_outer = [
        definition for definition in config["outer_folds"]
        if not only_folds or str(definition["name"]) in only_folds
    ]
    if not selected_outer:
        raise ValueError("no requested structural qualification folds")
    loaded_end = max(_exclusive_day_end(definition["held_end"]) for definition in selected_outer)
    target_free, target_free_view, source_audit = _read_panel(
        config, predictive_fields, sidecar_fields,
        start=config["research_period"][0], end=loaded_end,
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_parquet_exclusive(target_free_view, output / "target_free_candidate_universe.parquet")
    del target_free_view
    minimum_coverage = float(config["preprocessing"]["min_feature_coverage"])
    metrics: list[pd.DataFrame] = []
    definitions: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for offset, definition in enumerate(selected_outer):
        held_start = _utc(definition["held_start"])
        inner_start = held_start - pd.DateOffset(months=int(config["inner_months"]))
        train_target_free = target_free.loc[target_free["__decision_ts__"] < inner_start].copy()
        eligible_fields, audit = _coverage_qualified_fields(
            train_target_free, discovery_fields, minimum_coverage=minimum_coverage,
            fold=str(definition["name"]), stage="target_free_structural_qualification",
        )
        coverage.append(audit)
        try:
            _, candidate_metrics, candidate_definitions = _strict_structural_discovery_candidates(
                train_target_free, eligible_fields, config,
                seed=int(config["seed"]) + offset * 1_000_000,
                fold_name=str(definition["name"]),
            )
        except ArchetypeDiscoveryFailure as failure:
            candidate_metrics, candidate_definitions = failure.metrics, failure.definitions
        metrics.append(candidate_metrics)
        definitions.append(candidate_definitions)
        fold_rows.append({
            "fold": str(definition["name"]), "structural_training_end": str(inner_start),
            "target_free_rows": int(len(train_target_free)), "qualified_candidates": int(candidate_metrics["status"].eq("qualified").sum()),
            "decision": "QUALIFIED" if candidate_metrics["status"].eq("qualified").any() else "FAILED",
        })
    qualification = _concat([{"qualification": pd.concat(metrics, ignore_index=True)}], "qualification")
    structural_definitions = _concat([{"definitions": pd.concat(definitions, ignore_index=True)}], "definitions")
    coverage_audit = _concat([{"coverage": pd.concat(coverage, ignore_index=True)}], "coverage")
    all_passed = bool(pd.DataFrame(fold_rows)["decision"].eq("QUALIFIED").all())
    decision = "STRUCTURAL_QUALIFICATION_PASSED" if all_passed else "ARCHETYPE_DISCOVERY_FAILED"
    _write_parquet_exclusive(qualification, output / "archetype_qualification.parquet")
    _write_parquet_exclusive(structural_definitions, output / "archetype_structural_definitions.parquet")
    _write_parquet_exclusive(coverage_audit, output / "feature_coverage_audit.parquet")
    correctness = {
        "schema": config["schema_version"],
        "target_free_only": True,
        "policy_label_source_opened": False,
        "economic_labels_used_for_archetype_selection": False,
        "router_scores_used_for_archetype_selection": False,
        "probe_training_started": False,
        "all_discovery_features_causal_contract": True,
        "decision": decision,
    }
    manifest = {
        "schema": config["schema_version"], "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256_file(config_path), "decision": decision,
        "folds": fold_rows, "source_audit": source_audit,
        "canonical_router_changed": False,
        "next_step": "train probes only if every development fold has a qualified structural candidate",
    }
    _write_json_exclusive(output / "correctness_report.json", correctness)
    _write_json_exclusive(output / "frozen_validation_decision.json", {"decision": decision, "folds": fold_rows})
    _write_json_exclusive(output / "run_manifest.json", manifest)
    summary = pd.DataFrame(fold_rows).to_csv(index=False)
    evidence_columns = [
        "fold", "algorithm", "covariance", "k", "status", "max_category_mass_share",
        "min_ess_fraction", "min_ess_meets_5pct", "temporal_signature_correlation", "max_pairwise_signature_cosine",
        "median_max_membership", "median_effective_archetype_count", "gate_anti_collapse",
        "gate_support", "gate_membership_sparsity", "gate_membership_overlap", "gate_structural_distinctness", "gate_temporal_stability",
    ]
    evidence = qualification.loc[:, [column for column in evidence_columns if column in qualification.columns]].to_csv(index=False)
    (output / "ARCHETYPE_RECOVERY_STRUCTURAL_QUALIFICATION_REPORT.md").write_text(
        "# P8U Archetype Recovery — Target-Free Structural Qualification\n\n"
        f"## Decision\n\n`{decision}`\n\n"
        "No label file, policy outcome, Router score/rank, or probe target was opened during this run.\n\n"
        "## Fold summary\n\n" + summary + "\n## Candidate evidence\n\n" + evidence
    )


def _run(config_path: Path, output: Path, *, skip_validation: bool = False, only_folds: set[str] | None = None) -> None:
    config = _load_config(config_path)
    if config.get("schema_version") not in {SCHEMA, ARCHETYPE_RECOVERY_SCHEMA}:
        raise AssertionError(f"unexpected schema version {config.get('schema_version')!r}")
    if output.exists():
        raise FileExistsError(f"immutable experiment output already exists: {output}")
    discovery_key = config["feature_contract_keys"]["discovery"]
    predictive_key = config["feature_contract_keys"]["predictive"]
    sidecar_key = config["probe_feature_sidecar_fields_key"]
    try:
        discovery_source = _P8U_FEATURE_CONTRACTS[discovery_key]
        predictive_source = _P8U_FEATURE_CONTRACTS[predictive_key]
        sidecar_source = _P8U_FEATURE_CONTRACTS[sidecar_key]
    except KeyError as exc:
        raise KeyError(f"unknown lightweight P8U feature contract key: {exc.args[0]}") from exc
    discovery_fields = _assert_causal_feature_contract(discovery_source)
    predictive_fields = _assert_causal_feature_contract(predictive_source)
    if not set(discovery_fields).issubset(predictive_fields):
        raise AssertionError("P0 discovery contract must be a subset of P1 predictive contract")
    sidecar_fields = _assert_causal_feature_contract(sidecar_source)
    if not set(sidecar_fields).issubset(predictive_fields):
        raise AssertionError("all causal probe sidecar fields must be in the P1 predictive contract")
    if "probe_atr_bps_14h" not in sidecar_fields:
        raise AssertionError("ATR-normalised utility requires probe_atr_bps_14h in the causal sidecar contract")
    selected_outer = [
        definition for definition in config["outer_folds"]
        if not only_folds or str(definition["name"]) in only_folds
    ]
    include_validation = not skip_validation and (not only_folds or "frozen_2026" in only_folds)
    loaded_end = max(
        [_exclusive_day_end(definition["held_end"]) for definition in selected_outer]
        + ([_exclusive_day_end(config["frozen_validation_period"][1])] if include_validation else [])
    )
    target_free, target_free_view, source_audit = _read_panel(
        config, predictive_fields, sidecar_fields, start=config["research_period"][0], end=loaded_end,
    )
    # Persist and seal the pre-outcome population before labels are ever joined.
    output.mkdir(parents=True, exist_ok=False)
    (output / "bundles").mkdir()
    _write_parquet_exclusive(target_free_view, output / "target_free_candidate_universe.parquet")
    del target_free_view
    gc.collect()
    decision_start = target_free["__decision_ts__"].min()
    # The runner operates on half-open windows, so include every persisted
    # candidate timestamp through the panel's final hour and no later ledger
    # append.  This is a memory-only bound, not a label-availability filter.
    decision_end = target_free["__decision_ts__"].max() + pd.Timedelta(hours=1)
    labels = _load_labels(
        config["policy_label_path"],
        decision_start=decision_start,
        decision_end=decision_end,
    )
    panel = _attach_outcomes(target_free, labels)
    # ``panel`` is the same object, mutated in place by _attach_outcomes.
    # Drop aliases and the label index promptly before fold-level models are
    # fitted against the full feature panel.
    del target_free, labels
    gc.collect()
    _write_parquet_exclusive(
        panel[[
            *IDENTITY_COLUMNS, "label_join_status", "label_valid", "probe_target_valid", "policy_path_valid",
            "policy_label_available_ts", "probe_atr_bps_14h", "policy_net_atr", "policy_net_bps", "policy_gross_bps",
        ]],
        output / "outcome_joined_labels.parquet",
    )
    results: list[dict[str, Any]] = []
    for offset, definition in enumerate(selected_outer):
        held_end = _exclusive_day_end(definition["held_end"])
        try:
            result = _run_fold(
                panel, config, fold=str(definition["name"]), held_start=definition["held_start"], held_end=held_end,
                discovery_fields=discovery_fields, predictive_fields=predictive_fields,
                seed=int(config["seed"]) + offset * 1000000,
            )
        except ArchetypeDiscoveryFailure as failure:
            _write_archetype_failure_artifacts(output, config_path, config, failure, source_audit=source_audit)
            return
        results.append(result)
        joblib.dump(result["package"], output / "bundles" / f"{definition['name']}.joblib", compress=3)
        # The persisted bundle is the sole artifact needed after this point.  Do
        # not retain duplicate fitted boosters/preprocessors while later folds
        # run; all tabular fold outputs live outside ``package``.
        result.pop("package", None)
        gc.collect()
    if include_validation:
        validation_start = _utc(config["validation"]["fit_end"])
        validation_end = _exclusive_day_end(config["frozen_validation_period"][1])
        try:
            result = _run_fold(
                panel, config, fold="frozen_2026", held_start=validation_start, held_end=validation_end,
                discovery_fields=discovery_fields, predictive_fields=predictive_fields,
                seed=int(config["seed"]) + 9000000, frozen=True,
            )
        except ArchetypeDiscoveryFailure as failure:
            _write_archetype_failure_artifacts(output, config_path, config, failure, source_audit=source_audit)
            return
        results.append(result)
        joblib.dump(result["package"], output / "bundles" / "frozen_2026.joblib", compress=3)
        result.pop("package", None)
        gc.collect()
    metrics = _concat(results, "metrics")
    controls = _concat(results, "controls")
    predictions = _concat(results, "predictions")
    discovery_metrics = _concat(results, "discovery_metrics")
    discovery_definitions = _concat(results, "discovery_definitions")
    head_trials = _concat(results, "head_trials")
    head_selections = _concat(results, "head_selections")
    combination_trials = _concat(results, "combination_trials")
    category_definitions = _concat(results, "category_definitions")
    category_count_probe_ablation = _concat(results, "category_count_probe_ablation")
    complementarity = _concat(results, "complementarity")
    blind_spots = _concat(results, "blind_spots")
    category_representatives = _concat(results, "category_representatives")
    feature_coverage = _concat(results, "feature_coverage")
    fold_summary = pd.DataFrame([item["fold_summary"] for item in results])
    category_crossfold_stability = _cross_fold_category_stability(category_definitions, fold_summary)
    monthly = _monthly_metrics(predictions, config)
    quarterly = _quarterly_metrics(predictions, config)
    k_ablation = (
        discovery_metrics.loc[discovery_metrics["status"].isin(["ok", "qualified", "rejected_structural"])]
        .groupby(["algorithm", "covariance", "k"], dropna=False, as_index=False)
        .agg(
            folds=("fold", "nunique"), mean_stability=("stability", "mean"), min_stability=("stability", "min"),
            mean_ess=("mean_ess", "mean"), mean_utility=("structural_selection_score" if "structural_selection_score" in discovery_metrics.columns else "inner_selection_utility", "mean"),
            selected_folds=("selected_inner", "sum"),
        )
    )
    rescue_allocation = metrics.loc[metrics["strategy"] == "router_plus_probe_rescue"].copy()
    for name, frame in {
        "category_discovery_metrics.parquet": discovery_metrics,
        "category_definitions.parquet": category_definitions,
        "category_discovery_inner_definitions.parquet": discovery_definitions,
        "category_representatives.parquet": category_representatives,
        "category_crossfold_stability.parquet": category_crossfold_stability,
        "oof_memberships_predictions.parquet": predictions,
        "inner_selection_summary.parquet": head_selections,
        "head_trial_summary.parquet": head_trials,
        "combination_selection_summary.parquet": combination_trials,
        "matched_budget_recall.parquet": metrics,
        "rescue_allocation.parquet": rescue_allocation,
        "complementarity.parquet": complementarity,
        "category_blind_spots.parquet": blind_spots,
        "controls.parquet": controls,
        "category_count_ablation.parquet": k_ablation,
        "category_count_probe_ablation.parquet": category_count_probe_ablation,
        "feature_coverage_audit.parquet": feature_coverage,
        "monthly_metrics.parquet": monthly,
        "quarterly_metrics.parquet": quarterly,
        "fold_summary.parquet": fold_summary,
    }.items():
        _write_parquet_exclusive(frame, output / name)
    validation = metrics.loc[metrics["split"] == "validation"]
    development = metrics.loc[metrics["split"] == "outer_oof"]
    primary = float(config["evaluation"]["selection_primary_budget"])
    def chosen_rows(data: pd.DataFrame) -> pd.DataFrame:
        return data.loc[(data["strategy"] == "router_plus_probe_rescue") & np.isclose(data["budget_fraction"], primary) & data["selected_inner_allocation"].fillna(False)]
    dev_chosen, val_chosen = chosen_rows(development), chosen_rows(validation)
    dev_router = development.loc[(development["strategy"] == "router_only") & np.isclose(development["budget_fraction"], primary)]
    val_router = validation.loc[(validation["strategy"] == "router_only") & np.isclose(validation["budget_fraction"], primary)]
    def gain(chosen: pd.DataFrame, router: pd.DataFrame, column: str) -> float:
        return float(chosen[column].mean() - router[column].mean()) if not chosen.empty and not router.empty else float("nan")
    decision = "INCONCLUSIVE"
    dev_gain = gain(dev_chosen, dev_router, "recall_gt_50")
    val_gain = gain(val_chosen, val_router, "recall_gt_50")
    if np.isfinite(dev_gain) and np.isfinite(val_gain):
        decision = "PROMOTE_TO_ROUTER_CHALLENGER" if dev_gain >= float(config["evaluation"]["selection_min_gain_pp"]) / 100.0 and val_gain > 0.0 else "DO_NOT_PROMOTE"
    correctness = {
        "schema": SCHEMA,
        "target_free_population_written_before_outcome_join": True,
        "archetype_discovery_receives_only_target_free_panel": True,
        "probe_contract_has_no_forbidden_target_or_model_fields": True,
        "all_training_rows_resolved_before_train_cutoff": True,
        "all_folds_use_prior_inner_selection": True,
        "held_window_percentile_or_outcome_input": False,
        "invalid_or_missing_policy_or_atr_labels_excluded_from_supervised_fitting": True,
        "sparse_features_excluded_on_target_free_prior_training_coverage": True,
        "random_membership_and_hard_category_controls_retrain_specialists": True,
        "generic_multiseed_control_uses_learned_category_count": True,
        "category_specific_feature_selection_is_training_only": True,
        "canonical_probe_target": "policy_net_bps / decision_time_atr_bps",
        "timestamp_ranking_is_ablation_only": True,
        "development_uses_2024_2025_only": True,
        "validation_frozen_before_2026": not skip_validation,
        "router_input_used_only_for_evaluation_and_matched_rescue": True,
        "output_mutable": False,
    }
    manifest = {
        "schema": SCHEMA, "config_path": str(config_path.relative_to(ROOT)), "config_sha256": _sha256_file(config_path),
        "side": config["side"], "feature_contracts": {
            "discovery": {"key": config["feature_contract_keys"]["discovery"], "count": len(discovery_fields), "sha256": _hash_fields(discovery_fields), "fields": list(discovery_fields)},
            "predictive": {"key": config["feature_contract_keys"]["predictive"], "count": len(predictive_fields), "sha256": _hash_fields(predictive_fields), "fields": list(predictive_fields)},
            "causal_intraday_sidecar": {"key": config["probe_feature_sidecar_fields_key"], "count": len(sidecar_fields), "sha256": _hash_fields(sidecar_fields), "fields": list(sidecar_fields)},
        },
        "feature_coverage_gate": {
            "minimum_target_free_preheld_training_coverage": float(config["preprocessing"]["min_feature_coverage"]),
            "audit": "feature_coverage_audit.parquet",
            "rule": "features below the threshold are absent from the fold model contract; imputation never qualifies a feature",
        },
        "category_stability_audit": "category_crossfold_stability.parquet",
        "category_count_probe_ablation": "category_count_probe_ablation.parquet",
        "policy_label_path": config["policy_label_path"], "policy_label_sha256": _sha256_file(ROOT / config["policy_label_path"]),
        "target_definition": {
            "canonical": "policy_net_bps / causal_14h_atr_bps",
            "ordinal_bins_atr": ["<= -1", "(-1, 0]", "(0, 0.5]", "(0.5, 1]", "(1, 2]", "> 2"],
            "evaluation": "exact realised policy_net_bps; Router-recall metrics remain bps",
            "timestamp_ranker": "reported only as non-canonical ablation",
        },
        "source_audit": source_audit, "folds": fold_summary.to_dict(orient="records"),
        "decision": decision, "development_recall_gt50_delta": dev_gain, "frozen_2026_recall_gt50_delta": val_gain,
    }
    _write_json_exclusive(output / "correctness_report.json", correctness)
    _write_json_exclusive(output / "frozen_validation_decision.json", {"decision": decision, "development_recall_gt50_delta": dev_gain, "frozen_2026_recall_gt50_delta": val_gain})
    _write_json_exclusive(output / "run_manifest.json", manifest)
    _report_markdown(output, manifest, metrics, controls, monthly, fold_summary, feature_coverage, category_definitions, complementarity, category_crossfold_stability, category_count_probe_ablation, discovery_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-frozen-2026-validation", action="store_true")
    parser.add_argument("--only-fold", action="append", default=[], help="bounded smoke/debug run; repeated names are allowed")
    parser.add_argument(
        "--structural-qualification-only", action="store_true",
        help="seal target-free archetype gates without opening policy labels or training probes",
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.out if args.out.is_absolute() else ROOT / args.out
    if args.structural_qualification_only:
        if args.skip_frozen_2026_validation:
            raise ValueError("--skip-frozen-2026-validation is not applicable to structural-only qualification")
        _run_structural_qualification_only(config, output, only_folds=set(args.only_fold) or None)
    else:
        _run(config, output, skip_validation=args.skip_frozen_2026_validation, only_folds=set(args.only_fold) or None)


if __name__ == "__main__":
    main()
