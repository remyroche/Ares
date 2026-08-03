"""Bounded, sequential Stage-II meta conversion-archetype funnel.

Stage II is deliberately a *meta-only context* experiment.  A side-local
realised-path catalogue is discovered and recognised under the strict-OOF
contract in :mod:`stage_ii_meta_archetypes`; this module evaluates whether it
is economically distinct, stable and causally recognisable before allowing it
into a residual/meta comparison.  It never creates a side expert, hard-routes
rows, changes the base score, or performs local top-k ranking.

The module is in-memory orchestration only.  The caller provides a bounded
list of predeclared discovery candidates and an already strict-OOF meta
predictor.  It writes no data and starts no experiment itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_causal_admission import Causal21dAdmissionSpec, apply_causal_21d_side_admission
from .stage_ii_meta_archetypes import (
    META_ARCHETYPE_PREFIX,
    StageIIMetaArchetypeConfig,
    StageIIMetaArchetypeOOFResult,
    membership_feature_names,
    strict_oof_meta_archetype_features,
)


SCHEMA = "stage_ii_meta_conversion_archetype_funnel_v1"
_ALLOWED_COMPONENTS = frozenset({3, 4, 5, 6})
_CONTROL_ARMS = ("none", "soft_memberships", "prior", "both")
_TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
_ADMISSION_SCOPES = ("without_21d_admission", "with_21d_side_local_admission")
_DIRECT_R3_SEMANTICS = "same_side_direct_strict_oof_probabilities_without_conversion"


class StageIIFunnelError(ValueError):
    """Raised before a non-causal or unbounded Stage-II comparison is run."""


@dataclass(frozen=True)
class StageIIDiscoveryCandidate:
    """One explicitly predeclared realised-path discovery configuration."""

    candidate_id: str
    config: StageIIMetaArchetypeConfig
    causal_feature_cols: tuple[str, ...]


@dataclass(frozen=True)
class StageIIFunnelSpec:
    """Immutable inputs for the sequential Stage-II decision.

    ``base_expected_net_column`` must be the same-side strict-OOF base output.
    It is not mapped, calibrated, or converted before the meta model receives
    it.  ``meta_feature_cols`` are the frozen ordinary meta features; Stage-II
    additions are appended only to those meta inputs in the four controls.
    """

    candidate_id_column: str = "candidate_id"
    symbol_column: str = "symbol"
    decision_ts_column: str = "decision_ts"
    label_available_ts_column: str = "label_available_ts"
    side_column: str = "side_name"
    exact_net_column: str = "exact_net_bps"
    exact_gross_column: str = "exact_gross_bps"
    base_expected_net_column: str = "prequential_base_expected_net_bps"
    # The raw R3 simplex is a first-class meta handoff.  The expected-net map
    # alone is deliberately insufficient: it loses adverse/weak/clear shape.
    base_r3_probability_columns: tuple[str, str, str] = (
        "r3_p_adverse", "r3_p_weak", "r3_p_clear",
    )
    base_r3_oof_flag_column: str = "r3_is_strict_oof"
    base_r3_source_side_column: str = "r3_source_side"
    base_r3_fit_end_column: str = "r3_fit_end_ts"
    base_r3_semantics_column: str = "r3_score_semantics"
    base_r3_fold_id_column: str = "r3_oof_fold_id"
    base_r3_oof_fold_catalog: tuple[Mapping[str, Any], ...] = ()
    base_map_prequential_flag_column: str = "base_map_is_prequential"
    base_map_source_side_column: str = "base_map_source_side"
    base_map_max_label_available_column: str = "base_map_max_label_available_ts"
    meta_feature_cols: tuple[str, ...] = ()
    total_cost_bps: float = 100.0
    top_fractions: tuple[float, ...] = _TOP_FRACTIONS
    # Gates are intentionally small and explicit.  A candidate that cannot
    # clear them remains a diagnostic; no downstream meta control is run.
    min_oof_rows: int = 100
    min_economic_separation_bps: float = 10.0
    min_stable_fold_fraction: float = 0.50
    min_causal_log_loss_improvement: float = 0.00
    max_causal_membership_brier: float = 0.35
    max_symbol_share: float = 0.85
    min_control_segment_rows: int = 5
    control_selection_top_fraction: float = 0.10
    require_21d_admission_for_control_selection: bool = True
    admission_spec: Causal21dAdmissionSpec = field(default_factory=Causal21dAdmissionSpec)


@dataclass(frozen=True)
class StageIIMetaPredictionRequest:
    """A matched meta-only OOF request.

    The request contains no realised path descriptor as a feature.  ``target``
    exists for fitting only and is exactly net minus the frozen same-side base
    prediction.  Implementations must return validation predictions under the
    strict lineage stated in :class:`StageIIMetaPredictionResult`.
    """

    arm: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_residual_bps: np.ndarray
    candidate_ids: np.ndarray
    base_expected_net_bps: np.ndarray
    base_r3_probabilities: np.ndarray
    base_r3_probability_columns: tuple[str, str, str]
    base_handoff_provenance: Mapping[str, Any]
    decision_timestamps: np.ndarray
    label_available_timestamps: np.ndarray


@dataclass(frozen=True)
class StageIIMetaPredictionResult:
    """Strict-OOF residual output supplied by the downstream meta runner."""

    candidate_ids: Sequence[Any]
    predicted_residual_bps: Sequence[float]
    # Every output row names the strict validation fold which produced it.
    # This makes a global provenance claim insufficient on its own.
    oof_fold_ids: Sequence[int]
    provenance: Mapping[str, Any]


MetaOOFPredictor = Callable[[StageIIMetaPredictionRequest], StageIIMetaPredictionResult]


@dataclass(frozen=True)
class StageIIFunnelResult:
    candidate_audit: pd.DataFrame
    economic_stability: pd.DataFrame
    causal_predictability: pd.DataFrame
    control_metrics: pd.DataFrame
    selected_contributions: pd.DataFrame
    admission_audit: pd.DataFrame
    selected_candidate_id: str | None
    selected_control_arm: str | None
    manifest: Mapping[str, Any]
    oof_features: pd.DataFrame | None


def _utc(value: pd.Series, *, name: str) -> pd.Series:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if result.isna().any():
        raise StageIIFunnelError(f"{name} must contain valid UTC timestamps")
    return result


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    result = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(result).all():
        raise StageIIFunnelError(f"{column} must be finite")
    return result


def _canonical_boolean(series: pd.Series, *, name: str) -> np.ndarray:
    values: list[bool] = []
    for value in series.to_numpy(dtype=object):
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            values.append(bool(value))
        else:
            raise StageIIFunnelError(f"{name} must contain only explicit boolean/0/1 values")
    return np.asarray(values, dtype=bool)


def _validate_base_handoff(work: pd.DataFrame, spec: StageIIFunnelSpec) -> dict[str, Any]:
    """Require the direct same-side R3 simplex and map lineage row by row."""
    probability_columns = tuple(spec.base_r3_probability_columns)
    if len(probability_columns) != 3 or len(set(probability_columns)) != 3:
        raise StageIIFunnelError("base_r3_probability_columns must name adverse, weak and clear exactly once")
    required = {
        *probability_columns, spec.base_r3_oof_flag_column, spec.base_r3_source_side_column,
        spec.base_r3_fit_end_column, spec.base_r3_semantics_column, spec.base_r3_fold_id_column,
        spec.base_map_prequential_flag_column, spec.base_map_source_side_column,
        spec.base_map_max_label_available_column,
    }
    missing = sorted(required.difference(work.columns))
    if missing:
        raise StageIIFunnelError(f"direct same-side R3 base handoff is incomplete: {missing[:12]}")
    if not spec.base_r3_oof_fold_catalog:
        raise StageIIFunnelError("direct R3 handoff requires a strict row-level OOF fold catalogue")
    side = work[spec.side_column].astype(str).str.lower()
    if not _canonical_boolean(work[spec.base_r3_oof_flag_column], name=spec.base_r3_oof_flag_column).all():
        raise StageIIFunnelError("every Stage-II base R3 simplex row must be strict OOF")
    if not work[spec.base_r3_source_side_column].astype(str).str.lower().eq(side).all():
        raise StageIIFunnelError("Stage-II base R3 simplex must be direct same-side output")
    if not work[spec.base_r3_semantics_column].astype(str).eq(_DIRECT_R3_SEMANTICS).all():
        raise StageIIFunnelError("Stage-II base R3 simplex has converted or unverifiable semantics")
    probabilities = work.loc[:, list(probability_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise StageIIFunnelError("direct R3 adverse/weak/clear handoff must be a finite probability simplex")
    decision = work[spec.decision_ts_column]
    if not (_utc(work[spec.base_r3_fit_end_column], name=spec.base_r3_fit_end_column) < decision).all():
        raise StageIIFunnelError("every R3 fit must end strictly before its decision")
    if not _canonical_boolean(work[spec.base_map_prequential_flag_column], name=spec.base_map_prequential_flag_column).all():
        raise StageIIFunnelError("base expected-net map must be explicitly prequential")
    if not work[spec.base_map_source_side_column].astype(str).str.lower().eq(side).all():
        raise StageIIFunnelError("base expected-net map must use the matching same-side R3 simplex")
    if not (_utc(work[spec.base_map_max_label_available_column], name=spec.base_map_max_label_available_column) < decision).all():
        raise StageIIFunnelError("base expected-net map may not use current/future resolved labels")
    catalogue: dict[int, Mapping[str, Any]] = {}
    for item in spec.base_r3_oof_fold_catalog:
        if not isinstance(item, Mapping) or "fold_id" not in item:
            raise StageIIFunnelError("R3 fold catalogue needs mappings with explicit fold_id")
        fold_id = int(item["fold_id"])
        if fold_id in catalogue:
            raise StageIIFunnelError("R3 fold catalogue contains duplicate fold_id")
        train_max = pd.Timestamp(item.get("train_max_label_available_ts"))
        valid_start = pd.Timestamp(item.get("validation_start_ts"))
        train_max = train_max.tz_localize("UTC") if train_max.tzinfo is None else train_max.tz_convert("UTC")
        valid_start = valid_start.tz_localize("UTC") if valid_start.tzinfo is None else valid_start.tz_convert("UTC")
        if pd.isna(train_max) or pd.isna(valid_start) or not train_max < valid_start:
            raise StageIIFunnelError("R3 fold catalogue is not strictly prior-resolved")
        catalogue[fold_id] = item
    row_folds = pd.to_numeric(work[spec.base_r3_fold_id_column], errors="coerce").to_numpy(float)
    if not np.isfinite(row_folds).all() or not np.equal(row_folds, np.floor(row_folds)).all():
        raise StageIIFunnelError("R3 OOF fold ids must be finite integers")
    fold_ids = row_folds.astype(np.int64)
    if not set(fold_ids).issubset(catalogue):
        raise StageIIFunnelError("R3 rows reference a fold absent from the strict OOF catalogue")
    for fold_id in np.unique(fold_ids):
        item = catalogue[int(fold_id)]
        start = pd.Timestamp(item["validation_start_ts"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end_value = item.get("validation_end_ts")
        end = None if end_value is None or pd.isna(end_value) else pd.Timestamp(end_value)
        if end is not None:
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        mask = fold_ids == fold_id
        if not decision.loc[mask].ge(start).all() or (end is not None and not decision.loc[mask].lt(end).all()):
            raise StageIIFunnelError("R3 row fold assignment falls outside its validation interval")
        fit_end = _utc(work.loc[mask, spec.base_r3_fit_end_column], name=spec.base_r3_fit_end_column)
        if not fit_end.le(start).all():
            raise StageIIFunnelError("R3 fit end must not enter its strict OOF validation fold")
    return {
        "r3_probability_columns": probability_columns,
        "r3_semantics": _DIRECT_R3_SEMANTICS,
        "r3_strict_oof": True,
        "r3_fold_ids": tuple(sorted(int(value) for value in np.unique(fold_ids))),
        "base_map_prequential": True,
        "base_map_source": "same_side_direct_r3_simplex",
    }


def _validate_spec(frame: pd.DataFrame, spec: StageIIFunnelSpec) -> pd.DataFrame:
    if not spec.meta_feature_cols:
        raise StageIIFunnelError("Stage II requires an explicit frozen meta feature list")
    if tuple(sorted(set(spec.top_fractions))) != spec.top_fractions or any(
        not 0.0 < value <= 1.0 for value in spec.top_fractions
    ):
        raise StageIIFunnelError("top fractions must be unique, increasing and in (0, 1]")
    if not np.isfinite(spec.total_cost_bps) or spec.total_cost_bps <= 0:
        raise StageIIFunnelError("total_cost_bps must be positive and finite")
    if not 0.0 <= spec.max_causal_membership_brier <= 1.0:
        raise StageIIFunnelError("max_causal_membership_brier must lie in [0, 1]")
    if int(spec.min_control_segment_rows) < 1:
        raise StageIIFunnelError("min_control_segment_rows must be positive")
    if spec.control_selection_top_fraction not in spec.top_fractions:
        raise StageIIFunnelError("control_selection_top_fraction must be one declared pooled-global tail")
    required = {
        spec.candidate_id_column, spec.symbol_column, spec.decision_ts_column,
        spec.label_available_ts_column, spec.side_column, spec.exact_net_column,
        spec.exact_gross_column, spec.base_expected_net_column, *spec.meta_feature_cols,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StageIIFunnelError(f"Stage-II ledger lacks columns: {missing[:12]}")
    work = frame.copy()
    work[spec.decision_ts_column] = _utc(work[spec.decision_ts_column], name=spec.decision_ts_column)
    work[spec.label_available_ts_column] = _utc(work[spec.label_available_ts_column], name=spec.label_available_ts_column)
    if (work[spec.label_available_ts_column] <= work[spec.decision_ts_column]).any():
        raise StageIIFunnelError("labels must resolve strictly after their decision timestamp")
    candidate = work[spec.candidate_id_column].astype("string")
    symbol = work[spec.symbol_column].astype("string")
    side = work[spec.side_column].astype("string").str.lower()
    if candidate.isna().any() or candidate.str.strip().eq("").any() or candidate.duplicated().any():
        raise StageIIFunnelError("candidate identities must be non-empty and immutable")
    if symbol.isna().any() or symbol.str.strip().eq("").any():
        raise StageIIFunnelError("symbols must be non-empty")
    if not side.isin(("long", "short")).all():
        raise StageIIFunnelError("Stage II requires canonical long/short sides")
    identity = pd.DataFrame({"id": candidate, "symbol": symbol, "ts": work[spec.decision_ts_column], "side": side})
    if identity.duplicated().any():
        raise StageIIFunnelError("candidate/symbol/time/side identity must be unique")
    work[spec.side_column] = side
    net = _finite(work, spec.exact_net_column)
    gross = _finite(work, spec.exact_gross_column)
    _finite(work, spec.base_expected_net_column)
    for name in spec.meta_feature_cols:
        if not pd.api.types.is_numeric_dtype(work[name]):
            raise StageIIFunnelError(f"meta feature {name!r} must be numeric")
    # Labels are commonly persisted as float32 bps; permit only that storage
    # round-off, not a second cost debit or a gross/net unit mismatch.
    if not np.allclose(gross - spec.total_cost_bps, net, rtol=0.0, atol=1e-3):
        raise StageIIFunnelError("gross minus fixed cost must reconcile to exact net exactly once")
    work = work.sort_values([spec.decision_ts_column, spec.candidate_id_column], kind="stable").reset_index(drop=True)
    _validate_base_handoff(work, spec)
    return work


def _validate_candidates(
    candidates: Sequence[StageIIDiscoveryCandidate], frame: pd.DataFrame, spec: StageIIFunnelSpec,
) -> tuple[StageIIDiscoveryCandidate, ...]:
    values = tuple(candidates)
    if not values or len(values) > 8:
        raise StageIIFunnelError("Stage II requires one to eight explicitly bounded candidates")
    ids = [str(value.candidate_id) for value in values]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise StageIIFunnelError("candidate IDs must be non-empty and unique")
    for candidate in values:
        cfg = candidate.config
        expected_columns = {
            "decision_ts_col": spec.decision_ts_column,
            "label_available_ts_col": spec.label_available_ts_column,
            "side_col": spec.side_column,
            "exact_net_col": spec.exact_net_column,
            "base_expected_net_col": spec.base_expected_net_column,
        }
        for attribute, expected in expected_columns.items():
            if getattr(cfg, attribute) != expected:
                raise StageIIFunnelError(
                    f"candidate {candidate.candidate_id!r} {attribute} must use the frozen Stage-II ledger contract"
                )
        if int(cfg.components) not in _ALLOWED_COMPONENTS:
            raise StageIIFunnelError("candidate components must be one of 3, 4, 5 or 6")
        if not cfg.path_descriptor_cols or len(cfg.path_descriptor_cols) > 12:
            raise StageIIFunnelError("path descriptor view must contain one to twelve predeclared fields")
        if not candidate.causal_feature_cols:
            raise StageIIFunnelError("each discovery candidate requires explicit causal inputs")
        if any(name not in frame for name in candidate.causal_feature_cols):
            raise StageIIFunnelError("candidate causal feature is absent from the ledger")
        if set(candidate.causal_feature_cols).intersection({cfg.exact_net_col, *cfg.path_descriptor_cols}):
            raise StageIIFunnelError("realised outcomes cannot enter a causal recogniser")
        if set(spec.meta_feature_cols).intersection({cfg.exact_net_col, *cfg.path_descriptor_cols}):
            raise StageIIFunnelError("realised path descriptors cannot enter any Stage-II meta control")
    return values


def _fold_for_rows(frame: pd.DataFrame, audit: pd.DataFrame, *, decision_column: str) -> pd.Series:
    output = pd.Series(-1, index=frame.index, dtype=np.int32)
    if audit.empty:
        return output
    decision = frame[decision_column]
    for row in audit.loc[audit.status.eq("scored")].itertuples(index=False):
        start = pd.Timestamp(row.valid_start)
        end = getattr(row, "valid_end", None)
        mask = decision.ge(start) if pd.isna(end) else decision.ge(start) & decision.lt(pd.Timestamp(end))
        output.loc[mask] = int(row.fold)
    return output


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=float)
    value = np.asarray(value, dtype=float)
    return float(np.dot(value, weight) / weight.sum()) if weight.sum() > 0 else float("nan")


def _diagnose_candidate(
    candidate: StageIIDiscoveryCandidate,
    result: StageIIMetaArchetypeOOFResult,
    frame: pd.DataFrame,
    spec: StageIIFunnelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Economic separation/stability and causal-recogniser diagnostics."""
    names = membership_feature_names(candidate.config.components)
    truth = result.diagnostic_truth_memberships.loc[:, names]
    fold = _fold_for_rows(frame, result.fold_audit, decision_column=spec.decision_ts_column)
    work = pd.DataFrame({
        "fold": fold,
        "month": frame[spec.decision_ts_column].dt.strftime("%Y-%m"),
        "side": frame[spec.side_column].astype(str),
        "symbol": frame[spec.symbol_column].astype(str),
        "net": frame[spec.exact_net_column].to_numpy(float),
        "gross": frame[spec.exact_gross_column].to_numpy(float),
    }, index=frame.index)
    work["archetype_rank"] = truth.to_numpy(float).argmax(axis=1)
    work["membership"] = truth.to_numpy(float).max(axis=1)
    work = work.loc[(fold >= 0) & np.isfinite(truth.to_numpy(float)).all(axis=1)].copy()
    rows: list[dict[str, Any]] = []
    if not work.empty:
        for (fold_id, month, side, rank), group in work.groupby(["fold", "month", "side", "archetype_rank"], sort=True):
            weight = group.membership.to_numpy(float)
            dominant = group.symbol.value_counts(normalize=True).max()
            rows.append({
                "candidate_id": candidate.candidate_id, "fold": int(fold_id), "month": str(month),
                "side": str(side), "archetype_rank": int(rank), "support_rows": int(len(group)),
                "effective_support": float(weight.sum()), "mean_net_bps": _weighted_mean(group.net, weight),
                "mean_gross_bps": _weighted_mean(group.gross, weight), "max_symbol_share": float(dominant),
            })
    stability = pd.DataFrame(rows)
    fold_rows: list[dict[str, Any]] = []
    for (fold_id, side), group in stability.groupby(["fold", "side"], sort=True) if not stability.empty else []:
        valid = group.loc[group.support_rows.ge(1)]
        spread = float(valid.mean_net_bps.max() - valid.mean_net_bps.min()) if len(valid) >= 2 else np.nan
        ordered = valid.sort_values("archetype_rank", kind="stable")
        rank_corr = float(pd.Series(ordered.archetype_rank).corr(ordered.mean_net_bps, method="spearman")) if len(ordered) >= 2 else np.nan
        fold_rows.append({"candidate_id": candidate.candidate_id, "fold": int(fold_id), "side": str(side), "economic_separation_bps": spread, "rank_net_spearman": rank_corr, "max_symbol_share": float(valid.max_symbol_share.max())})
    fold_summary = pd.DataFrame(fold_rows)
    audit = result.fold_audit.loc[result.fold_audit.status.eq("scored")]
    log_loss = pd.to_numeric(audit.get("causal_membership_log_loss"), errors="coerce")
    brier = pd.to_numeric(audit.get("causal_membership_brier"), errors="coerce")
    uniform_loss = float(np.log(candidate.config.components))
    predictability = pd.DataFrame([{
        "candidate_id": candidate.candidate_id,
        "scored_folds": int(len(audit)),
        "diagnostic_labelled_rows": int(pd.to_numeric(audit.get("diagnostic_labelled_rows"), errors="coerce").fillna(0).sum()),
        "causal_membership_log_loss": float(log_loss.mean()) if len(log_loss) else np.nan,
        "causal_membership_brier": float(brier.mean()) if len(brier) else np.nan,
        "uniform_log_loss": uniform_loss,
        "causal_log_loss_improvement_vs_uniform": float(uniform_loss - log_loss.mean()) if len(log_loss) else np.nan,
    }])
    summary = {
        "oof_rows": int(result.manifest.get("scored_rows", 0)),
        "economic_separation_bps": float(fold_summary.economic_separation_bps.median()) if not fold_summary.empty else np.nan,
        "stable_fold_fraction": float(fold_summary.economic_separation_bps.ge(spec.min_economic_separation_bps).mean()) if not fold_summary.empty else 0.0,
        "max_symbol_share": float(stability.max_symbol_share.max()) if not stability.empty else np.nan,
        "causal_log_loss_improvement": float(predictability.causal_log_loss_improvement_vs_uniform.iloc[0]),
        "causal_membership_brier": float(predictability.causal_membership_brier.iloc[0]),
    }
    return stability, predictability, summary


def _feature_control_columns(result: StageIIMetaArchetypeOOFResult, candidate: StageIIDiscoveryCandidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = membership_feature_names(candidate.config.components)
    soft = tuple([*names, f"{META_ARCHETYPE_PREFIX}prob__unknown", f"{META_ARCHETYPE_PREFIX}entropy", f"{META_ARCHETYPE_PREFIX}confidence", f"{META_ARCHETYPE_PREFIX}support_log1p", f"{META_ARCHETYPE_PREFIX}available"])
    prior = (f"{META_ARCHETYPE_PREFIX}prior_residual_bps",)
    if any(name not in result.features for name in (*soft, *prior)):
        raise StageIIFunnelError("strict OOF archetype result has an incomplete published feature contract")
    return soft, prior


def _validate_meta_prediction(
    result: StageIIMetaPredictionResult,
    request: StageIIMetaPredictionRequest,
) -> np.ndarray:
    ids = np.asarray(result.candidate_ids, dtype=object).reshape(-1)
    expected = np.asarray(request.candidate_ids, dtype=object)
    values = np.asarray(result.predicted_residual_bps, dtype=float).reshape(-1)
    if len(ids) != len(expected) or not np.array_equal(ids, expected):
        raise StageIIFunnelError("all controls must return identical ordered candidate rows")
    if len(values) != len(expected) or not np.isfinite(values).all():
        raise StageIIFunnelError("meta output must be finite and row-aligned")
    row_folds = np.asarray(result.oof_fold_ids, dtype=np.int64).reshape(-1)
    if len(row_folds) != len(expected) or (row_folds < 0).any():
        raise StageIIFunnelError("meta output requires one non-negative strict fold id per row")
    provenance = result.provenance
    if not isinstance(provenance, Mapping) or provenance.get("strict_oof") is not True:
        raise StageIIFunnelError("meta output must declare strict_oof=True")
    if str(provenance.get("layer", "")).lower() not in {"meta", "residual", "meta_residual"}:
        raise StageIIFunnelError("Stage II controls may only fit a meta/residual layer")
    if str(provenance.get("score_semantics", "")) != "raw_predicted_residual_bps":
        raise StageIIFunnelError("meta output must be an unconverted predicted residual in bps")
    if provenance.get("base_model_changed", False) is not False:
        raise StageIIFunnelError("Stage II must not alter the base model")
    handoff = provenance.get("base_handoff")
    if not isinstance(handoff, Mapping):
        raise StageIIFunnelError("meta output must preserve direct R3 base-handoff provenance")
    if tuple(handoff.get("r3_probability_columns", ())) != request.base_r3_probability_columns:
        raise StageIIFunnelError("meta output omitted or changed direct R3 simplex columns")
    if handoff.get("r3_semantics") != _DIRECT_R3_SEMANTICS or handoff.get("r3_strict_oof") is not True:
        raise StageIIFunnelError("meta output did not preserve direct strict-OOF R3 semantics")
    if handoff.get("base_map_source") != "same_side_direct_r3_simplex" or handoff.get("base_map_prequential") is not True:
        raise StageIIFunnelError("meta output did not preserve same-side prequential base-map provenance")
    folds = provenance.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or not folds:
        raise StageIIFunnelError("meta output requires strict fold availability lineage")
    fold_by_id: dict[int, Mapping[str, Any]] = {}
    for value in folds:
        if not isinstance(value, Mapping):
            raise StageIIFunnelError("meta fold lineage is invalid")
        if "fold_id" not in value:
            raise StageIIFunnelError("meta fold lineage requires an explicit fold_id")
        fold_id = int(value["fold_id"])
        if fold_id in fold_by_id:
            raise StageIIFunnelError("meta fold lineage contains duplicate fold_id")
        fold_by_id[fold_id] = value
        maximum = pd.Timestamp(value.get("train_max_label_available_ts"))
        start = pd.Timestamp(value.get("validation_start_ts"))
        if maximum.tzinfo is None:
            maximum = maximum.tz_localize("UTC")
        else:
            maximum = maximum.tz_convert("UTC")
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if pd.isna(maximum) or pd.isna(start) or not maximum < start:
            raise StageIIFunnelError("meta fold is not strictly prior-resolved")
    if not set(row_folds).issubset(fold_by_id):
        raise StageIIFunnelError("meta output references a fold absent from lineage")
    decisions = pd.to_datetime(pd.Series(request.decision_timestamps), utc=True, errors="coerce")
    if decisions.isna().any():
        raise StageIIFunnelError("request decision timestamps are invalid")
    for fold_id in np.unique(row_folds):
        provenance_fold = fold_by_id[int(fold_id)]
        start = pd.Timestamp(provenance_fold["validation_start_ts"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end_value = provenance_fold.get("validation_end_ts")
        end = None if end_value is None or pd.isna(end_value) else pd.Timestamp(end_value)
        if end is not None:
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        mask = row_folds == fold_id
        if not decisions.loc[mask].ge(start).all() or (end is not None and not decisions.loc[mask].lt(end).all()):
            raise StageIIFunnelError("meta row fold assignment falls outside its declared validation interval")
    returned = tuple(provenance.get("feature_columns", ()))
    if returned != request.feature_columns:
        raise StageIIFunnelError("meta output feature contract differs from the matched request")
    return values.astype(np.float32)


def _pooled_global_metrics(
    frame: pd.DataFrame,
    *,
    arm: str,
    score_column: str,
    spec: StageIIFunnelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank once globally, then attribute the exact selected rows only."""
    base = frame.copy()
    base["__score"] = _finite(base, score_column)
    # Preserve the canonical exact-net column for reporting.  The admission
    # helper uses its historical ``net_bps`` default independently.
    base["net_bps"] = base[spec.exact_net_column].to_numpy(np.float64)
    mapped, admission_audit = apply_causal_21d_side_admission(
        base,
        score_column="__score", net_column="net_bps", decision_column=spec.decision_ts_column,
        label_available_column=spec.label_available_ts_column, identity_column=spec.candidate_id_column,
        spec=spec.admission_spec,
    )
    rows: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    original_n = len(mapped)
    # One and only one result per admission scope/tail.  The causal mapper
    # returns a full population; the admitted view is a filtered view of that
    # same ledger, not a second independently evaluated arm.
    populations = (
        ("without_21d_admission", mapped, "__score"),
        ("with_21d_side_local_admission", mapped.loc[mapped.causal_21d_side_admitted_ge_50bps & mapped.causal_21d_side_expected_net_bps.notna()], "causal_21d_side_expected_net_bps"),
    )
    if tuple(scope for scope, _, _ in populations) != _ADMISSION_SCOPES:
        raise AssertionError("Stage-II admission scopes must be canonical and non-duplicated")
    for scope, population, score in populations:
        ordered = population.sort_values([score, spec.candidate_id_column], ascending=[False, True], kind="stable")
        for fraction in spec.top_fractions:
            requested = max(1, int(np.ceil(fraction * original_n)))
            selected = ordered.head(min(requested, len(ordered))).copy()
            common = {"arm": arm, "admission_scope": scope, "top_fraction": float(fraction), "ranking_basis": "pooled_global_after_common_bps_mapping_no_side_or_month_rerank", "original_population_rows": int(original_n), "eligible_rows": int(len(population)), "selected_rows": int(len(selected))}
            rows.append({**common, "gross_bps_per_trade": float(selected[spec.exact_gross_column].mean()) if len(selected) else np.nan, "net_bps_per_trade": float(selected[spec.exact_net_column].mean()) if len(selected) else np.nan, "gross_bps_sum": float(selected[spec.exact_gross_column].sum()), "net_bps_sum": float(selected[spec.exact_net_column].sum()), "selected_long_rows": int(selected[spec.side_column].eq("long").sum()), "selected_short_rows": int(selected[spec.side_column].eq("short").sum())})
            selected["__month"] = selected[spec.decision_ts_column].dt.strftime("%Y-%m")
            for scope_name, columns in (("side", [spec.side_column]), ("month", ["__month"]), ("month_side", ["__month", spec.side_column])):
                for keys, group in selected.groupby(columns, sort=True, observed=True):
                    values = keys if isinstance(keys, tuple) else (keys,)
                    record = {**common, "scope": scope_name, "month": "__all__", "side": "__all__", "selected_rows": int(len(group)), "gross_bps_per_trade": float(group[spec.exact_gross_column].mean()), "net_bps_per_trade": float(group[spec.exact_net_column].mean()), "gross_bps_sum": float(group[spec.exact_gross_column].sum()), "net_bps_sum": float(group[spec.exact_net_column].sum())}
                    for name, value in zip(columns, values, strict=True):
                        record["month" if name == "__month" else "side"] = str(value)
                    contributions.append(record)
    metrics = pd.DataFrame(rows)
    if metrics.duplicated(["arm", "admission_scope", "top_fraction"]).any():
        raise AssertionError("Stage-II pooled metrics duplicated an admission arm")
    return metrics, pd.DataFrame(contributions), admission_audit.assign(arm=arm)


def _control_selection_summary(
    metrics: pd.DataFrame,
    contributions: pd.DataFrame,
    *,
    arm: str,
    spec: StageIIFunnelSpec,
) -> dict[str, Any]:
    """Robust sequential choice: worst period/side before aggregate tail EV.

    These are contributions of one frozen pooled-global tail, never monthly or
    side-local reranks.  An arm lacking a usable admission view is explicitly
    diagnostic when the frozen policy requires one rather than being rewarded
    for a strong unadmitted aggregate.
    """
    top = metrics.loc[np.isclose(metrics.top_fraction, spec.control_selection_top_fraction)].copy()
    if top.duplicated("admission_scope").any() or not set(top.admission_scope).issubset(_ADMISSION_SCOPES):
        raise StageIIFunnelError("control selection received duplicated or unknown admission metrics")
    scope_count = int(top.admission_scope.nunique())
    eligible_scopes = set(top.admission_scope)
    scope_complete = eligible_scopes == set(_ADMISSION_SCOPES)
    # Empty admitted populations produce NaN tail EV and cannot pass a robust
    # admission-aware choice.  We retain their metrics as a diagnostic.
    aggregate = pd.to_numeric(top.net_bps_per_trade, errors="coerce").to_numpy(float)
    valid_aggregate = aggregate[np.isfinite(aggregate)]
    selected = contributions.loc[
        np.isclose(contributions.top_fraction, spec.control_selection_top_fraction)
        & contributions.scope.isin(("month", "side"))
        & contributions.selected_rows.ge(int(spec.min_control_segment_rows))
    ].copy()
    month = selected.loc[selected.scope.eq("month"), "net_bps_per_trade"]
    side = selected.loc[selected.scope.eq("side"), "net_bps_per_trade"]
    monthly_values = pd.to_numeric(month, errors="coerce").dropna().to_numpy(float)
    side_values = pd.to_numeric(side, errors="coerce").dropna().to_numpy(float)
    side_share: list[float] = []
    for scope in _ADMISSION_SCOPES:
        rows = contributions.loc[
            contributions.scope.eq("side")
            & contributions.admission_scope.eq(scope)
            & np.isclose(contributions.top_fraction, spec.control_selection_top_fraction)
        ]
        if len(rows):
            total = float(rows.selected_rows.sum())
            if total > 0:
                side_share.append(float(rows.selected_rows.max() / total))
    has_segments = bool(len(monthly_values) and len(side_values))
    admission_ok = (not spec.require_21d_admission_for_control_selection) or (
        scope_complete and np.isfinite(top.loc[top.admission_scope.eq("with_21d_side_local_admission"), "net_bps_per_trade"].to_numpy(float)).all()
    )
    eligible = bool(admission_ok and has_segments and len(valid_aggregate))
    return {
        "record_type": "control_summary", "arm": arm,
        "candidate_rows": int(top.original_population_rows.max()) if len(top) else 0,
        "selection_tail_fraction": float(spec.control_selection_top_fraction),
        "selection_admission_scope_count": scope_count,
        "selection_admission_complete": bool(scope_complete),
        "selection_worst_month_net_bps_per_trade": float(monthly_values.min()) if len(monthly_values) else np.nan,
        "selection_worst_side_net_bps_per_trade": float(side_values.min()) if len(side_values) else np.nan,
        "selection_mean_top_tail_net_bps_per_trade": float(valid_aggregate.mean()) if len(valid_aggregate) else np.nan,
        "selection_max_side_share": float(max(side_share)) if side_share else np.nan,
        "selection_eligible": eligible,
    }


def run_stage_ii_meta_archetype_funnel(
    frame: pd.DataFrame,
    *,
    spec: StageIIFunnelSpec,
    candidates: Sequence[StageIIDiscoveryCandidate],
    meta_oof_predictor: MetaOOFPredictor,
) -> StageIIFunnelResult:
    """Run the prescribed Stage-II sequence and deterministically freeze one arm.

    Candidate discovery is evaluated first.  Only the best retained candidate
    may enter four same-row meta controls.  If no candidate clears discovery
    gates the result is an explicit ``NO_STAGE_II_ARCHETYPE_ADVANCES``
    diagnostic rather than a silently widened search or a base-model change.
    """
    work = _validate_spec(frame, spec)
    base_handoff_provenance = _validate_base_handoff(work, spec)
    validated = _validate_candidates(candidates, work, spec)
    candidate_rows: list[dict[str, Any]] = []
    stability_tables: list[pd.DataFrame] = []
    predictability_tables: list[pd.DataFrame] = []
    outputs: dict[str, StageIIMetaArchetypeOOFResult] = {}
    for candidate in validated:
        result = strict_oof_meta_archetype_features(work, config=candidate.config, causal_feature_cols=candidate.causal_feature_cols)
        stability, predictability, summary = _diagnose_candidate(candidate, result, work, spec)
        outputs[candidate.candidate_id] = result
        stability_tables.append(stability)
        predictability_tables.append(predictability)
        retained = (
            summary["oof_rows"] >= spec.min_oof_rows
            and np.isfinite(summary["economic_separation_bps"])
            and summary["economic_separation_bps"] >= spec.min_economic_separation_bps
            and summary["stable_fold_fraction"] >= spec.min_stable_fold_fraction
            and np.isfinite(summary["causal_log_loss_improvement"])
            and summary["causal_log_loss_improvement"] >= spec.min_causal_log_loss_improvement
            and np.isfinite(summary["causal_membership_brier"])
            and summary["causal_membership_brier"] <= spec.max_causal_membership_brier
            and np.isfinite(summary["max_symbol_share"])
            and summary["max_symbol_share"] <= spec.max_symbol_share
        )
        candidate_rows.append({"candidate_id": candidate.candidate_id, "components": candidate.config.components, "path_descriptor_cols": tuple(candidate.config.path_descriptor_cols), **summary, "disposition": "retained" if retained else "diagnostic", "selection_score": float(summary["economic_separation_bps"] + 25.0 * summary["causal_log_loss_improvement"] - 100.0 * max(0.0, summary["max_symbol_share"] - 0.50)) if np.isfinite(summary["economic_separation_bps"]) and np.isfinite(summary["causal_log_loss_improvement"]) and np.isfinite(summary["max_symbol_share"]) else -np.inf})
    candidate_audit = pd.DataFrame(candidate_rows).sort_values("candidate_id", kind="stable")
    retained = candidate_audit.loc[candidate_audit.disposition.eq("retained")].sort_values(["selection_score", "candidate_id"], ascending=[False, True], kind="stable")
    stability = pd.concat(stability_tables, ignore_index=True) if stability_tables else pd.DataFrame()
    predictability = pd.concat(predictability_tables, ignore_index=True) if predictability_tables else pd.DataFrame()
    if retained.empty:
        return StageIIFunnelResult(candidate_audit, stability, predictability, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None, {"schema": SCHEMA, "sequence": "discovery_then_meta_controls", "decision": "NO_STAGE_II_ARCHETYPE_ADVANCES", "hard_routing": False, "local_experts": False, "base_model_changed": False}, None)
    selected_id = str(retained.iloc[0].candidate_id)
    candidate_audit.loc[(candidate_audit.disposition.eq("retained")) & (~candidate_audit.candidate_id.eq(selected_id)), "disposition"] = "rejected"
    selected_candidate = next(value for value in validated if value.candidate_id == selected_id)
    selected_output = outputs[selected_id]
    soft, prior = _feature_control_columns(selected_output, selected_candidate)
    available = selected_output.features[f"{META_ARCHETYPE_PREFIX}available"].to_numpy(float) > 0.5
    # All downstream arms use exactly the same rows and target.  Burn-in/unknown
    # rows are excluded from every arm, never relabelled as an archetype.
    common = work.loc[available].copy().reset_index(drop=True)
    archetypes = selected_output.features.loc[available].reset_index(drop=True)
    if len(common) < spec.min_oof_rows:
        raise StageIIFunnelError("retained discovery candidate has insufficient shared OOF rows")
    # The direct same-side raw base output is always an explicit meta input;
    # it is never converted to a period-local rank or a mapped value first.
    base_meta = tuple(dict.fromkeys((
        spec.base_expected_net_column, *spec.base_r3_probability_columns, *spec.meta_feature_cols,
    )))
    control_additions = {"none": (), "soft_memberships": soft, "prior": prior, "both": (*soft, *prior)}
    metrics_tables: list[pd.DataFrame] = []
    contribution_tables: list[pd.DataFrame] = []
    admission_tables: list[pd.DataFrame] = []
    control_rows: list[dict[str, Any]] = []
    ids = common[spec.candidate_id_column].to_numpy(object)
    base = common[spec.base_expected_net_column].to_numpy(np.float32)
    target = common[spec.exact_net_column].to_numpy(np.float32) - base
    for arm in _CONTROL_ARMS:
        additions = tuple(control_additions[arm])
        model_frame = pd.concat([common.loc[:, list(base_meta)].reset_index(drop=True), archetypes.loc[:, list(additions)].reset_index(drop=True)], axis=1)
        request = StageIIMetaPredictionRequest(
            arm=arm, frame=model_frame, feature_columns=tuple(model_frame.columns),
            target_residual_bps=target, candidate_ids=ids, base_expected_net_bps=base,
            base_r3_probabilities=common.loc[:, list(spec.base_r3_probability_columns)].to_numpy(np.float32),
            base_r3_probability_columns=tuple(spec.base_r3_probability_columns),
            base_handoff_provenance=base_handoff_provenance,
            decision_timestamps=common[spec.decision_ts_column].to_numpy(),
            label_available_timestamps=common[spec.label_available_ts_column].to_numpy(),
        )
        residual = _validate_meta_prediction(meta_oof_predictor(request), request)
        ledger = common.copy()
        ledger["__stage_ii_reconstructed_common_bps"] = base + residual
        metric, contributions, admission = _pooled_global_metrics(ledger, arm=arm, score_column="__stage_ii_reconstructed_common_bps", spec=spec)
        metric.insert(0, "record_type", "pooled_tail_metric")
        metrics_tables.append(metric)
        contribution_tables.append(contributions)
        admission_tables.append(admission)
        summary = _control_selection_summary(metric, contributions, arm=arm, spec=spec)
        summary["feature_columns"] = tuple(model_frame.columns)
        control_rows.append(summary)
    controls = pd.DataFrame(control_rows)
    eligible_controls = controls.loc[controls.selection_eligible].sort_values(
        ["selection_worst_month_net_bps_per_trade", "selection_worst_side_net_bps_per_trade", "selection_mean_top_tail_net_bps_per_trade", "selection_max_side_share", "arm"],
        ascending=[False, False, False, True, True], kind="stable",
    )
    all_metrics = pd.concat([controls, pd.concat(metrics_tables, ignore_index=True)], ignore_index=True, sort=False)
    all_contributions = pd.concat(contribution_tables, ignore_index=True)
    all_admission = pd.concat(admission_tables, ignore_index=True)
    if eligible_controls.empty:
        manifest = {"schema": SCHEMA, "sequence": "bounded_discovery->strict_oof_causal_recogniser->gates->matched_meta_only_controls->causal_21d_admission->pooled_global_ranking", "decision": "NO_STAGE_II_META_CONTROL_ADVANCES", "selected_candidate_id": selected_id, "selected_control_arm": None, "candidate_count": len(validated), "control_arms": list(_CONTROL_ARMS), "top_fractions": list(spec.top_fractions), "hard_routing": False, "local_experts": False, "base_model_changed": False, "base_handoff": {"expected_net_bps_column": spec.base_expected_net_column, **base_handoff_provenance}, "control_selection": "lexicographic worst selected month, worst selected side, mean top-tail net, lower side concentration", "dispositions": candidate_audit.set_index("candidate_id").disposition.to_dict()}
        return StageIIFunnelResult(candidate_audit, stability, predictability, all_metrics, all_contributions, all_admission, selected_id, None, manifest, selected_output.features.copy())
    winning_arm = str(eligible_controls.iloc[0].arm)
    manifest = {"schema": SCHEMA, "sequence": "bounded_discovery->strict_oof_causal_recogniser->gates->matched_meta_only_controls->causal_21d_admission->pooled_global_ranking", "selected_candidate_id": selected_id, "selected_control_arm": winning_arm, "candidate_count": len(validated), "control_arms": list(_CONTROL_ARMS), "top_fractions": list(spec.top_fractions), "hard_routing": False, "local_experts": False, "base_model_changed": False, "base_handoff": {"expected_net_bps_column": spec.base_expected_net_column, **base_handoff_provenance}, "reconstruction": "same_side_frozen_base_expected_net_bps + predicted_meta_residual_bps", "ranking": "one pooled global common-bps ordering after optional side-local causal 21d admission", "control_selection": "lexicographic worst selected month, worst selected side, mean top-tail net, lower side concentration", "dispositions": candidate_audit.set_index("candidate_id").disposition.to_dict()}
    return StageIIFunnelResult(candidate_audit, stability, predictability, all_metrics, all_contributions, all_admission, selected_id, winning_arm, manifest, selected_output.features.copy())


__all__ = [
    "SCHEMA", "MetaOOFPredictor", "StageIIDiscoveryCandidate", "StageIIFunnelError",
    "StageIIFunnelResult", "StageIIFunnelSpec", "StageIIMetaPredictionRequest",
    "StageIIMetaPredictionResult", "run_stage_ii_meta_archetype_funnel",
]
