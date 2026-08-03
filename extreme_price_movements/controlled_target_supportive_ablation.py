"""Contracts for the controlled target/supportive-label ablation.

This is deliberately a research contract, not a promotion path.  It compares
five economic target designs under one candidate population and one frozen
model capacity.  Auxiliary/path labels may enter a target model *only* as
strict chronological out-of-fold predictions.  Raw future labels are never
features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.feature_provenance_gate import FeatureProvenanceError, validate_feature_columns


SCHEMA = "controlled_target_supportive_ablation_v1"
TOP_K_FRACTION = 0.10
DEFAULT_HURDLE_BPS = 25.0
TARGET_ARMS = ("T0_native24_control", "T1_clean_opportunity", "T2_direct_net", "T3_competing_risk_expected_net", "T4_hurdle_decomposition")
SUPPORT_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
SUPPORT_LABELS = (
    ("S1", "clean_opportunity", "__opportunity_occurred_12h__", "binary"),
    ("S2", "peak_mfe_atr", "__peak_mfe_atr_12h__", "regression"),
    ("S3", "time_to_meaningful_mfe_hours", "__time_to_first_meaningful_mfe_hours_12h__", "regression"),
    ("S4", "mae_before_meaningful_mfe_atr", "__mae_before_meaningful_mfe_atr_12h__", "regression"),
    ("S5", "future_slope_atr_per_hour", "__future_slope_atr_per_hour_12h__", "regression"),
)
# The attached roadmap groups supporting outcomes by the decision question they
# answer.  These labels are materialised as finite, normalized composites in
# the prepared ledger (their source fields and formula are retained in the
# ledger manifest).  Keeping them as a separate spec preserves the legacy
# five-head runner for reproducibility while allowing the grouped pipeline to
# be run through the same strict-OOF machinery.
GROUPED_SUPPORT_LABELS = (
    ("S1", "opportunity_reach_time", "__group_s1_opportunity_reach_time__", "regression"),
    ("S2", "adverse_path_risk", "__group_s2_adverse_path_risk__", "regression"),
    ("S3", "magnitude_net_margin", "__group_s3_magnitude_net_margin__", "regression"),
    ("S4", "persistence_giveback", "__group_s4_persistence_giveback__", "regression"),
    ("S5", "early_adverse_recovery", "__group_s5_early_adverse_recovery__", "regression"),
)
_FORBIDDEN_RAW_FEATURES = frozenset(
    {
        "__first_touch_target_soft__",
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "favorable_first",
        "adverse_first",
        "timeout",
        *[label for _, _, label, _ in SUPPORT_LABELS],
        *[label for _, _, label, _ in GROUPED_SUPPORT_LABELS],
    }
)


class ContractError(ValueError):
    """Raised when a research run would compare non-identical evidence."""


@dataclass(frozen=True)
class AcceptanceGates:
    """Predeclared diagnostics; passing them does not authorize promotion."""

    top_k_fraction: float = TOP_K_FRACTION
    minimum_selected_rows: int = 50
    require_positive_global_net: bool = True
    require_positive_latest_month_net: bool = True
    require_complete_month_coverage: bool = True

    def manifest(self) -> dict[str, object]:
        return asdict(self)


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        raise ContractError(f"missing required target column: {name}")
    result = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ContractError(f"non-finite values in target column: {name}")
    return result


def hurdle_return(hurdle_bps: float) -> float:
    """Convert the declared hurdle from bps to the ledger's return unit."""
    value = float(hurdle_bps)
    if not np.isfinite(value) or value < 0.0:
        raise ContractError("hurdle_bps must be finite and non-negative")
    return value / 10_000.0


def hurdle_decomposition_score(
    clear_probability: np.ndarray | Sequence[float],
    clear_conditional_excess: np.ndarray | Sequence[float],
    fail_probability: np.ndarray | Sequence[float],
    fail_conditional_shortfall: np.ndarray | Sequence[float],
    *,
    hurdle_bps: float = DEFAULT_HURDLE_BPS,
) -> np.ndarray:
    """Return the exact two-state hurdle expected-net decomposition.

    The output uses the same fractional-return unit as ``execution_net_ev_12h``:

    ``h + P(clear) E[max(net-h, 0) | clear]
       - P(fail) E[max(h-net, 0) | fail]``.

    Callers normally set ``P(fail) = 1 - P(clear)`` because clear/fail are an
    exhaustive two-state event.  Accepting both vectors keeps the composed
    score auditable while making a non-simplex caller fail closed.
    """
    h = hurdle_return(hurdle_bps)
    clear_p = np.asarray(clear_probability, dtype=float)
    clear_excess = np.asarray(clear_conditional_excess, dtype=float)
    fail_p = np.asarray(fail_probability, dtype=float)
    fail_shortfall = np.asarray(fail_conditional_shortfall, dtype=float)
    if not (clear_p.shape == clear_excess.shape == fail_p.shape == fail_shortfall.shape):
        raise ContractError("hurdle decomposition components must have identical shapes")
    if not all(np.isfinite(values).all() for values in (clear_p, clear_excess, fail_p, fail_shortfall)):
        raise ContractError("hurdle decomposition components must be finite")
    if ((clear_p < 0.0) | (clear_p > 1.0) | (fail_p < 0.0) | (fail_p > 1.0)).any():
        raise ContractError("hurdle decomposition probabilities must be in [0, 1]")
    if not np.allclose(clear_p + fail_p, 1.0, rtol=0.0, atol=1e-6):
        raise ContractError("clear/fail probabilities must form an exhaustive two-state simplex")
    if (clear_excess < 0.0).any() or (fail_shortfall < 0.0).any():
        raise ContractError("hurdle conditional excess and shortfall must be non-negative")
    return h + clear_p * clear_excess - fail_p * fail_shortfall


def validate_causal_raw_features(feature_columns: Iterable[str]) -> tuple[str, ...]:
    """Reject targets/path outcomes accidentally presented as model inputs."""
    try:
        columns = validate_feature_columns(feature_columns)
    except FeatureProvenanceError as error:
        # Preserve this module's public contract while retaining the stricter
        # universal provenance diagnostic in the chained exception.
        raise ContractError(f"non-causal raw feature contract: {error}") from error
    duplicate = len(columns) != len(set(columns))
    forbidden = sorted(set(columns) & _FORBIDDEN_RAW_FEATURES)
    if duplicate or forbidden:
        problem = ([] if not duplicate else ["duplicate feature names"]) + forbidden
        raise ContractError(f"non-causal raw feature contract: {problem}")
    return columns


def derive_economic_targets(frame: pd.DataFrame, *, hurdle_bps: float = DEFAULT_HURDLE_BPS) -> pd.DataFrame:
    """Materialize target labels, preserving exact row-level cost accounting.

    ``T3`` is represented by its mutually-exclusive class plus the exact net
    outcome.  ``T4`` is the exact two-state net-hurdle identity: an exhaustive
    clear/fail classifier and non-negative conditional excess/shortfall heads.
    The hurdle is in bps at the public API, then converted once to the ledger's
    fractional-return unit.
    """

    work = frame.copy()
    native = _numeric(work, "__first_touch_target_soft__")
    net = _numeric(work, "execution_net_ev_12h")
    gross = _numeric(work, "execution_gross_ev_12h")
    cost = _numeric(work, "execution_cost_return")
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise ContractError("execution gross - exact cost does not equal net")
    clean = _numeric(work, "__opportunity_occurred_12h__")
    favorable = _numeric(work, "favorable_first") > 0.5
    adverse = _numeric(work, "adverse_first") > 0.5
    timeout = _numeric(work, "timeout") > 0.5
    if not np.array_equal(favorable.astype(int) + adverse.astype(int) + timeout.astype(int), np.ones(len(work), dtype=int)):
        raise ContractError("competing-risk event labels must be mutually exclusive and exhaustive")
    if ((clean < 0.0) | (clean > 1.0)).any():
        raise ContractError("clean opportunity must be a probability/binary label in [0, 1]")
    work["target_t0_native24"] = np.clip(native, 0.0, 1.0).astype(np.float32)
    work["target_t1_clean_opportunity"] = clean.astype(np.float32)
    work["target_t2_direct_net"] = net.astype(np.float32)
    work["target_t3_competing_class"] = np.select([timeout, adverse, favorable], [0, 1, 2]).astype(np.int8)
    work["target_t3_expected_net"] = net.astype(np.float32)
    h = hurdle_return(hurdle_bps)
    clear = net > h
    work["target_t4_hurdle_bps"] = np.float32(hurdle_bps)
    work["target_t4_clear"] = clear.astype(np.int8)
    work["target_t4_fail"] = (~clear).astype(np.int8)
    work["target_t4_clear_excess_return"] = np.maximum(net - h, 0.0).astype(np.float32)
    work["target_t4_fail_shortfall_return"] = np.maximum(h - net, 0.0).astype(np.float32)
    return work


def support_columns(stage: str, support_labels: Sequence[tuple[str, str, str, str]] = SUPPORT_LABELS) -> tuple[str, ...]:
    if stage not in SUPPORT_STAGES:
        raise ContractError(f"unknown supportive stage: {stage}")
    stage_order = SUPPORT_STAGES.index(stage)
    return tuple(
        f"support_oof__{name}"
        for group, name, _, _ in support_labels
        if group in SUPPORT_STAGES and SUPPORT_STAGES.index(group) <= stage_order and group != "S0"
    )


def strict_oof_support_predictions(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    fold_column: str,
    timestamp_column: str = "__ts__",
    available_at_column: str = "__label_available_at__",
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, str], np.ndarray],
    support_labels: Sequence[tuple[str, str, str, str]] = SUPPORT_LABELS,
) -> pd.DataFrame:
    """Generate supportive features using only earlier resolved folds.

    ``predictor`` is injected so the contract can be tested without a heavy ML
    dependency.  It receives train X/y, test X and label kind, and must return
    one prediction per test row.  The first chronological fold is necessarily
    unsupported and remains null; callers must exclude it for *every* target
    arm, including S0, to preserve a matched-row comparison.
    """

    feature_columns = validate_causal_raw_features(feature_columns)
    required = {fold_column, timestamp_column, available_at_column, *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"support OOF input lacks columns: {sorted(missing)}")
    work = frame.copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], utc=True, errors="raise")
    work[available_at_column] = pd.to_datetime(work[available_at_column], utc=True, errors="raise")
    if work[fold_column].isna().any():
        raise ContractError("support OOF fold IDs must be complete")
    fold_order = (
        work.groupby(fold_column, observed=True)[timestamp_column].min().sort_values(kind="mergesort").index.tolist()
    )
    # LightGBM's native missing-value handling is part of the frozen raw
    # feature contract.  Requiring a globally finite matrix here would either
    # leak a full-population imputation statistic or silently discard the
    # causal missingness pattern.  We only reject rows with no usable causal
    # input at all; all-NaN *columns* are rejected by the run preparation
    # layer, where their exclusion is recorded in the feature manifest.
    matrix = work.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if (~np.isfinite(matrix).any(axis=1)).any():
        raise ContractError("support OOF raw features contain a row with no finite causal inputs")
    for _, support_name, label, kind in support_labels:
        values = _numeric(work, label)
        output = np.full(len(work), np.nan, dtype=np.float64)
        for position, fold in enumerate(fold_order):
            test_mask = work[fold_column].eq(fold).to_numpy()
            if position == 0:
                continue
            test_start = work.loc[test_mask, timestamp_column].min()
            earlier_folds = set(fold_order[:position])
            train_mask = work[fold_column].isin(earlier_folds).to_numpy()
            train_mask &= work[available_at_column].lt(test_start).to_numpy()
            if not train_mask.any():
                raise ContractError(f"no strictly resolved training rows before support fold {fold!r}")
            # The explicit max check makes accidental chronology regressions loud.
            if work.loc[train_mask, available_at_column].max() >= test_start:
                raise ContractError("support OOF chronology violation")
            prediction = np.asarray(predictor(matrix[train_mask], values[train_mask], matrix[test_mask], kind), dtype=float)
            if prediction.shape != (int(test_mask.sum()),) or not np.isfinite(prediction).all():
                raise ContractError(f"invalid support prediction for {support_name}/{fold!r}")
            output[test_mask] = prediction
        work[f"support_oof__{support_name}"] = output.astype(np.float32)
    return work


def matched_support_population(
    frame: pd.DataFrame,
    support_labels: Sequence[tuple[str, str, str, str]] = SUPPORT_LABELS,
) -> pd.DataFrame:
    """Use the same fully-supported OOF rows for every T/S configuration."""
    support = [f"support_oof__{name}" for _, name, _, _ in support_labels]
    missing = set(support) - set(frame.columns)
    if missing:
        raise ContractError(f"missing supportive OOF predictions: {sorted(missing)}")
    usable = np.isfinite(frame.loc[:, support].to_numpy(dtype=float)).all(axis=1)
    result = frame.loc[usable].copy()
    if result.empty:
        raise ContractError("no rows have the complete cumulative supportive OOF vector")
    return result


def stable_pooled_global_top_k(frame: pd.DataFrame, score_column: str, fraction: float = TOP_K_FRACTION) -> pd.DataFrame:
    """One global book—no timestamp, side, asset or replacement quota."""
    if not 0.0 < fraction <= 1.0:
        raise ContractError("top-k fraction must be in (0, 1]")
    if "candidate_id" not in frame:
        raise ContractError("candidate_id is required for deterministic global top-k ties")
    score = pd.to_numeric(frame[score_column], errors="coerce")
    if score.isna().any():
        raise ContractError("candidate policy cannot rank non-finite scores")
    count = max(1, int(np.ceil(len(frame) * fraction)))
    return frame.assign(__score__=score).sort_values(
        ["__score__", "candidate_id"], ascending=[False, True], kind="mergesort"
    ).head(count).drop(columns="__score__")


def pooled_global_top_k_metrics(frame: pd.DataFrame, score_column: str, *, gates: AcceptanceGates = AcceptanceGates(), timestamp_column: str = "__ts__") -> dict[str, object]:
    """Exact, post-score global candidate policy plus non-promotional gates."""
    if timestamp_column not in frame:
        raise ContractError(f"missing timestamp column: {timestamp_column}")
    selected = stable_pooled_global_top_k(frame, score_column, gates.top_k_fraction)
    net = _numeric(selected, "execution_net_ev_12h") * 10_000.0
    timestamps = pd.to_datetime(selected[timestamp_column], utc=True, errors="raise")
    all_months = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise").dt.to_period("M")
    selected_months = timestamps.dt.to_period("M")
    month_net = pd.DataFrame({"month": selected_months, "net_bps": net}).groupby("month", observed=True).net_bps.mean()
    expected_months = set(all_months.unique())
    observed_months = set(selected_months.unique())
    latest = max(expected_months)
    latest_value = float(month_net.get(latest, np.nan))
    complete = expected_months.issubset(observed_months)
    checks = {
        "selected_rows_sufficient": len(selected) >= gates.minimum_selected_rows,
        "complete_month_coverage": (not gates.require_complete_month_coverage) or complete,
        "positive_global_net": (not gates.require_positive_global_net) or float(net.mean()) > 0.0,
        "positive_latest_month_net": (not gates.require_positive_latest_month_net) or latest_value > 0.0,
    }
    cost = _numeric(selected, "execution_cost_return") * 10_000.0
    positive = net > 0.0
    negative = net < 0.0
    gross_positive = _numeric(selected, "execution_gross_ev_12h") > 0.0
    return {
        "selection_basis": "pooled_global_post_score_top_k",
        "top_k_fraction": gates.top_k_fraction,
        "population_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "global_topk_net_bps": float(net.mean()),
        "global_topk_gross_bps": float(_numeric(selected, "execution_gross_ev_12h").mean() * 10_000.0),
        "global_topk_cost_bps": float(cost.mean()),
        "global_topk_win_rate": float(positive.mean()),
        "global_topk_gross_positive_rate": float(gross_positive.mean()),
        "global_topk_profit_factor": float(
            net[positive].sum() / abs(net[negative].sum())
            if negative.any() and abs(net[negative].sum()) > 0.0
            else np.inf if positive.any() else 0.0
        ),
        "worst_month_topk_net_bps": float(month_net.min()),
        "latest_month": str(latest),
        "latest_month_topk_net_bps": latest_value,
        "months_expected": int(len(expected_months)),
        "months_selected": int(len(observed_months)),
        "acceptance_checks": checks,
        "acceptance_passed": bool(all(checks.values())),
    }


def aligned_run_contract(
    frame: pd.DataFrame,
    *,
    fold_column: str,
    feature_columns: Iterable[str],
    hurdle_bps: float = DEFAULT_HURDLE_BPS,
    support_labels: Sequence[tuple[str, str, str, str]] = SUPPORT_LABELS,
) -> Mapping[str, object]:
    """Serializable invariant evidence for each controlled experiment."""
    return {
        "schema": SCHEMA,
        "target_arms": list(TARGET_ARMS),
        "support_stages": list(SUPPORT_STAGES),
        "supportive_labels": [
            {"stage": stage, "name": name, "label": label, "kind": kind}
            for stage, name, label, kind in support_labels
        ],
        "rows": int(len(frame)),
        "fold_counts": {str(key): int(value) for key, value in frame[fold_column].value_counts(sort=False).sort_index().items()},
        "feature_columns": list(feature_columns),
        "raw_future_labels_as_features": False,
        "supportive_features": "strict chronological OOF predictions only",
        "t4_hurdle_decomposition": {
            "hurdle_bps": float(hurdle_bps),
            "formula": "h + P(clear)*E[max(net-h,0)|clear] - P(fail)*E[max(h-net,0)|fail]",
            "clear_fail_contract": "net > h versus net <= h; exhaustive two-state simplex",
            "unit": "fractional return internally; hurdle supplied in bps",
        },
        "candidate_policy": "one pooled global post-score top-k; candidate-id tie break; no side/timestamp/asset quota",
    }
