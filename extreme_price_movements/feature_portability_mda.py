"""Causal Stage-A/F4 evidence materialisation for the F3 portability arm.

This module deliberately sits *before* ``feature_portability_selection``.  It
uses an explicitly supplied frozen, side-local R3 classifier only on labels
resolved before a fold's evaluation boundary, maps its probability simplex
through a train-only class-to-common-net-bps map, and ranks the two sides once
together.  It is not an HPO facility and it never evaluates the final November
holdout.

The public dataframe API accepts an already-materialised candidate panel so a
future genuine F4 run does not need to regenerate features.  Its ``evidence``
output has precisely the columns consumed by ``feature_portability_selection``;
the supporting coverage/fold tables are intentionally separate artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_portability_selection import REQUIRED_EVIDENCE_COLUMNS
from .feature_portability_f4_compact import (
    DEFAULT_F4_GROUP_COUNTS,
    F4_TRANSFORM_GROUPS,
    compact_contract_payload,
    compact_contracts_for_ranked_groups,
    f4_transform_groups,
    restrict_f4_transform_groups_to_sources,
    validate_group_counts,
)


SCHEMA = "stage_a_f4_feature_portability_mda_v1"
F3_TRANSFORM_RE = re.compile(r"__causal_(?:rank_w(?:90|180)|robust_z_w(?:90|180)|delta_p(?:4|24))$")
F3_REQUIRED_SUFFIXES = (
    "__causal_rank_w90",
    "__causal_rank_w180",
    "__causal_robust_z_w90",
    "__causal_robust_z_w180",
    "__causal_delta_p4",
    "__causal_delta_p24",
)
IDENTITY_COLUMNS = frozenset({"candidate_id", "decision_ts", "label_available_ts", "side_name"})
OUTCOME_OR_CONTROL_RE = re.compile(
    r"(?:^|_)(?:label|target|outcome|realized|realised|future|mfe|mae|pnl|"
    r"first_touch|exit|take_profit|stop_loss|path|event|gross|net|r3_class)(?:_|$)",
    flags=re.IGNORECASE,
)


class FeaturePortabilityMDAError(ValueError):
    """Raised when the materialiser cannot prove its causal F4 contract."""


@dataclass(frozen=True)
class ChronologicalTransport:
    """One development-only, contiguous chronological evaluation transport."""

    name: str
    train_start: object
    evaluation_start: object
    evaluation_end: object

    def __post_init__(self) -> None:
        start = _utc(self.train_start, "train_start")
        evaluation_start = _utc(self.evaluation_start, "evaluation_start")
        evaluation_end = _utc(self.evaluation_end, "evaluation_end")
        if not self.name:
            raise FeaturePortabilityMDAError("transport name is required")
        if not start < evaluation_start < evaluation_end:
            raise FeaturePortabilityMDAError("transport must satisfy train_start < evaluation_start < evaluation_end")
        # November 2024 is reserved for the single terminal replay.  Rejecting
        # by window, rather than a caller-provided label, prevents relabelling a
        # final OOS run as development evidence.
        if evaluation_end > pd.Timestamp("2024-11-01", tz="UTC") or re.search(r"(?:final|oos|november)", self.name, re.I):
            raise FeaturePortabilityMDAError("F4 evidence must never consume final November OOS")
        object.__setattr__(self, "train_start", start)
        object.__setattr__(self, "evaluation_start", evaluation_start)
        object.__setattr__(self, "evaluation_end", evaluation_end)


@dataclass(frozen=True)
class R3CostContract:
    """Required outcome/weight contract for the frozen actual R3 base path.

    ``sample_weight_column`` is preferred for a pre-materialised panel.  When
    it is absent, the exact frozen R3 agreement/class-balance formula is
    reconstructed from the three robust-clear label columns.
    """

    class_column: str
    gross_bps_column: str
    net_bps_column: str
    expected_cost_bps: float
    sample_weight_column: str | None = None
    robust_clear_columns: tuple[str, str, str] = (
        "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50",
    )

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (
            self.class_column, self.gross_bps_column, self.net_bps_column,
        )):
            raise FeaturePortabilityMDAError("R3/cost column names must be non-empty")
        if not np.isfinite(float(self.expected_cost_bps)) or float(self.expected_cost_bps) < 0.0:
            raise FeaturePortabilityMDAError("R3 expected_cost_bps must be finite and non-negative")
        if self.sample_weight_column is None and len(self.robust_clear_columns) != 3:
            raise FeaturePortabilityMDAError("R3 needs a sample-weight field or its three robust-clear columns")


@dataclass(frozen=True)
class FrozenR3ModelContract:
    """Identity and immutable parameters of the externally frozen R3 fit."""

    model_id: str
    params: Mapping[str, Any]
    random_seed: int = 17
    model_hpo_performed: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not isinstance(self.params, Mapping) or not self.params:
            raise FeaturePortabilityMDAError("a frozen R3 model id and fixed parameters are required")
        if bool(self.model_hpo_performed):
            raise FeaturePortabilityMDAError("F4 cannot consume a model-HPO R3 contract")


# The callback owns only the frozen actual classifier implementation.  It must
# receive labels/weights supplied here and return P(adverse), P(weak),
# P(clear) in this exact order; it never receives net-bps as a fit target.
R3FitPredictCallback = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class F4EvidenceMaterialization:
    """Selector-ready evidence plus independently auditable support tables."""

    evidence: pd.DataFrame
    transformed_coverage: pd.DataFrame
    source_intersection_coverage: pd.DataFrame
    representation_coverage: pd.DataFrame
    fold_mda: pd.DataFrame
    feature_group_mda: pd.DataFrame
    compact_contracts: Mapping[str, Any]
    transport_audit: pd.DataFrame
    manifest: Mapping[str, Any]


def _utc(value: object, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    if pd.isna(result):
        raise FeaturePortabilityMDAError(f"{name} must be a finite UTC timestamp")
    return result


def _normalise_panel(panel: pd.DataFrame, *, r3_cost: R3CostContract) -> pd.DataFrame:
    required = [
        *IDENTITY_COLUMNS, r3_cost.class_column, r3_cost.gross_bps_column,
        r3_cost.net_bps_column,
    ]
    if r3_cost.sample_weight_column is not None:
        required.append(r3_cost.sample_weight_column)
    else:
        required.extend(r3_cost.robust_clear_columns)
    missing = sorted(set(required).difference(panel.columns))
    if missing:
        raise FeaturePortabilityMDAError(f"candidate panel lacks required columns: {missing}")
    work = panel.copy()
    if work.empty or work["candidate_id"].isna().any() or work["candidate_id"].duplicated().any():
        raise FeaturePortabilityMDAError("candidate_id must be non-empty, non-null, and globally unique")
    for name in ("decision_ts", "label_available_ts"):
        work[name] = pd.to_datetime(work[name], utc=True, errors="coerce")
    if work[["decision_ts", "label_available_ts"]].isna().any().any():
        raise FeaturePortabilityMDAError("decision_ts and label_available_ts must be finite UTC timestamps")
    if (work["label_available_ts"] < work["decision_ts"]).any():
        raise FeaturePortabilityMDAError("labels cannot be available before their decision timestamp")
    work["side_name"] = work["side_name"].astype(str)
    if set(work["side_name"]).difference({"long", "short"}):
        raise FeaturePortabilityMDAError("side_name must contain canonical long/short values")
    classes = pd.to_numeric(work[r3_cost.class_column], errors="coerce")
    if classes.isna().any() or not np.array_equal(classes.to_numpy(float), classes.to_numpy(np.int8).astype(float)):
        raise FeaturePortabilityMDAError("R3 class labels must be finite integers")
    if set(classes.astype(int)).difference({0, 1, 2}):
        raise FeaturePortabilityMDAError("R3 class labels must be adverse=0, weak=1, clear=2")
    work[r3_cost.class_column] = classes.astype(np.int8)
    for column in (r3_cost.gross_bps_column, r3_cost.net_bps_column):
        values = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise FeaturePortabilityMDAError(f"{column} must be finite R3 economic evidence")
        work[column] = values.astype(float)
    observed_cost = work[r3_cost.gross_bps_column].to_numpy(float) - work[r3_cost.net_bps_column].to_numpy(float)
    if not np.allclose(observed_cost, float(r3_cost.expected_cost_bps), atol=0.02, rtol=0.0):
        raise FeaturePortabilityMDAError("gross_bps - net_bps must equal the declared R3 cost exactly once")
    if r3_cost.sample_weight_column is not None:
        weights = pd.to_numeric(work[r3_cost.sample_weight_column], errors="coerce")
        if not np.isfinite(weights.to_numpy(float)).all() or (weights <= 0.0).any():
            raise FeaturePortabilityMDAError("materialised R3 sample weights must be finite and positive")
        work[r3_cost.sample_weight_column] = weights.astype(float)
    return work.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)


def _normalise_contracts(
    representation_features: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    if len(representation_features) < 2:
        raise FeaturePortabilityMDAError("F4 needs a candidate representation and an unpermuted control")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for representation, by_side in representation_features.items():
        if not str(representation) or not isinstance(by_side, Mapping):
            raise FeaturePortabilityMDAError("representation contracts must map a non-empty name to side fields")
        result[str(representation)] = {}
        for side in ("long", "short"):
            fields = tuple(dict.fromkeys(map(str, by_side.get(side, ()))))
            if not fields:
                raise FeaturePortabilityMDAError(f"{representation}/{side} requires a non-empty feature contract")
            forbidden = [field for field in fields if field in IDENTITY_COLUMNS or OUTCOME_OR_CONTROL_RE.search(field)]
            if forbidden:
                raise FeaturePortabilityMDAError(f"{representation}/{side} contains raw outcome/control inputs: {forbidden}")
            result[str(representation)][side] = fields
    return result


def f3_transformed_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Return actual F3 rank, robust-z, and delta fields, never sources."""
    return tuple(field for field in dict.fromkeys(map(str, fields)) if F3_TRANSFORM_RE.search(field))


def _validate_f3_contract(contract: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    transformed: dict[str, tuple[str, ...]] = {}
    for side in ("long", "short"):
        fields = f3_transformed_fields(contract[side])
        absent = [suffix for suffix in F3_REQUIRED_SUFFIXES if not any(field.endswith(suffix) for field in fields)]
        if absent:
            raise FeaturePortabilityMDAError(
                "F3 "
                f"{side} contract lacks actual rank90/rank180/robust-z90/robust-z180/delta4/delta24 fields: {absent}"
            )
        transformed[side] = fields
    return transformed


def audit_f3_transformed_coverage(
    panel: pd.DataFrame,
    *,
    transport: ChronologicalTransport,
    f3_features: Mapping[str, Sequence[str]],
    min_coverage: float = 0.99,
    require_all: bool = True,
) -> pd.DataFrame:
    """Audit generated F3 fields for each side and outer evaluation transport.

    The audit deliberately takes the transformed feature names.  It cannot be
    satisfied by coverage of their source columns.
    """
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise FeaturePortabilityMDAError("min_coverage must be in [0, 1]")
    rows: list[dict[str, Any]] = []
    evaluation = panel.loc[
        panel["decision_ts"].ge(transport.evaluation_start) & panel["decision_ts"].lt(transport.evaluation_end)
    ]
    for side in ("long", "short"):
        side_rows = evaluation.loc[evaluation["side_name"].eq(side)]
        if side_rows.empty:
            raise FeaturePortabilityMDAError(f"{transport.name}/{side} has no evaluation candidates")
        fields = tuple(map(str, f3_features[side]))
        missing = sorted(set(fields).difference(panel.columns))
        if missing:
            raise FeaturePortabilityMDAError(f"F3 transformed fields absent from panel: {missing}")
        for feature in fields:
            values = pd.to_numeric(side_rows[feature], errors="coerce").to_numpy(float)
            coverage = float(np.isfinite(values).mean())
            rows.append({
                "transport": transport.name,
                "side_name": side,
                "feature": feature,
                "is_actual_f3_transform": True,
                "evaluation_rows": int(len(side_rows)),
                "finite_rows": int(np.isfinite(values).sum()),
                "coverage": coverage,
                "passes_99pct_coverage": bool(coverage >= float(min_coverage)),
            })
    output = pd.DataFrame(rows)
    if require_all and not output["passes_99pct_coverage"].all():
        bad = output.loc[~output["passes_99pct_coverage"], "feature"].tolist()
        raise FeaturePortabilityMDAError(f"actual F3 transformed coverage below contract: {bad}")
    return output


def _f4_source_bundle_coverage(
    evaluation: pd.DataFrame,
    *,
    transport: ChronologicalTransport,
    transform_groups: Mapping[str, Mapping[str, Sequence[str]]],
    min_coverage: float,
) -> pd.DataFrame:
    """Audit every raw F3 source together with its six generated variants.

    This is the F4 eligibility unit.  A source is usable only when its whole
    causal bundle is finite on at least the declared coverage fraction; later
    code takes the intersection over *both* development transports per side.
    """
    rows: list[dict[str, Any]] = []
    group_names = [name for name, _ in F4_TRANSFORM_GROUPS]
    for side in ("long", "short"):
        side_rows = evaluation.loc[evaluation["side_name"].eq(side)]
        if side_rows.empty:
            raise FeaturePortabilityMDAError(f"{transport.name}/{side} has no evaluation candidates")
        sources = tuple(map(str, transform_groups[side]["portable_raw"]))
        for index, source in enumerate(sources):
            fields = [source, *(str(transform_groups[side][name][index]) for name in group_names)]
            missing = sorted(set(fields).difference(evaluation.columns))
            if missing:
                raise FeaturePortabilityMDAError(f"F4 source bundle fields are absent from panel: {missing}")
            coverage = [
                float(np.isfinite(pd.to_numeric(side_rows[field], errors="coerce").to_numpy(float)).mean())
                for field in fields
            ]
            rows.append({
                "transport": transport.name,
                "side_name": side,
                "source_field": source,
                "source_bundle_field_count": int(len(fields)),
                "source_bundle_min_coverage": float(min(coverage)),
                "source_raw_coverage": float(coverage[0]),
                "source_transform_min_coverage": float(min(coverage[1:])),
                "source_bundle_passes_99pct_coverage": bool(min(coverage) >= min_coverage),
                "evaluation_rows": int(len(side_rows)),
            })
    return pd.DataFrame(rows)


def _cross_transport_coverage_safe_sources(
    source_coverage: pd.DataFrame,
    *,
    transports: Sequence[ChronologicalTransport],
    min_coverage: float,
) -> tuple[dict[str, tuple[str, ...]], pd.DataFrame]:
    """Return the exact per-side source intersection safe in all transports."""
    expected = {transport.name for transport in transports}
    rows: list[pd.DataFrame] = []
    selected: dict[str, tuple[str, ...]] = {}
    for side in ("long", "short"):
        part = source_coverage.loc[source_coverage["side_name"].eq(side)].copy()
        counts = part.groupby("source_field", observed=True)["transport"].nunique()
        minima = part.groupby("source_field", observed=True)["source_bundle_min_coverage"].min()
        pass_all = (counts.reindex(minima.index, fill_value=0).eq(len(expected)) & minima.ge(min_coverage))
        annotated = part.merge(
            pd.DataFrame({
                "source_field": minima.index.astype(str),
                "cross_transport_min_source_bundle_coverage": minima.to_numpy(float),
                "transports_observed": counts.reindex(minima.index, fill_value=0).to_numpy(int),
                "selected_cross_transport_source_intersection": pass_all.to_numpy(bool),
            }),
            on="source_field", validate="many_to_one",
        )
        rows.append(annotated)
        source_order = list(dict.fromkeys(part["source_field"].astype(str)))
        selected[side] = tuple(source for source in source_order if bool(pass_all.get(source, False)))
        if not selected[side]:
            raise FeaturePortabilityMDAError(
                f"F4 has no {side} source whose raw-plus-transform bundle reaches {min_coverage:.0%} coverage in both transports"
            )
    return selected, pd.concat(rows, ignore_index=True)


def _matrix(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise FeaturePortabilityMDAError(f"feature contract lacks panel columns: {missing}")
    return frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _complete(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    return np.isfinite(_matrix(frame, fields)).all(axis=1)


def _r3_sample_weights(frame: pd.DataFrame, *, r3_cost: R3CostContract) -> np.ndarray:
    if r3_cost.sample_weight_column is not None:
        return frame[r3_cost.sample_weight_column].to_numpy(dtype=float, copy=True)
    labels = frame[r3_cost.class_column].to_numpy(np.int8)
    agreement = frame.loc[:, list(r3_cost.robust_clear_columns)].nunique(axis=1).eq(1).to_numpy(float)
    certainty = 0.5 + 0.5 * agreement
    counts = np.bincount(labels, minlength=3).astype(float)
    class_weight = np.sqrt(len(frame) / np.maximum(counts, 1.0))[labels]
    class_weight /= max(class_weight.mean(), 1e-12)
    weight = np.clip(certainty * class_weight, 0.25, 4.0)
    return weight / max(weight.mean(), 1e-12)


def _class_to_common_net_bps_map(train: pd.DataFrame, *, r3_cost: R3CostContract) -> np.ndarray:
    """Resolved-training R3 class payoff map in the shared net-bps unit.

    Net-bps are used only here as realised economic evidence, never as a
    classifier target.  The preceding cost reconciliation proves that the map
    is compatible with the frozen gross-minus-cost execution contract.
    """
    global_mean = float(train[r3_cost.net_bps_column].mean())
    labels = train[r3_cost.class_column]
    values = np.array(
        [
            float(train.loc[labels.eq(index), r3_cost.net_bps_column].mean())
            if labels.eq(index).any() else global_mean
            for index in range(3)
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise FeaturePortabilityMDAError("train-only R3 class-to-common-bps map is non-finite")
    return values


def _r3_probabilities(
    callback: R3FitPredictCallback,
    *,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_weight: np.ndarray,
    eval_matrix: np.ndarray,
    seed: int,
    model_contract: FrozenR3ModelContract,
) -> np.ndarray:
    try:
        probabilities = callback(
            train_matrix, train_labels, train_weight, eval_matrix,
            seed=int(seed), model_contract=model_contract,
        )
    except TypeError as exc:
        raise FeaturePortabilityMDAError(
            "R3 callback must accept train features/classes/sample weights, eval features, seed, and model_contract"
        ) from exc
    output = np.asarray(probabilities, dtype=float)
    if output.shape != (len(eval_matrix), 3) or not np.isfinite(output).all():
        raise FeaturePortabilityMDAError("frozen R3 callback must return finite [P(adverse), P(weak), P(clear)]")
    if (output < -1e-9).any() or not np.allclose(output.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise FeaturePortabilityMDAError("frozen R3 callback output must be a three-class probability simplex")
    return np.clip(output, 0.0, 1.0)


def fit_predict_frozen_r3_lgbm(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_weight: np.ndarray,
    eval_matrix: np.ndarray,
    *,
    seed: int,
    model_contract: FrozenR3ModelContract,
) -> np.ndarray:
    """Fit the explicitly frozen project R3 classifier for F4 evidence.

    The callback exposes no HPO surface and never receives realised net bps as
    a fit target.  The caller separately maps the probability simplex through
    a prior-resolved, fold-local class payoff map.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - required project runtime
        raise FeaturePortabilityMDAError("LightGBM is required for frozen R3 F4 scoring") from exc
    x_train = np.asarray(train_matrix, dtype=np.float32)
    x_eval = np.asarray(eval_matrix, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int8).reshape(-1)
    weights = np.asarray(train_weight, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_eval.ndim != 2 or x_train.shape[1] != x_eval.shape[1]:
        raise FeaturePortabilityMDAError("frozen R3 callback received incompatible feature matrices")
    if len(x_train) != len(labels) or len(labels) != len(weights) or not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise FeaturePortabilityMDAError("frozen R3 callback received invalid labels or sample weights")
    if set(labels).difference({0, 1, 2}):
        raise FeaturePortabilityMDAError("frozen R3 callback requires adverse/weak/clear labels")
    params = dict(model_contract.params)
    objective = str(params.pop("objective", "multiclass")).lower()
    num_class = int(params.pop("num_class", 3))
    if objective != "multiclass" or num_class != 3:
        raise FeaturePortabilityMDAError("frozen R3 contract must be multiclass with num_class=3")
    contract_seed = params.pop("random_state", None)
    if contract_seed is not None and int(contract_seed) != int(seed):
        raise FeaturePortabilityMDAError("frozen R3 contract random_state conflicts with fold seed")
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, random_state=int(seed), **params,
    )
    model.fit(x_train, labels, sample_weight=weights)
    probabilities = np.asarray(model.predict_proba(x_eval), dtype=np.float64)
    if probabilities.shape != (len(x_eval), 3):
        raise FeaturePortabilityMDAError("frozen R3 LGBM did not emit three probabilities")
    return probabilities


def _fit_predict_by_side(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    contract: Mapping[str, Sequence[str]],
    r3_cost: R3CostContract,
    r3_fit_predict: R3FitPredictCallback,
    r3_model: FrozenR3ModelContract,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for side_index, side in enumerate(("long", "short")):
        train_side = train.loc[train["side_name"].eq(side)]
        eval_side = evaluate.loc[evaluate["side_name"].eq(side)]
        train_ok = _complete(train_side, contract[side])
        eval_ok = _complete(eval_side, contract[side])
        train_side, eval_side = train_side.loc[train_ok], eval_side.loc[eval_ok]
        if train_side.empty or eval_side.empty:
            raise FeaturePortabilityMDAError(f"{side} has no complete train/evaluation rows for its fixed base contract")
        class_map = _class_to_common_net_bps_map(train_side, r3_cost=r3_cost)
        probabilities = _r3_probabilities(
            r3_fit_predict, train_matrix=_matrix(train_side, contract[side]),
            train_labels=train_side[r3_cost.class_column].to_numpy(np.int8),
            train_weight=_r3_sample_weights(train_side, r3_cost=r3_cost),
            eval_matrix=_matrix(eval_side, contract[side]), seed=int(seed) + side_index,
            model_contract=r3_model,
        )
        scored = eval_side.loc[:, ["candidate_id", "decision_ts", "side_name", r3_cost.net_bps_column]].copy()
        scored["score_common_net_bps"] = probabilities @ class_map
        if not np.isfinite(scored["score_common_net_bps"].to_numpy(float)).all():
            raise FeaturePortabilityMDAError("frozen R3/class-map path emitted non-finite common-bps scores")
        rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def _pooled_top10_net_bps(scored: pd.DataFrame, *, target_column: str) -> tuple[float, int]:
    if scored.empty:
        raise FeaturePortabilityMDAError("cannot calculate pooled global top-10 on an empty score surface")
    # This is intentionally the only rank operation in the module.  It pools
    # common-unit long/short scores and deterministic candidate-ID tie breaks.
    ordered = scored.sort_values(["score_common_net_bps", "candidate_id"], ascending=[False, True], kind="stable")
    count = max(1, int(np.ceil(len(ordered) * 0.10)))
    return float(ordered.head(count)[target_column].mean()), count


def _inner_boundaries(train: pd.DataFrame, *, outer_start: pd.Timestamp, folds: int) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    if folds < 1:
        raise FeaturePortabilityMDAError("at least one inner MDA fold is required")
    times = np.sort(train["decision_ts"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    if len(times) < folds + 2:
        raise FeaturePortabilityMDAError("insufficient chronology for requested inner MDA folds")
    # Later inner blocks leave an expanding history while keeping all MDA
    # labels inside outer training.  Boundaries derive from decision time only.
    indexes = np.linspace(max(1, len(times) // (folds + 1)), len(times) - 1, folds + 1, dtype=int)
    starts = [pd.Timestamp(times[index], tz="UTC") for index in indexes[:-1]]
    ends = [pd.Timestamp(times[index], tz="UTC") for index in indexes[1:]]
    if ends[-1] < outer_start:
        ends[-1] = outer_start
    return tuple((start, end) for start, end in zip(starts, ends) if start < end)


def _grouped_fold_mda(
    outer_train: pd.DataFrame,
    *,
    transport: ChronologicalTransport,
    contract: Mapping[str, Sequence[str]],
    transformed: Mapping[str, Sequence[str]],
    r3_cost: R3CostContract,
    r3_fit_predict: R3FitPredictCallback,
    r3_model: FrozenR3ModelContract,
    seed: int,
    inner_folds: int,
    control_cache: dict[tuple[object, ...], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(_inner_boundaries(outer_train, outer_start=transport.evaluation_start, folds=inner_folds)):
        train = outer_train.loc[
            outer_train["decision_ts"].lt(start) & outer_train["label_available_ts"].lt(start)
        ]
        evaluate = outer_train.loc[outer_train["decision_ts"].ge(start) & outer_train["decision_ts"].lt(end)]
        if train.empty or evaluate.empty:
            raise FeaturePortabilityMDAError("inner MDA fold has empty train/evaluation support")
        if not train["label_available_ts"].lt(start).all():
            raise FeaturePortabilityMDAError("MDA train labels are not resolved before fold evaluation")
        # The grouped permutation includes missing values when it shuffles a
        # field.  If we let each prediction pass apply its own complete-row
        # filter afterwards, a value missing in one transform can move to a
        # different candidate and create an invalid control/permuted universe.
        # Establish the exact F3-complete candidate universe *before* either
        # the unpermuted or permuted model sees it.  This is an MDA evaluation
        # eligibility rule, not a feature-dependent post-score filter.
        prefilter_rows = int(len(evaluate))
        complete = np.zeros(len(evaluate), dtype=bool)
        for side in ("long", "short"):
            positions = np.flatnonzero(evaluate["side_name"].eq(side).to_numpy())
            if not len(positions):
                raise FeaturePortabilityMDAError(f"inner MDA fold has no {side} candidates")
            complete[positions] = _complete(evaluate.iloc[positions], contract[side])
        evaluate = evaluate.loc[complete].copy()
        if set(evaluate["side_name"].astype(str)) != {"long", "short"}:
            raise FeaturePortabilityMDAError("inner MDA complete-row universe lost one side")
        # All transform-family permutations for one fixed contract share the
        # same causal train/evaluation surface.  Reuse the deterministic,
        # unpermuted score surface inside that narrow call scope rather than
        # refitting it once per candidate group.  The cache key contains the
        # exact ordered side contracts and prefiltered candidate IDs, so it
        # cannot cross a feature contract, fold, or eligibility universe.
        cache_key = (
            int(index), int(seed + 1000 * index), tuple(contract["long"]), tuple(contract["short"]),
            tuple(evaluate["candidate_id"].astype(str)),
        )
        control = control_cache.get(cache_key) if control_cache is not None else None
        if control is None:
            control = _fit_predict_by_side(
                train, evaluate, contract=contract, r3_cost=r3_cost,
                r3_fit_predict=r3_fit_predict, r3_model=r3_model, seed=seed + 1000 * index,
            )
            if control_cache is not None:
                control_cache[cache_key] = control.copy()
        # The identical-row requirement is enforced by predicting the same
        # scored candidate IDs after jointly permuting only the declared F3
        # transform group, independently within each side.
        permuted_evaluate = evaluate.copy()
        for side_index, side in enumerate(("long", "short")):
            positions = np.flatnonzero(permuted_evaluate["side_name"].eq(side).to_numpy())
            if len(positions) < 2:
                raise FeaturePortabilityMDAError("grouped MDA needs at least two evaluation rows per side")
            rng = np.random.default_rng(int(seed) + 100000 * index + side_index)
            source = permuted_evaluate.iloc[positions]
            permutation = rng.permutation(len(positions))
            for field in transformed[side]:
                permuted_evaluate.iloc[positions, permuted_evaluate.columns.get_loc(field)] = source[field].to_numpy()[permutation]
        permuted = _fit_predict_by_side(
            train, permuted_evaluate, contract=contract, r3_cost=r3_cost,
            r3_fit_predict=r3_fit_predict, r3_model=r3_model, seed=seed + 1000 * index,
        )
        if not np.array_equal(
            control.sort_values("candidate_id")["candidate_id"].to_numpy(),
            permuted.sort_values("candidate_id")["candidate_id"].to_numpy(),
        ):
            raise FeaturePortabilityMDAError("group MDA control/permuted candidates differ")
        control_bps, trades = _pooled_top10_net_bps(control, target_column=r3_cost.net_bps_column)
        permuted_bps, _ = _pooled_top10_net_bps(permuted, target_column=r3_cost.net_bps_column)
        row: dict[str, Any] = {
            "transport": transport.name,
            "fold_id": f"{transport.name}_mda_{index:02d}",
            "fold_evaluation_start": start,
            "fold_evaluation_end": end,
            "train_rows": int(len(train)),
            "evaluation_rows": int(len(control)),
            "evaluation_rows_before_contract_completeness": prefilter_rows,
            "evaluation_rows_after_contract_completeness": int(len(evaluate)),
            "train_max_label_available_ts": train["label_available_ts"].max(),
            "labels_resolved_before_fold_evaluation": True,
            "ranking_scope": "pooled_global",
            "control_top10_net_bps": control_bps,
            "permuted_top10_net_bps": permuted_bps,
            "group_mda_bps": control_bps - permuted_bps,
            "top10_trades": trades,
            "permutation_style": "joint_row_shuffle_by_side_of_actual_f3_transforms_on_prefiltered_complete_candidates",
        }
        # Persist the actual train-only common-unit map used by this fold so
        # the MDA artifact can be audited without reconstructing a callback.
        for side in ("long", "short"):
            fit_rows = train.loc[train["side_name"].eq(side)]
            fit_rows = fit_rows.loc[_complete(fit_rows, contract[side])]
            class_map = _class_to_common_net_bps_map(fit_rows, r3_cost=r3_cost)
            counts = fit_rows[r3_cost.class_column].value_counts().reindex(range(3), fill_value=0)
            for class_index, value in enumerate(class_map):
                row[f"{side}_r3_class_{class_index}_train_common_net_bps"] = float(value)
                row[f"{side}_r3_class_{class_index}_train_rows"] = int(counts.iloc[class_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _representation_coverage(
    evaluation: pd.DataFrame, *, transport: str, representation: str, contract: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        part = evaluation.loc[evaluation["side_name"].eq(side)]
        for field in contract[side]:
            values = pd.to_numeric(part[field], errors="coerce").to_numpy(float)
            rows.append({
                "representation": representation, "transport": transport, "side_name": side,
                "feature": field, "coverage": float(np.isfinite(values).mean()), "evaluation_rows": int(len(part)),
            })
    return pd.DataFrame(rows)


def _rank_f4_transform_groups(
    outer_train: pd.DataFrame,
    *,
    transport: ChronologicalTransport,
    f3_contract: Mapping[str, Sequence[str]],
    transform_groups: Mapping[str, Mapping[str, Sequence[str]]],
    r3_cost: R3CostContract,
    r3_fit_predict: R3FitPredictCallback,
    r3_model: FrozenR3ModelContract,
    seed: int,
    inner_folds: int,
    control_cache: dict[tuple[object, ...], pd.DataFrame],
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Rank F3 transform *families* on only predecessor inner-fold MDA.

    The ranking is intentionally global after the common-bps map: a separate
    long/short selection would make the later F4 result a pair of local books.
    Side-local base fits remain intact because every group still supplies its
    side-specific fields.  The result is a deterministic order used only to
    build nested compact candidates for this transport's outer evaluation.
    """
    rows: list[pd.DataFrame] = []
    group_order = {name: index for index, (name, _) in enumerate(F4_TRANSFORM_GROUPS)}
    for name, _ in F4_TRANSFORM_GROUPS:
        transformed = {side: tuple(transform_groups[side][name]) for side in ("long", "short")}
        local = _grouped_fold_mda(
            outer_train, transport=transport, contract=f3_contract, transformed=transformed,
            r3_cost=r3_cost, r3_fit_predict=r3_fit_predict, r3_model=r3_model,
            seed=seed, inner_folds=inner_folds, control_cache=control_cache,
        ).copy()
        local["feature_group"] = name
        local["mda_role"] = "inner_only_f4_transform_group_ranking"
        rows.append(local)
    output = pd.concat(rows, ignore_index=True)
    summary = (
        output.groupby("feature_group", observed=True)["group_mda_bps"]
        .agg([("group_mda_median_bps", "median"), ("group_mda_min_bps", "min")])
        .reset_index()
    )
    mad = (
        output.groupby("feature_group", observed=True)["group_mda_bps"]
        .apply(lambda values: float(np.median(np.abs(values.to_numpy(float) - np.median(values.to_numpy(float))))))
        .rename("group_mda_mad_bps")
        .reset_index()
    )
    summary = summary.merge(mad, on="feature_group", validate="one_to_one")
    summary["group_mda_stable_score_bps"] = summary["group_mda_median_bps"] - 0.5 * summary["group_mda_mad_bps"]
    summary["group_order"] = summary["feature_group"].map(group_order).astype(int)
    summary = summary.sort_values(
        ["group_mda_stable_score_bps", "group_mda_median_bps", "group_order"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    summary["inner_mda_rank"] = np.arange(1, len(summary) + 1, dtype=np.int16)
    output = output.merge(summary, on="feature_group", validate="many_to_one")
    ranking = tuple(summary["feature_group"].astype(str))
    if len(ranking) != len(F4_TRANSFORM_GROUPS):  # defensive: the grouping contract is fixed.
        raise FeaturePortabilityMDAError("F4 inner-MDA rank did not cover every transform family")
    return ranking, output


def materialize_feature_portability_f4_evidence(
    panel: pd.DataFrame,
    *,
    representation_features: Mapping[str, Mapping[str, Sequence[str]]],
    control_representation: str,
    f3_representation: str,
    transports: Sequence[ChronologicalTransport],
    r3_cost: R3CostContract,
    r3_model: FrozenR3ModelContract,
    r3_fit_predict: R3FitPredictCallback,
    inner_folds: int = 2,
    min_coverage: float = 0.99,
    f4_group_counts: Sequence[int] = DEFAULT_F4_GROUP_COUNTS,
) -> F4EvidenceMaterialization:
    """Materialise development-only F4 evidence from a pre-built panel.

    ``control_representation`` is an unpermuted base-contract comparator and
    is not emitted as a selection candidate.  F3 remains a diagnostic full
    transform comparator.  The actual F4 candidates are nested compact field
    contracts selected from *inner* transform-family MDA separately for each
    outer transport, then evaluated against the control on identical rows.
    """
    if not callable(r3_fit_predict):
        raise FeaturePortabilityMDAError("an explicit frozen R3 fit/predict callback is required; direct net regression is forbidden")
    work = _normalise_panel(panel, r3_cost=r3_cost)
    contracts = _normalise_contracts(representation_features)
    if control_representation not in contracts or f3_representation not in contracts:
        raise FeaturePortabilityMDAError("control_representation and f3_representation must be declared contracts")
    if control_representation == f3_representation:
        raise FeaturePortabilityMDAError("F3 candidate cannot also be its unpermuted control")
    if set(contracts).difference({control_representation, f3_representation}):
        raise FeaturePortabilityMDAError(
            "F4 evidence accepts exactly the frozen F0 control and full F3 source contract; "
            "compact candidates are generated from inner chronological MDA"
        )
    if not transports:
        raise FeaturePortabilityMDAError("at least one development transport is required")
    names = [transport.name for transport in transports]
    if len(names) != len(set(names)):
        raise FeaturePortabilityMDAError("transport names must be unique")
    try:
        group_counts = validate_group_counts(f4_group_counts)
        transform_groups = f4_transform_groups(contracts[f3_representation])
    except ValueError as exc:
        raise FeaturePortabilityMDAError(f"invalid F4 compact contract: {exc}") from exc
    transformed = _validate_f3_contract(contracts[f3_representation])
    # A source-field audit is insufficient: the F3 runner materialises a
    # concrete generated matrix.  Refuse a contract that omits any matching
    # rank90/rank180/robust-z90/robust-z180/delta4/delta24 column present in that matrix, rather than
    # letting a caller declare only a convenient transformed subset.
    actual_f3_columns = set(f3_transformed_fields(work.columns))
    declared_f3_columns = {field for values in transformed.values() for field in values}
    undeclared_f3_columns = sorted(actual_f3_columns.difference(declared_f3_columns))
    if undeclared_f3_columns:
        raise FeaturePortabilityMDAError(
            "F3 contract omits actual transformed panel fields from coverage audit: "
            f"{undeclared_f3_columns}"
        )
    evidence_rows: list[dict[str, Any]] = []
    transformed_coverage_rows: list[pd.DataFrame] = []
    source_coverage_rows: list[pd.DataFrame] = []
    representation_coverage_rows: list[pd.DataFrame] = []
    mda_rows: list[pd.DataFrame] = []
    feature_group_mda_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    compact_contracts_by_transport: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    compact_rankings_by_transport: dict[str, tuple[str, ...]] = {}
    transport_surfaces: list[tuple[int, ChronologicalTransport, pd.DataFrame, pd.DataFrame]] = []
    for transport_index, transport in enumerate(transports):
        outer_train = work.loc[
            work["decision_ts"].ge(transport.train_start)
            & work["decision_ts"].lt(transport.evaluation_start)
            & work["label_available_ts"].lt(transport.evaluation_start)
        ]
        evaluation = work.loc[
            work["decision_ts"].ge(transport.evaluation_start) & work["decision_ts"].lt(transport.evaluation_end)
        ]
        if outer_train.empty or evaluation.empty:
            raise FeaturePortabilityMDAError(f"{transport.name} has empty resolved train/evaluation support")
        if not outer_train["label_available_ts"].lt(transport.evaluation_start).all():
            raise FeaturePortabilityMDAError("outer train labels are not resolved before evaluation")
        # Full F3 is retained as a coverage diagnostic.  It must not abort F4
        # before the compact source intersection is known.
        transformed_coverage_rows.append(audit_f3_transformed_coverage(
            work, transport=transport, f3_features=transformed, min_coverage=min_coverage, require_all=False,
        ))
        source_coverage_rows.append(_f4_source_bundle_coverage(
            evaluation, transport=transport, transform_groups=transform_groups, min_coverage=min_coverage,
        ))
        transport_surfaces.append((transport_index, transport, outer_train, evaluation))
    source_intersection, source_intersection_coverage = _cross_transport_coverage_safe_sources(
        pd.concat(source_coverage_rows, ignore_index=True), transports=transports, min_coverage=min_coverage,
    )
    safe_transform_groups = restrict_f4_transform_groups_to_sources(
        transform_groups, source_fields_by_side=source_intersection,
    )
    safe_f3_contract: dict[str, tuple[str, ...]] = {
        side: tuple(
            field for group in ("portable_raw", *(name for name, _ in F4_TRANSFORM_GROUPS))
            for field in safe_transform_groups[side][group]
        )
        for side in ("long", "short")
    }
    full_f3_eligible = bool(
        all(frame["passes_99pct_coverage"].all() for frame in transformed_coverage_rows)
        and bool(source_intersection_coverage["source_bundle_passes_99pct_coverage"].all())
    )
    for transport_index, transport, outer_train, evaluation in transport_surfaces:
        inner_seed = int(r3_model.random_seed) + 10000 * transport_index
        f3_control_cache: dict[tuple[object, ...], pd.DataFrame] = {}
        f3_mda: pd.DataFrame | None = None
        if full_f3_eligible:
            f3_mda = _grouped_fold_mda(
                outer_train, transport=transport, contract=contracts[f3_representation], transformed=transformed,
                r3_cost=r3_cost, r3_fit_predict=r3_fit_predict, r3_model=r3_model,
                seed=inner_seed, inner_folds=inner_folds, control_cache=f3_control_cache,
            ).assign(mda_role="full_f3_diagnostic_eligible")
            mda_rows.append(f3_mda)
        ranked_groups, group_mda = _rank_f4_transform_groups(
            outer_train, transport=transport, f3_contract=safe_f3_contract,
            transform_groups=safe_transform_groups, r3_cost=r3_cost, r3_fit_predict=r3_fit_predict,
            r3_model=r3_model, seed=inner_seed, inner_folds=inner_folds,
            control_cache={},
        )
        feature_group_mda_rows.append(group_mda)
        compact_rankings_by_transport[transport.name] = ranked_groups
        transport_compacts = compact_contracts_for_ranked_groups(
            safe_transform_groups, ranked_transform_groups=ranked_groups, group_counts=group_counts,
        )
        compact_contracts_by_transport[transport.name] = transport_compacts
        candidate_contracts: dict[str, Mapping[str, Sequence[str]]] = {
            **({f3_representation: contracts[f3_representation]} if full_f3_eligible else {}),
            **transport_compacts,
        }
        f3_top10_net_bps: float | None = None
        for representation_index, (representation, candidate_contract) in enumerate(candidate_contracts.items()):
            representation_coverage_rows.append(_representation_coverage(
                evaluation, transport=transport.name, representation=representation, contract=candidate_contract,
            ))
            # Both score surfaces are first limited to the same rows, before
            # either model sees an evaluation matrix.  This prohibits a lift
            # caused by a representation quietly dropping difficult rows.
            common_evaluation = evaluation.copy()
            common = np.ones(len(common_evaluation), dtype=bool)
            for side in ("long", "short"):
                side_mask = common_evaluation["side_name"].eq(side).to_numpy()
                common[side_mask] &= _complete(common_evaluation.loc[side_mask], candidate_contract[side])
                common[side_mask] &= _complete(common_evaluation.loc[side_mask], contracts[control_representation][side])
            common_evaluation = common_evaluation.loc[common]
            if set(common_evaluation["side_name"]) != {"long", "short"}:
                raise FeaturePortabilityMDAError("identical-row pooled comparison lost one side")
            candidate_scores = _fit_predict_by_side(
                outer_train, common_evaluation, contract=candidate_contract, r3_cost=r3_cost,
                r3_fit_predict=r3_fit_predict, r3_model=r3_model,
                seed=int(r3_model.random_seed) + 10000 * transport_index + 100 * representation_index,
            )
            control_scores = _fit_predict_by_side(
                outer_train, common_evaluation, contract=contracts[control_representation], r3_cost=r3_cost,
                r3_fit_predict=r3_fit_predict, r3_model=r3_model,
                seed=int(r3_model.random_seed) + 10000 * transport_index + 100 * representation_index,
            )
            if not np.array_equal(
                candidate_scores.sort_values("candidate_id")["candidate_id"].to_numpy(),
                control_scores.sort_values("candidate_id")["candidate_id"].to_numpy(),
            ):
                raise FeaturePortabilityMDAError("candidate/control comparison does not use identical rows")
            candidate_top10, top10_trades = _pooled_top10_net_bps(candidate_scores, target_column=r3_cost.net_bps_column)
            control_top10, _ = _pooled_top10_net_bps(control_scores, target_column=r3_cost.net_bps_column)
            if representation == f3_representation:
                f3_top10_net_bps = candidate_top10
            elif full_f3_eligible and f3_top10_net_bps is None:
                raise FeaturePortabilityMDAError("F4 candidate was evaluated before its frozen full-F3 control")
            representation_coverage = representation_coverage_rows[-1]
            if representation == f3_representation:
                if f3_mda is None:
                    raise FeaturePortabilityMDAError("ineligible full F3 must not be emitted as a candidate")
                candidate_mda = f3_mda
            else:
                # This is a second, independent inner-only MDA calculation of
                # the selected compact contract as a whole.  It is not the
                # outer score nor a sum of individual group effects.
                selected_groups = [
                    name for name in ranked_groups
                    if any(field in candidate_contract["long"] for field in safe_transform_groups["long"][name])
                ]
                selected_fields = {
                    side: tuple(field for name in selected_groups for field in safe_transform_groups[side][name])
                    for side in ("long", "short")
                }
                candidate_mda = _grouped_fold_mda(
                    outer_train, transport=transport, contract=candidate_contract, transformed=selected_fields,
                    r3_cost=r3_cost, r3_fit_predict=r3_fit_predict, r3_model=r3_model,
                    seed=inner_seed + 1_000_000 + 100_000 * representation_index,
                    inner_folds=inner_folds,
                ).assign(
                    mda_role="selected_compact_f4_grouped_mda",
                    representation=representation,
                    selected_transform_groups="|".join(selected_groups),
                )
                mda_rows.append(candidate_mda)
            mda_value = float(candidate_mda["group_mda_bps"].mean())
            evidence_rows.append({
                "representation": representation,
                "transport": transport.name,
                "feature_count": int(max(len(candidate_contract["long"]), len(candidate_contract["short"]))),
                "coverage": float(representation_coverage["coverage"].min()),
                "incremental_top10_net_bps": candidate_top10 - control_top10,
                "incremental_vs_f3_top10_net_bps": (
                    candidate_top10 - float(f3_top10_net_bps) if full_f3_eligible else float("nan")
                ),
                "full_f3_control_eligible": bool(full_f3_eligible),
                "transport_mda_bps": mda_value,
                "development_stage": "development_transport",
                "chronological_verified": True,
                "global_ranking_verified": True,
                "ranking_scope": "pooled_global",
                "model_hpo_performed": False,
            })
            audit_rows.append({
                "transport": transport.name, "representation": representation,
                "outer_train_rows": int(len(outer_train)), "outer_evaluation_rows": int(len(evaluation)),
                "identical_row_comparison_rows": int(len(common_evaluation)), "top10_trades": top10_trades,
                "outer_train_max_label_available_ts": outer_train["label_available_ts"].max(),
                "labels_resolved_before_outer_evaluation": True,
                "candidate_top10_net_bps": candidate_top10, "unpermuted_control_top10_net_bps": control_top10,
                "global_ranking": "one_pooled_long_short_common_net_bps_ranking",
                "compact_selection_source": (
                    "inner_chronological_transform_group_mda" if representation.startswith("F4_compact_top")
                    else "not_applicable_full_f3_diagnostic"
                ),
                "selected_transform_groups": (
                    "|".join(name for name in ranked_groups if any(
                        field in candidate_contract["long"] for field in safe_transform_groups["long"][name]
                    )) if representation.startswith("F4_compact_top") else None
                ),
            })
    evidence = pd.DataFrame(evidence_rows)
    # Preserve the selector's documented input order, which makes CSV output
    # and downstream table comparisons stable without adding producer columns.
    evidence = evidence.loc[:, [
        "representation", "transport", "feature_count", "coverage", "incremental_top10_net_bps",
        "transport_mda_bps", "development_stage", "chronological_verified", "global_ranking_verified",
        "ranking_scope", "model_hpo_performed",
        "incremental_vs_f3_top10_net_bps",
        "full_f3_control_eligible",
    ]]
    compact_contracts = compact_contract_payload(
        source_representation=f3_representation, by_transport=compact_contracts_by_transport,
        ranking_by_transport=compact_rankings_by_transport, group_counts=group_counts,
    )
    compact_contracts["coverage_safe_source_intersection"] = {
        side: list(values) for side, values in source_intersection.items()
    }
    compact_contracts["full_f3_diagnostic_eligible"] = bool(full_f3_eligible)
    return F4EvidenceMaterialization(
        evidence=evidence.sort_values(["representation", "transport"], kind="stable").reset_index(drop=True),
        transformed_coverage=pd.concat(transformed_coverage_rows, ignore_index=True),
        source_intersection_coverage=source_intersection_coverage,
        representation_coverage=pd.concat(representation_coverage_rows, ignore_index=True),
        fold_mda=pd.concat(mda_rows, ignore_index=True),
        feature_group_mda=pd.concat(feature_group_mda_rows, ignore_index=True),
        compact_contracts=compact_contracts,
        transport_audit=pd.DataFrame(audit_rows),
        manifest={
            "schema": SCHEMA,
            "development_only": True,
            "final_november_oos_consumed": False,
            "base_model_hpo_performed": False,
            "frozen_r3_model_id": r3_model.model_id,
            "frozen_r3_model_params": dict(r3_model.params),
            "r3_class_to_common_bps_map": "fold_train_only_class_mean_net_bps_after_gross_minus_net_cost_reconciliation",
            "r3_cost_contract": {
                "class_column": r3_cost.class_column,
                "gross_bps_column": r3_cost.gross_bps_column,
                "net_bps_column": r3_cost.net_bps_column,
                "expected_cost_bps": float(r3_cost.expected_cost_bps),
                "sample_weight_column": r3_cost.sample_weight_column,
            },
            "grouped_mda": "fold_local_train_only_joint_row_shuffle_of_actual_f3_transforms_and_predeclared_compact_groups",
            "labels": "train labels resolved strictly before each fold evaluation boundary",
            "ranking": "one_pooled_global_long_short_common_net_bps_top10",
            "control": "identical_row_unpermuted_control",
            "f3_transformed_fields": {side: list(values) for side, values in transformed.items()},
            "full_f3_diagnostic": {
                "eligible": bool(full_f3_eligible),
                "ineligible_action": "diagnostic_only_not_an_f4_promotion_gate" if not full_f3_eligible else "full_f3_noninferiority_gate_required",
                "coverage_artifacts": [
                    "f4_actual_f3_transformed_coverage.parquet",
                    "f4_source_intersection_coverage.parquet",
                ],
            },
            "f4_compact": {
                "selection": "transform-family ranking from predecessor inner chronological grouped MDA only on the exact cross-transport per-side coverage-safe source intersection",
                "group_counts": list(group_counts),
                "requires_exact_cross_transport_contract_stability": True,
                "base_control_promotion": "positive F0 lift and stable grouped MDA in every development transport; require non-negative full-F3 lift only when full F3 is coverage-eligible",
                "compact_contract_artifact": "f4_compact_contracts.json",
            },
        },
    )


__all__ = [
    "SCHEMA", "ChronologicalTransport", "F4EvidenceMaterialization", "FeaturePortabilityMDAError",
    "FrozenR3ModelContract", "R3CostContract", "R3FitPredictCallback",
    "audit_f3_transformed_coverage", "f3_transformed_fields", "fit_predict_frozen_r3_lgbm",
    "materialize_feature_portability_f4_evidence",
]
