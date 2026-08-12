"""Memory-bounded, support-aware pairwise logistic objectives for LightGBM.

This is deliberately a *training-only* primitive.  A caller supplies the
already-resolved residual labels for a chronological training partition and
receives a LightGBM custom objective closure.  The closure needs only model
predictions during boosting; it has no access to the source frame, future rows,
or any inference-time candidates.

Pairs are constructed only within ``query_id`` (normally decision timestamp ×
side).  Construction avoids an :math:`O(n^2)` pair matrix: each query emits at
most ``max_pairs_per_query`` pairs from a deterministic, bounded sample.  Raw
pair importance is normalised within each query so an unusually broad
cross-section cannot dominate the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


SCHEMA = "support_aware_pairwise_logistic_objective_v1"
_EPS = 1e-12


class SupportAwarePairwiseObjectiveError(ValueError):
    """Raised when a pairwise-training contract is malformed."""


@dataclass(frozen=True)
class SupportAwarePairwiseColumns:
    """Column names required from a resolved training ledger.

    ``atr_residual`` is the primary ordering label.  ``bps_residual`` and
    ``atr_grade`` are consistency/separation evidence, while ``support`` and
    ``incumbent_base_score`` affect pair weighting only.  None of these fields
    is read by the objective after construction.
    """

    query_id: str = "query_id"
    atr_residual: str = "atr_residual"
    bps_residual: str = "candidate_residual_bps"
    atr_grade: str = "atr_residual_grade"
    support: str = "support_h12"
    incumbent_base_score: str = "prequential_base_expected_net_bps"


@dataclass(frozen=True)
class SupportAwarePairwiseConfig:
    """Bounded construction and confidence controls for residual ranking.

    A *loose* pair requires the ATR- and bps-residual orderings to agree and
    to clear the loose separation floors.  A *strict* pair additionally clears
    stricter separation floors and an ordinal ATR-grade gap.  Strict pairs are
    retained with a configurable multiplier; loose pairs remain available so
    the target does not collapse to a few extreme observations.

    The support multipliers are deliberately symmetric configuration knobs:
    callers can emphasise a supported winner, supported loser, both, or neither
    without using support as an inference-time gate.
    """

    max_pairs_per_query: int = 256
    sampling_attempt_multiplier: int = 24
    exhaustive_pair_limit: int = 4096
    random_state: int = 1729

    loose_atr_separation: float = 0.25
    loose_bps_separation: float = 25.0
    strict_atr_separation: float = 0.75
    strict_bps_separation: float = 75.0
    strict_grade_separation: int = 1
    loose_pair_multiplier: float = 1.0
    strict_pair_multiplier: float = 1.75

    both_supported_multiplier: float = 1.50
    winner_supported_multiplier: float = 1.20
    loser_supported_multiplier: float = 1.00
    neither_supported_multiplier: float = 0.80

    incumbent_misordered_multiplier: float = 1.35
    incumbent_correctly_ordered_multiplier: float = 1.00
    require_incumbent_misorder: bool = False
    incumbent_score_distance_scale: float = 100.0
    incumbent_score_proximity_floor: float = 0.35

    bps_gap_scale: float = 100.0
    atr_grade_gap_multiplier: float = 0.10
    query_total_weight: float = 1.0
    min_hessian: float = 1e-6

    def validate(self) -> None:
        if not 1 <= self.max_pairs_per_query <= 16_384:
            raise SupportAwarePairwiseObjectiveError(
                "max_pairs_per_query must lie in [1, 16384]"
            )
        if not 1 <= self.sampling_attempt_multiplier <= 512:
            raise SupportAwarePairwiseObjectiveError(
                "sampling_attempt_multiplier must lie in [1, 512]"
            )
        if not 1 <= self.exhaustive_pair_limit <= 1_000_000:
            raise SupportAwarePairwiseObjectiveError(
                "exhaustive_pair_limit must lie in [1, 1000000]"
            )
        if self.loose_atr_separation < 0.0 or self.loose_bps_separation < 0.0:
            raise SupportAwarePairwiseObjectiveError("loose separation floors must be non-negative")
        if self.strict_atr_separation < self.loose_atr_separation:
            raise SupportAwarePairwiseObjectiveError(
                "strict_atr_separation must be at least loose_atr_separation"
            )
        if self.strict_bps_separation < self.loose_bps_separation:
            raise SupportAwarePairwiseObjectiveError(
                "strict_bps_separation must be at least loose_bps_separation"
            )
        if self.strict_grade_separation < 0:
            raise SupportAwarePairwiseObjectiveError("strict_grade_separation must be non-negative")
        multipliers = (
            self.loose_pair_multiplier,
            self.strict_pair_multiplier,
            self.both_supported_multiplier,
            self.winner_supported_multiplier,
            self.loser_supported_multiplier,
            self.neither_supported_multiplier,
            self.incumbent_misordered_multiplier,
            self.incumbent_correctly_ordered_multiplier,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in multipliers):
            raise SupportAwarePairwiseObjectiveError("all pair multipliers must be finite and non-negative")
        if self.incumbent_score_distance_scale <= 0.0:
            raise SupportAwarePairwiseObjectiveError("incumbent_score_distance_scale must be positive")
        if not 0.0 < self.incumbent_score_proximity_floor <= 1.0:
            raise SupportAwarePairwiseObjectiveError(
                "incumbent_score_proximity_floor must lie in (0, 1]"
            )
        if self.bps_gap_scale <= 0.0:
            raise SupportAwarePairwiseObjectiveError("bps_gap_scale must be positive")
        if self.atr_grade_gap_multiplier < 0.0:
            raise SupportAwarePairwiseObjectiveError("atr_grade_gap_multiplier must be non-negative")
        if not np.isfinite(self.query_total_weight) or self.query_total_weight <= 0.0:
            raise SupportAwarePairwiseObjectiveError("query_total_weight must be finite and positive")
        if not 0.0 < self.min_hessian <= 1.0:
            raise SupportAwarePairwiseObjectiveError("min_hessian must lie in (0, 1]")


@dataclass(frozen=True)
class PairwiseObjectiveAudit:
    """Compact construction evidence, including rows excluded before fitting."""

    schema: str
    input_rows: int
    valid_rows: int
    invalid_rows: int
    queries: int
    eligible_queries: int
    skipped_queries: int
    candidate_pairs_examined: int
    loose_pairs: int
    strict_pairs: int
    selected_pairs: int
    selected_pairs_by_query: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "input_rows": self.input_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "queries": self.queries,
            "eligible_queries": self.eligible_queries,
            "skipped_queries": self.skipped_queries,
            "candidate_pairs_examined": self.candidate_pairs_examined,
            "loose_pairs": self.loose_pairs,
            "strict_pairs": self.strict_pairs,
            "selected_pairs": self.selected_pairs,
            "selected_pairs_by_query": [list(value) for value in self.selected_pairs_by_query],
        }


@dataclass(frozen=True)
class PairwiseLogisticObjective:
    """Frozen pair ledger plus a LightGBM-compatible logistic objective.

    ``winner_rows`` and ``loser_rows`` are positions in the exact training
    matrix.  They are never candidate IDs, so score/prediction alignment stays
    explicit at the LightGBM boundary.
    """

    winner_rows: np.ndarray
    loser_rows: np.ndarray
    pair_weights: np.ndarray
    is_strict: np.ndarray
    support_class: np.ndarray
    incumbent_misordered: np.ndarray
    query_codes: np.ndarray
    row_count: int
    config: SupportAwarePairwiseConfig
    audit: PairwiseObjectiveAudit

    def __post_init__(self) -> None:
        size = len(self.winner_rows)
        fields = (
            self.loser_rows,
            self.pair_weights,
            self.is_strict,
            self.support_class,
            self.incumbent_misordered,
            self.query_codes,
        )
        if any(len(field) != size for field in fields):
            raise SupportAwarePairwiseObjectiveError("pair ledger fields lost alignment")
        if self.row_count <= 0:
            raise SupportAwarePairwiseObjectiveError("row_count must be positive")
        if size and (
            self.winner_rows.min() < 0
            or self.loser_rows.min() < 0
            or self.winner_rows.max() >= self.row_count
            or self.loser_rows.max() >= self.row_count
        ):
            raise SupportAwarePairwiseObjectiveError("pair ledger contains an out-of-range row")
        if size and np.any(self.winner_rows == self.loser_rows):
            raise SupportAwarePairwiseObjectiveError("self-pairs are not permitted")
        if size and (not np.isfinite(self.pair_weights).all() or np.any(self.pair_weights <= 0.0)):
            raise SupportAwarePairwiseObjectiveError("pair weights must be finite and positive")

    @property
    def pair_count(self) -> int:
        return int(len(self.winner_rows))

    def pair_frame(self) -> pd.DataFrame:
        """Return a compact, row-position-only audit ledger for tests/reports."""
        return pd.DataFrame(
            {
                "winner_row": self.winner_rows,
                "loser_row": self.loser_rows,
                "pair_weight": self.pair_weights,
                "is_strict": self.is_strict,
                "support_class": self.support_class,
                "incumbent_misordered": self.incumbent_misordered,
                "query_code": self.query_codes,
            }
        )

    def __call__(self, preds: Sequence[float], train_data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return gradients and Hessians in LightGBM custom-objective form.

        The second argument is intentionally used only to validate prediction
        length when a LightGBM Dataset exposes ``num_data``.  Labels are not
        read here: the pair ledger was frozen from the caller's chronological
        training partition before fitting began.
        """
        prediction = np.asarray(preds, dtype=np.float64).reshape(-1)
        if len(prediction) != self.row_count:
            raise SupportAwarePairwiseObjectiveError(
                f"prediction length {len(prediction)} does not match pair ledger row_count {self.row_count}"
            )
        num_data = getattr(train_data, "num_data", None)
        if callable(num_data) and int(num_data()) != self.row_count:
            raise SupportAwarePairwiseObjectiveError("LightGBM Dataset row count does not match pair ledger")
        if not np.isfinite(prediction).all():
            raise SupportAwarePairwiseObjectiveError("pairwise objective received non-finite predictions")

        gradient = np.zeros(self.row_count, dtype=np.float64)
        hessian = np.full(self.row_count, self.config.min_hessian, dtype=np.float64)
        if self.pair_count == 0:
            return gradient, hessian

        margin = prediction[self.winner_rows] - prediction[self.loser_rows]
        # sigmoid(-margin), evaluated without overflow for large margins.
        loss_probability = np.empty_like(margin)
        positive = margin >= 0.0
        exp_negative = np.exp(-margin[positive])
        loss_probability[positive] = exp_negative / (1.0 + exp_negative)
        exp_positive = np.exp(margin[~positive])
        loss_probability[~positive] = 1.0 / (1.0 + exp_positive)

        weighted_gradient = self.pair_weights * loss_probability
        weighted_hessian = self.pair_weights * loss_probability * (1.0 - loss_probability)
        np.add.at(gradient, self.winner_rows, -weighted_gradient)
        np.add.at(gradient, self.loser_rows, weighted_gradient)
        np.add.at(hessian, self.winner_rows, weighted_hessian)
        np.add.at(hessian, self.loser_rows, weighted_hessian)
        return gradient, np.maximum(hessian, self.config.min_hessian)

    def lightgbm_objective(self) -> Callable[[Sequence[float], Any], tuple[np.ndarray, np.ndarray]]:
        """Return ``self`` as an explicit closure for ``lightgbm.train``.

        Use either ``params[\"objective\"] = objective.lightgbm_objective()`` or
        pass the returned callable in the equivalent LightGBM custom-objective
        slot for the installed LightGBM version.
        """
        return self


def _stable_query_seed(query: object, random_state: int) -> int:
    digest = blake2b(
        f"{int(random_state)}|{type(query).__name__}|{query!r}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise SupportAwarePairwiseObjectiveError(f"missing required training column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)


def _support_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise SupportAwarePairwiseObjectiveError(f"missing required training column {column!r}")
    value = frame[column]
    if pd.api.types.is_bool_dtype(value) or pd.api.types.is_numeric_dtype(value):
        return value.fillna(False).astype(bool).to_numpy(copy=False)
    normalised = value.astype("string").str.strip().str.lower()
    return normalised.isin(("1", "true", "t", "yes", "y")).to_numpy(dtype=bool, copy=False)


def _sample_candidate_pairs(
    positions: np.ndarray,
    *,
    query: object,
    config: SupportAwarePairwiseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce bounded unordered pairs without allocating an n-by-n matrix."""
    count = len(positions)
    possible = count * (count - 1) // 2
    if possible == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(_stable_query_seed(query, config.random_state))
    if possible <= config.exhaustive_pair_limit:
        left_local, right_local = np.triu_indices(count, k=1)
        order = rng.permutation(len(left_local))
        return positions[left_local[order]], positions[right_local[order]]

    attempts = min(
        possible,
        max(config.max_pairs_per_query, config.max_pairs_per_query * config.sampling_attempt_multiplier),
    )
    selected: set[int] = set()
    left_local: list[int] = []
    right_local: list[int] = []
    # The set is bounded by ``attempts`` and avoids duplicate pair contribution.
    while len(left_local) < attempts:
        first = int(rng.integers(0, count))
        second = int(rng.integers(0, count - 1))
        if second >= first:
            second += 1
        lo, hi = (first, second) if first < second else (second, first)
        key = lo * count + hi
        if key in selected:
            if len(selected) >= possible:
                break
            continue
        selected.add(key)
        left_local.append(lo)
        right_local.append(hi)
    return positions[np.asarray(left_local)], positions[np.asarray(right_local)]


def _support_multiplier(
    winner_support: np.ndarray,
    loser_support: np.ndarray,
    config: SupportAwarePairwiseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    # Values double as a human-readable audit category and a compact model input.
    category = np.full(len(winner_support), "neither", dtype=object)
    both = winner_support & loser_support
    winner_only = winner_support & ~loser_support
    loser_only = ~winner_support & loser_support
    category[both] = "both"
    category[winner_only] = "winner_only"
    category[loser_only] = "loser_only"
    multiplier = np.full(len(category), config.neither_supported_multiplier, dtype=np.float64)
    multiplier[both] = config.both_supported_multiplier
    multiplier[winner_only] = config.winner_supported_multiplier
    multiplier[loser_only] = config.loser_supported_multiplier
    return category, multiplier


def build_support_aware_pairwise_objective(
    frame: pd.DataFrame,
    *,
    columns: SupportAwarePairwiseColumns = SupportAwarePairwiseColumns(),
    config: SupportAwarePairwiseConfig = SupportAwarePairwiseConfig(),
) -> PairwiseLogisticObjective:
    """Build a deterministic, query-normalised objective from resolved rows.

    The caller is responsible for chronological label availability, purging and
    embargo before invoking this function.  To prevent accidental outcome use
    at inference, this function accepts a concrete *training frame* and the
    resulting objective stores only row positions, labels' pair ordering, and
    frozen pair weights.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SupportAwarePairwiseObjectiveError("frame must be a pandas DataFrame")
    if frame.empty:
        raise SupportAwarePairwiseObjectiveError("cannot build a pairwise objective from an empty frame")
    config.validate()
    if columns.query_id not in frame.columns:
        raise SupportAwarePairwiseObjectiveError(f"missing required training column {columns.query_id!r}")

    atr_residual = _numeric_column(frame, columns.atr_residual)
    bps_residual = _numeric_column(frame, columns.bps_residual)
    atr_grade = _numeric_column(frame, columns.atr_grade)
    incumbent = _numeric_column(frame, columns.incumbent_base_score)
    support = _support_column(frame, columns.support)
    query_values = frame[columns.query_id].to_numpy(copy=False)
    valid = (
        pd.notna(query_values)
        & np.isfinite(atr_residual)
        & np.isfinite(bps_residual)
        & np.isfinite(atr_grade)
        & np.isfinite(incumbent)
    )
    valid_positions = np.flatnonzero(valid).astype(np.int64, copy=False)
    query_codes, query_uniques = pd.factorize(query_values[valid], sort=False)

    winners: list[np.ndarray] = []
    losers: list[np.ndarray] = []
    pair_weights: list[np.ndarray] = []
    strict_flags: list[np.ndarray] = []
    support_classes: list[np.ndarray] = []
    misordered_flags: list[np.ndarray] = []
    selected_query_codes: list[np.ndarray] = []
    selected_pairs_by_query: list[tuple[str, int]] = []
    examined = loose_total = strict_total = eligible_queries = 0

    for query_code, query in enumerate(query_uniques):
        positions = valid_positions[query_codes == query_code]
        if len(positions) < 2:
            continue
        first, second = _sample_candidate_pairs(positions, query=query, config=config)
        examined += len(first)
        if not len(first):
            continue

        delta_atr = atr_residual[first] - atr_residual[second]
        delta_bps = bps_residual[first] - bps_residual[second]
        delta_grade = atr_grade[first] - atr_grade[second]
        same_direction = np.sign(delta_atr) == np.sign(delta_bps)
        nonzero_direction = (np.sign(delta_atr) != 0.0) & (np.sign(delta_bps) != 0.0)
        loose = (
            same_direction
            & nonzero_direction
            & (np.abs(delta_atr) >= config.loose_atr_separation)
            & (np.abs(delta_bps) >= config.loose_bps_separation)
        )
        strict = (
            loose
            & (np.abs(delta_atr) >= config.strict_atr_separation)
            & (np.abs(delta_bps) >= config.strict_bps_separation)
            & (np.abs(delta_grade) >= config.strict_grade_separation)
        )
        loose_total += int(loose.sum())
        strict_total += int(strict.sum())
        if not loose.any():
            continue

        first, second = first[loose], second[loose]
        delta_atr, delta_bps, delta_grade = delta_atr[loose], delta_bps[loose], delta_grade[loose]
        strict = strict[loose]
        winner_is_first = delta_atr > 0.0
        winner = np.where(winner_is_first, first, second).astype(np.int64, copy=False)
        loser = np.where(winner_is_first, second, first).astype(np.int64, copy=False)

        base_direction = incumbent[winner] - incumbent[loser]
        incumbent_misordered = base_direction <= 0.0
        if config.require_incumbent_misorder:
            keep = incumbent_misordered
            winner, loser, strict = winner[keep], loser[keep], strict[keep]
            delta_bps, delta_grade, base_direction = (
                delta_bps[keep], delta_grade[keep], base_direction[keep]
            )
            incumbent_misordered = incumbent_misordered[keep]
        if not len(winner):
            continue

        category, support_weight = _support_multiplier(
            support[winner], support[loser], config
        )
        strict_weight = np.where(
            strict, config.strict_pair_multiplier, config.loose_pair_multiplier
        )
        incumbent_weight = np.where(
            incumbent_misordered,
            config.incumbent_misordered_multiplier,
            config.incumbent_correctly_ordered_multiplier,
        )
        score_proximity = config.incumbent_score_proximity_floor + (
            1.0 - config.incumbent_score_proximity_floor
        ) * np.exp(-np.abs(base_direction) / config.incumbent_score_distance_scale)
        magnitude_weight = np.log1p(np.abs(delta_bps) / config.bps_gap_scale)
        grade_weight = 1.0 + config.atr_grade_gap_multiplier * np.abs(delta_grade)
        raw_weight = (
            strict_weight
            * support_weight
            * incumbent_weight
            * score_proximity
            * magnitude_weight
            * grade_weight
        )
        finite_positive = np.isfinite(raw_weight) & (raw_weight > 0.0)
        winner, loser, strict, category, incumbent_misordered, raw_weight = (
            winner[finite_positive], loser[finite_positive], strict[finite_positive],
            category[finite_positive], incumbent_misordered[finite_positive], raw_weight[finite_positive],
        )
        if not len(winner):
            continue
        # The sample itself is random but deterministic; retain its sampled
        # order, then cap.  This prevents score/label sorting from becoming an
        # unintended pair-selection rule.
        retained = min(len(winner), config.max_pairs_per_query)
        winner, loser, strict, category, incumbent_misordered, raw_weight = (
            winner[:retained], loser[:retained], strict[:retained], category[:retained],
            incumbent_misordered[:retained], raw_weight[:retained],
        )
        normalised_weight = config.query_total_weight * raw_weight / raw_weight.sum()
        winners.append(winner)
        losers.append(loser)
        pair_weights.append(normalised_weight.astype(np.float64, copy=False))
        strict_flags.append(strict.astype(bool, copy=False))
        support_classes.append(category.astype(object, copy=False))
        misordered_flags.append(incumbent_misordered.astype(bool, copy=False))
        selected_query_codes.append(np.full(retained, query_code, dtype=np.int32))
        selected_pairs_by_query.append((str(query), int(retained)))
        eligible_queries += 1

    def _concat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    audit = PairwiseObjectiveAudit(
        schema=SCHEMA,
        input_rows=int(len(frame)),
        valid_rows=int(valid.sum()),
        invalid_rows=int((~valid).sum()),
        queries=int(len(query_uniques)),
        eligible_queries=int(eligible_queries),
        skipped_queries=int(len(query_uniques) - eligible_queries),
        candidate_pairs_examined=int(examined),
        loose_pairs=int(loose_total),
        strict_pairs=int(strict_total),
        selected_pairs=int(sum(len(part) for part in winners)),
        selected_pairs_by_query=tuple(selected_pairs_by_query),
    )
    return PairwiseLogisticObjective(
        winner_rows=_concat(winners, np.int64),
        loser_rows=_concat(losers, np.int64),
        pair_weights=_concat(pair_weights, np.float64),
        is_strict=_concat(strict_flags, bool),
        support_class=_concat(support_classes, object),
        incumbent_misordered=_concat(misordered_flags, bool),
        query_codes=_concat(selected_query_codes, np.int32),
        row_count=len(frame),
        config=config,
        audit=audit,
    )


__all__ = [
    "SCHEMA",
    "PairwiseLogisticObjective",
    "PairwiseObjectiveAudit",
    "SupportAwarePairwiseColumns",
    "SupportAwarePairwiseConfig",
    "SupportAwarePairwiseObjectiveError",
    "build_support_aware_pairwise_objective",
]
