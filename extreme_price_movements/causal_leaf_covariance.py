"""Strictly causal, frozen-reference covariance diagnostics for leaf families.

This is intentionally a state builder, rather than a fitter or a selector.
For an evaluation block, every diagnostic is emitted *before* the row is
allowed to update any state.  The comparison reference is a snapshot made at
the start of that block, consequently neither the row nor any later row can
change its own output.  Leaf tokens are deliberately not part of this
contract: a leaf family is a human-declared, stable family label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


RAW_LEAF_MARKERS = ("leaf_id", "leafid", "leaf_token", "raw_leaf")
HORIZONS: tuple[int, int] = (24, 24 * 7)


class CausalLeafCovarianceError(ValueError):
    """Raised when the causal covariance input contract is not satisfied."""


@dataclass(frozen=True)
class CausalLeafCovarianceConfig:
    """Small, bounded numerical contract for covariance-reference state."""

    timestamp_col: str = "source_utc"
    evaluation_block_col: str = "evaluation_block"
    family_col: str = "family"
    side_col: str = "side_name"
    head_col: str = "head_name"
    horizons_hours: tuple[int, int] = HORIZONS
    max_fields_per_family: int = 15
    min_reference_rows: int = 2
    shrinkage_support: float = 12.0
    value_clip: float = 1_000_000.0
    covariance_epsilon: float = 1e-8
    distance_clip: float = 100.0

    def validate(self) -> None:
        if tuple(int(item) for item in self.horizons_hours) != HORIZONS:
            raise CausalLeafCovarianceError("horizons_hours must be exactly (24, 168)")
        if not 1 <= int(self.max_fields_per_family) <= 15:
            raise CausalLeafCovarianceError("max_fields_per_family must be in [1, 15]")
        if int(self.min_reference_rows) < 2:
            raise CausalLeafCovarianceError("min_reference_rows must be at least two")
        if not np.isfinite(self.shrinkage_support) or float(self.shrinkage_support) <= 0.0:
            raise CausalLeafCovarianceError("shrinkage_support must be finite and positive")
        if not np.isfinite(self.value_clip) or float(self.value_clip) <= 0.0:
            raise CausalLeafCovarianceError("value_clip must be finite and positive")


@dataclass(frozen=True)
class CausalLeafCovarianceResult:
    """Causal row diagnostics and the corresponding bounded field contract."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass
class _EWMCovariance:
    """Float64 EWMA first/second moments for one hierarchy member."""

    dimension: int
    mean: np.ndarray
    second: np.ndarray
    observations: int = 0
    last_timestamp_ns: int | None = None

    @classmethod
    def empty(cls, dimension: int) -> "_EWMCovariance":
        return cls(
            dimension=dimension,
            mean=np.zeros(dimension, dtype=np.float64),
            second=np.zeros((dimension, dimension), dtype=np.float64),
        )

    def copy(self) -> "_EWMCovariance":
        return _EWMCovariance(
            self.dimension, self.mean.copy(), self.second.copy(), self.observations,
            self.last_timestamp_ns,
        )

    def covariance(self, epsilon: float) -> np.ndarray | None:
        if self.observations < 2:
            return None
        result = self.second - np.outer(self.mean, self.mean)
        result = (result + result.T) * 0.5
        # Floating point cancellation can produce a tiny negative eigenvalue.
        diagonal = np.maximum(np.diag(result), float(epsilon))
        result[np.diag_indices_from(result)] = diagonal
        return result

    def update(self, value: np.ndarray, timestamp_ns: int, half_life_hours: int, clip: float) -> None:
        x = np.clip(value.astype(np.float64, copy=False), -float(clip), float(clip))
        if self.last_timestamp_ns is None:
            weight = 1.0
        else:
            elapsed_hours = max(0.0, (int(timestamp_ns) - self.last_timestamp_ns) / 3_600_000_000_000.0)
            weight = 1.0 - np.exp(-np.log(2.0) * elapsed_hours / float(half_life_hours))
            # Distinct ordered rows may share a timestamp.  They must still
            # enter subsequent rows' states without a zero-weight no-op.
            weight = max(weight, 1.0 / max(float(half_life_hours), 1.0))
        weight = float(np.clip(weight, 0.0, 1.0))
        self.mean *= 1.0 - weight
        self.mean += weight * x
        self.second *= 1.0 - weight
        self.second += weight * np.outer(x, x)
        self.observations += 1
        self.last_timestamp_ns = int(timestamp_ns)


def covariance_feature_names(prefix: str = "leaf_covariance") -> tuple[str, ...]:
    """Return the stable, compact diagnostic fields (14 total)."""

    names: list[str] = []
    for horizon in HORIZONS:
        stem = f"{prefix}__{horizon}h"
        names.extend((
            f"{stem}__weighted_covariance_distance",
            f"{stem}__weighted_correlation_distance",
            f"{stem}__effective_rank_shift",
            f"{stem}__principal_angle_proxy",
            f"{stem}__correlation_sign_flip_share",
        ))
    names.extend((
        f"{prefix}__family_weight",
        f"{prefix}__side_head_weight",
        f"{prefix}__global_weight",
        f"{prefix}__reference_support",
    ))
    return tuple(names)


def _has_raw_leaf_identifier(name: object) -> bool:
    value = str(name).lower()
    if value.startswith("base_reasoning__g1_leaf_assignment_count"):
        return False
    return any(marker in value for marker in RAW_LEAF_MARKERS)


def _require_utc(values: pd.Series, column: str) -> pd.Series:
    # ``pd.to_datetime(..., utc=True)`` silently treats naive timestamps as
    # UTC.  That is not an acceptable interpretation for a causal ordering
    # contract, so reject naive scalar values before normalising them.
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    else:
        try:
            raw = values.tolist()
            if any(pd.Timestamp(item).tzinfo is None for item in raw if not pd.isna(item)):
                raise CausalLeafCovarianceError(f"{column} must contain timezone-aware UTC timestamps")
        except (TypeError, ValueError) as exc:
            raise CausalLeafCovarianceError(f"{column} has invalid UTC timestamps") from exc
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise CausalLeafCovarianceError(f"{column} has invalid UTC timestamps")
    return parsed


def _validate_input(
    frame: pd.DataFrame, feature_columns: Sequence[str], config: CausalLeafCovarianceConfig,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.Series]:
    config.validate()
    fields = tuple(str(name) for name in feature_columns)
    if not fields:
        raise CausalLeafCovarianceError("at least one pre-entry feature field is required")
    if len(fields) > int(config.max_fields_per_family):
        raise CausalLeafCovarianceError("too many feature fields per family")
    if len(set(fields)) != len(fields):
        raise CausalLeafCovarianceError("feature fields must be unique")
    if any(_has_raw_leaf_identifier(name) for name in (*frame.columns, *fields)):
        raise CausalLeafCovarianceError("raw leaf identifiers are not accepted")
    required = (config.timestamp_col, config.evaluation_block_col, config.family_col, config.side_col, config.head_col, *fields)
    missing = [name for name in required if name not in frame]
    if missing:
        raise CausalLeafCovarianceError(f"input lacks required columns: {missing}")
    if frame.empty:
        return frame.copy(), fields, pd.Series([], dtype="datetime64[ns, UTC]")
    result = frame.copy()
    timestamps = _require_utc(result[config.timestamp_col], config.timestamp_col)
    if not timestamps.is_monotonic_increasing:
        raise CausalLeafCovarianceError("rows must be in non-decreasing UTC order")
    for column in (config.evaluation_block_col, config.family_col, config.side_col, config.head_col):
        values = result[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise CausalLeafCovarianceError(f"{column} must be non-empty")
        result[column] = values
    # Blocks must be contiguous.  Re-entering a block would make its reference
    # ambiguous and can allow an already evaluated row into a later snapshot.
    blocks = result[config.evaluation_block_col].tolist()
    seen: set[str] = set()
    previous: str | None = None
    for block in blocks:
        if block != previous:
            if block in seen:
                raise CausalLeafCovarianceError("evaluation blocks must be contiguous and chronological")
            seen.add(block)
            previous = block
    numeric = result.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise CausalLeafCovarianceError("pre-entry feature fields must not contain infinity")
    return result, fields, timestamps


def _state_key(level: str, family: str, side: str, head: str) -> Hashable:
    if level == "family":
        return family
    if level == "side_head":
        return (side, head)
    return "__global__"


def _matrix(state: _EWMCovariance | None, config: CausalLeafCovarianceConfig) -> np.ndarray | None:
    return None if state is None else state.covariance(float(config.covariance_epsilon))


def _hierarchical_matrices(
    current: dict[str, _EWMCovariance], reference: dict[str, _EWMCovariance], config: CausalLeafCovarianceConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[float, float, float], float]:
    """Support-aware family -> side/head -> global covariance shrinkage."""

    levels = ("family", "side_head", "global")
    available: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for level in levels:
        now, frozen = current.get(level), reference.get(level)
        current_cov, reference_cov = _matrix(now, config), _matrix(frozen, config)
        if current_cov is None or reference_cov is None:
            continue
        support = float(min(now.observations, frozen.observations))
        if support < int(config.min_reference_rows):
            continue
        available.append((level, current_cov, reference_cov, support))
    if not available:
        return None, None, (np.nan, np.nan, np.nan), np.nan

    # Sequential allocation preserves the intended hierarchy.  Strong family
    # support claims most of the mass; otherwise side/head and global absorb
    # the uncertainty.  Any residual always lands on global when available.
    weights = {level: 0.0 for level in levels}
    remaining = 1.0
    for level, _, _, support in available:
        claim = support / (support + float(config.shrinkage_support))
        weight = remaining * claim
        weights[level] += weight
        remaining -= weight
    if "global" in {level for level, *_ in available}:
        weights["global"] += remaining
    else:
        weights[available[-1][0]] += remaining
    current_cov = sum(weights[level] * now for level, now, _, _ in available)
    reference_cov = sum(weights[level] * frozen for level, _, frozen, _ in available)
    support = min(item[3] for item in available)
    return current_cov, reference_cov, (weights["family"], weights["side_head"], weights["global"]), support


def _correlation(covariance: np.ndarray, epsilon: float) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), epsilon))
    result = covariance / np.outer(scale, scale)
    result = np.clip((result + result.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def _effective_rank(covariance: np.ndarray, epsilon: float) -> float:
    values = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(values.sum())
    if total <= epsilon:
        return 0.0
    probability = values / total
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, epsilon)))))


def _diagnostics(current: np.ndarray, reference: np.ndarray, config: CausalLeafCovarianceConfig) -> tuple[float, float, float, float, float]:
    epsilon = float(config.covariance_epsilon)
    # Feature-wise inverse-scale weights stop a large-unit field from silently
    # dominating a covariance distance.  Correlation distance separately uses
    # the same weights, restricted to unique off-diagonal pairs.
    scale_weight = 1.0 / np.sqrt(np.maximum(np.diag(reference), epsilon))
    pair_weight = np.outer(scale_weight, scale_weight)
    covariance_delta = (current - reference) * pair_weight
    covariance_reference = reference * pair_weight
    covariance_distance = float(np.linalg.norm(covariance_delta, "fro") / max(np.linalg.norm(covariance_reference, "fro"), epsilon))
    current_corr, reference_corr = _correlation(current, epsilon), _correlation(reference, epsilon)
    upper = np.triu_indices_from(current, k=1)
    if upper[0].size:
        weights = pair_weight[upper]
        corr_distance = float(np.sqrt(np.average(np.square((current_corr - reference_corr)[upper]), weights=weights)))
        active = (np.abs(current_corr[upper]) > epsilon) & (np.abs(reference_corr[upper]) > epsilon)
        sign_flips = float(np.mean(np.sign(current_corr[upper][active]) != np.sign(reference_corr[upper][active]))) if active.any() else 0.0
    else:
        corr_distance, sign_flips = 0.0, 0.0
    current_rank, reference_rank = _effective_rank(current, epsilon), _effective_rank(reference, epsilon)
    rank_shift = (current_rank - reference_rank) / max(reference_rank, 1.0)
    current_vector = np.linalg.eigh(current)[1][:, -1]
    reference_vector = np.linalg.eigh(reference)[1][:, -1]
    cosine = float(np.clip(abs(np.dot(current_vector, reference_vector)), 0.0, 1.0))
    angle_proxy = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    return (
        float(np.clip(covariance_distance, 0.0, config.distance_clip)),
        float(np.clip(corr_distance, 0.0, 2.0)),
        float(np.clip(rank_shift, -1.0, config.distance_clip)),
        angle_proxy,
        float(np.clip(sign_flips, 0.0, 1.0)),
    )


def build_causal_leaf_covariance_state(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: CausalLeafCovarianceConfig = CausalLeafCovarianceConfig(),
    prefix: str = "leaf_covariance",
) -> CausalLeafCovarianceResult:
    """Emit pre-entry covariance-reference diagnostics for chronological rows.

    ``evaluation_block`` identifies an externally defined chronological block.
    Its frozen reference is fitted only from preceding blocks.  This function
    does not sort, fit a model, choose fields, write artifacts, or use targets.
    """

    panel, fields, timestamps = _validate_input(frame, feature_columns, config)
    names = covariance_feature_names(prefix)
    if panel.empty:
        for name in names:
            panel[name] = pd.Series(dtype=np.float32)
        return CausalLeafCovarianceResult(panel, names)

    values = panel.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    result = np.full((len(panel), len(names)), np.nan, dtype=np.float32)
    states: dict[int, dict[str, dict[Hashable, _EWMCovariance]]] = {
        horizon: {level: {} for level in ("family", "side_head", "global")}
        for horizon in HORIZONS
    }
    references: dict[int, dict[str, dict[Hashable, _EWMCovariance]]] | None = None
    active_block: str | None = None

    # Every row with one decision timestamp sees the same state.  This is
    # stricter than merely excluding the row itself: a contemporaneous asset
    # or duplicate-tree-collapsed family cannot leak into another candidate's
    # health field at that timestamp.  The caller has already proved the
    # order is non-decreasing, so timestamp batches are contiguous.
    position = 0
    while position < len(panel):
        timestamp_ns = int(timestamps.iloc[position].value)
        end = position + 1
        while end < len(panel) and int(timestamps.iloc[end].value) == timestamp_ns:
            end += 1

        block = str(panel.iloc[position][config.evaluation_block_col])
        if not panel.iloc[position:end][config.evaluation_block_col].astype(str).eq(block).all():
            raise CausalLeafCovarianceError(
                "one decision timestamp must belong to exactly one evaluation block"
            )
        if block != active_block:
            # Copying the tiny bounded moment states is the frozen-reference
            # boundary.  No row in this block exists in ``references``.
            references = {
                horizon: {
                    level: {key: state.copy() for key, state in by_key.items()}
                    for level, by_key in by_level.items()
                }
                for horizon, by_level in states.items()
            }
            active_block = block
        assert references is not None

        # First read the pre-timestamp state for *all* contemporaneous rows.
        # Do not update ``states`` in this loop.
        batch_keys: list[dict[str, Hashable]] = []
        for row in range(position, end):
            family = str(panel.iloc[row][config.family_col])
            side = str(panel.iloc[row][config.side_col])
            head = str(panel.iloc[row][config.head_col])
            keys = {
                level: _state_key(level, family, side, head)
                for level in ("family", "side_head", "global")
            }
            batch_keys.append(keys)
            for horizon_index, horizon in enumerate(HORIZONS):
                current = {
                    level: states[horizon][level].get(keys[level])
                    for level in keys
                }
                frozen = {
                    level: references[horizon][level].get(keys[level])
                    for level in keys
                }
                current = {
                    level: value for level, value in current.items()
                    if value is not None
                }
                frozen = {
                    level: value for level, value in frozen.items()
                    if value is not None
                }
                now, reference, weights, support = _hierarchical_matrices(
                    current, frozen, config
                )
                start = horizon_index * 5
                if now is not None and reference is not None:
                    result[row, start:start + 5] = np.asarray(
                        _diagnostics(now, reference, config), dtype=np.float32
                    )
                if horizon_index == 0:
                    result[row, 10:14] = np.asarray(
                        (*weights, support), dtype=np.float32
                    )

        # A partially observed vector cannot define a full covariance.  It is
        # intentionally skipped rather than imputed from a future estimate.
        # Crucially, this update begins only after every same-timestamp output
        # above has been issued.
        for row, keys in zip(range(position, end), batch_keys, strict=True):
            if not np.isfinite(values[row]).all():
                continue
            for horizon in HORIZONS:
                for level, key in keys.items():
                    state = states[horizon][level].get(key)
                    if state is None:
                        state = _EWMCovariance.empty(len(fields))
                        states[horizon][level][key] = state
                    state.update(
                        values[row], timestamp_ns, horizon, config.value_clip
                    )
        position = end

    for index, name in enumerate(names):
        panel[name] = result[:, index]
    return CausalLeafCovarianceResult(panel, names)


# A descriptive alias keeps callers from having to choose between the words
# "state" and "features"; both names preserve the same strict contract.
build_causal_leaf_covariance_features = build_causal_leaf_covariance_state


__all__ = [
    "CausalLeafCovarianceConfig",
    "CausalLeafCovarianceError",
    "CausalLeafCovarianceResult",
    "HORIZONS",
    "build_causal_leaf_covariance_features",
    "build_causal_leaf_covariance_state",
    "covariance_feature_names",
]
