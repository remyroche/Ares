"""Leakage-safe archetype-level residual-aware market-state models.

The causal market representation is stored once per ``side x timestamp`` to
avoid counting a synchronized move once per asset.  Model fitting is a separate
concern: each existing inference archetype receives its own AE/MLP/GMM fit over
the matching side's representation and its own train-only economic targets.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

try:  # pragma: no cover - optional at import time.
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

OUTCOME_COLUMNS = frozenset(
    {
        "ev_after_1pct",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "hit_probability",
        "selected_for_monitor",
        "outcomes_available",
    }
)

MARKET_PREFIXES = (
    "mkt_",
    "market_",
    "cross_asset_",
    "return_dispersion",
    "pct_assets_",
    "state_spectral_",
)

PORTABLE_ASSET_HINTS = (
    "oi_drawdown_from_peak",
    "oi_recovery_fraction",
    "bars_since_oi_",
    "bars_since_max_oi_",
    "oi_drop_acceleration",
    "oi_drop_deceleration",
    "price_down_oi_",
    "price_up_oi_",
    "price_recovery",
    "price_minus_oi_recovery",
    "funding_sign_persistence",
    "funding_crowding_release",
    "downside_deceleration",
    "volume_climax_decay",
    "range_climax_decay",
    "wick_recovery_intensity",
    "asset_minus_mkt_",
    "asset_liquidation_phase_score",
    "asset_flush_exhaustion_score",
    "asset_short_covering_score",
    "asset_mkt_",
    "rv_rel_universe",
    "log_quote_volume",
)

PHASE_STATE_PREFIX = "state_phase__"
PHASE_STATE_FEATURES = (
    "state_phase__liquidation_onset",
    "state_phase__liquidation_climax",
    "state_phase__flush_exhaustion",
    "state_phase__post_liquidation_rebound",
    "state_phase__leverage_rebuild",
    "state_phase__late_continuation_risk",
    "state_phase__synchronized_shock",
    "state_phase__onset_delta_1step",
    "state_phase__exhaustion_delta_1step",
    "state_phase__rebound_delta_1step",
)


@dataclass(frozen=True)
class StateVectorConfig:
    timestamp_col: str = "__ts__"
    symbol_col: str = "__symbol__"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    selected_col: str = "selected_for_monitor"
    score_col: str = "score_meta_base_soft_label"
    probability_col: str = "hit_probability"
    ev_col: str = "ev_after_1pct"
    hit_col: str = "clean_exec"
    bad_mae_col: str = "full_path_bad_mae_1r"
    timeout_col: str = "timeout"
    max_asset_features: int = 36
    max_market_features: int = 48
    min_feature_coverage: float = 0.65


@dataclass(frozen=True)
class ResidualAEConfig:
    latent_dim: int = 8
    hidden_dim: int = 48
    lambda_surprise: float = 0.05
    lambda_ev: float = 0.05
    lambda_asymmetry: float = 0.02
    epochs: int = 120
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.15
    patience: int = 15
    max_input_features: int = 384
    correlation_prune_threshold: float = 0.995
    torch_num_threads: int = 1
    random_state: int = 20260711


@dataclass(frozen=True)
class ResidualEncoderConfig:
    """Configuration for economically supervised global-state encoders.

    ``encoder_kind`` controls only the representation loss.  All variants use
    the same train-fitted preprocessing and emit the same latent/head contract.
    Outcome-derived signature columns are targets during fit and are never
    accepted as encoder inputs or required by :meth:`transform`.
    """

    encoder_kind: str = "hybrid_mlp"
    latent_dim: int = 12
    hidden_dims: tuple[int, ...] = (256, 128, 32)
    dropout: float = 0.10
    reconstruction_weight: float = 0.10
    signature_weight: float = 1.0
    epochs: int = 160
    batch_size: int = 512
    learning_rate: float = 7.5e-4
    weight_decay: float = 2e-4
    validation_fraction: float = 0.15
    patience: int = 18
    max_input_features: int = 384
    correlation_prune_threshold: float = 0.995
    torch_num_threads: int = 1
    random_state: int = 20260711


ENCODER_PRESETS: dict[str, dict[str, float]] = {
    "unsupervised_ae": {"reconstruction_weight": 1.0, "signature_weight": 0.0},
    "residual_aware_ae": {"reconstruction_weight": 1.0, "signature_weight": 0.10},
    "supervised_mlp": {"reconstruction_weight": 0.0, "signature_weight": 1.0},
    "hybrid_mlp": {"reconstruction_weight": 0.10, "signature_weight": 1.0},
}


def _signature_target_allowed_for_frame(name: str, frame: pd.DataFrame) -> bool:
    """Keep only targets belonging to the fitted side/archetype partition."""
    if not str(name).startswith("target_signature_") or "side_name" not in frame:
        return True
    sides = {
        str(value).strip().lower()
        for value in frame["side_name"].dropna().astype(str).unique()
        if str(value).strip()
    }
    if len(sides) != 1:
        return True
    side = next(iter(sides))
    if "archetype_policy_key" in frame:
        archetypes = {
            str(value).strip()
            for value in frame["archetype_policy_key"].dropna().astype(str).unique()
            if str(value).strip()
        }
        if len(archetypes) == 1:
            token = archetype_state_token(side, next(iter(archetypes)))
            return str(name).startswith(f"target_signature_arch__{token}_")
    return str(name).startswith(
        (
            "target_signature_global_",
            f"target_signature_{side}_",
            f"target_signature_arch__{side}_",
        )
    )


GLOBAL_RESIDUAL_SIGNATURE_BASES: tuple[str, ...] = (
    "signed_surprise",
    "positive_surprise",
    "negative_surprise",
    "mean_ev",
    "negative_ev",
    "clean_rate",
    "dirty_positive_rate",
    "bad_mae_rate",
    "timeout_rate",
    "payoff_asymmetry",
    "missed_clean_rate",
    "missed_positive_ev",
    "residual_archetype_dispersion",
    "adverse_archetype_concentration",
    "favorable_archetype_concentration",
)


@dataclass(frozen=True)
class GMMGridConfig:
    components: tuple[int, ...] = (4, 6, 8, 10, 12, 16)
    covariance_types: tuple[str, ...] = ("diag", "full")
    reg_covars: tuple[float, ...] = (1e-4, 1e-3)
    n_init: int = 3
    max_iter: int = 300
    min_component_occupancy: float = 0.01
    shrinkage_rows: float = 120.0
    min_enrichment_target_rows: int = 120
    random_state: int = 20260711


@dataclass(frozen=True)
class HMMGridConfig:
    """Search and filtering controls for the categorical GMM-state HMM."""

    hidden_states: tuple[int, ...] = (3, 4, 5)
    n_iter: int = 200
    tol: float = 1e-4
    min_state_occupancy: float = 0.01
    shrinkage_rows: float = 120.0
    max_contiguous_gap_hours: float = 2.0
    random_state: int = 20260740


class CausalCategoricalStateHMM:
    """Causal HMM over GMM state IDs with train-only economic enrichments."""

    enrichment_targets = (
        "target_signed_surprise",
        "target_positive_surprise",
        "target_negative_surprise",
        "target_negative_ev",
        "target_payoff_asymmetry",
    )

    def __init__(self, config: HMMGridConfig | None = None) -> None:
        self.config = config or HMMGridConfig()
        self.model: Any | None = None
        self.grid = pd.DataFrame()
        self.enrichments: dict[str, np.ndarray] = {}
        self.global_targets: dict[str, float] = {}
        self.fitted_enrichment_targets: tuple[str, ...] = ()
        self.final_filtered_probability: np.ndarray | None = None
        self.train_last_timestamp: pd.Timestamp | None = None

    def _sequence_boundaries(self, timestamps: pd.Series) -> np.ndarray:
        ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
        gap = ts.diff().dt.total_seconds().div(3600.0)
        return (
            gap.isna() | gap.gt(float(self.config.max_contiguous_gap_hours))
        ).to_numpy()

    def _filter(
        self,
        observations: np.ndarray,
        timestamps: pd.Series,
        *,
        initial: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Categorical HMM is not fitted")
        obs = np.asarray(observations, dtype=np.int64).reshape(-1)
        posterior = np.zeros((len(obs), int(self.model.n_components)), dtype=np.float64)
        starts = self._sequence_boundaries(timestamps)
        transition = np.asarray(self.model.transmat_, dtype=np.float64)
        emission = np.asarray(self.model.emissionprob_, dtype=np.float64)
        start_probability = np.asarray(self.model.startprob_, dtype=np.float64)
        previous = None if initial is None else np.asarray(initial, dtype=np.float64)
        for idx, category in enumerate(obs):
            if starts[idx]:
                prior = (
                    start_probability
                    if idx or previous is None
                    else previous @ transition
                )
            else:
                prior = posterior[idx - 1] @ transition
            category = int(np.clip(category, 0, emission.shape[1] - 1))
            filtered = prior * emission[:, category]
            denominator = float(filtered.sum())
            posterior[idx] = (
                filtered / denominator
                if denominator > 0.0
                else prior / max(float(prior.sum()), 1e-12)
            )
        return posterior.astype(np.float32)

    def fit(
        self,
        state_frame: pd.DataFrame,
        targets: pd.DataFrame,
        timestamps: pd.Series,
    ) -> "CausalCategoricalStateHMM":
        try:
            from hmmlearn.hmm import CategoricalHMM
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("hmmlearn is required for causal HMM fitting") from exc

        observations = (
            pd.to_numeric(state_frame["global_state_id"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
        sequence_starts = self._sequence_boundaries(timestamps)
        start_indices = np.flatnonzero(sequence_starts)
        lengths = np.diff(np.r_[start_indices, len(observations)]).astype(int).tolist()
        x = observations.reshape(-1, 1)
        category_count = int(observations.max()) + 1
        candidates: list[tuple[dict[str, Any], Any, np.ndarray]] = []
        for hidden_states in self.config.hidden_states:
            if len(observations) < int(hidden_states) * 50:
                continue
            model = CategoricalHMM(
                n_components=int(hidden_states),
                n_iter=int(self.config.n_iter),
                tol=float(self.config.tol),
                random_state=int(self.config.random_state),
                implementation="scaling",
            )
            model.n_features = category_count
            model.fit(x, lengths=lengths)
            self.model = model
            posterior = self._filter(observations, timestamps)
            occupancy = posterior.mean(axis=0)
            parameter_count = (
                int(hidden_states)
                - 1
                + int(hidden_states) * (int(hidden_states) - 1)
                + int(hidden_states) * (category_count - 1)
            )
            bic_per_row = float(
                (
                    -2.0 * model.score(x, lengths=lengths)
                    + parameter_count * math.log(len(x))
                )
                / len(x)
            )
            candidates.append(
                (
                    {
                        "hidden_states": int(hidden_states),
                        "bic_per_row": bic_per_row,
                        "min_occupancy": float(occupancy.min()),
                        "occupancy_entropy": float(
                            entropy(occupancy) / max(math.log(len(occupancy)), 1e-12)
                        ),
                        "converged": bool(getattr(model.monitor_, "converged", True)),
                    },
                    model,
                    posterior,
                )
            )
        if not candidates:
            raise ValueError("No valid categorical HMM candidates")
        grid = pd.DataFrame([row for row, _, _ in candidates])
        eligible = (
            grid["min_occupancy"].ge(float(self.config.min_state_occupancy))
            & grid["converged"]
        )
        chosen_index = (
            grid.loc[eligible, "bic_per_row"].idxmin()
            if eligible.any()
            else grid["bic_per_row"].idxmin()
        )
        _, self.model, posterior = candidates[int(chosen_index)]
        self.grid = grid.sort_values("bic_per_row", kind="stable").reset_index(
            drop=True
        )

        signature_targets = [
            name
            for name in targets.columns
            if name.startswith("target_signature_")
            and _signature_target_allowed_for_frame(name, targets)
            and pd.to_numeric(targets[name], errors="coerce").notna().any()
        ]
        self.fitted_enrichment_targets = tuple(
            dict.fromkeys(
                [name for name in self.enrichment_targets if name in targets]
                + signature_targets
            )
        )
        occupancy_rows = posterior.sum(axis=0)
        for target in self.fitted_enrichment_targets:
            values = pd.to_numeric(targets[target], errors="coerce").to_numpy(
                dtype=float
            )
            global_value = float(np.nanmean(values))
            local = _posterior_weighted_mean(posterior, values)
            strength = occupancy_rows / (
                occupancy_rows + float(self.config.shrinkage_rows)
            )
            self.enrichments[target] = (
                strength * local + (1.0 - strength) * global_value
            ).astype(np.float32)
            self.global_targets[target] = global_value
        self.final_filtered_probability = posterior[-1].copy()
        self.train_last_timestamp = pd.to_datetime(timestamps, utc=True).max()
        return self

    def transform(
        self,
        state_frame: pd.DataFrame,
        timestamps: pd.Series,
        *,
        continue_from_train: bool = False,
    ) -> pd.DataFrame:
        observations = (
            pd.to_numeric(state_frame["global_state_id"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
        initial = None
        if (
            continue_from_train
            and self.final_filtered_probability is not None
            and len(observations)
        ):
            first = pd.to_datetime(timestamps, utc=True).min()
            if self.train_last_timestamp is not None:
                gap_hours = (first - self.train_last_timestamp).total_seconds() / 3600.0
                if 0.0 <= gap_hours <= float(self.config.max_contiguous_gap_hours):
                    initial = self.final_filtered_probability
        posterior = self._filter(observations, timestamps, initial=initial)
        output = pd.DataFrame(
            posterior,
            columns=[
                f"global_hmm_posterior_{idx}" for idx in range(posterior.shape[1])
            ],
            index=state_frame.index,
        )
        output["global_hmm_state_id"] = posterior.argmax(axis=1).astype(np.int16)
        output["global_hmm_entropy"] = (
            -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1)
        ).astype(np.float32)
        for target, values in self.enrichments.items():
            output[f"global_hmm_expected_{target.removeprefix('target_')}"] = (
                posterior @ np.asarray(values, dtype=np.float32)
            ).astype(np.float32)
        return output

    def manifest(self) -> dict[str, Any]:
        if self.model is None:
            return {"fitted": False}
        return {
            "schema": "archetype_partition_causal_categorical_hmm_v1",
            "config": asdict(self.config),
            "selected_hidden_states": int(self.model.n_components),
            "global_targets": self.global_targets,
            "fitted_enrichment_targets": list(self.fitted_enrichment_targets),
            "inference_contract": (
                "Causal forward filtering only; OOS rows use frozen transition/emission "
                "parameters and train-derived enrichment priors."
            ),
        }


def select_state_features(
    frame: pd.DataFrame,
    config: StateVectorConfig | None = None,
) -> tuple[list[str], list[str]]:
    """Choose portable market and asset-relative coordinates from a handoff."""
    cfg = config or StateVectorConfig()
    numeric = frame.select_dtypes(include=[np.number, "bool"]).columns
    candidates: list[tuple[str, float]] = []
    for name in numeric:
        if name in OUTCOME_COLUMNS or name in {"row_id", cfg.score_col}:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        coverage = float(values.notna().mean())
        if coverage < cfg.min_feature_coverage or values.nunique(dropna=True) < 8:
            continue
        candidates.append((str(name), coverage))
    market = [
        name
        for name, _ in candidates
        if name.startswith(MARKET_PREFIXES)
        or any(prefix in name for prefix in MARKET_PREFIXES)
    ]
    asset = [
        name
        for name, _ in candidates
        if name not in market and any(hint in name for hint in PORTABLE_ASSET_HINTS)
    ]
    priority = {name: coverage for name, coverage in candidates}
    market = sorted(market, key=lambda name: (-priority[name], name))[
        : cfg.max_market_features
    ]
    asset = sorted(asset, key=lambda name: (-priority[name], name))[
        : cfg.max_asset_features
    ]
    return market, asset


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _state_coordinate(
    frame: pd.DataFrame,
    *names: str,
) -> np.ndarray:
    """Return the first populated aggregate coordinate without fitting a transform."""
    values = np.full(len(frame), np.nan, dtype=np.float32)
    for name in names:
        if name not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float32
        )
        missing = ~np.isfinite(values)
        values[missing] = candidate[missing]
    return values


def _bounded_positive(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.clip(
        np.maximum(np.asarray(values, dtype=np.float32), 0.0) / float(scale), 0.0, 3.0
    )


def _phase_average(*components: np.ndarray) -> np.ndarray:
    matrix = np.stack(components, axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(matrix)
    count = finite.sum(axis=1)
    total = np.nansum(matrix, axis=1, dtype=np.float32)
    return np.where(count >= 2, total / np.maximum(count, 1), np.nan).astype(np.float32)


def add_causal_phase_state_features(
    states: pd.DataFrame,
    *,
    timestamp_col: str = "__ts__",
    side_col: str = "side_name",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize fixed, pre-entry liquidation lifecycle coordinates.

    The formulas only combine contemporaneous or already-point-in-time feature
    store columns. No cross-period centering is fitted here: AE/GMM scalers are
    still train-only. This keeps the phase representation portable to frozen
    transforms while exposing the market transition that raw shock magnitudes
    alone often miss.
    """
    output = states.copy(deep=False)
    if output.empty:
        return output, {
            "schema": "causal_phase_state_features_v1",
            "features": list(PHASE_STATE_FEATURES),
            "source_contract": "empty",
        }

    def coordinate(name: str) -> np.ndarray:
        return _state_coordinate(
            output,
            f"full_universe__median__{name}",
            f"universe__median__{name}",
            f"selected__median__{name}",
            name,
        )

    oi_drop_4h = coordinate("mkt_median_oi_chg_4h_rz")
    oi_drop_1h = coordinate("mkt_median_oi_chg_1h_rz")
    oi_flush_breadth = coordinate("mkt_pct_oi_chg_4h_rz_lt_minus1")
    extreme_oi_flush_breadth = coordinate("mkt_pct_oi_chg_4h_rz_lt_minus2")
    oi_flush_accel = coordinate("mkt_oi_flush_breadth_accel_1h")
    oi_flush_recovery = coordinate("mkt_oi_flush_breadth_recovery_4h")
    downside_breadth = coordinate("mkt_pct_price_down_oi_down_4h")
    rebound_breadth = coordinate("mkt_pct_price_up_oi_down_1h")
    rebuild_breadth = coordinate("mkt_pct_price_up_oi_up_4h")
    breadth_change = coordinate("market_breadth_chg_1h")
    breadth_recovery = coordinate("market_breadth_recovery_from_6h_min")
    systemic = coordinate("mkt_systemic_deleveraging_score")
    exhaustion = coordinate("mkt_flush_exhaustion_score")
    rebuild = coordinate("mkt_leverage_rebuild_score")
    pc1_share = coordinate("market_pc1_variance_share_12h")
    asset_exhaustion = coordinate("asset_flush_exhaustion_score")
    asset_short_cover = coordinate("asset_short_covering_score")

    onset = _phase_average(
        _bounded_positive(-oi_drop_4h, 1.0),
        _bounded_positive(oi_flush_breadth, 0.25),
        _bounded_positive(downside_breadth, 0.25),
        _bounded_positive(-breadth_change, 0.10),
        _bounded_positive(systemic, 1.0),
    )
    climax = _phase_average(
        _bounded_positive(-oi_drop_1h, 1.0),
        _bounded_positive(extreme_oi_flush_breadth, 0.15),
        _bounded_positive(oi_flush_accel, 0.10),
        _bounded_positive(systemic, 1.0),
        _bounded_positive(pc1_share, 0.50),
    )
    flush_exhaustion = _phase_average(
        _bounded_positive(exhaustion, 1.0),
        _bounded_positive(oi_flush_recovery, 0.15),
        _bounded_positive(breadth_recovery, 0.15),
        _bounded_positive(rebound_breadth, 0.25),
        _bounded_positive(asset_exhaustion, 1.0),
    )
    rebound = _phase_average(
        flush_exhaustion,
        _bounded_positive(rebound_breadth, 0.25),
        _bounded_positive(asset_short_cover, 1.0),
        _bounded_positive(breadth_recovery, 0.15),
    )
    leverage_rebuild = _phase_average(
        _bounded_positive(rebuild, 1.0),
        _bounded_positive(rebuild_breadth, 0.25),
        _bounded_positive(oi_drop_4h, 1.0),
    )
    late_continuation_risk = _phase_average(
        flush_exhaustion,
        _bounded_positive(rebound_breadth, 0.25),
        _bounded_positive(systemic, 1.0),
    )
    synchronized = _phase_average(
        _bounded_positive(systemic, 1.0),
        _bounded_positive(downside_breadth, 0.25),
        _bounded_positive(pc1_share, 0.50),
    )
    phase_values = {
        "state_phase__liquidation_onset": onset,
        "state_phase__liquidation_climax": climax,
        "state_phase__flush_exhaustion": flush_exhaustion,
        "state_phase__post_liquidation_rebound": rebound,
        "state_phase__leverage_rebuild": leverage_rebuild,
        "state_phase__late_continuation_risk": late_continuation_risk,
        "state_phase__synchronized_shock": synchronized,
    }
    for name, values in phase_values.items():
        output[name] = values.astype(np.float32, copy=False)

    if timestamp_col in output.columns and side_col in output.columns:
        ordered = output.sort_values([side_col, timestamp_col], kind="stable")
        deltas = ordered.groupby(side_col, observed=True)[
            [
                "state_phase__liquidation_onset",
                "state_phase__flush_exhaustion",
                "state_phase__post_liquidation_rebound",
            ]
        ].diff()
        delta_names = (
            "state_phase__onset_delta_1step",
            "state_phase__exhaustion_delta_1step",
            "state_phase__rebound_delta_1step",
        )
        for source, destination in zip(deltas.columns, delta_names, strict=True):
            output.loc[ordered.index, destination] = np.clip(
                deltas[source].to_numpy(dtype=np.float32), -3.0, 3.0
            )

    return output, {
        "schema": "causal_phase_state_features_v1",
        "features": list(PHASE_STATE_FEATURES),
        "source_contract": (
            "Fixed current/past OI, price-OI, breadth, synchronization, and lifecycle "
            "coordinates only. Train-fitted scalers remain downstream in each side x archetype AE/GMM."
        ),
    }


def select_partition_state_features(
    frame: pd.DataFrame,
    requested: Sequence[str],
    *,
    max_features: int,
    min_coverage: float = 0.65,
    max_rows: int = 24_000,
    mi_bins: int = 8,
    max_views_per_source: int = 6,
) -> tuple[list[str], pd.DataFrame]:
    """Rank state coordinates for one archetype using train rows only.

    Relevance combines linear association with adverse/favorable signatures and
    nonlinear tail lift plus quantile-binned mutual information. MI is measured
    globally and in beginning/middle/end temporal thirds so a feature cannot win
    solely from one short regime. The calculation uses a time-spread row sample
    and returns continuous diagnostics rather than a hard state gate.
    """
    names = [
        str(name)
        for name in requested
        if str(name) in frame.columns
        and not str(name).startswith(("target_", "placebo_target_"))
    ]
    if not names:
        return [], pd.DataFrame()
    size = len(frame)
    sample_pos = np.arange(size, dtype=np.int64)
    if size > int(max_rows):
        sample_pos = np.unique(np.linspace(0, size - 1, int(max_rows), dtype=np.int64))
    sample = frame.iloc[sample_pos]
    targets: list[tuple[str, float]] = [
        ("target_negative_surprise", 0.20),
        ("target_negative_ev", 0.20),
        ("target_bad_mae_rate", 0.15),
        ("target_timeout_rate", 0.10),
        ("target_positive_surprise", 0.15),
        ("target_mean_ev", 0.15),
        ("target_payoff_asymmetry", 0.05),
    ]
    prepared_targets: list[tuple[str, float, np.ndarray, float, float]] = []
    for name, weight in targets:
        if name not in sample.columns:
            continue
        values = pd.to_numeric(sample[name], errors="coerce").to_numpy(dtype=np.float32)
        finite = values[np.isfinite(values)]
        if finite.size < 32 or float(np.nanstd(finite)) <= 1e-8:
            continue
        q25, q75 = np.nanquantile(finite, [0.25, 0.75])
        scale = max(float(q75 - q25), float(np.nanstd(finite)), 1e-6)
        prepared_targets.append(
            (name, weight, values, float(np.nanmean(finite)), scale)
        )
    if not prepared_targets:
        return names[: int(max_features)], pd.DataFrame(
            {"feature": names[: int(max_features)], "relevance": 0.0}
        )

    rows: list[dict[str, Any]] = []
    for name in names:
        values = pd.to_numeric(sample[name], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        coverage = float(finite.mean())
        if coverage < float(min_coverage):
            continue
        finite_values = values[finite]
        if finite_values.size < 32 or float(np.nanstd(finite_values)) <= 1e-8:
            continue
        lower, upper = np.nanquantile(finite_values, [0.15, 0.85])
        lower_mask = finite & (values <= lower)
        upper_mask = finite & (values >= upper)
        relevance = 0.0
        nonlinear = 0.0
        mi_relevance = 0.0
        details: dict[str, float] = {}
        for target_name, weight, target, target_mean, target_scale in prepared_targets:
            valid = finite & np.isfinite(target)
            if int(valid.sum()) < 32:
                continue
            x = values[valid].astype(np.float64, copy=False)
            y = target[valid].astype(np.float64, copy=False)
            x_std = float(np.std(x))
            y_std = float(np.std(y))
            corr = (
                float(np.corrcoef(x, y)[0, 1])
                if x_std > 1e-10 and y_std > 1e-10
                else 0.0
            )
            high = target[upper_mask & np.isfinite(target)]
            low = target[lower_mask & np.isfinite(target)]
            tail_lift = 0.0
            if high.size >= 8 and low.size >= 8:
                tail_lift = float((np.nanmean(high) - np.nanmean(low)) / target_scale)
            binned_mi = _normalized_binned_mutual_information(
                values[valid], target[valid], bins=int(mi_bins)
            )
            temporal_mi: list[float] = []
            valid_positions = np.flatnonzero(valid)
            for positions in np.array_split(valid_positions, 3):
                if len(positions) < 64:
                    continue
                temporal_mi.append(
                    _normalized_binned_mutual_information(
                        values[positions], target[positions], bins=int(mi_bins)
                    )
                )
            stable_mi = binned_mi
            if temporal_mi:
                stable_mi = float(
                    0.50 * binned_mi
                    + 0.30 * np.mean(temporal_mi)
                    + 0.20 * np.min(temporal_mi)
                )
            score = 0.25 * abs(corr) + 0.25 * abs(tail_lift) + 0.50 * stable_mi
            relevance += float(weight) * score
            nonlinear += float(weight) * (0.40 * abs(tail_lift) + 0.60 * stable_mi)
            mi_relevance += float(weight) * stable_mi
            details[f"corr__{target_name}"] = corr
            details[f"tail_lift__{target_name}"] = tail_lift
            details[f"binned_mi__{target_name}"] = binned_mi
            details[f"stable_binned_mi__{target_name}"] = stable_mi
        if not details:
            continue
        rows.append(
            {
                "feature": name,
                "coverage": coverage,
                "relevance": float(relevance),
                "nonlinear_relevance": float(nonlinear),
                "mi_relevance": float(mi_relevance),
                "phase_feature": bool(name.startswith(PHASE_STATE_PREFIX)),
                **details,
            }
        )
    diagnostic = pd.DataFrame(rows)
    if diagnostic.empty:
        return [], diagnostic
    diagnostic = diagnostic.sort_values(
        ["relevance", "mi_relevance", "nonlinear_relevance", "coverage", "feature"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    selected: list[str] = []
    source_counts: dict[str, int] = {}
    for feature in diagnostic["feature"]:
        source = _state_feature_source(str(feature))
        if source_counts.get(source, 0) >= int(max_views_per_source):
            continue
        selected.append(str(feature))
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= int(max_features):
            break
    selected_rank = {name: rank + 1 for rank, name in enumerate(selected)}
    diagnostic["selected"] = diagnostic["feature"].isin(selected_rank)
    diagnostic["selected_rank"] = diagnostic["feature"].map(selected_rank)
    return selected, diagnostic


def _quantile_codes(values: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.full(len(values), -1, dtype=np.int16)
    if int(finite.sum()) < 32:
        return output, 0
    quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.unique(np.quantile(values[finite], quantiles))
    if len(edges) < 3:
        return output, 0
    output[finite] = np.searchsorted(edges[1:-1], values[finite], side="right")
    return output, int(len(edges) - 1)


def _normalized_binned_mutual_information(
    feature: np.ndarray,
    target: np.ndarray,
    *,
    bins: int = 8,
) -> float:
    """Normalized discrete MI using robust quantile bins and NumPy only."""
    x_codes, x_bins = _quantile_codes(feature, bins)
    y_codes, y_bins = _quantile_codes(target, bins)
    valid = (x_codes >= 0) & (y_codes >= 0)
    count = int(valid.sum())
    if count < 64 or x_bins < 2 or y_bins < 2:
        return 0.0
    joint = (
        np.bincount(
            x_codes[valid].astype(np.int64) * y_bins + y_codes[valid].astype(np.int64),
            minlength=x_bins * y_bins,
        )
        .reshape(x_bins, y_bins)
        .astype(np.float64)
    )
    joint /= float(count)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    populated = joint > 0.0
    mi = float(
        np.sum(joint[populated] * np.log(joint[populated] / expected[populated]))
    )
    target_probability = py.reshape(-1)
    target_probability = target_probability[target_probability > 0.0]
    target_entropy = float(-np.sum(target_probability * np.log(target_probability)))
    return float(np.clip(mi / max(target_entropy, 1e-12), 0.0, 1.0))


def _state_feature_source(name: str) -> str:
    """Collapse aggregate views so one raw concept cannot consume the screen."""
    source = str(name)
    source = re.sub(
        r"^(?:full_universe|universe|selected)__(?:median|q10|q90|std|coverage)__",
        "",
        source,
    )
    source = re.sub(r"^selected_minus_universe__", "", source)
    return source


def _aggregate_features(
    frame: pd.DataFrame,
    keys: Sequence[str],
    features: Sequence[str],
    prefix: str,
    *,
    include_quantiles: bool,
) -> pd.DataFrame:
    if not features:
        return frame[list(keys)].drop_duplicates().reset_index(drop=True)
    grouped = frame.groupby(list(keys), observed=True, sort=True)
    median = grouped[list(features)].median().add_prefix(f"{prefix}median__")
    coverage = (
        grouped[list(features)]
        .count()
        .div(grouped.size(), axis=0)
        .add_prefix(f"{prefix}coverage__")
    )
    output = [median, coverage]
    if include_quantiles:
        output.extend(
            [
                grouped[list(features)].quantile(0.10).add_prefix(f"{prefix}q10__"),
                grouped[list(features)].quantile(0.90).add_prefix(f"{prefix}q90__"),
                grouped[list(features)].std(ddof=0).add_prefix(f"{prefix}std__"),
            ]
        )
    return pd.concat(output, axis=1).reset_index()


def _payoff_asymmetry(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    winners = numeric[numeric > 0.0]
    losers = numeric[numeric < 0.0]
    if winners.empty or losers.empty:
        return np.nan
    return float(winners.mean() + losers.mean())


def _mass_concentration(values: pd.Series) -> float:
    mass = pd.to_numeric(values, errors="coerce").clip(lower=0.0).dropna()
    total = float(mass.sum())
    if total <= 1e-12:
        return 0.0
    shares = mass.to_numpy(dtype=np.float64) / total
    return float(np.sum(shares * shares))


def _archetype_token(side: str, archetype: str) -> str:
    return (
        re.sub(r"[^a-zA-Z0-9]+", "_", f"{str(side).lower()}__{str(archetype)}")
        .strip("_")
        .lower()
    )


def archetype_state_token(side: str, archetype: str) -> str:
    """Stable feature-key token shared by train and inference materializers."""
    return _archetype_token(side, archetype)


def _single_state_partition(frame: pd.DataFrame) -> dict[str, str]:
    """Return and validate the one side x archetype represented by ``frame``."""
    required = {"side_name", "archetype_policy_key"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Partition-local state fitting/transform requires columns {missing}"
        )
    sides = sorted(
        {
            str(value).strip().lower()
            for value in frame["side_name"].dropna().astype(str)
            if str(value).strip()
        }
    )
    archetypes = sorted(
        {
            str(value).strip()
            for value in frame["archetype_policy_key"].dropna().astype(str)
            if str(value).strip()
        }
    )
    if len(sides) != 1 or len(archetypes) != 1:
        raise ValueError(
            "AE/MLP/GMM state models must fit and transform exactly one "
            f"side x archetype partition; sides={sides}, archetypes={archetypes}"
        )
    return {
        "side_name": sides[0],
        "archetype_policy_key": archetypes[0],
        "token": archetype_state_token(sides[0], archetypes[0]),
    }


def _validate_state_partition(
    frame: pd.DataFrame,
    fitted_partition: dict[str, str] | None,
) -> dict[str, str]:
    if fitted_partition is None:
        raise RuntimeError(
            "State model has no frozen side x archetype identity; refit it with "
            "the partition-local contract"
        )
    current = _single_state_partition(frame)
    if fitted_partition is not None and current != fitted_partition:
        raise ValueError(
            "State model partition mismatch: "
            f"fitted={fitted_partition['token']}, received={current['token']}"
        )
    return current


def _finite_column_center_scale(
    values: np.ndarray,
    *,
    minimum_scale: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust target transforms without warnings for unsupported local heads."""
    center = np.zeros(values.shape[1], dtype=np.float32)
    scale = np.ones(values.shape[1], dtype=np.float32)
    for index in range(values.shape[1]):
        finite = values[np.isfinite(values[:, index]), index]
        if not len(finite):
            continue
        center[index] = np.float32(np.median(finite))
        q25, q75 = np.quantile(finite, (0.25, 0.75))
        scale[index] = np.float32(max(float(q75 - q25), minimum_scale))
    return center, scale


def prepare_archetype_state_partition(
    states: pd.DataFrame,
    *,
    side: str,
    archetype: str,
) -> pd.DataFrame:
    """Route one side-level market-state sequence to one archetype model.

    The observable state geometry remains one row per side x timestamp.  This
    function exposes only the requested archetype's train-only residual targets
    before fitting its independent scaler, encoder, and GMM.
    """
    side_name = str(side).strip().lower()
    archetype_name = str(archetype).strip()
    token = archetype_state_token(side_name, archetype_name)
    prefix = f"target_signature_arch__{token}_"
    local = states.loc[states["side_name"].astype(str).str.lower().eq(side_name)].copy()
    signature_columns = [
        name for name in local.columns if name.startswith("target_signature_")
    ]
    local_targets = [name for name in signature_columns if name.startswith(prefix)]
    if not local_targets:
        raise ValueError(
            f"No residual signature targets found for side/archetype partition {token}"
        )
    local = local.drop(
        columns=[name for name in signature_columns if name not in local_targets],
        errors="ignore",
    )
    target_map = {
        "target_signed_surprise": f"{prefix}signed_surprise",
        "target_positive_surprise": f"{prefix}positive_surprise",
        "target_negative_surprise": f"{prefix}negative_surprise",
        "target_mean_ev": f"{prefix}mean_ev",
        "target_negative_ev": f"{prefix}negative_ev",
        "target_payoff_asymmetry": f"{prefix}payoff_asymmetry",
        "target_bad_mae_rate": f"{prefix}bad_mae_rate",
        "target_timeout_rate": f"{prefix}timeout_rate",
    }
    for target, source in target_map.items():
        if source in local:
            local[target] = pd.to_numeric(local[source], errors="coerce").astype(
                np.float32
            )
    local["archetype_policy_key"] = archetype_name
    local["state_partition_token"] = token
    return local


def _archetype_surprise_persistence_signature(
    selected: pd.DataFrame,
    *,
    timestamp_col: str,
    side_col: str,
    archetype_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Create per-archetype signed autocorrelation-contribution targets.

    Targets at day ``t`` combine the realized surprise resolving on ``t`` with
    already-earlier daily surprises.  They therefore do not extend the label
    horizon beyond the current trade outcome and never become inference inputs.
    """
    if selected.empty or archetype_col not in selected.columns:
        return pd.DataFrame(columns=["__signature_day__"]), []
    daily = selected.assign(
        __signature_day__=pd.to_datetime(
            selected[timestamp_col], utc=True, errors="coerce"
        ).dt.floor("D")
    )
    daily = (
        daily.groupby(
            ["__signature_day__", side_col, archetype_col],
            observed=True,
            sort=True,
        )
        .agg(
            signed_surprise=("_signature_signed_surprise", "mean"),
            support=("_signature_signed_surprise", "count"),
        )
        .reset_index()
        .sort_values([side_col, archetype_col, "__signature_day__"], kind="stable")
    )
    keys = [side_col, archetype_col]
    enriched_groups: list[pd.DataFrame] = []
    for _, local in daily.groupby(keys, observed=True, sort=False):
        local = local.sort_values("__signature_day__", kind="stable").copy()
        observed_days = pd.DatetimeIndex(local["__signature_day__"])
        calendar = pd.date_range(observed_days.min(), observed_days.max(), freq="D")
        surprise = pd.Series(
            pd.to_numeric(local["signed_surprise"], errors="coerce").to_numpy(),
            index=observed_days,
            dtype=np.float64,
        ).reindex(calendar)
        lag1 = surprise.shift(1)
        current = surprise
        local_calendar = pd.DataFrame(index=calendar)
        local_calendar["signed_autocov_2d"] = current * lag1
        local_calendar["positive_persistence_2d"] = current.clip(lower=0.0) * lag1.clip(
            lower=0.0
        )
        local_calendar["negative_persistence_2d"] = (-current).clip(lower=0.0) * (
            -lag1
        ).clip(lower=0.0)

        # The primary persistence targets compare today's surprise with the
        # prior seven calendar days for the same side x archetype.  The shift
        # excludes day t, and separate positive/negative histories prevent
        # favorable and adverse persistence from cancelling each other.
        prior_signed = surprise.shift(1).rolling(7, min_periods=3).mean()
        prior_positive = (
            surprise.clip(lower=0.0).shift(1).rolling(7, min_periods=3).mean()
        )
        prior_negative = (
            (-surprise).clip(lower=0.0).shift(1).rolling(7, min_periods=3).mean()
        )
        local_calendar["signed_alignment_prev7d"] = current * prior_signed
        local_calendar["positive_persistence_prev7d"] = (
            current.clip(lower=0.0) * prior_positive
        )
        local_calendar["negative_persistence_prev7d"] = (-current).clip(
            lower=0.0
        ) * prior_negative
        values = local_calendar.reindex(observed_days).reset_index(drop=True)
        for name in values:
            local[name] = values[name].to_numpy(dtype=np.float64)
        enriched_groups.append(local)
    daily = pd.concat(enriched_groups, ignore_index=True, sort=False)
    metrics = (
        "signed_alignment_prev7d",
        "positive_persistence_prev7d",
        "negative_persistence_prev7d",
        "signed_autocov_2d",
        "positive_persistence_2d",
        "negative_persistence_2d",
    )
    output: pd.DataFrame | None = None
    components: list[str] = []
    for (side, archetype), local in daily.groupby(keys, observed=True, sort=True):
        token = _archetype_token(str(side), str(archetype))
        components.append(token)
        columns = {
            metric: f"target_signature_arch__{token}_{metric}" for metric in metrics
        }
        part = local[["__signature_day__", *metrics]].rename(columns=columns)
        output = (
            part
            if output is None
            else output.merge(
                part, on="__signature_day__", how="outer", validate="one_to_one"
            )
        )
    if output is None:
        output = pd.DataFrame(columns=["__signature_day__"])
    return output, components


def _residual_signature_scope(
    selected: pd.DataFrame,
    missed: pd.DataFrame,
    *,
    timestamp_col: str,
    archetype_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate a train-label residual signature without rolling outcomes."""
    if selected.empty:
        return pd.DataFrame(columns=[timestamp_col])
    grouped = selected.groupby(timestamp_col, observed=True, sort=True)
    output = grouped.agg(
        signed_surprise=("_signature_signed_surprise", "mean"),
        positive_surprise=("_signature_positive_surprise", "mean"),
        negative_surprise=("_signature_negative_surprise", "mean"),
        mean_ev=("_signature_ev", "mean"),
        negative_ev=("_signature_negative_ev", "mean"),
        clean_rate=("_signature_clean", "mean"),
        dirty_positive_rate=("_signature_dirty", "mean"),
        bad_mae_rate=("_signature_bad_mae", "mean"),
        timeout_rate=("_signature_timeout", "mean"),
        payoff_asymmetry=("_signature_ev", _payoff_asymmetry),
    )
    if missed.empty:
        output["missed_clean_rate"] = np.nan
        output["missed_positive_ev"] = np.nan
    else:
        missed_grouped = missed.groupby(timestamp_col, observed=True, sort=True).agg(
            missed_clean_rate=("_signature_clean", "mean"),
            missed_positive_ev=("_signature_positive_ev", "mean"),
        )
        output = output.join(missed_grouped, how="left")

    if archetype_col in selected.columns:
        local = (
            selected.groupby([timestamp_col, archetype_col], observed=True, sort=True)
            .agg(
                archetype_residual=("_signature_signed_surprise", "mean"),
                adverse_mass=("_signature_adverse_mass", "sum"),
                favorable_mass=("_signature_favorable_mass", "sum"),
            )
            .reset_index()
        )
        archetype = local.groupby(timestamp_col, observed=True, sort=True).agg(
            residual_archetype_dispersion=(
                "archetype_residual",
                lambda values: float(
                    pd.to_numeric(values, errors="coerce").std(ddof=0)
                ),
            ),
            adverse_archetype_concentration=("adverse_mass", _mass_concentration),
            favorable_archetype_concentration=("favorable_mass", _mass_concentration),
        )
        output = output.join(archetype, how="left")
    else:
        output["residual_archetype_dispersion"] = np.nan
        output["adverse_archetype_concentration"] = np.nan
        output["favorable_archetype_concentration"] = np.nan

    output = output.rename(
        columns={name: f"target_signature_{prefix}_{name}" for name in output.columns}
    )
    return output.reset_index()


def build_global_residual_signature(
    candidate_rows: pd.DataFrame,
    config: StateVectorConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one global timestamp signature with overall and side components.

    The signature is computed from realized training outcomes.  It is suitable
    as an encoder target or state-enrichment label, but never as a direct OOS
    feature.  Existing archetypes contribute only concentration/dispersion
    summaries; they do not define separate market-state models.
    """
    cfg = config or StateVectorConfig()
    frame = candidate_rows.copy(deep=False)
    frame[cfg.timestamp_col] = pd.to_datetime(
        frame[cfg.timestamp_col], utc=True, errors="coerce"
    )
    frame = frame.loc[frame[cfg.timestamp_col].notna()].copy()
    selected_mask = frame.get(cfg.selected_col, False)
    if not isinstance(selected_mask, pd.Series):
        selected_mask = pd.Series(False, index=frame.index)
    selected_mask = selected_mask.fillna(False).astype(bool)

    frame["_signature_ev"] = _numeric(frame, cfg.ev_col)
    frame["_signature_signed_surprise"] = _numeric(frame, cfg.hit_col) - _numeric(
        frame, cfg.probability_col
    )
    frame["_signature_positive_surprise"] = frame["_signature_signed_surprise"].clip(
        lower=0.0
    )
    frame["_signature_negative_surprise"] = (-frame["_signature_signed_surprise"]).clip(
        lower=0.0
    )
    frame["_signature_positive_ev"] = frame["_signature_ev"].clip(lower=0.0)
    frame["_signature_negative_ev"] = (-frame["_signature_ev"]).clip(lower=0.0)
    frame["_signature_clean"] = _numeric(frame, cfg.hit_col)
    frame["_signature_bad_mae"] = _numeric(frame, cfg.bad_mae_col)
    frame["_signature_timeout"] = _numeric(frame, cfg.timeout_col)
    if "dirty_positive" in frame.columns:
        frame["_signature_dirty"] = _numeric(frame, "dirty_positive")
    else:
        frame["_signature_dirty"] = (
            frame["_signature_ev"].gt(0.0)
            & (
                frame["_signature_bad_mae"].gt(0.5)
                | frame["_signature_timeout"].gt(0.5)
            )
        ).astype(np.float32)
    frame["_signature_adverse_mass"] = (
        frame["_signature_negative_surprise"].fillna(0.0)
        + frame["_signature_negative_ev"].fillna(0.0)
        + frame["_signature_bad_mae"].fillna(0.0)
        + frame["_signature_timeout"].fillna(0.0)
    )
    frame["_signature_favorable_mass"] = (
        frame["_signature_positive_surprise"].fillna(0.0)
        + frame["_signature_positive_ev"].fillna(0.0)
        + frame["_signature_clean"].fillna(0.0)
    )
    selected = frame.loc[selected_mask]
    missed = frame.loc[~selected_mask]
    parts = [
        _residual_signature_scope(
            selected,
            missed,
            timestamp_col=cfg.timestamp_col,
            archetype_col=cfg.archetype_col,
            prefix="global",
        )
    ]
    side_values = sorted(frame[cfg.side_col].dropna().astype(str).str.lower().unique())
    for side in side_values:
        parts.append(
            _residual_signature_scope(
                selected.loc[selected[cfg.side_col].astype(str).str.lower().eq(side)],
                missed.loc[missed[cfg.side_col].astype(str).str.lower().eq(side)],
                timestamp_col=cfg.timestamp_col,
                archetype_col=cfg.archetype_col,
                prefix=side,
            )
        )
    archetype_components: list[str] = []
    if cfg.archetype_col in frame.columns:
        combinations = (
            frame[[cfg.side_col, cfg.archetype_col]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values([cfg.side_col, cfg.archetype_col], kind="stable")
        )
        for side_raw, archetype_raw in combinations.itertuples(index=False, name=None):
            side = str(side_raw).lower()
            archetype = str(archetype_raw)
            selected_local = selected[cfg.side_col].astype(str).str.lower().eq(side)
            selected_local &= selected[cfg.archetype_col].astype(str).eq(archetype)
            missed_local = missed[cfg.side_col].astype(str).str.lower().eq(side)
            missed_local &= missed[cfg.archetype_col].astype(str).eq(archetype)
            if not bool(selected_local.any()):
                continue
            token = _archetype_token(side, archetype)
            prefix = f"arch__{token}"
            archetype_components.append(prefix)
            parts.append(
                _residual_signature_scope(
                    selected.loc[selected_local],
                    missed.loc[missed_local],
                    timestamp_col=cfg.timestamp_col,
                    archetype_col=cfg.archetype_col,
                    prefix=prefix,
                )
            )
    signature = parts[0]
    for part in parts[1:]:
        signature = signature.merge(
            part, on=cfg.timestamp_col, how="outer", validate="one_to_one"
        )
    persistence, persistence_components = _archetype_surprise_persistence_signature(
        selected,
        timestamp_col=cfg.timestamp_col,
        side_col=cfg.side_col,
        archetype_col=cfg.archetype_col,
    )
    if not persistence.empty:
        signature["__signature_day__"] = pd.to_datetime(
            signature[cfg.timestamp_col], utc=True, errors="coerce"
        ).dt.floor("D")
        signature = signature.merge(
            persistence,
            on="__signature_day__",
            how="left",
            validate="many_to_one",
        ).drop(columns="__signature_day__")
    signature = signature.sort_values(cfg.timestamp_col, kind="stable").reset_index(
        drop=True
    )
    manifest = {
        "schema": "global_residual_signature_v1",
        "timestamp_col": cfg.timestamp_col,
        "target_columns": [
            name for name in signature if name.startswith("target_signature_")
        ],
        "side_components": side_values,
        "side_archetype_components": archetype_components,
        "surprise_persistence_components": persistence_components,
        "surprise_persistence_contract": (
            "Per side x archetype primary targets measure today's magnitude-weighted "
            "signed/positive/negative surprise alignment with the shifted mean of the prior "
            "seven calendar days (minimum three observed days). Two-day products are retained "
            "as fast-onset auxiliary targets. "
            "Lagged surprises are target construction only, never inference inputs."
        ),
        "selected_population": cfg.selected_col,
        "definition": (
            "Timestamp-level realized champion residual signature across assets and archetypes. "
            "It is a train-only encoder target, not a recent-performance or inference feature."
        ),
    }
    return signature, manifest


def build_side_timestamp_states(
    candidate_rows: pd.DataFrame,
    market_features: Sequence[str],
    asset_features: Sequence[str],
    config: StateVectorConfig | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Collapse candidate rows to one causal market-state observation per side/hour."""
    cfg = config or StateVectorConfig()
    frame = candidate_rows.copy(deep=False)
    frame[cfg.timestamp_col] = pd.to_datetime(
        frame[cfg.timestamp_col], utc=True, errors="coerce"
    )
    frame = frame.loc[frame[cfg.timestamp_col].notna()].copy()
    available_market = [name for name in market_features if name in frame.columns]
    available_asset = [name for name in asset_features if name in frame.columns]
    for name in available_market + available_asset:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(np.float32)
    keys = [cfg.timestamp_col, cfg.side_col]
    universe = _aggregate_features(
        frame,
        keys,
        available_market + available_asset,
        "universe__",
        include_quantiles=True,
    )
    selected_mask = frame.get(cfg.selected_col, False)
    if not isinstance(selected_mask, pd.Series):
        selected_mask = pd.Series(False, index=frame.index)
    selected = frame.loc[selected_mask.fillna(False).astype(bool)].copy()
    selected_agg = _aggregate_features(
        selected,
        keys,
        available_market + available_asset,
        "selected__",
        include_quantiles=False,
    )
    states = universe.merge(selected_agg, on=keys, how="left", validate="one_to_one")

    counts = (
        frame.groupby(keys, observed=True)
        .agg(
            universe_rows=(cfg.timestamp_col, "size"),
            universe_assets=(cfg.symbol_col, "nunique"),
            universe_score_mean=(cfg.score_col, "mean"),
            universe_score_std=(cfg.score_col, "std"),
        )
        .reset_index()
    )
    states = states.merge(counts, on=keys, how="left", validate="one_to_one")
    if not selected.empty:
        selected_counts = (
            selected.groupby(keys, observed=True)
            .agg(
                selected_rows=(cfg.timestamp_col, "size"),
                selected_assets=(cfg.symbol_col, "nunique"),
                selected_score_mean=(cfg.score_col, "mean"),
                selected_score_std=(cfg.score_col, "std"),
            )
            .reset_index()
        )
        states = states.merge(
            selected_counts, on=keys, how="left", validate="one_to_one"
        )
    else:
        states["selected_rows"] = 0
        states["selected_assets"] = 0

    archetype_mix = pd.DataFrame(index=states.set_index(keys).index)
    if not selected.empty and cfg.archetype_col in selected.columns:
        mix = pd.crosstab(
            [selected[cfg.timestamp_col], selected[cfg.side_col]],
            selected[cfg.archetype_col].astype(str),
            normalize="index",
        ).add_prefix("selected_archetype_share__")
        archetype_mix = mix
        states = states.set_index(keys).join(mix, how="left").reset_index()

    # Selected-minus-universe deltas express whether the traded slice is unusual.
    for name in available_market + available_asset:
        selected_name = f"selected__median__{name}"
        universe_name = f"universe__median__{name}"
        if selected_name in states and universe_name in states:
            states[f"selected_minus_universe__{name}"] = (
                pd.to_numeric(states[selected_name], errors="coerce")
                - pd.to_numeric(states[universe_name], errors="coerce")
            ).astype(np.float32)

    # Realized targets are retained separately and never included in state_features.
    if not selected.empty:
        selected["_signed_surprise"] = _numeric(selected, cfg.hit_col) - _numeric(
            selected, cfg.probability_col
        )
        selected["_positive_ev"] = _numeric(selected, cfg.ev_col).clip(lower=0.0)
        selected["_negative_ev"] = (-_numeric(selected, cfg.ev_col)).clip(lower=0.0)

        targets = (
            selected.groupby(keys, observed=True)
            .agg(
                target_signed_surprise=("_signed_surprise", "mean"),
                target_positive_surprise=(
                    "_signed_surprise",
                    lambda values: float(np.maximum(values, 0.0).mean()),
                ),
                target_negative_surprise=(
                    "_signed_surprise",
                    lambda values: float(np.maximum(-values, 0.0).mean()),
                ),
                target_mean_ev=(cfg.ev_col, "mean"),
                target_negative_ev=("_negative_ev", "mean"),
                target_payoff_asymmetry=(cfg.ev_col, _payoff_asymmetry),
                target_bad_mae_rate=(cfg.bad_mae_col, "mean"),
                target_timeout_rate=(cfg.timeout_col, "mean"),
            )
            .reset_index()
        )
        states = states.merge(targets, on=keys, how="left", validate="one_to_one")

    signature, signature_manifest = build_global_residual_signature(frame, cfg)
    if not signature.empty:
        states = states.merge(
            signature,
            on=cfg.timestamp_col,
            how="left",
            validate="many_to_one",
        )

    # Lower-ranked/non-admitted rows are a placebo population only. Their
    # outcomes never enter AE auxiliary heads or GMM enrichment estimates.
    placebo = frame.loc[~selected_mask.fillna(False).astype(bool)].copy()
    if not placebo.empty:
        placebo_targets = (
            placebo.groupby(keys, observed=True)
            .agg(
                placebo_target_mean_ev=(cfg.ev_col, "mean"),
                placebo_target_clean_rate=(cfg.hit_col, "mean"),
                placebo_target_bad_mae_rate=(cfg.bad_mae_col, "mean"),
            )
            .reset_index()
        )
        states = states.merge(
            placebo_targets, on=keys, how="left", validate="one_to_one"
        )

    # These are fixed observable combinations. They deliberately run before
    # target columns are excluded so the state table can retain train-only
    # auxiliaries alongside an outcome-free inference coordinate set.
    states, phase_manifest = add_causal_phase_state_features(
        states,
        timestamp_col=cfg.timestamp_col,
        side_col=cfg.side_col,
    )

    states["selected_rows"] = (
        _numeric(states, "selected_rows", 0.0).fillna(0).astype(np.int32)
    )
    states["selected_assets"] = (
        _numeric(states, "selected_assets", 0.0).fillna(0).astype(np.int16)
    )
    target_columns = {
        name
        for name in states
        if name.startswith("target_") or name.startswith("placebo_target_")
    }
    state_features = [
        name
        for name in states.select_dtypes(include=[np.number, "bool"]).columns
        if name not in target_columns
    ]
    states = states.sort_values(keys, kind="stable").reset_index(drop=True)
    manifest = {
        "schema": "global_side_timestamp_market_state_v1",
        "config": asdict(cfg),
        "rows": len(states),
        "market_features": available_market,
        "asset_features": available_asset,
        "state_features": state_features,
        "archetype_mix_features": list(archetype_mix.columns),
        "causal_phase_features": phase_manifest,
        "global_residual_signature": signature_manifest,
        "population_contract": (
            "Universe aggregates use the fixed base top30 candidate universe; broadcast mkt_/market_ "
            "features were generated from the full eligible asset universe. Selected aggregates use "
            "the exact frozen policy selection."
        ),
        "leakage_contract": (
            "target_* columns are train auxiliaries and placebo_target_* columns are diagnostics; "
            "both are excluded from AE inputs."
        ),
    }
    return states, state_features, manifest


if nn is not None:  # pragma: no branch

    class _ResidualAwareNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
            self.head = nn.Linear(latent_dim, 4)

        def forward(
            self, values: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            latent = self.encoder(values)
            return self.decoder(latent), self.head(latent), latent


class ResidualAwareAutoencoder:
    """Robust-scaled AE with weak, non-cancelling residual auxiliary heads."""

    target_columns = (
        "target_positive_surprise",
        "target_negative_surprise",
        "target_negative_ev",
        "target_payoff_asymmetry",
    )

    def __init__(self, config: ResidualAEConfig | None = None) -> None:
        self.config = config or ResidualAEConfig()
        self.partition: dict[str, str] | None = None
        self.feature_names: list[str] = []
        self.feature_medians: np.ndarray | None = None
        self.scaler: RobustScaler | None = None
        self.target_center: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None
        self.model_state: dict[str, Any] | None = None
        self.training_report: dict[str, Any] = {}

    def _prepare_x(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        values = (
            frame.reindex(columns=self.feature_names)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        if fit:
            self.feature_medians = np.nanmedian(values, axis=0).astype(np.float32)
            self.feature_medians = np.where(
                np.isfinite(self.feature_medians), self.feature_medians, 0.0
            ).astype(np.float32)
        if self.feature_medians is None:
            raise RuntimeError("Autoencoder is not fitted")
        missing = ~np.isfinite(values)
        values[missing] = np.take(self.feature_medians, np.where(missing)[1])
        if fit:
            self.scaler = RobustScaler(
                quantile_range=(10.0, 90.0), unit_variance=True
            ).fit(values)
        if self.scaler is None:
            raise RuntimeError("Autoencoder scaler is not fitted")
        return np.clip(self.scaler.transform(values), -8.0, 8.0).astype(np.float32)

    def _prepare_y(
        self, frame: pd.DataFrame, fit: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        values = (
            frame.reindex(columns=self.target_columns)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        mask = np.isfinite(values).astype(np.float32)
        if fit:
            self.target_center, self.target_scale = _finite_column_center_scale(values)
        if self.target_center is None or self.target_scale is None:
            raise RuntimeError("Autoencoder target transform is not fitted")
        values = np.where(np.isfinite(values), values, self.target_center)
        values = np.clip((values - self.target_center) / self.target_scale, -8.0, 8.0)
        return values.astype(np.float32), mask

    def fit(
        self, frame: pd.DataFrame, feature_names: Sequence[str]
    ) -> "ResidualAwareAutoencoder":
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for residual-aware AE fitting")
        self.partition = _single_state_partition(frame)
        requested = [str(name) for name in feature_names if str(name) in frame.columns]
        sample = frame[requested]
        coverage = sample.notna().mean()
        variance = sample.apply(pd.to_numeric, errors="coerce").var(ddof=0)
        eligible = [
            name
            for name in requested
            if coverage.get(name, 0.0) >= 0.65
            and np.isfinite(variance.get(name, np.nan))
            and variance.get(name, 0.0) > 1e-10
        ]
        priority = sorted(
            eligible,
            key=lambda name: (
                -int(
                    name.startswith(
                        (
                            "selected_minus_full_universe__",
                            "full_universe__median__",
                            "selected_minus_universe__",
                            "universe__median__",
                            "selected__median__",
                        )
                    )
                ),
                -float(coverage[name]),
                name,
            ),
        )
        correlation_sample = (
            sample[priority]
            .iloc[:: max(1, len(sample) // 12_000)]
            .apply(pd.to_numeric, errors="coerce")
        )
        correlation_sample = correlation_sample.fillna(
            correlation_sample.median()
        ).fillna(0.0)
        correlation = correlation_sample.corr().abs().fillna(0.0)
        keep: list[str] = []
        for name in priority:
            if (
                not keep
                or float(correlation.loc[name, keep].max())
                < self.config.correlation_prune_threshold
            ):
                keep.append(name)
            if len(keep) >= int(self.config.max_input_features):
                break
        self.feature_names = keep
        if not self.feature_names:
            raise ValueError("No AE input features")
        x = self._prepare_x(frame, fit=True)
        y, y_mask = self._prepare_y(frame, fit=True)
        torch.manual_seed(int(self.config.random_state))
        np.random.seed(int(self.config.random_state))
        model = _ResidualAwareNetwork(
            len(self.feature_names), self.config.hidden_dim, self.config.latent_dim
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        split = max(
            1, min(len(x) - 1, int(len(x) * (1.0 - self.config.validation_fraction)))
        )
        train_idx = np.arange(split)
        valid_idx = np.arange(split, len(x))
        rng = np.random.default_rng(self.config.random_state)
        best_loss = np.inf
        best_state: dict[str, Any] | None = None
        stale = 0
        history: list[dict[str, float]] = []
        head_weights = torch.tensor(
            [
                self.config.lambda_surprise,
                self.config.lambda_surprise,
                self.config.lambda_ev,
                self.config.lambda_asymmetry,
            ],
            dtype=torch.float32,
        )

        def loss_for(indices: np.ndarray, train_mode: bool) -> torch.Tensor:
            xb = torch.from_numpy(x[indices])
            yb = torch.from_numpy(y[indices])
            mb = torch.from_numpy(y_mask[indices])
            reconstruction, prediction, _ = model(xb)
            recon_loss = torch.mean((reconstruction - xb) ** 2)
            denominator = torch.clamp(mb.sum(dim=0), min=1.0)
            head_loss = (((prediction - yb) ** 2) * mb).sum(dim=0) / denominator
            return recon_loss + torch.sum(head_weights * head_loss)

        for epoch in range(int(self.config.epochs)):
            model.train()
            order = rng.permutation(train_idx)
            losses: list[float] = []
            for start in range(0, len(order), int(self.config.batch_size)):
                batch = order[start : start + int(self.config.batch_size)]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(batch, True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                valid_loss = float(loss_for(valid_idx, False))
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "valid_loss": valid_loss,
                }
            )
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= int(self.config.patience):
                    break
        if best_state is None:
            best_state = model.state_dict()
        self.model_state = best_state
        self.training_report = {
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "epochs_run": int(len(history)),
            "best_validation_loss": float(best_loss),
            "history_tail": history[-10:],
            "config": asdict(self.config),
            "requested_features": int(len(requested)),
            "eligible_features": int(len(eligible)),
            "selected_input_features": int(len(self.feature_names)),
        }
        return self

    def _model(self) -> Any:
        if torch is None or self.model_state is None:
            raise RuntimeError("Autoencoder is not fitted")
        model = _ResidualAwareNetwork(
            len(self.feature_names), self.config.hidden_dim, self.config.latent_dim
        )
        model.load_state_dict(self.model_state)
        model.eval()
        return model

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        _validate_state_partition(frame, getattr(self, "partition", None))
        x = self._prepare_x(frame, fit=False)
        model = self._model()
        latent_parts: list[np.ndarray] = []
        reconstruction_parts: list[np.ndarray] = []
        head_parts: list[np.ndarray] = []
        inference_batch_size = max(256, min(4096, int(self.config.batch_size) * 4))
        with torch.no_grad():
            for start in range(0, len(x), inference_batch_size):
                stop = min(start + inference_batch_size, len(x))
                batch = x[start:stop]
                reconstruction, heads, latent = model(torch.from_numpy(batch))
                latent_parts.append(latent.numpy().astype(np.float32, copy=False))
                reconstruction_parts.append(
                    np.mean((reconstruction.numpy() - batch) ** 2, axis=1).astype(
                        np.float32, copy=False
                    )
                )
                head_parts.append(heads.numpy().astype(np.float32, copy=False))
        latent_np = np.concatenate(latent_parts, axis=0)
        reconstruction_error = np.concatenate(reconstruction_parts, axis=0)
        heads_np = (
            np.concatenate(head_parts, axis=0) * self.target_scale + self.target_center
        )
        output = pd.DataFrame(
            latent_np,
            columns=[f"global_state_latent_{idx}" for idx in range(latent_np.shape[1])],
            index=frame.index,
        )
        output["global_state_reconstruction_error"] = reconstruction_error.astype(
            np.float32
        )
        for idx, name in enumerate(self.target_columns):
            output[f"global_state_pred_{name.removeprefix('target_')}"] = heads_np[
                :, idx
            ].astype(np.float32)
        return output

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "global_residual_aware_autoencoder_v1",
            "partition": self.partition,
            "feature_names": self.feature_names,
            "target_columns": list(self.target_columns),
            "training_report": self.training_report,
            "inference_contract": (
                "Frozen partition-local scaler and AE use only pre-entry features; target "
                "columns are not required and side/archetype routing must match the fit."
            ),
        }


if nn is not None:  # pragma: no branch

    class _ResidualSignatureEncoderNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            latent_dim: int,
            target_dim: int,
            dropout: float,
            with_decoder: bool,
        ) -> None:
            super().__init__()
            widths = [
                int(input_dim),
                *[int(value) for value in hidden_dims],
                int(latent_dim),
            ]
            encoder_layers: list[nn.Module] = []
            for idx, (source, destination) in enumerate(
                zip(widths[:-1], widths[1:], strict=True)
            ):
                encoder_layers.append(nn.Linear(source, destination))
                if idx < len(widths) - 2:
                    encoder_layers.extend(
                        [
                            nn.LayerNorm(destination),
                            nn.GELU(),
                            nn.Dropout(float(dropout)),
                        ]
                    )
            self.encoder = nn.Sequential(*encoder_layers)
            self.signature_head = (
                nn.Linear(int(latent_dim), int(target_dim)) if target_dim else None
            )
            if with_decoder:
                decoder_widths = [
                    int(latent_dim),
                    *reversed([int(value) for value in hidden_dims]),
                    int(input_dim),
                ]
                decoder_layers: list[nn.Module] = []
                for idx, (source, destination) in enumerate(
                    zip(decoder_widths[:-1], decoder_widths[1:], strict=True)
                ):
                    decoder_layers.append(nn.Linear(source, destination))
                    if idx < len(decoder_widths) - 2:
                        decoder_layers.extend([nn.GELU(), nn.Dropout(float(dropout))])
                self.decoder: nn.Module | None = nn.Sequential(*decoder_layers)
            else:
                self.decoder = None

        def forward(
            self, values: torch.Tensor
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
            latent = self.encoder(values)
            reconstruction = self.decoder(latent) if self.decoder is not None else None
            signature = (
                self.signature_head(latent) if self.signature_head is not None else None
            )
            return reconstruction, signature, latent


class GlobalResidualSignatureEncoder:
    """Train-only partition-local encoder with a frozen OOS transform."""

    def __init__(self, config: ResidualEncoderConfig | None = None) -> None:
        self.config = config or ResidualEncoderConfig()
        if self.config.encoder_kind not in ENCODER_PRESETS:
            raise ValueError(
                f"Unknown encoder_kind={self.config.encoder_kind!r}; "
                f"expected one of {sorted(ENCODER_PRESETS)}"
            )
        preset = ENCODER_PRESETS[self.config.encoder_kind]
        self.reconstruction_weight = float(preset["reconstruction_weight"])
        self.signature_weight = float(preset["signature_weight"])
        self.partition: dict[str, str] | None = None
        self.feature_names: list[str] = []
        self.target_columns: list[str] = []
        self.feature_medians: np.ndarray | None = None
        self.scaler: RobustScaler | None = None
        self.target_center: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None
        self.target_weights: np.ndarray | None = None
        self.model_state: dict[str, Any] | None = None
        self.training_report: dict[str, Any] = {}

    @staticmethod
    def _target_weight(name: str) -> float:
        lowered = str(name).lower()
        if "prev7d" in lowered:
            if "negative" in lowered:
                return 1.75
            if "positive" in lowered:
                return 1.45
            return 1.30
        if lowered.endswith("_2d"):
            if "negative" in lowered:
                return 1.10
            if "positive" in lowered:
                return 0.95
            return 0.80
        if any(
            token in lowered for token in ("negative", "dirty", "bad_mae", "timeout")
        ):
            return 1.40
        if any(token in lowered for token in ("positive", "clean", "missed")):
            return 1.15
        if "payoff_asymmetry" in lowered or "mean_ev" in lowered:
            return 1.00
        if "concentration" in lowered or "dispersion" in lowered:
            return 0.70
        return 0.85

    def _select_features(self, frame: pd.DataFrame, requested: Sequence[str]) -> None:
        names = [str(name) for name in requested if str(name) in frame.columns]
        if not names:
            raise ValueError("No encoder input features")
        sample = frame[names]
        coverage = sample.notna().mean()
        variance = sample.apply(pd.to_numeric, errors="coerce").var(ddof=0)
        eligible = [
            name
            for name in names
            if float(coverage.get(name, 0.0)) >= 0.65
            and np.isfinite(float(variance.get(name, np.nan)))
            and float(variance.get(name, 0.0)) > 1e-10
            and not name.startswith(("target_", "placebo_target_"))
        ]
        priority = sorted(
            eligible,
            key=lambda name: (
                -int(
                    name.startswith(
                        (
                            "selected_minus_full_universe__",
                            "full_universe__median__",
                            "selected_minus_universe__",
                            "universe__median__",
                            "selected__median__",
                        )
                    )
                ),
                -float(coverage[name]),
                name,
            ),
        )
        if not priority:
            raise ValueError("No eligible encoder input features")
        stride = max(1, len(sample) // 12_000)
        correlation_sample = (
            sample[priority].iloc[::stride].apply(pd.to_numeric, errors="coerce")
        )
        correlation_sample = correlation_sample.fillna(
            correlation_sample.median()
        ).fillna(0.0)
        correlation = correlation_sample.corr().abs().fillna(0.0)
        keep: list[str] = []
        for name in priority:
            if not keep or float(correlation.loc[name, keep].max()) < float(
                self.config.correlation_prune_threshold
            ):
                keep.append(name)
            if len(keep) >= int(self.config.max_input_features):
                break
        self.feature_names = keep

    def _prepare_x(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values = (
            frame.reindex(columns=self.feature_names)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        if fit:
            medians = np.nanmedian(values, axis=0).astype(np.float32)
            self.feature_medians = np.where(np.isfinite(medians), medians, 0.0).astype(
                np.float32
            )
        if self.feature_medians is None:
            raise RuntimeError("Residual signature encoder is not fitted")
        missing = ~np.isfinite(values)
        if np.any(missing):
            values[missing] = np.take(self.feature_medians, np.where(missing)[1])
        if fit:
            self.scaler = RobustScaler(
                quantile_range=(10.0, 90.0), unit_variance=True
            ).fit(values)
        if self.scaler is None:
            raise RuntimeError("Residual signature encoder scaler is not fitted")
        return np.clip(self.scaler.transform(values), -8.0, 8.0).astype(np.float32)

    def _prepare_y(
        self, frame: pd.DataFrame, *, fit: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.target_columns:
            return (
                np.zeros((len(frame), 0), dtype=np.float32),
                np.zeros((len(frame), 0), dtype=np.float32),
            )
        values = (
            frame.reindex(columns=self.target_columns)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        mask = np.isfinite(values).astype(np.float32)
        if fit:
            self.target_center, self.target_scale = _finite_column_center_scale(values)
            raw_weights = np.asarray(
                [self._target_weight(name) for name in self.target_columns],
                dtype=np.float32,
            )
            self.target_weights = raw_weights / max(float(raw_weights.mean()), 1e-8)
        if self.target_center is None or self.target_scale is None:
            raise RuntimeError("Residual signature target transform is not fitted")
        values = np.where(np.isfinite(values), values, self.target_center)
        values = np.clip((values - self.target_center) / self.target_scale, -8.0, 8.0)
        return values.astype(np.float32), mask

    def _network(self) -> Any:
        if torch is None or nn is None or self.model_state is None:
            raise RuntimeError("Residual signature encoder is not fitted")
        network = _ResidualSignatureEncoderNetwork(
            len(self.feature_names),
            self.config.hidden_dims,
            self.config.latent_dim,
            len(self.target_columns),
            self.config.dropout,
            self.reconstruction_weight > 0.0,
        )
        network.load_state_dict(self.model_state)
        network.eval()
        return network

    def fit(
        self, frame: pd.DataFrame, feature_names: Sequence[str]
    ) -> "GlobalResidualSignatureEncoder":
        if torch is None or nn is None:
            raise RuntimeError(
                "PyTorch is required for residual signature encoder fitting"
            )
        torch.set_num_threads(max(1, int(self.config.torch_num_threads)))
        self.partition = _single_state_partition(frame)
        self._select_features(frame, feature_names)
        self.target_columns = (
            sorted(
                name
                for name in frame.columns
                if name.startswith("target_signature_")
                and _signature_target_allowed_for_frame(name, frame)
                and pd.to_numeric(frame[name], errors="coerce").notna().any()
            )
            if self.signature_weight > 0.0
            else []
        )
        if self.signature_weight > 0.0 and not self.target_columns:
            raise ValueError(
                "Supervised encoder requires global residual signature targets"
            )
        x = self._prepare_x(frame, fit=True)
        y, y_mask = self._prepare_y(frame, fit=True)
        torch.manual_seed(int(self.config.random_state))
        np.random.seed(int(self.config.random_state))
        model = _ResidualSignatureEncoderNetwork(
            len(self.feature_names),
            self.config.hidden_dims,
            self.config.latent_dim,
            len(self.target_columns),
            self.config.dropout,
            self.reconstruction_weight > 0.0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        split = max(
            1,
            min(
                len(x) - 1,
                int(len(x) * (1.0 - float(self.config.validation_fraction))),
            ),
        )
        train_idx = np.arange(split, dtype=np.int32)
        valid_idx = np.arange(split, len(x), dtype=np.int32)
        if not len(valid_idx):
            raise ValueError(
                "Residual signature encoder requires a temporal validation tail"
            )
        rng = np.random.default_rng(self.config.random_state)
        target_weights = torch.from_numpy(
            self.target_weights
            if self.target_weights is not None
            else np.ones(len(self.target_columns), dtype=np.float32)
        )

        def _loss(indices: np.ndarray) -> torch.Tensor:
            xb = torch.from_numpy(x[indices])
            reconstruction, prediction, _ = model(xb)
            total = torch.zeros((), dtype=torch.float32)
            if reconstruction is not None and self.reconstruction_weight > 0.0:
                total = total + float(self.reconstruction_weight) * torch.mean(
                    (reconstruction - xb) ** 2
                )
            if prediction is not None and self.signature_weight > 0.0:
                yb = torch.from_numpy(y[indices])
                mb = torch.from_numpy(y_mask[indices])
                denominator = torch.clamp(mb.sum(dim=0), min=1.0)
                head_loss = (((prediction - yb) ** 2) * mb).sum(dim=0) / denominator
                total = total + float(self.signature_weight) * torch.mean(
                    target_weights * head_loss
                )
            return total

        best_loss = np.inf
        best_state: dict[str, Any] | None = None
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(int(self.config.epochs)):
            model.train()
            order = rng.permutation(train_idx)
            batch_losses: list[float] = []
            for start in range(0, len(order), int(self.config.batch_size)):
                batch = order[start : start + int(self.config.batch_size)]
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                valid_loss = float(_loss(valid_idx))
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(np.mean(batch_losses)),
                    "valid_loss": valid_loss,
                }
            )
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= int(self.config.patience):
                    break
        self.model_state = best_state or {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        self.training_report = {
            "encoder_kind": self.config.encoder_kind,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "epochs_run": int(len(history)),
            "best_validation_loss": float(best_loss),
            "selected_input_features": int(len(self.feature_names)),
            "signature_targets": int(len(self.target_columns)),
            "reconstruction_weight": float(self.reconstruction_weight),
            "signature_weight": float(self.signature_weight),
            "history_tail": history[-10:],
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        _validate_state_partition(frame, getattr(self, "partition", None))
        x = self._prepare_x(frame, fit=False)
        model = self._network()
        with torch.no_grad():
            reconstruction, prediction, latent = model(torch.from_numpy(x))
        latent_values = latent.numpy().astype(np.float32)
        output = pd.DataFrame(
            latent_values,
            columns=[
                f"global_state_latent_{idx}" for idx in range(latent_values.shape[1])
            ],
            index=frame.index,
        )
        generated: dict[str, np.ndarray] = {
            "global_state_input_novelty": np.mean(np.abs(x), axis=1).astype(np.float32)
        }
        if reconstruction is not None:
            generated["global_state_reconstruction_error"] = np.mean(
                (reconstruction.numpy() - x) ** 2, axis=1
            ).astype(np.float32)
        if prediction is not None and self.target_columns:
            if self.target_center is None or self.target_scale is None:
                raise RuntimeError("Residual signature target transform is unavailable")
            predictions = prediction.numpy() * self.target_scale + self.target_center
            for idx, name in enumerate(self.target_columns):
                generated[f"global_state_pred_{name.removeprefix('target_')}"] = (
                    predictions[:, idx].astype(np.float32)
                )
        return pd.concat(
            [output, pd.DataFrame(generated, index=frame.index)], axis=1, copy=False
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "global_residual_signature_encoder_v1",
            "config": asdict(self.config),
            "effective_reconstruction_weight": self.reconstruction_weight,
            "effective_signature_weight": self.signature_weight,
            "feature_names": self.feature_names,
            "target_columns": self.target_columns,
            "partition": self.partition,
            "training_report": self.training_report,
            "inference_contract": (
                "transform requires the matching side/archetype plus frozen pre-entry feature "
                "columns; realized residual signature targets and recent model outcomes are "
                "not inference inputs"
            ),
        }


def _posterior_weighted_mean(posterior: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    weights = posterior[valid]
    numerator = weights.T @ values[valid]
    denominator = np.maximum(weights.sum(axis=0), 1e-8)
    return numerator / denominator


class GlobalGMMStateModel:
    """Select and enrich one partition-local GMM using train-only economics."""

    enrichment_targets = (
        "target_signed_surprise",
        "target_positive_surprise",
        "target_negative_surprise",
        "target_negative_ev",
        "target_payoff_asymmetry",
    )

    def __init__(self, config: GMMGridConfig | None = None) -> None:
        self.config = config or GMMGridConfig()
        self.partition: dict[str, str] | None = None
        self.model: GaussianMixture | None = None
        self.grid: pd.DataFrame = pd.DataFrame()
        self.enrichments: dict[str, np.ndarray] = {}
        self.global_targets: dict[str, float] = {}
        self.months: tuple[str, ...] = ()
        self.fitted_enrichment_targets: tuple[str, ...] = ()

    def fit(
        self, latent: pd.DataFrame, targets: pd.DataFrame, timestamps: pd.Series
    ) -> "GlobalGMMStateModel":
        self.partition = _single_state_partition(targets)
        x = latent.filter(regex=r"^global_state_latent_").to_numpy(dtype=np.float64)
        months = pd.to_datetime(timestamps, utc=True).dt.strftime("%Y-%m").to_numpy()
        self.months = tuple(sorted(set(months)))
        signature_targets = [
            name
            for name in targets.columns
            if name.startswith("target_signature_")
            and _signature_target_allowed_for_frame(name, targets)
            and int(pd.to_numeric(targets[name], errors="coerce").notna().sum())
            >= int(self.config.min_enrichment_target_rows)
        ]
        self.fitted_enrichment_targets = tuple(
            dict.fromkeys(
                [
                    target
                    for target in self.enrichment_targets
                    if target in targets.columns
                    and pd.to_numeric(targets[target], errors="coerce").notna().any()
                ]
                + signature_targets
            )
        )
        candidates: list[tuple[dict[str, Any], GaussianMixture, np.ndarray]] = []
        for components in self.config.components:
            if len(x) < components * 20:
                continue
            for covariance in self.config.covariance_types:
                for reg_covar in self.config.reg_covars:
                    model = GaussianMixture(
                        n_components=int(components),
                        covariance_type=str(covariance),
                        reg_covar=float(reg_covar),
                        n_init=int(self.config.n_init),
                        max_iter=int(self.config.max_iter),
                        random_state=int(self.config.random_state),
                    ).fit(x)
                    posterior = model.predict_proba(x)
                    occupancy = posterior.mean(axis=0)
                    month_presence = []
                    for month in self.months:
                        month_presence.append(
                            (posterior[months == month].mean(axis=0) >= 0.005).mean()
                        )
                    component_month_shares = np.vstack(
                        [
                            posterior[months == month].sum(axis=0)
                            / np.maximum(posterior.sum(axis=0), 1e-8)
                            for month in self.months
                        ]
                    )
                    event_diversity = np.nan
                    largest_event_share = np.nan
                    if "diagnostic_event_ids" in targets.columns:
                        event_ids = (
                            targets["diagnostic_event_ids"]
                            .fillna("")
                            .astype(str)
                            .to_numpy()
                        )
                        event_mask = event_ids != ""
                        if event_mask.any():
                            unique_events = sorted(set(event_ids[event_mask]))
                            event_mass = np.vstack(
                                [
                                    posterior[event_ids == event].sum(axis=0)
                                    for event in unique_events
                                ]
                            )
                            component_event_total = np.maximum(
                                event_mass.sum(axis=0), 1e-8
                            )
                            event_share = event_mass / component_event_total
                            event_diversity = float(
                                np.mean((event_share >= 0.02).sum(axis=0))
                            )
                            largest_event_share = float(event_share.max())
                    enrichment_values: list[float] = []
                    for target in self.fitted_enrichment_targets:
                        values = pd.to_numeric(
                            targets[target], errors="coerce"
                        ).to_numpy(dtype=float)
                        means = _posterior_weighted_mean(posterior, values)
                        enrichment_values.append(float(np.nanstd(means)))
                    row = {
                        "components": int(components),
                        "covariance_type": str(covariance),
                        "reg_covar": float(reg_covar),
                        "bic_per_row": float(model.bic(x) / len(x)),
                        "min_occupancy": float(occupancy.min()),
                        "occupancy_entropy": float(
                            entropy(occupancy) / math.log(len(occupancy))
                        ),
                        "month_coverage": float(np.mean(month_presence)),
                        "largest_month_share": float(component_month_shares.max()),
                        "event_diversity": event_diversity,
                        "largest_event_share": largest_event_share,
                        "residual_enrichment": float(np.nanmean(enrichment_values)),
                        "converged": bool(model.converged_),
                    }
                    candidates.append((row, model, posterior))
        if not candidates:
            raise ValueError("No valid GMM candidates")
        grid = pd.DataFrame([row for row, _, _ in candidates])
        for source, destination, lower_is_better in (
            ("bic_per_row", "bic_score", True),
            ("month_coverage", "coverage_score", False),
            ("residual_enrichment", "enrichment_score", False),
            ("min_occupancy", "occupancy_score", False),
            ("event_diversity", "event_diversity_score", False),
        ):
            if grid[source].notna().sum() == 0:
                grid[destination] = 0.5
                continue
            low = float(grid[source].min())
            high = float(grid[source].max())
            scaled = (grid[source] - low) / max(high - low, 1e-12)
            grid[destination] = 1.0 - scaled if lower_is_better else scaled
        grid["selection_score"] = (
            0.35 * grid["bic_score"]
            + 0.20 * grid["coverage_score"]
            + 0.25 * grid["enrichment_score"]
            + 0.15 * grid["occupancy_score"]
            + 0.05 * grid["event_diversity_score"]
            - 0.15 * grid["largest_month_share"]
            - 0.10 * grid["largest_event_share"].fillna(0.0)
        )
        valid = (
            grid["min_occupancy"].ge(self.config.min_component_occupancy)
            & grid["converged"]
        )
        chosen_index = (
            grid.loc[valid, "selection_score"].idxmax()
            if valid.any()
            else grid["selection_score"].idxmax()
        )
        self.grid = grid.sort_values(
            "selection_score", ascending=False, kind="stable"
        ).reset_index(drop=True)
        chosen_position = int(chosen_index)
        _, self.model, posterior = candidates[chosen_position]
        occupancy_rows = posterior.sum(axis=0)
        for target in self.fitted_enrichment_targets:
            values = pd.to_numeric(targets[target], errors="coerce").to_numpy(
                dtype=float
            )
            global_value = float(np.nanmean(values))
            local = _posterior_weighted_mean(posterior, values)
            strength = occupancy_rows / (
                occupancy_rows + float(self.config.shrinkage_rows)
            )
            self.enrichments[target] = (
                strength * local + (1.0 - strength) * global_value
            ).astype(np.float32)
            self.global_targets[target] = global_value
        return self

    def transform(self, latent: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("GMM is not fitted")
        x = latent.filter(regex=r"^global_state_latent_").to_numpy(dtype=np.float64)
        posterior = self.model.predict_proba(x).astype(np.float32)
        output = pd.DataFrame(
            posterior,
            columns=[
                f"global_state_posterior_{idx}" for idx in range(posterior.shape[1])
            ],
            index=latent.index,
        )
        output["global_state_id"] = posterior.argmax(axis=1).astype(np.int16)
        output["global_state_entropy"] = (
            -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1)
        ).astype(np.float32)
        output["global_state_novelty"] = (-self.model.score_samples(x)).astype(
            np.float32
        )
        for target, values in self.enrichments.items():
            output[f"global_state_expected_{target.removeprefix('target_')}"] = (
                posterior @ values
            ).astype(np.float32)
        return output

    def manifest(self) -> dict[str, Any]:
        if self.model is None:
            return {"fitted": False}
        return {
            "schema": "archetype_partition_gmm_state_v1",
            "config": asdict(self.config),
            "partition": self.partition,
            "selected": {
                "components": int(self.model.n_components),
                "covariance_type": self.model.covariance_type,
                "reg_covar": float(self.model.reg_covar),
            },
            "global_targets": self.global_targets,
            "fitted_enrichment_targets": list(self.fitted_enrichment_targets),
        }


class SideArchetypeStatePriors:
    """Frozen posterior-weighted outcome priors by side and base archetype.

    Realized outcomes are consumed only by :meth:`fit`.  OOS transformation
    requires posterior probabilities plus side/archetype identity.  The
    partition-local mode used by the champion path has no cross-archetype
    fallback; the legacy hierarchical mode may shrink toward a side parent.
    """

    target_sources: dict[str, str] = {
        "ev": "ev_after_1pct",
        "clean": "clean_exec",
        "dirty_positive": "dirty_positive",
        "bad_mae": "full_path_bad_mae_1r",
        "timeout": "timeout",
        "hit_surprise": "__derived_hit_surprise__",
    }

    def __init__(
        self,
        *,
        side_col: str = "side_name",
        archetype_col: str = "archetype_policy_key",
        shrinkage_rows: float = 120.0,
        output_prefix: str = "global_state_expected_arch_",
        partition_local: bool = False,
        strict_unknown: bool = False,
    ) -> None:
        self.side_col = str(side_col)
        self.archetype_col = str(archetype_col)
        self.shrinkage_rows = float(shrinkage_rows)
        self.output_prefix = str(output_prefix)
        self.partition_local = bool(partition_local)
        self.strict_unknown = bool(strict_unknown)
        self.posterior_columns: list[str] = []
        self.parent_priors: dict[str, dict[str, np.ndarray]] = {}
        self.local_priors: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self.support: dict[tuple[str, str], float] = {}

    @staticmethod
    def _posterior_prior(
        posterior: np.ndarray,
        values: np.ndarray,
        fallback: np.ndarray,
        shrinkage_rows: float,
    ) -> tuple[np.ndarray, float]:
        valid = np.isfinite(values) & np.isfinite(posterior).all(axis=1)
        if not np.any(valid):
            return fallback.astype(np.float32, copy=True), 0.0
        weights = posterior[valid].astype(np.float64, copy=False)
        target = values[valid].astype(np.float64, copy=False)
        mass = weights.sum(axis=0)
        local = (weights.T @ target) / np.maximum(mass, 1e-8)
        strength = mass / (mass + max(float(shrinkage_rows), 1e-8))
        prior = strength * local + (1.0 - strength) * fallback
        return prior.astype(np.float32), float(valid.sum())

    def _target_values(self, frame: pd.DataFrame, key: str) -> np.ndarray:
        source = self.target_sources[key]
        if source == "__derived_hit_surprise__":
            clean = pd.to_numeric(frame.get("clean_exec"), errors="coerce")
            probability_source = (
                frame["hit_probability"]
                if "hit_probability" in frame.columns
                else frame.get("score_meta_base_soft_label")
            )
            probability = pd.to_numeric(probability_source, errors="coerce")
            return (clean - probability).to_numpy(dtype=np.float64)
        if source not in frame.columns:
            return np.full(len(frame), np.nan, dtype=np.float64)
        return pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=np.float64)

    def fit(
        self,
        frame: pd.DataFrame,
        posterior_columns: Sequence[str],
    ) -> "SideArchetypeStatePriors":
        self.posterior_columns = [
            str(name) for name in posterior_columns if str(name) in frame.columns
        ]
        if not self.posterior_columns:
            raise ValueError("Side/archetype state priors require posterior columns")
        posterior_all = frame[self.posterior_columns].to_numpy(
            dtype=np.float64, copy=False
        )
        side_values = frame[self.side_col].astype(str).str.lower()
        archetypes = frame[self.archetype_col].astype(str)
        n_components = len(self.posterior_columns)
        target_values = {
            target: self._target_values(frame, target) for target in self.target_sources
        }
        if self.partition_local:
            groups = (
                pd.DataFrame(
                    {
                        "side": side_values.to_numpy(),
                        "archetype": archetypes.to_numpy(),
                    }
                )
                .groupby(["side", "archetype"], sort=True)
                .indices
            )
            for (side, archetype), positions_raw in groups.items():
                positions = np.asarray(positions_raw, dtype=np.int64)
                local: dict[str, np.ndarray] = {}
                local_support = 0.0
                for target in self.target_sources:
                    values = target_values[target][positions]
                    finite = np.isfinite(values)
                    local_mean = (
                        float(np.nanmean(values[finite])) if np.any(finite) else 0.0
                    )
                    fallback = np.full(n_components, local_mean, dtype=np.float64)
                    local[target], support = self._posterior_prior(
                        posterior_all[positions],
                        values,
                        fallback,
                        self.shrinkage_rows,
                    )
                    local_support = max(local_support, support)
                key = (str(side), str(archetype))
                self.local_priors[key] = local
                self.support[key] = float(local_support)
            return self
        for side in sorted(side_values.unique()):
            side_mask = side_values.eq(side).to_numpy()
            parent: dict[str, np.ndarray] = {}
            for target in self.target_sources:
                values = target_values[target]
                valid = side_mask & np.isfinite(values)
                global_mean = float(np.nanmean(values[valid])) if np.any(valid) else 0.0
                fallback = np.full(n_components, global_mean, dtype=np.float64)
                parent[target], _ = self._posterior_prior(
                    posterior_all[side_mask],
                    values[side_mask],
                    fallback,
                    self.shrinkage_rows,
                )
            self.parent_priors[side] = parent
            local_archetypes = sorted(archetypes.loc[side_mask].unique())
            for archetype in local_archetypes:
                local_mask = side_mask & archetypes.eq(archetype).to_numpy()
                local: dict[str, np.ndarray] = {}
                local_support = 0.0
                for target in self.target_sources:
                    values = target_values[target]
                    local[target], support = self._posterior_prior(
                        posterior_all[local_mask],
                        values[local_mask],
                        parent[target].astype(np.float64),
                        self.shrinkage_rows,
                    )
                    local_support = max(local_support, support)
                key = (side, str(archetype))
                self.local_priors[key] = local
                self.support[key] = float(local_support)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.posterior_columns or (
            not self.parent_priors and not self.local_priors
        ):
            raise RuntimeError("Side/archetype state priors are not fitted")
        posterior = frame.reindex(columns=self.posterior_columns).to_numpy(
            dtype=np.float32, copy=False
        )
        posterior = np.nan_to_num(posterior, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = posterior.sum(axis=1, keepdims=True)
        posterior = np.divide(
            posterior,
            np.maximum(row_sum, 1e-8),
            out=np.full_like(posterior, 1.0 / max(posterior.shape[1], 1)),
            where=row_sum > 1e-8,
        )
        side_values = frame[self.side_col].astype(str).str.lower().to_numpy()
        archetypes = frame[self.archetype_col].astype(str).to_numpy()
        output = {
            target: np.zeros(len(frame), dtype=np.float32)
            for target in self.target_sources
        }
        output_support = np.zeros(len(frame), dtype=np.float32)
        groups = (
            pd.DataFrame({"side": side_values, "archetype": archetypes})
            .groupby(["side", "archetype"], sort=False)
            .indices
        )
        for (side, archetype), positions_raw in groups.items():
            positions = np.asarray(positions_raw, dtype=np.int64)
            if self.partition_local:
                priors = self.local_priors.get((str(side), str(archetype)))
                if priors is None:
                    if self.strict_unknown:
                        raise ValueError(
                            "No frozen state prior for side/archetype partition "
                            f"{side}__{archetype}"
                        )
                    continue
                for target in self.target_sources:
                    output[target][positions] = posterior[positions] @ priors[target]
                output_support[positions] = float(
                    self.support.get((str(side), str(archetype)), 0.0)
                )
                continue
            parent = self.parent_priors.get(str(side))
            if parent is None:
                parent = next(iter(self.parent_priors.values()))
            priors = self.local_priors.get((str(side), str(archetype)), parent)
            for target in self.target_sources:
                output[target][positions] = posterior[positions] @ priors[target]
            output_support[positions] = float(
                self.support.get((str(side), str(archetype)), 0.0)
            )
        result = pd.DataFrame(index=frame.index)
        for target, values in output.items():
            result[f"{self.output_prefix}{target}"] = values
        result[f"{self.output_prefix}support_log1p"] = np.log1p(output_support).astype(
            np.float32
        )
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "side_archetype_state_priors_v1",
            "posterior_columns": self.posterior_columns,
            "side_col": self.side_col,
            "archetype_col": self.archetype_col,
            "shrinkage_rows": self.shrinkage_rows,
            "output_prefix": self.output_prefix,
            "partition_local": self.partition_local,
            "strict_unknown": self.strict_unknown,
            "local_groups": int(len(self.local_priors)),
            "support": {
                f"{side}__{archetype}": value
                for (side, archetype), value in self.support.items()
            },
            "leakage_contract": (
                "Priors use fit rows only; OOS transform consumes frozen posteriors plus "
                "side/archetype identity and never realized outcomes"
            ),
        }


def add_temporal_state_features(
    state_features: pd.DataFrame,
    timestamps: pd.Series,
) -> pd.DataFrame:
    output = state_features.copy()
    order = np.argsort(pd.to_datetime(timestamps, utc=True).to_numpy(), kind="stable")
    posterior_cols = [
        name for name in output if name.startswith("global_state_posterior_")
    ]
    posterior = output.iloc[order][posterior_cols].to_numpy(dtype=np.float32)
    delta1 = np.vstack(
        [
            np.zeros((1, posterior.shape[1]), dtype=np.float32),
            np.diff(posterior, axis=0),
        ]
    )
    delta4 = (
        posterior - np.vstack([np.repeat(posterior[:1], 4, axis=0), posterior[:-4]])
        if len(posterior) > 4
        else np.zeros_like(posterior)
    )
    acceleration = np.vstack(
        [np.zeros((1, posterior.shape[1]), dtype=np.float32), np.diff(delta1, axis=0)]
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    for idx, name in enumerate(posterior_cols):
        output[f"{name}_delta_1h"] = delta1[inverse, idx]
        output[f"{name}_delta_4h"] = delta4[inverse, idx]
        output[f"{name}_acceleration"] = acceleration[inverse, idx]
    speed = np.linalg.norm(delta1, axis=1)
    accel = np.linalg.norm(acceleration, axis=1)
    output["global_state_latent_speed"] = speed[inverse].astype(np.float32)
    output["global_state_latent_acceleration"] = accel[inverse].astype(np.float32)
    labels = posterior.argmax(axis=1)
    dwell = np.ones(len(labels), dtype=np.int32)
    for idx in range(1, len(labels)):
        dwell[idx] = dwell[idx - 1] + 1 if labels[idx] == labels[idx - 1] else 1
    output["global_state_dwell_time"] = dwell[inverse]
    output["global_state_transition_entropy"] = (
        -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1)
    )[inverse].astype(np.float32)
    return output


def state_recognition_metrics(
    frame: pd.DataFrame,
    risk_col: str,
    opportunity_col: str,
) -> dict[str, float]:
    risk = pd.to_numeric(frame[risk_col], errors="coerce")
    opportunity = pd.to_numeric(frame[opportunity_col], errors="coerce")
    negative_event = pd.to_numeric(frame["target_negative_ev"], errors="coerce").gt(0.0)
    positive_event = pd.to_numeric(
        frame["target_positive_surprise"], errors="coerce"
    ).gt(0.0)

    def auprc(target: pd.Series, score: pd.Series) -> float:
        valid = target.notna() & score.notna()
        return (
            float(average_precision_score(target[valid].astype(int), score[valid]))
            if valid.sum() > 10 and target[valid].nunique() > 1
            else np.nan
        )

    top_risk = risk.ge(risk.quantile(0.95))
    top_opportunity = opportunity.ge(opportunity.quantile(0.95))
    return {
        "negative_ev_auprc": auprc(negative_event, risk),
        "positive_surprise_auprc": auprc(positive_event, opportunity),
        "negative_ev_precision_top5pct": float(negative_event[top_risk].mean()),
        "positive_surprise_precision_top5pct": float(
            positive_event[top_opportunity].mean()
        ),
        "incremental_ev_top_opportunity_state": float(
            pd.to_numeric(
                frame.loc[top_opportunity, "target_mean_ev"], errors="coerce"
            ).mean()
            - pd.to_numeric(frame["target_mean_ev"], errors="coerce").mean()
        ),
    }


def centroid_matched_ari(
    reference_model: GaussianMixture,
    challenger_model: GaussianMixture,
    x: np.ndarray,
) -> float:
    distance = np.linalg.norm(
        reference_model.means_[:, None, :] - challenger_model.means_[None, :, :], axis=2
    )
    ref_idx, challenger_idx = linear_sum_assignment(distance)
    mapping = {
        int(challenger): int(reference)
        for reference, challenger in zip(ref_idx, challenger_idx, strict=True)
    }
    reference_labels = reference_model.predict(x)
    challenger_labels = np.array(
        [mapping.get(int(label), -1) for label in challenger_model.predict(x)]
    )
    return float(adjusted_rand_score(reference_labels, challenger_labels))
