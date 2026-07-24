"""Side-local AE/GMM representation search for base and meta research.

This module deliberately separates representation selection from production
model fitting.  A state block is fitted only on a fold's authorised train
reference rows, refined only against an *inner* chronological validation
slice, and can then be materialised as an optional sidecar.  It never changes
the existing base/meta feature contract implicitly.

The state identifiers emitted here are local to ``layer x side``.  Consumers
must use the continuous posterior/distance outputs and preserve ``side_name``;
``*_component_id_local`` is diagnostic only and is not comparable between
long/short or base/meta blocks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - environment specific.
    LGBMRegressor = None

from extreme_price_movements.alternative_latent_encoders import (
    AlternativeLatentEncoder,
    EncoderConfig,
)
from extreme_price_movements.config import CFG
from extreme_price_movements.representation_proxy_metrics import (
    bounded_diagonal_gmm_overlap,
    diagonal_gmm_statistics,
    refine_diagonal_gmm,
)


SCHEMA_VERSION = "side_local_ae_gmm_search_v1"
SIDES = ("long", "short")
PREFIX_SIZES = (30, 75, 150)
PROXY_GMM_SPECS = ((3, "diag", 0.003), (3, "diag", 0.01), (5, "diag", 0.003), (5, "diag", 0.01))
FULL_REG_COVARS = (0.001, 0.003, 0.01, 0.03)
OUTCOME_TOKENS = (
    "target", "outcome", "realized", "realised", "future", "mfe", "mae", "timeout",
    "stop", "first_touch", "ret_net", "net_return", "exec_margin", "pnl", "profit",
)
META_EXTRA_TOKENS = (
    "base_", "oof", "score", "rank", "margin", "uncert", "calibr", "drift", "ood",
    "residual", "error", "leaf", "surprise",
)
PATH_COLUMNS = (
    "mfe", "mae", "mfe_mae", "tp", "sl", "stop", "timeout", "time_to_mfe", "time_to_mae",
)


@dataclass(frozen=True)
class SearchConfig:
    """Fixed compute contract for one side/layer representation search."""

    layer: str
    side_name: str
    random_state: int = 20260720
    proxy_rows: int = 50_000
    final_rows: int = 150_000
    prefixes: tuple[int, ...] = PREFIX_SIZES
    proxy_gmm_specs: tuple[tuple[int, str, float], ...] = PROXY_GMM_SPECS
    final_components: tuple[int, ...] = (3, 4, 5, 6)
    final_covariance_types: tuple[str, ...] = ("diag", "tied")
    final_reg_covars: tuple[float, ...] = FULL_REG_COVARS
    lambda_ladder: tuple[float, ...] = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)
    encoder_kinds: tuple[str, ...] = ("dae", "masked")
    proxy_epochs: int = 12
    final_epochs: int = 28
    max_gmm_workers: int = 2
    refinement_rows: int = 15_000
    refinement_torch_threads: int = 1
    # MPS is useful for encoders, but repeated mixture-NLL/overlap backward
    # passes have proven unstable across torch/macOS builds. This ladder is
    # deliberately small, so deterministic CPU float64 is preferable.
    refinement_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.layer not in {"base", "meta"}:
            raise ValueError("layer must be base or meta")
        if self.side_name not in SIDES:
            raise ValueError("side_name must be long or short")
        if not self.prefixes or sorted(set(self.prefixes)) != list(self.prefixes):
            raise ValueError("prefixes must be sorted unique positive sizes")


@dataclass
class SideLocalState:
    """Frozen transform state for one side-local state block."""

    config: SearchConfig
    feature_names: list[str]
    encoder_state: dict[str, Any]
    latent_scaler: StandardScaler
    gmm: GaussianMixture | dict[str, Any]
    novelty_reference: np.ndarray
    component_count: int
    selected_candidate_id: str
    feature_schema_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return f"{self.config.layer}_{self.config.side_name}_state"

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen state without access to outcomes or targets."""
        _ensure_transform_frame(frame)
        result = empty_state_block(
            frame.index, self.prefix, self.component_count, self.latent_scaler.n_features_in_
        )
        side = _side_series(frame)
        eligible = side.eq(self.config.side_name).to_numpy()
        available = [name for name in self.feature_names if name in frame]
        if len(available) != len(self.feature_names):
            result[f"{self.prefix}_valid"] = 0
            result[f"{self.prefix}_missing_feature_count"] = len(self.feature_names) - len(available)
            return result
        valid = eligible & np.isfinite(frame[self.feature_names].to_numpy(dtype=np.float32)).any(axis=1)
        if not np.any(valid):
            return result
        encoder = AlternativeLatentEncoder.from_state(self.encoder_state)
        values = _impute_matrix(frame.loc[valid, self.feature_names])
        native = encoder.transform_native(values, sides=np.repeat(self.config.side_name, len(values)))
        latent = self.latent_scaler.transform(np.asarray(native.latent, dtype=np.float32))
        stats = _gmm_statistics(self.gmm, latent)
        output_index = result.index[valid]
        for j in range(self.component_count):
            result.loc[output_index, f"{self.prefix}_posterior_{j:02d}"] = stats["posterior"][:, j]
            result.loc[output_index, f"{self.prefix}_mahalanobis_{j:02d}"] = stats["mahalanobis"][:, j]
        reconstruction = native.reconstruction_error
        if reconstruction is None:
            reconstruction = np.zeros(len(output_index), dtype=np.float32)
        novelty_raw = stats["min_mahalanobis"] + np.asarray(reconstruction, dtype=np.float32)
        result.loc[output_index, f"{self.prefix}_component_id_local"] = stats["posterior"].argmax(axis=1)
        result.loc[output_index, f"{self.prefix}_entropy"] = stats["entropy"]
        result.loc[output_index, f"{self.prefix}_posterior_margin"] = stats["margin"]
        result.loc[output_index, f"{self.prefix}_expected_mahalanobis"] = stats["expected_mahalanobis"]
        result.loc[output_index, f"{self.prefix}_min_mahalanobis"] = stats["min_mahalanobis"]
        result.loc[output_index, f"{self.prefix}_reconstruction_error"] = reconstruction
        result.loc[output_index, f"{self.prefix}_novelty_percentile"] = _empirical_percentile(
            novelty_raw, self.novelty_reference
        )
        for j in range(latent.shape[1]):
            result.loc[output_index, f"{self.prefix}_latent_{j:02d}"] = latent[:, j]
        result.loc[output_index, f"{self.prefix}_active"] = 1
        result.loc[output_index, f"{self.prefix}_valid"] = 1
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "layer": self.config.layer,
            "side_name": self.config.side_name,
            "state_prefix": self.prefix,
            "component_ids_globally_comparable": False,
            "selected_candidate_id": self.selected_candidate_id,
            "feature_names": self.feature_names,
            "feature_schema_hash": self.feature_schema_hash,
            "component_count": self.component_count,
            "output_columns": state_block_columns(self.prefix, self.component_count, self.latent_scaler.n_features_in_),
            "metadata": self.metadata,
        }


def config_feature_universe(layer: str) -> list[str]:
    """Expand the layer's config groups, preserving the config order."""
    if layer not in {"base", "meta"}:
        raise ValueError("layer must be base or meta")
    group_names = list(CFG.get(f"{layer}_shared_feature_keys", ()))
    if layer == "base":
        group_names += list(CFG.get("base_long_feature_keys", ())) + list(CFG.get("base_short_feature_keys", ()))
    values: list[str] = []
    for item in group_names:
        group = CFG.get(str(item))
        if isinstance(group, (tuple, list)):
            values.extend(map(str, group))
        else:
            values.append(str(item))
    return list(dict.fromkeys(value for value in values if value and not _is_outcome_column(value)))


def available_layer_features(frame: pd.DataFrame, layer: str) -> list[str]:
    """Return observable config features present in ``frame``.

    Meta also admits materialised OOF base/calibration/uncertainty/drift
    columns.  Target-derived realised outcomes remain excluded even if a
    source ledger happens to contain them.
    """
    configured = set(config_feature_universe(layer))
    observed = {str(col) for col in frame.columns}
    if layer == "meta":
        configured.update(
            column for column in observed
            if any(token in column.lower() for token in META_EXTRA_TOKENS)
            and not _is_outcome_column(column)
        )
    return [column for column in frame.columns if str(column) in configured and _is_numeric_like(frame[column])]


def stable_feature_filter(frame: pd.DataFrame, features: Sequence[str], *, timestamp: str = "timestamp") -> pd.DataFrame:
    """Availability/variance/drift screen before supervised ranking."""
    work = frame.sort_values(timestamp, kind="stable") if timestamp in frame else frame
    bands = _time_bands(len(work))
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(work[feature], errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        coverage = float(finite.mean())
        if finite.any():
            q25, median, q75 = np.nanpercentile(values[finite], (25, 50, 75))
            scale = max(float(q75 - q25), 1e-6)
            outlier = float(np.mean(np.abs((values[finite] - median) / scale) > 25.0))
            band_medians = [float(np.nanmedian(values[index])) for index in bands if np.isfinite(values[index]).any()]
            drift = float(np.std(band_medians) / scale) if len(band_medians) > 1 else 0.0
            variance = float(np.nanvar(values[finite]))
        else:
            outlier, drift, variance = 1.0, np.inf, 0.0
        stable = bool(coverage >= 0.70 and variance > 1e-10 and outlier <= 0.02 and drift <= 8.0)
        rows.append({"feature": feature, "coverage": coverage, "variance": variance, "outlier_rate": outlier, "median_drift_iqr": drift, "stable": stable, "stability_score": coverage - 0.5 * outlier - 0.05 * min(drift, 10.0)})
    return pd.DataFrame(rows).sort_values(["stable", "stability_score"], ascending=False, kind="stable")


def correlation_cluster_representatives(
    frame: pd.DataFrame,
    stats: pd.DataFrame,
    relevance: Mapping[str, float],
    *, threshold: float = 0.90,
    max_rows: int = 25_000,
) -> tuple[list[str], pd.DataFrame]:
    """Keep one stable/relevant feature from each abs-Spearman >= threshold group."""
    stable = stats.loc[stats["stable"].astype(bool), "feature"].astype(str).tolist()
    if not stable:
        return [], pd.DataFrame(columns=["feature", "correlation_group", "representative"])
    sampled = frame[stable].iloc[_time_spread_indices(len(frame), min(max_rows, len(frame)))].copy()
    sampled = sampled.apply(pd.to_numeric, errors="coerce")
    corr = sampled.corr(method="spearman").abs().fillna(0.0).to_numpy(dtype=np.float32)
    parent = np.arange(len(stable), dtype=np.int32)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def union(a: int, b: int) -> None:
        left, right = find(a), find(b)
        if left != right:
            parent[right] = left
    for i in range(len(stable)):
        for j in np.flatnonzero(corr[i, i + 1 :] >= threshold) + i + 1:
            union(i, int(j))
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(stable):
        groups.setdefault(find(i), []).append(name)
    stable_scores = stats.set_index("feature")["stability_score"].to_dict()
    records, keep = [], []
    for group_id, names in enumerate(groups.values()):
        best = max(names, key=lambda name: (float(relevance.get(name, 0.0)), float(stable_scores.get(name, 0.0)), name))
        keep.append(best)
        records.extend({"feature": name, "correlation_group": group_id, "representative": name == best} for name in names)
    return keep, pd.DataFrame(records)


def univariate_relief_mda_ranking(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    target_column: str,
    net_ev_column: str | None,
    timestamp: str = "timestamp",
    random_state: int = 20260720,
) -> pd.DataFrame:
    """Economically informed canonical ranking: uni + Relief rescue + chronological MDA.

    The univariate term is side-local Spearman against the *soft* target.
    Relief is an intentionally lightweight nearest-neighbour rescue. MDA is
    measured over three chronological folds with a fixed small LGBM probe.
    """
    if target_column not in frame:
        raise KeyError(f"Missing target column {target_column!r}")
    ordered = frame.sort_values(timestamp, kind="stable").reset_index(drop=True) if timestamp in frame else frame.reset_index(drop=True)
    y = pd.to_numeric(ordered[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    x = _impute_matrix(ordered[list(features)])
    uni = _spearman_columns(x, y)
    top_uni = np.argsort(np.abs(uni))[::-1][: min(200, x.shape[1])]
    relief = _relief_scores(x, y, random_state=random_state)
    rescued = [idx for idx in np.argsort(relief)[::-1][: min(100, x.shape[1])] if idx not in set(top_uni)]
    candidate_idx = np.array(sorted(set(top_uni).union(rescued)), dtype=np.int32)
    mda_mean, mda_std = _chronological_mda(
        x[:, candidate_idx], y, None if net_ev_column is None or net_ev_column not in ordered else pd.to_numeric(ordered[net_ev_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        random_state=random_state,
    )
    records = []
    for local, original in enumerate(candidate_idx):
        records.append({
            "feature": str(features[int(original)]),
            "univariate_spearman": float(uni[int(original)]),
            "univariate_rank": int(np.where(np.argsort(np.abs(uni))[::-1] == original)[0][0] + 1),
            "relief_score": float(relief[int(original)]),
            "relief_rescued": bool(int(original) in set(rescued)),
            "mda_mean": float(mda_mean[local]),
            "mda_std": float(mda_std[local]),
            "mda_stable_importance": float(max(mda_mean[local] - 0.5 * mda_std[local], 0.0)),
        })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["selection_score"] = result["mda_stable_importance"] + 0.05 * result["univariate_spearman"].abs()
    return result.sort_values(["selection_score", "univariate_spearman"], ascending=False, kind="stable").reset_index(drop=True)


def nested_feature_prefixes(ranking: pd.DataFrame, sizes: Sequence[int] = PREFIX_SIZES) -> dict[int, list[str]]:
    """Derive nested 30/75/150 sets from the one frozen side-local ranking.

    The small prefix is the stable-MDA core.  Larger prefixes intentionally
    extend into lower-MDA candidates rather than collapsing whenever fewer
    than 75/150 features have positive ``mean - .5*std`` importance.  That is
    required for a representation search: an AE may retain useful joint state
    structure from features whose standalone supervised importance is weak.
    All extensions still come from the already filtered, correlation-pruned,
    univariate/Relief-screened ranking.
    """
    if ranking.empty:
        return {int(size): [] for size in sizes}
    order_column = "selection_score" if "selection_score" in ranking else "mda_stable_importance"
    ordered = ranking.sort_values(order_column, ascending=False, kind="stable")
    return {
        int(size): ordered["feature"].head(min(int(size), len(ordered))).astype(str).tolist()
        for size in sizes
    }


def fit_encoder(frame: pd.DataFrame, features: Sequence[str], config: SearchConfig, *, rows: int, kind: str, latent_dim: int, epochs: int) -> tuple[AlternativeLatentEncoder, np.ndarray, np.ndarray]:
    indices = _time_spread_indices(len(frame), min(rows, len(frame)))
    values = _impute_matrix(frame.iloc[indices][list(features)])
    encoder = AlternativeLatentEncoder(EncoderConfig(
        kind=kind,
        latent_dim=int(latent_dim),
        hidden_dim=max(48, min(128, 2 * int(latent_dim))),
        epochs=int(epochs),
        pretrain_epochs=max(1, int(epochs * 0.66)),
        batch_size=512,
        corruption_rate=0.0,
        element_mask_rate=0.05 if kind != "dae" else 0.0,
        additive_noise_std=0.01 if kind != "dae" else 0.0,
        ssl_objective="masked_reconstruction",
        random_state=config.random_state,
        # The caller controls the staged reference budget.  Proxy comparisons
        # use 50k rows; selected-package refits must actually consume their
        # larger frozen B/M/E reference rather than silently falling back to
        # the proxy budget.
        dae_max_train_rows=int(rows),
        device="auto",
    )).fit(values, sides=np.repeat(config.side_name, len(values)))
    native = encoder.transform_native(values, sides=np.repeat(config.side_name, len(values)))
    # Return native coordinates.  Callers own the single serialized scaler used
    # by both GMM fitting and ``SideLocalState.transform``.  Returning scaled
    # coordinates here would fit the GMM on one scale but transform inference
    # rows on another, breaking density/posterior parity.
    return encoder, np.asarray(native.latent, dtype=np.float32), indices


def fit_gmm(latent: np.ndarray, *, components: int, covariance_type: str, reg_covar: float, random_state: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=int(components), covariance_type=str(covariance_type), reg_covar=float(reg_covar),
        n_init=1, max_iter=250, random_state=int(random_state), init_params="kmeans",
    ).fit(np.asarray(latent, dtype=np.float64))


def evaluate_density_candidate(
    frame: pd.DataFrame,
    *,
    encoder: AlternativeLatentEncoder,
    scaler: StandardScaler,
    gmm: GaussianMixture | Mapping[str, Any],
    features: Sequence[str],
    target_column: str,
    net_ev_column: str | None,
    archetype_column: str | None,
    timestamp: str = "timestamp",
    random_state: int = 20260720,
) -> dict[str, float]:
    """Inner chronological economic evaluation; outer OOS is deliberately absent."""
    ordered = frame.sort_values(timestamp, kind="stable").reset_index(drop=True) if timestamp in frame else frame.reset_index(drop=True)
    if len(ordered) < 800:
        return {"incremental_top10_ev": 0.0, "incremental_top20_ev": 0.0, "archetype_rank_gain": 0.0, "path_separation": 0.0, "stability": 0.0, "density_quality": 0.0}
    split = max(400, int(len(ordered) * 0.75))
    # Both probe predictions are indexed from zero.  Reset the validation
    # index before any archetype-local selection so group positions remain
    # aligned with the compact validation arrays below.
    train = ordered.iloc[:split].reset_index(drop=True)
    valid = ordered.iloc[split:].reset_index(drop=True)
    y_train = pd.to_numeric(train[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_valid = pd.to_numeric(valid[target_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    raw_features = _probe_features(train, valid, y_train, limit=40)
    side_label = next(iter(encoder.side_mapping), "__all__")
    native_train = encoder.transform_native(_impute_matrix(train[list(features)]), sides=np.repeat(side_label, len(train)))
    native_valid = encoder.transform_native(_impute_matrix(valid[list(features)]), sides=np.repeat(side_label, len(valid)))
    z_train = scaler.transform(native_train.latent)
    z_valid = scaler.transform(native_valid.latent)
    state_train = _state_probe_matrix(gmm, z_train, native_train.reconstruction_error)
    state_valid = _state_probe_matrix(gmm, z_valid, native_valid.reconstruction_error)
    raw_pred = _fit_probe_predict(raw_features[0], y_train, raw_features[1], random_state)
    enriched_pred = _fit_probe_predict(np.column_stack((raw_features[0], state_train)), y_train, np.column_stack((raw_features[1], state_valid)), random_state)
    ev = pd.to_numeric(valid[net_ev_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if net_ev_column in valid else y_valid
    raw10, raw20 = _top_mean(ev, raw_pred, .10), _top_mean(ev, raw_pred, .20)
    state10, state20 = _top_mean(ev, enriched_pred, .10), _top_mean(ev, enriched_pred, .20)
    archetype_gain = _archetype_gain(valid, ev, raw_pred, enriched_pred, archetype_column)
    path_sep = _path_separation(valid, _gmm_statistics(gmm, z_valid)["posterior"].argmax(axis=1))
    density = _density_quality(gmm, z_valid)
    stats_valid = _gmm_statistics(gmm, z_valid)
    perturbed = z_valid + np.random.default_rng(random_state + 7).normal(0.0, 0.01, z_valid.shape)
    perturbation_consistency = 1.0 - float(np.mean(np.abs(stats_valid["posterior"] - _gmm_statistics(gmm, perturbed)["posterior"])))
    occupancy_blocks = [stats_valid["posterior"][chunk].mean(axis=0) for chunk in _time_bands(len(z_valid)) if len(chunk)]
    temporal_stability = 1.0 - float(np.clip(np.mean(np.std(np.vstack(occupancy_blocks), axis=0)), 0.0, 1.0)) if occupancy_blocks else 0.0
    seed_stability = 0.0
    if isinstance(gmm, GaussianMixture):
        alternate = fit_gmm(z_train, components=gmm.n_components, covariance_type=gmm.covariance_type, reg_covar=gmm.reg_covar, random_state=random_state + 991)
        alternate_entropy = _gmm_statistics(alternate, z_valid)["entropy"]
        seed_stability = 1.0 - float(np.clip(np.mean(np.abs(stats_valid["entropy"] - alternate_entropy)), 0.0, 1.0))
    stability = float(np.mean([perturbation_consistency, temporal_stability, seed_stability]))
    posterior_target_spearman = float(np.max(np.abs(_spearman_columns(stats_valid["posterior"], y_valid))))
    return {
        "incremental_top10_ev": float(state10 - raw10), "incremental_top20_ev": float(state20 - raw20),
        "archetype_rank_gain": float(archetype_gain), "path_separation": float(path_sep),
        "stability": float(np.clip(stability, 0.0, 1.0)), "perturbation_consistency": float(perturbation_consistency),
        "temporal_occupancy_stability": float(temporal_stability), "seed_stability": float(seed_stability),
        "posterior_target_spearman_abs_max": posterior_target_spearman, "density_quality": float(density),
        "raw_top10_ev": float(raw10), "state_top10_ev": float(state10), "raw_top20_ev": float(raw20), "state_top20_ev": float(state20),
    }


def score_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalised selection objective from the representation contract."""
    weights = {
        "incremental_ev": .40, "archetype_rank_gain": .20, "path_separation": .15,
        "stability": .15, "density_quality": .10,
    }
    output = metrics.copy()
    output["incremental_ev"] = .5 * output["incremental_top10_ev"] + .5 * output["incremental_top20_ev"]
    output["candidate_score"] = 0.0
    for column, weight in weights.items():
        ranks = output[column].rank(pct=True, method="average") if len(output) > 1 else pd.Series(1.0, index=output.index)
        output[f"rank_{column}"] = ranks
        output["candidate_score"] += float(weight) * ranks
    return output.sort_values("candidate_score", ascending=False, kind="stable").reset_index(drop=True)


def refine_top_diagonal_candidates(
    latent: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    inner_validation: pd.DataFrame,
    *,
    evaluate: Any,
    config: SearchConfig,
) -> list[dict[str, Any]]:
    """Refine at most three unpenalized, distinct-K diagonal models on inner data.

    The caller supplies ``evaluate`` so it can be bound only to its inner
    validation fold. Two consecutive inferior lambda values stop the ladder.
    """
    chosen: list[Mapping[str, Any]] = []
    used_k: set[int] = set()
    for candidate in candidates:
        if str(candidate.get("covariance_type")) != "diag":
            continue
        k = int(candidate["components"])
        if k in used_k:
            continue
        chosen.append(candidate); used_k.add(k)
        if len(chosen) == 3:
            break
    results: list[dict[str, Any]] = []
    tensor_cache: dict[tuple[Any, ...], Any] = {}
    if config.refinement_device == "cpu":
        # The local refinement uses small tensors. Serial torch execution is
        # more than sufficient and avoids native BLAS/OpenMP oversubscription
        # observed on macOS under concurrent research workloads.
        try:
            import torch
            torch.set_num_threads(int(config.refinement_torch_threads))
        except ImportError:
            pass
    # The overlap objective is a local density adjustment, not the primary
    # fit. A deterministic B/M/E subset bounds autograd memory on M1 while the
    # lambda decision remains on the complete inner chronological validation.
    refinement_latent = latent[_time_spread_indices(len(latent), min(int(config.refinement_rows), len(latent)))]
    for candidate in chosen:
        inferior = 0
        best = -np.inf
        for lam in config.lambda_ladder:
            refined = refine_diagonal_gmm(refinement_latent, candidate["gmm"], overlap_lambda=float(lam), overlap_metric="bhattacharyya", steps=60, learning_rate=.01, device=config.refinement_device, tensor_cache=tensor_cache)
            inner_score = float(evaluate(refined["state"], inner_validation))
            record = {"candidate_id": candidate["candidate_id"], "components": int(candidate["components"]), "lambda": float(lam), "inner_score": inner_score, "overlap_before": refined["overlap_before"], "overlap_after": refined["overlap_after"], "refined": bool(refined["refined"])}
            results.append(record)
            if inner_score > best + 1e-12:
                best, inferior = inner_score, 0
            else:
                inferior += 1
            if inferior >= 2:
                record["early_stop_after_two_inferior"] = True
                break
    return results


def state_block_columns(prefix: str, components: int, latent_dim: int) -> list[str]:
    return [
        *[f"{prefix}_latent_{j:02d}" for j in range(int(latent_dim))],
        *[f"{prefix}_posterior_{j:02d}" for j in range(int(components))],
        *[f"{prefix}_mahalanobis_{j:02d}" for j in range(int(components))],
        f"{prefix}_component_id_local", f"{prefix}_entropy", f"{prefix}_posterior_margin",
        f"{prefix}_expected_mahalanobis", f"{prefix}_min_mahalanobis", f"{prefix}_reconstruction_error",
        f"{prefix}_novelty_percentile", f"{prefix}_active", f"{prefix}_valid", f"{prefix}_missing_feature_count",
    ]


def empty_state_block(index: pd.Index, prefix: str, components: int, latent_dim: int = 0) -> pd.DataFrame:
    output = pd.DataFrame(0.0, index=index, columns=state_block_columns(prefix, components, latent_dim), dtype=np.float32)
    output[f"{prefix}_component_id_local"] = -1
    return output


def feature_schema_hash(features: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, features)).encode()).hexdigest()


def _is_outcome_column(name: str) -> bool:
    lowered = str(name).lower()
    return any(token in lowered for token in OUTCOME_TOKENS)


def _ensure_transform_frame(frame: pd.DataFrame) -> None:
    forbidden = [str(name) for name in frame.columns if _is_outcome_column(str(name))]
    if forbidden:
        raise ValueError("State transform received forbidden outcome-derived columns: " + ", ".join(sorted(forbidden)[:12]))


def _side_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("side_name", "side"):
        if column in frame:
            return frame[column].astype(str).str.lower()
    raise KeyError("frame requires side_name or side")


def _is_numeric_like(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series) or pd.to_numeric(series, errors="coerce").notna().mean() >= .70)


def _time_bands(n: int) -> list[np.ndarray]:
    return [np.arange(start, end) for start, end in ((0, max(1, n // 3)), (n // 3, max(1, 2 * n // 3)), (2 * n // 3, n)) if end > start]


def _time_spread_indices(n: int, take: int) -> np.ndarray:
    if take >= n:
        return np.arange(n, dtype=np.int32)
    bands = _time_bands(n)
    counts = [take // len(bands) + (i < take % len(bands)) for i in range(len(bands))]
    pieces = [band[np.linspace(0, len(band) - 1, min(len(band), int(count)), dtype=np.int32)] for band, count in zip(bands, counts) if count]
    return np.sort(np.concatenate(pieces)).astype(np.int32)


def _impute_matrix(frame: pd.DataFrame) -> np.ndarray:
    array = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    med = np.nanmedian(array, axis=0)
    med[~np.isfinite(med)] = 0.0
    array[~np.isfinite(array)] = np.take(med, np.where(~np.isfinite(array))[1])
    return np.ascontiguousarray(np.clip(array, -1e6, 1e6), dtype=np.float32)


def _spearman_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx = pd.DataFrame(x).rank(pct=True).to_numpy(dtype=np.float32)
    ry = pd.Series(y).rank(pct=True).to_numpy(dtype=np.float32)
    rx -= rx.mean(axis=0, keepdims=True); ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx, axis=0) * np.sum(ry * ry))
    return np.divide(np.sum(rx * ry[:, None], axis=0), denom, out=np.zeros(x.shape[1], dtype=np.float32), where=denom > 1e-12)


def _relief_scores(x: np.ndarray, y: np.ndarray, *, random_state: int) -> np.ndarray:
    # Fast deterministic approximate Relief: between-bin minus within-bin separation.
    ranks = pd.Series(y).rank(pct=True).to_numpy()
    labels = np.minimum((ranks * 5).astype(np.int8), 4)
    scores = np.zeros(x.shape[1], dtype=np.float32)
    total = max(len(x), 1)
    global_med = np.median(x, axis=0)
    for label in np.unique(labels):
        member = labels == label
        if member.sum() < 10 or (~member).sum() < 10:
            continue
        scores += (member.sum() / total) * np.abs(np.median(x[member], axis=0) - np.median(x[~member], axis=0))
    return scores


def _chronological_mda(x: np.ndarray, y: np.ndarray, ev: np.ndarray | None, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if LGBMRegressor is None or len(x) < 1500:
        return np.abs(_spearman_columns(x, y)), np.zeros(x.shape[1], dtype=np.float32)
    folds = _time_bands(len(x))
    all_scores: list[np.ndarray] = []
    for fold_id in range(1, len(folds)):
        train_idx = np.concatenate(folds[:fold_id]); valid_idx = folds[fold_id]
        if len(train_idx) < 500 or len(valid_idx) < 250:
            continue
        model = _fit_probe_model(x[train_idx], y[train_idx], random_state + fold_id)
        baseline = _economic_probe_objective(y[valid_idx], model.predict(x[valid_idx]), None if ev is None else ev[valid_idx])
        values = x[valid_idx].copy(); rng = np.random.default_rng(random_state + 100 + fold_id)
        drops = np.empty(x.shape[1], dtype=np.float32)
        for j in range(x.shape[1]):
            values[:, j] = values[rng.permutation(len(values)), j]
            drops[j] = baseline - _economic_probe_objective(y[valid_idx], model.predict(values), None if ev is None else ev[valid_idx])
            values[:, j] = x[valid_idx, j]
        all_scores.append(drops)
    matrix = np.vstack(all_scores) if all_scores else np.zeros((1, x.shape[1]), dtype=np.float32)
    return matrix.mean(axis=0), matrix.std(axis=0)


def _new_probe(seed: int, n_rows: int) -> Any:
    return LGBMRegressor(
        n_estimators=300, max_depth=4, num_leaves=15,
        min_child_samples=max(20, int(math.ceil(.0025 * n_rows))),
        learning_rate=.05, reg_alpha=.5, reg_lambda=.5, subsample=.8,
        colsample_bytree=.8, subsample_freq=1, random_state=seed, n_jobs=2, verbosity=-1,
    )


def _fit_probe_model(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    """Fit the specified LGBM probe, with a deterministic sklearn fallback.

    Some local environments ship a LightGBM/sklearn combination whose sklearn
    wrapper cannot call ``check_X_y``.  A fallback keeps the representation
    comparison runnable but is recorded by the caller's runtime environment;
    it must not be compared across runs with a working LightGBM backend.
    """
    if LGBMRegressor is not None:
        try:
            return _new_probe(seed, len(x)).fit(x, y)
        except TypeError as exc:
            if "force_all_finite" not in str(exc):
                raise
    return HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, learning_rate=.05, l2_regularization=.5, min_samples_leaf=max(20, int(math.ceil(.0025 * len(x)))), random_state=seed).fit(x, y)


def _economic_probe_objective(y: np.ndarray, pred: np.ndarray, ev: np.ndarray | None) -> float:
    signal = ev if ev is not None else y
    top10 = _top_mean(signal, pred, .10); top20 = _top_mean(signal, pred, .20)
    return .6 * top10 + .4 * top20


def _top_mean(values: np.ndarray, score: np.ndarray, fraction: float) -> float:
    take = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(values[np.argsort(score)[-take:]]))


def _probe_features(train: pd.DataFrame, valid: pd.DataFrame, y_train: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    shared = [column for column in train.columns if column in valid and _is_numeric_like(train[column]) and not _is_outcome_column(str(column))]
    if not shared:
        raise ValueError("No observable raw features are available for probe")
    sample = _impute_matrix(train[shared])
    relevance = np.abs(_spearman_columns(sample, y_train))
    ordered = np.argsort(relevance)[::-1]
    # Greedy abs-Spearman pruning produces a frozen 40-column raw basis used
    # for every representation candidate in this side/layer run.
    ranks = pd.DataFrame(sample[:, ordered[: min(len(ordered), 250)]]).rank(pct=True).to_numpy(dtype=np.float32)
    selected_local: list[int] = []
    for local_idx, original_idx in enumerate(ordered[: ranks.shape[1]]):
        if len(selected_local) >= limit:
            break
        if not selected_local:
            selected_local.append(local_idx); continue
        corr = np.asarray([
            np.corrcoef(ranks[:, local_idx], ranks[:, prior])[0, 1]
            for prior in selected_local
        ])
        if not np.any(np.abs(corr) >= .90):
            selected_local.append(local_idx)
    selected = [shared[int(ordered[int(local)])] for local in selected_local]
    return _impute_matrix(train[selected]), _impute_matrix(valid[selected])


def _fit_probe_predict(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, seed: int) -> np.ndarray:
    if LGBMRegressor is None:
        return x_valid[:, 0]
    return _fit_probe_model(x_train, y_train, seed).predict(x_valid).astype(np.float32)


def _gmm_statistics(gmm: GaussianMixture | Mapping[str, Any], latent: np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(gmm, GaussianMixture):
        posterior = gmm.predict_proba(latent).astype(np.float32)
        means, cov, weights = gmm.means_, gmm.covariances_, gmm.weights_
        if gmm.covariance_type == "tied":
            inv = np.linalg.pinv(cov)
            distances = np.sqrt(np.maximum(0., np.einsum("nkd,df,nkf->nk", latent[:, None, :] - means[None, :, :], inv, latent[:, None, :] - means[None, :, :])))
        else:
            distances = np.sqrt(np.maximum(0., ((latent[:, None, :] - means[None, :, :]) ** 2 / cov[None, :, :]).sum(axis=2)))
    else:
        means = np.asarray(gmm["means"], dtype=np.float64); cov = np.asarray(gmm["covariances"], dtype=np.float64); weights = np.asarray(gmm["weights"], dtype=np.float64)
        diff = latent[:, None, :] - means[None, :, :]
        mahal2 = (diff * diff / cov[None, :, :]).sum(axis=2)
        logp = np.log(np.maximum(weights, 1e-12))[None, :] - .5 * (mahal2 + np.log(cov).sum(axis=1)[None, :] + latent.shape[1] * np.log(2. * np.pi))
        logp -= logp.max(axis=1, keepdims=True); posterior = np.exp(logp); posterior /= posterior.sum(axis=1, keepdims=True); distances = np.sqrt(np.maximum(0., mahal2))
    ordered = np.sort(posterior, axis=1)
    entropy = -(posterior * np.log(np.maximum(posterior, 1e-12))).sum(axis=1) / np.log(max(posterior.shape[1], 2))
    return {"posterior": posterior, "mahalanobis": distances.astype(np.float32), "entropy": entropy.astype(np.float32), "margin": (ordered[:, -1] - ordered[:, -2]).astype(np.float32), "expected_mahalanobis": (posterior * distances).sum(axis=1).astype(np.float32), "min_mahalanobis": distances.min(axis=1).astype(np.float32)}


def _state_probe_matrix(gmm: GaussianMixture | Mapping[str, Any], latent: np.ndarray, reconstruction_error: np.ndarray | None) -> np.ndarray:
    stats = _gmm_statistics(gmm, latent)
    pieces = [latent, stats["posterior"], stats["mahalanobis"], stats["entropy"][:, None], stats["margin"][:, None], stats["expected_mahalanobis"][:, None], stats["min_mahalanobis"][:, None]]
    if reconstruction_error is not None:
        pieces.append(np.asarray(reconstruction_error, dtype=np.float32).reshape(-1, 1))
    return np.ascontiguousarray(np.column_stack(pieces), dtype=np.float32)


def _archetype_gain(frame: pd.DataFrame, ev: np.ndarray, raw: np.ndarray, enriched: np.ndarray, archetype_column: str | None) -> float:
    if not archetype_column or archetype_column not in frame:
        return _top_mean(ev, enriched, .20) - _top_mean(ev, raw, .20)
    gains = []
    for _, group in frame.groupby(archetype_column, observed=True):
        positions = group.index.to_numpy(dtype=int)
        if len(positions) >= 20:
            gains.append(_top_mean(ev[positions], enriched[positions], .20) - _top_mean(ev[positions], raw[positions], .20))
    return float(np.mean(gains)) if gains else 0.0


def _path_separation(frame: pd.DataFrame, labels: np.ndarray) -> float:
    values = []
    for token in PATH_COLUMNS:
        columns = [column for column in frame.columns if token in str(column).lower() and _is_numeric_like(frame[column])]
        if columns:
            data = pd.to_numeric(frame[columns[0]], errors="coerce").fillna(0.0).to_numpy()
            means = pd.Series(data).groupby(labels).mean().to_numpy()
            if len(means) > 1:
                values.append(float(np.std(means) / (np.std(data) + 1e-6)))
    return float(np.mean(values)) if values else 0.0


def _density_quality(gmm: GaussianMixture | Mapping[str, Any], latent: np.ndarray) -> float:
    stats = _gmm_statistics(gmm, latent)
    occupancy = stats["posterior"].mean(axis=0)
    usable = float(np.mean(occupancy >= .01))
    entropy = float(np.mean(stats["entropy"]))
    overlap = (
        float(bounded_diagonal_gmm_overlap(gmm)["bounded_overlap"])
        if isinstance(gmm, GaussianMixture) and gmm.covariance_type == "diag"
        else 0.5
    )
    return float(np.clip(.5 * usable + .35 * (1.0 - abs(entropy - .5) * 2.0) + .15 * (1.0 - overlap), 0.0, 1.0))


def _empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float32))
    if len(ref) == 0:
        return np.zeros(len(values), dtype=np.float32)
    return (np.searchsorted(ref, values, side="right") / len(ref)).astype(np.float32)


__all__ = [
    "FULL_REG_COVARS", "PREFIX_SIZES", "PROXY_GMM_SPECS", "SCHEMA_VERSION", "SIDES",
    "SearchConfig", "SideLocalState", "available_layer_features", "config_feature_universe",
    "correlation_cluster_representatives", "empty_state_block", "evaluate_density_candidate",
    "feature_schema_hash", "fit_encoder", "fit_gmm", "nested_feature_prefixes",
    "refine_top_diagonal_candidates", "score_candidates", "stable_feature_filter", "state_block_columns",
    "univariate_relief_mda_ranking",
]
