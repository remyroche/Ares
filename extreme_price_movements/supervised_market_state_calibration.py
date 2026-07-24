"""Leakage-safe local market-state encoders for EV-aware score calibration.

The supervised MLP is used only to learn an economically relevant embedding on
chronologically prior rows.  OOS rows are transformed from observable features
only.  GMM outcome values are train-derived posterior-weighted priors, never OOS
outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression


EncoderArm = Literal["ae_gmm", "mlp_gmm", "mlp_direct", "ae_mlp_gmm"]


@dataclass
class LocalStateEncoder:
    side: str
    archetype: str
    arm: EncoderArm
    features: list[str]
    medians: np.ndarray
    scales: np.ndarray
    ae_features: list[str]
    ae_medians: np.ndarray
    ae_scales: np.ndarray
    mlp: Any | None
    gmm: Any | None
    cluster_ev: np.ndarray
    cluster_support: np.ndarray
    ev_center: float
    ev_scale: float
    rows: int
    # Score-distribution references are observable and are fitted only on the
    # local encoder's training rows.  They make reliability margins comparable
    # within, rather than across, side x archetype populations.
    reliability_reference: dict[str, float] = field(default_factory=dict)


@dataclass
class HierarchicalEVCalibrator:
    """Map a ranking score to a comparable net-EV unit.

    Side curves are deliberately shrunk toward the global monotone curve, and
    local side/archetype curves are shrunk toward their side prediction.  The
    fitted object requires only observable score and identity columns at
    inference.
    """

    global_model: IsotonicRegression
    local_models: dict[tuple[str, str], IsotonicRegression]
    local_weights: dict[tuple[str, str], float]
    local_support: dict[tuple[str, str], int]
    shrink_rows: float
    local_weight_cap: float
    rank_reference: np.ndarray
    rank_blend: float = 1.0
    monotonic_refinement_slope: float = 0.00025
    refinement_score_min: float = 0.0
    refinement_score_max: float = 1.0
    # Appended with defaults so joblib calibrators serialized before the
    # side-parent hierarchy remain readable by prediction and payload helpers.
    side_models: dict[str, IsotonicRegression] = field(default_factory=dict)
    side_weights: dict[str, float] = field(default_factory=dict)
    side_support: dict[str, int] = field(default_factory=dict)


@dataclass
class FrozenLocalMLPOverlay:
    """Joblib-safe local overlay consumed by replay and live calibration."""

    internally_derived_feature_columns = frozenset(
        {
            "meta_parent_rank_local_top10_margin",
            "meta_hit_probability_local_top10_margin",
            "meta_parent_reliability_local_support_log1p",
        }
    )

    encoder: LocalStateEncoder
    alpha: float
    cap: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy(deep=False)
        reference = self.encoder.reliability_reference or {}
        rank = pd.to_numeric(work.get("policy_parent_rank"), errors="coerce").fillna(0.5)
        hit = pd.to_numeric(work.get("hit_probability"), errors="coerce").fillna(0.5)
        rank_q90 = float(reference.get("parent_rank_q90", 0.90))
        hit_q90 = float(reference.get("hit_probability_q90", 0.90))
        support = max(float(reference.get("support", self.encoder.rows)), 0.0)
        work["meta_parent_rank_local_top10_margin"] = (rank - rank_q90).astype(np.float32)
        work["meta_hit_probability_local_top10_margin"] = (hit - hit_q90).astype(np.float32)
        work["meta_parent_reliability_local_support_log1p"] = np.float32(
            np.log1p(support)
        )
        pred = predict_local_state_encoder(
            self.encoder, work, ae_features=self.encoder.ae_features
        )
        correction = pred["ev_correction"] / max(self.encoder.ev_scale, 1e-4)
        delta = np.clip(float(self.alpha) * correction, -self.cap, self.cap)
        # regime_ev_calibration subtracts risk from the source score.
        return (-delta).astype(np.float32)


@dataclass
class FrozenLocalExtremeTailOverlay:
    """Frozen v9-style local empirical-tail demotion."""

    source_score_col: str
    features: list[str]
    directions: np.ndarray
    references: list[np.ndarray]
    threshold: float = 0.95
    alpha_down: float = 0.02

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        components: list[np.ndarray] = []
        for feature, direction, reference in zip(
            self.features, self.directions, self.references
        ):
            values = float(direction) * pd.to_numeric(
                frame[feature], errors="coerce"
            ).to_numpy(dtype=np.float32)
            ref = np.asarray(reference, dtype=np.float32)
            left = np.searchsorted(ref, values, side="left")
            right = np.searchsorted(ref, values, side="right")
            pct = ((left + right) / (2.0 * max(len(ref), 1))).astype(np.float32)
            pct[~np.isfinite(values)] = 0.5
            components.append(pct)
        composite = (
            np.mean(np.column_stack(components), axis=1)
            if components
            else np.full(len(frame), 0.5, dtype=np.float32)
        )
        intensity = np.clip(
            (composite - self.threshold) / max(1.0 - self.threshold, 1e-6),
            0.0,
            1.0,
        )
        score = pd.to_numeric(
            frame[self.source_score_col], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)
        return (self.alpha_down * intensity * (score >= 0.90)).astype(np.float32)


@dataclass
class FrozenCompositeLocalOverlay:
    """Sequential v9 tail-95 demotion followed by the local MLP overlay."""

    predecessor: FrozenLocalExtremeTailOverlay | None
    mlp_overlay: FrozenLocalMLPOverlay

    @property
    def internally_derived_feature_columns(self) -> frozenset[str]:
        return self.mlp_overlay.internally_derived_feature_columns

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy(deep=False)
        source_col = (
            self.predecessor.source_score_col
            if self.predecessor is not None
            else "calibrated_score"
        )
        source = pd.to_numeric(work[source_col], errors="coerce").fillna(0.0)
        predecessor_risk = (
            self.predecessor.predict(work)
            if self.predecessor is not None
            else np.zeros(len(work), dtype=np.float32)
        )
        work["policy_parent_rank"] = np.clip(
            source.to_numpy(dtype=np.float32) - predecessor_risk, 0.0, 1.0
        )
        mlp_risk = self.mlp_overlay.predict(work)
        return (predecessor_risk + mlp_risk).astype(np.float32)


def fit_hierarchical_ev_calibrator(
    frame: pd.DataFrame,
    score: np.ndarray,
    realized_net_ev: np.ndarray,
    *,
    shrink_rows: float = 2_000.0,
    min_local_rows: int = 400,
    local_weight_cap: float = 0.85,
    min_side_rows: int | None = None,
    side_weight_cap: float | None = None,
    tail_weight_top10: float = 4.0,
    tail_weight_top20: float = 2.0,
    tail_weight_by_score_quantile: bool = False,
    rank_blend: float = 1.0,
    monotonic_refinement_slope: float = 0.00025,
) -> HierarchicalEVCalibrator:
    """Fit global, side, and local monotone EV curves on authorized OOF rows only."""
    x = np.asarray(score, dtype=np.float64)
    y = np.asarray(realized_net_ev, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 100:
        raise ValueError("EV calibration requires at least 100 finite OOF rows")
    # Concentrate calibration capacity on the region that can enter the traded
    # book without discarding lower-ranked rows needed to anchor the curve.
    if tail_weight_by_score_quantile:
        q80, q90 = np.quantile(x[valid], [0.80, 0.90])
    else:
        q80, q90 = 0.80, 0.90
    weight = np.where(
        x >= q90,
        float(tail_weight_top10),
        np.where(x >= q80, float(tail_weight_top20), 0.25),
    )
    global_model = IsotonicRegression(out_of_bounds="clip", y_min=-0.25, y_max=0.25)
    global_model.fit(x[valid], y[valid], sample_weight=weight[valid])
    sides = frame["side_name"].astype(str).to_numpy()
    arches = frame["archetype_policy_key"].astype(str).to_numpy()
    effective_min_side_rows = int(
        min_local_rows if min_side_rows is None else min_side_rows
    )
    effective_side_weight_cap = float(
        local_weight_cap if side_weight_cap is None else side_weight_cap
    )
    side_models: dict[str, IsotonicRegression] = {}
    side_weights: dict[str, float] = {}
    side_support: dict[str, int] = {}
    for side in sorted(set(sides)):
        pos = valid & (sides == side)
        n = int(pos.sum())
        side_support[side] = n
        if n < effective_min_side_rows or np.unique(x[pos]).size < 8:
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=-0.25, y_max=0.25)
        model.fit(x[pos], y[pos], sample_weight=weight[pos])
        side_models[side] = model
        side_weights[side] = float(
            min(effective_side_weight_cap, n / (n + float(shrink_rows)))
        )
    local_models: dict[tuple[str, str], IsotonicRegression] = {}
    local_weights: dict[tuple[str, str], float] = {}
    local_support: dict[tuple[str, str], int] = {}
    for side, archetype in sorted(set(zip(sides, arches))):
        pos = valid & (sides == side) & (arches == archetype)
        n = int(pos.sum())
        local_support[(side, archetype)] = n
        if n < min_local_rows or np.unique(x[pos]).size < 8:
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=-0.25, y_max=0.25)
        model.fit(x[pos], y[pos], sample_weight=weight[pos])
        local_models[(side, archetype)] = model
        # Cap local authority: even large groups retain a side-parent anchor.
        local_weights[(side, archetype)] = float(
            min(float(local_weight_cap), n / (n + float(shrink_rows)))
        )
    # The auction rank must use the same hierarchical EV unit emitted at
    # inference.  Ranking against global-only EV predictions would make the
    # percentile contract inconsistent with side and side x archetype curves.
    hierarchical_ev = np.asarray(
        global_model.predict(x[valid]), dtype=np.float64
    )
    valid_sides = sides[valid]
    valid_arches = arches[valid]
    valid_scores = x[valid]
    for side, model in side_models.items():
        side_pos = valid_sides == side
        if not side_pos.any():
            continue
        side_ev = np.asarray(model.predict(valid_scores[side_pos]), dtype=np.float64)
        alpha = side_weights[side]
        hierarchical_ev[side_pos] = (
            (1.0 - alpha) * hierarchical_ev[side_pos] + alpha * side_ev
        )
    for key, model in local_models.items():
        local_pos = (valid_sides == key[0]) & (valid_arches == key[1])
        if not local_pos.any():
            continue
        local_ev = np.asarray(model.predict(valid_scores[local_pos]), dtype=np.float64)
        alpha = local_weights[key]
        hierarchical_ev[local_pos] = (
            (1.0 - alpha) * hierarchical_ev[local_pos] + alpha * local_ev
        )
    refinement_score_min, refinement_score_max = np.quantile(
        x[valid], [0.005, 0.995]
    )
    refinement_span = max(
        float(refinement_score_max - refinement_score_min), 1e-8
    )
    refinement_rank = (valid_scores - refinement_score_min) / refinement_span
    hierarchical_ev += max(float(monotonic_refinement_slope), 0.0) * (
        refinement_rank - 0.5
    )
    rank_reference = np.sort(hierarchical_ev.astype(np.float32, copy=False))
    return HierarchicalEVCalibrator(
        global_model=global_model,
        local_models=local_models,
        local_weights=local_weights,
        local_support=local_support,
        shrink_rows=float(shrink_rows),
        local_weight_cap=float(local_weight_cap),
        rank_reference=rank_reference,
        rank_blend=float(rank_blend),
        monotonic_refinement_slope=max(float(monotonic_refinement_slope), 0.0),
        refinement_score_min=float(refinement_score_min),
        refinement_score_max=float(refinement_score_max),
        side_models=side_models,
        side_weights=side_weights,
        side_support=side_support,
    )


def predict_hierarchical_ev(
    calibrator: HierarchicalEVCalibrator,
    frame: pd.DataFrame,
    score: np.ndarray,
) -> np.ndarray:
    """Return expected net EV/trade in the same decimal-return unit as labels."""
    x = np.asarray(score, dtype=np.float64)
    out = np.asarray(calibrator.global_model.predict(x), dtype=np.float64)
    sides = frame["side_name"].astype(str).to_numpy()
    arches = frame["archetype_policy_key"].astype(str).to_numpy()
    # ``getattr`` preserves prediction behavior for pre-side-parent joblib
    # calibrators: they retain the historical global -> side x archetype path.
    side_models = getattr(calibrator, "side_models", {})
    side_weights = getattr(calibrator, "side_weights", {})
    for side, model in side_models.items():
        pos = sides == side
        if not pos.any():
            continue
        parent = np.asarray(model.predict(x[pos]), dtype=np.float64)
        alpha = float(side_weights.get(side, 0.0))
        out[pos] = (1.0 - alpha) * out[pos] + alpha * parent
    for key, model in getattr(calibrator, "local_models", {}).items():
        pos = (sides == key[0]) & (arches == key[1])
        if not pos.any():
            continue
        local = np.asarray(model.predict(x[pos]), dtype=np.float64)
        alpha = float(getattr(calibrator, "local_weights", {}).get(key, 0.0))
        out[pos] = (1.0 - alpha) * out[pos] + alpha * local
    refinement_slope = max(
        float(getattr(calibrator, "monotonic_refinement_slope", 0.0)), 0.0
    )
    refinement_min = float(getattr(calibrator, "refinement_score_min", 0.0))
    refinement_max = float(getattr(calibrator, "refinement_score_max", 1.0))
    refinement_rank = (x - refinement_min) / max(
        refinement_max - refinement_min, 1e-8
    )
    out += refinement_slope * (refinement_rank - 0.5)
    return out.astype(np.float32)


def expected_ev_rank(
    calibrator: HierarchicalEVCalibrator,
    expected_ev: np.ndarray,
    raw_score: np.ndarray | None = None,
) -> np.ndarray:
    """Convert common-unit EV to a frozen train-derived global percentile."""
    ref = np.asarray(calibrator.rank_reference, dtype=np.float32)
    values = np.asarray(expected_ev, dtype=np.float32)
    if not len(ref):
        return np.full(len(values), 0.5, dtype=np.float32)
    mapped = (
        np.searchsorted(ref, values, side="right") / float(len(ref))
    ).astype(np.float32)
    if raw_score is None or calibrator.rank_blend >= 1.0:
        return mapped
    raw = np.asarray(raw_score, dtype=np.float32)
    blend = float(np.clip(calibrator.rank_blend, 0.0, 1.0))
    return ((1.0 - blend) * raw + blend * mapped).astype(np.float32)


def hierarchical_ev_calibrator_payload(
    calibrator: HierarchicalEVCalibrator,
) -> dict[str, Any]:
    """Serialize the calibrator to the JSON contract used by replay/live."""
    def curve(model: IsotonicRegression) -> dict[str, list[float]]:
        return {
            "x": np.asarray(model.X_thresholds_, dtype=float).tolist(),
            "y": np.asarray(model.y_thresholds_, dtype=float).tolist(),
        }

    side_models = getattr(calibrator, "side_models", {})
    side_weights = getattr(calibrator, "side_weights", {})
    side_support = getattr(calibrator, "side_support", {})
    local_models = getattr(calibrator, "local_models", {})
    local_weights = getattr(calibrator, "local_weights", {})
    local_support = getattr(calibrator, "local_support", {})
    has_side_parent = bool(side_models)
    return {
        "schema": "hierarchical_monotonic_expected_ev_v3",
        "unit": "net_return_after_1pct",
        "mapping_scope": (
            "global_to_side_to_side_x_archetype"
            if has_side_parent
            else "side_x_archetype_shrunk_to_global"
        ),
        "rank_reference_scope": (
            "hierarchical_global_side_side_x_archetype_expected_ev"
            if has_side_parent
            else "hierarchical_side_x_archetype_expected_ev"
        ),
        "global": curve(calibrator.global_model),
        "side": {
            side: {
                **curve(model),
                "weight": float(side_weights.get(side, 0.0)),
                "support": int(side_support.get(side, 0)),
            }
            for side, model in side_models.items()
        },
        "local": {
            f"{side}||{arch}": {
                **curve(model),
                "weight": float(local_weights.get((side, arch), 0.0)),
                "support": int(local_support.get((side, arch), 0)),
            }
            for (side, arch), model in local_models.items()
        },
        "rank_reference": np.asarray(
            calibrator.rank_reference, dtype=float
        ).tolist(),
        "shrink_rows": float(calibrator.shrink_rows),
        "local_weight_cap": float(calibrator.local_weight_cap),
        "rank_blend": float(calibrator.rank_blend),
        "monotonic_refinement": {
            "enabled": bool(calibrator.monotonic_refinement_slope > 0.0),
            "slope": float(calibrator.monotonic_refinement_slope),
            "score_min": float(calibrator.refinement_score_min),
            "score_max": float(calibrator.refinement_score_max),
            "centering": 0.5,
            "contract": (
                "A centered strictly increasing score-percentile term refines "
                "isotonic plateaus without changing monotonic ordering."
            ),
        },
    }


def _numeric_matrix(
    frame: pd.DataFrame,
    features: list[str],
    medians: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    arr = x.to_numpy(dtype=np.float32, copy=True)
    if medians is None:
        medians = np.nanmedian(arr, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(medians, np.nonzero(missing)[1])
    if scales is None:
        q25, q75 = np.nanquantile(arr, (0.25, 0.75), axis=0)
        scales = np.asarray(q75 - q25, dtype=np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
    arr = np.clip((arr - medians) / scales, -8.0, 8.0).astype(np.float32)
    return arr, medians.astype(np.float32), scales.astype(np.float32)


def mlp_hidden_embedding(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    """Return the final hidden activation from a fitted sklearn MLP."""
    h = np.asarray(x, dtype=np.float32)
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        h = h @ np.asarray(weights, dtype=np.float32) + np.asarray(
            bias, dtype=np.float32
        )
        if model.activation == "relu":
            np.maximum(h, 0.0, out=h)
        elif model.activation == "tanh":
            np.tanh(h, out=h)
        elif model.activation == "logistic":
            h = (1.0 / (1.0 + np.exp(-h))).astype(np.float32)
    return h.astype(np.float32, copy=False)


def _posterior_ev_prior(
    posterior: np.ndarray,
    ev_residual: np.ndarray,
    *,
    shrink_rows: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = posterior.sum(axis=0).astype(np.float32)
    sums = posterior.T @ ev_residual.astype(np.float32)
    means = np.divide(sums, support, out=np.zeros_like(sums), where=support > 0)
    shrink = support / (support + float(shrink_rows))
    return (means * shrink).astype(np.float32), support


def fit_local_state_encoder(
    frame: pd.DataFrame,
    *,
    side: str,
    archetype: str,
    arm: EncoderArm,
    features: list[str],
    ae_features: list[str],
    ev_residual: np.ndarray,
    hit_residual: np.ndarray,
    sample_weight: np.ndarray,
    n_components: int | Sequence[int] = (3, 4, 5, 6),
    shrink_rows: float = 300.0,
    seed: int = 42,
    mlp_params: dict[str, Any] | None = None,
) -> LocalStateEncoder:
    parent_rank = (
        pd.to_numeric(frame["policy_parent_rank"], errors="coerce")
        if "policy_parent_rank" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    hit_probability = (
        pd.to_numeric(frame["hit_probability"], errors="coerce")
        if "hit_probability" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    reliability_reference = {
        "parent_rank_q90": float(parent_rank.quantile(0.90))
        if parent_rank.notna().any()
        else 0.90,
        "hit_probability_q90": float(hit_probability.quantile(0.90))
        if hit_probability.notna().any()
        else 0.90,
        "support": float(len(frame)),
    }
    # Rebuild the local fields from this encoder's authorized training rows.
    # This keeps the learned scale identical to FrozenLocalMLPOverlay.predict.
    frame = frame.copy(deep=False)
    frame["meta_parent_rank_local_top10_margin"] = (
        parent_rank - reliability_reference["parent_rank_q90"]
    ).astype(np.float32)
    frame["meta_hit_probability_local_top10_margin"] = (
        hit_probability - reliability_reference["hit_probability_q90"]
    ).astype(np.float32)
    frame["meta_parent_reliability_local_support_log1p"] = np.float32(
        np.log1p(reliability_reference["support"])
    )
    x, medians, scales = _numeric_matrix(frame, features)
    ev = np.asarray(ev_residual, dtype=np.float32)
    hit = np.asarray(hit_residual, dtype=np.float32)
    ev_center = float(np.nanmedian(ev))
    ev_scale = float(max(np.nanquantile(np.abs(ev - ev_center), 0.75), 1e-4))
    y = np.column_stack(
        [np.clip((ev - ev_center) / ev_scale, -5.0, 5.0), np.clip(hit, -1.0, 1.0)]
    ).astype(np.float32)
    mlp: MLPRegressor | None = None
    if arm != "ae_gmm":
        params = dict(mlp_params or {})
        hidden = tuple(params.get("hidden_layer_sizes", (32, 16, 8)))
        hit_target_weight = float(params.get("hit_target_weight", 1.0))
        y[:, 1] *= np.float32(hit_target_weight)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            solver="adam",
            alpha=float(params.get("alpha", 0.15)),
            batch_size=min(
                int(params.get("batch_size", 512)), max(64, len(frame) // 20)
            ),
            learning_rate_init=float(params.get("learning_rate_init", 5e-4)),
            max_iter=int(params.get("max_iter", 140)),
            early_stopping=False,
            n_iter_no_change=int(params.get("n_iter_no_change", 12)),
            tol=float(params.get("tol", 1e-4)),
            random_state=seed,
        )
        noise_rng = np.random.default_rng(seed + 409)
        noisy_x = np.clip(
            x + noise_rng.normal(
                0.0, float(params.get("noise_std", 0.03)), size=x.shape
            ).astype(np.float32),
            -8.0,
            8.0,
        )
        mlp.fit(
            noisy_x, y, sample_weight=np.asarray(sample_weight, dtype=np.float32)
        )
    if arm == "mlp_direct":
        return LocalStateEncoder(
            side, archetype, arm, features, medians, scales,
            [], np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
            mlp, None,
            np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
            ev_center, ev_scale, len(frame), reliability_reference,
        )
    ae_cols = [c for c in ae_features if c in frame]
    ae_x, ae_medians, ae_scales = _numeric_matrix(frame, ae_cols) if ae_cols else (
        np.empty((len(frame), 0), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    if arm == "ae_gmm":
        latent = ae_x
    else:
        assert mlp is not None
        mlp_latent = mlp_hidden_embedding(mlp, x)
        latent = mlp_latent if arm == "mlp_gmm" else np.column_stack([ae_x, mlp_latent])
    if latent.shape[1] < 2:
        raise ValueError(f"{arm} requires at least two latent dimensions")
    latent_full = np.asarray(latent, dtype=np.float64)
    if len(latent_full) > 60_000:
        blocks = np.array_split(np.arange(len(latent_full), dtype=np.int64), 3)
        fit_idx = np.concatenate([
            np.linspace(block[0], block[-1], min(20_000, len(block)), dtype=np.int64)
            for block in blocks if len(block)
        ])
        latent_fit = latent_full[fit_idx]
    else:
        latent_fit = latent_full
    requested = (n_components,) if isinstance(n_components, int) else tuple(n_components)
    candidates: list[tuple[float, GaussianMixture]] = []
    min_support = max(100.0, float(shrink_rows) * 0.5)
    for requested_k in requested:
        k = min(int(requested_k), max(2, len(frame) // 500), len(frame) - 1)
        if k < 2 or any(model.n_components == k for _, model in candidates):
            continue
        try:
            candidate = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=3e-3,
                n_init=2,
                max_iter=180,
                random_state=seed + 101 + k,
            ).fit(latent_fit)
        except ValueError:
            continue
        candidate_posterior = candidate.predict_proba(latent_fit)
        effective_support = candidate_posterior.sum(axis=0)
        if float(np.min(effective_support)) < min_support:
            continue
        candidates.append((float(candidate.bic(latent_fit)), candidate))
    if not candidates:
        fallback_k = min(3, max(2, len(frame) // 500), len(frame) - 1)
        gmm = GaussianMixture(
            n_components=fallback_k, covariance_type="diag", reg_covar=1e-3,
            n_init=2, max_iter=180, random_state=seed + 104,
        ).fit(latent_fit)
    else:
        gmm = min(candidates, key=lambda item: item[0])[1]
    posterior = gmm.predict_proba(latent_full).astype(np.float32)
    cluster_ev, cluster_support = _posterior_ev_prior(
        posterior, ev, shrink_rows=shrink_rows
    )
    return LocalStateEncoder(
        side, archetype, arm, features, medians, scales,
        ae_cols, ae_medians, ae_scales, mlp, gmm,
        cluster_ev, cluster_support, ev_center, ev_scale, len(frame), reliability_reference,
    )


def predict_local_state_encoder(
    model: LocalStateEncoder,
    frame: pd.DataFrame,
    *,
    ae_features: list[str],
) -> dict[str, np.ndarray]:
    x, _, _ = _numeric_matrix(frame, model.features, model.medians, model.scales)
    if model.mlp is not None:
        direct = np.asarray(model.mlp.predict(x), dtype=np.float32)
        if direct.ndim == 1:
            direct = direct[:, None]
        direct_ev = direct[:, 0] * model.ev_scale + model.ev_center
    else:
        direct_ev = np.zeros(len(frame), dtype=np.float32)
    if model.arm == "mlp_direct":
        return {
            "ev_correction": direct_ev.astype(np.float32),
            "posterior_confidence": np.ones(len(frame), dtype=np.float32),
            "posterior_entropy": np.zeros(len(frame), dtype=np.float32),
        }
    ae_cols = [c for c in model.ae_features if c in frame]
    ae_x, _, _ = _numeric_matrix(
        frame, ae_cols, model.ae_medians, model.ae_scales
    ) if ae_cols else (
        np.empty((len(frame), 0), dtype=np.float32), None, None
    )
    if model.arm == "ae_gmm":
        latent = ae_x
    else:
        assert model.mlp is not None
        mlp_latent = mlp_hidden_embedding(model.mlp, x)
        latent = mlp_latent if model.arm == "mlp_gmm" else np.column_stack([ae_x, mlp_latent])
    posterior = np.asarray(model.gmm.predict_proba(latent), dtype=np.float32)
    entropy = -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1)
    entropy /= max(float(np.log(posterior.shape[1])), 1e-6)
    confidence = np.max(posterior, axis=1) * (1.0 - entropy)
    correction = posterior @ model.cluster_ev
    return {
        "ev_correction": correction.astype(np.float32),
        "posterior_confidence": np.clip(confidence, 0.0, 1.0).astype(np.float32),
        "posterior_entropy": np.clip(entropy, 0.0, 1.0).astype(np.float32),
    }
