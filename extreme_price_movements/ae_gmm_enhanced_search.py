"""Outcome-free staged DAE/IDEC plus GMM representation search.

This module deliberately keeps encoder selection separate from the supervised
base/meta objective.  It is imported lazily by :mod:`features_gmm_ae` so legacy
states and inference retain their existing numerical path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .alternative_latent_encoders import AlternativeLatentEncoder, EncoderConfig
from .features_denoising_ae import (
    fit_denoising_autoencoder_state,
    refit_denoising_autoencoder_state,
    transform_denoising_autoencoder_features,
)


def _rank01(values: Sequence[float], *, higher: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(len(arr), 0.5, dtype=np.float64)
    fill = float(np.nanmedian(arr[finite]))
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    ranks /= max(len(arr) - 1, 1)
    return ranks if higher else 1.0 - ranks


def _safe_side_values(frame: pd.DataFrame) -> np.ndarray:
    if "side" not in frame.columns:
        return np.repeat("global", len(frame)).astype(object)
    return frame["side"].astype(str).to_numpy(dtype=object, copy=False)


def _latent_scale_fit(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(z, axis=0).astype(np.float32)
    q75 = np.nanquantile(z, 0.75, axis=0)
    q25 = np.nanquantile(z, 0.25, axis=0)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    return center, scale


def _latent_scale_apply(z: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((z - center.reshape(1, -1)) / scale.reshape(1, -1), -12.0, 12.0).astype(
        np.float32, copy=False
    )


def _occupancy_score(labels: np.ndarray, k: int) -> tuple[np.ndarray, float, bool]:
    counts = np.bincount(labels, minlength=int(k)).astype(np.float64)
    occupancy = counts / max(float(counts.sum()), 1.0)
    smallest = float(occupancy.min()) if len(occupancy) else 0.0
    largest = float(occupancy.max()) if len(occupancy) else 1.0
    balance = float(1.0 - np.clip(largest - smallest, 0.0, 1.0))
    return occupancy, balance, bool(smallest >= 0.01 and largest <= 0.80)


def _side_balance(labels: np.ndarray, sides: np.ndarray, k: int) -> tuple[float, bool]:
    if len(sides) != len(labels) or len(set(map(str, sides))) < 2:
        return 1.0, True
    scores: list[float] = []
    valid = True
    for cluster in range(int(k)):
        mask = labels == cluster
        if not mask.any():
            valid = False
            continue
        values = pd.Series(sides[mask]).value_counts(normalize=True)
        if len(values) < 2:
            valid = False
            scores.append(0.0)
            continue
        scores.append(float(np.clip(1.0 - abs(values.iloc[0] - values.iloc[1]), 0.0, 1.0)))
    return (float(np.mean(scores)) if scores else 0.0), valid


def _time_coverage(labels: np.ndarray, timestamps: pd.Series, k: int) -> float:
    if len(timestamps) != len(labels):
        return 0.0
    buckets = pd.to_datetime(timestamps, utc=True, errors="coerce").dt.to_period("M")
    total = max(int(buckets.nunique()), 1)
    coverage: list[float] = []
    for cluster in range(int(k)):
        mask = labels == cluster
        coverage.append(float(buckets[mask].nunique() / total) if mask.any() else 0.0)
    return float(np.mean(coverage)) if coverage else 0.0


def _gmm_log_likelihood(
    z: np.ndarray,
    *,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    covariance_type: str,
) -> float:
    if len(z) == 0:
        return float("-inf")
    x = np.asarray(z, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    dim = max(x.shape[1], 1)
    log_parts: list[np.ndarray] = []
    if covariance_type == "diag":
        vars_ = np.maximum(np.asarray(covariances, dtype=np.float64), 1e-10)
        for idx in range(len(weights)):
            delta = x - means[idx]
            q = np.sum(delta * delta / vars_[idx], axis=1)
            norm = np.sum(np.log(2.0 * np.pi * vars_[idx]))
            log_parts.append(np.log(weights[idx]) - 0.5 * (norm + q))
    else:
        cov = np.asarray(covariances, dtype=np.float64)
        cov = cov if cov.ndim == 2 else cov[0]
        cov = cov + np.eye(dim) * 1e-10
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return float("-inf")
        inv = np.linalg.inv(cov)
        for idx in range(len(weights)):
            delta = x - means[idx]
            q = np.einsum("ij,jk,ik->i", delta, inv, delta, optimize=True)
            log_parts.append(np.log(weights[idx]) - 0.5 * (dim * np.log(2.0 * np.pi) + logdet + q))
    matrix = np.vstack(log_parts)
    maximum = np.max(matrix, axis=0)
    return float(np.mean(maximum + np.log(np.exp(matrix - maximum).sum(axis=0))))


def _bhattacharyya_overlap(
    means: np.ndarray, covariances: np.ndarray, covariance_type: str
) -> float:
    means = np.asarray(means, dtype=np.float64)
    if len(means) < 2:
        return 0.0
    values: list[float] = []
    if covariance_type == "diag":
        vars_ = np.maximum(np.asarray(covariances, dtype=np.float64), 1e-10)
        for left in range(len(means)):
            for right in range(left + 1, len(means)):
                avg = 0.5 * (vars_[left] + vars_[right])
                delta = means[left] - means[right]
                distance = 0.125 * np.sum(delta * delta / avg)
                distance += 0.5 * np.sum(
                    np.log(avg) - 0.5 * (np.log(vars_[left]) + np.log(vars_[right]))
                )
                values.append(float(np.exp(-np.clip(distance, 0.0, 80.0))))
    else:
        cov = np.asarray(covariances, dtype=np.float64)
        cov = cov if cov.ndim == 2 else cov[0]
        inv = np.linalg.pinv(cov)
        for left in range(len(means)):
            for right in range(left + 1, len(means)):
                delta = means[left] - means[right]
                distance = 0.125 * float(delta @ inv @ delta)
                values.append(float(np.exp(-np.clip(distance, 0.0, 80.0))))
    return float(np.mean(values)) if values else 0.0


def _refine_repulsion(
    z_train: np.ndarray,
    *,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    covariance_type: str,
    penalty: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Refine means/weights locally while keeping EM covariance fixed.

    Fixed covariance makes the refinement numerically conservative and keeps a
    tied covariance truly tied.  The lambda=0 EM model remains in the search.
    """
    if penalty <= 0.0:
        return None
    try:
        import torch
    except Exception:
        return None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    if len(x) > 50_000:
        # The local refinement needs a representative density gradient, not all
        # rows.  Deterministic time-spread indexing avoids a random subsample.
        idx = np.linspace(0, len(x) - 1, 50_000, dtype=np.int64)
        x = x[idx]
    mean_param = torch.nn.Parameter(torch.as_tensor(means, dtype=torch.float32, device=device))
    logits = torch.nn.Parameter(torch.log(torch.as_tensor(np.maximum(weights, 1e-8), dtype=torch.float32, device=device)))
    optimizer = torch.optim.Adam([mean_param, logits], lr=2e-3)
    if covariance_type == "diag":
        vars_t = torch.as_tensor(np.maximum(covariances, 1e-10), dtype=torch.float32, device=device)
        log_norm = torch.sum(torch.log(2.0 * np.pi * vars_t), dim=1)
    else:
        cov = np.asarray(covariances, dtype=np.float64)
        cov = cov if cov.ndim == 2 else cov[0]
        inv_t = torch.as_tensor(np.linalg.pinv(cov), dtype=torch.float32, device=device)
        _sign, logdet = np.linalg.slogdet(cov)
        log_norm_shared = float(x.shape[1] * np.log(2.0 * np.pi) + logdet)
    for _ in range(int(steps)):
        delta = x[:, None, :] - mean_param[None, :, :]
        if covariance_type == "diag":
            quad = torch.sum(delta.square() / vars_t[None, :, :], dim=2)
            log_prob = -0.5 * (quad + log_norm[None, :])
            overlap_terms: list[Any] = []
            for left in range(mean_param.shape[0]):
                for right in range(left + 1, mean_param.shape[0]):
                    avg = 0.5 * (vars_t[left] + vars_t[right])
                    d = mean_param[left] - mean_param[right]
                    bd = 0.125 * torch.sum(d.square() / avg)
                    bd = bd + 0.5 * torch.sum(torch.log(avg) - 0.5 * (torch.log(vars_t[left]) + torch.log(vars_t[right])))
                    overlap_terms.append(torch.exp(-torch.clamp(bd, 0.0, 80.0)))
        else:
            quad = torch.einsum("nkd,df,nkf->nk", delta, inv_t, delta)
            log_prob = -0.5 * (quad + log_norm_shared)
            overlap_terms = []
            for left in range(mean_param.shape[0]):
                for right in range(left + 1, mean_param.shape[0]):
                    d = mean_param[left] - mean_param[right]
                    bd = 0.125 * torch.einsum("d,df,f->", d, inv_t, d)
                    overlap_terms.append(torch.exp(-torch.clamp(bd, 0.0, 80.0)))
        nll = -torch.logsumexp(log_prob + torch.log_softmax(logits, dim=0)[None, :], dim=1).mean()
        overlap = torch.stack(overlap_terms).mean() if overlap_terms else torch.zeros((), device=device)
        loss = nll + float(penalty) * overlap
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    refined_means = mean_param.detach().cpu().numpy().astype(np.float32)
    refined_weights = torch.softmax(logits.detach(), dim=0).cpu().numpy().astype(np.float32)
    overlap = _bhattacharyya_overlap(refined_means, covariances, covariance_type)
    return refined_weights, refined_means, overlap


def _score_gmm_records(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    ll_rank = _rank01([float(r["validation_log_likelihood"]) for r in records])
    occupancy_rank = _rank01([float(r["occupancy_balance_score"]) for r in records])
    side_rank = _rank01([float(r["side_balance_score"]) for r in records])
    time_rank = _rank01([float(r["time_coverage_score"]) for r in records])
    overlap_rank = _rank01([float(r["bhattacharyya_overlap"]) for r in records], higher=False)
    for idx, row in enumerate(records):
        # Overlap has a deliberately small weight: EM density fit is the default
        # and a repelled candidate wins only when it retains other properties.
        row["final_score"] = float(
            0.45 * ll_rank[idx]
            + 0.20 * occupancy_rank[idx]
            + 0.20 * side_rank[idx]
            + 0.10 * time_rank[idx]
            + 0.05 * overlap_rank[idx]
        )


def _fit_gmm_records(
    latent: np.ndarray,
    *,
    sides: np.ndarray,
    timestamps: pd.Series,
    clusters: Sequence[int],
    regs: Sequence[float],
    covariance_types: Sequence[str],
    random_state: int,
    repulsion_lambdas: Sequence[float],
    repulsion_top_configs: int,
    repulsion_steps: int,
) -> list[dict[str, Any]]:
    n = len(latent)
    split = max(1, int(round(0.80 * n)))
    split = min(split, max(1, n - max(50, int(0.05 * n))))
    train, valid = latent[:split], latent[split:]
    if len(valid) < 50:
        valid = train
    records: list[dict[str, Any]] = []
    for components in clusters:
        if len(train) < int(components) * 30:
            continue
        for covariance_type in covariance_types:
            for reg in regs:
                try:
                    model = GaussianMixture(
                        n_components=int(components),
                        covariance_type=str(covariance_type),
                        reg_covar=float(reg),
                        n_init=3,
                        max_iter=200,
                        random_state=int(random_state + components * 101 + int(reg * 1e8)),
                    ).fit(train)
                    probabilities = model.predict_proba(latent)
                    labels = probabilities.argmax(axis=1).astype(np.int32)
                    occupancy, occupancy_score, occupancy_ok = _occupancy_score(labels, int(components))
                    side_score, side_ok = _side_balance(labels, sides, int(components))
                    if not occupancy_ok or not side_ok:
                        continue
                    params = {
                        "gmm_weights": model.weights_.astype(np.float32),
                        "gmm_means": model.means_.astype(np.float32),
                        "gmm_covariances": model.covariances_.astype(np.float32),
                        "gmm_covariance_type": str(covariance_type),
                    }
                    records.append(
                        {
                            "n_components": int(components),
                            "covariance_type": str(covariance_type),
                            "reg_covar": float(reg),
                            "search_phase": "ordinary_em",
                            "repulsion_lambda": 0.0,
                            "bhattacharyya_overlap": _bhattacharyya_overlap(
                                params["gmm_means"], params["gmm_covariances"], str(covariance_type)
                            ),
                            "validation_log_likelihood": _gmm_log_likelihood(
                                valid,
                                weights=params["gmm_weights"],
                                means=params["gmm_means"],
                                covariances=params["gmm_covariances"],
                                covariance_type=str(covariance_type),
                            ),
                            "occupancy": occupancy.astype(float).tolist(),
                            "min_occupancy": float(occupancy.min()),
                            "max_occupancy": float(occupancy.max()),
                            "occupancy_balance_score": occupancy_score,
                            "side_balance_score": side_score,
                            "time_coverage_score": _time_coverage(labels, timestamps, int(components)),
                            "converged": bool(model.converged_),
                            "params": params,
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "n_components": int(components),
                            "covariance_type": str(covariance_type),
                            "reg_covar": float(reg),
                            "search_phase": "ordinary_em",
                            "error": str(exc),
                        }
                    )
    valid_records = [r for r in records if "params" in r]
    _score_gmm_records(valid_records)
    leaders = sorted(valid_records, key=lambda row: float(row["final_score"]), reverse=True)[: int(repulsion_top_configs)]
    for base in leaders:
        params = base["params"]
        for penalty in repulsion_lambdas:
            if float(penalty) <= 0.0:
                continue
            refined = _refine_repulsion(
                train,
                weights=np.asarray(params["gmm_weights"]),
                means=np.asarray(params["gmm_means"]),
                covariances=np.asarray(params["gmm_covariances"]),
                covariance_type=str(base["covariance_type"]),
                penalty=float(penalty),
                steps=int(repulsion_steps),
            )
            if refined is None:
                continue
            weights, means, overlap = refined
            refined_params = {**params, "gmm_weights": weights, "gmm_means": means}
            probabilities = _predict_proba(latent, refined_params)
            labels = probabilities.argmax(axis=1).astype(np.int32)
            occupancy, occupancy_score, occupancy_ok = _occupancy_score(labels, int(base["n_components"]))
            side_score, side_ok = _side_balance(labels, sides, int(base["n_components"]))
            if not occupancy_ok or not side_ok:
                continue
            valid_records.append(
                {
                    **{key: value for key, value in base.items() if key not in {"params", "final_score"}},
                    "search_phase": "bhattacharyya_refinement",
                    "repulsion_lambda": float(penalty),
                    "bhattacharyya_overlap": float(overlap),
                    "validation_log_likelihood": _gmm_log_likelihood(
                        valid,
                        weights=weights,
                        means=means,
                        covariances=np.asarray(params["gmm_covariances"]),
                        covariance_type=str(base["covariance_type"]),
                    ),
                    "occupancy": occupancy.astype(float).tolist(),
                    "min_occupancy": float(occupancy.min()),
                    "max_occupancy": float(occupancy.max()),
                    "occupancy_balance_score": occupancy_score,
                    "side_balance_score": side_score,
                    "time_coverage_score": _time_coverage(labels, timestamps, int(base["n_components"])),
                    "params": refined_params,
                }
            )
    _score_gmm_records(valid_records)
    return valid_records


def _predict_proba(z: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    weights = np.maximum(np.asarray(params["gmm_weights"], dtype=np.float64), 1e-12)
    means = np.asarray(params["gmm_means"], dtype=np.float64)
    covariances = np.asarray(params["gmm_covariances"], dtype=np.float64)
    x = np.asarray(z, dtype=np.float64)
    dim = x.shape[1]
    parts: list[np.ndarray] = []
    if str(params["gmm_covariance_type"]) == "diag":
        vars_ = np.maximum(covariances, 1e-10)
        for idx in range(len(weights)):
            delta = x - means[idx]
            parts.append(np.log(weights[idx]) - 0.5 * (np.sum(np.log(2.0 * np.pi * vars_[idx])) + np.sum(delta * delta / vars_[idx], axis=1)))
    else:
        cov = covariances if covariances.ndim == 2 else covariances[0]
        inv = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return np.full((len(x), len(weights)), 1.0 / len(weights), dtype=np.float32)
        for idx in range(len(weights)):
            delta = x - means[idx]
            q = np.einsum("ij,jk,ik->i", delta, inv, delta, optimize=True)
            parts.append(np.log(weights[idx]) - 0.5 * (dim * np.log(2.0 * np.pi) + logdet + q))
    logits = np.vstack(parts).T
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _idec_proxy_score(native: Any) -> float:
    reconstruction = np.asarray(native.reconstruction_error, dtype=np.float64)
    probabilities = np.asarray(native.cluster_probabilities, dtype=np.float64)
    if reconstruction.size == 0 or probabilities.ndim != 2:
        return float("-inf")
    occupancy = probabilities.mean(axis=0)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    entropy /= max(np.log(max(probabilities.shape[1], 2)), 1e-6)
    return float(
        -np.nanmean(reconstruction)
        + 0.20 * (1.0 - np.clip(occupancy.max() - occupancy.min(), 0.0, 1.0))
        - 0.10 * (np.mean(entropy < 0.01) + np.mean(entropy > 0.99))
    )


def _fit_idec_candidates(
    values: np.ndarray,
    sides: np.ndarray,
    *,
    dae_widths: Mapping[int, str],
    random_state: int,
    initial_promote: int = 2,
) -> list[dict[str, Any]]:
    """Hierarchical IDEC search; only promoted configs reach the GMM panel."""
    stage_one: list[dict[str, Any]] = []
    width_order = ("small", "medium")
    for latent_dim in (8, 16):
        preferred = str(dae_widths.get(latent_dim, "small"))
        widths = tuple(dict.fromkeys((preferred,) + width_order))
        for width in widths:
            hidden = 64 if width == "small" else 128
            for clusters in (4, 6, 8, 12):
                for cluster_weight in (1e-3, 1e-2, 1e-1, 1.0):
                    config = EncoderConfig(
                        kind="idec", latent_dim=latent_dim, hidden_dim=hidden,
                        epochs=18, pretraining_fraction=2.0 / 3.0,
                        n_clusters=clusters, cluster_weight=cluster_weight,
                        student_t_df=1.0, target_update_frequency=5,
                        initialization="kmeans++", random_state=random_state + latent_dim * 100 + clusters,
                        device="auto",
                    )
                    try:
                        encoder = AlternativeLatentEncoder(config).fit(values, sides=sides)
                        native = encoder.transform_native(values, sides=sides)
                        stage_one.append(
                            {
                                "encoder": encoder,
                                "native": native,
                                "config": config,
                                "proxy_score": _idec_proxy_score(native),
                                "search_stage": "idec_initial",
                            }
                        )
                    except Exception as exc:
                        stage_one.append({"error": str(exc), "config": config, "proxy_score": float("-inf")})
    eligible = [row for row in stage_one if "encoder" in row]
    leaders = sorted(eligible, key=lambda row: float(row["proxy_score"]), reverse=True)[: int(initial_promote)]
    refined: list[dict[str, Any]] = []
    for leader in leaders:
        base = leader["config"]
        # "Incumbent GMM means" must live in this IDEC encoder's raw latent
        # coordinates.  DAE GMM means are separately scaled and can have a
        # different component count, so using them here would be invalid.
        try:
            initial_means = GaussianMixture(
                n_components=int(base.n_clusters),
                covariance_type="diag",
                reg_covar=0.003,
                n_init=3,
                random_state=int(base.random_state),
            ).fit(np.asarray(leader["native"].latent, dtype=np.float32)).means_
            init_state: Mapping[str, Any] | None = {"gmm_means": initial_means}
        except Exception:
            init_state = None
        # Coordinate refinement keeps this stage practical: each parameter family
        # is varied against the strongest initial setting rather than a 72-point grid.
        variants: list[dict[str, Any]] = []
        for field, values_to_try in {
            "target_update_frequency": (1, 5, 20),
            "student_t_df": (1.0, 5.0),
            "initialization": ("kmeans++", "incumbent_means"),
            "pretraining_fraction": (0.50, 2.0 / 3.0, 0.80),
        }.items():
            for value in values_to_try:
                payload = dict(base.__dict__)
                payload[field] = value
                variants.append(payload)
        best = leader
        for payload in variants:
            if payload.get("initialization") == "incumbent_means" and init_state is None:
                continue
            try:
                config = EncoderConfig(**payload)
                encoder = AlternativeLatentEncoder(config).fit(
                    values,
                    sides=sides,
                    initialization_state=(
                        init_state if payload.get("initialization") == "incumbent_means" else None
                    ),
                )
                native = encoder.transform_native(values, sides=sides)
                candidate = {
                    "encoder": encoder,
                    "native": native,
                    "config": config,
                    "proxy_score": _idec_proxy_score(native),
                    "search_stage": "idec_coordinate_refine",
                }
                if float(candidate["proxy_score"]) > float(best["proxy_score"]):
                    best = candidate
            except Exception:
                continue
        refined.append(best)
    return refined or leaders


def fit_enhanced_ae_gmm_state(
    x_reference: Any,
    *,
    timestamps: Any = None,
    random_state: int,
    ae_max_train_rows: int,
    gmm_max_train_rows: int,
    final_refit_rows: int,
    final_ae_rows: int,
    ae_max_iter: int,
    cluster_candidates: Sequence[int],
    reg_covar_candidates: Sequence[float],
    covariance_type_candidates: Sequence[str],
    repulsion_lambdas: Sequence[float],
    repulsion_top_configs: int,
    repulsion_steps: int,
    encoder_families: Sequence[str],
) -> dict[str, Any]:
    """Fit the enhanced, outcome-free DAE/IDEC plus ordinary/refined GMM search."""
    from . import features_gmm_ae as core

    x_df = core._as_float_frame(x_reference)
    if len(x_df) < 300 or x_df.shape[1] < 2:
        return {"enabled": False, "reason": "insufficient_rows_or_features", "feature_columns": list(x_df.columns)}
    reference_cap = min(len(x_df), max(int(final_refit_rows), int(gmm_max_train_rows), int(ae_max_train_rows)))
    reference_idx = core._time_spread_sample_indices(len(x_df), reference_cap)
    reference = x_df.iloc[reference_idx].reset_index(drop=True)
    initial_gmm_idx = core._time_spread_sample_indices(len(reference), int(gmm_max_train_rows))
    initial_ae_idx = core._time_spread_sample_indices(len(reference), int(ae_max_train_rows))
    gmm_frame = reference.iloc[initial_gmm_idx].reset_index(drop=True)
    ae_frame = reference.iloc[initial_ae_idx].reset_index(drop=True)
    side_gmm = _safe_side_values(gmm_frame)
    if timestamps is None or len(np.asarray(timestamps)) != len(x_df):
        time_reference = pd.Series(pd.date_range("2000-01-01", periods=len(x_df), freq="h", tz="UTC"))
    else:
        time_reference = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
        if bool(time_reference.isna().any()):
            raise ValueError("Enhanced AE/GMM search requires valid UTC timestamps")
    time_reference = time_reference.iloc[reference_idx].reset_index(drop=True)
    time_gmm = time_reference.iloc[initial_gmm_idx].reset_index(drop=True)
    center, scale = core._robust_scale_fit(ae_frame)
    dae_state = fit_denoising_autoencoder_state(
        core._robust_scale_apply(ae_frame, center, scale),
        random_state=int(random_state), max_train_rows=int(ae_max_train_rows), max_iter=int(ae_max_iter),
    )
    if not bool(dae_state.get("enabled", False)):
        return {"enabled": False, "reason": "dae_fit_failed", "feature_columns": list(x_df.columns)}
    dae_features = transform_denoising_autoencoder_features(
        core._robust_scale_apply(gmm_frame, center, scale), dae_state
    )
    dae_widths = {
        dim: str((dae_state.get("report", {}).get(f"b{dim}", {}) or {}).get("best_width", "small"))
        for dim in (8, 16)
    }
    candidates: list[dict[str, Any]] = []
    if "dae" in set(encoder_families):
        for dim in (8, 16):
            columns = [f"ae_b{dim}_{i:02d}" for i in range(dim)]
            if not set(columns).issubset(dae_features.columns):
                continue
            candidates.append(
                {
                    "family": "dae", "latent_dim": dim, "candidate_key": f"dae_b{dim}",
                    "encoder_state": dae_state,
                    "encoder_config": {"kind": "dae", "selected_bottleneck": dim},
                    "latent": dae_features[columns].to_numpy(dtype=np.float32, copy=False),
                    "reconstruction": dae_features[f"ae_b{dim}_reconstruction_error"].to_numpy(dtype=np.float32, copy=False),
                }
            )
    all_records: list[dict[str, Any]] = []
    for candidate in candidates:
        latent_center, latent_scale = _latent_scale_fit(candidate["latent"])
        normalized = _latent_scale_apply(candidate["latent"], latent_center, latent_scale)
        records = _fit_gmm_records(
            normalized, sides=side_gmm, timestamps=time_gmm,
            clusters=cluster_candidates, regs=reg_covar_candidates,
            covariance_types=covariance_type_candidates, random_state=random_state + candidate["latent_dim"],
            repulsion_lambdas=repulsion_lambdas, repulsion_top_configs=repulsion_top_configs,
            repulsion_steps=repulsion_steps,
        )
        for row in records:
            row["candidate"] = candidate
            row["latent_center"] = latent_center
            row["latent_scale"] = latent_scale
            all_records.append(row)
    if "idec" in set(encoder_families):
        idec_rows = _fit_idec_candidates(
            ae_frame.to_numpy(dtype=np.float32, copy=False), _safe_side_values(ae_frame),
            dae_widths=dae_widths, random_state=random_state + 19_000,
        )
        # IDEC proxy fits use the AE sample.  Refit each promoted encoder once on
        # the GMM reference before density screening so every GMM sees 100k rows.
        for idec_index, idec in enumerate(idec_rows):
            encoder = idec.get("encoder")
            if encoder is None:
                continue
            config = idec["config"]
            try:
                init_state = (
                    idec["encoder"].to_state()
                    if str(config.initialization) == "incumbent_means"
                    else None
                )
                encoder = AlternativeLatentEncoder(config).fit(
                    gmm_frame.to_numpy(dtype=np.float32, copy=False), sides=side_gmm,
                    initialization_state=init_state,
                )
                native = encoder.transform_native(gmm_frame.to_numpy(dtype=np.float32, copy=False), sides=side_gmm)
            except Exception:
                continue
            latent_center, latent_scale = _latent_scale_fit(native.latent)
            normalized = _latent_scale_apply(native.latent, latent_center, latent_scale)
            records = _fit_gmm_records(
                normalized, sides=side_gmm, timestamps=time_gmm,
                clusters=cluster_candidates, regs=reg_covar_candidates,
                covariance_types=covariance_type_candidates, random_state=random_state + 29_000 + int(config.latent_dim),
                repulsion_lambdas=repulsion_lambdas, repulsion_top_configs=repulsion_top_configs,
                repulsion_steps=repulsion_steps,
            )
            candidate = {
                "family": "idec", "latent_dim": int(config.latent_dim),
                "candidate_key": f"idec_{idec_index}_b{config.latent_dim}_k{config.n_clusters}",
                "encoder_state": encoder.to_state(), "encoder_config": dict(config.__dict__),
                "latent": native.latent, "reconstruction": native.reconstruction_error,
                "idec_proxy_score": float(idec.get("proxy_score", float("nan"))),
            }
            for row in records:
                row["candidate"] = candidate
                row["latent_center"] = latent_center
                row["latent_scale"] = latent_scale
                all_records.append(row)
    valid = [row for row in all_records if "params" in row]
    if not valid:
        return {"enabled": False, "reason": "no_valid_enhanced_gmm", "feature_columns": list(x_df.columns)}
    _score_gmm_records(valid)
    winner = max(valid, key=lambda row: float(row["final_score"]))
    chosen = winner["candidate"]
    final_idx = core._time_spread_sample_indices(len(reference), int(final_refit_rows))
    final_frame = reference.iloc[final_idx].reset_index(drop=True)
    final_ae_idx = core._time_spread_sample_indices(len(final_frame), int(final_ae_rows))
    final_ae_frame = final_frame.iloc[final_ae_idx].reset_index(drop=True)
    final_center, final_scale = core._robust_scale_fit(final_ae_frame)
    if chosen["family"] == "dae":
        final_encoder_state = refit_denoising_autoencoder_state(
            core._robust_scale_apply(final_ae_frame, final_center, final_scale),
            selected_state=chosen["encoder_state"], random_state=random_state + 70_001,
            max_iter=int(ae_max_iter),
        )
        dim = int(chosen["latent_dim"])
        final_features = transform_denoising_autoencoder_features(
            core._robust_scale_apply(final_frame, final_center, final_scale), final_encoder_state
        )
        final_latent = final_features[[f"ae_b{dim}_{i:02d}" for i in range(dim)]].to_numpy(dtype=np.float32, copy=False)
        final_recon = final_features[f"ae_b{dim}_reconstruction_error"].to_numpy(dtype=np.float32, copy=False)
        latent_encoder_state: Any = final_encoder_state
    else:
        config = EncoderConfig(**dict(chosen["encoder_config"]))
        init_state = (
            chosen["encoder_state"]
            if str(config.initialization) == "incumbent_means"
            else None
        )
        encoder = AlternativeLatentEncoder(config).fit(
            final_ae_frame.to_numpy(dtype=np.float32, copy=False), sides=_safe_side_values(final_ae_frame),
            initialization_state=init_state,
        )
        native = encoder.transform_native(final_frame.to_numpy(dtype=np.float32, copy=False), sides=_safe_side_values(final_frame))
        final_latent, final_recon = native.latent, native.reconstruction_error
        latent_encoder_state = encoder.to_state()
        dim = int(config.latent_dim)
    final_latent_center, final_latent_scale = _latent_scale_fit(final_latent)
    final_normalized = _latent_scale_apply(final_latent, final_latent_center, final_latent_scale)
    params = dict(winner["params"])
    final_model = GaussianMixture(
        n_components=int(winner["n_components"]), covariance_type=str(winner["covariance_type"]),
        reg_covar=float(winner["reg_covar"]), n_init=5, max_iter=300, random_state=random_state + 70_019,
    ).fit(final_normalized)
    params = {
        "gmm_weights": final_model.weights_.astype(np.float32), "gmm_means": final_model.means_.astype(np.float32),
        "gmm_covariances": final_model.covariances_.astype(np.float32), "gmm_covariance_type": str(final_model.covariance_type),
    }
    if float(winner.get("repulsion_lambda", 0.0)) > 0.0:
        refined = _refine_repulsion(
            final_normalized, weights=params["gmm_weights"], means=params["gmm_means"],
            covariances=params["gmm_covariances"], covariance_type=str(winner["covariance_type"]),
            penalty=float(winner["repulsion_lambda"]), steps=int(repulsion_steps),
        )
        if refined is not None:
            weights, means, _overlap = refined
            params["gmm_weights"], params["gmm_means"] = weights, means
    recon_mean = float(np.nanmean(final_recon))
    recon_std = float(np.nanstd(final_recon) + 1e-6)
    gmm_state = {**params, "gmm_reg_covar": float(winner["reg_covar"])}
    _dist, mahal = core._gmm_distances(final_normalized, gmm_state)
    min_mahal = np.min(mahal, axis=1)
    reports = []
    for row in sorted(valid, key=lambda item: float(item["final_score"]), reverse=True):
        safe = {key: value for key, value in row.items() if key not in {"params", "candidate", "latent_center", "latent_scale"}}
        safe["encoder_family"] = row["candidate"]["family"]
        safe["encoder_latent_dim"] = int(row["candidate"]["latent_dim"])
        safe["candidate_key"] = str(row["candidate"]["candidate_key"])
        reports.append(safe)
    selected = next(
        report
        for report in reports
        if report["candidate_key"] == chosen["candidate_key"]
        and report["n_components"] == int(winner["n_components"])
        and report["search_phase"] == winner["search_phase"]
        and abs(float(report["repulsion_lambda"]) - float(winner["repulsion_lambda"])) < 1e-12
    )
    return {
        "enabled": True, "schema_version": "ae_gmm_v3_enhanced", "enhanced_search": True,
        "learned_transform_hash_version": core.AE_GMM_TRANSFORM_HASH_V3,
        "feature_columns": list(x_df.columns), "input_feature_order_hash": core.ae_gmm_input_feature_order_hash(list(x_df.columns)),
        "center": final_center.astype(float).tolist(), "scale": final_scale.astype(float).tolist(),
        "cycle_input_fill_values": {str(col): float(final_center[pos]) for pos, col in enumerate(x_df.columns)},
        "clip": [-8.0, 8.0], "latent_encoder_kind": chosen["family"], "latent_encoder_state": latent_encoder_state,
        "latent_dim": int(dim), "latent_gmm_center": final_latent_center.astype(float).tolist(), "latent_gmm_scale": final_latent_scale.astype(float).tolist(),
        "ae_state": latent_encoder_state if chosen["family"] == "dae" else {},
        "train_rows_available": int(len(x_df)), "ae_fit_rows": int(len(final_ae_frame)), "gmm_fit_rows": int(len(final_frame)),
        "ae_max_train_rows": int(final_ae_rows), "gmm_max_train_rows": int(final_refit_rows),
        "initial_ae_fit_rows": int(len(ae_frame)), "initial_gmm_fit_rows": int(len(gmm_frame)),
        "final_refit_rows": int(len(final_frame)), "final_refit_all_rows": False,
        "sample_policy": "cycle_reference_beginning_middle_end_time_spread",
        "latent_columns": [f"dae_b16_{idx:02d}" for idx in range(int(dim))],
        "gmm_n_components": int(winner["n_components"]), "gmm_covariance_type": str(winner["covariance_type"]),
        "gmm_numeric_contract": core.AE_GMM_NUMERIC_CONTRACT, "gmm_reg_covar": float(winner["reg_covar"]),
        "gmm_weights": params["gmm_weights"].astype(float).tolist(), "gmm_means": params["gmm_means"].astype(float).tolist(),
        "gmm_covariances": params["gmm_covariances"].astype(float).tolist(), "smooth_lambda": 0.0,
        "max_components": int(max(cluster_candidates)), "reconstruction_error_mean": recon_mean, "reconstruction_error_std": recon_std,
        "ood_mahal_q95": float(np.nanquantile(min_mahal, 0.95)), "ood_mahal_q99": float(np.nanquantile(min_mahal, 0.99)),
        "ood_reconstruction_q95": float(recon_mean + 1.645 * recon_std), "ood_reconstruction_q99": float(recon_mean + 2.326 * recon_std),
        "representation_selection_outcome_free": True, "representation_selection_context_keys": ["side", "time_bucket"], "representation_selection_outcome_keys": [],
        "temporal_feature_contract": "row_independent_v1", "selected_config": selected, "top_configs": reports[:12], "hpo_reports": reports,
        "hpo_report_count": int(len(reports)),
        "hpo_grid": {
            "encoder_families": list(encoder_families), "initial_ae_rows": int(ae_max_train_rows), "initial_gmm_rows": int(gmm_max_train_rows),
            "final_ae_rows": int(final_ae_rows), "final_gmm_rows": int(final_refit_rows), "cluster_candidates": [int(v) for v in cluster_candidates],
            "covariance_type_candidates": list(covariance_type_candidates), "reg_covar_candidates": [float(v) for v in reg_covar_candidates],
            "repulsion_penalty": "Bhattacharyya_coefficient", "repulsion_lambdas": [float(v) for v in repulsion_lambdas],
            "repulsion_top_configs": int(repulsion_top_configs), "repulsion_steps": int(repulsion_steps),
            "idec_initial": {"latent_dim": [8, 16], "clusters": [4, 6, 8, 12], "cluster_weight": [1e-3, 1e-2, 1e-1, 1.0], "student_t_df": [1.0], "target_update_frequency": [5], "initialization": ["kmeans++"], "pretraining_fraction": [2.0 / 3.0]},
            "idec_refinement": {"target_update_frequency": [1, 5, 20], "student_t_df": [1.0, 5.0], "initialization": ["kmeans++", "incumbent_means"], "pretraining_fraction": [0.50, 2.0 / 3.0, 0.80]},
        },
    }
