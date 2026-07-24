#!/usr/bin/env python3
"""Leakage-safe fixed-activity MLP replacement test for the V9 tail.

This is intentionally a non-promoting research runner.  It tests whether a
pooled long/short residual model can replace marginal V9-tail decisions using
the full inference-available side contract.  All selection, residual targets,
EV maps and policy gates are fitted only on rows before the scored month.

The current full-contract cache starts in April 2026, therefore the strict
walk-forward evaluation is May-July 2026.  The manifest makes this boundary
explicit instead of silently substituting an older, narrower feature contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.lgbm_pipeline import _recent_feature_coverage_survivors  # noqa: E402
from extreme_price_movements.supervised_market_state_calibration import (  # noqa: E402
    fit_hierarchical_ev_calibrator,
    predict_hierarchical_ev,
)
from scripts.run_meta_market_state_encoder_ablation import _select_ev_features  # noqa: E402


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOME_TOKENS = (
    "target", "label", "future", "oracle", "realized_", "bad_mae",
    "timeout", "full_stop", "exec_margin", "ev_after_1pct", "clean_exec",
    "dirty_positive", "first_touch", "mfe", "mae", "outcome", "return",
)
RELIABILITY_TOKENS = (
    "leaf", "support", "uncertainty", "ood", "mahal", "reconstruction",
    "posterior", "entropy", "drift", "gmm", "aegmm", "cluster",
)
SUBTYPE_STATE_TOKENS = (
    "mkt_", "market_", "xasset", "xs_", "oi_", "funding", "breadth",
    "shock", "entropy", "vol", "volume", "range", "recovery", "liquidation",
    "delever", "gmm", "aegmm", "mahal", "reconstruction", "cluster",
)
JULY_DRIFT_FIELDS = (
    "mark_perp_dislocation", "seasonality_strength", "spike_score",
    "xs_dispersion__funding_per_hour",
    "q_tail_width__bars_in_high_vol_state_log_norm", "oi_vol_10d_robust_z",
    "ffd_rv_2h_06", "ob_mid_close_dislocation_bps_z_24h", "rv_24h_peer_resid",
)
PROTECTED_ANCHORS = (
    "score", "base_score_rank_pct_train_prior", "base_margin_to_cutoff",
    "base_margin_to_cutoff_z", "base_signal_zscore_within_archetype",
    "policy_parent_rank", "hit_probability",
    "existing_sparse_parent_score",
)


@dataclass
class SideModel:
    side: str
    features: list[str]
    categories: list[str]
    medians: np.ndarray
    scales: np.ndarray
    target_center: np.ndarray
    target_scale: np.ndarray
    model: MLPRegressor
    ood_q50: float
    ood_q95: float
    archetype_support: dict[str, int]
    params: dict[str, Any]


@dataclass
class LocalSubtypeModel:
    side: str
    archetype: str
    features: list[str]
    medians: np.ndarray
    scales: np.ndarray
    encoder_kind: str
    pca: PCA | None
    ae: MLPRegressor | None
    gmm: GaussianMixture
    fit_rows: int
    selected_components: int
    train_cluster_ev: list[float]


def _num(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), default, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _feature_ok(name: str) -> bool:
    lower = str(name).lower()
    if name in PROTECTED_ANCHORS or name in JULY_DRIFT_FIELDS:
        return True
    return not any(token in lower for token in OUTCOME_TOKENS)


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    wanted = None if columns is None else [c for c in columns if c in available]
    return pd.read_parquet(path, columns=wanted)


def _merge_missing(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    extra = [c for c in right.columns if c not in left.columns and c not in KEYS]
    if not extra:
        return left
    source = right.loc[:, [*KEYS, *extra]].drop_duplicates(KEYS, keep="last")
    return left.merge(source, on=KEYS, how="left", validate="one_to_one", copy=False)


def _merge_state_context(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join an externally materialized decision-time state contract by keys."""
    available = set(pq.ParquetFile(path).schema.names)
    state_columns = [name for name in available if name.startswith("state__")]
    context = _read_parquet(path, [*KEYS, *state_columns])
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True, errors="coerce")
    return _merge_missing(frame, context)


def _load_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = _read_parquet(args.canonical_oos)
    sparse = _read_parquet(args.sparse_mlp_oos)
    ledger_schema = pq.ParquetFile(args.scored_ledger).schema.names
    ledger_extra = [
        c for c in ledger_schema
        if c in KEYS
        or c == "__archetype_policy_key__"
        or any(token in c.lower() for token in RELIABILITY_TOKENS)
        or c.startswith("regime_lgbm_leaf_")
    ]
    ledger = _read_parquet(args.scored_ledger, ledger_extra)
    if "archetype_policy_key" not in ledger and "__archetype_policy_key__" in ledger:
        ledger = ledger.rename(columns={"__archetype_policy_key__": "archetype_policy_key"})
    frame = _merge_missing(canonical, sparse)
    frame = _merge_missing(frame, ledger)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].notna()].copy()
    frame["side_name"] = frame["side_name"].astype(str)
    frame["archetype_policy_key"] = frame["archetype_policy_key"].astype(str)
    frame = frame.sort_values("__ts__", kind="stable").reset_index(drop=True)
    duplicate_rows = int(frame.duplicated(KEYS).sum())
    if duplicate_rows:
        raise ValueError(f"merged source has duplicate keys={duplicate_rows}")
    with args.canonical_manifest.open() as handle:
        manifest = json.load(handle)
    audit = {
        "canonical_rows": int(len(canonical)),
        "sparse_context_rows": int(len(sparse)),
        "ledger_context_rows": int(len(ledger)),
        "merged_rows": int(len(frame)),
        "merged_columns": int(len(frame.columns)),
        "canonical_feature_contract": manifest.get("feature_contract", {}),
        "required_july_drift_present": {
            key: bool(key in frame and frame[key].notna().any())
            for key in JULY_DRIFT_FIELDS
        },
    }
    return frame, audit


def _direct_causal_residual(frame: pd.DataFrame) -> np.ndarray:
    """Residual to a train-only side x archetype rank-to-EV expectation."""
    ordered = frame.sort_values("__ts__", kind="stable").reset_index(drop=True)
    ev = _num(ordered, "ev_after_1pct")
    rank = _num(ordered, "policy_parent_rank", 0.5)
    arch = ordered["archetype_policy_key"].astype(str).to_numpy()
    blocks = np.array_split(np.arange(len(ordered), dtype=np.int32), 5)
    result = np.full(len(ordered), np.nan, dtype=np.float32)
    for fold in range(1, len(blocks)):
        train_idx = np.concatenate(blocks[:fold])
        valid_idx = blocks[fold]
        if len(train_idx) < 400 or not len(valid_idx):
            continue
        edges = np.unique(np.quantile(rank[train_idx], np.linspace(0.0, 1.0, 13)))
        if len(edges) < 4:
            result[valid_idx] = ev[valid_idx] - np.float32(np.mean(ev[train_idx]))
            continue
        bins = len(edges) - 1
        train_bin = np.clip(np.searchsorted(edges, rank[train_idx], side="right") - 1, 0, bins - 1)
        global_sum = np.bincount(train_bin, weights=ev[train_idx], minlength=bins)
        global_n = np.bincount(train_bin, minlength=bins).astype(np.float32)
        global_mean = float(np.mean(ev[train_idx]))
        global_ev = (global_sum + 100.0 * global_mean) / (global_n + 100.0)
        valid_bin = np.clip(np.searchsorted(edges, rank[valid_idx], side="right") - 1, 0, bins - 1)
        expected = global_ev[valid_bin].astype(np.float32)
        for archetype in np.unique(arch[train_idx]):
            local_train = train_idx[arch[train_idx] == archetype]
            local_valid_mask = arch[valid_idx] == archetype
            if not len(local_train) or not local_valid_mask.any():
                continue
            local_bin = np.clip(np.searchsorted(edges, rank[local_train], side="right") - 1, 0, bins - 1)
            local_sum = np.bincount(local_bin, weights=ev[local_train], minlength=bins)
            local_n = np.bincount(local_bin, minlength=bins).astype(np.float32)
            local_ev = (local_sum + 120.0 * global_ev) / (local_n + 120.0)
            local_weight = min(0.75, len(local_train) / (len(local_train) + 900.0))
            positions = np.flatnonzero(local_valid_mask)
            expected[positions] = (
                (1.0 - local_weight) * global_ev[valid_bin[positions]]
                + local_weight * local_ev[valid_bin[positions]]
            )
        result[valid_idx] = ev[valid_idx] - expected
    out = np.full(len(frame), np.nan, dtype=np.float32)
    out[ordered.index.to_numpy()] = result
    return out


def _tail_fit_mask(frame: pd.DataFrame) -> np.ndarray:
    rank = _num(frame, "policy_parent_rank", 0.5)
    return (rank >= 0.80) & (rank <= 0.995)


def _rank_weights(frame: pd.DataFrame) -> np.ndarray:
    rank = _num(frame, "policy_parent_rank", 0.5)
    cutoff_focus = np.exp(-0.5 * ((rank - 0.90) / 0.035) ** 2)
    return (0.35 + 1.00 * (rank >= 0.80) + 3.0 * cutoff_focus).astype(np.float32)


def _joint_coverage_features(
    frame: pd.DataFrame, candidates: list[str]
) -> tuple[list[str], dict[str, Any]]:
    available = [c for c in candidates if c in frame]
    survivors, report = _recent_feature_coverage_survivors(
        frame.loc[:, available],
        frame["__ts__"].to_numpy(),
        require_joint_complete_case=True,
        min_feature_coverage=0.90,
        coverage_scope="all_post_warmup",
        warmup_days=30,
        warmup_reference_start=frame["__ts__"].min(),
    )
    survivors = list(dict.fromkeys([*PROTECTED_ANCHORS, *survivors]))
    survivors = [c for c in survivors if c in frame]
    return survivors, report


def _select_side_features(
    train: pd.DataFrame,
    candidates: list[str],
    side: str,
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    group = train.loc[train["side_name"].eq(side)].copy()
    covered, coverage = _joint_coverage_features(group, candidates)
    selected, report = _select_ev_features(
        group,
        covered,
        max_features=None,
        seed=seed,
        auto_feature_ceiling=72,
    )
    selected = list(dict.fromkeys([
        *[f for f in PROTECTED_ANCHORS if f in covered], *selected,
    ]))
    if not report.empty:
        report.insert(0, "side_name", side)
        report["protected_anchor"] = report["feature"].isin(PROTECTED_ANCHORS)
    return selected, report, coverage


def _subtype_state_candidates(frame: pd.DataFrame) -> list[str]:
    """Observable market/latent state inputs, excluding score and outcome priors."""
    result: list[str] = []
    for name in frame.columns:
        lower = str(name).lower()
        if not _feature_ok(name) or name in PROTECTED_ANCHORS:
            continue
        if any(token in lower for token in SUBTYPE_STATE_TOKENS):
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.notna().mean() >= 0.85 and values.nunique(dropna=True) >= 8:
                result.append(name)
    return result


def _time_spread_sample_positions(length: int, cap: int) -> np.ndarray:
    if length <= cap:
        return np.arange(length, dtype=np.int32)
    blocks = np.array_split(np.arange(length, dtype=np.int32), 3)
    per_block = max(1, cap // 3)
    return np.concatenate([
        np.linspace(block[0], block[-1], min(per_block, len(block)), dtype=np.int32)
        for block in blocks if len(block)
    ])


def _subtype_matrix(
    frame: pd.DataFrame,
    features: list[str],
    medians: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    if medians is None:
        medians = np.nanmedian(raw, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(raw)
    if missing.any():
        raw[missing] = np.take(medians, np.nonzero(missing)[1])
    if scales is None:
        q25, q75 = np.quantile(raw, (0.25, 0.75), axis=0)
        scales = (q75 - q25).astype(np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
    return np.clip((raw - medians) / scales, -8.0, 8.0).astype(np.float32), medians, scales


def _fit_denoising_autoencoder(x: np.ndarray, seed: int) -> MLPRegressor:
    """Fit a conservative train-only denoising encoder for local state discovery.

    The model reconstructs pre-entry state inputs only.  It is deliberately
    regularized and uses a fixed training budget, avoiding a target-derived
    early-stopping split or a high-capacity local representation.
    """
    latent_dim = min(8, max(4, x.shape[1] - 2))
    encoder = MLPRegressor(
        hidden_layer_sizes=(16, latent_dim, 16),
        activation="relu",
        solver="adam",
        alpha=1.5,
        batch_size=min(1024, max(128, len(x) // 20)),
        learning_rate_init=3e-4,
        max_iter=120,
        shuffle=False,
        early_stopping=False,
        tol=2e-4,
        random_state=seed,
    )
    rng = np.random.default_rng(seed + 37)
    noisy = np.clip(x + rng.normal(0.0, 0.12, size=x.shape).astype(np.float32), -8.0, 8.0)
    encoder.fit(noisy, x)
    return encoder


def _ae_latent(encoder: MLPRegressor, x: np.ndarray) -> np.ndarray:
    """Extract the central hidden layer from sklearn's symmetric AE."""
    hidden = x.astype(np.float32, copy=False)
    # The first two hidden layers are encoder layers for 16 -> latent -> 16.
    for weights, bias in zip(encoder.coefs_[:2], encoder.intercepts_[:2]):
        hidden = np.maximum(hidden @ weights + bias, 0.0, dtype=np.float32)
    return hidden.astype(np.float32, copy=False)


def _subtype_latent(model: LocalSubtypeModel, x: np.ndarray) -> np.ndarray:
    if model.encoder_kind == "ae":
        if model.ae is None:
            raise ValueError("AE subtype model missing encoder")
        return _ae_latent(model.ae, x)
    if model.pca is None:
        raise ValueError("PCA subtype model missing encoder")
    return model.pca.transform(x).astype(np.float32)


def _fit_local_subtype(
    group: pd.DataFrame,
    candidates: list[str],
    side: str,
    archetype: str,
    seed: int,
    encoder_kind: str,
    tail_only: bool = True,
    objective_kind: str = "ev",
) -> tuple[LocalSubtypeModel | None, dict[str, Any]]:
    """Fit one side x archetype market-state GMM from observable train rows."""
    tail = group.loc[_tail_fit_mask(group) if tail_only else np.ones(len(group), dtype=bool)].sort_values("__ts__", kind="stable").reset_index(drop=True)
    if len(tail) < 3_000:
        return None, {"side": side, "archetype": archetype, "status": "insufficient_support", "rows": int(len(tail))}
    selection_frame = tail.copy()
    # Path outcomes are train-only supervision. The AE/GMM never receives
    # them as transform inputs; they only decide which observable state fields
    # and density split best isolate adverse high-confidence decisions.
    if objective_kind == "path_precision":
        clean = _num(selection_frame, "clean_exec")
        bad = _num(selection_frame, "full_path_bad_mae_1r")
        timeout = _num(selection_frame, "timeout")
        ev = np.clip(_num(selection_frame, "ev_after_1pct") / 0.02, -2.0, 2.0)
        base_conf = _num(selection_frame, "base_score_rank_pct_train_prior", _num(selection_frame, "policy_parent_rank", 0.5))
        # High base confidence paired with an ugly path is the exact residual
        # error the state layer should learn to demote.
        overconfident_path_failure = base_conf * np.maximum(bad + 0.6 * timeout - clean, 0.0)
        selection_frame["ev_after_1pct"] = (
            0.55 * clean - 0.85 * bad - 0.55 * timeout + 0.35 * ev
            - 0.75 * overconfident_path_failure
        ).astype(np.float32)
    selected, feature_report = _select_ev_features(
        selection_frame, candidates, max_features=10, seed=seed, auto_feature_ceiling=10
    )
    selected = [name for name in selected if name in tail]
    if len(selected) < 4:
        return None, {"side": side, "archetype": archetype, "status": "insufficient_state_features", "rows": int(len(tail))}
    x_all, medians, scales = _subtype_matrix(tail, selected)
    sample_pos = _time_spread_sample_positions(len(tail), 20_000)
    x_fit = x_all[sample_pos]
    pca: PCA | None = None
    ae: MLPRegressor | None = None
    if encoder_kind == "ae":
        ae = _fit_denoising_autoencoder(x_fit, seed)
        z_fit = _ae_latent(ae, x_fit)
        z_all = _ae_latent(ae, x_all)
        representation_detail = {
            "encoder_kind": "denoising_ae",
            "latent_dim": int(z_fit.shape[1]),
            "noise_std": 0.12,
            "l2_alpha": 1.5,
        }
    else:
        pca_dim = min(5, x_fit.shape[1], max(2, x_fit.shape[0] // 300))
        pca = PCA(n_components=pca_dim, whiten=False, random_state=seed).fit(x_fit)
        z_fit = pca.transform(x_fit).astype(np.float32)
        z_all = pca.transform(x_all).astype(np.float32)
        representation_detail = {
            "encoder_kind": "pca",
            "latent_dim": int(pca_dim),
            "pca_variance": float(pca.explained_variance_ratio_.sum()),
        }
    residual = _direct_causal_residual(selection_frame)
    clean_target = _num(tail, "clean_exec") > 0.5
    base_confidence = _num(tail, "base_score_rank_pct_train_prior", _num(tail, "policy_parent_rank", 0.5))
    best: tuple[float, GaussianMixture, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    for components in range(3, 7):
        if len(z_fit) < components * 250:
            continue
        gmm = GaussianMixture(
            n_components=components, covariance_type="diag", reg_covar=1e-3,
            n_init=2, max_iter=180, random_state=seed + components,
        ).fit(z_fit)
        labels = gmm.predict(z_all)
        support = np.bincount(labels, minlength=components)
        min_share = float(support.min() / max(len(labels), 1))
        valid = np.isfinite(residual)
        cluster_ev = np.array([
            float(np.nanmean(residual[(labels == cluster) & valid]))
            if np.any((labels == cluster) & valid) else 0.0
            for cluster in range(components)
        ])
        separation = float(np.std(cluster_ev))
        ap_values = []
        for cluster in range(components):
            cluster_mask = labels == cluster
            if cluster_mask.sum() < 200 or clean_target[cluster_mask].min() == clean_target[cluster_mask].max():
                continue
            ap_values.append(float(average_precision_score(clean_target[cluster_mask], base_confidence[cluster_mask])))
        base_ap_dispersion = float(np.std(ap_values)) if len(ap_values) > 1 else 0.0
        # BIC establishes the state density; a small train-only economic term
        # avoids selecting a geometrically neat but behaviorally inert split.
        score = float(gmm.bic(z_fit) / len(z_fit) - 0.20 * separation - 0.08 * base_ap_dispersion + (0.5 if min_share < 0.05 else 0.0))
        rows.append({"components": components, "bic_per_row": float(gmm.bic(z_fit) / len(z_fit)), "min_share": min_share, "train_residual_separation": separation, "base_average_precision_dispersion": base_ap_dispersion, "selection_score": score})
        if best is None or score < best[0]:
            best = (score, gmm, cluster_ev)
    if best is None:
        return None, {"side": side, "archetype": archetype, "status": "no_supported_gmm", "rows": int(len(tail))}
    _score, gmm, cluster_ev = best
    model = LocalSubtypeModel(
        side=side, archetype=archetype, features=selected, medians=medians,
        scales=scales, encoder_kind=encoder_kind, pca=pca, ae=ae, gmm=gmm, fit_rows=int(len(tail)),
        selected_components=int(gmm.n_components), train_cluster_ev=cluster_ev.astype(float).tolist(),
    )
    return model, {
        "side": side, "archetype": archetype, "status": "fit",
        "rows": int(len(tail)), "features": selected,
        **representation_detail,
        "selected_components": int(gmm.n_components), "cluster_ev_residual": model.train_cluster_ev,
        "grid": rows,
        "objective_kind": objective_kind,
        "feature_report": feature_report.loc[feature_report["selected"], ["feature", "final_score", "conditional_oof_gain", "weighted_binned_mi"]].to_dict("records"),
    }


def _append_local_subtypes(
    train: pd.DataFrame,
    score: pd.DataFrame,
    seed: int,
    encoder_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[LocalSubtypeModel]]:
    """Append frozen local subtype posteriors; no outcome reaches score rows."""
    train = train.copy()
    score = score.copy()
    candidates = _subtype_state_candidates(train)
    catalog: list[dict[str, Any]] = []
    models: list[LocalSubtypeModel] = []
    for index, ((side, archetype), group) in enumerate(
        train.groupby(["side_name", "archetype_policy_key"], observed=True, sort=True)
    ):
        model, detail = _fit_local_subtype(
            group, candidates, str(side), str(archetype), seed + index * 101, encoder_kind
        )
        catalog.append(detail)
        if model is None:
            continue
        models.append(model)
        train_pos = (train["side_name"].astype(str).eq(model.side) & train["archetype_policy_key"].astype(str).eq(model.archetype)).to_numpy()
        score_pos = (score["side_name"].astype(str).eq(model.side) & score["archetype_policy_key"].astype(str).eq(model.archetype)).to_numpy()
        for frame, pos in ((train, train_pos), (score, score_pos)):
            if not pos.any():
                continue
            x, _, _ = _subtype_matrix(frame.loc[pos], model.features, model.medians, model.scales)
            z = _subtype_latent(model, x)
            posterior = model.gmm.predict_proba(z).astype(np.float32)
            distance = -model.gmm.score_samples(z).astype(np.float32)
            prefix = f"local_subtype__{model.side}__{model.archetype}"
            for component in range(model.selected_components):
                name = f"{prefix}__posterior_{component}"
                if name not in frame:
                    frame[name] = np.float32(0.0)
                frame.loc[pos, name] = posterior[:, component]
            entropy = -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1) / np.log(model.selected_components)
            for name, values in ((f"{prefix}__entropy", entropy), (f"{prefix}__distance", distance)):
                if name not in frame:
                    frame[name] = np.float32(0.0)
                frame.loc[pos, name] = values.astype(np.float32)
    return train, score, catalog, models


def _apply_local_subtypes(
    frame: pd.DataFrame,
    models: list[LocalSubtypeModel],
) -> pd.DataFrame:
    """Assign already-frozen local subtype encoders to decision-time rows."""
    frame = frame.copy()
    research_prior = np.zeros(len(frame), dtype=np.float32)
    research_quality = np.zeros(len(frame), dtype=np.float32)
    for model in models:
        pos = (
            frame["side_name"].astype(str).eq(model.side)
            & frame["archetype_policy_key"].astype(str).eq(model.archetype)
        ).to_numpy()
        if not pos.any():
            continue
        x, _, _ = _subtype_matrix(frame.loc[pos], model.features, model.medians, model.scales)
        z = _subtype_latent(model, x)
        posterior = model.gmm.predict_proba(z).astype(np.float32)
        distance = -model.gmm.score_samples(z).astype(np.float32)
        prefix = f"local_subtype__{model.side}__{model.archetype}"
        for component in range(model.selected_components):
            name = f"{prefix}__posterior_{component}"
            if name not in frame:
                frame[name] = np.float32(0.0)
            frame.loc[pos, name] = posterior[:, component]
        entropy = -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1) / np.log(model.selected_components)
        for name, values in ((f"{prefix}__entropy", entropy), (f"{prefix}__distance", distance)):
            if name not in frame:
                frame[name] = np.float32(0.0)
            frame.loc[pos, name] = values.astype(np.float32)
        # This is an empirical-Bayes context feature, not an OOS outcome:
        # the state means were fit entirely before the scored period. High
        # posterior concentration and adequate research support are required
        # before the policy may use it.
        prior = posterior @ np.asarray(model.train_cluster_ev, dtype=np.float32)
        confidence = np.sqrt(np.max(posterior, axis=1)) * min(1.0, model.fit_rows / 12_000.0)
        research_prior[pos] = prior.astype(np.float32)
        research_quality[pos] = confidence.astype(np.float32)
    frame["local_subtype_research_residual_prior"] = research_prior
    frame["local_subtype_research_prior_quality"] = research_quality
    return frame


def _fit_frozen_research_subtypes(
    research: pd.DataFrame,
    seed: int,
    encoder_kind: str,
    allowed_archetypes: set[str] | None = None,
    objective_kind: str = "ev",
) -> tuple[list[dict[str, Any]], list[LocalSubtypeModel]]:
    """Fit one 2025/early-2026 state contract, frozen before evaluation."""
    candidates = _subtype_state_candidates(research)
    catalog: list[dict[str, Any]] = []
    models: list[LocalSubtypeModel] = []
    for index, ((side, archetype), group) in enumerate(
        research.groupby(["side_name", "archetype_policy_key"], observed=True, sort=True)
    ):
        if allowed_archetypes is not None and str(archetype) not in allowed_archetypes:
            catalog.append({
                "side": str(side), "archetype": str(archetype),
                "status": "excluded_not_in_target_ablation", "rows": int(len(group)),
                "fit_scope": "research_frozen_pre_2026_04",
            })
            continue
        if "dirtyavoid" in str(archetype).lower():
            catalog.append({
                "side": str(side), "archetype": str(archetype),
                "status": "excluded_nontradable_family", "rows": int(len(group)),
                "fit_scope": "research_frozen_pre_2026_04",
            })
            continue
        model, detail = _fit_local_subtype(
            group, candidates, str(side), str(archetype), seed + index * 101,
            encoder_kind, tail_only=False, objective_kind=objective_kind,
        )
        detail["fit_scope"] = "research_frozen_pre_2026_04"
        catalog.append(detail)
        if model is not None:
            models.append(model)
    return catalog, models


def _matrix(
    frame: pd.DataFrame,
    features: list[str],
    categories: list[str],
    medians: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    if medians is None:
        medians = np.nanmedian(raw, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    missing = ~np.isfinite(raw)
    if missing.any():
        raw[missing] = np.take(medians, np.nonzero(missing)[1])
    if scales is None:
        q25, q75 = np.quantile(raw, (0.25, 0.75), axis=0)
        scales = (q75 - q25).astype(np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
    x = np.clip((raw - medians) / scales, -8.0, 8.0).astype(np.float32)
    # Missingness is decision-time information, especially for drift/OOD fields.
    x = np.column_stack([x, missing.astype(np.float32)])
    codes = frame["archetype_policy_key"].astype(str).to_numpy()
    one_hot = np.zeros((len(frame), len(categories)), dtype=np.float32)
    index = {name: pos for pos, name in enumerate(categories)}
    for row, value in enumerate(codes):
        if value in index:
            one_hot[row, index[value]] = 1.0
    return np.column_stack([x, one_hot]).astype(np.float32), medians, scales, raw


def _fit_side_model(
    train: pd.DataFrame,
    side: str,
    features: list[str],
    params: dict[str, Any],
    seed: int,
) -> SideModel | None:
    group = train.loc[train["side_name"].eq(side)].sort_values("__ts__", kind="stable").reset_index(drop=True)
    residual = _direct_causal_residual(group)
    eligible = _tail_fit_mask(group) & np.isfinite(residual)
    if int(eligible.sum()) < 1_500:
        return None
    fit = group.loc[eligible].reset_index(drop=True)
    residual = residual[eligible]
    categories = sorted(fit["archetype_policy_key"].astype(str).unique())
    x, medians, scales, raw = _matrix(fit, features, categories)
    clean = _num(fit, "clean_exec")
    bad = _num(fit, "full_path_bad_mae_1r")
    # Keep clean-hit and bad-path as auxiliary outcomes, but repeat the direct
    # EV residual in the shared loss.  sklearn's MLP has no per-output loss
    # weights; this makes fixed-activity EV the dominant learned objective
    # without discarding path diagnostics.
    target = np.column_stack([residual, residual, residual, clean, bad]).astype(np.float32)
    target_center = np.nanmean(target, axis=0).astype(np.float32)
    target_scale = np.nanstd(target, axis=0).astype(np.float32)
    target_scale[target_scale < 1e-4] = 1.0
    y = (target - target_center) / target_scale
    # sklearn's MLPRegressor has no stable per-row weight support across our
    # dependency versions.  Deterministic tail replication keeps the loss
    # focused on cutoff replacements without random resampling.
    weight = _rank_weights(fit)
    repeats = np.where(weight >= 3.0, 3, np.where(weight >= 1.5, 2, 1)).astype(np.int8)
    idx = np.repeat(np.arange(len(fit), dtype=np.int32), repeats)
    model = MLPRegressor(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation="relu",
        solver="adam",
        alpha=float(params["alpha"]),
        batch_size=int(params["batch_size"]),
        learning_rate_init=float(params["learning_rate_init"]),
        max_iter=int(params["max_iter"]),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=16,
        tol=float(params["tol"]),
        random_state=seed,
    )
    noise = float(params["noise_std"])
    rng = np.random.default_rng(seed + 17)
    x_fit = x[idx].copy()
    if noise:
        x_fit[:, : 2 * len(features)] += rng.normal(0.0, noise, size=(len(x_fit), 2 * len(features))).astype(np.float32)
    model.fit(x_fit, y[idx])
    ood = np.sqrt(np.mean(np.square(x[:, : len(features)]), axis=1))
    support = fit["archetype_policy_key"].astype(str).value_counts().to_dict()
    return SideModel(
        side=side, features=features, categories=categories, medians=medians,
        scales=scales, target_center=target_center, target_scale=target_scale,
        model=model, ood_q50=float(np.quantile(ood, 0.50)),
        ood_q95=float(np.quantile(ood, 0.95)),
        archetype_support={str(k): int(v) for k, v in support.items()},
        params=dict(params),
    )


def _predict_side_model(model: SideModel | None, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correction = np.zeros(len(frame), dtype=np.float32)
    quality = np.zeros(len(frame), dtype=np.float32)
    ood = np.full(len(frame), np.nan, dtype=np.float32)
    if model is None or not len(frame):
        return correction, quality, ood
    x, _, _, raw = _matrix(frame, model.features, model.categories, model.medians, model.scales)
    pred = model.model.predict(x).astype(np.float32)
    pred = pred * model.target_scale + model.target_center
    correction = np.mean(pred[:, :3], axis=1, dtype=np.float32)
    ood = np.sqrt(np.mean(np.square(x[:, : len(model.features)]), axis=1))
    denom = max(model.ood_q95 - model.ood_q50, 1e-4)
    ood_conf = np.clip((model.ood_q95 - ood) / denom, 0.0, 1.0)
    arches = frame["archetype_policy_key"].astype(str).to_numpy()
    local_support = np.array([model.archetype_support.get(key, 0) for key in arches], dtype=np.float32)
    support_conf = np.minimum(1.0, local_support / 1_800.0)
    quality = (ood_conf * support_conf).astype(np.float32)
    return correction, quality, ood


def _month_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")


def _top_mask(score: np.ndarray, budget: int) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    finite = np.isfinite(score)
    n = min(int(budget), int(finite.sum()))
    if n <= 0:
        return mask
    if n >= int(finite.sum()):
        mask[finite] = True
        return mask
    cutoff = np.partition(score[finite], int(finite.sum()) - n)[int(finite.sum()) - n]
    above = np.flatnonzero(finite & (score > cutoff))
    mask[above] = True
    remaining = n - len(above)
    if remaining:
        mask[np.flatnonzero(finite & (score == cutoff))[:remaining]] = True
    return mask


def _fixed_activity_mask(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    result = np.zeros(len(frame), dtype=bool)
    months = _month_ids(frame)
    parent = _num(frame, "policy_parent_rank", 0.0)
    for month in sorted(months.unique()):
        pos = np.flatnonzero(months.eq(month).to_numpy())
        budget = int(np.sum(parent[pos] >= 0.90))
        result[pos] = _top_mask(score[pos], budget)
    return result


def _metric_rows(frame: pd.DataFrame, selected: np.ndarray, arm: str) -> list[dict[str, Any]]:
    work = frame.loc[selected].copy()
    if work.empty:
        return []
    work["month"] = _month_ids(work).to_numpy()
    ts = pd.to_datetime(work["__ts__"], utc=True)
    work["week_start"] = (ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    ev = _num(work, "ev_after_1pct")
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("global", []), ("month", ["month"]), ("week", ["week_start"]),
        ("side", ["side_name"]), ("archetype", ["archetype_policy_key"]),
        ("month_side_archetype", ["month", "side_name", "archetype_policy_key"]),
    ]
    for scope, keys in group_specs:
        iterator: Iterable[tuple[Any, pd.DataFrame]]
        if keys:
            iterator = work.groupby(keys, observed=True, sort=True)
        else:
            iterator = [((), work)]
        for group_key, group in iterator:
            value = _num(group, "ev_after_1pct")
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = {"arm": arm, "scope": scope, "selected_rows": int(len(group)),
                   "mean_ev_after_1pct": float(np.mean(value)),
                   "sum_ev_after_1pct": float(np.sum(value)),
                   "positive_ev_rate": float(np.mean(value > 0.0)),
                   "clean_exec_rate": float(np.mean(_num(group, "clean_exec") > 0.5)),
                   "bad_mae_rate": float(np.mean(_num(group, "full_path_bad_mae_1r") > 0.5)),
                   "timeout_rate": float(np.mean(_num(group, "timeout") > 0.5))}
            row.update(dict(zip(keys, group_key)))
            rows.append(row)
    return rows


def _stability(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    work = frame.loc[selected].copy()
    if work.empty:
        return {"mean_ev": -np.inf, "worst_week": -np.inf, "worst_month": -np.inf}
    work["month"] = _month_ids(work).to_numpy()
    ts = pd.to_datetime(work["__ts__"], utc=True)
    work["week"] = (ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    weekly = work.groupby("week", observed=True)["ev_after_1pct"].mean()
    monthly = work.groupby("month", observed=True)["ev_after_1pct"].mean()
    return {"mean_ev": float(work["ev_after_1pct"].mean()), "worst_week": float(weekly.min()), "worst_month": float(monthly.min())}


def _local_gate(history: pd.DataFrame) -> dict[tuple[str, str], bool]:
    gates: dict[tuple[str, str], bool] = {}
    if history.empty:
        return gates
    for key, group in history.groupby(["side_name", "archetype_policy_key"], observed=True):
        if len(group) < 120:
            gates[(str(key[0]), str(key[1]))] = False
            continue
        parent = _num(group, "parent_expected_ev") + 1e-7 * _num(group, "policy_parent_rank")
        base = _fixed_activity_mask(group, parent)
        best_gain, best_swaps = -np.inf, 0
        for alpha in (0.05, 0.10, 0.25, 0.50, 1.0):
            candidate = parent + alpha * _num(group, "mlp_correction") * _num(group, "mlp_quality")
            alt = _fixed_activity_mask(group, candidate)
            swaps = int(np.sum(base ^ alt))
            gain = (
                float(np.mean(_num(group.loc[alt], "ev_after_1pct")) - np.mean(_num(group.loc[base], "ev_after_1pct")))
                if base.any() and alt.any() else -np.inf
            )
            if gain > best_gain:
                best_gain, best_swaps = gain, swaps
        gates[(str(key[0]), str(key[1]))] = bool(best_swaps >= 8 and best_gain > 0.0)
    return gates


def _policy_params(history: pd.DataFrame) -> tuple[float, dict[tuple[str, str], bool], dict[str, float]]:
    gates = _local_gate(history)
    if history.empty:
        return 0.0, gates, {"objective": 0.0, "baseline_mean_ev": np.nan}
    parent_score = _num(history, "parent_expected_ev") + 1e-7 * _num(history, "policy_parent_rank")
    baseline = _fixed_activity_mask(history, parent_score)
    base_stats = _stability(history, baseline)
    chosen, best = 0.0, {"objective": 0.0, **base_stats}
    gate_arr = np.array([
        gates.get((str(s), str(a)), False)
        for s, a in zip(history["side_name"], history["archetype_policy_key"])
    ], dtype=np.float32)
    for alpha in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0):
        score = parent_score + alpha * _num(history, "mlp_correction") * _num(history, "mlp_quality") * gate_arr
        selected = _fixed_activity_mask(history, score)
        stats = _stability(history, selected)
        gain = stats["mean_ev"] - base_stats["mean_ev"]
        # A higher mean EV may pay a small stability cost, but not a hidden
        # one: any worst-week/month decline must be at most one fifth of gain.
        stable = (
            gain > 0.0
            and base_stats["worst_week"] - stats["worst_week"] <= gain / 5.0 + 1e-9
            and base_stats["worst_month"] - stats["worst_month"] <= gain / 5.0 + 1e-9
        )
        objective = gain + 0.20 * (stats["worst_week"] - base_stats["worst_week"]) + 0.20 * (stats["worst_month"] - base_stats["worst_month"])
        if stable and objective > best["objective"]:
            chosen, best = float(alpha), {"objective": float(objective), **stats}
    best["baseline_mean_ev"] = base_stats["mean_ev"]
    return chosen, gates, best


def _local_alphas(history: pd.DataFrame, gates: dict[tuple[str, str], bool]) -> dict[tuple[str, str], float]:
    """Select conservative side/archetype blend weights from prior OOS only."""
    result: dict[tuple[str, str], float] = {}
    for key, group in history.groupby(["side_name", "archetype_policy_key"], observed=True):
        side_key = (str(key[0]), str(key[1]))
        if not gates.get(side_key, False) or len(group) < 120:
            result[side_key] = 0.0
            continue
        parent = _num(group, "parent_expected_ev") + 1e-7 * _num(group, "policy_parent_rank")
        baseline = _fixed_activity_mask(group, parent)
        base = _stability(group, baseline)
        best_alpha, best_objective = 0.0, 0.0
        for alpha in (0.05, 0.10, 0.25, 0.50, 0.75, 1.0):
            score = parent + alpha * _num(group, "mlp_correction") * _num(group, "mlp_quality")
            selected = _fixed_activity_mask(group, score)
            stats = _stability(group, selected)
            gain = stats["mean_ev"] - base["mean_ev"]
            stable = (
                gain > 0.0
                and base["worst_week"] - stats["worst_week"] <= gain / 5.0 + 1e-9
                and base["worst_month"] - stats["worst_month"] <= gain / 5.0 + 1e-9
            )
            objective = gain + 0.20 * (stats["worst_week"] - base["worst_week"]) + 0.20 * (stats["worst_month"] - base["worst_month"])
            if stable and objective > best_objective:
                best_alpha, best_objective = alpha, objective
        result[side_key] = float(best_alpha)
    return result


def _hpo_params(train: pd.DataFrame, side_features: dict[str, list[str]], seed: int, trials: int) -> tuple[dict[str, Any], pd.DataFrame]:
    default = {"hidden_layer_sizes": (48, 24, 12), "alpha": 0.45, "noise_std": 0.07,
               "learning_rate_init": 0.00035, "batch_size": 1024, "max_iter": 180, "tol": 0.0002}
    if trials <= 0:
        return default, pd.DataFrame()
    cutoff = train["__ts__"].quantile(0.78)
    fit = train.loc[train["__ts__"] < cutoff]
    valid = train.loc[train["__ts__"] >= cutoff]
    if len(fit) < 5_000 or len(valid) < 1_500:
        return default, pd.DataFrame()
    def objective(trial: optuna.Trial) -> float:
        params = dict(default)
        params.update({
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(32, 16), (48, 24, 12), (64, 32, 16)]),
            "alpha": trial.suggest_float("alpha", 0.20, 1.60, log=True),
            "noise_std": trial.suggest_float("noise_std", 0.035, 0.13),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.00015, 0.00075, log=True),
        })
        models = {side: _fit_side_model(fit, side, features, params, seed + trial.number * 31 + i)
                  for i, (side, features) in enumerate(side_features.items())}
        scored = valid.copy()
        scored["parent_expected_ev"] = 0.0
        scored["mlp_correction"] = 0.0
        scored["mlp_quality"] = 0.0
        for side, model in models.items():
            pos = scored["side_name"].eq(side).to_numpy()
            mapping = fit_hierarchical_ev_calibrator(fit.loc[fit["side_name"].eq(side)], _num(fit.loc[fit["side_name"].eq(side)], "policy_parent_rank"), _num(fit.loc[fit["side_name"].eq(side)], "ev_after_1pct"), min_local_rows=500, shrink_rows=1500.0, local_weight_cap=0.65)
            scored.loc[pos, "parent_expected_ev"] = predict_hierarchical_ev(mapping, scored.loc[pos], _num(scored.loc[pos], "policy_parent_rank"))
            corr, quality, _ = _predict_side_model(model, scored.loc[pos])
            scored.loc[pos, "mlp_correction"] = corr
            scored.loc[pos, "mlp_quality"] = quality
        base = _fixed_activity_mask(scored, _num(scored, "parent_expected_ev"))
        b = _stability(scored, base)
        best_value = -np.inf
        for alpha in (0.05, 0.10, 0.25, 0.50, 1.0):
            alt = _fixed_activity_mask(scored, _num(scored, "parent_expected_ev") + alpha * _num(scored, "mlp_correction") * _num(scored, "mlp_quality"))
            a = _stability(scored, alt)
            gain = a["mean_ev"] - b["mean_ev"]
            penalty = max(0.0, b["worst_week"] - a["worst_week"] - max(gain, 0.0) / 5.0)
            best_value = max(best_value, float(gain - 2.0 * penalty))
        return best_value
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    rows = [{**trial.params, "value": trial.value, "state": str(trial.state)} for trial in study.trials]
    winner = dict(default)
    # A non-positive internal fixed-activity result is an explicit no-HPO
    # outcome.  Keep the conservative default instead of selecting the least
    # harmful noise configuration.
    if study.best_value > 0.0:
        winner.update(study.best_params)
    winner["hidden_layer_sizes"] = tuple(winner["hidden_layer_sizes"])
    return winner, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-oos", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260713/ev_mapped_side_base_residual_expert_fullcurrent_top30_replay_20260714/oos_predictions.parquet"))
    parser.add_argument("--canonical-manifest", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260713/ev_mapped_side_base_residual_expert_fullcurrent_top30_replay_20260714/manifest.json"))
    parser.add_argument("--sparse-mlp-oos", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260713/canonical_meta_postprocessor_20260714/mlp_hier_ev_hpo20_expanding_sparse_v3_retry1/oos_predictions.parquet"))
    parser.add_argument("--scored-ledger", type=Path, default=Path("data_perp/reports/s59_h5_fullthroughjul10_base_configfull_freshmda_fixedparams_wf30_20260713/meta_handoff_top30_allsafe_aegmmfull_fullcoverage_20260714/s52_trailing_regime_scored_ledger.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260714/full_contract_pooled_mlp_fixed_activity_ablation"))
    parser.add_argument("--hpo-trials", type=int, default=12)
    parser.add_argument(
        "--mlp-params-json", type=Path, default=None,
        help="Reuse an earlier chronological MLP HPO winner; do not retune model shape.",
    )
    parser.add_argument(
        "--subtype-objective", choices=("ev", "path_precision"), default="ev",
        help="Train-only objective for state feature/K selection; transform inputs remain pre-entry only.",
    )
    parser.add_argument(
        "--sparse-parent", action="store_true",
        help="Use the existing sparse local MLP EV-ranked output as the immutable parent score.",
    )
    parser.add_argument(
        "--subtype-target-archetypes", default="",
        help="Optional comma-separated archetypes to receive subtype context in a narrow ablation.",
    )
    parser.add_argument(
        "--local-subtypes", action="store_true",
        help="Fit frozen data-driven side x archetype encoder/GMM subtype posteriors in every fold.",
    )
    parser.add_argument(
        "--subtype-encoder", choices=("pca", "ae"), default="pca",
        help="Latent encoder before local GMM. AE is a regularized denoising autoencoder.",
    )
    parser.add_argument(
        "--subtype-research-context", type=Path, default=None,
        help="Historical side/archetype research cache used to fit one frozen subtype state contract.",
    )
    parser.add_argument(
        "--subtype-current-state-context", type=Path, default=None,
        help="Current OOS state cache with the identical state__ feature contract.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_rows(args)
    start, end = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")
    frame = frame.loc[frame["__ts__"].between(start, end, inclusive="left")].reset_index(drop=True)
    frame["existing_sparse_parent_score"] = _num(frame, "expected_ev_rank_score", np.nan)
    if args.dry_run:
        print(json.dumps({
            "current_rows": int(len(frame)),
            "local_subtypes": bool(args.local_subtypes),
            "encoder": args.subtype_encoder,
            "research_context": str(args.subtype_research_context) if args.subtype_research_context else None,
            "current_state_context": str(args.subtype_current_state_context) if args.subtype_current_state_context else None,
        }, indent=2))
        return
    frozen_subtype_catalog: list[dict[str, Any]] = []
    frozen_subtype_models: list[LocalSubtypeModel] = []
    allowed_archetypes: set[str] | None = None
    if args.subtype_research_context is not None:
        if args.subtype_current_state_context is None:
            raise ValueError("--subtype-current-state-context is required with --subtype-research-context")
        frame = _merge_state_context(frame, args.subtype_current_state_context)
        research = _read_parquet(args.subtype_research_context)
        research["__ts__"] = pd.to_datetime(research["__ts__"], utc=True, errors="coerce")
        research = research.loc[research["__ts__"].notna()].copy()
        allowed_archetypes = {x.strip() for x in args.subtype_target_archetypes.split(",") if x.strip()} or None
        frozen_subtype_catalog, frozen_subtype_models = _fit_frozen_research_subtypes(
            research, args.seed, args.subtype_encoder, allowed_archetypes, args.subtype_objective
        )
        frame = _apply_local_subtypes(frame, frozen_subtype_models)
        source_audit["frozen_subtype_research_rows"] = int(len(research))
        source_audit["frozen_subtype_models"] = int(len(frozen_subtype_models))
    source_audit["evaluation_source_start"] = str(frame["__ts__"].min())
    source_audit["evaluation_source_end"] = str(frame["__ts__"].max())
    source_audit["monthly_rows"] = _month_ids(frame).value_counts().sort_index().to_dict()
    available_contract = source_audit["canonical_feature_contract"]
    contract_features = list(dict.fromkeys(sum((list(v) for v in available_contract.values()), [])))
    observable_extra = [
        c for c in frame.columns
        if not c.startswith("state__")
        and _feature_ok(c)
        and (any(token in c.lower() for token in RELIABILITY_TOKENS) or c in JULY_DRIFT_FIELDS)
    ]
    candidates = list(dict.fromkeys([
        *contract_features, *PROTECTED_ANCHORS, *JULY_DRIFT_FIELDS, *observable_extra,
    ]))
    candidates = [c for c in candidates if c in frame and _feature_ok(c)]
    source_audit["candidate_count"] = len(candidates)
    source_audit["required_drift_candidates"] = [c for c in JULY_DRIFT_FIELDS if c in candidates]
    (args.output_dir / "source_audit.json").write_text(json.dumps(source_audit, indent=2, default=str) + "\n")
    april = frame.loc[_month_ids(frame).eq("2026-04")].copy()
    selection: dict[str, list[str]] = {}
    feature_reports: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    for offset, side in enumerate(("long", "short")):
        selection[side], report, coverage[side] = _select_side_features(april, candidates, side, args.seed + offset * 100)
        feature_reports.append(report)
    pd.concat(feature_reports, ignore_index=True).to_csv(args.output_dir / "feature_selection_april.csv", index=False)
    (args.output_dir / "joint_coverage_april.json").write_text(json.dumps(coverage, indent=2, default=str) + "\n")
    if args.mlp_params_json is not None:
        params = json.loads(args.mlp_params_json.read_text())
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        hpo = pd.DataFrame()
    else:
        params, hpo = _hpo_params(april, selection, args.seed, args.hpo_trials)
    hpo.to_csv(args.output_dir / "mlp_hpo_april.csv", index=False)
    (args.output_dir / "best_mlp_params.json").write_text(json.dumps(params, indent=2) + "\n")
    print(f"Full-contract inputs={len(candidates)} side-selected={{long:{len(selection['long'])}, short:{len(selection['short'])}}} params={params}", flush=True)
    scored_folds: list[pd.DataFrame] = []
    fold_manifest: list[dict[str, Any]] = []
    months = ("2026-05", "2026-06", "2026-07")
    for fold_no, month in enumerate(months):
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        train = frame.loc[frame["__ts__"] < month_start].copy()
        valid = frame.loc[frame["__ts__"].between(month_start, month_end, inclusive="left")].copy()
        subtype_catalog: list[dict[str, Any]] = []
        subtype_models: list[LocalSubtypeModel] = []
        if frozen_subtype_models:
            subtype_catalog = frozen_subtype_catalog
            subtype_models = frozen_subtype_models
        elif args.local_subtypes:
            train, valid, subtype_catalog, subtype_models = _append_local_subtypes(
                train, valid, args.seed + fold_no * 10_000, args.subtype_encoder
            )
        fold_candidates = list(dict.fromkeys([
            *candidates,
            *[column for column in train.columns if column.startswith("local_subtype__")],
        ]))
        current_features: dict[str, list[str]] = {}
        fold_reports: list[pd.DataFrame] = []
        for offset, side in enumerate(("long", "short")):
            current_features[side], report, fold_cov = _select_side_features(train, fold_candidates, side, args.seed + fold_no * 1_000 + offset)
            report["fold_month"] = month
            fold_reports.append(report)
            (args.output_dir / f"joint_coverage_{month}_{side}.json").write_text(json.dumps(fold_cov, indent=2, default=str) + "\n")
        pd.concat(fold_reports, ignore_index=True).to_csv(args.output_dir / f"feature_selection_{month}.csv", index=False)
        valid["parent_expected_ev"] = np.nan
        valid["mlp_correction"] = 0.0
        valid["mlp_quality"] = 0.0
        valid["mlp_ood_distance"] = np.nan
        for offset, side in enumerate(("long", "short")):
            train_side = train.loc[train["side_name"].eq(side)]
            valid_pos = valid["side_name"].eq(side).to_numpy()
            ev_map = fit_hierarchical_ev_calibrator(
                train_side, _num(train_side, "policy_parent_rank"), _num(train_side, "ev_after_1pct"),
                shrink_rows=1_500.0, min_local_rows=500, local_weight_cap=0.65,
                tail_weight_top10=5.0, tail_weight_top20=2.5,
            )
            valid.loc[valid_pos, "parent_expected_ev"] = predict_hierarchical_ev(ev_map, valid.loc[valid_pos], _num(valid.loc[valid_pos], "policy_parent_rank"))
            model = _fit_side_model(train, side, current_features[side], params, args.seed + fold_no * 10 + offset)
            corr, quality, ood = _predict_side_model(model, valid.loc[valid_pos])
            valid.loc[valid_pos, "mlp_correction"] = corr
            valid.loc[valid_pos, "mlp_quality"] = quality
            valid.loc[valid_pos, "mlp_ood_distance"] = ood
        if args.sparse_parent:
            sparse_parent = _num(valid, "existing_sparse_parent_score", np.nan)
            if not np.isfinite(sparse_parent).all():
                raise ValueError("sparse-parent mode requires finite expected_ev_rank_score rows")
            valid["parent_expected_ev"] = sparse_parent
        prior = pd.concat(scored_folds, ignore_index=True) if scored_folds else pd.DataFrame(columns=valid.columns)
        alpha, gates, policy_info = _policy_params(prior)
        state_history = prior.copy()
        if not state_history.empty:
            state_history["mlp_correction"] = _num(state_history, "local_subtype_research_residual_prior")
            state_history["mlp_quality"] = _num(state_history, "local_subtype_research_prior_quality")
        state_alpha, state_gates, state_policy_info = _policy_params(state_history)
        state_local_alphas = _local_alphas(state_history, state_gates)
        gate_array = np.array([gates.get((str(s), str(a)), False) for s, a in zip(valid["side_name"], valid["archetype_policy_key"])], dtype=np.float32)
        state_gate_array = np.array([state_gates.get((str(s), str(a)), False) for s, a in zip(valid["side_name"], valid["archetype_policy_key"])], dtype=np.float32)
        state_alpha_array = np.array([state_local_alphas.get((str(s), str(a)), 0.0) for s, a in zip(valid["side_name"], valid["archetype_policy_key"])], dtype=np.float32)
        valid["causal_alpha"] = np.float32(alpha)
        valid["local_positive_swap_gate"] = gate_array
        valid["state_prior_causal_alpha"] = state_alpha_array
        valid["state_prior_positive_swap_gate"] = state_gate_array
        valid["score_parent_ev"] = _num(valid, "parent_expected_ev") + 1e-7 * _num(valid, "policy_parent_rank")
        valid["score_pooled_raw"] = _num(valid, "parent_expected_ev") + _num(valid, "mlp_correction") * _num(valid, "mlp_quality")
        valid["score_pooled_shrunk"] = _num(valid, "score_parent_ev") + alpha * _num(valid, "mlp_correction") * _num(valid, "mlp_quality") * gate_array
        valid["score_state_prior_shrunk"] = _num(valid, "score_parent_ev") + state_alpha_array * _num(valid, "local_subtype_research_residual_prior") * _num(valid, "local_subtype_research_prior_quality") * state_gate_array
        valid["score_existing_sparse"] = _num(valid, "expected_ev_rank_score", np.nan)
        valid["selected_parent_v9"] = _fixed_activity_mask(valid, _num(valid, "score_parent_ev"))
        valid["selected_pooled_raw"] = _fixed_activity_mask(valid, _num(valid, "score_pooled_raw"))
        valid["selected_pooled_shrunk"] = _fixed_activity_mask(valid, _num(valid, "score_pooled_shrunk"))
        valid["selected_state_prior_shrunk"] = _fixed_activity_mask(valid, _num(valid, "score_state_prior_shrunk"))
        valid["selected_existing_sparse"] = _fixed_activity_mask(valid, _num(valid, "score_existing_sparse"))
        scored_folds.append(valid)
        fold_manifest.append({"month": month, "train_end_exclusive": str(month_start), "oos_start": str(month_start), "oos_end_exclusive": str(month_end), "train_rows": int(len(train)), "oos_rows": int(len(valid)), "features_by_side": current_features, "local_subtype_catalog": subtype_catalog, "local_subtype_model_count": len(subtype_models), "policy_alpha_from_prior_oos": alpha, "active_local_gates": [f"{s}||{a}" for (s, a), enabled in gates.items() if enabled], "policy_tuning": policy_info, "state_prior_alpha_from_prior_oos": state_alpha, "state_prior_local_alphas": {f"{s}||{a}": value for (s, a), value in state_local_alphas.items() if value > 0.0}, "active_state_prior_gates": [f"{s}||{a}" for (s, a), enabled in state_gates.items() if enabled], "state_prior_policy_tuning": state_policy_info})
        print(f"fold={month} train={len(train):,} oos={len(valid):,} alpha={alpha:.2f} gates={int(gate_array.sum())}/{len(valid)} state_alpha={state_alpha:.2f} state_gates={int(state_gate_array.sum())}/{len(valid)}", flush=True)
    scored = pd.concat(scored_folds, ignore_index=True)
    arms = {
        "parent_v9_tail": "selected_parent_v9",
        "existing_sparse_local_mlp": "selected_existing_sparse",
        "full_contract_pooled_raw": "selected_pooled_raw",
        "full_contract_pooled_causal_shrunk": "selected_pooled_shrunk",
        "frozen_subtype_prior_causal_shrunk": "selected_state_prior_shrunk",
    }
    rows: list[dict[str, Any]] = []
    for arm, column in arms.items():
        rows.extend(_metric_rows(scored, scored[column].to_numpy(dtype=bool), arm))
    metrics = pd.DataFrame(rows)
    global_rows = metrics.loc[metrics["scope"].eq("global")].copy()
    baseline = global_rows.loc[global_rows["arm"].eq("parent_v9_tail"), ["mean_ev_after_1pct", "sum_ev_after_1pct", "clean_exec_rate", "bad_mae_rate", "timeout_rate"]].iloc[0]
    for col in baseline.index:
        metrics[f"delta_vs_parent_{col}"] = metrics[col] - float(baseline[col])
    metrics.to_csv(args.output_dir / "metrics_by_scope.csv", index=False)
    scored.to_parquet(args.output_dir / "oos_predictions.parquet", index=False, compression="zstd")
    joblib.dump({"folds": fold_manifest, "params": params, "non_promoting": True}, args.output_dir / "ablation_models_metadata.joblib")
    manifest = {
        "schema": "v9_full_contract_pooled_mlp_fixed_activity_ablation_v1",
        "purpose": "non-promoting fixed-activity V9-tail replacement test",
        "evaluation": "May-July 2026 only; full current cache begins April 2026",
        "candidate_contract": "canonical side contract plus inference-available AE/GMM, leaf, support, OOD, uncertainty and drift fields",
        "direct_target": "realized EV - train-only side x archetype expected EV(parent rank)",
        "auxiliary_targets": ["clean_exec", "full_path_bad_mae_1r"],
        "architecture": "one conservative MLP per side with archetype one-hot inputs; optional local subtype posteriors are frozen train-only encoder/GMM state context; local corrections are only admitted through causal OOF positive rank-swap gates",
        "selection_region": "policy_parent_rank 0.80-0.995 with extra deterministic training weight around 0.90",
        "activity_contract": "each month retains exactly the V9 parent tail count (policy_parent_rank >= 0.90)",
        "policy_contract": "alpha is tuned on prior OOS folds only; zero is eligible and is the default when no positive stable gain exists",
        "feature_selection_contract": "side-specific causal residual screening, 90% joint coverage after 30-day warm-up, automatic stopping with ceiling 72",
        "outcome_columns_excluded": list(OUTCOME_TOKENS),
        "folds": fold_manifest,
        "source_audit": source_audit,
        "params": params,
        "local_subtypes_enabled": bool(args.local_subtypes),
        "local_subtype_encoder": args.subtype_encoder if args.local_subtypes else None,
        "frozen_subtype_research_context": str(args.subtype_research_context) if args.subtype_research_context else None,
        "sparse_parent": bool(args.sparse_parent),
        "subtype_target_archetypes": sorted(allowed_archetypes) if args.subtype_research_context and allowed_archetypes else None,
        "subtype_objective": args.subtype_objective,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(global_rows.to_string(index=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    main()
