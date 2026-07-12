"""Leakage-safe residual archetypes for the alternative meta model.

Realized outcomes are used only to discover train-side failure signatures.
Stable semantic probabilities are then predicted from pre-entry features and
are the only residual-archetype values exposed to OOS rows or inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

try:  # pragma: no cover - LightGBM is optional in unit-test environments.
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

from .features_gmm_ae import fit_ae_gmm_state, transform_ae_gmm_features

SEMANTIC_ARCHETYPES: tuple[str, ...] = (
    "base_clean_high_confidence",
    "base_dirty_high_confidence",
    "base_slow_timeout_positive",
    "base_bad_mae_false_positive",
    "base_missed_clean_opportunity",
    "base_high_variance_uncertain",
    "base_low_edge_noise",
)

RESIDUAL_FEATURE_PREFIX = "meta_resid_arch_"
RESIDUAL_AE_PREFIX = "meta_resid_ae_"

OUTCOME_COLUMNS = {
    "__first_touch_target_soft__",
    "__first_touch_policy_soft__",
    "target_soft",
    "__target_soft__",
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "clean_exec",
    "dirty_positive",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "ret_net",
    "u_policy_net",
}

REFERENCE_DERIVED_COLUMNS = {
    "reference_rank_pct",
    "reference_rank_band",
    "hit_surprise",
    "negative_hit_surprise",
    "positive_hit_surprise",
    "negative_tail_label",
    "positive_tail_label",
    "ev_surprise",
    "market_signed_surprise",
    "market_adjusted_hit_surprise",
    "local_daily_signed_surprise",
    "local_prior_daily_signed_surprise",
    "large_negative_surprise_label",
    "large_positive_surprise_label",
    "negative_autocorr_label",
    "positive_autocorr_label",
    "reference_ev_equivalent_threshold",
    "reference_ev_equivalent_selected",
    "assessment_hr_8d_expected",
    "assessment_hr_8d_surprise",
}

STATELESS_AE_GMM_TOKENS = (
    "dae_b16_",
    "gmm_prob_",
    "gmm_cluster_posterior_",
    "gmm_entropy",
    "gmm_posterior_max",
    "gmm_posterior_margin",
    "gmm_mahal_",
    "gmm_dist_center_",
    "min_mahalanobis",
    "expected_mahalanobis",
    "reconstruction_error",
)


@dataclass(frozen=True)
class ResidualArchetypeConfig:
    score_col: str = "score_regime_calibrated"
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    hit_col: str = "clean_exec"
    ev_col: str = "ev_after_1pct"
    min_side_rows: int = 2_500
    min_local_rows: int = 1_200
    min_cluster_rows: int = 100
    cluster_candidates: tuple[int, ...] = (3, 4, 5)
    max_cluster_fit_rows: int = 60_000
    max_recognizer_fit_rows: int = 120_000
    max_recognizer_features: int = 96
    mutual_info_rows: int = 18_000
    use_residual_ae_gmm: bool = False
    ae_gmm_max_rows: int = 5_000
    ae_gmm_max_iter: int = 80
    ae_gmm_cluster_candidates: tuple[int, ...] = (4, 6, 8, 10, 12)
    fit_local_models: bool = True
    rank_scope: str = "global"
    label_mode: str = "gmm"
    allow_side_fallback: bool = False
    top10_quantile: float = 0.90
    top20_quantile: float = 0.80
    surprise_tail_quantile: float = 0.80
    ev_equivalent_min_rows: int = 40
    feature_screen_mode: str = "binned_mi_lgbm"
    feature_screen_bins: int = 8
    feature_screen_lgbm_rounds: int = 60
    semantic_min_temporal_segments: int = 2
    semantic_min_segment_rows: int = 20
    random_state: int = 20260711


@dataclass
class _LocalModel:
    key: str
    feature_columns: list[str]
    medians: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray
    recognizer: Any
    class_semantics: dict[int, str]
    cluster_priors: dict[int, dict[str, float]]
    support_rows: int
    catalog: list[dict[str, Any]] = field(default_factory=list)
    feature_relevance: list[dict[str, Any]] = field(default_factory=list)
    ae_gmm_state: dict[str, Any] = field(default_factory=dict)
    ae_gmm_input_features: list[str] = field(default_factory=list)
    ae_gmm_output_features: list[str] = field(default_factory=list)


@dataclass
class _NativeMulticlassRecognizer:
    booster: Any
    classes_: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.booster.predict(x), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, len(self.classes_))
        return raw


def residual_feature_names(*, include_ae_gmm: bool = False) -> list[str]:
    names = [f"{RESIDUAL_FEATURE_PREFIX}prob__{name}" for name in SEMANTIC_ARCHETYPES]
    names += [
        f"{RESIDUAL_FEATURE_PREFIX}expected_hit_surprise",
        f"{RESIDUAL_FEATURE_PREFIX}expected_ev",
        f"{RESIDUAL_FEATURE_PREFIX}expected_bad_mae",
        f"{RESIDUAL_FEATURE_PREFIX}expected_timeout",
        f"{RESIDUAL_FEATURE_PREFIX}expected_dirty_positive",
        f"{RESIDUAL_FEATURE_PREFIX}entropy",
        f"{RESIDUAL_FEATURE_PREFIX}confidence",
        f"{RESIDUAL_FEATURE_PREFIX}support_log1p",
        f"{RESIDUAL_FEATURE_PREFIX}local_model",
    ]
    if include_ae_gmm:
        names += residual_ae_gmm_feature_names()
    return names


def residual_ae_gmm_feature_names() -> list[str]:
    from .features_gmm_ae import ae_gmm_feature_columns

    return [
        name
        for name in ae_gmm_feature_columns(RESIDUAL_AE_PREFIX)
        if any(token in name for token in STATELESS_AE_GMM_TOKENS)
        and not any(token in name for token in ("delta", "accel", "speed"))
    ]


def _num(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _archetype(frame: pd.DataFrame, preferred: str) -> pd.Series:
    for name in (
        preferred,
        "__archetype_policy_key__",
        "archetype_label_family",
        "__archetype_label_family__",
        "policy_archetype",
        "local_side_archetype",
    ):
        if name in frame.columns:
            values = frame[name].astype(str).replace({"nan": "", "None": ""})
            if values.str.len().gt(0).any():
                return values.where(values.str.len().gt(0), "missing")
    return pd.Series("missing", index=frame.index, dtype="object")


def add_reference_surprise_targets(
    frame: pd.DataFrame,
    config: ResidualArchetypeConfig,
    *,
    rank_group_col: str = "oos_fold",
    ev_equivalent_thresholds: Mapping[tuple[str, str], float] | None = None,
    global_top10_ev: float | None = None,
    score_reference_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Add offline residual descriptors from the frozen reference prediction."""

    out = frame.copy(deep=False)
    score_col = next(
        (
            name
            for name in (
                config.score_col,
                "score_regime_calibrated",
                "score",
                "score_meta_base_soft_label",
                "hit_probability",
            )
            if name in out.columns
            and pd.to_numeric(out[name], errors="coerce").notna().any()
        ),
        None,
    )
    if score_col is None:
        raise ValueError(
            "No usable frozen score column is available for residual targets"
        )
    score = _num(out, score_col).clip(0.0, 1.0)
    rank_scope = str(config.rank_scope).strip().lower()
    if rank_scope in {"global", "train_global", "history_global"}:
        if score_reference_values is None:
            rank = score.rank(method="average", pct=True)
            score_reference = np.sort(
                score.dropna().to_numpy(dtype=np.float32, copy=True)
            )
        else:
            score_reference = np.sort(
                np.asarray(score_reference_values, dtype=np.float32)
            )
            score_values = score.to_numpy(dtype=np.float32)
            rank_values = np.searchsorted(
                score_reference, score_values, side="right"
            ) / max(len(score_reference), 1)
            rank = pd.Series(rank_values, index=out.index, dtype=np.float32)
    elif rank_scope == "timestamp_global":
        timestamp = pd.to_datetime(out.get("__ts__"), utc=True, errors="coerce")
        groupers = [timestamp]
    elif rank_scope == "timestamp_side":
        timestamp = pd.to_datetime(out.get("__ts__"), utc=True, errors="coerce")
        groupers = [timestamp]
        if config.side_col in out.columns:
            groupers.append(out[config.side_col].astype(str).str.lower())
    elif rank_group_col in out.columns:
        groupers: list[pd.Series] = [out[rank_group_col].astype(str)]
    else:
        ts = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
        groupers = [ts.dt.to_period("M").astype(str)]
    if rank_scope not in {"global", "train_global", "history_global"}:
        if (
            rank_scope not in {"timestamp_global", "timestamp_side"}
            and config.side_col in out.columns
        ):
            groupers.append(out[config.side_col].astype(str).str.lower())
        group_key = pd.concat(groupers, axis=1).astype(str).agg("|".join, axis=1)
        rank = score.groupby(group_key, sort=False).rank(method="average", pct=True)
    hit = _num(out, config.hit_col, 0.0).fillna(0.0).clip(0.0, 1.0)
    ev = _num(out, config.ev_col, 0.0).fillna(0.0)
    surprise = (hit - score).astype(np.float32)
    out["reference_rank_pct"] = rank.fillna(0.0).astype(np.float32)
    out["reference_rank_band"] = pd.cut(
        out["reference_rank_pct"],
        bins=[-np.inf, 0.70, 0.80, 0.90, np.inf],
        labels=["below_top30", "top20_30", "top10_20", "top10"],
        include_lowest=True,
    ).astype(str)
    out["hit_surprise"] = surprise
    out["negative_hit_surprise"] = (-surprise).clip(lower=0.0).astype(np.float32)
    out["positive_hit_surprise"] = surprise.clip(lower=0.0).astype(np.float32)
    out["ev_surprise"] = ev.astype(np.float32)
    out["_reference_score"] = score.astype(np.float32)
    out["_reference_score_column"] = str(score_col)
    thresholds, target_ev = _add_ev_equivalent_archetype_thresholds(
        out,
        config,
        thresholds=ev_equivalent_thresholds,
        target_ev=global_top10_ev,
    )
    out.attrs["ev_equivalent_thresholds"] = thresholds
    out.attrs["global_top10_ev"] = target_ev
    top20 = out["reference_rank_pct"].ge(float(config.top20_quantile))
    top10 = out["reference_ev_equivalent_selected"].fillna(0).astype(bool) & top20
    band_10_20 = top20 & ~top10
    out["negative_tail_label"] = (top10 & ((hit < 0.5) | (ev <= 0.0))).astype(np.int8)
    out["positive_tail_label"] = (band_10_20 & (hit >= 0.5) & (ev > 0.0)).astype(
        np.int8
    )
    _add_local_surprise_event_targets(out, config)
    return out


def _add_local_surprise_event_targets(
    frame: pd.DataFrame,
    config: ResidualArchetypeConfig,
) -> None:
    """Add train-only local residual events with same-timestamp market drift removed."""

    timestamp = pd.to_datetime(frame.get("__ts__"), utc=True, errors="coerce")
    side = (
        frame.get(config.side_col, pd.Series("missing", index=frame.index))
        .astype(str)
        .str.lower()
    )
    arch = _archetype(frame, config.archetype_col).astype(str)
    surprise = _num(frame, "hit_surprise", 0.0).fillna(0.0)
    top20 = _num(frame, "reference_rank_pct", 0.0).ge(float(config.top20_quantile))
    market = surprise.where(top20).groupby(timestamp, sort=False).transform("mean")
    market = market.fillna(surprise.groupby(timestamp, sort=False).transform("mean"))
    adjusted = surprise - market
    frame["market_signed_surprise"] = market.astype(np.float32)
    frame["market_adjusted_hit_surprise"] = adjusted.astype(np.float32)

    day = timestamp.dt.floor("D")
    local_key = pd.DataFrame(
        {"side": side, "arch": arch, "day": day}, index=frame.index
    )
    daily = adjusted.groupby([side, arch, day], sort=False).transform("mean")
    daily_cells = pd.DataFrame(
        {"side": side, "arch": arch, "day": day, "daily": daily}, index=frame.index
    ).drop_duplicates(["side", "arch", "day"])
    daily_cells = daily_cells.sort_values(["side", "arch", "day"], kind="stable")
    daily_cells["prior"] = daily_cells.groupby(["side", "arch"], sort=False)[
        "daily"
    ].shift(1)
    prior_map = daily_cells.set_index(["side", "arch", "day"])["prior"]
    row_index = pd.MultiIndex.from_frame(local_key[["side", "arch", "day"]])
    prior = prior_map.reindex(row_index).to_numpy(dtype=np.float32)
    frame["local_daily_signed_surprise"] = daily.astype(np.float32)
    frame["local_prior_daily_signed_surprise"] = prior

    group_key = pd.MultiIndex.from_arrays([side.to_numpy(), arch.to_numpy()])
    tail_q = float(np.clip(config.surprise_tail_quantile, 0.50, 0.99))
    adjusted_series = pd.Series(adjusted.to_numpy(dtype=np.float32), index=frame.index)
    negative_cut = adjusted_series.groupby(group_key, sort=False).transform(
        lambda values: values.quantile(1.0 - tail_q)
    )
    positive_cut = adjusted_series.groupby(group_key, sort=False).transform(
        lambda values: values.quantile(tail_q)
    )
    top10 = _num(frame, "reference_ev_equivalent_selected", 0.0).ge(0.5) & top20
    top10_20 = top20 & ~top10
    large_negative = top10 & adjusted_series.le(negative_cut)
    large_positive = top10_20 & adjusted_series.ge(positive_cut)
    frame["large_negative_surprise_label"] = large_negative.astype(np.int8)
    frame["large_positive_surprise_label"] = large_positive.astype(np.int8)
    frame["negative_autocorr_label"] = (
        large_negative & pd.Series(prior, index=frame.index).lt(0.0) & daily.lt(0.0)
    ).astype(np.int8)
    frame["positive_autocorr_label"] = (
        large_positive & pd.Series(prior, index=frame.index).gt(0.0) & daily.gt(0.0)
    ).astype(np.int8)


def _add_ev_equivalent_archetype_thresholds(
    frame: pd.DataFrame,
    config: ResidualArchetypeConfig,
    *,
    thresholds: Mapping[tuple[str, str], float] | None = None,
    target_ev: float | None = None,
) -> tuple[dict[tuple[str, str], float], float]:
    """Find train-only local score thresholds matching global top-10 EV quality."""

    score = _num(frame, "_reference_score")
    ev = _num(frame, config.ev_col, 0.0).fillna(0.0)
    rank = _num(frame, "reference_rank_pct", 0.0)
    global_top10 = rank.ge(float(config.top10_quantile))
    if target_ev is None or not np.isfinite(target_ev):
        target_ev = (
            float(ev.loc[global_top10].mean())
            if global_top10.any()
            else float(ev.mean())
        )
    side = (
        frame.get(config.side_col, pd.Series("missing", index=frame.index))
        .astype(str)
        .str.lower()
    )
    arch = _archetype(frame, config.archetype_col).astype(str)
    fitted_thresholds: dict[tuple[str, str], float] = {
        (str(key[0]), str(key[1])): float(value)
        for key, value in (thresholds or {}).items()
        if np.isfinite(value)
    }
    if thresholds is None:
        for key, idx in (
            pd.DataFrame({"side": side, "arch": arch}, index=frame.index)
            .groupby(["side", "arch"], sort=False, dropna=False)
            .groups.items()
        ):
            part = pd.DataFrame({"score": score.loc[idx], "ev": ev.loc[idx]}).dropna()
            if len(part) < int(config.ev_equivalent_min_rows):
                continue
            part = part.sort_values("score", ascending=False, kind="stable")
            count = np.arange(1, len(part) + 1, dtype=np.int32)
            cumulative_ev = part["ev"].to_numpy(dtype=np.float64).cumsum() / count
            eligible = np.flatnonzero(count >= int(config.ev_equivalent_min_rows))
            if eligible.size == 0:
                continue
            distance = np.abs(cumulative_ev[eligible] - float(target_ev))
            chosen = int(eligible[int(np.nanargmin(distance))])
            fitted_thresholds[(str(key[0]), str(key[1]))] = float(
                part["score"].iloc[chosen]
            )
    threshold = np.asarray(
        [fitted_thresholds.get((str(s), str(a)), np.nan) for s, a in zip(side, arch)],
        dtype=np.float32,
    )
    frame["reference_ev_equivalent_threshold"] = threshold
    frame["reference_ev_equivalent_selected"] = (
        np.isfinite(threshold) & (score.to_numpy(dtype=np.float32) >= threshold)
    ).astype(np.int8)
    return fitted_thresholds, float(target_ev)


def inference_feature_columns(
    frame: pd.DataFrame, candidates: Iterable[str]
) -> list[str]:
    excluded = (
        OUTCOME_COLUMNS
        | REFERENCE_DERIVED_COLUMNS
        | {
            "__ts__",
            "__symbol__",
            "month",
            "oos_fold",
            "calendar_month",
            "valid_start",
            "valid_end",
        }
    )
    out: list[str] = []
    for name in candidates:
        col = str(name)
        if col in excluded or col not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or frame[col].dtype == bool:
            out.append(col)
    return list(dict.fromkeys(out))


def _time_spread_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    thirds = ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))
    per = max(1, cap // 3)
    parts = [
        np.linspace(lo, hi - 1, min(per, hi - lo), dtype=np.int64)
        for lo, hi in thirds
        if hi > lo
    ]
    idx = np.unique(np.concatenate(parts))
    return idx[:cap].astype(np.int64, copy=False)


def _descriptor_matrix(
    frame: pd.DataFrame, config: ResidualArchetypeConfig
) -> pd.DataFrame:
    prepared = (
        frame
        if "hit_surprise" in frame.columns
        else add_reference_surprise_targets(frame, config)
    )
    return pd.DataFrame(
        {
            "rank_pct": _num(prepared, "reference_rank_pct", 0.0).fillna(0.0),
            "signed_surprise": _num(prepared, "hit_surprise", 0.0).fillna(0.0),
            "negative_surprise": _num(prepared, "negative_hit_surprise", 0.0).fillna(
                0.0
            ),
            "positive_surprise": _num(prepared, "positive_hit_surprise", 0.0).fillna(
                0.0
            ),
            "ev": _num(prepared, config.ev_col, 0.0).fillna(0.0).clip(-0.10, 0.10),
            "clean": _num(prepared, config.hit_col, 0.0).fillna(0.0).clip(0.0, 1.0),
            "dirty": _num(prepared, "dirty_positive", 0.0).fillna(0.0).clip(0.0, 1.0),
            "bad_mae": _num(prepared, "full_path_bad_mae_1r", 0.0)
            .fillna(0.0)
            .clip(0.0, 1.0),
            "timeout": _num(prepared, "timeout", 0.0).fillna(0.0).clip(0.0, 1.0),
            "negative_tail": _num(prepared, "negative_tail_label", 0.0).fillna(0.0),
            "positive_tail": _num(prepared, "positive_tail_label", 0.0).fillna(0.0),
            "market_adjusted_surprise": _num(
                prepared, "market_adjusted_hit_surprise", 0.0
            ).fillna(0.0),
            "large_negative_surprise": _num(
                prepared, "large_negative_surprise_label", 0.0
            ).fillna(0.0),
            "large_positive_surprise": _num(
                prepared, "large_positive_surprise_label", 0.0
            ).fillna(0.0),
            "negative_autocorr": _num(prepared, "negative_autocorr_label", 0.0).fillna(
                0.0
            ),
            "positive_autocorr": _num(prepared, "positive_autocorr_label", 0.0).fillna(
                0.0
            ),
        },
        index=frame.index,
    ).astype(np.float32)


def _cluster_semantic(metrics: Mapping[str, float]) -> str:
    if (
        float(metrics.get("positive_tail_rate", 0.0)) >= 0.20
        and float(metrics.get("mean_ev", 0.0)) > 0.0
    ):
        return "base_missed_clean_opportunity"
    if float(metrics.get("negative_tail_rate", 0.0)) >= 0.35:
        if float(metrics.get("bad_mae_rate", 0.0)) >= 0.55:
            return "base_bad_mae_false_positive"
        if float(metrics.get("timeout_rate", 0.0)) >= 0.12:
            return "base_slow_timeout_positive"
        if float(metrics.get("dirty_rate", 0.0)) >= 0.45:
            return "base_dirty_high_confidence"
    if (
        float(metrics.get("clean_rate", 0.0)) >= 0.65
        and float(metrics.get("mean_hit_surprise", 0.0)) >= 0.0
    ):
        return "base_clean_high_confidence"
    if float(metrics.get("surprise_std", 0.0)) >= 0.40:
        return "base_high_variance_uncertain"
    return "base_low_edge_noise"


def _cluster_catalog(
    desc: pd.DataFrame, labels: np.ndarray, key: str
) -> tuple[list[dict[str, Any]], dict[int, str], dict[int, dict[str, float]]]:
    catalog: list[dict[str, Any]] = []
    semantics: dict[int, str] = {}
    priors: dict[int, dict[str, float]] = {}
    for cluster in sorted(np.unique(labels).tolist()):
        mask = np.asarray(labels) == int(cluster)
        part = desc.loc[mask]
        metrics = {
            "rows": int(len(part)),
            "mean_hit_surprise": float(part["signed_surprise"].mean()),
            "surprise_std": float(part["signed_surprise"].std(ddof=0)),
            "mean_ev": float(part["ev"].mean()),
            "clean_rate": float(part["clean"].mean()),
            "dirty_rate": float(part["dirty"].mean()),
            "bad_mae_rate": float(part["bad_mae"].mean()),
            "timeout_rate": float(part["timeout"].mean()),
            "negative_tail_rate": float(part["negative_tail"].mean()),
            "positive_tail_rate": float(part["positive_tail"].mean()),
            "negative_autocorr_rate": float(part["negative_autocorr"].mean()),
            "positive_autocorr_rate": float(part["positive_autocorr"].mean()),
        }
        semantic = _cluster_semantic(metrics)
        semantics[int(cluster)] = semantic
        priors[int(cluster)] = {
            "hit_surprise": metrics["mean_hit_surprise"],
            "ev": metrics["mean_ev"],
            "bad_mae": metrics["bad_mae_rate"],
            "timeout": metrics["timeout_rate"],
            "dirty": metrics["dirty_rate"],
        }
        catalog.append(
            {"model_key": key, "cluster": int(cluster), "semantic": semantic, **metrics}
        )
    return catalog, semantics, priors


def _economic_semantic_labels(
    work: pd.DataFrame,
    desc: pd.DataFrame,
    config: ResidualArchetypeConfig,
) -> tuple[np.ndarray, dict[int, str], dict[str, Any]]:
    """Create stable path/economic state labels on train outcomes only."""
    rank = desc["rank_pct"].to_numpy(dtype=np.float32)
    surprise = desc["signed_surprise"].to_numpy(dtype=np.float32)
    ev = desc["ev"].to_numpy(dtype=np.float32)
    clean = desc["clean"].to_numpy(dtype=np.float32) >= 0.5
    dirty = desc["dirty"].to_numpy(dtype=np.float32) >= 0.5
    bad_mae = desc["bad_mae"].to_numpy(dtype=np.float32) >= 0.5
    timeout = desc["timeout"].to_numpy(dtype=np.float32) >= 0.5
    negative_autocorr = desc["negative_autocorr"].to_numpy(dtype=np.float32) >= 0.5
    positive_autocorr = desc["positive_autocorr"].to_numpy(dtype=np.float32) >= 0.5
    top10 = rank >= 0.90
    top10_20 = (rank >= 0.80) & ~top10
    finite_surprise = surprise[np.isfinite(surprise)]
    finite_ev = np.abs(ev[np.isfinite(ev)])
    surprise_tail = (
        float(np.nanquantile(np.abs(finite_surprise), 0.75))
        if finite_surprise.size
        else 0.25
    )
    ev_tail = float(np.nanquantile(finite_ev, 0.80)) if finite_ev.size else 0.01
    semantic = np.full(len(desc), "base_low_edge_noise", dtype=object)
    semantic[top10_20 & clean & (ev > 0.0) & (surprise > 0.0)] = (
        "base_missed_clean_opportunity"
    )
    semantic[positive_autocorr & clean & (ev > 0.0)] = "base_missed_clean_opportunity"
    semantic[top10 & timeout] = "base_slow_timeout_positive"
    semantic[top10 & bad_mae & ((surprise < 0.0) | (ev <= 0.0))] = (
        "base_bad_mae_false_positive"
    )
    semantic[top10 & dirty & (ev > 0.0) & ~timeout] = "base_dirty_high_confidence"
    semantic[negative_autocorr & bad_mae] = "base_bad_mae_false_positive"
    semantic[negative_autocorr & timeout] = "base_slow_timeout_positive"
    semantic[negative_autocorr & dirty & ~bad_mae & ~timeout] = (
        "base_dirty_high_confidence"
    )
    semantic[top10 & clean & (ev > 0.0) & ~dirty & ~bad_mae & ~timeout] = (
        "base_clean_high_confidence"
    )
    unresolved = semantic == "base_low_edge_noise"
    semantic[
        unresolved
        & top10
        & ((np.abs(surprise) >= surprise_tail) | (np.abs(ev) >= ev_tail))
    ] = "base_high_variance_uncertain"

    timestamp = pd.to_datetime(work.get("__ts__"), utc=True, errors="coerce")
    order = np.argsort(timestamp.astype("int64", copy=False).to_numpy(), kind="stable")
    segment = np.zeros(len(work), dtype=np.int8)
    for segment_id, positions in enumerate(np.array_split(order, 3)):
        segment[positions] = np.int8(segment_id)
    stability: dict[str, Any] = {}
    fallback = "base_low_edge_noise"
    for name in SEMANTIC_ARCHETYPES:
        mask = semantic == name
        counts = [int(np.sum(mask & (segment == idx))) for idx in range(3)]
        stable_segments = int(
            np.sum(np.asarray(counts) >= int(config.semantic_min_segment_rows))
        )
        stable = bool(
            name == fallback
            or (
                int(mask.sum()) >= int(config.min_cluster_rows)
                and stable_segments >= int(config.semantic_min_temporal_segments)
            )
        )
        stability[name] = {
            "rows": int(mask.sum()),
            "segment_rows": counts,
            "stable_segments": stable_segments,
            "retained": stable,
        }
        if not stable:
            semantic[mask] = fallback
    labels = np.asarray(
        [SEMANTIC_ARCHETYPES.index(str(name)) for name in semantic], dtype=np.int32
    )
    semantics = {
        int(index): str(name)
        for index, name in enumerate(SEMANTIC_ARCHETYPES)
        if int(np.sum(labels == index)) > 0
    }
    return labels, semantics, stability


def _semantic_catalog(
    desc: pd.DataFrame,
    labels: np.ndarray,
    semantics: Mapping[int, str],
    key: str,
    stability: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    catalog: list[dict[str, Any]] = []
    priors: dict[int, dict[str, float]] = {}
    for label, semantic in semantics.items():
        part = desc.loc[np.asarray(labels) == int(label)]
        metrics = {
            "rows": int(len(part)),
            "mean_hit_surprise": float(part["signed_surprise"].mean()),
            "surprise_std": float(part["signed_surprise"].std(ddof=0)),
            "mean_ev": float(part["ev"].mean()),
            "clean_rate": float(part["clean"].mean()),
            "dirty_rate": float(part["dirty"].mean()),
            "bad_mae_rate": float(part["bad_mae"].mean()),
            "timeout_rate": float(part["timeout"].mean()),
            "negative_tail_rate": float(part["negative_tail"].mean()),
            "positive_tail_rate": float(part["positive_tail"].mean()),
            "negative_autocorr_rate": float(part["negative_autocorr"].mean()),
            "positive_autocorr_rate": float(part["positive_autocorr"].mean()),
        }
        priors[int(label)] = {
            "hit_surprise": metrics["mean_hit_surprise"],
            "ev": metrics["mean_ev"],
            "bad_mae": metrics["bad_mae_rate"],
            "timeout": metrics["timeout_rate"],
            "dirty": metrics["dirty_rate"],
        }
        temporal = dict(stability.get(str(semantic), {}))
        catalog.append(
            {
                "model_key": key,
                "cluster": int(label),
                "semantic": str(semantic),
                "label_mode": "economic_semantic",
                **metrics,
                "temporal_segment_rows": temporal.get("segment_rows", []),
                "stable_temporal_segments": temporal.get("stable_segments", 0),
            }
        )
    return catalog, priors


def _choose_gmm(
    desc: pd.DataFrame, config: ResidualArchetypeConfig, seed: int
) -> tuple[GaussianMixture | None, np.ndarray, RobustScaler | None]:
    if len(desc) < max(config.min_cluster_rows * 3, 300):
        return None, np.zeros(len(desc), dtype=np.int32), None
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    z = scaler.fit_transform(desc).astype(np.float32)
    fit_idx = _time_spread_indices(len(z), config.max_cluster_fit_rows)
    z_fit = z[fit_idx]
    best: tuple[float, GaussianMixture] | None = None
    for k in config.cluster_candidates:
        if len(z_fit) < int(k) * config.min_cluster_rows:
            continue
        try:
            model = GaussianMixture(
                n_components=int(k),
                covariance_type="diag",
                reg_covar=1e-3,
                max_iter=160,
                n_init=2,
                random_state=int(seed + k),
            ).fit(z_fit)
        except ValueError:
            continue
        labels_fit = model.predict(z_fit)
        occupancy = np.bincount(labels_fit, minlength=int(k)) / max(len(labels_fit), 1)
        if float(occupancy.min()) < 0.015 or float(occupancy.max()) > 0.80:
            continue
        score = float(model.bic(z_fit)) + 1_000.0 * float(
            np.maximum(0.03 - occupancy, 0.0).sum()
        )
        if best is None or score < best[0]:
            best = (score, model)
    if best is None:
        return None, np.zeros(len(desc), dtype=np.int32), scaler
    return best[1], best[1].predict(z).astype(np.int32), scaler


def _prepare_numeric_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    medians: np.ndarray | None = None,
    clip_low: np.ndarray | None = None,
    clip_high: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = (
        frame.reindex(columns=list(columns))
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, copy=False)
    )
    x[~np.isfinite(x)] = np.nan
    if medians is None:
        medians = np.nanmedian(x, axis=0).astype(np.float32)
        medians = np.nan_to_num(medians, nan=0.0)
    missing = ~np.isfinite(x)
    if missing.any():
        x = x.copy()
        x[missing] = np.take(medians, np.nonzero(missing)[1])
    if clip_low is None:
        clip_low = np.nanpercentile(x, 0.5, axis=0).astype(np.float32)
    if clip_high is None:
        clip_high = np.nanpercentile(x, 99.5, axis=0).astype(np.float32)
    np.clip(x, clip_low, clip_high, out=x)
    return x, medians, clip_low, clip_high


def _quantile_codes(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    codes = np.full(len(values), -1, dtype=np.int16)
    if int(finite.sum()) < max(20, int(bins) * 3):
        return codes
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)[1:-1]
    edges = np.unique(np.nanquantile(values[finite], quantiles))
    if edges.size < 2:
        return codes
    codes[finite] = np.searchsorted(edges, values[finite], side="right").astype(
        np.int16
    )
    return codes


def _normalized_binned_mi(feature_codes: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int16)
    valid = (feature_codes >= 0) & np.isfinite(target)
    if int(valid.sum()) < 30 or np.unique(target[valid]).size < 2:
        return 0.0
    target_valid = target[valid]
    mi = float(mutual_info_score(target_valid, feature_codes[valid]))
    counts = np.bincount(target_valid - int(target_valid.min()))
    probs = counts[counts > 0].astype(np.float64)
    probs /= max(float(probs.sum()), 1.0)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(mi / max(entropy, 1e-8))


def _screen_targets(frame: pd.DataFrame, labels: np.ndarray) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {"state": np.asarray(labels, dtype=np.int16)}
    for name in (
        "large_negative_surprise_label",
        "large_positive_surprise_label",
        "negative_autocorr_label",
        "positive_autocorr_label",
    ):
        if name in frame.columns:
            values = _num(frame, name, 0.0).fillna(0.0).to_numpy(dtype=np.int16)
            if np.unique(values).size >= 2 and int(values.sum()) >= 8:
                targets[name] = values
    return targets


def _select_recognizer_features(
    frame: pd.DataFrame,
    labels: np.ndarray,
    candidates: Sequence[str],
    config: ResidualArchetypeConfig,
    seed: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    usable = inference_feature_columns(frame, candidates)
    idx = _time_spread_indices(len(frame), config.mutual_info_rows)
    sample = frame.iloc[idx]
    sample_labels = np.asarray(labels, dtype=np.int32)[idx]
    x, _, _, _ = _prepare_numeric_matrix(sample, usable)
    targets = _screen_targets(sample, sample_labels)
    target_arrays = {
        name: np.asarray(values, dtype=np.int16) for name, values in targets.items()
    }
    thirds = [
        part
        for part in np.array_split(np.arange(len(sample), dtype=np.int64), 3)
        if len(part) >= 30
    ]
    mi = np.zeros(len(usable), dtype=np.float32)
    mi_stability = np.zeros(len(usable), dtype=np.float32)
    for column_idx in range(len(usable)):
        codes = _quantile_codes(x[:, column_idx], int(config.feature_screen_bins))
        full_scores = [
            _normalized_binned_mi(codes, target) for target in target_arrays.values()
        ]
        mi[column_idx] = np.float32(max(full_scores, default=0.0))
        segment_scores: list[float] = []
        for positions in thirds:
            segment_scores.append(
                max(
                    (
                        _normalized_binned_mi(codes[positions], target[positions])
                        for target in target_arrays.values()
                    ),
                    default=0.0,
                )
            )
        mi_stability[column_idx] = np.float32(
            np.nanmedian(segment_scores) if segment_scores else 0.0
        )

    gain = np.zeros(len(usable), dtype=np.float32)
    if lgb is not None and str(config.feature_screen_mode).lower() == "binned_mi_lgbm":
        for target_name, target in target_arrays.items():
            if target_name == "state" or np.unique(target).size != 2:
                continue
            positives = int(target.sum())
            if positives < 8 or positives > len(target) - 8:
                continue
            dataset = lgb.Dataset(x, label=target, free_raw_data=True)
            booster = lgb.train(
                {
                    "objective": "binary",
                    "learning_rate": 0.05,
                    "num_leaves": 7,
                    "max_depth": 3,
                    "min_data_in_leaf": max(20, min(100, len(target) // 100)),
                    "feature_fraction": 0.75,
                    "lambda_l1": 0.10,
                    "lambda_l2": 4.0,
                    "seed": int(seed + len(target_name)),
                    "num_threads": 2,
                    "verbosity": -1,
                    "force_col_wise": True,
                },
                dataset,
                num_boost_round=int(config.feature_screen_lgbm_rounds),
            )
            local_gain = booster.feature_importance(importance_type="gain").astype(
                np.float32
            )
            local_gain /= max(float(local_gain.max()), 1e-8)
            gain = np.maximum(gain, local_gain)
    coverage = sample[usable].notna().mean(axis=0).to_numpy(dtype=np.float32)
    variance = np.nanstd(x, axis=0).astype(np.float32)
    score = (
        np.nan_to_num(mi, nan=0.0)
        + 0.50 * np.nan_to_num(mi_stability, nan=0.0)
        + 0.35 * np.nan_to_num(gain, nan=0.0)
        + 0.005 * coverage * np.log1p(np.maximum(variance, 0.0))
    )
    eligible = np.flatnonzero(
        (coverage >= 0.05) & np.isfinite(variance) & (variance > 1e-8)
    )
    keep = min(len(eligible), int(config.max_recognizer_features))
    order = eligible[np.argsort(score[eligible], kind="stable")[::-1][:keep]]
    relevance = [
        {
            "feature": str(usable[int(i)]),
            "score": float(score[int(i)]),
            "binned_mi": float(mi[int(i)]),
            "binned_mi_stability": float(mi_stability[int(i)]),
            "shallow_lgbm_gain": float(gain[int(i)]),
            "coverage": float(coverage[int(i)]),
        }
        for i in order
    ]
    return [usable[int(i)] for i in order], relevance


def _fit_local_model(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    config: ResidualArchetypeConfig,
    *,
    key: str,
    seed: int,
) -> _LocalModel | None:
    prepared = (
        frame
        if "hit_surprise" in frame.columns
        else add_reference_surprise_targets(frame, config)
    )
    population = prepared["reference_rank_pct"].ge(0.80).to_numpy(dtype=bool)
    if int(population.sum()) < max(100, config.min_cluster_rows * 3):
        return None
    work = prepared.loc[population].reset_index(drop=True)
    desc = _descriptor_matrix(work, config)
    label_mode = str(config.label_mode).strip().lower()
    if label_mode == "economic_semantic":
        labels, semantics, stability = _economic_semantic_labels(work, desc, config)
        if len(np.unique(labels)) < 2:
            return None
        catalog, priors = _semantic_catalog(desc, labels, semantics, key, stability)
    else:
        _gmm, labels, _descriptor_scaler = _choose_gmm(desc, config, seed)
        if _gmm is None or len(np.unique(labels)) < 2:
            return None
        catalog, semantics, priors = _cluster_catalog(desc, labels, key)
    features, feature_relevance = _select_recognizer_features(
        work, labels, candidates, config, seed
    )
    local_ae_state: dict[str, Any] = {}
    local_ae_inputs: list[str] = []
    local_ae_outputs: list[str] = []
    if config.use_residual_ae_gmm and len(features) >= 2:
        local_ae_inputs = features[: min(80, len(features))]
        timestamp = pd.to_datetime(work.get("__ts__"), utc=True, errors="coerce")
        timestamp_ns = timestamp.astype("int64", copy=False).to_numpy(dtype=np.int64)
        time_bucket = (
            timestamp_ns // np.int64(7 * 24 * 60 * 60 * 1_000_000_000)
        ).astype(np.float64)
        economic_targets = {
            "state": labels.astype(np.float32),
            "hit_surprise": desc["signed_surprise"].to_numpy(dtype=np.float32),
            "market_adjusted_hit_surprise": desc["market_adjusted_surprise"].to_numpy(
                dtype=np.float32
            ),
            "negative_autocorr": desc["negative_autocorr"].to_numpy(dtype=np.float32),
            "positive_autocorr": desc["positive_autocorr"].to_numpy(dtype=np.float32),
            "returns": desc["ev"].to_numpy(dtype=np.float32),
            "bad_mae": desc["bad_mae"].to_numpy(dtype=np.float32),
            "timeout": desc["timeout"].to_numpy(dtype=np.float32),
            "time_bucket": time_bucket,
        }
        local_ae_state = fit_ae_gmm_state(
            work.reindex(columns=local_ae_inputs),
            economic_targets=economic_targets,
            random_state=int(seed + 701),
            max_train_rows=int(config.ae_gmm_max_rows),
            gmm_max_train_rows=int(config.ae_gmm_max_rows),
            ae_max_iter=int(config.ae_gmm_max_iter),
            cluster_candidates=config.ae_gmm_cluster_candidates,
            reg_covar_candidates=(1e-4, 1e-3, 3e-3),
            smooth_lambda_candidates=(0.0,),
            path_aware_hpo=True,
            temporal_concentration_hpo=True,
        )
        transformed = transform_ae_gmm_features(
            work.reindex(columns=local_ae_inputs),
            local_ae_state,
            index=work.index,
            prefix=RESIDUAL_AE_PREFIX,
        )
        local_ae_outputs = [
            name
            for name in residual_ae_gmm_feature_names()
            if name in transformed.columns
        ]
        if local_ae_outputs:
            work = work.copy(deep=False)
            for name in local_ae_outputs:
                work[name] = (
                    pd.to_numeric(transformed[name], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float32)
                )
            features = list(dict.fromkeys(features + local_ae_outputs))
    if len(features) < 2 or lgb is None:
        return None
    x, medians, clip_low, clip_high = _prepare_numeric_matrix(work, features)
    fit_idx = _time_spread_indices(len(work), config.max_recognizer_fit_rows)
    classes = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int32)
    class_to_local = {int(value): idx for idx, value in enumerate(classes.tolist())}
    labels_local = np.asarray(
        [class_to_local[int(value)] for value in labels], dtype=np.int32
    )
    params = {
        "objective": "multiclass",
        "num_class": int(len(classes)),
        "learning_rate": 0.035,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "feature_fraction": 0.75,
        "lambda_l1": 0.05,
        "lambda_l2": 4.0,
        "seed": int(seed),
        "n_jobs": 2,
        "verbosity": -1,
        "force_col_wise": True,
    }
    counts = np.bincount(labels_local, minlength=len(classes)).astype(np.float32)
    class_weight = np.sqrt(float(len(work)) / np.maximum(counts, 1.0))
    sample_weight = class_weight[labels_local].astype(np.float32)
    timestamp = pd.to_datetime(work.get("__ts__"), utc=True, errors="coerce")
    day = timestamp.dt.floor("D")
    day_count = (
        day.groupby(day, sort=False).transform("size").to_numpy(dtype=np.float32)
    )
    sample_weight /= np.sqrt(np.maximum(day_count, 1.0))
    surprise_magnitude = np.abs(
        desc["market_adjusted_surprise"].to_numpy(dtype=np.float32)
    )
    sample_weight *= 1.0 + np.clip(surprise_magnitude, 0.0, 1.0)
    sample_weight *= 1.0 + 2.0 * desc["negative_autocorr"].to_numpy(dtype=np.float32)
    sample_weight *= 1.0 + 1.5 * desc["positive_autocorr"].to_numpy(dtype=np.float32)
    sample_weight /= max(float(np.mean(sample_weight)), 1e-6)
    sample_weight = np.clip(sample_weight, 0.10, 8.0)
    dataset = lgb.Dataset(
        x[fit_idx],
        label=labels_local[fit_idx],
        weight=sample_weight[fit_idx],
        free_raw_data=True,
    )
    booster = lgb.train(params, dataset, num_boost_round=120)
    recognizer = _NativeMulticlassRecognizer(booster=booster, classes_=classes)
    return _LocalModel(
        key=key,
        feature_columns=features,
        medians=medians,
        clip_low=clip_low,
        clip_high=clip_high,
        recognizer=recognizer,
        class_semantics=semantics,
        cluster_priors=priors,
        support_rows=int(len(work)),
        catalog=catalog,
        feature_relevance=feature_relevance,
        ae_gmm_state=local_ae_state,
        ae_gmm_input_features=local_ae_inputs,
        ae_gmm_output_features=local_ae_outputs,
    )


@dataclass
class ResidualArchetypeRecognizer:
    config: ResidualArchetypeConfig
    candidate_features: list[str]
    side_models: dict[str, _LocalModel] = field(default_factory=dict)
    local_models: dict[tuple[str, str], _LocalModel] = field(default_factory=dict)
    ae_gmm_state: dict[str, Any] = field(default_factory=dict)
    ae_gmm_input_features: list[str] = field(default_factory=list)
    catalog_: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_start_: str | None = None
    train_end_: str | None = None
    ev_equivalent_thresholds_: dict[tuple[str, str], float] = field(
        default_factory=dict
    )
    global_top10_ev_: float | None = None
    score_reference_values_: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )

    def _side_arch(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        side = (
            frame.get(self.config.side_col, pd.Series("missing", index=frame.index))
            .astype(str)
            .str.lower()
        )
        arch = _archetype(frame, self.config.archetype_col).astype(str)
        return side, arch

    def _append_ae_features(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        if not self.ae_gmm_state or not self.ae_gmm_input_features:
            return frame, []
        values = frame.reindex(columns=self.ae_gmm_input_features).apply(
            pd.to_numeric, errors="coerce"
        )
        transformed = transform_ae_gmm_features(
            values,
            self.ae_gmm_state,
            index=frame.index,
            prefix=RESIDUAL_AE_PREFIX,
        )
        wanted = [
            name
            for name in residual_ae_gmm_feature_names()
            if name in transformed.columns
        ]
        out = frame.copy(deep=False)
        for name in wanted:
            out[name] = (
                pd.to_numeric(transformed[name], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
            )
        return out, wanted

    def fit(self, train: pd.DataFrame) -> "ResidualArchetypeRecognizer":
        prepared = add_reference_surprise_targets(train, self.config)
        self.ev_equivalent_thresholds_ = dict(
            prepared.attrs.get("ev_equivalent_thresholds", {})
        )
        self.global_top10_ev_ = prepared.attrs.get("global_top10_ev")
        self.score_reference_values_ = np.sort(
            _num(prepared, "_reference_score")
            .dropna()
            .to_numpy(dtype=np.float32, copy=True)
        )
        ts = pd.to_datetime(prepared.get("__ts__"), utc=True, errors="coerce")
        self.train_start_ = str(ts.min())
        self.train_end_ = str(ts.max())
        candidates = inference_feature_columns(prepared, self.candidate_features)
        side, arch = self._side_arch(prepared)
        catalog: list[dict[str, Any]] = []
        self.side_models = {}
        self.local_models = {}
        if self.config.allow_side_fallback:
            for side_key, idx in (
                pd.Series(side, index=prepared.index)
                .groupby(side, sort=True)
                .groups.items()
            ):
                group = prepared.loc[idx]
                if len(group) < self.config.min_side_rows:
                    continue
                model = _fit_local_model(
                    group,
                    candidates,
                    self.config,
                    key=f"side::{side_key}",
                    seed=self.config.random_state + len(self.side_models) * 17,
                )
                if model is not None:
                    self.side_models[str(side_key)] = model
                    catalog.extend(model.catalog)
        if self.config.fit_local_models:
            keys = pd.DataFrame({"side": side, "arch": arch}, index=prepared.index)
            for (side_key, arch_key), idx in keys.groupby(
                ["side", "arch"], sort=True, dropna=False
            ).groups.items():
                group = prepared.loc[idx]
                if len(group) < self.config.min_local_rows:
                    continue
                model = _fit_local_model(
                    group,
                    candidates,
                    self.config,
                    key=f"local::{side_key}::{arch_key}",
                    seed=self.config.random_state + len(self.local_models) * 31 + 11,
                )
                if model is not None:
                    self.local_models[(str(side_key), str(arch_key))] = model
                    catalog.extend(model.catalog)
        self.catalog_ = pd.DataFrame(catalog)
        return self

    def _transform_model(
        self, frame: pd.DataFrame, model: _LocalModel, *, local: bool
    ) -> pd.DataFrame:
        ae_values: pd.DataFrame | None = None
        if model.ae_gmm_state and model.ae_gmm_input_features:
            ae_values = transform_ae_gmm_features(
                frame.reindex(columns=model.ae_gmm_input_features),
                model.ae_gmm_state,
                index=frame.index,
                prefix=RESIDUAL_AE_PREFIX,
            )
            frame = frame.copy(deep=False)
            for name in model.ae_gmm_output_features:
                frame[name] = (
                    pd.to_numeric(ae_values[name], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float32)
                )
        x, _, _, _ = _prepare_numeric_matrix(
            frame,
            model.feature_columns,
            medians=model.medians,
            clip_low=model.clip_low,
            clip_high=model.clip_high,
        )
        raw = model.recognizer.predict_proba(x)
        classes = [int(value) for value in model.recognizer.classes_]
        semantic_prob = np.zeros(
            (len(frame), len(SEMANTIC_ARCHETYPES)), dtype=np.float32
        )
        expected = {
            name: np.zeros(len(frame), dtype=np.float32)
            for name in ("hit_surprise", "ev", "bad_mae", "timeout", "dirty")
        }
        for col_idx, cluster in enumerate(classes):
            p = raw[:, col_idx].astype(np.float32)
            semantic = model.class_semantics.get(cluster, "base_low_edge_noise")
            semantic_prob[:, SEMANTIC_ARCHETYPES.index(semantic)] += p
            prior = model.cluster_priors.get(cluster, {})
            for name in expected:
                expected[name] += p * np.float32(prior.get(name, 0.0))
        rowsum = semantic_prob.sum(axis=1, keepdims=True)
        semantic_prob = np.divide(semantic_prob, np.where(rowsum <= 0.0, 1.0, rowsum))
        entropy = -np.sum(
            semantic_prob * np.log(np.maximum(semantic_prob, 1e-8)), axis=1
        ) / math.log(len(SEMANTIC_ARCHETYPES))
        out = pd.DataFrame(index=frame.index)
        for idx, semantic in enumerate(SEMANTIC_ARCHETYPES):
            out[f"{RESIDUAL_FEATURE_PREFIX}prob__{semantic}"] = semantic_prob[:, idx]
        for name, values in expected.items():
            suffix = (
                "expected_dirty_positive" if name == "dirty" else f"expected_{name}"
            )
            out[f"{RESIDUAL_FEATURE_PREFIX}{suffix}"] = values
        out[f"{RESIDUAL_FEATURE_PREFIX}entropy"] = entropy.astype(np.float32)
        out[f"{RESIDUAL_FEATURE_PREFIX}confidence"] = np.max(
            semantic_prob, axis=1
        ).astype(np.float32)
        out[f"{RESIDUAL_FEATURE_PREFIX}support_log1p"] = np.float32(
            np.log1p(model.support_rows)
        )
        out[f"{RESIDUAL_FEATURE_PREFIX}local_model"] = np.float32(1.0 if local else 0.0)
        if ae_values is not None:
            for name in model.ae_gmm_output_features:
                out[name] = (
                    pd.to_numeric(ae_values[name], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float32)
                )
        return out

    def transform_oos(self, oos: pd.DataFrame) -> pd.DataFrame:
        forbidden = sorted(
            (OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS).intersection(oos.columns)
        )
        if forbidden:
            raise ValueError(
                f"OOS residual-archetype transform received outcomes: {forbidden[:12]}"
            )
        safe = oos
        output = pd.DataFrame(
            0.0,
            index=oos.index,
            columns=residual_feature_names(
                include_ae_gmm=self.config.use_residual_ae_gmm
            ),
            dtype=np.float32,
        )
        fallback_prob = f"{RESIDUAL_FEATURE_PREFIX}prob__base_low_edge_noise"
        output[fallback_prob] = np.float32(1.0)
        output[f"{RESIDUAL_FEATURE_PREFIX}entropy"] = np.float32(1.0)
        output[f"{RESIDUAL_FEATURE_PREFIX}confidence"] = np.float32(0.0)
        side, arch = self._side_arch(safe)
        keys = pd.DataFrame({"side": side, "arch": arch}, index=safe.index)
        for (side_key, arch_key), idx in keys.groupby(
            ["side", "arch"], sort=False, dropna=False
        ).groups.items():
            model = self.local_models.get((str(side_key), str(arch_key)))
            local = model is not None
            if model is None and self.config.allow_side_fallback:
                model = self.side_models.get(str(side_key))
            if model is None:
                continue
            transformed = self._transform_model(safe.loc[idx], model, local=local)
            output.loc[idx, transformed.columns] = transformed.to_numpy(
                dtype=np.float32, copy=False
            )
        return output.astype(np.float32, copy=False)

    def prepare_evaluation_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Materialize outcome labels using thresholds frozen from train only."""

        return add_reference_surprise_targets(
            frame,
            self.config,
            ev_equivalent_thresholds=self.ev_equivalent_thresholds_,
            global_top10_ev=self.global_top10_ev_,
            score_reference_values=self.score_reference_values_,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "meta_residual_archetype_recognizer_v1",
            "train_start": self.train_start_,
            "train_end": self.train_end_,
            "candidate_feature_count": int(len(self.candidate_features)),
            "side_model_count": int(len(self.side_models)),
            "local_model_count": int(len(self.local_models)),
            "fit_local_models": bool(self.config.fit_local_models),
            "allow_side_fallback": bool(self.config.allow_side_fallback),
            "rank_scope": str(self.config.rank_scope),
            "label_mode": str(self.config.label_mode),
            "score_column": str(self.config.score_col),
            "top10_quantile": float(self.config.top10_quantile),
            "top20_quantile": float(self.config.top20_quantile),
            "global_top10_ev": self.global_top10_ev_,
            "score_reference_rows": int(len(self.score_reference_values_)),
            "ev_equivalent_thresholds": {
                f"{side}::{arch}": float(value)
                for (side, arch), value in self.ev_equivalent_thresholds_.items()
            },
            "feature_screen_mode": str(self.config.feature_screen_mode),
            "selected_features_by_model": {
                model.key: list(model.feature_columns)
                for model in list(self.side_models.values())
                + list(self.local_models.values())
            },
            "feature_relevance_by_model": {
                model.key: list(model.feature_relevance)
                for model in list(self.side_models.values())
                + list(self.local_models.values())
            },
            "semantic_archetypes": list(SEMANTIC_ARCHETYPES),
            "generated_features": residual_feature_names(
                include_ae_gmm=bool(self.config.use_residual_ae_gmm)
            ),
            "residual_ae_gmm_enabled": bool(
                any(
                    model.ae_gmm_state.get("enabled", False)
                    for model in self.local_models.values()
                )
            ),
            "residual_ae_gmm_scope": "side_x_archetype_local",
            "residual_ae_gmm_input_features_by_model": {
                model.key: list(model.ae_gmm_input_features)
                for model in self.local_models.values()
            },
            "leakage_contract": {
                "discovery": "realized outcomes and frozen current-meta OOS predictions on train rows only",
                "recognition": "pre-entry numeric features only",
                "oos_transform": "frozen recognizers and train-derived cluster priors; outcomes rejected",
                "raw_cluster_ids_exposed": False,
                "stable_outputs": "semantic probabilities and posterior-weighted train priors",
                "recent_performance_features": "excluded; 8-day hit-rate smoothing is assessment-only",
            },
        }


def strip_outcomes_for_oos(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[
            col
            for col in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if col in frame.columns
        ],
        errors="ignore",
    )
