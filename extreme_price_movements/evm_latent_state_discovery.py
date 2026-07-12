"""Latent-state discovery for the side x archetype EV calibrator.

This module finds observable states where a given ``side_name x policy_archetype``
stream is more or less favorable.  It deliberately targets the EV/regime
calibration layer, not GMM clustering: state definitions are built from
pre-entry meta/policy/context features, with AE/GMM columns excluded by default.

The core contract is leakage-safe for OOS assessment:

* state thresholds are fitted on train rows only,
* OOS rows receive frozen low/mid/high/missing assignments,
* outcomes are used only to score assigned states,
* metrics are emitted by top-k rank slices because those are what we trade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional acceleration.
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


DEFAULT_TOP_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("all", 1.00),
    ("top10", 0.10),
    ("top20", 0.20),
    ("top30", 0.30),
)

DEFAULT_EVM_FEATURE_HINTS: tuple[str, ...] = (
    "__derived_",
    "score_",
    "rank_",
    "base_",
    "meta_",
    "margin",
    "support_",
    "leaf_",
    "drift_",
    "source_drift_",
    "support_drift_",
    "leaf_drift_",
    "archetype_hit_surprise_",
    "hit_surprise_",
    "recent_",
    "market_",
    "mkt_",
    "xs_",
    "cs_",
    "corr_",
    "eig_",
    "state_spectral_eig_",
    "breadth",
    "dispersion",
    "amihud",
    "liquid",
    "spread",
    "funding",
    "oi_",
    "orderbook",
    "book_",
    "vwap",
    "range_",
    "loc_",
    "trend",
    "adx",
    "bollinger",
    "vol_",
    "atr_",
    "compression",
    "residual",
    "pressure",
    "shock",
    "entropy",
    "uncertainty",
    "effective_rank",
    "participation",
    "aegmm",
    "gmm",
    "mahalanobis",
    "reconstruction",
    "posterior",
    "cluster_speed",
    "cluster_acceleration",
)

DEFAULT_EVM_PRIORITY_FEATURE_HINTS: tuple[str, ...] = (
    "shock",
    "entropy",
    "market_",
    "mkt_",
    "breadth",
    "dispersion",
    "aegmm",
    "gmm",
    "mahalanobis",
    "reconstruction",
    "posterior",
)


def is_market_context_shock_entropy_feature(name: str) -> bool:
    """Return true when shock/entropy features are market/cross-asset context.

    This regime-learning path is meant to modulate archetype precision with
    market states, not reintroduce per-asset path descriptors.  Plain asset
    features such as ``vol_range_shock`` or ``direction_entropy_20`` are
    intentionally excluded here.  Residualized asset descriptors such as
    ``asset_minus_mkt_*shock*`` are also excluded: they are asset-relative, not
    market-wide.
    """

    lower = str(name).lower()
    has_shock_entropy = "shock" in lower or "entropy" in lower
    if not has_shock_entropy:
        return True
    residual_context = (
        "asset_minus_mkt",
        "mkt_resid",
        "peer_resid",
        "ts_resid",
        "symbol_minus_mkt",
    )
    if any(hint in lower for hint in residual_context):
        return False
    market_context = (
        "market",
        "mkt",
        "xs_",
        "xs_",
        "xs",
        "cs_",
        "cross_asset",
        "crossasset",
        "cross_section",
        "xasset",
        "state_spectral",
        "eig_",
        "basket",
        "factor",
        "breadth",
        "dispersion",
        "pct_assets",
        "liquidation",
        "bars_since_mkt",
    )
    return any(hint in lower for hint in market_context)


def evm_feature_priority_score(name: str) -> float:
    lower = str(name).lower()
    high_priority = (
        "shock",
        "entropy",
        "market_",
        "market_breadth",
        "market_dispersion",
        "mkt_",
        "pct_assets",
        "liquidation",
        "aegmm",
        "gmm",
        "mahalanobis",
        "reconstruction",
        "posterior",
    )
    medium_priority = (
        "breadth",
        "dispersion",
        "effective_rank",
        "participation",
    )
    if any(h in lower for h in high_priority):
        return 2.0
    if any(h in lower for h in medium_priority):
        return 1.0
    return 0.0


DEFAULT_TARGET_DERIVED_HINTS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "oracle",
    "realized",
    "ret_net",
    "net_return",
    "gross_return",
    "pnl",
    "profit",
    "loss",
    "first_touch",
    "full_path",
    "timeout",
    "stop",
    "bad_mae",
    "bad_MAE",
    "exec_margin",
    "clean_exec",
    "clean_positive",
    "dirty_positive",
    "adverse",
    "diagnostic_only",
    "outcome",
    "exit_",
    "position_",
)

DEFAULT_AEGMM_HINTS: tuple[str, ...] = (
    "aegmm",
    "gmm",
    "mahalanobis",
    "reconstruction",
    "posterior",
    "cluster_speed",
    "cluster_acceleration",
    "autoencoder",
    "dae_",
    "ae_",
    "AE_",
)

DEFAULT_OUTCOME_COL = "ev_after_1pct"
DEFAULT_SCORE_COL = "score_meta_base_soft_label"


@dataclass(frozen=True)
class EvmLatentStateConfig:
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    score_col: str = DEFAULT_SCORE_COL
    outcome_col: str = DEFAULT_OUTCOME_COL
    month_col: str = "month"
    week_col: str = "week_start"
    clean_col: str = "clean_exec"
    dirty_col: str = "dirty_positive"
    bad_mae_col: str = "full_path_bad_mae_1r"
    timeout_col: str = "timeout"
    stop_col: str = "stop_or_adverse"
    top_fractions: tuple[tuple[str, float], ...] = DEFAULT_TOP_FRACTIONS
    quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    min_group_rows: int = 160
    min_state_rows: int = 30
    min_feature_coverage: float = 0.35
    min_unique_values: int = 8
    max_features_per_group: int = 24
    top_features_for_pairs: int = 6
    min_oos_objective_delta: float = 0.0002
    max_pair_states_per_group: int = 80
    include_pair_states: bool = True


@dataclass(frozen=True)
class FeatureThreshold:
    feature: str
    q_low: float
    q_high: float
    coverage: float
    n_unique: int


@dataclass(frozen=True)
class EvmLatentStateResult:
    feature_state_metrics: pd.DataFrame
    pair_state_metrics: pd.DataFrame
    baselines: pd.DataFrame
    catalog: pd.DataFrame
    thresholds: pd.DataFrame
    manifest: dict[str, Any]


def _lower_any(text: str, hints: Sequence[str]) -> bool:
    lower = str(text).lower()
    return any(str(hint).lower() in lower for hint in hints)


def select_evm_state_feature_columns(
    frame: pd.DataFrame,
    *,
    include_aegmm: bool = False,
    feature_hints: Sequence[str] = DEFAULT_EVM_FEATURE_HINTS,
    target_hints: Sequence[str] = DEFAULT_TARGET_DERIVED_HINTS,
    aegmm_hints: Sequence[str] = DEFAULT_AEGMM_HINTS,
    required_columns: Iterable[str] = (),
    max_columns: int = 0,
) -> list[str]:
    """Select numeric, inference-compatible features for EVM state discovery."""

    required = {str(col) for col in required_columns}
    selected: list[str] = []
    for col in frame.columns:
        name = str(col)
        if name in required:
            continue
        if _lower_any(name, target_hints):
            continue
        if not include_aegmm and _lower_any(name, aegmm_hints):
            continue
        if not is_market_context_shock_entropy_feature(name):
            continue
        if not _lower_any(name, feature_hints):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            selected.append(name)
    if max_columns and len(selected) > int(max_columns):
        finite_share = np.asarray(
            [
                np.isfinite(
                    pd.to_numeric(frame[col], errors="coerce").to_numpy(
                        dtype=np.float64, copy=False
                    )
                ).mean()
                for col in selected
            ],
            dtype=np.float32,
        )
        priority = np.asarray(
            [evm_feature_priority_score(col) for col in selected], dtype=np.float32
        )
        order = np.lexsort((finite_share, priority))[::-1][: int(max_columns)]
        selected = [selected[int(i)] for i in order]
    return selected


def _safe_numeric(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return pd.Series(default, index=frame.index, dtype="float32")


def _as_float32(frame: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    return _safe_numeric(frame, col, default).to_numpy(dtype=np.float32, copy=False)


def _as_int8_bool(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.zeros(len(frame), dtype=np.int8)
    values = (
        pd.to_numeric(frame[col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    return (values > 0.5).astype(np.int8, copy=False)


def _rank_mask(
    frame: pd.DataFrame, score_col: str, month_col: str, top_fraction: float
) -> np.ndarray:
    if top_fraction >= 0.999:
        return np.ones(len(frame), dtype=bool)
    score = _safe_numeric(frame, score_col)
    if month_col in frame.columns:
        rank = score.groupby(frame[month_col], sort=False).rank(
            pct=True, method="first"
        )
    else:
        rank = score.rank(pct=True, method="first")
    return rank.ge(1.0 - float(top_fraction)).to_numpy(dtype=bool, copy=False)


if njit is not None:

    @njit(cache=True)
    def _stats_numba(
        mask: np.ndarray,
        ev: np.ndarray,
        clean: np.ndarray,
        dirty: np.ndarray,
        bad: np.ndarray,
        timeout: np.ndarray,
        stop: np.ndarray,
    ) -> tuple[int, float, float, float, float, float, float, float]:
        n = 0
        ev_sum = 0.0
        ev_pos = 0
        clean_sum = 0
        dirty_sum = 0
        bad_sum = 0
        timeout_sum = 0
        stop_sum = 0
        for i in range(mask.shape[0]):
            if not mask[i]:
                continue
            val = float(ev[i])
            if not np.isfinite(val):
                continue
            n += 1
            ev_sum += val
            if val > 0.0:
                ev_pos += 1
            clean_sum += int(clean[i])
            dirty_sum += int(dirty[i])
            bad_sum += int(bad[i])
            timeout_sum += int(timeout[i])
            stop_sum += int(stop[i])
        if n <= 0:
            return (0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        inv = 1.0 / n
        return (
            n,
            ev_sum * inv,
            ev_pos * inv,
            clean_sum * inv,
            dirty_sum * inv,
            bad_sum * inv,
            timeout_sum * inv,
            stop_sum * inv,
        )

else:
    _stats_numba = None


def _stats(
    mask: np.ndarray,
    *,
    ev: np.ndarray,
    clean: np.ndarray,
    dirty: np.ndarray,
    bad: np.ndarray,
    timeout: np.ndarray,
    stop: np.ndarray,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    if _stats_numba is not None:
        n, mean_ev, pos, clean_rate, dirty_rate, bad_rate, timeout_rate, stop_rate = (
            _stats_numba(
                mask,
                ev,
                clean,
                dirty,
                bad,
                timeout,
                stop,
            )
        )
    else:
        valid = mask & np.isfinite(ev)
        n = int(valid.sum())
        if n <= 0:
            mean_ev = pos = clean_rate = dirty_rate = bad_rate = timeout_rate = (
                stop_rate
            ) = np.nan
        else:
            mean_ev = float(ev[valid].mean())
            pos = float((ev[valid] > 0.0).mean())
            clean_rate = float(clean[valid].mean())
            dirty_rate = float(dirty[valid].mean())
            bad_rate = float(bad[valid].mean())
            timeout_rate = float(timeout[valid].mean())
            stop_rate = float(stop[valid].mean())
    return {
        "rows": int(n),
        "mean_ev_after_1pct": float(mean_ev),
        "positive_ev_rate": float(pos),
        "clean_exec_rate": float(clean_rate),
        "dirty_positive_rate": float(dirty_rate),
        "full_path_bad_mae_rate": float(bad_rate),
        "timeout_rate": float(timeout_rate),
        "stop_or_adverse_rate": float(stop_rate),
    }


def _objective_delta(
    metrics: Mapping[str, float], baseline: Mapping[str, float]
) -> float:
    ev_delta = float(metrics.get("mean_ev_after_1pct", np.nan)) - float(
        baseline.get("mean_ev_after_1pct", np.nan)
    )
    clean_delta = float(metrics.get("clean_exec_rate", np.nan)) - float(
        baseline.get("clean_exec_rate", np.nan)
    )
    dirty_delta = float(metrics.get("dirty_positive_rate", np.nan)) - float(
        baseline.get("dirty_positive_rate", np.nan)
    )
    bad_delta = float(metrics.get("full_path_bad_mae_rate", np.nan)) - float(
        baseline.get("full_path_bad_mae_rate", np.nan)
    )
    timeout_delta = float(metrics.get("timeout_rate", np.nan)) - float(
        baseline.get("timeout_rate", np.nan)
    )
    stop_delta = float(metrics.get("stop_or_adverse_rate", np.nan)) - float(
        baseline.get("stop_or_adverse_rate", np.nan)
    )
    total = 0.0
    if np.isfinite(ev_delta):
        total += ev_delta
    for value in (clean_delta, -dirty_delta, -bad_delta, -timeout_delta, -stop_delta):
        if np.isfinite(value):
            total += 0.0020 * value
    return float(total)


def _state_direction(train_delta: float, eval_delta: float, min_abs: float) -> str:
    train_ok = np.isfinite(train_delta) and abs(train_delta) >= min_abs
    eval_ok = np.isfinite(eval_delta) and abs(eval_delta) >= min_abs
    if not eval_ok:
        return "neutral_oos"
    if train_ok and np.sign(train_delta) == np.sign(eval_delta):
        return "favorable" if eval_delta > 0.0 else "unfavorable"
    if train_ok and np.sign(train_delta) != np.sign(eval_delta):
        return "unstable_flip"
    return "oos_only_favorable" if eval_delta > 0.0 else "oos_only_unfavorable"


def _feature_thresholds(
    group: pd.DataFrame,
    feature_columns: Sequence[str],
    config: EvmLatentStateConfig,
) -> list[FeatureThreshold]:
    out: list[FeatureThreshold] = []
    for feature in feature_columns:
        values = _safe_numeric(group, feature)
        coverage = float(values.notna().mean())
        if coverage < config.min_feature_coverage:
            continue
        unique = int(values.nunique(dropna=True))
        if unique < config.min_unique_values:
            continue
        q_low, q_high = values.quantile(list(config.quantiles)).to_numpy(dtype=float)
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
            continue
        out.append(
            FeatureThreshold(
                feature=str(feature),
                q_low=float(q_low),
                q_high=float(q_high),
                coverage=coverage,
                n_unique=unique,
            )
        )
    return out


def _assign_bins(values: pd.Series, threshold: FeatureThreshold) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series("mid", index=values.index, dtype=object)
    out.loc[numeric.le(threshold.q_low)] = "low"
    out.loc[numeric.gt(threshold.q_high)] = "high"
    out.loc[numeric.isna()] = "missing"
    return out


def _build_metric_context(
    frame: pd.DataFrame, config: EvmLatentStateConfig
) -> dict[str, np.ndarray]:
    return {
        "ev": _as_float32(frame, config.outcome_col),
        "clean": _as_int8_bool(frame, config.clean_col),
        "dirty": _as_int8_bool(frame, config.dirty_col),
        "bad": _as_int8_bool(frame, config.bad_mae_col),
        "timeout": _as_int8_bool(frame, config.timeout_col),
        "stop": _as_int8_bool(frame, config.stop_col),
    }


def _baseline_rows(
    frame: pd.DataFrame,
    *,
    period_label: str,
    side: str,
    archetype: str,
    config: EvmLatentStateConfig,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    rows: list[dict[str, Any]] = []
    ctx = _build_metric_context(frame, config)
    masks: dict[str, np.ndarray] = {}
    metrics_by_scope: dict[str, dict[str, float]] = {}
    for scope, top_fraction in config.top_fractions:
        mask = _rank_mask(frame, config.score_col, config.month_col, top_fraction)
        masks[scope] = mask
        metrics = _stats(mask, **ctx)
        metrics_by_scope[scope] = metrics
        rows.append(
            {
                "period": period_label,
                "side_name": side,
                "archetype_policy_key": archetype,
                "scope": scope,
                **metrics,
            }
        )
    return rows, metrics_by_scope, masks, ctx


def _state_metric_row(
    *,
    kind: str,
    state_name: str,
    side: str,
    archetype: str,
    feature: str,
    feature_b: str,
    bin_name: str,
    bin_b: str,
    scope: str,
    train_metrics: Mapping[str, float],
    eval_metrics: Mapping[str, float],
    train_baseline: Mapping[str, float],
    eval_baseline: Mapping[str, float],
    threshold: FeatureThreshold | None,
    threshold_b: FeatureThreshold | None,
    config: EvmLatentStateConfig,
) -> dict[str, Any]:
    train_delta = _objective_delta(train_metrics, train_baseline)
    eval_delta = _objective_delta(eval_metrics, eval_baseline)
    direction = _state_direction(
        train_delta, eval_delta, config.min_oos_objective_delta
    )
    row: dict[str, Any] = {
        "state_kind": kind,
        "state_name": state_name,
        "side_name": side,
        "archetype_policy_key": archetype,
        "feature": feature,
        "feature_b": feature_b,
        "feature_bin": bin_name,
        "feature_b_bin": bin_b,
        "scope": scope,
        "train_objective_delta": train_delta,
        "oos_objective_delta": eval_delta,
        "direction": direction,
        "q_low": threshold.q_low if threshold else np.nan,
        "q_high": threshold.q_high if threshold else np.nan,
        "q_low_b": threshold_b.q_low if threshold_b else np.nan,
        "q_high_b": threshold_b.q_high if threshold_b else np.nan,
    }
    for prefix, metrics, baseline in (
        ("train", train_metrics, train_baseline),
        ("oos", eval_metrics, eval_baseline),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
        row[f"{prefix}_ev_lift"] = float(
            metrics.get("mean_ev_after_1pct", np.nan)
        ) - float(baseline.get("mean_ev_after_1pct", np.nan))
        for rate_key in (
            "clean_exec_rate",
            "dirty_positive_rate",
            "full_path_bad_mae_rate",
            "timeout_rate",
            "stop_or_adverse_rate",
        ):
            base = float(baseline.get(rate_key, np.nan))
            value = float(metrics.get(rate_key, np.nan))
            row[f"{prefix}_{rate_key}_lift"] = (
                value / base if np.isfinite(base) and abs(base) > 1e-12 else np.nan
            )
            row[f"{prefix}_{rate_key}_delta"] = (
                value - base if np.isfinite(value) and np.isfinite(base) else np.nan
            )
    return row


def discover_evm_latent_states(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: EvmLatentStateConfig | None = None,
    eval_label: str = "oos",
) -> EvmLatentStateResult:
    """Fit train thresholds and assess side x archetype states on OOS rows."""

    cfg = config or EvmLatentStateConfig()
    required = [cfg.side_col, cfg.archetype_col, cfg.score_col, cfg.outcome_col]
    missing_train = [col for col in required if col not in train.columns]
    missing_oos = [col for col in required if col not in oos.columns]
    if missing_train or missing_oos:
        raise ValueError(
            f"Missing required columns: train={missing_train} oos={missing_oos}"
        )

    feature_columns = [
        str(col)
        for col in feature_columns
        if col in train.columns
        and col in oos.columns
        and pd.api.types.is_numeric_dtype(train[col])
    ]
    feature_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    train_groups = train.groupby(
        [cfg.side_col, cfg.archetype_col], sort=False, observed=True
    ).groups
    for (side, archetype), train_idx in train_groups.items():
        side_text = str(side)
        arch_text = str(archetype)
        train_group = train.loc[train_idx]
        oos_group = oos.loc[
            oos[cfg.side_col].astype(str).eq(side_text)
            & oos[cfg.archetype_col].astype(str).eq(arch_text)
        ]
        if len(train_group) < cfg.min_group_rows or len(oos_group) < max(
            cfg.min_state_rows, 10
        ):
            continue
        train_base_rows, train_base, train_masks, train_ctx = _baseline_rows(
            train_group,
            period_label="train",
            side=side_text,
            archetype=arch_text,
            config=cfg,
        )
        oos_base_rows, oos_base, oos_masks, oos_ctx = _baseline_rows(
            oos_group,
            period_label=eval_label,
            side=side_text,
            archetype=arch_text,
            config=cfg,
        )
        baseline_rows.extend(train_base_rows)
        baseline_rows.extend(oos_base_rows)

        thresholds = _feature_thresholds(train_group, feature_columns, cfg)
        if not thresholds:
            continue
        # Prefer features that have enough train coverage and variance, but keep
        # the candidate count bounded for pairwise discovery.
        thresholds = sorted(
            thresholds, key=lambda t: (t.coverage, t.n_unique), reverse=True
        )[: max(1, int(cfg.max_features_per_group))]
        threshold_by_feature = {t.feature: t for t in thresholds}
        for threshold in thresholds:
            threshold_rows.append(
                {
                    "side_name": side_text,
                    "archetype_policy_key": arch_text,
                    **asdict(threshold),
                }
            )
            train_bins = _assign_bins(train_group[threshold.feature], threshold)
            oos_bins = _assign_bins(oos_group[threshold.feature], threshold)
            for bin_name in ("low", "mid", "high", "missing"):
                train_state = train_bins.eq(bin_name).to_numpy(dtype=bool, copy=False)
                oos_state = oos_bins.eq(bin_name).to_numpy(dtype=bool, copy=False)
                if int(train_state.sum()) < cfg.min_state_rows or int(
                    oos_state.sum()
                ) < max(10, cfg.min_state_rows // 2):
                    continue
                for scope, _top_fraction in cfg.top_fractions:
                    train_mask = train_masks[scope] & train_state
                    oos_mask = oos_masks[scope] & oos_state
                    train_metrics = _stats(train_mask, **train_ctx)
                    oos_metrics = _stats(oos_mask, **oos_ctx)
                    if train_metrics["rows"] < cfg.min_state_rows or oos_metrics[
                        "rows"
                    ] < max(10, cfg.min_state_rows // 2):
                        continue
                    feature_rows.append(
                        _state_metric_row(
                            kind="feature_bin",
                            state_name=f"{threshold.feature}={bin_name}",
                            side=side_text,
                            archetype=arch_text,
                            feature=threshold.feature,
                            feature_b="",
                            bin_name=bin_name,
                            bin_b="",
                            scope=scope,
                            train_metrics=train_metrics,
                            eval_metrics=oos_metrics,
                            train_baseline=train_base[scope],
                            eval_baseline=oos_base[scope],
                            threshold=threshold,
                            threshold_b=None,
                            config=cfg,
                        )
                    )

        if not cfg.include_pair_states or len(thresholds) < 2:
            continue
        feature_df = pd.DataFrame(feature_rows)
        local = feature_df.loc[
            feature_df["side_name"].eq(side_text)
            & feature_df["archetype_policy_key"].eq(arch_text)
            & feature_df["scope"].eq("top20")
            & feature_df["state_kind"].eq("feature_bin")
        ].copy()
        if local.empty:
            continue
        local["abs_train_delta"] = pd.to_numeric(
            local["train_objective_delta"], errors="coerce"
        ).abs()
        top_feature_names = (
            local.sort_values("abs_train_delta", ascending=False)["feature"]
            .drop_duplicates()
            .head(max(2, int(cfg.top_features_for_pairs)))
        ).tolist()
        pair_count = 0
        for i, feature_a in enumerate(top_feature_names):
            for feature_b in top_feature_names[i + 1 :]:
                t_a = threshold_by_feature.get(feature_a)
                t_b = threshold_by_feature.get(feature_b)
                if t_a is None or t_b is None:
                    continue
                train_a = _assign_bins(train_group[feature_a], t_a)
                train_b = _assign_bins(train_group[feature_b], t_b)
                oos_a = _assign_bins(oos_group[feature_a], t_a)
                oos_b = _assign_bins(oos_group[feature_b], t_b)
                for bin_a in ("low", "mid", "high"):
                    for bin_b in ("low", "mid", "high"):
                        train_state = (train_a.eq(bin_a) & train_b.eq(bin_b)).to_numpy(
                            dtype=bool, copy=False
                        )
                        oos_state = (oos_a.eq(bin_a) & oos_b.eq(bin_b)).to_numpy(
                            dtype=bool, copy=False
                        )
                        if int(train_state.sum()) < cfg.min_state_rows or int(
                            oos_state.sum()
                        ) < max(10, cfg.min_state_rows // 2):
                            continue
                        for scope, _top_fraction in cfg.top_fractions:
                            train_mask = train_masks[scope] & train_state
                            oos_mask = oos_masks[scope] & oos_state
                            train_metrics = _stats(train_mask, **train_ctx)
                            oos_metrics = _stats(oos_mask, **oos_ctx)
                            if train_metrics[
                                "rows"
                            ] < cfg.min_state_rows or oos_metrics["rows"] < max(
                                10, cfg.min_state_rows // 2
                            ):
                                continue
                            pair_rows.append(
                                _state_metric_row(
                                    kind="pair_bin",
                                    state_name=f"{feature_a}={bin_a} & {feature_b}={bin_b}",
                                    side=side_text,
                                    archetype=arch_text,
                                    feature=feature_a,
                                    feature_b=feature_b,
                                    bin_name=bin_a,
                                    bin_b=bin_b,
                                    scope=scope,
                                    train_metrics=train_metrics,
                                    eval_metrics=oos_metrics,
                                    train_baseline=train_base[scope],
                                    eval_baseline=oos_base[scope],
                                    threshold=t_a,
                                    threshold_b=t_b,
                                    config=cfg,
                                )
                            )
                        pair_count += 1
                        if pair_count >= cfg.max_pair_states_per_group:
                            break
                    if pair_count >= cfg.max_pair_states_per_group:
                        break
                if pair_count >= cfg.max_pair_states_per_group:
                    break
            if pair_count >= cfg.max_pair_states_per_group:
                break

    feature_metrics = pd.DataFrame(feature_rows)
    pair_metrics = pd.DataFrame(pair_rows)
    all_metrics = pd.concat(
        [df for df in (feature_metrics, pair_metrics) if not df.empty],
        ignore_index=True,
        copy=False,
    )
    if all_metrics.empty:
        catalog = pd.DataFrame()
    else:
        catalog = all_metrics.loc[
            all_metrics["direction"].isin(["favorable", "unfavorable"])
            & pd.to_numeric(all_metrics["oos_objective_delta"], errors="coerce")
            .abs()
            .ge(cfg.min_oos_objective_delta)
        ].copy()
        if not catalog.empty:
            catalog["oos_abs_objective_delta"] = pd.to_numeric(
                catalog["oos_objective_delta"], errors="coerce"
            ).abs()
            catalog = catalog.sort_values(
                ["scope", "oos_abs_objective_delta", "oos_rows"],
                ascending=[True, False, False],
            )

    manifest = {
        "method": "train-threshold latent state discovery for side x archetype EV calibration",
        "config": asdict(cfg),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "groups_evaluated": int(
            len(
                pd.DataFrame(baseline_rows)[
                    ["side_name", "archetype_policy_key"]
                ].drop_duplicates()
            )
        )
        if baseline_rows
        else 0,
        "feature_state_rows": int(len(feature_metrics)),
        "pair_state_rows": int(len(pair_metrics)),
        "catalog_rows": int(len(catalog)),
        "leakage_contract": (
            "feature thresholds are fitted on train rows only; OOS rows are assigned with frozen thresholds; "
            "target/outcome columns are excluded from state features and used only for OOS metrics"
        ),
    }
    return EvmLatentStateResult(
        feature_state_metrics=feature_metrics,
        pair_state_metrics=pair_metrics,
        baselines=pd.DataFrame(baseline_rows),
        catalog=catalog,
        thresholds=pd.DataFrame(threshold_rows),
        manifest=manifest,
    )


def downcast_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast a frame in place where possible and return it."""

    out = frame
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype(np.float32, copy=False)
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out
