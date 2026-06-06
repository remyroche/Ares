"""
Candidate Selector for Inference.

This module applies candidate thresholds to select trade candidates:
- Uses select_trade_candidates_vectorized from candidates.py
- Applies thresholds: extreme_pct=0.05, min_range_pct=0.06, min_vol_zscore=1.5
- Returns long_candidates and short_candidates
"""

from typing import Dict, Iterable, List, Tuple, Any, Optional

import pandas as pd
import numpy as np

from extreme_price_movements.inference.config import _resolve_runtime_cfg
from extreme_price_movements.utils import tprint


def _feature_to_panel_flat(
    name: str,
    value: Any,
    close_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Coerce a live feature into the flat panel shape expected by mask mining.

    Live feature dictionaries can contain full feature panels as well as
    latest-only symbol vectors. LGBM rule evaluation expects every raw feature to
    share the same flattened ``timestamp x symbol`` length as the active price
    panel, so latest-only vectors are placed on the latest row and left NaN
    elsewhere.
    """
    idx = close_df.index
    columns = close_df.columns
    n_ts, n_syms = close_df.shape
    expected = int(n_ts * n_syms)

    try:
        if isinstance(value, pd.DataFrame):
            return (
                value.reindex(index=idx, columns=columns)
                .to_numpy(dtype=np.float32, copy=False)
                .reshape(-1)
            )

        if isinstance(value, pd.Series):
            series = pd.to_numeric(value, errors="coerce")
            if series.index.equals(idx) or set(series.index).issubset(set(idx)):
                arr_2d = np.repeat(
                    series.reindex(idx).to_numpy(dtype=np.float32)[:, None],
                    n_syms,
                    axis=1,
                )
                return arr_2d.reshape(-1)
            if series.index.equals(columns) or set(series.index).issubset(set(columns)):
                arr_2d = np.full((n_ts, n_syms), np.nan, dtype=np.float32)
                arr_2d[-1, :] = series.reindex(columns).to_numpy(dtype=np.float32)
                return arr_2d.reshape(-1)

        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == close_df.shape:
            return arr.reshape(-1)
        flat = arr.reshape(-1)
        if flat.size == expected:
            return flat.astype(np.float32, copy=False)
        if flat.size == n_syms:
            arr_2d = np.full((n_ts, n_syms), np.nan, dtype=np.float32)
            arr_2d[-1, :] = flat
            return arr_2d.reshape(-1)
        if flat.size == n_ts:
            return np.repeat(flat[:, None], n_syms, axis=1).reshape(-1)
    except Exception as exc:
        tprint(f"candidate_selector: failed to align mask feature {name}: {exc}")
        return None

    tprint(
        "candidate_selector: skipping mask feature with incompatible shape "
        f"name={name} size={np.asarray(value).size if value is not None else 0} "
        f"expected={expected} n_ts={n_ts} n_syms={n_syms}"
    )
    return None


def _fallback_rank_candidates(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    *,
    metric: str,
    cfg: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Select candidates with the legacy rank/range filters when masks are silent."""
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return [], []

    metric_df = feats.get(metric)
    if not isinstance(metric_df, pd.DataFrame):
        metric_df = close.pct_change(12).fillna(0.0)
    latest_metric = (
        metric_df.reindex(columns=close.columns)
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = latest_metric.notna()

    min_move = float(cfg.get("train_min_move_12h_pct", 0.06) or 0.0)
    if min_move > 0.0:
        valid &= latest_metric.abs() >= min_move

    range_df = feats.get("range_12h_pct")
    min_range = float(cfg.get("train_min_range_pct", 0.06) or 0.0)
    if isinstance(range_df, pd.DataFrame) and min_range > 0.0:
        latest_range = (
            range_df.reindex(columns=close.columns)
            .iloc[-1]
            .replace([np.inf, -np.inf], np.nan)
        )
        valid &= latest_range >= min_range

    vol_df = feats.get("volatility_zscore")
    min_vol = float(cfg.get("train_min_vol_zscore", 1.5) or 0.0)
    if isinstance(vol_df, pd.DataFrame) and min_vol > 0.0:
        latest_vol = (
            vol_df.reindex(columns=close.columns)
            .iloc[-1]
            .replace([np.inf, -np.inf], np.nan)
        )
        vol_valid = latest_vol >= min_vol
        if bool(vol_valid.any()):
            valid &= vol_valid
        else:
            tprint(
                "candidate_selector: volatility fallback filter skipped because "
                "no latest symbols passed it"
            )

    eligible = latest_metric[valid].dropna()
    if eligible.empty:
        return [], []

    extreme_pct = float(cfg.get("train_extreme_pct_hourly", 0.05) or 0.05)
    n_keep = max(1, int(np.ceil(len(latest_metric.dropna()) * extreme_pct)))
    long_candidates = (
        eligible[eligible > 0.0]
        .sort_values(ascending=False)
        .head(n_keep)
        .index.tolist()
    )
    short_candidates = (
        eligible[eligible < 0.0].sort_values().head(n_keep).index.tolist()
    )
    return long_candidates, short_candidates


def _rescue_rank_candidates(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    *,
    metric: str,
    cfg: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return a bounded rank-only pool when live masks are silent."""
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return [], []

    metric_df = feats.get(metric)
    if not isinstance(metric_df, pd.DataFrame):
        metric_df = close.pct_change(12).fillna(0.0)
    latest_metric = (
        metric_df.reindex(columns=close.columns)
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if latest_metric.empty:
        return [], []

    extreme_pct = float(cfg.get("candidate_mask_rescue_extreme_pct", 0.02) or 0.02)
    min_per_side = int(cfg.get("candidate_mask_rescue_min_per_side", 5) or 5)
    max_per_side = int(cfg.get("candidate_mask_rescue_max_per_side", 12) or 12)
    n_keep = int(np.ceil(float(len(latest_metric)) * max(extreme_pct, 0.0)))
    n_keep = max(1, min(max_per_side, max(min_per_side, n_keep)))

    long_candidates = (
        latest_metric[latest_metric > 0.0]
        .sort_values(ascending=False)
        .head(n_keep)
        .index.tolist()
    )
    short_candidates = (
        latest_metric[latest_metric < 0.0]
        .sort_values(ascending=True)
        .head(n_keep)
        .index.tolist()
    )
    return long_candidates, short_candidates


def _prepare_shared_mask_context(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    mask_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the expensive LGBM rule resolver once for a live inference cycle."""
    from extreme_price_movements.lgbm_based_mask_generation import (
        CanonicalRuleMaskResolver,
        FeatureProcessor,
    )

    close_df = panel["close"]
    n_ts, n_syms = close_df.shape
    idx_flat = np.repeat(close_df.index.to_numpy(), n_syms)
    sym_flat = np.tile(close_df.columns.to_numpy(), n_ts)

    feats_1d = {}
    for k, v in feats.items():
        arr = _feature_to_panel_flat(str(k), v, close_df)
        if arr is not None:
            feats_1d[str(k)] = arr

    fp = FeatureProcessor()
    X, metadata, _ = fp.prepare_features(feats_1d, idx_flat, sym_flat, mask_cfg)
    resolver = CanonicalRuleMaskResolver(X, metadata, raw_feature_lookup=feats_1d)
    tprint(
        "candidate_selector: shared CanonicalRuleMaskResolver initialized "
        f"features={len(feats_1d)} rows={int(n_ts * n_syms)}"
    )
    return {
        "feats_1d": feats_1d,
        "resolver": resolver,
        "X": X,
        "metadata": metadata,
        "symbols": list(close_df.columns),
        "timestamps": list(close_df.index),
    }


def build_latest_prepared_feature_frames(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    mask_cfg: Dict[str, Any],
    *,
    symbols: Optional[Iterable[str]] = None,
    required_columns: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Return latest FeatureProcessor columns as live feature frames.

    LGBM base models are trained on the FeatureProcessor output contract, not
    only the raw hourly feature panels. Live inference must therefore expose
    these prepared columns to the model scorer after the shared training-path
    raw features have been computed.
    """

    close_df = panel.get("close")
    if not isinstance(close_df, pd.DataFrame) or close_df.empty:
        return {}
    context = _prepare_shared_mask_context(panel, feats, mask_cfg)
    X = context.get("X")
    metadata = context.get("metadata") or []
    if not isinstance(X, np.ndarray) or X.size == 0 or not metadata:
        return {}

    all_symbols = [str(sym) for sym in context.get("symbols", list(close_df.columns))]
    selected_symbols = [
        str(sym)
        for sym in (symbols if symbols is not None else all_symbols)
        if str(sym) in set(all_symbols)
    ]
    if not selected_symbols:
        return {}

    feature_names = [str(getattr(meta, "feature_name", "")) for meta in metadata]
    feature_names = [name for name in feature_names if name]
    if not feature_names:
        return {}

    keep: Optional[set[str]] = None
    if required_columns is not None:
        keep = {str(col) for col in required_columns if str(col)}
    n_syms = int(len(all_symbols))
    latest_start = max(0, int(X.shape[0]) - n_syms)
    latest = np.asarray(X[latest_start : latest_start + n_syms, :], dtype=np.float32)
    latest_df = pd.DataFrame(latest, index=all_symbols, columns=feature_names)
    latest_df = latest_df.reindex(selected_symbols)
    latest_ts = pd.Timestamp(close_df.index.max())

    out: Dict[str, pd.DataFrame] = {}
    for col in feature_names:
        if keep is not None and col not in keep:
            continue
        series = latest_df[col]
        out[col] = pd.DataFrame(
            [series.to_numpy(dtype=np.float32, copy=False)],
            index=pd.DatetimeIndex([latest_ts]),
            columns=latest_df.index,
        )
    tprint(
        "candidate_selector: prepared latest FeatureProcessor frames "
        f"features={len(out)} symbols={len(selected_symbols)} ts={latest_ts}"
    )
    return out


def _build_mask_for_mode(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    mask_cfg: Dict[str, Any],
    prepared_context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    from extreme_price_movements.mask_optimiser import (
        _compute_z_cache,
        _generate_event_masks,
        _generate_event_masks_fast,
    )

    close_df = panel["close"]
    n_ts, n_syms = close_df.shape

    if prepared_context is None:
        prepared_context = _prepare_shared_mask_context(panel, feats, mask_cfg)
    feats_1d = dict(prepared_context.get("feats_1d") or {})
    resolver = prepared_context.get("resolver")

    def _normalize_legacy_base_event(base_event: str) -> str:
        base_event = str(base_event or "").strip()
        if base_event and "|" not in base_event and base_event.startswith("price_"):
            return f"({base_event}==1)|(*)|(*)"
        return base_event

    base = _normalize_legacy_base_event(
        mask_cfg.get("base_event_trigger") or mask_cfg.get("canonical_key") or ""
    )
    if base and resolver is not None:
        try:
            mask_1d = resolver.get_mask(base)
            mask_2d = mask_1d.reshape((n_ts, n_syms))
            return pd.DataFrame(
                mask_2d, index=close_df.index, columns=close_df.columns, dtype=bool
            )
        except Exception:
            pass

    family = str(mask_cfg.get("family", "")).strip()
    if not family:
        return pd.DataFrame(
            False, index=close_df.index, columns=close_df.columns, dtype=bool
        )

    idx = close_df.index
    if len(idx) >= 2:
        delta_seconds = max((idx[1] - idx[0]).total_seconds(), 1.0)
        bph = max(int(round(3600.0 / delta_seconds)), 1)
    else:
        bph = 1

    close_arr = close_df.to_numpy(dtype=np.float32, copy=False).ravel()
    high_arr = (
        panel["high"]
        .reindex(index=idx, columns=close_df.columns)
        .to_numpy(dtype=np.float32, copy=False)
        .ravel()
    )
    low_arr = (
        panel["low"]
        .reindex(index=idx, columns=close_df.columns)
        .to_numpy(dtype=np.float32, copy=False)
        .ravel()
    )
    volume_df = panel.get("volume")
    volume_arr = None
    if isinstance(volume_df, pd.DataFrame):
        volume_arr = (
            volume_df.reindex(index=idx, columns=close_df.columns)
            .to_numpy(dtype=np.float32, copy=False)
            .ravel()
        )

    ret1_df = feats.get("ret1h")
    if isinstance(ret1_df, pd.DataFrame):
        ret_1 = (
            ret1_df.reindex(index=idx, columns=close_df.columns)
            .to_numpy(dtype=np.float32, copy=False)
            .ravel()
        )
    else:
        ret_1 = (
            close_df.pct_change()
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
            .ravel()
        )

    vol_df = feats.get("atr_pct_base")
    if not isinstance(vol_df, pd.DataFrame):
        vol_df = feats.get("atr_pct")
    if isinstance(vol_df, pd.DataFrame):
        vol_g = (
            vol_df.reindex(index=idx, columns=close_df.columns)
            .to_numpy(dtype=np.float32, copy=False)
            .ravel()
        )
    else:
        close_safe = np.maximum(close_df.to_numpy(dtype=np.float32, copy=False), 1e-6)
        vol_g = (
            (
                (
                    panel["high"]
                    .reindex(index=idx, columns=close_df.columns)
                    .to_numpy(dtype=np.float32, copy=False)
                    - panel["low"]
                    .reindex(index=idx, columns=close_df.columns)
                    .to_numpy(dtype=np.float32, copy=False)
                )
                / close_safe
            )
            .astype(np.float32)
            .ravel()
        )

    asset_groups = {
        int(i): np.arange(i, n_ts * n_syms, n_syms, dtype=np.int32)
        for i in range(n_syms)
    }

    z_hours = float(mask_cfg.get("z_hours", 1.0) or 1.0)
    duration_hours = float(mask_cfg.get("duration_hours", 1.0) or 1.0)
    z_bars = max(int(round(z_hours * bph)), 1)
    duration_bars = max(int(round(duration_hours * bph)), 1)
    tprint("candidate_selector: calling _compute_z_cache...")
    if not hasattr(_build_mask_for_mode, "_zc_cache"):
        _build_mask_for_mode._zc_cache = {}
    _zc_cache = _build_mask_for_mode._zc_cache
    _zc_key = (
        int(z_bars),
        int(n_ts),
        int(n_syms),
        str(close_df.index[0]) if len(close_df.index) else "",
        str(close_df.index[-1]) if len(close_df.index) else "",
        tuple(map(str, close_df.columns)),
    )
    if _zc_key in _zc_cache:
        zc = _zc_cache[_zc_key]
        tprint("candidate_selector: _compute_z_cache complete (cached).")
    else:
        zc = _compute_z_cache(
            high=high_arr,
            low=low_arr,
            close=close_arr,
            ret_1=ret_1,
            vol_g=vol_g,
            asset_groups=asset_groups,
            z=z_bars,
            bph=bph,
            volume=volume_arr,
            precomputed=feats_1d,
        )
        _zc_cache[_zc_key] = zc
        tprint("candidate_selector: _compute_z_cache complete.")

    name = str(mask_cfg.get("name", "") or "")
    feature_base = str(mask_cfg.get("feature_base", "") or "")
    param_token = None
    if name and "|p=" in name:
        param_token = name.split("|p=", 1)[1].split("|", 1)[0]

    candidate = None
    parsed_token = param_token or str(mask_cfg.get("param", "") or "")
    if parsed_token:
        if "_gt_" in parsed_token:
            parsed_feature_base, parsed_threshold = parsed_token.rsplit("_gt_", 1)
            candidate = {
                "family": family,
                "feature_base": feature_base or parsed_feature_base,
                "direction": "gt",
                "threshold": float(parsed_threshold),
            }
        elif "_lt_" in parsed_token:
            parsed_feature_base, parsed_threshold = parsed_token.rsplit("_lt_", 1)
            candidate = {
                "family": family,
                "feature_base": feature_base or parsed_feature_base,
                "direction": "lt",
                "threshold": float(parsed_threshold),
            }

    if candidate is None and family in {
        "std_threshold",
        "abs_move_threshold",
        "std_plus_abs",
    }:
        move_df = feats.get("ret12h")
        if not isinstance(move_df, pd.DataFrame):
            move_df = close_df.pct_change(12).fillna(0.0)
        move_df = move_df.reindex(index=idx, columns=close_df.columns).fillna(0.0)
        if family == "std_threshold":
            threshold_df = move_df.rolling(24 * 30, min_periods=2).std().fillna(
                0.0
            ) * float(mask_cfg.get("param", 0.0) or 0.0)
            mask_h_df = move_df >= threshold_df
            mask_l_df = (-move_df) >= threshold_df
        elif family == "abs_move_threshold":
            threshold = float(mask_cfg.get("param", 0.0) or 0.0) / 100.0
            mask_h_df = move_df >= threshold
            mask_l_df = (-move_df) >= threshold
        else:
            param_val = mask_cfg.get("param", (0.0, 0.0))
            if isinstance(param_val, (list, tuple)) and len(param_val) >= 2:
                std_val = float(param_val[0])
                abs_val = float(param_val[1]) / 100.0
            else:
                std_val = 0.0
                abs_val = float(param_val or 0.0) / 100.0
            threshold_df = (
                move_df.rolling(24 * 30, min_periods=2).std().fillna(0.0) * std_val
            )
            mask_h_df = (move_df >= threshold_df) & (move_df >= abs_val)
            mask_l_df = ((-move_df) >= threshold_df) & ((-move_df) >= abs_val)

        mask_df = (mask_h_df | mask_l_df).fillna(False).astype(bool)
        if duration_bars > 1:
            for lag in range(1, duration_bars):
                mask_df = mask_df | mask_df.shift(lag, fill_value=False)
        return mask_df.astype(bool)

    try:
        if candidate is not None:
            tprint("candidate_selector: Calling _generate_event_masks_fast...")
            mask_h, mask_l = _generate_event_masks_fast(candidate=candidate, zc=zc)
            tprint("candidate_selector: _generate_event_masks_fast complete.")
        else:
            param_val = mask_cfg.get("param")
            if param_val is None:
                return pd.DataFrame(
                    False, index=close_df.index, columns=close_df.columns, dtype=bool
                )
            mask_h, mask_l = _generate_event_masks(
                family=family,
                param_val=param_val,
                up_move=zc["up"],
                dn_move=zc["dn"],
                rolling_std_up=zc["std_up"],
                rolling_std_dn=zc["std_dn"],
                asset_groups=asset_groups,
                duration_bars=duration_bars,
            )
            tprint("candidate_selector: _generate_event_masks complete.")
        mask_2d = (mask_h | mask_l).reshape((n_ts, n_syms))
        return pd.DataFrame(
            mask_2d, index=close_df.index, columns=close_df.columns, dtype=bool
        )
    except Exception:
        return pd.DataFrame(
            False, index=close_df.index, columns=close_df.columns, dtype=bool
        )


def build_strategy_candidate_masks(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    strategies: Iterable[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Return latest symbols passing each LGBM-generated strategy mask.

    Strategies are rows from
    load_inference_candidate_mask_params_per_bucket(), where strategy_id is the
    safe ID used by train_base/train_meta and base_event_trigger is the
    canonical LGBM rule expression.
    """
    out: Dict[str, List[str]] = {}
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return out

    prepared_strategies: List[Tuple[Dict[str, Any], str, Dict[str, Any]]] = []
    for strategy in strategies:
        if not isinstance(strategy, dict):
            continue
        strategy_id = str(strategy.get("strategy_id", "") or "")
        if not strategy_id:
            continue
        mask_cfg = dict(strategy.get("mask_params", {}) or {})
        mask_cfg.update(strategy)
        canonical_key = str(
            strategy.get("base_event_trigger")
            or strategy.get("canonical_key")
            or mask_cfg.get("canonical_key")
            or ""
        )
        if canonical_key:
            mask_cfg.setdefault("base_event_trigger", canonical_key)
            mask_cfg.setdefault("canonical_key", canonical_key)
        prepared_strategies.append((strategy, strategy_id, mask_cfg))

    shared_context: Optional[Dict[str, Any]] = None
    if prepared_strategies:
        try:
            shared_context = _prepare_shared_mask_context(
                panel, feats, prepared_strategies[0][2]
            )
        except Exception as exc:
            tprint(
                "candidate_selector: shared mask context init failed; "
                f"falling back to per-strategy preparation: {exc}"
            )

    for strategy, strategy_id, mask_cfg in prepared_strategies:
        try:
            mask_df = _build_mask_for_mode(
                panel, feats, mask_cfg, prepared_context=shared_context
            )
        except Exception as exc:
            tprint(
                "candidate_selector: strategy mask failed "
                f"strategy_id={strategy_id} error={exc}"
            )
            out[strategy_id] = []
            continue
        if not isinstance(mask_df, pd.DataFrame) or mask_df.empty:
            out[strategy_id] = []
            continue
        latest = mask_df.reindex(index=close.index, columns=close.columns).iloc[-1]
        latest_bool = latest.fillna(False).astype(bool)
        passed = latest_bool[latest_bool].index.astype(str).tolist()
        denominator = int(latest_bool.shape[0])
        side = str(strategy.get("trade_side") or strategy.get("side") or "")
        support = (float(len(passed)) / float(denominator)) if denominator > 0 else 0.0
        tprint(
            "candidate_selector: strategy mask latest support "
            f"side={side or 'unknown'} strategy_id={strategy_id} "
            f"passed={len(passed)}/{denominator} support={support:.2%}"
        )
        out[strategy_id] = passed
    return out


def _up_down_zones(
    feats: Dict[str, pd.DataFrame], panel: Dict[str, pd.DataFrame], metric: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric in feats:
        metric_df = feats[metric]
    else:
        metric_df = panel["close"].pct_change(24).fillna(0.0)
    ranks = metric_df.rank(axis=1, method="first", na_option="keep", pct=True)
    up_zone = (ranks > 0.5).fillna(False).astype(bool)
    down_zone = (ranks <= 0.5).fillna(False).astype(bool)
    return up_zone, down_zone


def _require_mode_cfg(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mode_cfg = dict(cfg.get("candidate_mask_params_by_mode", {}) or {})
    required = ["price_up_tf", "price_up_mr", "price_down_tf", "price_down_mr"]
    missing = [m for m in required if m not in mode_cfg]
    if missing:
        raise ValueError(
            "Per-mode mask params missing; refusing legacy fallback. "
            f"missing={missing} available={sorted(mode_cfg.keys())}"
        )
    return mode_cfg


def select_candidates(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    extreme_pct: Optional[float] = None,
    min_move_12h_pct: Optional[float] = None,
    min_range_pct: Optional[float] = None,
    min_vol_zscore: Optional[float] = None,
    metric: str = "ret12h",
    chop_thr: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Select trade candidates using mask optimiser logic.

    Applies the candidate selection algorithm with optimized parameters from
    mask_optimiser.py instead of the legacy threshold selection.

    Args:
        panel: Price panel with open, high, low, close, volume DataFrames
        feats: Feature dictionary with computed market features
        extreme_pct: Deprecated and unsupported (raises)
        min_move_12h_pct: Deprecated and unsupported (raises)
        min_range_pct: Deprecated and unsupported (raises)
        min_vol_zscore: Deprecated and unsupported (raises)
        metric: Performance metric to rank by
        chop_thr: Maximum choppiness score threshold

    Returns:
        Tuple of (long_candidates, short_candidates) - lists of symbol strings
    """
    cfg = _resolve_runtime_cfg()
    if any(
        v is not None
        for v in (extreme_pct, min_move_12h_pct, min_range_pct, min_vol_zscore)
    ):
        raise ValueError(
            "Legacy threshold overrides are not supported after per-mode mask migration. "
            "Use persisted candidate_mask_params_by_mode instead."
        )
    mode_cfg = _require_mode_cfg(cfg)
    default_cfg = {
        "family": cfg.get("family", "top_movers"),
        "param": cfg.get("param", 5.0),
        "z_hours": cfg.get("z_hours", 12.0),
        "duration_hours": cfg.get("duration_hours", 1.0),
    }

    try:
        up_zone, down_zone = _up_down_zones(feats, panel, metric=metric)
        shared_context: Optional[Dict[str, Any]] = None
        try:
            shared_context = _prepare_shared_mask_context(
                panel, feats, mode_cfg.get("price_up_tf", default_cfg)
            )
        except Exception as exc:
            tprint(
                "candidate_selector: shared select_candidates mask context init "
                f"failed; falling back to per-mode preparation: {exc}"
            )
        m_up_tf = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_up_tf", default_cfg),
            prepared_context=shared_context,
        )
        m_up_mr = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_up_mr", default_cfg),
            prepared_context=shared_context,
        )
        m_down_tf = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_down_tf", default_cfg),
            prepared_context=shared_context,
        )
        m_down_mr = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_down_mr", default_cfg),
            prepared_context=shared_context,
        )

        long_mask = (up_zone & m_up_tf) | (down_zone & m_down_mr)
        short_mask = (up_zone & m_up_mr) | (down_zone & m_down_tf)

    except Exception as e:
        raise RuntimeError(f"Per-mode candidate mask generation failed: {e}") from e

    if long_mask.empty and short_mask.empty:
        tprint("No candidates found - candidate masks are empty")
        return [], []

    latest_ts = long_mask.index[-1]
    latest_long = long_mask.loc[latest_ts]
    latest_short = short_mask.loc[latest_ts]
    long_candidates = latest_long[latest_long].index.tolist()
    short_candidates = latest_short[latest_short].index.tolist()

    if (
        not long_candidates
        and not short_candidates
        and bool(cfg.get("candidate_mask_empty_fallback_enabled", True))
    ):
        long_candidates, short_candidates = _fallback_rank_candidates(
            panel,
            feats,
            metric=metric,
            cfg=cfg,
        )
        tprint(
            "candidate_selector: optimized masks were silent; "
            "used rank/threshold fallback "
            f"long={len(long_candidates)} short={len(short_candidates)}"
        )
    if (
        not long_candidates
        and not short_candidates
        and bool(cfg.get("candidate_mask_rank_rescue_enabled", True))
    ):
        long_candidates, short_candidates = _rescue_rank_candidates(
            panel,
            feats,
            metric=metric,
            cfg=cfg,
        )
        tprint(
            "candidate_selector: strict fallback was silent; "
            "used rank-only rescue pool "
            f"long={len(long_candidates)} short={len(short_candidates)}"
        )

    tprint(
        f"Selected {len(long_candidates)} long candidates, "
        f"{len(short_candidates)} short candidates"
    )

    return long_candidates, short_candidates


def select_candidates_at_timestamp(
    panel: Dict[str, pd.DataFrame],
    feats: Dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    extreme_pct: Optional[float] = None,
    min_move_12h_pct: Optional[float] = None,
    min_range_pct: Optional[float] = None,
    min_vol_zscore: Optional[float] = None,
    metric: str = "ret12h",
    chop_thr: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """Select candidates at a specific timestamp using mask_optimiser logic."""
    cfg = _resolve_runtime_cfg()
    if any(
        v is not None
        for v in (extreme_pct, min_move_12h_pct, min_range_pct, min_vol_zscore)
    ):
        raise ValueError(
            "Legacy threshold overrides are not supported after per-mode mask migration. "
            "Use persisted candidate_mask_params_by_mode instead."
        )
    mode_cfg = _require_mode_cfg(cfg)
    default_cfg = {
        "family": cfg.get("family", "top_movers"),
        "param": cfg.get("param", 5.0),
        "z_hours": cfg.get("z_hours", 12.0),
        "duration_hours": cfg.get("duration_hours", 1.0),
    }

    try:
        up_zone, down_zone = _up_down_zones(feats, panel, metric=metric)
        shared_context: Optional[Dict[str, Any]] = None
        try:
            shared_context = _prepare_shared_mask_context(
                panel, feats, mode_cfg.get("price_up_tf", default_cfg)
            )
        except Exception as exc:
            tprint(
                "candidate_selector: shared timestamp mask context init failed; "
                f"falling back to per-mode preparation: {exc}"
            )
        m_up_tf = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_up_tf", default_cfg),
            prepared_context=shared_context,
        )
        m_up_mr = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_up_mr", default_cfg),
            prepared_context=shared_context,
        )
        m_down_tf = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_down_tf", default_cfg),
            prepared_context=shared_context,
        )
        m_down_mr = _build_mask_for_mode(
            panel,
            feats,
            mode_cfg.get("price_down_mr", default_cfg),
            prepared_context=shared_context,
        )

        long_mask = (up_zone & m_up_tf) | (down_zone & m_down_mr)
        short_mask = (up_zone & m_up_mr) | (down_zone & m_down_tf)

    except Exception as e:
        raise RuntimeError(
            f"Per-mode candidate mask generation at timestamp failed: {e}"
        ) from e

    if long_mask.empty and short_mask.empty:
        return [], []

    # Check if requested timestamp exists
    if ts not in long_mask.index:
        # Find nearest timestamp
        tprint(f"Timestamp {ts} not in mask, using nearest")
        ts = long_mask.index[np.abs(long_mask.index - ts).argmin()]

    long_candidates = long_mask.loc[ts]
    short_candidates = short_mask.loc[ts]
    long_candidates = long_candidates[long_candidates].index.tolist()
    short_candidates = short_candidates[short_candidates].index.tolist()
    if (
        not long_candidates
        and not short_candidates
        and bool(cfg.get("candidate_mask_empty_fallback_enabled", True))
    ):
        sliced_panel = {
            key: value.loc[:ts] if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
        sliced_feats = {
            key: value.loc[:ts] if isinstance(value, pd.DataFrame) else value
            for key, value in feats.items()
        }
        long_candidates, short_candidates = _fallback_rank_candidates(
            sliced_panel,
            sliced_feats,
            metric=metric,
            cfg=cfg,
        )
    if (
        not long_candidates
        and not short_candidates
        and bool(cfg.get("candidate_mask_rank_rescue_enabled", True))
    ):
        sliced_panel = {
            key: value.loc[:ts] if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
        sliced_feats = {
            key: value.loc[:ts] if isinstance(value, pd.DataFrame) else value
            for key, value in feats.items()
        }
        long_candidates, short_candidates = _rescue_rank_candidates(
            sliced_panel,
            sliced_feats,
            metric=metric,
            cfg=cfg,
        )

    return long_candidates, short_candidates


def filter_candidates_by_direction(
    candidates: List[str],
    panel: Dict[str, pd.DataFrame],
    side: str,
    lookback_hours: int = 24,
) -> List[str]:
    """Filter candidates based on price direction.

    Args:
        candidates: List of candidate symbols
        panel: Price panel
        side: "long" or "short"
        lookback_hours: Hours to look back for direction

    Returns:
        Filtered list of candidates
    """
    if not candidates:
        return []

    close = panel.get("close")
    # Safely check for empty - handle case where close might be a string or other type
    try:
        is_empty = (
            close is None
            or not isinstance(close, (pd.DataFrame, pd.Series))
            or (hasattr(close, "empty") and close.empty)
        )
    except Exception as e:
        tprint(f"Error checking close.empty: {e}, type: {type(close)}")
        is_empty = True

    if is_empty:
        return candidates

    filtered = []
    for sym in candidates:
        if sym not in close.columns:
            continue

        try:
            # Get recent prices
            recent_prices = close[sym].dropna()
            if len(recent_prices) < 2:
                continue

            # Calculate return over lookback period
            current_price = recent_prices.iloc[-1]
            past_price = recent_prices.iloc[-min(lookback_hours, len(recent_prices))]

            if past_price > 0:
                ret = (current_price / past_price) - 1

                if side == "long" and ret > 0:
                    filtered.append(sym)
                elif side == "short" and ret < 0:
                    filtered.append(sym)
        except (KeyError, IndexError, ZeroDivisionError):
            continue

    return filtered
