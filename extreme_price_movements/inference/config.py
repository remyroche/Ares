"""
Inference Configuration Loader.

This module loads all configuration needed for inference:
- Best parameters from offline optimizer (candidate thresholds, TBM params, etc.)
- Model paths using find_latest_run_id and load_full_state
- Other config parameters
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.inference.symbol_mapping import (
    convert_symbol_quote,
    normalise_symbol,
    symbol_bases,
)
from extreme_price_movements.model_loader import (
    find_latest_run_id,
    load_full_state,
    load_model_bundle,
)
from extreme_price_movements.offline_optimisers.params_store import (
    INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV,
    apply_offline_optimizer_best_params,
)
from extreme_price_movements.utils import tprint


def _resolve_runtime_cfg() -> Dict[str, Any]:
    """Refresh runtime config from persisted offline optimiser outputs."""
    return apply_offline_optimizer_best_params(dict(CFG))


def _load_inference_candidate_mask_params() -> Dict[str, Any]:
    path = Path(INFERENCE_CANDIDATE_MASK_BEST_PARAMS_CSV)
    if not path.exists():
        return {}
    try:
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        return rows[-1]
    except Exception:
        return {}


# Default paths
DEFAULT_DATA_ROOT = "data"
DEFAULT_EXECUTION_ACCOUNT = "margin"
DEFAULT_MARGIN_MODE = "cross"
DEFAULT_LIVE_QUOTE_CURRENCY = "USDC"


def get_candidate_thresholds(thresholds_csv: Optional[str] = None) -> Dict[str, float]:
    """Load candidate thresholds from runtime config (populated by offline optimizer).

    Args:
        thresholds_csv: Deprecated parameter, kept for backward compatibility.

    Returns:
        Dictionary with threshold parameters:
        - extreme_pct: Percentage of top/bottom performers to consider
        - min_range_pct: Minimum 12h high/low range percentage
        - min_vol_zscore: Minimum volatility z-score threshold
    """
    runtime_cfg = _resolve_runtime_cfg()
    thresholds = {
        "extreme_pct": runtime_cfg.get("train_extreme_pct_hourly", 0.05),
        "min_move_12h_pct": runtime_cfg.get("train_min_move_12h_pct", 0.06),
        "min_range_pct": runtime_cfg.get("train_min_range_pct", 0.06),
        "min_vol_zscore": runtime_cfg.get("train_min_vol_zscore", 1.5),
        "metric": runtime_cfg.get("train_candidate_metric", "ret12h"),
    }
    infer_mask = _load_inference_candidate_mask_params()
    if infer_mask:
        if infer_mask.get("train_extreme_pct_hourly"):
            thresholds["extreme_pct"] = float(infer_mask["train_extreme_pct_hourly"])
        if infer_mask.get("train_min_move_12h_pct"):
            thresholds["min_move_12h_pct"] = float(infer_mask["train_min_move_12h_pct"])
        if infer_mask.get("train_min_vol_zscore"):
            thresholds["min_vol_zscore"] = float(infer_mask["train_min_vol_zscore"])
        if infer_mask.get("train_candidate_metric"):
            thresholds["metric"] = str(infer_mask["train_candidate_metric"])
    return thresholds


def load_inference_config(
    data_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load complete inference configuration.

    Args:
        data_root: Root data directory. If None, uses "data"
        run_id: Specific run ID to load. If None, finds latest run

    Returns:
        Dictionary with all config needed for inference:
        - run_id: The run ID used
        - thresholds: Candidate threshold parameters
        - tbm_params: TBM barrier parameters from offline optimizer
        - model_bundle: Loaded model bundle
        - full_state: Complete training state
        - data_root: Data root path
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    # Find latest run ID if not provided
    if run_id is None:
        run_id = find_latest_run_id(data_root)
        if run_id is None:
            raise ValueError("No run ID found and none provided")

    tprint(f"Loading inference config for run_id: {run_id}")

    # Load thresholds from runtime_cfg (populated by offline optimizer)
    runtime_cfg = _resolve_runtime_cfg()
    thresholds = get_candidate_thresholds()
    tprint(f"Using thresholds: {thresholds}")

    # Load TBM params from runtime_cfg
    tbm_params = get_tbm_params()
    tprint(f"Using TBM params: {tbm_params}")

    # Load model bundle
    model_bundle = load_model_bundle(run_id, data_root)

    # Load full state
    full_state = load_full_state(run_id, data_root)

    config = {
        "run_id": run_id,
        "thresholds": thresholds,
        "tbm_params": tbm_params,
        "runtime_cfg": runtime_cfg,
        "model_bundle": model_bundle,
        "full_state": full_state,
        "data_root": data_root,
        "execution_account": DEFAULT_EXECUTION_ACCOUNT,
        "margin_mode": DEFAULT_MARGIN_MODE,
        "live_quote_currency": DEFAULT_LIVE_QUOTE_CURRENCY,
    }

    tprint(f"Inference config loaded successfully for run {run_id}")
    return config


def get_tbm_params() -> Dict[str, Any]:
    """Get TBM (Triple Barrier Model) parameters from runtime config.

    These parameters are populated by apply_offline_optimizer_best_params()
    and are the same parameters that were optimized during training.

    Returns:
        Dictionary with TBM barrier parameters:
        - barrier_k_tp: TP multiplier
        - barrier_sl_base_mult: SL as TP percentage
        - barrier_tp_lo: TP absolute lower bound
        - barrier_tp_hi: TP absolute upper bound
        - barrier_sl_lo: SL absolute lower bound
        - barrier_sl_hi: SL absolute upper bound
        - barrier_tp_base_pct: TP base percentage
        - barrier_tp_method: TP method
        - barrier_sl_method: SL method
        - barrier_atr_window: ATR window for barrier calculation
        - label_horizon_base: Base horizon for labels
        - label_horizon_scaling: Horizon scaling factor
        - barrier_mode: Barrier mode
    """
    # TBM parameters from runtime_cfg (populated by apply_offline_optimizer_best_params)
    tbm_keys = [
        "barrier_k_tp",
        "barrier_sl_base_mult",
        "barrier_tp_lo",
        "barrier_tp_hi",
        "barrier_sl_lo",
        "barrier_sl_hi",
        "barrier_tp_base_pct",
        "barrier_tp_abs_pct",
        "barrier_tp_method",
        "barrier_sl_method",
        "barrier_atr_window",
        "label_horizon_base",
        "label_horizon_scaling",
        "barrier_mode",
    ]

    runtime_cfg = _resolve_runtime_cfg()
    params = {}
    for key in tbm_keys:
        if key in runtime_cfg and runtime_cfg[key] is not None:
            params[key] = runtime_cfg[key]

    return params


def get_sample_weight_params() -> Dict[str, Any]:
    """Get sample weight parameters from runtime config.

    These parameters are populated by apply_offline_optimizer_best_params()
    and are the same parameters that were optimized during training.

    Returns:
        Dictionary with sample weight parameters:
        - sample_weight_component_alphas: Component alphas for sample weighting
        - sample_weight_component_alphas_base: Base component alphas
        - sample_weight_component_alphas_meta: Meta component alphas
        - sample_weight_vol_power: Volume power for sample weighting
        - sample_weight_distance_k: Distance parameter k
        - sample_weight_distance_min_dist: Minimum distance
        - sample_weight_recency_half_life_bars: Recency half-life in bars
    """
    # Sample weight parameters from runtime_cfg
    sample_weight_keys = [
        "sample_weight_component_alphas",
        "sample_weight_component_alphas_base",
        "sample_weight_component_alphas_meta",
        "sample_weight_vol_power",
        "sample_weight_distance_k",
        "sample_weight_distance_min_dist",
        "sample_weight_recency_half_life_bars",
    ]

    runtime_cfg = _resolve_runtime_cfg()
    params = {}
    for key in sample_weight_keys:
        if key in runtime_cfg and runtime_cfg[key] is not None:
            params[key] = runtime_cfg[key]

    return params


def get_runtime_cfg() -> Dict[str, Any]:
    """Get the full runtime config with all optimized parameters.

    Returns:
        Dictionary with all runtime config parameters including:
        - Candidate thresholds (extreme_pct, min_range_pct, min_vol_zscore)
        - TBM barrier parameters
        - Sample weight parameters
        - All other config from CFG
    """
    return _resolve_runtime_cfg()


def get_inference_defaults() -> Dict[str, Any]:
    """Get default inference parameters.

    Returns:
        Dictionary with default parameters for inference
    """
    return {
        # Data fetching
        "lookback_periods": 24 * 60,  # Number of 1h periods to look back (~2 months)
        "symbols_per_batch": 50,  # Symbols to fetch per batch
        # Feature generation
        "trend_sma_hours": 24 * 14,  # 14 days
        "gate_vol_lookback_hours": 24 * 7,  # 7 days
        "gate_trend_thr": 0.0,
        # Model inference
        "use_multi_horizon": True,
        # Execution
        "max_position_size": 0.1,  # 10% of capital
        "default_take_profit_pct": 0.15,  # 15%
    }


# Margin universe cache - lazy loaded and refreshed once per UTC day.
_MARGIN_UNIVERSE_CACHE = None
_MARGIN_UNIVERSE_CACHE_DAY = None
_MARGIN_UNIVERSE_CACHE_QUOTE = None


def _normalise_symbol(symbol: str) -> str:
    return normalise_symbol(symbol)


def _parse_market_listed_at(meta: Dict[str, Any]) -> Optional[int]:
    """Return exchange listing/onboarding timestamp in milliseconds when present."""
    info = meta.get("info", {}) if isinstance(meta, dict) else {}
    candidates = []
    if isinstance(meta, dict):
        candidates.extend(
            [
                meta.get("onboardDate"),
                meta.get("onboardTimestamp"),
                meta.get("listingTime"),
                meta.get("listedAt"),
                meta.get("created"),
                meta.get("createdAt"),
            ]
        )
    if isinstance(info, dict):
        candidates.extend(
            [
                info.get("onboardDate"),
                info.get("onboardTimestamp"),
                info.get("listingTime"),
                info.get("listedAt"),
                info.get("created"),
                info.get("createdAt"),
            ]
        )
    for value in candidates:
        if value in (None, ""):
            continue
        try:
            if isinstance(value, str) and not value.isdigit():
                parsed = pd.Timestamp(value)
                if parsed.tzinfo is None:
                    parsed = parsed.tz_localize("UTC")
                else:
                    parsed = parsed.tz_convert("UTC")
                return int(parsed.value // 10**6)
            numeric = float(value)
            if numeric < 10**11:
                numeric *= 1000.0
            return int(numeric)
        except Exception:
            continue
    return None


def _load_universe_from_exchange(
    exchange: Any,
    *,
    min_age_days: int = 14,
    quote_currency: str = "USDC",
) -> List[str]:
    """Build inference universe from exchange.load_markets() metadata.

    Filters to active quote-currency symbols and margin-capable markets. If listing age
    metadata exists, symbols younger than ``min_age_days`` are excluded.
    """
    markets = exchange.load_markets()
    selected: List[str] = []
    quote_currency = str(quote_currency or "USDC").upper()
    age_cutoff_ms = int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=int(min_age_days))).value // 10**6
    )
    skipped_young = 0
    for symbol, meta in (markets or {}).items():
        if not isinstance(meta, dict):
            continue
        quote = str(meta.get("quote") or "").upper()
        if quote != quote_currency:
            continue
        if not bool(meta.get("active", True)):
            continue
        info = meta.get("info", {}) if isinstance(meta.get("info", {}), dict) else {}
        margin_ok = bool(
            meta.get("margin", False) or info.get("isMarginTradingAllowed", False)
        )
        if not margin_ok:
            continue
        listed_at_ms = _parse_market_listed_at(meta)
        if listed_at_ms is not None and listed_at_ms > age_cutoff_ms:
            skipped_young += 1
            continue
        selected.append(_normalise_symbol(str(symbol)))
    if skipped_young:
        tprint(
            f"Exchange universe age filter removed {skipped_young} symbols "
            f"younger than {min_age_days}d"
        )
    return sorted(set(selected))


def get_margin_universe(
    exchange=None,
    *,
    force_refresh: bool = False,
    min_age_days: int = 14,
    quote_currency: str = "USDC",
) -> List[str]:
    """Get list of margin-enabled symbols from cache.

    Args:
        exchange: Optional exchange instance (ignored, kept for API compatibility)

    Returns:
        List of margin-enabled trading symbols
    """
    global _MARGIN_UNIVERSE_CACHE, _MARGIN_UNIVERSE_CACHE_DAY, _MARGIN_UNIVERSE_CACHE_QUOTE

    import json
    import os

    today = pd.Timestamp.utcnow().floor("D")
    cache_stale = (
        _MARGIN_UNIVERSE_CACHE_DAY is None or _MARGIN_UNIVERSE_CACHE_DAY < today
    )
    quote_currency = str(quote_currency or "USDC").upper()
    quote_changed = _MARGIN_UNIVERSE_CACHE_QUOTE != quote_currency
    if force_refresh or cache_stale or quote_changed or _MARGIN_UNIVERSE_CACHE is None:
        if exchange is not None and hasattr(exchange, "load_markets"):
            try:
                _MARGIN_UNIVERSE_CACHE = _load_universe_from_exchange(
                    exchange,
                    min_age_days=min_age_days,
                    quote_currency=quote_currency,
                )
                _MARGIN_UNIVERSE_CACHE_DAY = today
                _MARGIN_UNIVERSE_CACHE_QUOTE = quote_currency
                if _MARGIN_UNIVERSE_CACHE:
                    tprint(
                        f"Loaded {len(_MARGIN_UNIVERSE_CACHE)} margin-enabled symbols from exchange markets"
                    )
                    return _MARGIN_UNIVERSE_CACHE
            except Exception as exc:
                tprint(f"Exchange universe load failed, falling back to cache: {exc}")
        cache_path = os.path.join(
            os.path.dirname(__file__), "..", ".margin_universe_cache.json"
        )

        # Try multiple possible locations
        possible_paths = [
            cache_path,
            os.path.join(os.path.dirname(__file__), ".margin_universe_cache.json"),
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                ".margin_universe_cache.json",
            ),
            "/Users/remyroche/Documents/Ares/extreme_price_movements/.margin_universe_cache.json",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                cache_path = path
                break

        tprint(f"Loading margin universe from: {cache_path}")

        with open(cache_path, "r") as f:
            margin_data = json.load(f)

        # Extract symbols that have margin trading enabled for the requested quote.
        _MARGIN_UNIVERSE_CACHE = sorted(
            {
                _normalise_symbol(str(item["symbol"]))
                for item in margin_data
                if item.get("isMarginTradingAllowed", False)
                and _normalise_symbol(str(item.get("symbol", ""))).endswith(
                    f"/{quote_currency}"
                )
            }
        )
        _MARGIN_UNIVERSE_CACHE_DAY = today
        _MARGIN_UNIVERSE_CACHE_QUOTE = quote_currency

        tprint(f"Loaded {len(_MARGIN_UNIVERSE_CACHE)} margin-enabled symbols")

    return _MARGIN_UNIVERSE_CACHE


def _symbols_from_csv(path: Path) -> Set[str]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        symbols = {
            _normalise_symbol(row.get("symbol", ""))
            for row in reader
            if row.get("symbol")
        }
    symbols.discard("")
    return symbols


def _symbols_from_parquet(path: Path, column: str) -> Set[str]:
    frame = pd.read_parquet(path, columns=[column])
    symbols = {
        _normalise_symbol(str(sym)) for sym in frame[column].dropna().unique().tolist()
    }
    symbols.discard("")
    return symbols


def _label_manifest_symbol_sources(
    run_dir: Path,
    *,
    max_sources: int = 5,
) -> List[Tuple[str, Path, Set[str]]]:
    """Read symbol universes from label parquet files referenced by manifests."""
    out: List[Tuple[str, Path, Set[str]]] = []
    manifest_paths = [
        run_dir / "labels" / "labels_manifest.json",
        run_dir / "labels_backup_20260424" / "labels_manifest.json",
    ]
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text())
            datasets = payload.get("datasets", {})
            if not isinstance(datasets, dict):
                continue
            rows = [
                (str(name), meta)
                for name, meta in datasets.items()
                if isinstance(meta, dict) and meta.get("file")
            ]
            rows.sort(key=lambda item: int(item[1].get("rows", 0) or 0), reverse=True)
            for name, meta in rows[:max_sources]:
                columns = set(meta.get("columns", []) or [])
                symbol_col = "__symbol__" if "__symbol__" in columns else "symbol"
                if symbol_col not in columns:
                    continue
                label_path = manifest_path.parent / str(meta["file"])
                if not label_path.exists():
                    continue
                symbols = _symbols_from_parquet(label_path, symbol_col)
                if symbols:
                    out.append(
                        (f"{manifest_path.parent.name}:{name}", label_path, symbols)
                    )
        except Exception as exc:
            tprint(f"Warning: failed to inspect label manifest {manifest_path}: {exc}")
    return out


def load_trained_symbol_universe(data_root: str, run_id: str) -> Set[str]:
    """Load symbols covered by the training artifacts for the active run."""
    run_dir = Path(data_root) / "artifacts" / str(run_id)
    sources: List[Tuple[str, Path, Set[str]]] = []
    csv_candidates = [
        run_dir / "features" / "feature_health_symbol_summary.csv",
        run_dir / "feature_health_symbol_summary.csv",
    ]
    for path in csv_candidates:
        if not path.exists():
            continue
        try:
            symbols = _symbols_from_csv(path)
            if symbols:
                sources.append((path.name, path, symbols))
        except Exception as exc:
            tprint(f"Warning: failed to load trained symbol summary {path}: {exc}")

    parquet_candidates = [
        (run_dir / "baseline_events.parquet", "symbol"),
        (run_dir / "ohlc.parquet", "symbol"),
    ]
    for path, column in parquet_candidates:
        if not path.exists():
            continue
        try:
            symbols = _symbols_from_parquet(path, column)
            if symbols:
                sources.append((path.name, path, symbols))
        except Exception as exc:
            tprint(f"Warning: failed to load trained symbols from {path}: {exc}")

    sources.extend(_label_manifest_symbol_sources(run_dir))
    if not sources:
        return set()

    sources.sort(key=lambda item: len(item[2]), reverse=True)
    source_sizes = ", ".join(f"{label}={len(symbols)}" for label, _, symbols in sources)
    tprint(f"Trained symbol source sizes: {source_sizes}")

    label, path, symbols = sources[0]
    if len(symbols) < 25:
        tprint(
            f"Warning: trained symbol universe is very small ({len(symbols)} symbols); "
            "inspect training artifacts before deployment"
        )
    tprint(f"Loaded {len(symbols)} trained symbols from {path} ({label})")
    return symbols


def resolve_inference_universes(
    exchange: Any,
    *,
    data_root: str,
    run_id: str,
    explicit_symbols: Optional[List[str]] = None,
    live_quote_currency: str = "USDC",
) -> Dict[str, List[str]]:
    """Resolve Step-9 download and tradable universes.

    ``download_symbols`` is the daily Binance margin/live-quote universe.
    ``tradable_symbols`` is further restricted by base asset to symbols represented
    in training artifacts, so a model trained on ``BTC/USDT`` can trade
    ``BTC/USDC``.
    """
    live_quote_currency = str(live_quote_currency or "USDC").upper()
    if explicit_symbols:
        live_symbols = sorted(
            {
                convert_symbol_quote(_normalise_symbol(s), live_quote_currency)
                for s in explicit_symbols
            }
        )
    else:
        live_symbols = get_margin_universe(
            exchange, force_refresh=True, quote_currency=live_quote_currency
        )
    trained_symbols = load_trained_symbol_universe(data_root, run_id)
    if trained_symbols:
        trained_bases = symbol_bases(trained_symbols)
        tradable_symbols = sorted(
            sym
            for sym in set(live_symbols)
            if _normalise_symbol(sym).split("/")[0] in trained_bases
        )
        dropped = len(set(live_symbols) - set(tradable_symbols))
        tprint(
            f"Tradable universe restricted to trained symbols: "
            f"download={len(live_symbols)} tradable={len(tradable_symbols)} "
            f"dropped_untrained={dropped} live_quote={live_quote_currency}"
        )
    else:
        tradable_symbols = live_symbols
        tprint(
            "Warning: no trained symbol universe artifact found; "
            "using live universe for tradability"
        )
    return {
        "download_symbols": live_symbols,
        "tradable_symbols": tradable_symbols,
        "trained_symbols": sorted(trained_symbols),
        "live_quote_currency": live_quote_currency,
    }
