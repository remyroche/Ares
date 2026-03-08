"""Inference package with lazy exports to avoid importing live dependencies eagerly."""

from importlib import import_module

__all__ = [
    "load_inference_config",
    "get_candidate_thresholds",
    "make_exchange",
    "fetch_ohlcv_for_symbols",
    "select_candidates",
    "generate_features",
    "ModelOrchestrator",
    "run_inference_chain",
    "TradeExecutor",
    "execute_live_trade",
    "record_shadow_trade",
    "TradeLogger",
    "log_trade_decision",
]

_EXPORTS = {
    "load_inference_config": ("extreme_price_movements.inference.config", "load_inference_config"),
    "get_candidate_thresholds": ("extreme_price_movements.inference.config", "get_candidate_thresholds"),
    "make_exchange": ("extreme_price_movements.inference.data_fetcher", "make_exchange"),
    "fetch_ohlcv_for_symbols": ("extreme_price_movements.inference.data_fetcher", "fetch_ohlcv_for_symbols"),
    "select_candidates": ("extreme_price_movements.inference.candidate_selector", "select_candidates"),
    "generate_features": ("extreme_price_movements.inference.feature_generator", "generate_features"),
    "ModelOrchestrator": ("extreme_price_movements.inference.model_orchestrator", "ModelOrchestrator"),
    "run_inference_chain": ("extreme_price_movements.inference.model_orchestrator", "run_inference_chain"),
    "TradeExecutor": ("extreme_price_movements.inference.trade_executor", "TradeExecutor"),
    "execute_live_trade": ("extreme_price_movements.inference.trade_executor", "execute_live_trade"),
    "record_shadow_trade": ("extreme_price_movements.inference.trade_executor", "record_shadow_trade"),
    "TradeLogger": ("extreme_price_movements.inference.trade_logger", "TradeLogger"),
    "log_trade_decision": ("extreme_price_movements.inference.trade_logger", "log_trade_decision"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
