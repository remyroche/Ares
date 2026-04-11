from __future__ import annotations

from typing import Any

from extreme_price_movements.training_utils import (
    dedupe_keep_order,
    get_base_feature_keys,
    get_meta_feature_keys,
)


_LEGACY_STRATEGIES: tuple[dict[str, Any], ...] = (
    {
        "strategy_id": "(price_up_tf==1)|(*)|(*)",  # Full canonical 3-slot key
        "trade_side": "long",
        "base_event_trigger": "price_up_tf",
        "regime_filters": [],
    },
    {
        "strategy_id": "(price_down_mr==1)|(*)|(*)",  # Full canonical 3-slot key
        "trade_side": "long",
        "base_event_trigger": "price_down_mr",
        "regime_filters": [],
    },
    {
        "strategy_id": "(price_down_tf==1)|(*)|(*)",  # Full canonical 3-slot key
        "trade_side": "short",
        "base_event_trigger": "price_down_tf",
        "regime_filters": [],
    },
    {
        "strategy_id": "(price_up_mr==1)|(*)|(*)",  # Full canonical 3-slot key
        "trade_side": "short",
        "base_event_trigger": "price_up_mr",
        "regime_filters": [],
    },
)


def get_strategies(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return normalized strategy definitions.

    Strategy schema:
      - strategy_id: unique key for artifact/model names
      - trade_side: long|short
      - base_event_trigger: candidate mask mode key
      - regime_filters: optional list of additional mask mode keys to AND
      - feature_keys: list of feature keys for the alpha model
      - meta_feature_keys: list of feature keys for the meta model
      - is_mr: boolean indicating if this is a mean-reversion strategy
      - is_tf: boolean indicating if this is a trend-following strategy
    """
    raw = (cfg or {}).get("strategies")

    def _enrich_legacy(s: dict[str, Any]) -> dict[str, Any]:
        cfg_dict = cfg or {}
        sid = s["strategy_id"].lower()
        trigger = s["base_event_trigger"].lower()
        trade_side = str(s.get("trade_side", "long")).lower()
        is_mr_raw = s.get("is_mr")
        is_tf_raw = s.get("is_tf")
        is_mr = (
            bool(is_mr_raw)
            if isinstance(is_mr_raw, bool)
            else ("mr" in sid or trigger.endswith("_mr"))
        )
        is_tf = (
            bool(is_tf_raw)
            if isinstance(is_tf_raw, bool)
            else ("tf" in sid or trigger.endswith("_tf"))
        )

        feat_keys = s.get("feature_keys")
        if not feat_keys:
            feat_keys = get_base_feature_keys(
                "short" if trade_side == "short" else "long", cfg_dict
            )

        out = {
            **s,
            "is_mr": is_mr,
            "is_tf": is_tf,
            "feature_keys": feat_keys,
        }
        meta_keys = s.get("meta_feature_keys")
        if meta_keys:
            out["meta_feature_keys"] = meta_keys
        return out

    if not isinstance(raw, list) or not raw:
        return [_enrich_legacy(dict(s)) for s in _LEGACY_STRATEGIES]

    out: list[dict[str, Any]] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            continue
        strategy_id = str(
            s.get("strategy_id")
            or s.get("name")
            or s.get("id")
            or f"strategy_{i}"
        )
        trade_side = str(s.get("trade_side", "long")).lower()
        base_event_trigger = str(
            s.get("base_event_trigger") or s.get("mode") or ""
        ).strip()
        if not base_event_trigger:
            continue
        filters = s.get("regime_filters") or s.get("filters") or []
        norm_filters = [str(v) for v in filters if isinstance(v, str) and v]

        parsed_s = {
            "strategy_id": strategy_id,
            "trade_side": "short" if trade_side == "short" else "long",
            "base_event_trigger": base_event_trigger,
            "regime_filters": norm_filters,
        }
        if "move_bucket" in s:
            parsed_s["move_bucket"] = s["move_bucket"]
        if "candidate_bucket" in s:
            parsed_s["candidate_bucket"] = s["candidate_bucket"]
        if "feature_keys" in s:
            parsed_s["feature_keys"] = s["feature_keys"]
        if "meta_feature_keys" in s:
            parsed_s["meta_feature_keys"] = s["meta_feature_keys"]
        if "is_mr" in s:
            parsed_s["is_mr"] = s["is_mr"]
        if "is_tf" in s:
            parsed_s["is_tf"] = s["is_tf"]
        if "source_target" in s:
            parsed_s["source_target"] = s["source_target"]
        if "source_horizon" in s:
            parsed_s["source_horizon"] = s["source_horizon"]

        out.append(_enrich_legacy(parsed_s))

    if not out:
        return [_enrich_legacy(dict(s)) for s in _LEGACY_STRATEGIES]
    return out


def strategy_map(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return {s["strategy_id"]: s for s in get_strategies(cfg)}


def normalize_strategy_horizon(horizon: Any) -> int:
    """Normalize runtime horizons to the TBM geometry support contract.

    Dynamic strategies may come from discovery with horizons like H3. The
    simple/generated TBM geometry contract only supports H5 as the shortest
    horizon, so any horizon below 5 is promoted to H5.
    """
    try:
        h_int = int(horizon)
    except Exception:
        h_int = 0
    if h_int <= 0:
        return 5
    return 5 if h_int < 5 else h_int


def strategy_runtime_horizons(
    strategy: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    requested_horizons: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Return the runtime horizons for a strategy.

    Priority:
    1. explicit requested horizons from the CLI/caller
    2. strategy-level source_horizon
    3. config label_horizons_hours
    """
    if requested_horizons:
        vals = [normalize_strategy_horizon(h) for h in requested_horizons]
    elif strategy.get("source_horizon") is not None:
        vals = [normalize_strategy_horizon(strategy.get("source_horizon"))]
    else:
        cfg_horizons = (cfg or {}).get("label_horizons_hours", [])
        vals = [normalize_strategy_horizon(h) for h in cfg_horizons]
    out: list[int] = []
    seen: set[int] = set()
    for h in vals:
        if h in seen:
            continue
        seen.add(h)
        out.append(int(h))
    return out
