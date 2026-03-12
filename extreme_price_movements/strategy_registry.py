from __future__ import annotations

from typing import Any


_LEGACY_STRATEGIES: tuple[dict[str, Any], ...] = (
    {
        "strategy_id": "long_tf",
        "trade_side": "long",
        "base_event_trigger": "price_up_tf",
        "regime_filters": [],
    },
    {
        "strategy_id": "long_mr",
        "trade_side": "long",
        "base_event_trigger": "price_down_mr",
        "regime_filters": [],
    },
    {
        "strategy_id": "short_tf",
        "trade_side": "short",
        "base_event_trigger": "price_down_tf",
        "regime_filters": [],
    },
    {
        "strategy_id": "short_mr",
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
    """
    raw = (cfg or {}).get("strategies")
    if not isinstance(raw, list) or not raw:
        return [dict(s) for s in _LEGACY_STRATEGIES]

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
        out.append(
            {
                "strategy_id": strategy_id,
                "trade_side": "short" if trade_side == "short" else "long",
                "base_event_trigger": base_event_trigger,
                "regime_filters": norm_filters,
            }
        )

    return out or [dict(s) for s in _LEGACY_STRATEGIES]


def strategy_map(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return {s["strategy_id"]: s for s in get_strategies(cfg)}

