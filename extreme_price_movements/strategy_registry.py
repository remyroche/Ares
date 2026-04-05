from __future__ import annotations

from typing import Any


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
        is_mr = "mr" in sid or trigger.endswith("_mr")
        is_tf = "tf" in sid or trigger.endswith("_tf")

        feat_keys = s.get("feature_keys")
        if not feat_keys:
            feat_keys = cfg_dict.get("mr_feature_keys", []) if is_mr else cfg_dict.get("tf_feature_keys", [])

        meta_keys = s.get("meta_feature_keys")
        if not meta_keys:
            base = list(cfg_dict.get("meta_feature_keys", []))
            extra = list(cfg_dict.get("mr_meta_feature_keys", [])) if is_mr else list(cfg_dict.get("tf_meta_feature_keys", []))
            meta_keys = list(dict.fromkeys(base + extra)) # unique preserve order

        return {
            **s,
            "is_mr": is_mr,
            "is_tf": is_tf,
            "feature_keys": feat_keys,
            "meta_feature_keys": meta_keys,
        }

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

        out.append(_enrich_legacy(parsed_s))

    if not out:
        return [_enrich_legacy(dict(s)) for s in _LEGACY_STRATEGIES]
    return out


def strategy_map(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    return {s["strategy_id"]: s for s in get_strategies(cfg)}

