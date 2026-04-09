import pandas as pd

from typing import List, Dict, Any

def dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def expand_feature_group_refs(keys: List[str], cfg: Dict[str, Any], visited=None) -> List[str]:
    if visited is None:
        visited = set()

    expanded = []
    for k in keys:
        if k in visited:
            continue
        visited.add(k)

        # Support dict-like or object-like config access
        if isinstance(cfg, dict):
            val = cfg.get(k)
            if val is None:
                # Check module level imports just in case
                import extreme_price_movements.config as cfg_mod
                val = getattr(cfg_mod, k, None)
        else:
            val = getattr(cfg, k, None)

        if isinstance(val, list):
            expanded.extend(expand_feature_group_refs(val, cfg, visited.copy()))
        else:
            expanded.append(k)
    return expanded

def get_base_feature_keys(side: str, cfg: Dict[str, Any]) -> List[str]:
    if side not in {"long", "short"}:
        raise ValueError(f"Invalid side {side}")

    shared = expand_feature_group_refs(cfg.get("base_shared_feature_keys", []), cfg)
    if side == "long":
        specific = expand_feature_group_refs(cfg.get("base_long_feature_keys", []), cfg)
    else:
        specific = expand_feature_group_refs(cfg.get("base_short_feature_keys", []), cfg)

    return dedupe_keep_order(shared + specific)

def get_meta_feature_keys(head: str, cfg: Dict[str, Any]) -> List[str]:
    if head not in {"reg", "clf", "mfe", "mae", "asym"}:
        raise ValueError(f"Invalid head {head}")

    shared = expand_feature_group_refs(cfg.get("meta_shared_feature_keys", []), cfg)

    if head == "reg":
        specific = expand_feature_group_refs(cfg.get("meta_reg_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "clf":
        specific = expand_feature_group_refs(cfg.get("meta_clf_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "mfe":
        specific = expand_feature_group_refs(cfg.get("meta_mfe_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "mae":
        specific = expand_feature_group_refs(cfg.get("meta_mae_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "asym":
        mfe_specific = expand_feature_group_refs(cfg.get("meta_mfe_feature_keys", []), cfg)
        mae_specific = expand_feature_group_refs(cfg.get("meta_mae_feature_keys", []), cfg)
        return dedupe_keep_order(shared + mfe_specific + mae_specific)

def validate_feature_keys_exist(df, keys: List[str], context: str) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature keys for {context}: {missing}")

def compute_unused_features(all_feature_columns: List[str], configured_feature_keys: List[str]) -> List[str]:
    configured = set(configured_feature_keys)
    unused = [c for c in all_feature_columns if c not in configured]
    return sorted(unused)

def audit_feature_coverage(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    exclude_prefixes = [
        "__y", "__w", "id", "timestamp", "symbol",
        "fold_", "oof_", "p_", "pred_", "is_up"
    ]

    all_cols = []
    for c in df.columns:
        if c in {"id", "timestamp", "symbol"}:
            continue
        if any(c.startswith(p) for p in exclude_prefixes) and not c.startswith("p_vol_high") and not c.startswith("p_cusum_high") and not c.startswith("p_liq_low"):
            continue
        all_cols.append(c)

    base_long = set(get_base_feature_keys("long", cfg))
    base_short = set(get_base_feature_keys("short", cfg))
    base_all = base_long.union(base_short)

    meta_reg = set(get_meta_feature_keys("reg", cfg))
    meta_clf = set(get_meta_feature_keys("clf", cfg))
    meta_mfe = set(get_meta_feature_keys("mfe", cfg))
    meta_mae = set(get_meta_feature_keys("mae", cfg))
    meta_asym = set(get_meta_feature_keys("asym", cfg))
    meta_all = meta_reg.union(meta_clf).union(meta_mfe).union(meta_mae).union(meta_asym)

    global_all = base_all.union(meta_all)

    base_unused = sorted(list(set(all_cols) - base_all))
    meta_unused = sorted(list(set(all_cols) - meta_all))
    global_unused = sorted(list(set(all_cols) - global_all))

    stale_orphans = []
    for f in global_unused:
        if (
            f.startswith("tf_")
            or f.startswith("mr_")
            or "legacy" in f.lower()
            or f in {"tf_feature_keys", "mr_feature_keys", "meta_feature_keys"}
        ):
            stale_orphans.append(f)

    return {
        "base_unused": base_unused,
        "meta_unused": meta_unused,
        "global_unused": global_unused,
        "stale_orphans": stale_orphans
    }
