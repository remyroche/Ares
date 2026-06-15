from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint


RULE_MASK_FEATURE_PREFIX = "lgbm_rule_mask_"


@dataclass(frozen=True)
class RuleMaskSpec:
    feature_name: str
    canonical_key: str
    side: str
    source_horizon: int
    source_target: str
    row_index: int


def _truthy(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def rule_mask_features_enabled(cfg: Mapping[str, Any] | None = None) -> bool:
    cfg = cfg or {}
    raw = os.environ.get(
        "EPM_LGBM_RULE_MASK_FEATURES_ENABLED",
        cfg.get("lgbm_rule_mask_features_enabled", False),
    )
    return _truthy(raw)


def is_rule_mask_feature_name(name: Any) -> bool:
    return str(name).startswith(RULE_MASK_FEATURE_PREFIX)


def _default_rule_csv(cfg: Mapping[str, Any] | None = None) -> Path | None:
    cfg = cfg or {}
    explicit = str(
        os.environ.get("EPM_LGBM_RULE_MASK_FEATURES_CSV")
        or cfg.get("lgbm_rule_mask_features_csv")
        or ""
    ).strip()
    if explicit:
        return Path(explicit).expanduser()

    data_root = Path(str(cfg.get("data_root", "data_perp")))
    feature_run = str(
        os.environ.get("EPM_FEATURE_SOURCE_RUN_ID")
        or cfg.get("feature_source_run_id")
        or "20260523_015947"
    ).strip()
    root = data_root / "artifacts" / feature_run
    candidates = sorted(
        root.glob("lgbm_based_mask_generation_v2*/run_*/diversified_final_selection.csv")
    )
    if not candidates:
        candidates = sorted(root.glob("**/diversified_final_selection.csv"))
    return candidates[-1] if candidates else None


def _stable_feature_name(row_index: int, row: Mapping[str, Any]) -> str:
    canonical_key = str(row.get("canonical_key") or "")
    side = str(row.get("side") or "any").strip().lower() or "any"
    horizon_raw = row.get("source_horizon", "")
    target = str(row.get("source_target") or "target").strip().lower() or "target"
    try:
        horizon = int(float(horizon_raw))
    except Exception:
        horizon = -1
    digest_src = f"{row_index}|{side}|{horizon}|{target}|{canonical_key}".encode(
        "utf-8", errors="ignore"
    )
    digest = hashlib.blake2s(digest_src, digest_size=5).hexdigest()
    h_part = f"h{horizon}" if horizon >= 0 else "hany"
    side_part = side if side in {"long", "short"} else "any"
    return f"{RULE_MASK_FEATURE_PREFIX}{row_index:03d}_{side_part}_{h_part}_{digest}"


@lru_cache(maxsize=16)
def _load_rule_mask_specs_cached(
    csv_path_str: str,
    max_rules: int,
    side_filter: str,
) -> tuple[tuple[RuleMaskSpec, ...], tuple[str, ...]]:
    path = Path(csv_path_str)
    if not path.exists():
        raise FileNotFoundError(f"rule mask CSV not found: {path}")
    df = pd.read_csv(path)
    if "canonical_key" not in df.columns:
        raise ValueError(f"rule mask CSV has no canonical_key column: {path}")
    df = df.copy()
    df["canonical_key"] = df["canonical_key"].astype(str)
    df = df[df["canonical_key"].str.len() > 0]
    if side_filter in {"long", "short"} and "side" in df.columns:
        df = df[df["side"].astype(str).str.lower() == side_filter]
    df = df.drop_duplicates(subset=["canonical_key"], keep="first").reset_index(drop=True)
    if max_rules > 0:
        df = df.head(max_rules)

    try:
        from extreme_price_movements.lgbm_based_mask_generation_v2 import (
            extract_feature_names_from_key,
        )
    except Exception:
        from extreme_price_movements.lgbm_based_mask_generation import (  # type: ignore
            extract_feature_names_from_key,
        )

    specs: list[RuleMaskSpec] = []
    source_keys: set[str] = set()
    for i, row in df.iterrows():
        rec = row.to_dict()
        canonical_key = str(rec.get("canonical_key") or "")
        if not canonical_key:
            continue
        try:
            extracted_features = extract_feature_names_from_key(canonical_key)
        except Exception as exc:
            tprint(
                "Rule-mask feature registry: keeping rule but skipping source-key "
                f"preload for malformed canonical_key row={int(i)}: {exc}"
            )
            extracted_features = []
        for feat in extracted_features:
            if str(feat).strip():
                source_keys.add(str(feat).strip())
        try:
            horizon = int(float(rec.get("source_horizon", -1)))
        except Exception:
            horizon = -1
        specs.append(
            RuleMaskSpec(
                feature_name=_stable_feature_name(int(i), rec),
                canonical_key=canonical_key,
                side=str(rec.get("side") or "").strip().lower(),
                source_horizon=horizon,
                source_target=str(rec.get("source_target") or "").strip(),
                row_index=int(i),
            )
        )
    return tuple(specs), tuple(sorted(source_keys))


def load_rule_mask_specs(
    cfg: Mapping[str, Any] | None = None,
    *,
    side: str | None = None,
) -> tuple[list[RuleMaskSpec], list[str], Path | None]:
    cfg = cfg or {}
    path = _default_rule_csv(cfg)
    if path is None:
        return [], [], None
    max_rules = int(
        float(
            os.environ.get(
                "EPM_LGBM_RULE_MASK_FEATURES_MAX_RULES",
                cfg.get("lgbm_rule_mask_features_max_rules", 0),
            )
            or 0
        )
    )
    side_filter = ""
    if _truthy(
        os.environ.get(
            "EPM_LGBM_RULE_MASK_FEATURES_SIDE_FILTER",
            cfg.get("lgbm_rule_mask_features_side_filter", False),
        )
    ):
        side_filter = str(side or "").strip().lower()
    specs, source_keys = _load_rule_mask_specs_cached(
        str(path.resolve()), max_rules, side_filter
    )
    return list(specs), list(source_keys), path


def rule_mask_feature_source_keys(cfg: Mapping[str, Any] | None = None) -> list[str]:
    if not rule_mask_features_enabled(cfg):
        return []
    try:
        _, source_keys, _ = load_rule_mask_specs(cfg)
        return source_keys
    except Exception as exc:
        tprint(f"Rule-mask feature sources unavailable: {exc}")
        return []


def rule_mask_feature_names(cfg: Mapping[str, Any] | None = None) -> list[str]:
    if not rule_mask_features_enabled(cfg):
        return []
    try:
        specs, _, _ = load_rule_mask_specs(cfg)
        return [s.feature_name for s in specs]
    except Exception as exc:
        tprint(f"Rule-mask feature names unavailable: {exc}")
        return []


def append_rule_mask_features(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any] | None = None,
    *,
    side: str | None = None,
    context: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not rule_mask_features_enabled(cfg):
        return frame, {"enabled": False, "reason": "disabled"}
    specs, source_keys, path = load_rule_mask_specs(cfg, side=side)
    diag: dict[str, Any] = {
        "enabled": True,
        "csv": str(path) if path is not None else "",
        "context": str(context),
        "n_rules": int(len(specs)),
        "n_source_keys": int(len(source_keys)),
        "feature_names": [s.feature_name for s in specs],
    }
    if frame is None or frame.empty or not specs:
        diag["reason"] = "empty_frame_or_no_rules"
        return frame, diag

    available = [k for k in source_keys if k in frame.columns]
    missing = [k for k in source_keys if k not in frame.columns]
    diag["available_source_keys"] = int(len(available))
    diag["missing_source_keys"] = int(len(missing))
    if missing:
        diag["missing_source_key_sample"] = missing[:20]
    if not available:
        diag["reason"] = "no_source_features_available"
        tprint(
            f"Rule-mask features [{context}]: no source columns available "
            f"for {len(specs)} rules from {path}."
        )
        return frame, diag

    try:
        from extreme_price_movements.lgbm_based_mask_generation_v2 import (
            CanonicalRuleMaskResolver,
            FeatureMetadata,
        )
    except Exception:
        from extreme_price_movements.lgbm_based_mask_generation import (  # type: ignore
            CanonicalRuleMaskResolver,
            FeatureMetadata,
        )

    source_frame = frame.reindex(columns=source_keys)
    values = source_frame.apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32, copy=True
    )
    metadata = [
        FeatureMetadata(
            feature_name=str(name),
            feature_index=int(i),
            group="trigger",
            source_name=str(name),
            source_family="lgbm_rule_mask_source",
            source_type="continuous",
            description="Source feature for diversified LGBM rule-mask input",
        )
        for i, name in enumerate(source_keys)
    ]
    resolver = CanonicalRuleMaskResolver(values, metadata)
    valid_specs: list[RuleMaskSpec] = []
    mask_rows: list[np.ndarray] = []
    invalid_rules: list[dict[str, Any]] = []
    for spec in specs:
        try:
            mask_matrix = resolver.get_masks_matrix([spec.canonical_key])
            if mask_matrix.shape != (1, len(frame)):
                raise RuntimeError(
                    f"rule-mask matrix shape mismatch: {mask_matrix.shape}, expected "
                    f"{(1, len(frame))}"
                )
        except Exception as exc:
            invalid_rules.append(
                {
                    "row_index": int(spec.row_index),
                    "feature_name": spec.feature_name,
                    "canonical_key": spec.canonical_key,
                    "error": str(exc)[:240],
                }
            )
            continue
        valid_specs.append(spec)
        mask_rows.append(mask_matrix[0].astype(np.float32, copy=False))

    diag["invalid_rule_count"] = int(len(invalid_rules))
    if invalid_rules:
        diag["invalid_rule_sample"] = invalid_rules[:10]
    diag["added_feature_names"] = [s.feature_name for s in valid_specs]
    if not valid_specs:
        diag["reason"] = "no_valid_rule_masks"
        tprint(
            f"Rule-mask features [{context}]: no valid rule masks from {path}; "
            f"invalid_rules={len(invalid_rules)}."
        )
        return frame, diag

    add_df = pd.DataFrame(
        {
            spec.feature_name: mask_rows[i]
            for i, spec in enumerate(valid_specs)
        },
        index=frame.index,
    )
    out = pd.concat([frame, add_df], axis=1, copy=False)
    supports = add_df.mean(axis=0).astype(float).to_dict()
    diag["support_mean"] = float(np.nanmean(list(supports.values()))) if supports else 0.0
    diag["support_min"] = float(np.nanmin(list(supports.values()))) if supports else 0.0
    diag["support_max"] = float(np.nanmax(list(supports.values()))) if supports else 0.0
    diag["unresolved_feature_count"] = int(getattr(resolver, "unresolved_feature_count", 0))
    diag["unresolved_feature_sample"] = sorted(
        list(getattr(resolver, "unresolved_feature_names", set()))
    )[:20]
    tprint(
        f"Rule-mask features [{context}]: added {len(valid_specs)}/{len(specs)} columns "
        f"from {path}; source_available={len(available)}/{len(source_keys)} "
        f"support_mean={diag['support_mean']:.4f} invalid_rules={len(invalid_rules)}."
    )
    return out, diag
