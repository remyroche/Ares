"""Causal soft bad-regime archetype features.

The functions here turn a compact archetype definition artifact into
timestamp/row-level scores.  They are deliberately deterministic and use only
trailing robust baselines so the same transform can be cross-fitted during
training and replayed at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


GLOBAL_PREFIXES = (
    "mkt_",
    "xs_",
    "q_",
    "eig_",
    "xs_cov_",
    "pct_assets_",
    "cs_",
    "xasset_",
    "regime_",
)


@dataclass(frozen=True)
class BadRegimeArchetypeFeatureConfig:
    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    trailing_window: int = 24 * 28
    min_periods: int = 24 * 7
    min_resolved_features: int = 2
    clip_z: float = 6.0
    archetype_prefix: str = "badregime__"
    include_deployable_aliases: bool = True
    include_ranked_probability_aliases: bool = True
    eps: float = 1e-8


def load_bad_regime_archetype_definitions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a ``soft_archetype_definitions.json`` style artifact."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            out[str(key)] = dict(value)
    return out


def _safe_name(value: str) -> str:
    out = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return out or "unknown"


def _feature_aliases(name: str) -> list[str]:
    raw = str(name)
    aliases = [raw]
    for prefix in ("export__", "pred_H5_", "base_H5_", "oof_"):
        if raw.startswith(prefix):
            aliases.append(raw[len(prefix) :])
    if "_H5_" in raw:
        aliases.append(raw.split("_H5_", 1)[1])
    if raw.startswith("pred_") and "_" in raw:
        aliases.append(raw.rsplit("_", 1)[-1])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _resolve_feature(frame: pd.DataFrame, name: str) -> str | None:
    for alias in _feature_aliases(str(name)):
        if alias in frame.columns:
            return alias
    return None


def _is_global_feature(name: str) -> bool:
    lower = str(name).lower()
    return lower.startswith(GLOBAL_PREFIXES) or "__" in lower and lower.split("__", 1)[0] in {"q", "eig", "xs", "xs_cov"}


def _trailing_robust_z_series(
    values: pd.Series,
    *,
    window: int,
    min_periods: int,
    clip_z: float,
    eps: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    roll = numeric.rolling(window=window, min_periods=min_periods)
    center = roll.median().shift(1)
    q25 = roll.quantile(0.25).shift(1)
    q75 = roll.quantile(0.75).shift(1)
    scale = ((q75 - q25) / 1.349).replace(0.0, np.nan)
    z = (numeric - center) / scale.clip(lower=eps)
    return z.clip(lower=-float(clip_z), upper=float(clip_z))


def _global_trailing_z(
    frame: pd.DataFrame,
    column: str,
    *,
    config: BadRegimeArchetypeFeatureConfig,
) -> pd.Series:
    ts = pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce")
    by_time = pd.DataFrame({"timestamp": ts, "value": pd.to_numeric(frame[column], errors="coerce")})
    by_time = by_time.dropna(subset=["timestamp"]).groupby("timestamp", sort=True)["value"].median()
    z_time = _trailing_robust_z_series(
        by_time,
        window=int(config.trailing_window),
        min_periods=int(config.min_periods),
        clip_z=float(config.clip_z),
        eps=float(config.eps),
    )
    mapped = ts.map(z_time)
    return pd.Series(mapped.to_numpy(dtype=np.float32, copy=False), index=frame.index)


def _asset_trailing_z(
    frame: pd.DataFrame,
    column: str,
    *,
    config: BadRegimeArchetypeFeatureConfig,
) -> pd.Series:
    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce"),
            "symbol": frame[config.symbol_col].astype(str) if config.symbol_col in frame.columns else "",
            "value": pd.to_numeric(frame[column], errors="coerce"),
            "__row_pos": np.arange(len(frame), dtype=np.int64),
        }
    ).sort_values(["symbol", "timestamp", "__row_pos"], kind="mergesort")
    out = np.full(len(work), np.nan, dtype=np.float32)
    for _symbol, group in work.groupby("symbol", sort=False):
        z = _trailing_robust_z_series(
            group["value"],
            window=int(config.trailing_window),
            min_periods=int(config.min_periods),
            clip_z=float(config.clip_z),
            eps=float(config.eps),
        )
        out[group["__row_pos"].to_numpy(dtype=np.int64, copy=False)] = z.to_numpy(dtype=np.float32, copy=False)
    result = pd.Series(out, index=frame.index)
    return result


def _trailing_z(
    frame: pd.DataFrame,
    column: str,
    original_name: str,
    *,
    config: BadRegimeArchetypeFeatureConfig,
) -> pd.Series:
    if _is_global_feature(original_name) or config.symbol_col not in frame.columns:
        return _global_trailing_z(frame, column, config=config)
    return _asset_trailing_z(frame, column, config=config)


def _ranked_archetypes(definitions: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    return sorted(
        ((str(name), dict(payload)) for name, payload in definitions.items()),
        key=lambda item: float(item[1].get("evidence_score", 0.0) or 0.0),
        reverse=True,
    )


def build_bad_regime_archetype_feature_frame(
    frame: pd.DataFrame,
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    config: BadRegimeArchetypeFeatureConfig = BadRegimeArchetypeFeatureConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build continuous causal bad-regime archetype features.

    Scores are based on trailing robust-z intensity of each archetype's
    available top features.  They are not fit on future rows and do not use
    labels.  Cross-fitting is achieved by calling this transform separately on
    each train/validation fold with fold-appropriate input history.
    """

    if frame.empty or not definitions or config.timestamp_col not in frame.columns:
        return pd.DataFrame(index=frame.index), {
            "status": "empty_input",
            "output_feature_count": 0,
        }

    data: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {
        "status": "completed",
        "input_rows": int(len(frame)),
        "trailing_window": int(config.trailing_window),
        "min_periods": int(config.min_periods),
        "archetypes": {},
        "train_inference_parity_surface": "causal_trailing_robust_z_from_observable_features",
    }
    archetype_scores: list[np.ndarray] = []
    archetype_supports: list[np.ndarray] = []
    ranked = _ranked_archetypes(definitions)
    for rank, (archetype_name, payload) in enumerate(ranked, start=1):
        top_features = [str(feature) for feature in payload.get("top_features", []) if str(feature)]
        resolved: list[tuple[str, str]] = []
        missing: list[str] = []
        for feature in top_features:
            column = _resolve_feature(frame, feature)
            if column is None:
                missing.append(feature)
            else:
                resolved.append((feature, column))
        z_parts: list[np.ndarray] = []
        weights: list[float] = []
        for pos, (original, column) in enumerate(resolved):
            z = _trailing_z(frame, column, original, config=config)
            z_parts.append(z.to_numpy(dtype=np.float32, copy=False))
            weights.append(1.0 / np.sqrt(float(pos + 1)))
        safe_archetype = _safe_name(archetype_name)
        if z_parts and len(z_parts) >= int(config.min_resolved_features):
            z_stack = np.vstack(z_parts).astype(np.float32, copy=False)
            finite = np.isfinite(z_stack)
            weight_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
            weighted_abs = np.where(finite, np.abs(z_stack) * weight_arr, 0.0)
            weight_sum = np.sum(np.where(finite, weight_arr, 0.0), axis=0)
            intensity = np.divide(
                np.sum(weighted_abs, axis=0),
                np.maximum(weight_sum, float(config.eps)),
                out=np.zeros(len(frame), dtype=np.float32),
                where=weight_sum > float(config.eps),
            )
            support = np.mean(finite, axis=0).astype(np.float32, copy=False)
            score = (1.0 - np.exp(-np.clip(intensity, 0.0, float(config.clip_z)) / 2.0)).astype(np.float32, copy=False)
            score = (score * np.clip(support, 0.0, 1.0)).astype(np.float32, copy=False)
        else:
            score = np.zeros(len(frame), dtype=np.float32)
            support = np.zeros(len(frame), dtype=np.float32)
        score_name = f"{config.archetype_prefix}{safe_archetype}_score"
        support_name = f"{config.archetype_prefix}{safe_archetype}_support"
        data[score_name] = score
        data[support_name] = support
        archetype_scores.append(score)
        archetype_supports.append(support)
        alias_columns: list[str] = []
        if bool(config.include_ranked_probability_aliases):
            ranked_score_name = f"archetype_{rank}_score"
            ranked_support_name = f"archetype_{rank}_support"
            data[ranked_score_name] = score
            data[ranked_support_name] = support
            alias_columns.extend([ranked_score_name, ranked_support_name])
        if bool(config.include_deployable_aliases):
            for alias in payload.get("deployable_features", []) or []:
                alias_name = _safe_name(str(alias))
                if alias_name and alias_name not in data:
                    data[alias_name] = score
                    alias_columns.append(alias_name)
        diagnostics["archetypes"][archetype_name] = {
            "rank": int(rank),
            "mechanism_channel": str(payload.get("mechanism_channel", "")),
            "evidence_score": float(payload.get("evidence_score", 0.0) or 0.0),
            "requested_features": int(len(top_features)),
            "resolved_features": int(len(resolved)),
            "min_resolved_features": int(config.min_resolved_features),
            "active": bool(len(resolved) >= int(config.min_resolved_features)),
            "missing_features": missing,
            "resolved_feature_map": {original: column for original, column in resolved},
            "score_column": score_name,
            "support_column": support_name,
            "alias_columns": alias_columns,
        }

    if archetype_scores:
        score_matrix = np.vstack(archetype_scores).astype(np.float32, copy=False)
        support_matrix = np.vstack(archetype_supports).astype(np.float32, copy=False)
        total = np.sum(score_matrix, axis=0)
        probs = np.divide(
            score_matrix,
            np.maximum(total.reshape(1, -1), float(config.eps)),
            out=np.zeros_like(score_matrix, dtype=np.float32),
            where=total.reshape(1, -1) > float(config.eps),
        )
        for rank, (archetype_name, _payload) in enumerate(ranked, start=1):
            safe_archetype = _safe_name(archetype_name)
            prob = probs[rank - 1].astype(np.float32, copy=False)
            probability_column = f"{config.archetype_prefix}{safe_archetype}_probability"
            data[probability_column] = prob
            diag_entry = diagnostics.get("archetypes", {}).get(archetype_name)
            if isinstance(diag_entry, dict):
                diag_entry["probability_column"] = probability_column
            if bool(config.include_ranked_probability_aliases):
                probability_alias = f"archetype_{rank}_probability"
                data[probability_alias] = prob
                if isinstance(diag_entry, dict):
                    diag_entry.setdefault("alias_columns", []).append(probability_alias)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(np.where(probs > 0.0, probs * np.log(np.maximum(probs, float(config.eps))), 0.0), axis=0)
        max_entropy = np.log(float(max(2, probs.shape[0])))
        data["archetype_uncertainty"] = np.clip(entropy / max_entropy, 0.0, 1.0).astype(np.float32)
        data["archetype_support_score"] = np.nanmean(support_matrix, axis=0).astype(np.float32)
        data["historical_support"] = np.divide(
            np.sum(support_matrix * np.maximum(score_matrix, 0.0), axis=0),
            np.maximum(np.sum(np.maximum(score_matrix, 0.0), axis=0), float(config.eps)),
            out=np.nanmean(support_matrix, axis=0).astype(np.float32),
            where=np.sum(np.maximum(score_matrix, 0.0), axis=0) > float(config.eps),
        ).astype(np.float32)
        data["dominant_archetype_score"] = np.nanmax(score_matrix, axis=0).astype(np.float32)

    out = pd.DataFrame(data, index=frame.index).replace([np.inf, -np.inf], np.nan)
    out = out.astype(np.float32, copy=False)
    diagnostics["output_feature_count"] = int(out.shape[1])
    diagnostics["feature_columns"] = list(out.columns)
    return out, diagnostics
