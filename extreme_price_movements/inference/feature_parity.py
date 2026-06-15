from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.feature_transform_contract import FeatureTransformContract
from extreme_price_movements.utils import tprint


class FeatureParityError(RuntimeError):
    def __init__(self, message: str, report: dict | None = None):
        self.report = report or {}
        detail = ""
        if self.report:
            errors = self.report.get("global_errors") or []
            nonfinite = self.report.get("nonfinite_features") or []
            missing = self.report.get("missing_features") or []
            parts = []
            if errors:
                parts.append(f"errors={errors[:5]}")
            if nonfinite:
                parts.append(
                    f"nonfinite_features={nonfinite[:20]}"
                    + (f" (+{len(nonfinite) - 20} more)" if len(nonfinite) > 20 else "")
                )
            if missing:
                parts.append(
                    f"missing_features={missing[:20]}"
                    + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
                )
            if parts:
                detail = ": " + "; ".join(parts)
        super().__init__(f"{message}{detail}")


@dataclass
class FeatureParityReport:
    ok: bool = True
    mode: str = "strict"
    scope: str = "symbol"
    run_id: str = ""
    contract_hash: str | None = None
    state_contract_hash: str | None = None
    end_ts: str = ""
    global_errors: list[str] = field(default_factory=list)
    per_symbol_errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    accepted_symbols: list[str] = field(default_factory=list)
    rejected_symbols: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "scope": self.scope,
            "run_id": self.run_id,
            "contract_hash": self.contract_hash,
            "state_contract_hash": self.state_contract_hash,
            "end_ts": self.end_ts,
            "global_errors": list(self.global_errors),
            "per_symbol_errors": dict(self.per_symbol_errors),
            "accepted_symbols": list(self.accepted_symbols),
            "rejected_symbols": list(self.rejected_symbols),
        }


def _strict(cfg: dict[str, Any] | None, strict: bool | None = None) -> bool:
    if strict is not None:
        return bool(strict)
    return bool((cfg or {}).get("strict_feature_parity", True))


def _scope(cfg: dict[str, Any] | None, scope: str | None = None) -> str:
    value = str(
        scope or (cfg or {}).get("strict_feature_parity_scope", "symbol")
    ).lower()
    return value if value in {"symbol", "global"} else "symbol"


def _state_hash(model_bundle: dict[str, Any] | None) -> str | None:
    if not isinstance(model_bundle, dict):
        return None
    value = model_bundle.get("feature_transform_contract_hash")
    if value:
        return str(value)
    inner = model_bundle.get("bundle")
    if isinstance(inner, dict) and inner.get("feature_transform_contract_hash"):
        return str(inner.get("feature_transform_contract_hash"))
    return None


def _contract_from_bundle(
    model_bundle: dict[str, Any] | None,
) -> FeatureTransformContract | None:
    if not isinstance(model_bundle, dict):
        return None
    contract = model_bundle.get("feature_transform_contract")
    if isinstance(contract, FeatureTransformContract):
        return contract
    inner = model_bundle.get("bundle")
    if isinstance(inner, dict) and isinstance(
        inner.get("feature_transform_contract"), FeatureTransformContract
    ):
        return inner.get("feature_transform_contract")
    return None


def _manifest_from_bundle(model_bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(model_bundle, dict):
        return None
    manifest = model_bundle.get("feature_transform_manifest")
    if isinstance(manifest, dict):
        return manifest
    inner = model_bundle.get("bundle")
    if isinstance(inner, dict) and isinstance(
        inner.get("feature_transform_manifest"), dict
    ):
        return inner.get("feature_transform_manifest")
    return None


def validate_model_bundle_transform_contract(
    model_bundle: dict[str, Any] | None,
    cfg: dict[str, Any] | None,
    run_id: str,
) -> FeatureTransformContract:
    report = FeatureParityReport(
        mode="strict" if _strict(cfg) else "permissive",
        scope=_scope(cfg),
        run_id=str(run_id),
    )
    contract = _contract_from_bundle(model_bundle)
    state_hash = _state_hash(model_bundle)
    manifest = _manifest_from_bundle(model_bundle)
    report.state_contract_hash = state_hash
    report.contract_hash = (
        getattr(contract, "contract_hash", None) if contract is not None else None
    )

    if contract is None:
        report.global_errors.append("missing_feature_transform_contract")
    if not state_hash:
        report.global_errors.append("missing_state_feature_transform_contract_hash")
    if contract is not None and state_hash and contract.contract_hash != state_hash:
        report.global_errors.append("feature_transform_contract_hash_mismatch")
    if (
        isinstance(manifest, dict)
        and state_hash
        and str(manifest.get("contract_hash") or "") != state_hash
    ):
        report.global_errors.append("feature_transform_manifest_hash_mismatch")
    if contract is not None:
        if str(getattr(contract, "run_id", "")) != str(run_id):
            report.global_errors.append("feature_transform_contract_run_id_mismatch")
        cfg_market = str((cfg or {}).get("market_mode", "") or "")
        contract_market = str(getattr(contract, "market_mode", "") or "")
        if cfg_market and contract_market and cfg_market != contract_market:
            report.global_errors.append(
                "feature_transform_contract_market_mode_mismatch"
            )
        try:
            contract.validate_no_fit_required()
        except Exception as exc:
            report.global_errors.append(f"feature_transform_contract_invalid:{exc}")
        transformable = list(getattr(contract, "transformable_cols", []) or [])
        if not transformable:
            passthrough = set(getattr(contract, "passthrough_cols", []) or [])
            never = set(getattr(contract, "never_transform_cols", []) or [])
            transformable = [
                c
                for c in (getattr(contract, "raw_feature_cols", []) or [])
                if c not in passthrough and c not in never
            ]
        missing_stats = [
            c
            for c in transformable
            if c not in getattr(contract, "per_column_stats", {})
        ]
        if missing_stats:
            report.global_errors.append(
                "feature_transform_contract_missing_stats:"
                + ",".join(missing_stats[:50])
            )

    if report.global_errors and _strict(cfg):
        report.ok = False
        tprint(f"FEATURE PARITY: global refusal {report.global_errors[:5]}")
        raise FeatureParityError(
            "Feature transform contract parity failed", report.asdict()
        )
    if contract is None:
        raise FeatureParityError(
            "Feature transform contract is missing", report.asdict()
        )
    tprint(
        "FEATURE PARITY: strict mode enabled "
        f"run_id={run_id} hash={contract.contract_hash} "
        f"required_contract_cols={len(getattr(contract, 'transformed_feature_cols', []) or [])}"
    )
    return contract


def validate_required_features_against_contract(
    required_feature_keys: Iterable[str],
    contract: FeatureTransformContract,
    strict: bool = True,
) -> None:
    required = {str(k) for k in (required_feature_keys or []) if str(k)}
    output = set(getattr(contract, "transformed_feature_cols", []) or []) | set(
        getattr(contract, "passthrough_cols", []) or []
    )
    never = set(getattr(contract, "never_transform_cols", []) or [])
    missing = sorted(required - output)
    never_requested = sorted(required & never)
    if strict and (missing or never_requested):
        raise FeatureParityError(
            "Required model features are absent from the transform contract",
            {
                "global_errors": [
                    *(
                        [f"missing_from_contract:{','.join(missing[:50])}"]
                        if missing
                        else []
                    ),
                    *(
                        [f"never_transform_requested:{','.join(never_requested[:50])}"]
                        if never_requested
                        else []
                    ),
                ]
            },
        )


def validate_transform_stat_completeness(
    contract: FeatureTransformContract,
    required_feature_keys: Iterable[str],
    strict: bool = True,
) -> None:
    required = {str(k) for k in (required_feature_keys or []) if str(k)}
    passthrough = set(getattr(contract, "passthrough_cols", []) or [])
    never = set(getattr(contract, "never_transform_cols", []) or [])
    kind = str(getattr(contract, "transform_config", {}).get("kind", "robust")).lower()
    stats_map = getattr(contract, "per_column_stats", {}) or {}
    errors: list[str] = []
    for feature in sorted(required):
        if feature in passthrough or feature in never:
            continue
        stats = stats_map.get(feature)
        if not isinstance(stats, dict):
            errors.append(f"{feature}:missing_stats")
            continue
        keys = ["median", "iqr"] if kind == "robust" else ["mean", "std"]
        if (
            str(
                getattr(contract, "transform_config", {}).get("impute", "median")
            ).lower()
            == "median"
        ):
            keys.append("median")
        for key in sorted(set(keys)):
            value = stats.get(key)
            if value is None or not np.isfinite(float(value)):
                errors.append(f"{feature}:invalid_{key}")
        scale_key = "iqr" if kind == "robust" else "std"
        scale = stats.get(scale_key, 1.0)
        if not np.isfinite(float(scale)) or abs(float(scale)) < 1e-12:
            errors.append(f"{feature}:invalid_scale")
        for key in ("clip_lo", "clip_hi"):
            value = stats.get(key)
            if value is not None and not np.isfinite(float(value)):
                errors.append(f"{feature}:invalid_{key}")
    if errors and strict:
        raise FeatureParityError(
            "Feature transform stats are incomplete",
            {
                "global_errors": ["missing_or_invalid_transform_stats"],
                "details": errors[:100],
            },
        )


def _add_symbol_error(
    report: FeatureParityReport, symbol: str, category: str, value: Any
) -> None:
    bucket = report.per_symbol_errors.setdefault(str(symbol), {})
    existing = bucket.setdefault(category, [])
    if isinstance(existing, list):
        if isinstance(value, list):
            existing.extend(value)
        else:
            existing.append(value)
    else:
        bucket[category] = value


_SOURCE_ALTERNATIVES: dict[str, tuple[tuple[str, ...], ...]] = {
    "mark": (("mark_price",),),
    "perp_volume": (("quote_volume",), ("volume", "close")),
    "spot_or_index": (
        ("spot_close",),
        ("index_price",),
        ("index_close",),
        ("canonical_index",),
    ),
    "spot_close": (("spot_close",),),
    "spot_ohlc": (("spot_open", "spot_high", "spot_low", "spot_close"),),
    "spot_volume": (("spot_volume", "spot_close"),),
    "funding": (("funding_rate",),),
    "open_interest": (("open_interest",),),
    "orderbook": (
        ("orderbook_best_bid", "orderbook_best_ask"),
        ("best_bid", "best_ask"),
        ("bid", "ask"),
        ("orderbook_bid", "orderbook_ask"),
    ),
}

_SOURCE_MAX_STALENESS_HOURS: dict[str, float] = {
    "mark": 2.0,
    "perp_volume": 2.0,
    "spot_or_index": 2.0,
    "spot_close": 2.0,
    "spot_ohlc": 2.0,
    "spot_volume": 2.0,
    "funding": 12.0,
    "open_interest": 2.0,
    "orderbook": 1.0,
}


def _feature_source_requirements(feature_key: str) -> set[str]:
    key = str(feature_key or "")
    lower = key.lower()
    req: set[str] = set()

    if (
        lower == "mark_price"
        or lower.startswith("mark_")
        or "_mark_" in lower
        or lower.endswith("_mark_frac")
    ):
        req.add("mark")
    if (
        lower in {"index_price", "canonical_index", "basis", "basis_pct", "basis_frac"}
        or lower.startswith(("basis_", "premium_", "perp_index_", "perp_vs_index_"))
        or lower.startswith(("mark_index_", "mark_vs_index_"))
        or lower.startswith("basis")
        or "basis_funding_div" in lower
        or "basis_adjusted" in lower
        or "premium_mean_reversion" in lower
    ):
        req.add("spot_or_index")
    if (
        lower == "spot_price"
        or lower.startswith("spot_ret")
        or lower.startswith("perp_minus_spot")
        or lower.startswith("spot_perp")
        or lower in {"spot_available", "spot_perp_ratio", "spot_perp_log_ratio"}
    ):
        req.add("spot_close")
    if lower.startswith(("spot_range", "spot_breakout", "spot_liquidity_sweep")):
        req.add("spot_ohlc")
    if (
        lower.startswith("spot_volume")
        or lower.startswith("spot_quote_volume")
        or lower.startswith("spot_perp_volume")
    ):
        req.add("spot_volume")
    if (
        lower.startswith(("fund", "funding"))
        or "_funding_" in lower
        or "basis_funding_div" in lower
        or lower.startswith("carry_adj")
        or "fund_" in lower
    ):
        req.add("funding")
    if (
        lower.startswith(("oi_", "open_interest"))
        or lower in {
            "leverage_build",
            "leverage_build_score",
            "unwind",
            "unwind_score",
            "squeeze_prob",
        }
    ):
        req.add("open_interest")
    if (
        lower in {
            "dist_vwap_norm",
            "dist_vwap_12_atr",
            "dist_vwap_24_atr",
            "dist_vwap_96_atr",
            "trapped_longs_12",
            "trapped_longs_24",
            "trapped_longs_96",
            "vwap_zone_1d_atr",
            "vwap_zone_7d_atr",
            "dist_stack",
            "distance_to_vwap",
            "dist_vwap_atr",
            "dist_weekly_vwap",
            "z_vwap_12",
            "z_vwap_24",
            "z_dist_vwap_24",
        }
        or lower.startswith("dist_vwap_")
        or lower.startswith("trapped_longs_")
        or lower.startswith("vwap_zone_")
        or lower.startswith("oi_rel_vol_")
    ):
        req.add("perp_volume")
    if lower.startswith(("ob_", "obw_", "orderbook_", "book_")):
        req.add("orderbook")

    return req


def required_source_groups_for_features(
    required_feature_keys: Iterable[str] | None,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature in sorted({str(k) for k in (required_feature_keys or []) if str(k)}):
        for group in sorted(_feature_source_requirements(feature)):
            groups.setdefault(group, []).append(feature)
    return groups


def _source_staleness_hours(cfg: dict[str, Any] | None, group: str) -> float:
    overrides = (cfg or {}).get("feature_source_max_staleness_hours", {})
    if isinstance(overrides, dict) and group in overrides:
        try:
            return float(overrides[group])
        except (TypeError, ValueError):
            pass
    return float(_SOURCE_MAX_STALENESS_HOURS.get(group, 2.0))


def _finite_source_timestamp(
    panel: dict[str, pd.DataFrame],
    source_key: str,
    symbol: str,
    end_ts: pd.Timestamp,
) -> pd.Timestamp | None:
    frame = panel.get(source_key) if isinstance(panel, dict) else None
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or symbol not in frame.columns
        or not isinstance(frame.index, pd.DatetimeIndex)
    ):
        return None
    series = pd.to_numeric(frame[symbol], errors="coerce")
    if series.index.tz is None and end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)
    elif series.index.tz is not None and end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(series.index.tz)
    elif series.index.tz is not None and end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(series.index.tz)
    series = series.loc[series.index <= end_ts]
    if series.empty:
        return None
    values = series.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    source_key_l = str(source_key or "").lower()
    positive_required = any(
        token in source_key_l
        for token in ("price", "close", "open", "high", "low", "interest")
    )
    if positive_required:
        finite &= values > 0.0
    elif "volume" in source_key_l:
        finite &= values >= 0.0
    if not finite.any():
        return None
    return pd.Timestamp(series.index[np.flatnonzero(finite)[-1]])


def _source_alternative_is_available(
    panel: dict[str, pd.DataFrame],
    alternative: tuple[str, ...],
    symbol: str,
    end_ts: pd.Timestamp,
    max_age: pd.Timedelta,
) -> tuple[bool, dict[str, Any]]:
    seen_ts: list[pd.Timestamp] = []
    missing: list[str] = []
    stale: dict[str, str] = {}
    for source_key in alternative:
        ts = _finite_source_timestamp(panel, source_key, symbol, end_ts)
        if ts is None:
            missing.append(source_key)
            continue
        if end_ts - ts > max_age:
            stale[source_key] = ts.isoformat()
            continue
        seen_ts.append(ts)
    if len(seen_ts) == len(alternative):
        return True, {"latest_ts": min(seen_ts).isoformat()}
    return False, {"missing": missing, "stale": stale}


def _source_rejection_summary(report: dict[str, Any]) -> dict[str, Any]:
    per_symbol = dict(report.get("per_symbol_errors") or {})
    groups: dict[str, int] = {}
    features: dict[str, int] = {}
    source_keys: dict[str, int] = {}
    missing_source_keys: dict[str, int] = {}
    stale_source_keys: dict[str, int] = {}
    group_samples: dict[str, list[str]] = {}
    feature_samples: dict[str, list[str]] = {}
    source_key_samples: dict[str, list[str]] = {}

    for symbol, errors in per_symbol.items():
        values = (errors or {}).get("missing_or_stale_source_panels", [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if not isinstance(value, dict):
                continue
            group = str(value.get("group") or "")
            if group:
                groups[group] = groups.get(group, 0) + 1
                bucket = group_samples.setdefault(group, [])
                if len(bucket) < 10:
                    bucket.append(str(symbol))
            for feature in value.get("features") or []:
                feature_s = str(feature)
                features[feature_s] = features.get(feature_s, 0) + 1
                bucket = feature_samples.setdefault(feature_s, [])
                if len(bucket) < 5:
                    bucket.append(str(symbol))
            for attempt in value.get("alternatives") or []:
                if not isinstance(attempt, dict):
                    continue
                for source_key in attempt.get("missing") or []:
                    source_s = str(source_key)
                    source_keys[source_s] = source_keys.get(source_s, 0) + 1
                    missing_source_keys[source_s] = (
                        missing_source_keys.get(source_s, 0) + 1
                    )
                    bucket = source_key_samples.setdefault(source_s, [])
                    if len(bucket) < 5:
                        bucket.append(str(symbol))
                stale = attempt.get("stale") or {}
                if isinstance(stale, dict):
                    for source_key in stale:
                        source_s = str(source_key)
                        source_keys[source_s] = source_keys.get(source_s, 0) + 1
                        stale_source_keys[source_s] = (
                            stale_source_keys.get(source_s, 0) + 1
                        )
                        bucket = source_key_samples.setdefault(source_s, [])
                        if len(bucket) < 5:
                            bucket.append(str(symbol))

    def _top(counter: dict[str, int], limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"key": key, "count": int(count)}
            for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
                :limit
            ]
        ]

    return {
        "rejected_symbol_count": int(len(report.get("rejected_symbols") or [])),
        "accepted_symbol_count": int(len(report.get("accepted_symbols") or [])),
        "by_group": _top(groups),
        "by_feature": _top(features),
        "by_source_key": _top(source_keys),
        "missing_source_keys": _top(missing_source_keys),
        "stale_source_keys": _top(stale_source_keys),
        "sample_symbols_by_group": group_samples,
        "sample_symbols_by_feature": feature_samples,
        "sample_symbols_by_source_key": source_key_samples,
    }


def validate_required_source_panels(
    panel: dict[str, pd.DataFrame],
    symbols: list[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Iterable[str] | None,
    cfg: dict[str, Any] | None = None,
    strict: bool | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    strict_b = _strict(cfg, strict)
    source_groups = required_source_groups_for_features(required_feature_keys)
    report = FeatureParityReport(
        mode="strict" if strict_b else "permissive",
        scope=_scope(cfg, scope),
        end_ts=pd.Timestamp(end_ts).isoformat(),
        accepted_symbols=list(symbols),
    )
    report_dict = report.asdict()
    report_dict["required_source_groups"] = {
        group: features[:100] for group, features in sorted(source_groups.items())
    }
    if not source_groups:
        return apply_feature_parity_scope(report_dict, symbols, report.scope, strict_b)

    end_ts = pd.Timestamp(end_ts)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    if not isinstance(panel, dict):
        report.global_errors.append("missing_source_panel")
        return apply_feature_parity_scope(
            report.asdict() | {"required_source_groups": report_dict["required_source_groups"]},
            symbols,
            report.scope,
            strict_b,
        )

    for symbol in symbols:
        for group, features in sorted(source_groups.items()):
            alternatives = _SOURCE_ALTERNATIVES.get(group, ())
            if not alternatives:
                continue
            max_age = pd.Timedelta(hours=_source_staleness_hours(cfg, group))
            attempts: list[dict[str, Any]] = []
            available = False
            for alternative in alternatives:
                ok, detail = _source_alternative_is_available(
                    panel, alternative, str(symbol), end_ts, max_age
                )
                if ok:
                    available = True
                    break
                attempts.append({"sources": list(alternative), **detail})
            if not available:
                _add_symbol_error(
                    report,
                    str(symbol),
                    "missing_or_stale_source_panels",
                    {
                        "group": group,
                        "features": features[:20],
                        "alternatives": attempts[:5],
                    },
                )

    scoped = report.asdict()
    scoped["required_source_groups"] = report_dict["required_source_groups"]
    return apply_feature_parity_scope(scoped, symbols, report.scope, strict=strict_b)


def validate_raw_history_sufficiency(
    panel: dict[str, pd.DataFrame],
    contract: FeatureTransformContract,
    symbols: list[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Iterable[str] | None = None,
    cfg: dict[str, Any] | None = None,
    strict: bool | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    strict_b = _strict(cfg, strict)
    report = FeatureParityReport(
        mode="strict" if strict_b else "permissive",
        scope=_scope(cfg, scope),
        run_id=str(getattr(contract, "run_id", "")),
        contract_hash=getattr(contract, "contract_hash", None),
        end_ts=pd.Timestamp(end_ts).isoformat(),
        accepted_symbols=list(symbols),
    )
    close = panel.get("close") if isinstance(panel, dict) else None
    if (
        not isinstance(close, pd.DataFrame)
        or close.empty
        or not isinstance(close.index, pd.DatetimeIndex)
    ):
        report.global_errors.append("missing_close_panel")
        report.ok = False
        if strict_b:
            raise FeatureParityError("Raw history parity failed", report.asdict())
        return report.asdict()
    if close.index.has_duplicates:
        report.global_errors.append("duplicate_close_timestamps")
    if not close.index.is_monotonic_increasing:
        report.global_errors.append("non_monotonic_close_index")
    min_ratio = float(
        (cfg or {}).get("feature_parity_min_raw_history_finite_ratio", 0.95)
    )
    warmup = int(getattr(contract, "required_warmup_hours", 0) or 0)
    lookbacks = getattr(contract, "required_lookback_hours_by_feature", {}) or {}
    required = [str(k) for k in (required_feature_keys or []) if str(k)]
    required_hours = max(
        [warmup, *[int(lookbacks.get(k, 0) or 0) for k in required], 1]
    )
    end_ts = pd.Timestamp(end_ts)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    window_start = end_ts - pd.Timedelta(hours=required_hours)
    for symbol in symbols:
        if symbol not in close.columns:
            _add_symbol_error(
                report,
                symbol,
                "insufficient_history",
                {"reason": "missing_close_column"},
            )
            continue
        series = pd.to_numeric(close[symbol], errors="coerce")
        if bool((cfg or {}).get("feature_parity_require_current_timestamp", True)):
            if end_ts not in close.index or not np.isfinite(series.loc[end_ts]):
                _add_symbol_error(report, symbol, "stale_features", ["close"])
        window = series.loc[(series.index >= window_start) & (series.index <= end_ts)]
        finite_ratio = (
            float(np.isfinite(window.to_numpy(dtype=np.float64)).mean())
            if len(window)
            else 0.0
        )
        available_hours = int(len(window))
        if available_hours < required_hours or finite_ratio < min_ratio:
            _add_symbol_error(
                report,
                symbol,
                "insufficient_history",
                {
                    "available_hours": available_hours,
                    "required_hours": required_hours,
                    "finite_ratio": finite_ratio,
                },
            )
    return apply_feature_parity_scope(
        report.asdict(), symbols, report.scope, strict=strict_b
    )


def validate_raw_feature_availability(
    raw_feats: dict[str, pd.DataFrame],
    contract: FeatureTransformContract,
    symbols: list[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Iterable[str],
    cfg: dict[str, Any] | None = None,
    strict: bool | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    strict_b = _strict(cfg, strict)
    report = FeatureParityReport(
        mode="strict" if strict_b else "permissive",
        scope=_scope(cfg, scope),
        run_id=str(getattr(contract, "run_id", "")),
        contract_hash=getattr(contract, "contract_hash", None),
        end_ts=pd.Timestamp(end_ts).isoformat(),
        accepted_symbols=list(symbols),
    )
    end_ts = pd.Timestamp(end_ts)
    require_current = bool(
        (cfg or {}).get("feature_parity_require_current_timestamp", True)
    )
    required = [str(k) for k in (required_feature_keys or []) if str(k)]
    for feature in required:
        frame = raw_feats.get(feature)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            report.global_errors.append(f"missing_raw_feature:{feature}")
            continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            report.global_errors.append(f"raw_feature_non_datetime_index:{feature}")
            continue
        for symbol in symbols:
            if symbol not in frame.columns:
                _add_symbol_error(report, symbol, "missing_raw_features", [feature])
                continue
            if require_current:
                if end_ts not in frame.index:
                    _add_symbol_error(report, symbol, "stale_features", [feature])
                    continue
                value = frame.at[end_ts, symbol]
            else:
                series = frame[symbol].loc[frame.index <= end_ts].ffill()
                value = np.nan if series.empty else series.iloc[-1]
            if not np.isfinite(
                pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            ):
                _add_symbol_error(report, symbol, "nonfinite_features", [feature])
    return apply_feature_parity_scope(
        report.asdict(), symbols, report.scope, strict=strict_b
    )


def validate_transformed_feature_panels(
    feats: dict[str, pd.DataFrame],
    contract: FeatureTransformContract,
    symbols: list[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Iterable[str],
    cfg: dict[str, Any] | None = None,
    strict: bool | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    strict_b = _strict(cfg, strict)
    report = FeatureParityReport(
        mode="strict" if strict_b else "permissive",
        scope=_scope(cfg, scope),
        run_id=str(getattr(contract, "run_id", "")),
        contract_hash=getattr(contract, "contract_hash", None),
        end_ts=pd.Timestamp(end_ts).isoformat(),
        accepted_symbols=list(symbols),
    )
    end_ts = pd.Timestamp(end_ts)
    required = [str(k) for k in (required_feature_keys or []) if str(k)]
    for feature in required:
        frame = feats.get(feature)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            report.global_errors.append(f"missing_feature:{feature}")
            continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            report.global_errors.append(f"feature_non_datetime_index:{feature}")
            continue
        if frame.index.has_duplicates:
            report.global_errors.append(f"feature_duplicate_timestamps:{feature}")
        for symbol in symbols:
            if symbol not in frame.columns:
                _add_symbol_error(report, symbol, "missing_features", [feature])
                continue
            if end_ts not in frame.index:
                _add_symbol_error(report, symbol, "stale_features", [feature])
                continue
            value = frame.at[end_ts, symbol]
            if not np.isfinite(
                pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            ):
                _add_symbol_error(report, symbol, "nonfinite_features", [feature])
    return apply_feature_parity_scope(
        report.asdict(), symbols, report.scope, strict=strict_b
    )


def validate_final_model_matrix(
    X: pd.DataFrame,
    model_feature_cols: Iterable[str],
    model_key: str,
    strict: bool = True,
) -> pd.DataFrame:
    cols = [str(c) for c in (model_feature_cols or [])]
    report = {"model_key": model_key, "global_errors": []}
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        report["global_errors"].append("empty_model_matrix")
    elif list(map(str, X.columns)) != cols:
        report["global_errors"].append("model_matrix_column_order_mismatch")
        report["expected_columns_sample"] = cols[:50]
        report["actual_columns_sample"] = list(map(str, X.columns))[:50]
    if not cols:
        report["global_errors"].append("empty_model_feature_contract")
    if report["global_errors"] and strict:
        raise FeatureParityError("Final model matrix parity failed", report)
    X = X.reindex(columns=cols)
    try:
        X_float = X.astype(np.float32, copy=False)
    except Exception as exc:
        report["global_errors"].append(f"model_matrix_float32_cast_failed:{exc}")
        if strict:
            raise FeatureParityError(
                "Final model matrix dtype parity failed", report
            ) from exc
        X_float = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    values = X_float.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        bad_cols = [
            str(col)
            for col in X_float.columns
            if not np.isfinite(
                X_float[col].to_numpy(dtype=np.float32, copy=False)
            ).all()
        ]
        report["global_errors"].append("model_matrix_nonfinite")
        report["nonfinite_features"] = bad_cols[:100]
        if strict:
            raise FeatureParityError(
                "Final model matrix contains non-finite values", report
            )
    return X_float


def apply_feature_parity_scope(
    report: dict[str, Any],
    symbols: list[str],
    scope: str = "symbol",
    strict: bool = True,
) -> dict[str, Any]:
    global_errors = list(report.get("global_errors") or [])
    per_symbol = dict(report.get("per_symbol_errors") or {})
    rejected = sorted(str(s) for s in per_symbol)
    accepted = [str(s) for s in symbols if str(s) not in set(rejected)]
    report["rejected_symbols"] = rejected
    report["accepted_symbols"] = accepted
    report["ok"] = not global_errors and not rejected
    report["source_rejection_summary"] = _source_rejection_summary(report)
    if not strict:
        return report
    if global_errors:
        raise FeatureParityError("Global feature parity failure", report)
    if scope == "global" and rejected:
        raise FeatureParityError("Global feature parity scope rejected symbols", report)
    if not accepted:
        reasons: dict[str, int] = {}
        groups: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        for symbol, errors in per_symbol.items():
            for key, values in (errors or {}).items():
                reasons[key] = reasons.get(key, 0) + 1
                if key != "missing_or_stale_source_panels":
                    continue
                for value in values if isinstance(values, list) else [values]:
                    if not isinstance(value, dict):
                        continue
                    group = str(value.get("group") or "")
                    if not group:
                        continue
                    groups[group] = groups.get(group, 0) + 1
                    bucket = samples.setdefault(group, [])
                    if len(bucket) < 5:
                        bucket.append(str(symbol))
        tprint(
            "FEATURE PARITY: no symbols accepted "
            f"rejected={len(rejected)} top_reasons={reasons} "
            f"source_groups={groups} sample_symbols={samples}"
        )
        raise FeatureParityError("No symbols passed strict feature parity", report)
    if rejected:
        reasons: dict[str, int] = {}
        for errors in per_symbol.values():
            for key in errors:
                reasons[key] = reasons.get(key, 0) + 1
        tprint(
            "FEATURE PARITY: rejected symbols "
            f"n={len(rejected)} accepted={len(accepted)} top_reasons={reasons}"
        )
        source_summary = report.get("source_rejection_summary") or {}
        if source_summary.get("by_group"):
            tprint(
                "FEATURE PARITY: rejected source groups "
                f"{source_summary.get('by_group', [])[:10]} "
                f"missing_sources={source_summary.get('missing_source_keys', [])[:10]} "
                f"stale_sources={source_summary.get('stale_source_keys', [])[:10]}"
            )
    return report


def validate_feature_parity_before_prediction(
    *,
    feats: dict[str, pd.DataFrame],
    contract: FeatureTransformContract,
    symbols: list[str],
    end_ts: pd.Timestamp,
    required_feature_keys: Iterable[str],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_required_features_against_contract(
        required_feature_keys,
        contract,
        strict=_strict(cfg),
    )
    validate_transform_stat_completeness(
        contract,
        required_feature_keys,
        strict=_strict(cfg),
    )
    return validate_transformed_feature_panels(
        feats,
        contract,
        symbols,
        end_ts,
        required_feature_keys,
        cfg=cfg,
        strict=_strict(cfg),
        scope=_scope(cfg),
    )
