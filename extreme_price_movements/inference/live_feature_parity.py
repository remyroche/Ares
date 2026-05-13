"""Decision-time feature parity checks for live-vs-OOS diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

FEATURE_PARITY_COLUMNS = [
    "timestamp",
    "decision_ts",
    "signal_bar_ts",
    "symbol",
    "side",
    "strategy_id",
    "run_id",
    "model_artifact_run_id",
    "feature_contract_hash",
    "feature",
    "live_value",
    "oos_value",
    "abs_diff",
    "rel_diff",
    "live_feature_ts",
    "oos_feature_ts",
    "live_feature_bar_ts",
    "oos_feature_bar_ts",
    "live_feature_available_ts",
    "oos_feature_available_ts",
    "live_source_max_ts",
    "oos_source_max_ts",
    "feature_bar_after_decision",
    "feature_available_after_decision",
    "source_max_after_decision",
    "lookahead_violation",
    "availability_unknown",
    "parity_status",
]


def _normalise_symbol_for_join(symbol: object) -> str:
    """Normalize symbols conservatively for decision/feature joins."""
    text = str(symbol or "").upper().strip()
    return text.replace(":USDT", "").replace("/", "_").replace("-", "_")


def _normalise_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a UTC-indexed, sorted feature frame for deterministic as-of lookup."""
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.loc[pd.notna(out.index)]
    out = out.sort_index()
    return out


def _normalise_decisions(
    decisions: Optional[pd.DataFrame],
    *,
    timestamps: Optional[Iterable[pd.Timestamp]],
    symbols: Optional[Iterable[str]],
) -> pd.DataFrame:
    if decisions is not None and not decisions.empty:
        out = decisions.copy()
        if "timestamp" not in out.columns and "decision_ts" in out.columns:
            out["timestamp"] = out["decision_ts"]
        if "timestamp" not in out.columns or "symbol" not in out.columns:
            raise ValueError("decisions must contain timestamp/decision_ts and symbol columns")
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["decision_ts"] = pd.to_datetime(
            out.get("decision_ts", out["timestamp"]), utc=True, errors="coerce"
        )
        signal_raw = out["signal_bar_ts"] if "signal_bar_ts" in out.columns else out["timestamp"]
        signal = pd.to_datetime(signal_raw, utc=True, errors="coerce")
        out["signal_bar_ts"] = pd.Series(signal, index=out.index).fillna(out["timestamp"])
        out["symbol"] = out["symbol"].map(_normalise_symbol_for_join)
        for col in (
            "side",
            "strategy_id",
            "run_id",
            "model_artifact_run_id",
            "feature_contract_hash",
            "feature_available_ts",
            "live_feature_available_ts",
            "oos_feature_available_ts",
            "feature_source_max_ts",
            "live_source_max_ts",
            "oos_source_max_ts",
        ):
            if col not in out.columns:
                out[col] = pd.NA
        return out.dropna(subset=["timestamp", "signal_bar_ts", "symbol"]).drop_duplicates()
    if timestamps is None or symbols is None:
        raise ValueError("Provide either decisions or both timestamps and symbols")
    ts = pd.to_datetime(list(timestamps), utc=True, errors="coerce")
    rows = [
        {"timestamp": t, "decision_ts": t, "signal_bar_ts": t, "symbol": _normalise_symbol_for_join(s)}
        for t in ts
        for s in symbols
        if pd.notna(t)
    ]
    return _normalise_decisions(pd.DataFrame(rows), timestamps=None, symbols=None)


def _feature_keys(
    live_features: Dict[str, pd.DataFrame],
    oos_features: Dict[str, pd.DataFrame],
    requested: Optional[Iterable[str]],
    *,
    include_extra_features: bool,
) -> Sequence[str]:
    if requested is not None:
        return [str(k) for k in requested if str(k)]
    live = set(live_features.keys())
    oos = set(oos_features.keys())
    return sorted(live | oos) if include_extra_features else sorted(live & oos)


def _column_for_symbol(df: pd.DataFrame, symbol: str) -> Optional[str]:
    if symbol in df.columns:
        return symbol
    normalized = {_normalise_symbol_for_join(col): col for col in df.columns}
    return normalized.get(_normalise_symbol_for_join(symbol))


def _value_at_or_before(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    symbol: str,
    *,
    allow_asof: bool,
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan, pd.NaT, "missing_feature"
    frame = _normalise_feature_frame(df)
    if frame.empty:
        return np.nan, pd.NaT, "missing_timestamp"
    col = _column_for_symbol(frame, symbol)
    if col is None:
        return np.nan, pd.NaT, "missing_symbol"
    timestamp = pd.Timestamp(timestamp)
    if timestamp in frame.index:
        value = frame.loc[timestamp, col]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return value, timestamp, "ok"
    if not allow_asof:
        return np.nan, pd.NaT, "missing_timestamp"
    idx = pd.DatetimeIndex(frame.index)
    positions = np.flatnonzero(idx <= timestamp)
    if positions.size == 0:
        return np.nan, pd.NaT, "missing_timestamp"
    pos = int(positions[-1])
    return frame.iloc[pos][col], idx[pos], "asof"


def _first_ts(row: pd.Series, names: Sequence[str]) -> pd.Timestamp:
    for name in names:
        if name in row and pd.notna(row[name]):
            ts = pd.to_datetime(row[name], utc=True, errors="coerce")
            if pd.notna(ts):
                return pd.Timestamp(ts)
    return pd.NaT


def build_feature_parity_report(
    live_features: Dict[str, pd.DataFrame],
    oos_features: Dict[str, pd.DataFrame],
    *,
    decisions: Optional[pd.DataFrame] = None,
    timestamps: Optional[Iterable[pd.Timestamp]] = None,
    symbols: Optional[Iterable[str]] = None,
    feature_keys: Optional[Iterable[str]] = None,
    include_extra_features: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    allow_asof: bool = False,
) -> pd.DataFrame:
    """Compare live and OOS feature values at each decision's signal bar.

    Values are looked up at ``signal_bar_ts`` (defaulting to ``timestamp``), while
    leakage checks are evaluated against the actual live decision ``timestamp``.
    Supplied feature-availability/source timestamps after the decision time are
    marked as lookahead violations; unknown availability is reported but does not
    fail the leakage check.
    """
    live_features = live_features or {}
    oos_features = oos_features or {}
    decisions_df = _normalise_decisions(decisions, timestamps=timestamps, symbols=symbols)
    keys = _feature_keys(
        live_features,
        oos_features,
        feature_keys,
        include_extra_features=include_extra_features,
    )

    rows = []
    for decision in decisions_df.to_dict("records"):
        drow = pd.Series(decision)
        decision_ts = pd.Timestamp(decision["timestamp"])
        signal_bar_ts = pd.Timestamp(decision.get("signal_bar_ts", decision_ts))
        symbol = _normalise_symbol_for_join(decision["symbol"])
        for key in keys:
            live_val, live_bar_ts, live_status = _value_at_or_before(
                live_features.get(key), signal_bar_ts, symbol, allow_asof=allow_asof
            )
            oos_val, oos_bar_ts, oos_status = _value_at_or_before(
                oos_features.get(key), signal_bar_ts, symbol, allow_asof=allow_asof
            )

            live_available_ts = _first_ts(drow, ["live_feature_available_ts", "feature_available_ts"])
            oos_available_ts = _first_ts(drow, ["oos_feature_available_ts", "feature_available_ts"])
            live_source_max_ts = _first_ts(drow, ["live_source_max_ts", "feature_source_max_ts"])
            oos_source_max_ts = _first_ts(drow, ["oos_source_max_ts", "feature_source_max_ts"])

            live_num = pd.to_numeric(pd.Series([live_val]), errors="coerce").iloc[0]
            oos_num = pd.to_numeric(pd.Series([oos_val]), errors="coerce").iloc[0]
            availability_unknown = not any(
                pd.notna(ts)
                for ts in (live_available_ts, oos_available_ts, live_source_max_ts, oos_source_max_ts)
            )
            feature_available_after_decision = any(
                pd.notna(ts) and pd.Timestamp(ts) > decision_ts
                for ts in (live_available_ts, oos_available_ts)
            )
            source_max_after_decision = any(
                pd.notna(ts) and pd.Timestamp(ts) > decision_ts
                for ts in (live_source_max_ts, oos_source_max_ts)
            )
            feature_bar_after_decision = any(
                pd.notna(ts) and pd.Timestamp(ts) > decision_ts
                for ts in (live_bar_ts, oos_bar_ts)
            )
            lookahead_violation = (
                feature_available_after_decision
                or source_max_after_decision
                or feature_bar_after_decision
            )

            if pd.isna(live_num) or pd.isna(oos_num):
                abs_diff = np.nan
                rel_diff = np.nan
                status = live_status if live_status != "ok" else oos_status
            else:
                abs_diff = float(abs(live_num - oos_num))
                rel_diff = float(abs_diff / max(abs(float(oos_num)), 1e-12))
                if lookahead_violation:
                    status = "lookahead_violation"
                elif abs_diff <= float(atol) + float(rtol) * abs(float(oos_num)):
                    status = "match_asof" if (live_status == "asof" or oos_status == "asof") else "match"
                else:
                    status = "mismatch"

            rows.append(
                {
                    "timestamp": decision_ts,
                    "decision_ts": decision_ts,
                    "signal_bar_ts": signal_bar_ts,
                    "symbol": symbol,
                    "side": decision.get("side", pd.NA),
                    "strategy_id": decision.get("strategy_id", pd.NA),
                    "run_id": decision.get("run_id", pd.NA),
                    "model_artifact_run_id": decision.get("model_artifact_run_id", pd.NA),
                    "feature_contract_hash": decision.get("feature_contract_hash", pd.NA),
                    "feature": key,
                    "live_value": live_num,
                    "oos_value": oos_num,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "live_feature_ts": live_bar_ts,
                    "oos_feature_ts": oos_bar_ts,
                    "live_feature_bar_ts": live_bar_ts,
                    "oos_feature_bar_ts": oos_bar_ts,
                    "live_feature_available_ts": live_available_ts,
                    "oos_feature_available_ts": oos_available_ts,
                    "live_source_max_ts": live_source_max_ts,
                    "oos_source_max_ts": oos_source_max_ts,
                    "feature_bar_after_decision": bool(feature_bar_after_decision),
                    "feature_available_after_decision": bool(feature_available_after_decision),
                    "source_max_after_decision": bool(source_max_after_decision),
                    "lookahead_violation": bool(lookahead_violation),
                    "availability_unknown": bool(availability_unknown),
                    "parity_status": status,
                }
            )
    return pd.DataFrame(rows, columns=FEATURE_PARITY_COLUMNS)


def summarize_feature_parity(report: pd.DataFrame) -> pd.DataFrame:
    """Summarize parity failures by feature."""
    if report is None or report.empty:
        return pd.DataFrame(
            columns=["feature", "rows", "matches", "mismatches", "missing", "lookahead", "max_abs_diff"]
        )
    df = report.copy()
    status = df["parity_status"].astype(str)
    df["is_match"] = status.isin({"match", "match_asof"})
    df["is_missing"] = status.str.startswith("missing")
    df["is_lookahead"] = df.get("lookahead_violation", False).astype(bool)
    df["is_mismatch"] = status.eq("mismatch")
    grouped = df.groupby("feature", dropna=False)
    out = grouped.agg(
        rows=("feature", "size"),
        matches=("is_match", "sum"),
        missing=("is_missing", "sum"),
        lookahead=("is_lookahead", "sum"),
        mismatches=("is_mismatch", "sum"),
        max_abs_diff=("abs_diff", "max"),
    ).reset_index()
    return out[["feature", "rows", "matches", "mismatches", "missing", "lookahead", "max_abs_diff"]]
