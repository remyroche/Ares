#!/usr/bin/env python3
"""Compare live prediction-ledger feature JSON against source feature matrices."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.data_store import load_features_selected  # noqa: E402


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _token(ts: pd.Timestamp) -> str:
    return _timestamp(ts).strftime("%Y%m%dT%H%M%SZ")


def _json_float_map(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in parsed.items():
        val = _safe_float(value)
        if math.isfinite(val):
            out[str(key)] = val
    return out


def _raw_json_map(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _logged_feature_quality(
    ledger: pd.DataFrame, layers: set[str], max_rows: int | None
) -> dict[str, Any]:
    work = ledger.tail(int(max_rows)) if max_rows and max_rows > 0 else ledger
    output: dict[str, Any] = {}
    for layer, column in (
        ("base", "base_model_feature_values_json"),
        ("meta", "meta_model_feature_values_json"),
    ):
        if layer not in layers:
            continue
        feature_values: dict[str, list[float]] = {}
        missing_payload_rows = 0
        nonfinite_cells = 0
        null_cells = 0
        for raw in work.get(column, pd.Series(index=work.index, dtype=object)):
            values = _raw_json_map(raw)
            if not values:
                missing_payload_rows += 1
                continue
            for feature, value in values.items():
                if value is None:
                    null_cells += 1
                    continue
                numeric = _safe_float(value)
                if not math.isfinite(numeric):
                    nonfinite_cells += 1
                    continue
                feature_values.setdefault(str(feature), []).append(numeric)
        all_zero = sorted(
            feature
            for feature, values in feature_values.items()
            if values and np.all(np.asarray(values, dtype=np.float64) == 0.0)
        )
        output[layer] = {
            "ledger_rows": int(len(work)),
            "features_observed": int(len(feature_values)),
            "finite_cells": int(sum(len(values) for values in feature_values.values())),
            "missing_payload_rows": int(missing_payload_rows),
            "null_cells": int(null_cells),
            "nonfinite_cells": int(nonfinite_cells),
            "all_zero_feature_count": int(len(all_zero)),
            "all_zero_features": all_zero[:100],
            "all_zero_note": (
                "Zeros can be valid for binary, sparse event, centered, and inactive "
                "regime features; this list is diagnostic, not an automatic failure."
            ),
        }
    return output


def _load_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_parquet(path)
    out = matrix.copy()
    for col in ("symbol", "__symbol__"):
        if col in out.columns:
            out = out.set_index(col)
            break
    drop_cols = [c for c in out.columns if str(c).startswith("__index_level_")]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out.index = pd.Index([str(v) for v in out.index], name="symbol")
    out.columns = [str(c) for c in out.columns]
    if not out.index.is_unique:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _sidecar_timestamps(sidecar_dir: Path) -> dict[pd.Timestamp, Path]:
    out: dict[pd.Timestamp, Path] = {}
    for path in sorted(sidecar_dir.glob("matrix_*.parquet")):
        match = re.match(r"matrix_(\d{8}T\d{6}Z)\.parquet$", path.name)
        if not match:
            continue
        out[_timestamp(pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%SZ", utc=True))] = path
    return out


def _feature_store_ts(run_id: str) -> pd.Timestamp:
    text = str(run_id)
    try:
        return pd.to_datetime(text, format="%Y%m%d_%H%M%S")
    except Exception as exc:
        raise SystemExit(f"Could not parse feature run id as timestamp: {run_id}") from exc


def _row_layer_values(row: pd.Series, layers: set[str]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    if "base" in layers:
        values["base"] = _json_float_map(row.get("base_model_feature_values_json"))
    if "meta" in layers:
        values["meta"] = _json_float_map(row.get("meta_model_feature_values_json"))
    return values


def _ledger_feature_keys(ledger: pd.DataFrame, layers: set[str]) -> set[str]:
    keys: set[str] = set()
    layer_columns = []
    if "base" in layers:
        layer_columns.append("base_model_feature_values_json")
    if "meta" in layers:
        layer_columns.append("meta_model_feature_values_json")
    for col in layer_columns:
        if col not in ledger.columns:
            continue
        for raw in ledger[col].dropna():
            keys.update(_json_float_map(raw).keys())
    return keys


def _compare_ledger_to_logical_store(
    ledger: pd.DataFrame,
    *,
    data_root: Path,
    feature_run_id: str,
    layers: set[str],
    tolerance: float,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if ledger.empty:
        summary = {
            "mode": "logical_feature_store",
            "ledger_rows": 0,
            "compared_rows": 0,
            "common_cells": 0,
            "mismatch_cells": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    work = ledger.copy()
    work["signal_bar_ts_utc"] = pd.to_datetime(work["signal_bar_ts"], utc=True)
    work = work.dropna(subset=["signal_bar_ts_utc", "symbol"])
    work = work.sort_values("signal_bar_ts_utc")
    if max_rows is not None and max_rows > 0:
        work = work.tail(int(max_rows))

    symbols = sorted(set(str(sym) for sym in work["symbol"].dropna()))
    ledger_keys = _ledger_feature_keys(work, layers)
    min_ts = _timestamp(work["signal_bar_ts_utc"].min())
    max_ts = _timestamp(work["signal_bar_ts_utc"].max())
    feats = load_features_selected(
        _feature_store_ts(feature_run_id),
        str(data_root),
        feature_keys=sorted(ledger_keys),
        symbols=symbols,
        start_ts=min_ts,
        end_ts=max_ts,
    )
    available_keys = set(str(k) for k in (feats.keys() if feats is not None else []))

    compare_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    compared_row_keys: set[tuple[int, str, pd.Timestamp]] = set()
    no_store_feature_cells = 0
    no_exact_value_cells = 0

    value_cache: dict[tuple[str, pd.Timestamp], pd.Series] = {}

    def source_value(feature: str, symbol: str, ts: pd.Timestamp) -> float:
        nonlocal no_exact_value_cells
        if feats is None or feature not in available_keys:
            return float("nan")
        key = (feature, ts)
        if key not in value_cache:
            try:
                value_cache[key] = feats.latest_values_at(
                    feature,
                    symbols,
                    ts,
                    stale_sensitive=True,
                )
            except Exception:
                value_cache[key] = pd.Series(dtype=np.float32)
        val = _safe_float(value_cache[key].get(symbol, np.nan))
        if not math.isfinite(val):
            no_exact_value_cells += 1
        return val

    for row_idx, row in work.iterrows():
        ts = _timestamp(row["signal_bar_ts_utc"])
        symbol = str(row["symbol"])
        row_key = (int(row_idx), symbol, ts)
        layer_values = _row_layer_values(row, layers)
        row_has_common = False
        for layer, values in layer_values.items():
            if not values:
                continue
            missing_features = sorted(set(values) - available_keys)
            no_store_feature_cells += len(missing_features)
            if missing_features:
                missing_rows.append(
                    {
                        "row_index": int(row_idx),
                        "signal_bar_ts": ts.isoformat(),
                        "symbol": symbol,
                        "layer": layer,
                        "reason": "features_missing_from_logical_store",
                        "missing_feature_count": len(missing_features),
                        "sample_missing_features": ",".join(missing_features[:20]),
                    }
                )
            for feature in sorted(set(values) & available_keys):
                source_val = source_value(feature, symbol, ts)
                ledger_val = _safe_float(values.get(feature))
                if not (math.isfinite(source_val) and math.isfinite(ledger_val)):
                    continue
                abs_diff = abs(source_val - ledger_val)
                compare_rows.append(
                    {
                        "row_index": int(row_idx),
                        "signal_bar_ts": ts.isoformat(),
                        "symbol": symbol,
                        "strategy_id": row.get("strategy_id"),
                        "layer": layer,
                        "feature": feature,
                        "ledger_value": ledger_val,
                        "logical_store_value": source_val,
                        "abs_diff": abs_diff,
                        "matches_tolerance": bool(abs_diff <= tolerance),
                    }
                )
                row_has_common = True
        if row_has_common:
            compared_row_keys.add(row_key)

    comparisons = pd.DataFrame(compare_rows)
    missing = pd.DataFrame(missing_rows)
    summary = {
        "mode": "logical_feature_store",
        "data_root": str(data_root),
        "feature_run_id": str(feature_run_id),
        "ledger_rows": int(len(work)),
        "ledger_signal_ts": int(work["signal_bar_ts_utc"].nunique()),
        "ledger_feature_keys": int(len(ledger_keys)),
        "logical_store_feature_keys": int(len(available_keys)),
        "missing_feature_cells": int(no_store_feature_cells),
        "missing_exact_value_cells": int(no_exact_value_cells),
        "compared_rows": int(len(compared_row_keys)),
        "common_cells": int(len(comparisons)),
        "mismatch_cells": int((comparisons["abs_diff"] > tolerance).sum()) if not comparisons.empty else 0,
        "max_abs_diff": float(comparisons["abs_diff"].max()) if not comparisons.empty else float("nan"),
        "mean_abs_diff": float(comparisons["abs_diff"].mean()) if not comparisons.empty else float("nan"),
        "tolerance": float(tolerance),
        "start_ts": min_ts.isoformat(),
        "end_ts": max_ts.isoformat(),
        "logged_feature_quality": _logged_feature_quality(work, layers, max_rows),
    }
    return comparisons, missing, summary


def _compare_ledger_to_sidecars(
    ledger: pd.DataFrame,
    sidecars: dict[pd.Timestamp, Path],
    *,
    layers: set[str],
    tolerance: float,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if ledger.empty:
        summary = {
            "ledger_rows": 0,
            "compared_rows": 0,
            "common_cells": 0,
            "mismatch_cells": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), summary

    work = ledger.copy()
    work["signal_bar_ts_utc"] = pd.to_datetime(work["signal_bar_ts"], utc=True)
    work = work.dropna(subset=["signal_bar_ts_utc", "symbol"])
    work = work.sort_values("signal_bar_ts_utc")
    if max_rows is not None and max_rows > 0:
        work = work.tail(int(max_rows))

    matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    compare_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    compared_row_keys: set[tuple[int, str, pd.Timestamp]] = set()
    missing_sidecar_rows = 0
    missing_symbol_rows = 0

    for row_idx, row in work.iterrows():
        ts = _timestamp(row["signal_bar_ts_utc"])
        symbol = str(row["symbol"])
        matrix_path = sidecars.get(ts)
        row_key = (int(row_idx), symbol, ts)
        if matrix_path is None:
            missing_sidecar_rows += 1
            missing_rows.append(
                {
                    "row_index": int(row_idx),
                    "signal_bar_ts": ts.isoformat(),
                    "symbol": symbol,
                    "reason": "missing_exact_sidecar",
                    "missing_feature_count": np.nan,
                    "sample_missing_features": "",
                }
            )
            continue
        if ts not in matrices:
            matrices[ts] = _load_matrix(matrix_path)
        matrix = matrices[ts]
        if symbol not in matrix.index:
            missing_symbol_rows += 1
            missing_rows.append(
                {
                    "row_index": int(row_idx),
                    "signal_bar_ts": ts.isoformat(),
                    "symbol": symbol,
                    "reason": "symbol_missing_from_sidecar",
                    "missing_feature_count": np.nan,
                    "sample_missing_features": "",
                }
            )
            continue

        source_row = matrix.loc[symbol]
        source_cols = set(str(c) for c in matrix.columns)
        layer_values = _row_layer_values(row, layers)
        row_has_common = False
        for layer, values in layer_values.items():
            if not values:
                continue
            missing_features = sorted(set(values) - source_cols)
            if missing_features:
                missing_rows.append(
                    {
                        "row_index": int(row_idx),
                        "signal_bar_ts": ts.isoformat(),
                        "symbol": symbol,
                        "layer": layer,
                        "reason": "features_missing_from_sidecar",
                        "missing_feature_count": len(missing_features),
                        "sample_missing_features": ",".join(missing_features[:20]),
                    }
                )
            for feature in sorted(set(values) & source_cols):
                source_val = _safe_float(source_row.get(feature))
                ledger_val = _safe_float(values.get(feature))
                if not (math.isfinite(source_val) and math.isfinite(ledger_val)):
                    continue
                abs_diff = abs(source_val - ledger_val)
                compare_rows.append(
                    {
                        "row_index": int(row_idx),
                        "signal_bar_ts": ts.isoformat(),
                        "symbol": symbol,
                        "strategy_id": row.get("strategy_id"),
                        "layer": layer,
                        "feature": feature,
                        "ledger_value": ledger_val,
                        "sidecar_value": source_val,
                        "abs_diff": abs_diff,
                        "matches_tolerance": bool(abs_diff <= tolerance),
                    }
                )
                row_has_common = True
        if row_has_common:
            compared_row_keys.add(row_key)

    comparisons = pd.DataFrame(compare_rows)
    missing = pd.DataFrame(missing_rows)
    ledger_ts = set(_timestamp(v) for v in work["signal_bar_ts_utc"].dropna().unique())
    overlap_ts = sorted(ledger_ts.intersection(sidecars.keys()))
    summary = {
        "mode": "live_latest_sidecar",
        "ledger_rows": int(len(work)),
        "ledger_signal_ts": int(len(ledger_ts)),
        "sidecar_signal_ts": int(len(sidecars)),
        "exact_signal_ts_overlap": int(len(overlap_ts)),
        "overlap_timestamps": [ts.isoformat() for ts in overlap_ts],
        "missing_exact_sidecar_rows": int(missing_sidecar_rows),
        "missing_symbol_rows": int(missing_symbol_rows),
        "compared_rows": int(len(compared_row_keys)),
        "common_cells": int(len(comparisons)),
        "mismatch_cells": int((comparisons["abs_diff"] > tolerance).sum()) if not comparisons.empty else 0,
        "max_abs_diff": float(comparisons["abs_diff"].max()) if not comparisons.empty else float("nan"),
        "mean_abs_diff": float(comparisons["abs_diff"].mean()) if not comparisons.empty else float("nan"),
        "tolerance": float(tolerance),
        "logged_feature_quality": _logged_feature_quality(work, layers, max_rows),
    }
    return comparisons, missing, summary


def _write_outputs(
    *,
    output_dir: Path | None,
    comparisons: pd.DataFrame,
    missing: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
    comparisons.to_csv(output_dir / "feature_value_comparisons.csv", index=False)
    missing.to_csv(output_dir / "missing_feature_or_sidecar_rows.csv", index=False)


def _parse_layers(raw: str) -> set[str]:
    layers = {part.strip().lower() for part in str(raw or "").split(",") if part.strip()}
    bad = layers - {"base", "meta"}
    if bad:
        raise argparse.ArgumentTypeError(f"invalid layer(s): {sorted(bad)}")
    return layers or {"base", "meta"}


def _coverage_gate_failures(
    summary: dict[str, Any],
    *,
    min_compared_rows: int,
    min_common_cells: int,
    require_complete_sidecar_coverage: bool,
) -> list[str]:
    failures: list[str] = []
    compared_rows = int(summary.get("compared_rows", 0) or 0)
    common_cells = int(summary.get("common_cells", 0) or 0)
    if compared_rows <= 0:
        failures.append("no_compared_rows")
    elif compared_rows < int(min_compared_rows):
        failures.append(
            f"compared_rows={compared_rows}<min_compared_rows={int(min_compared_rows)}"
        )
    if common_cells <= 0:
        failures.append("no_common_cells")
    elif common_cells < int(min_common_cells):
        failures.append(
            f"common_cells={common_cells}<min_common_cells={int(min_common_cells)}"
        )
    if require_complete_sidecar_coverage:
        ledger_rows = int(summary.get("ledger_rows", 0) or 0)
        missing_exact = int(summary.get("missing_exact_sidecar_rows", 0) or 0)
        missing_symbols = int(summary.get("missing_symbol_rows", 0) or 0)
        if compared_rows != ledger_rows or missing_exact or missing_symbols:
            failures.append(
                "incomplete_exact_sidecar_coverage:"
                f"compared={compared_rows} ledger={ledger_rows} "
                f"missing_sidecar={missing_exact} missing_symbol={missing_symbols}"
            )
    return failures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--sidecar-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--feature-run-id")
    parser.add_argument("--layers", default="base,meta", type=_parse_layers)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-compared-rows", type=int, default=1)
    parser.add_argument("--min-common-cells", type=int, default=1)
    parser.add_argument(
        "--require-complete-sidecar-coverage",
        action="store_true",
        help="Fail unless every audited ledger row has an exact timestamp/symbol sidecar.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.ledger.exists():
        raise SystemExit(f"Ledger not found: {args.ledger}")

    ledger = pd.read_parquet(args.ledger)
    if args.feature_run_id:
        if args.data_root is None:
            raise SystemExit("--data-root is required with --feature-run-id")
        comparisons, missing, summary = _compare_ledger_to_logical_store(
            ledger,
            data_root=args.data_root,
            feature_run_id=args.feature_run_id,
            layers=args.layers,
            tolerance=float(args.tolerance),
            max_rows=args.max_rows if args.max_rows > 0 else None,
        )
    else:
        if args.sidecar_dir is None:
            raise SystemExit("Pass either --feature-run-id with --data-root, or --sidecar-dir")
        if not args.sidecar_dir.exists():
            raise SystemExit(f"Sidecar dir not found: {args.sidecar_dir}")
        sidecars = _sidecar_timestamps(args.sidecar_dir)
        comparisons, missing, summary = _compare_ledger_to_sidecars(
            ledger,
            sidecars,
            layers=args.layers,
            tolerance=float(args.tolerance),
            max_rows=args.max_rows if args.max_rows > 0 else None,
        )
    coverage_failures = _coverage_gate_failures(
        summary,
        min_compared_rows=max(0, int(args.min_compared_rows)),
        min_common_cells=max(0, int(args.min_common_cells)),
        require_complete_sidecar_coverage=bool(
            args.require_complete_sidecar_coverage
        ),
    )
    mismatch_cells = int(summary.get("mismatch_cells", 0) or 0)
    summary["coverage_gate_failures"] = coverage_failures
    summary["parity_gate_pass"] = bool(not coverage_failures and mismatch_cells == 0)
    _write_outputs(
        output_dir=args.output_dir,
        comparisons=comparisons,
        missing=missing,
        summary=summary,
    )

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if not comparisons.empty:
        worst = comparisons.sort_values("abs_diff", ascending=False).head(10)
        print(worst.to_string(index=False))
    if not missing.empty:
        print(missing.head(20).to_string(index=False))
    return 1 if not summary["parity_gate_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
