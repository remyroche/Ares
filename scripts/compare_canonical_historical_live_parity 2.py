#!/usr/bin/env python3
"""Compare canonical historical scoring with one persisted live inference batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["signal_ts", "symbol", "side_name"]


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _json_mapping(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for name, item in raw.items():
        try:
            out[str(name)] = float(item)
        except (TypeError, ValueError):
            continue
    return out


def _live_batch(
    path: Path,
    signal_ts: pd.Timestamp,
    *,
    decision_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["signal_ts"] = _utc(frame["signal_bar_ts"])
    frame = frame.loc[frame["signal_ts"].eq(signal_ts)].copy()
    frame["decision_ts"] = _utc(frame["decision_ts"])
    if frame.empty:
        return frame
    batch_sizes = frame.groupby("decision_ts", sort=True).size()
    if decision_ts is None:
        # Retries for the same signal bar can contain partial universes. Compare
        # one coherent inference invocation, never a per-symbol union of retries.
        max_rows = int(batch_sizes.max())
        decision_ts = batch_sizes.loc[batch_sizes.eq(max_rows)].index.max()
    frame = frame.loc[frame["decision_ts"].eq(decision_ts)].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["side_name"] = frame["side"].astype(str).str.lower()
    result = (
        frame.sort_values("decision_ts", kind="stable")
        .drop_duplicates(KEYS, keep="last")
        .reset_index(drop=True)
    )
    result.attrs["selected_decision_ts"] = decision_ts.isoformat()
    result.attrs["available_batches"] = {
        ts.isoformat(): int(rows) for ts, rows in batch_sizes.items()
    }
    return result


def _historical_rows(predictions: Path, signal_ts: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(predictions)
    frame["signal_ts"] = _utc(frame["__ts__"])
    frame["symbol"] = frame["__symbol__"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    return frame.loc[frame["signal_ts"].eq(signal_ts)].reset_index(drop=True)


def _input_comparison(
    merged: pd.DataFrame,
    *,
    historical_input: pd.DataFrame,
    prefix: str,
    live_json_col: str,
    layer: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [name for name in historical_input.columns if name.startswith(prefix)]
    historical_input = historical_input.copy()
    historical_input["signal_ts"] = _utc(historical_input["__ts__"])
    historical_input["symbol"] = historical_input["__symbol__"].astype(str)
    historical_input["side_name"] = historical_input["side_name"].astype(str).str.lower()
    keyed = historical_input.set_index(KEYS)
    rows: list[dict[str, Any]] = []
    for live in merged.itertuples(index=False):
        key = tuple(getattr(live, name) for name in KEYS)
        if key not in keyed.index:
            continue
        historical = keyed.loc[key]
        if isinstance(historical, pd.DataFrame):
            historical = historical.iloc[-1]
        live_values = _json_mapping(getattr(live, live_json_col, "{}"))
        for name in columns:
            feature = name[len(prefix) :]
            replay_value = pd.to_numeric(pd.Series([historical[name]]), errors="coerce").iloc[0]
            live_value = live_values.get(feature, np.nan)
            delta = float(replay_value - live_value) if np.isfinite(replay_value) and np.isfinite(live_value) else np.nan
            rows.append(
                {
                    **dict(zip(KEYS, key)),
                    "layer": layer,
                    "feature": feature,
                    "historical_value": replay_value,
                    "live_value": live_value,
                    "delta": delta,
                    "abs_delta": abs(delta) if np.isfinite(delta) else np.nan,
                    "both_finite": bool(np.isfinite(replay_value) and np.isfinite(live_value)),
                }
            )
    detail = pd.DataFrame(rows)
    finite = detail.loc[detail.get("both_finite", False)].copy() if not detail.empty else detail
    summary = {
        "layer": layer,
        "features": int(detail["feature"].nunique()) if not detail.empty else 0,
        "comparisons": int(len(detail)),
        "finite_comparisons": int(len(finite)),
        "missing_comparisons": int(len(detail) - len(finite)),
        "mean_abs_delta": float(finite["abs_delta"].mean()) if not finite.empty else None,
        "max_abs_delta": float(finite["abs_delta"].max()) if not finite.empty else None,
    }
    return detail, summary


def _score_comparison(
    merged: pd.DataFrame,
    *,
    layer: str,
    historical_col: str,
    live_col: str,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if historical_col not in merged or live_col not in merged:
        return pd.DataFrame(), {
            "layer": layer,
            "status": "missing_column",
            "historical_col": historical_col,
            "live_col": live_col,
        }
    replay = pd.to_numeric(merged[historical_col], errors="coerce")
    live = pd.to_numeric(merged[live_col], errors="coerce")
    finite = np.isfinite(replay) & np.isfinite(live)
    detail = merged.loc[finite, KEYS].copy()
    detail["layer"] = layer
    detail["historical_value"] = replay.loc[finite].to_numpy()
    detail["live_value"] = live.loc[finite].to_numpy()
    detail["delta"] = detail["historical_value"] - detail["live_value"]
    detail["abs_delta"] = detail["delta"].abs()
    max_delta = float(detail["abs_delta"].max()) if not detail.empty else np.nan
    return detail, {
        "layer": layer,
        "historical_col": historical_col,
        "live_col": live_col,
        "rows": int(len(detail)),
        "mean_abs_delta": float(detail["abs_delta"].mean()) if not detail.empty else None,
        "max_abs_delta": max_delta if np.isfinite(max_delta) else None,
        "tolerance": tolerance,
        "pass": bool(np.isfinite(max_delta) and max_delta <= tolerance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--historical-model-input", type=Path, required=True)
    parser.add_argument("--live-ledger", type=Path, required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument(
        "--live-decision-ts",
        default=None,
        help=(
            "Exact live inference decision timestamp to compare. By default, "
            "the latest maximum-size retry cohort is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-tolerance", type=float, default=1e-6)
    parser.add_argument("--score-tolerance", type=float, default=2e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal_ts = pd.Timestamp(args.signal_ts)
    signal_ts = signal_ts.tz_localize("UTC") if signal_ts.tzinfo is None else signal_ts.tz_convert("UTC")
    historical = _historical_rows(args.historical_predictions, signal_ts)
    model_input = pd.read_parquet(args.historical_model_input)
    historical_scores = model_input.copy()
    historical_scores["signal_ts"] = _utc(historical_scores["__ts__"])
    historical_scores["symbol"] = historical_scores["__symbol__"].astype(str)
    historical_scores["side_name"] = (
        historical_scores["side_name"].astype(str).str.lower()
    )
    score_columns = [
        name
        for name in ("score", "score_meta_base_soft_label")
        if name in historical_scores.columns
    ]
    historical = historical.merge(
        historical_scores[KEYS + score_columns],
        on=KEYS,
        how="left",
        suffixes=("", "__model_input"),
        validate="one_to_one",
    )
    live_decision_ts = None
    if args.live_decision_ts:
        live_decision_ts = pd.Timestamp(args.live_decision_ts)
        live_decision_ts = (
            live_decision_ts.tz_localize("UTC")
            if live_decision_ts.tzinfo is None
            else live_decision_ts.tz_convert("UTC")
        )
    live = _live_batch(
        args.live_ledger,
        signal_ts,
        decision_ts=live_decision_ts,
    )
    selected_live_decision_ts = live.attrs.get("selected_decision_ts")
    available_live_batches = live.attrs.get("available_batches", {})
    merged = historical.merge(live, on=KEYS, how="outer", suffixes=("__historical", "__live"), indicator=True)
    overlap = merged.loc[merged["_merge"].eq("both")].copy()

    input_specs = (
        ("base_input__", "base_model_feature_values_json", "base_model_input"),
        ("meta_input__", "meta_model_feature_values_json", "meta_model_input"),
    )
    input_details: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for prefix, live_json_col, layer in input_specs:
        detail, summary = _input_comparison(
            overlap,
            historical_input=model_input,
            prefix=prefix,
            live_json_col=live_json_col,
            layer=layer,
        )
        input_details.append(detail)
        summary["tolerance"] = args.input_tolerance
        summary["pass"] = bool(
            summary["max_abs_delta"] is not None
            and summary["max_abs_delta"] <= args.input_tolerance
            and summary["missing_comparisons"] == 0
        )
        summaries.append(summary)

    score_specs = (
        ("base_prediction", "score", "base_pred"),
        ("meta_prediction", "score_meta_base_soft_label", "meta_pred_aligned"),
        ("v9_tail95_rank", "historical_rank", "v9_tail95_predecessor_rank"),
        (
            "mlp_hierarchical_ev",
            "expected_net_ev_after_1pct",
            "market_state_mlp_expected_net_ev_after_1pct",
        ),
        (
            "mapped_side_archetype_ev",
            "threshold_basis_mapped_expected_ev_side_archetype",
            "threshold_basis_mapped_expected_ev_side_archetype",
        ),
        (
            "recent_ev_correction",
            "threshold_basis_side_archetype_recent_ev_correction",
            "threshold_basis_side_archetype_recent_ev_correction",
        ),
        (
            "corrected_expected_ev",
            "threshold_basis_corrected_expected_ev",
            "threshold_basis_corrected_expected_ev",
        ),
        (
            "corrected_expected_ev_rank",
            "threshold_basis_corrected_expected_ev_rank",
            "threshold_basis_corrected_expected_ev_rank",
        ),
    )
    score_details: list[pd.DataFrame] = []
    for layer, historical_col, live_col in score_specs:
        historical_name = historical_col + "__historical" if historical_col in live.columns else historical_col
        live_name = live_col + "__live" if live_col in historical.columns else live_col
        detail, summary = _score_comparison(
            overlap,
            layer=layer,
            historical_col=historical_name,
            live_col=live_name,
            tolerance=args.score_tolerance,
        )
        score_details.append(detail)
        summaries.append(summary)

    historical_selected = pd.Series(
        historical.get("threshold_basis_selected", False), index=historical.index
    ).fillna(False).astype(bool)
    live_selected = pd.Series(
        live.get("threshold_basis_selected", False), index=live.index
    ).fillna(False).astype(bool)
    historical_decisions = set(map(tuple, historical.loc[historical_selected, KEYS].itertuples(index=False, name=None)))
    live_decisions = set(map(tuple, live.loc[live_selected, KEYS].itertuples(index=False, name=None)))
    decision_summary = {
        "layer": "admission_decision",
        "historical_selected": len(historical_decisions),
        "live_selected": len(live_decisions),
        "historical_only": len(historical_decisions - live_decisions),
        "live_only": len(live_decisions - historical_decisions),
        "pass": historical_decisions == live_decisions,
    }
    summaries.append(decision_summary)

    ordered_layers = [summary["layer"] for summary in summaries]
    first_divergence = next(
        (summary["layer"] for summary in summaries if not summary.get("pass", False)),
        None,
    )
    report = {
        "schema": "canonical_historical_live_parity_v1",
        "signal_ts": signal_ts.isoformat(),
        "selected_live_decision_ts": selected_live_decision_ts,
        "available_live_batches": available_live_batches,
        "historical_rows": int(len(historical)),
        "live_rows": int(len(live)),
        "overlap_rows": int(len(overlap)),
        "historical_only_rows": int((merged["_merge"] == "left_only").sum()),
        "live_only_rows": int((merged["_merge"] == "right_only").sum()),
        "layer_order": ordered_layers,
        "first_divergence": first_divergence,
        "strict_parity": bool(first_divergence is None and len(historical) == len(live) == len(overlap)),
        "layers": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(input_details, ignore_index=True).to_parquet(
        args.output_dir / "input_feature_deltas.parquet", index=False, compression="zstd"
    )
    pd.concat(score_details, ignore_index=True).to_parquet(
        args.output_dir / "score_deltas.parquet", index=False, compression="zstd"
    )
    merged[KEYS + ["_merge"]].to_csv(args.output_dir / "row_alignment.csv", index=False)
    (args.output_dir / "parity_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["strict_parity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
