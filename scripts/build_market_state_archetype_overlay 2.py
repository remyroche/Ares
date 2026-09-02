#!/usr/bin/env python3
"""Build a side x archetype market-state overlay calibration artifact.

The output JSON is compatible with
``extreme_price_movements.regime_ev_calibration.apply_regime_ev_calibration``.
It is intended to be consumed by replay and live inference as a frozen overlay,
not as an in-sample hard gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.market_state_archetype_overlay import (
    DEFAULT_MARKET_STATE_PREFIXES,
    DEFAULT_STOP_COLUMNS,
    DEFAULT_TIMEOUT_COLUMNS,
    MarketStateOverlayConfig,
    fit_market_state_archetype_overlay,
    resolve_archetype_column,
    resolve_outcome_column,
    select_market_state_columns,
    topk_precision_metrics,
)
from extreme_price_movements.regime_ev_calibration import apply_regime_ev_calibration


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


def _downcast_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype(np.float32, copy=False)
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def _parse_ts(value: str) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _time_mask(frame: pd.DataFrame, ts_col: str, *, start: str = "", end: str = "") -> np.ndarray:
    ts = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    mask = ts.notna().to_numpy(dtype=bool, copy=False)
    start_ts = _parse_ts(start)
    end_ts = _parse_ts(end)
    if start_ts is not None:
        mask &= ts.ge(start_ts).to_numpy(dtype=bool, copy=False)
    if end_ts is not None:
        mask &= ts.lt(end_ts).to_numpy(dtype=bool, copy=False)
    return mask


def _time_spread_sample(frame: pd.DataFrame, ts_col: str, max_rows: int) -> pd.DataFrame:
    if int(max_rows) <= 0 or len(frame) <= int(max_rows):
        return frame
    ts = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    take_pos = np.linspace(0, len(order) - 1, int(max_rows)).round().astype(np.int64)
    take = np.unique(order[take_pos])
    return frame.iloc[np.sort(take)]


def _read_candidate_frame(
    path: Path,
    *,
    feature_cols: list[str],
    required_cols: list[str],
    prefixes: tuple[str, ...],
    max_feature_cols: int,
) -> tuple[pd.DataFrame, list[str]]:
    schema_cols = _parquet_columns(path)
    schema_set = set(schema_cols)
    if not feature_cols:
        feature_cols = [
            col
            for col in schema_cols
            if col.startswith(prefixes) and col not in set(required_cols)
        ]
    feature_cols = [col for col in feature_cols if col in schema_set]
    if max_feature_cols and len(feature_cols) > int(max_feature_cols):
        feature_cols = feature_cols[: int(max_feature_cols)]
    read_cols = list(dict.fromkeys([col for col in [*required_cols, *feature_cols] if col in schema_set]))
    if not read_cols:
        raise ValueError(f"No readable columns resolved for {path}")
    table = pq.read_table(path, columns=read_cols)
    frame = table.to_pandas()
    return _downcast_numeric(frame), feature_cols


def _join_outcomes(
    frame: pd.DataFrame,
    outcomes_path: Path,
    *,
    join_key: str,
    outcome_return_col: str,
    output_return_col: str,
) -> pd.DataFrame:
    join_keys = [key.strip() for key in str(join_key).split(",") if key.strip()]
    if not join_keys:
        raise ValueError("join_key must contain at least one column")
    if join_keys == ["candidate_index"] and "candidate_index" not in frame.columns:
        frame["candidate_index"] = np.arange(len(frame), dtype=np.int64)
    missing_frame_keys = [key for key in join_keys if key not in frame.columns]
    if missing_frame_keys:
        raise ValueError(f"candidate frame missing outcome join keys: {missing_frame_keys}")
    schema_cols = set(_parquet_columns(outcomes_path))
    missing_outcome_keys = [key for key in join_keys if key not in schema_cols]
    if missing_outcome_keys:
        raise ValueError(f"outcomes frame missing join keys: {missing_outcome_keys}")
    read_cols = list(join_keys)
    for col in (outcome_return_col, "position_exit_reason", "accepted"):
        if col in schema_cols:
            read_cols.append(col)
    outcomes = pq.read_table(outcomes_path, columns=list(dict.fromkeys(read_cols))).to_pandas()
    if outcome_return_col not in outcomes.columns:
        raise ValueError(f"outcome return column missing in {outcomes_path}: {outcome_return_col}")
    rename = {outcome_return_col: output_return_col}
    outcomes = outcomes.rename(columns=rename)
    outcomes[output_return_col] = pd.to_numeric(outcomes[output_return_col], errors="coerce").astype(
        np.float32,
        copy=False,
    )
    if "position_exit_reason" in outcomes.columns:
        reason = outcomes["position_exit_reason"].astype(str).str.lower()
        outcomes["full_sl"] = reason.str.contains("stop", regex=False).astype(np.int8)
        outcomes["timeout"] = reason.str.contains("timeout", regex=False).astype(np.int8)
    keep_cols = [col for col in [*join_keys, output_return_col, "full_sl", "timeout"] if col in outcomes.columns]
    outcomes = outcomes.loc[:, keep_cols].drop_duplicates(subset=join_keys, keep="last")
    for key in join_keys:
        if "timestamp" in key or key.endswith("_ts"):
            frame[key] = pd.to_datetime(frame[key], utc=True, errors="coerce")
            outcomes[key] = pd.to_datetime(outcomes[key], utc=True, errors="coerce")
    merged = frame.merge(outcomes, on=join_keys, how="left", copy=False)
    return _downcast_numeric(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-start", default="")
    parser.add_argument("--train-end", default="")
    parser.add_argument("--eval-start", default="")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument("--archetype-col", default="policy_archetype")
    parser.add_argument("--outcome-col", default="")
    parser.add_argument("--outcomes", type=Path, default=None)
    parser.add_argument("--join-key", default="candidate_index")
    parser.add_argument("--outcome-return-col", default="position_net_return")
    parser.add_argument("--source-score-col", default="rank_pct")
    parser.add_argument("--feature-col", action="append", default=[])
    parser.add_argument("--feature-prefix", action="append", default=[])
    parser.add_argument("--max-feature-cols", type=int, default=0)
    parser.add_argument("--max-fit-rows", type=int, default=0)
    parser.add_argument("--n-buckets", type=int, default=5)
    parser.add_argument("--min-group-rows", type=int, default=80)
    parser.add_argument("--min-bucket-rows", type=int, default=15)
    parser.add_argument("--max-features-per-group", type=int, default=8)
    parser.add_argument("--min-abs-effect", type=float, default=0.0025)
    parser.add_argument("--max-abs-effect", type=float, default=0.04)
    parser.add_argument("--risk-cap", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(args.feature_prefix or DEFAULT_MARKET_STATE_PREFIXES)
    schema_cols = set(_parquet_columns(args.candidates))
    outcome_col = args.outcome_col if args.outcome_col else ""
    if not outcome_col:
        for candidate in (
            "ret_net_notional",
            "net_return_notional",
            "net_return",
            "ret_net",
            "net_pnl_pct",
        ):
            if candidate in schema_cols:
                outcome_col = candidate
                break
    join_keys = [key.strip() for key in str(args.join_key).split(",") if key.strip()]
    required_cols = [
        args.timestamp_col,
        args.side_col,
        args.archetype_col,
        *join_keys,
        "archetype_policy_key",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
        outcome_col,
        args.source_score_col,
        *DEFAULT_STOP_COLUMNS,
        *DEFAULT_TIMEOUT_COLUMNS,
    ]
    required_cols = [col for col in dict.fromkeys(required_cols) if col and col in schema_cols]
    frame, feature_cols = _read_candidate_frame(
        args.candidates,
        feature_cols=[str(c) for c in args.feature_col],
        required_cols=required_cols,
        prefixes=prefixes,
        max_feature_cols=int(args.max_feature_cols),
    )
    if args.outcomes is not None:
        joined_outcome_col = args.outcome_col or "ret_net_notional"
        frame = _join_outcomes(
            frame,
            args.outcomes,
            join_key=str(args.join_key),
            outcome_return_col=str(args.outcome_return_col),
            output_return_col=str(joined_outcome_col),
        )
        outcome_col = joined_outcome_col
    if args.timestamp_col not in frame.columns:
        raise ValueError(f"timestamp column missing: {args.timestamp_col}")
    frame[args.timestamp_col] = pd.to_datetime(frame[args.timestamp_col], utc=True, errors="coerce")
    arch_col = resolve_archetype_column(frame, args.archetype_col)
    outcome = resolve_outcome_column(frame, outcome_col or None)
    if "archetype_policy_key" not in frame.columns:
        frame["archetype_policy_key"] = frame[arch_col].astype(str)
    if arch_col != "archetype_policy_key":
        frame["archetype_policy_key"] = frame[arch_col].astype(str)
    feature_cols = select_market_state_columns(
        frame,
        include_prefixes=prefixes,
        required_columns=[
            args.timestamp_col,
            args.side_col,
            arch_col,
            outcome,
            args.source_score_col,
        ],
        max_columns=int(args.max_feature_cols),
    )
    finite_outcome = np.isfinite(pd.to_numeric(frame[outcome], errors="coerce").to_numpy(dtype=np.float32, copy=False))
    train_mask = _time_mask(frame, args.timestamp_col, start=args.train_start, end=args.train_end) & finite_outcome
    train = frame.loc[train_mask]
    train = _time_spread_sample(train, args.timestamp_col, int(args.max_fit_rows))
    cfg = MarketStateOverlayConfig(
        side_col=args.side_col,
        archetype_col=arch_col,
        timestamp_col=args.timestamp_col,
        source_score_col=args.source_score_col,
        n_buckets=int(args.n_buckets),
        min_group_rows=int(args.min_group_rows),
        min_bucket_rows=int(args.min_bucket_rows),
        max_features_per_group=int(args.max_features_per_group),
        min_abs_effect=float(args.min_abs_effect),
        max_abs_effect=float(args.max_abs_effect),
        risk_cap=float(args.risk_cap),
    )
    result = fit_market_state_archetype_overlay(
        train,
        feature_columns=feature_cols,
        outcome_col=outcome,
        config=cfg,
        valid_from=args.eval_start,
        valid_to=args.eval_end,
    )
    artifact_path = args.out_dir / "market_state_archetype_overlay_regime_ev_calibration.json"
    artifact_path.write_text(json.dumps(result.artifact, indent=2, sort_keys=True, default=_json_default))
    result.effect_metrics.to_csv(args.out_dir / "market_state_effect_bucket_metrics.csv", index=False)
    result.group_metrics.to_csv(args.out_dir / "market_state_group_baselines.csv", index=False)

    eval_mask = _time_mask(frame, args.timestamp_col, start=args.eval_start, end=args.eval_end)
    eval_frame = frame.loc[eval_mask].copy()
    before_after: dict[str, Any] = {}
    if not eval_frame.empty and args.source_score_col in eval_frame.columns:
        adjusted = apply_regime_ev_calibration(
            eval_frame,
            result.artifact,
            source_score_col=args.source_score_col,
            side_col=args.side_col,
            archetype_col="archetype_policy_key",
            copy=True,
        )
        overall_before = topk_precision_metrics(
            eval_frame,
            score_col=args.source_score_col,
            outcome_col=outcome,
        )
        overall_before["score_version"] = "before"
        overall_after = topk_precision_metrics(
            adjusted,
            score_col=str(result.artifact["adjusted_score_col"]),
            outcome_col=outcome,
        )
        overall_after["score_version"] = "after"
        by_arch_before = topk_precision_metrics(
            eval_frame,
            score_col=args.source_score_col,
            outcome_col=outcome,
            group_cols=[args.side_col, "archetype_policy_key"],
        )
        by_arch_before["score_version"] = "before"
        by_arch_after = topk_precision_metrics(
            adjusted,
            score_col=str(result.artifact["adjusted_score_col"]),
            outcome_col=outcome,
            group_cols=[args.side_col, "archetype_policy_key"],
        )
        by_arch_after["score_version"] = "after"
        pd.concat([overall_before, overall_after], ignore_index=True).to_csv(
            args.out_dir / "market_state_eval_topk_overall.csv",
            index=False,
        )
        pd.concat([by_arch_before, by_arch_after], ignore_index=True).to_csv(
            args.out_dir / "market_state_eval_topk_side_archetype.csv",
            index=False,
        )
        before_after = {
            "eval_rows": int(len(eval_frame)),
            "adjusted_score_col": str(result.artifact["adjusted_score_col"]),
            "risk_nonzero_share": float(
                pd.to_numeric(adjusted[str(result.artifact["effect_count_col"])], errors="coerce")
                .fillna(0)
                .gt(0)
                .mean()
            ),
        }
    summary = {
        "candidates": str(args.candidates),
        "artifact_path": str(artifact_path),
        "train_rows": int(len(train)),
        "train_start": str(pd.to_datetime(train[args.timestamp_col], utc=True, errors="coerce").min())
        if not train.empty
        else "",
        "train_end": str(pd.to_datetime(train[args.timestamp_col], utc=True, errors="coerce").max())
        if not train.empty
        else "",
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "feature_count": int(len(feature_cols)),
        "effect_count": int(len(result.artifact.get("effects") or [])),
        "group_count": int(len(result.group_metrics)),
        "outcome_col": outcome,
        "source_score_col": args.source_score_col,
        "numba_bucket_stats": bool(result.artifact.get("numba_bucket_stats")),
        "eval": before_after,
    }
    (args.out_dir / "market_state_overlay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
    )
    print(json.dumps(summary, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
