#!/usr/bin/env python3
"""Frozen action-file replay for S52 meta-threshold handoffs.

This audit starts from clean materialized action files and joins realized S52
trailing-policy outcomes only after the decisions are fixed.  It is a replay of
the frozen candidate file against the scored execution ledger, not a selector
rerun and not threshold optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_meta_handoff_gate3_readiness import OUTCOME_COLUMNS


DEFAULT_HANDOFF_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1"
)
DEFAULT_SCORED_LEDGER = DEFAULT_HANDOFF_ROOT / "s52_trailing_regime_scored_ledger.parquet"
DEFAULT_VARIANTS = (
    "s52_meta_threshold_current_top5_sidecap80_v2",
    "s52_meta_threshold_current_top10_sidecap80_v2",
)
LEDGER_OUTCOME_COLUMNS = (
    "u_policy_net",
    "ret_net",
    "exec_margin",
    "first_touch_net",
    "first_touch_gross",
    "ev_after_1pct",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "clean_exec",
    "dirty_positive",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "mae_norm",
    "mfe_norm",
    "first_touch_mae_norm",
    "first_touch_full_path_mae_norm",
    "underwater_bars_before_mfe_1r",
)
GROUPING_SPECS = {
    "month": ("month",),
    "side": ("side_name",),
    "month_side": ("month", "side_name"),
    "source": ("source_semantic_family",),
    "side_source": ("side_name", "source_semantic_family"),
    "side_aegmm": ("side_name", "aegmm_cluster"),
    "side_side_aegmm": ("side_name", "side_aegmm_cluster"),
    "side_reconstruction": ("side_name", "reconstruction_bin"),
    "side_leaf_exec_margin": ("side_name", "regime_lgbm_leaf_exec_margin_k4"),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _sum(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.sum()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _worst_group_mean(frame: pd.DataFrame, group_col: str, value_col: str) -> float:
    if frame.empty or group_col not in frame.columns or value_col not in frame.columns:
        return float("nan")
    vals = frame.groupby(group_col, dropna=False)[value_col].apply(_mean)
    return float(vals.min()) if len(vals) else float("nan")


def _max_group_rate(frame: pd.DataFrame, group_col: str, value_col: str) -> float:
    if frame.empty or group_col not in frame.columns or value_col not in frame.columns:
        return float("nan")
    vals = frame.groupby(group_col, dropna=False)[value_col].apply(_rate)
    return float(vals.max()) if len(vals) else float("nan")


def _dominant_side_share(frame: pd.DataFrame) -> float:
    if frame.empty or "side_name" not in frame.columns:
        return float("nan")
    vc = frame["side_name"].astype(str).str.lower().value_counts(normalize=True)
    return float(vc.iloc[0]) if len(vc) else float("nan")


def _key_timestamp(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce", utc=True)
    return ts.dt.tz_convert(None)


def _decision_keys(frame: pd.DataFrame) -> pd.DataFrame:
    ts_col = "timestamp" if "timestamp" in frame.columns else "__ts__"
    symbol_col = "symbol" if "symbol" in frame.columns else "__symbol__"
    out = pd.DataFrame(index=frame.index)
    out["__join_ts__"] = _key_timestamp(frame[ts_col])
    out["__join_symbol__"] = frame[symbol_col].astype(str)
    out["__join_side__"] = frame["side_name"].astype(str).str.lower()
    return out


def _clean_forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col in OUTCOME_COLUMNS)


def _read_variant(variant_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    clean_path = variant_dir / "s52_meta_threshold_guarded_candidates.parquet"
    offline_path = variant_dir / "s52_meta_threshold_guarded_offline_eval_candidates.parquet"
    audit_path = variant_dir / "s52_meta_threshold_guarded_leakage_audit.json"
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)
    clean = pd.read_parquet(clean_path)
    offline = pd.read_parquet(offline_path) if offline_path.exists() else None
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return clean, offline, audit


def _prepare_ledger(scored_ledger: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = {"__ts__", "__symbol__", "side_name"}
    missing = sorted(required - set(scored_ledger.columns))
    if missing:
        raise ValueError(f"Scored ledger is missing join columns: {missing}")
    keys = _decision_keys(scored_ledger)
    payload_cols = [
        col
        for col in LEDGER_OUTCOME_COLUMNS
        if col in scored_ledger.columns
    ]
    ledger = pd.concat([keys, scored_ledger[payload_cols].reset_index(drop=True)], axis=1)
    duplicate_count = int(ledger.duplicated(["__join_ts__", "__join_symbol__", "__join_side__"]).sum())
    ledger = ledger.drop_duplicates(["__join_ts__", "__join_symbol__", "__join_side__"], keep="first")
    return ledger, duplicate_count


def _replay_variant(
    *,
    variant_name: str,
    variant_dir: Path,
    ledger: pd.DataFrame,
    ledger_duplicate_keys: int,
    notional_per_trade: float,
    stop_mult: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clean, offline, audit = _read_variant(variant_dir)
    clean_forbidden = _clean_forbidden_columns(clean)
    clean_keys = _decision_keys(clean)
    clean_key_dup = int(clean_keys.duplicated(["__join_ts__", "__join_symbol__", "__join_side__"]).sum())
    decision = pd.concat([clean.reset_index(drop=True), clean_keys.reset_index(drop=True)], axis=1)
    replay = decision.merge(
        ledger,
        on=["__join_ts__", "__join_symbol__", "__join_side__"],
        how="left",
        validate="one_to_one",
    )
    replay["replay_outcome_matched"] = _num(replay.get("ret_net"), index=replay.index).notna()
    replay["frozen_action_notional"] = float(notional_per_trade)
    replay["frozen_action_net_pnl"] = _num(replay.get("ret_net"), index=replay.index).fillna(0.0) * float(notional_per_trade)
    replay["frozen_action_exec_margin_pnl"] = _num(replay.get("exec_margin"), index=replay.index).fillna(0.0) * float(
        notional_per_trade
    )
    decision_mae = _num(replay.get("first_touch_mae_norm"), index=replay.index)
    full_path_mae = _num(replay.get("first_touch_full_path_mae_norm"), index=replay.index)
    if full_path_mae.isna().all():
        full_path_mae = _num(replay.get("mae_norm"), index=replay.index)
    replay["inferred_decision_stop_touch"] = decision_mae.ge(float(stop_mult)).astype(float)
    replay["inferred_full_path_stop_touch"] = full_path_mae.ge(float(stop_mult)).astype(float)

    summary = _summary_row(variant_name, replay, notional_per_trade=notional_per_trade)
    summary.update(
        {
            "variant_dir": str(variant_dir),
            "clean_handoff_forbidden_columns": ",".join(clean_forbidden),
            "clean_handoff_has_no_realized_outcomes": not clean_forbidden,
            "clean_duplicate_decision_key_rows": int(clean_key_dup),
            "ledger_duplicate_decision_key_rows": int(ledger_duplicate_keys),
            "audit_clean_handoff_has_no_realized_outcomes": bool(
                audit.get("clean_handoff_has_no_realized_outcomes", not clean_forbidden)
            ),
            "audit_duplicate_decision_key_rows": int(audit.get("duplicate_decision_key_rows", -1)),
        }
    )
    parity = _offline_parity(replay, offline)
    summary.update({f"offline_parity_{key}": value for key, value in parity.items()})
    breakdown = _breakdowns(variant_name, replay)
    return summary, breakdown, replay, parity


def _summary_row(variant: str, frame: pd.DataFrame, *, notional_per_trade: float) -> dict[str, Any]:
    return {
        "variant": variant,
        "rows": int(len(frame)),
        "matched_rows": int(_num(frame.get("replay_outcome_matched"), index=frame.index).sum()),
        "unmatched_rows": int(len(frame) - _num(frame.get("replay_outcome_matched"), index=frame.index).sum()),
        "symbols": int(frame.get("symbol", frame.get("__symbol__", pd.Series(dtype=str))).nunique()),
        "months": int(frame.get("month", pd.Series(dtype=str)).nunique()),
        "notional_per_trade": float(notional_per_trade),
        "sum_net_pnl": _sum(frame.get("frozen_action_net_pnl")),
        "sum_exec_margin_pnl": _sum(frame.get("frozen_action_exec_margin_pnl")),
        "mean_ret_net": _mean(frame.get("ret_net")),
        "sum_ret_net": _sum(frame.get("ret_net")),
        "worst_month_ret_net": _worst_group_mean(frame, "month", "ret_net"),
        "mean_exec_margin": _mean(frame.get("exec_margin")),
        "worst_month_exec_margin": _worst_group_mean(frame, "month", "exec_margin"),
        "hit_rate_ret_net": _rate(_num(frame.get("ret_net"), index=frame.index).gt(0.0)),
        "positive_exec_margin_rate": _rate(_num(frame.get("exec_margin"), index=frame.index).gt(0.0)),
        "full_path_bad_mae": _rate(frame.get("full_path_bad_mae_1r")),
        "max_month_full_path_bad_mae": _max_group_rate(frame, "month", "full_path_bad_mae_1r"),
        "timeout": _rate(frame.get("timeout")),
        "max_month_timeout": _max_group_rate(frame, "month", "timeout"),
        "clean_exec_precision": _rate(frame.get("clean_exec")),
        "dirty_positive_rate": _rate(frame.get("dirty_positive")),
        "mfe_before_mae_rate": _rate(frame.get("mfe_before_mae_1r")),
        "mae_before_mfe_rate": _rate(frame.get("mae_before_mfe_1r")),
        "inferred_decision_stop_touch": _rate(frame.get("inferred_decision_stop_touch")),
        "inferred_full_path_stop_touch": _rate(frame.get("inferred_full_path_stop_touch")),
        "dominant_side_share": _dominant_side_share(frame),
    }


def _offline_parity(replay: pd.DataFrame, offline: pd.DataFrame | None) -> dict[str, Any]:
    if offline is None or offline.empty:
        return {
            "available": False,
            "row_count_match": False,
            "key_set_match": False,
            "max_abs_ret_net_diff": float("nan"),
            "max_abs_exec_margin_diff": float("nan"),
        }
    left = _decision_keys(replay)
    right = _decision_keys(offline)
    left_keys = set(map(tuple, left.to_numpy(dtype=object)))
    right_keys = set(map(tuple, right.to_numpy(dtype=object)))
    cols = [col for col in ("ret_net", "exec_margin", "full_path_bad_mae_1r", "timeout") if col in offline.columns and col in replay.columns]
    if cols:
        cmp = replay[["__join_ts__", "__join_symbol__", "__join_side__"] + cols].merge(
            pd.concat([right, offline[cols].reset_index(drop=True)], axis=1),
            on=["__join_ts__", "__join_symbol__", "__join_side__"],
            how="inner",
            suffixes=("_replay", "_offline"),
        )
    else:
        cmp = pd.DataFrame()
    out: dict[str, Any] = {
        "available": True,
        "row_count_match": bool(len(replay) == len(offline)),
        "key_set_match": bool(left_keys == right_keys),
        "keys_only_in_clean_replay": int(len(left_keys - right_keys)),
        "keys_only_in_offline": int(len(right_keys - left_keys)),
    }
    for col in cols:
        diff = (
            _num(cmp.get(f"{col}_replay"), index=cmp.index)
            - _num(cmp.get(f"{col}_offline"), index=cmp.index)
        ).abs()
        out[f"max_abs_{col}_diff"] = float(diff.max()) if len(diff) else float("nan")
    return out


def _breakdowns(variant: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grouping, cols in GROUPING_SPECS.items():
        if not all(col in frame.columns for col in cols):
            continue
        for key, group in frame.groupby(list(cols), dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rec = _summary_row(
                variant,
                group,
                notional_per_trade=float(group["frozen_action_notional"].iloc[0])
                if len(group) and "frozen_action_notional" in group.columns
                else 1.0,
            )
            rec["grouping"] = grouping
            for col, value in zip(cols, key, strict=False):
                rec[col] = value
            rows.append(rec)
    return pd.DataFrame(rows)


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val * 100:.2f}%"


def _write_report(path: Path, summary: pd.DataFrame, breakdown: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# S52 Frozen Meta-Threshold Action Replay",
        "",
        "## Scope",
        "",
        "Replays clean, already-materialized S52 meta-threshold action files against the scored trailing-policy ledger.",
        "Selection is not recomputed here; realized outcome columns are attached only after frozen decision keys are fixed.",
        "",
        "## Leakage Contract",
        "",
        f"- scored ledger: `{manifest['scored_ledger_path']}`",
        f"- scored ledger hash: `{manifest['scored_ledger_sha256']}`",
        "- clean handoff files must contain no realized outcome columns",
        "- join key: `(timestamp, symbol, side_name)`",
        "- replay outcome source: S52 scored trailing-profit ledger",
        "",
        "## Summary",
        "",
        summary[
            [
                "variant",
                "rows",
                "matched_rows",
                "symbols",
                "sum_net_pnl",
                "mean_ret_net",
                "worst_month_ret_net",
                "mean_exec_margin",
                "worst_month_exec_margin",
                "hit_rate_ret_net",
                "full_path_bad_mae",
                "max_month_full_path_bad_mae",
                "timeout",
                "inferred_decision_stop_touch",
                "inferred_full_path_stop_touch",
                "dominant_side_share",
                "clean_handoff_has_no_realized_outcomes",
                "offline_parity_key_set_match",
            ]
        ].to_markdown(index=False),
        "",
        "## Month / Side Breakdown",
        "",
    ]
    month_side = breakdown[breakdown["grouping"].eq("month_side")] if not breakdown.empty else pd.DataFrame()
    if month_side.empty:
        lines.append("_No month-side breakdown._")
    else:
        cols = [
            "variant",
            "month",
            "side_name",
            "rows",
            "symbols",
            "sum_net_pnl",
            "mean_ret_net",
            "mean_exec_margin",
            "full_path_bad_mae",
            "timeout",
            "inferred_decision_stop_touch",
            "inferred_full_path_stop_touch",
        ]
        lines.append(month_side[[col for col in cols if col in month_side.columns]].to_markdown(index=False))
    lines += [
        "",
        "## Interpretation",
        "",
        "Bad-MAE above 50% is treated as a path-risk diagnostic in the PnL-override gate, not an automatic veto, when net PnL and non-risk hygiene checks pass.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_replay(
    *,
    handoff_root: Path,
    variants: tuple[str, ...],
    scored_ledger_path: Path,
    out_dir: Path,
    notional_per_trade: float = 1.0,
    stop_mult: float = 0.50,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not scored_ledger_path.exists():
        raise FileNotFoundError(scored_ledger_path)
    scored_ledger = pd.read_parquet(scored_ledger_path)
    ledger, ledger_duplicate_keys = _prepare_ledger(scored_ledger)
    summary_rows: list[dict[str, Any]] = []
    breakdown_frames: list[pd.DataFrame] = []
    replay_paths: dict[str, str] = {}
    parity: dict[str, Any] = {}
    for variant_name in variants:
        variant_dir = handoff_root / variant_name
        if not variant_dir.exists():
            raise FileNotFoundError(variant_dir)
        summary, breakdown, replay, variant_parity = _replay_variant(
            variant_name=variant_name,
            variant_dir=variant_dir,
            ledger=ledger,
            ledger_duplicate_keys=ledger_duplicate_keys,
            notional_per_trade=float(notional_per_trade),
            stop_mult=float(stop_mult),
        )
        summary_rows.append(summary)
        if not breakdown.empty:
            breakdown_frames.append(breakdown)
        replay_path = out_dir / f"{variant_name}__frozen_action_replay.parquet"
        replay.to_parquet(replay_path, index=False)
        replay_paths[variant_name] = str(replay_path)
        parity[variant_name] = variant_parity

    summary = pd.DataFrame(summary_rows)
    breakdown = pd.concat(breakdown_frames, ignore_index=True) if breakdown_frames else pd.DataFrame()
    paths = {
        "summary": out_dir / "s52_frozen_action_replay_summary.csv",
        "breakdown": out_dir / "s52_frozen_action_replay_breakdown.csv",
        "report": out_dir / "s52_frozen_action_replay.md",
        "manifest": out_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    breakdown.to_csv(paths["breakdown"], index=False)
    manifest = {
        "generated_by": "replay_s52_meta_threshold_frozen_actions",
        "handoff_root": str(handoff_root),
        "variants": list(variants),
        "scored_ledger_path": str(scored_ledger_path),
        "scored_ledger_sha256": _sha256_path(scored_ledger_path),
        "notional_per_trade": float(notional_per_trade),
        "stop_mult": float(stop_mult),
        "outputs": {key: str(value) for key, value in paths.items()},
        "replay_candidates": replay_paths,
        "offline_parity": parity,
        "scope": "frozen_clean_action_file_joined_to_s52_scored_trailing_policy_outcomes",
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_report(paths["report"], summary, breakdown, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--scored-ledger", type=Path, default=DEFAULT_SCORED_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_HANDOFF_ROOT / "s52_frozen_action_replay_current_v1")
    parser.add_argument("--notional-per-trade", type=float, default=1.0)
    parser.add_argument("--stop-mult", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_replay(
        handoff_root=args.handoff_root,
        variants=tuple(args.variants) if args.variants else DEFAULT_VARIANTS,
        scored_ledger_path=args.scored_ledger,
        out_dir=args.out_dir,
        notional_per_trade=float(args.notional_per_trade),
        stop_mult=float(args.stop_mult),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
