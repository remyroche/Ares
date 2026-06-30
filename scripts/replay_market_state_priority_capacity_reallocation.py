#!/usr/bin/env python3
"""Replay market-state priority with shadow per-strategy capacity reallocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_market_state_head_priority_learning import (  # noqa: E402
    BASELINE_ARM,
    _accepted_overlap,
    _load_candidates,
    _load_json,
    _replay_arm,
    replay_selection_metrics,
)
from scripts.audit_market_state_priority_shadow_promotion import resolve_arm_selector  # noqa: E402
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_CAP_SWEEP_DIR = Path(
    "data_perp/reports/market_state_priority_shadow_windows_20260626_v1"
    "/01_jun_15_22_utc/cap_sweep"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_priority_capacity_reallocation_20260626")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value < 0:
            raise ValueError(f"invalid nonnegative integer value: {text!r}")
        values.append(value)
    if not values:
        raise ValueError("at least one integer value is required")
    return sorted(set(values))


def _select_arm(metrics: pd.DataFrame, arm_contains: str) -> str:
    if metrics.empty or "arm" not in metrics.columns:
        raise ValueError("cap-sweep metrics are missing arm rows")
    mask = metrics["arm"].astype(str).str.contains(str(arm_contains), regex=False, na=False)
    selected = metrics.loc[mask, "arm"].dropna().astype(str)
    if selected.empty:
        raise ValueError(f"no cap-sweep arm contains {arm_contains!r}")
    return str(selected.iloc[0])


def _arm_name(
    base_arm: str,
    *,
    bar_uplift: int,
    concurrent_uplift: int,
    reduce_disfavored: bool,
) -> str:
    mode = "reduce" if reduce_disfavored else "favoronly"
    return f"{base_arm}_capalloc_bar{bar_uplift}_conc{concurrent_uplift}_{mode}"


def apply_capacity_reallocation(
    candidates: pd.DataFrame,
    *,
    base_strategy_bar_cap: int,
    base_strategy_concurrent_cap: int | None,
    bar_uplift: int,
    concurrent_uplift: int,
    reduce_disfavored: bool = True,
    min_strategy_bar_cap: int = 1,
    min_strategy_concurrent_cap: int = 1,
) -> pd.DataFrame:
    """Apply row-level per-strategy cap overrides from signed priority adjustment."""
    if "portfolio_priority_adjustment" not in candidates.columns:
        raise ValueError("candidates must include portfolio_priority_adjustment")
    out = candidates.copy()
    adjustment = pd.to_numeric(out["portfolio_priority_adjustment"], errors="coerce").fillna(0.0)
    fav = adjustment > 1e-12
    disfav = adjustment < -1e-12
    if int(bar_uplift) > 0:
        bar_cap = pd.Series(np.nan, index=out.index, dtype=float)
        base = max(int(base_strategy_bar_cap), int(min_strategy_bar_cap))
        bar_cap.loc[fav] = float(base + int(bar_uplift))
        if reduce_disfavored:
            bar_cap.loc[disfav] = float(max(int(min_strategy_bar_cap), base - int(bar_uplift)))
        out["portfolio_max_new_entries_per_strategy_per_bar"] = bar_cap
    if int(concurrent_uplift) > 0 and base_strategy_concurrent_cap is not None:
        concurrent_cap = pd.Series(np.nan, index=out.index, dtype=float)
        base_conc = max(int(base_strategy_concurrent_cap), int(min_strategy_concurrent_cap))
        concurrent_cap.loc[fav] = float(base_conc + int(concurrent_uplift))
        if reduce_disfavored:
            concurrent_cap.loc[disfav] = float(
                max(int(min_strategy_concurrent_cap), base_conc - int(concurrent_uplift))
            )
        out["portfolio_max_concurrent_per_strategy"] = concurrent_cap
    return out


def _read_cap_sweep_inputs(
    cap_sweep_dir: Path,
    arm_contains: str,
    *,
    use_selected_challenger: bool = False,
) -> dict[str, Any]:
    metrics_path = cap_sweep_dir / "head_priority_cap_sweep_metrics.csv"
    manifest_path = cap_sweep_dir / "manifest.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    metrics = pd.read_csv(metrics_path)
    resolved_arm_contains, arm_selector_source = resolve_arm_selector(
        [cap_sweep_dir],
        arm_contains=str(arm_contains),
        use_selected_challenger=bool(use_selected_challenger),
    )
    arm = _select_arm(metrics, resolved_arm_contains)
    manifest = _load_json(manifest_path)
    inputs = dict(manifest.get("inputs") or {})
    candidates_path = cap_sweep_dir / f"{arm}_candidates.parquet"
    train_deployable_path = Path(inputs.get("train_deployable_candidates") or "")
    policy_manifest_path = Path(inputs.get("policy_manifest") or "")
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    if not train_deployable_path.exists():
        raise FileNotFoundError(train_deployable_path)
    if not policy_manifest_path.exists():
        raise FileNotFoundError(policy_manifest_path)
    return {
        "arm": arm,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "requested_arm_contains": str(arm_contains),
        "use_selected_challenger": bool(use_selected_challenger),
        "resolved_arm_contains": resolved_arm_contains,
        "arm_selector_source": arm_selector_source,
        "candidates_path": candidates_path,
        "train_deployable_path": train_deployable_path,
        "policy_manifest_path": policy_manifest_path,
    }


def _metric_row(
    *,
    arm: str,
    summary_part: pd.DataFrame,
    accepted: pd.DataFrame,
    base_summary: pd.DataFrame,
    base_accepted: pd.DataFrame,
    bar_uplift: int | None,
    concurrent_uplift: int | None,
    reduce_disfavored: bool | None,
) -> dict[str, Any]:
    replay_metrics = replay_selection_metrics(
        arm=arm,
        candidate_summary=summary_part,
        candidate_accepted=accepted,
        base_summary=base_summary,
        base_accepted=base_accepted,
    )
    cand_row = summary_part.iloc[0].to_dict() if not summary_part.empty else {}
    base_row = base_summary.iloc[0].to_dict() if not base_summary.empty else {}
    return {
        "arm": arm,
        "bar_uplift": bar_uplift,
        "concurrent_uplift": concurrent_uplift,
        "reduce_disfavored": reduce_disfavored,
        "trade_count": int(cand_row.get("trade_count", 0) or 0),
        "net_pnl": float(cand_row.get("net_pnl", np.nan)),
        "delta_net_pnl": float(cand_row.get("net_pnl", 0.0) or 0.0)
        - float(base_row.get("net_pnl", 0.0) or 0.0),
        "full_sl_rate": float(cand_row.get("full_sl_rate", np.nan)),
        "delta_full_sl_rate": float(cand_row.get("full_sl_rate", np.nan))
        - float(base_row.get("full_sl_rate", np.nan)),
        "timeout_rate": float(cand_row.get("timeout_rate", np.nan)),
        "delta_timeout_rate": float(cand_row.get("timeout_rate", np.nan))
        - float(base_row.get("timeout_rate", np.nan)),
        "accepted_jaccard": float(replay_metrics.get("replay_accepted_jaccard", np.nan)),
        "entrants": int(float(replay_metrics.get("replay_entrants", 0) or 0)),
        "removed": int(float(replay_metrics.get("replay_removed", 0) or 0)),
        "net_replacement_pnl": float(replay_metrics.get("replay_net_replacement_pnl", np.nan)),
        "net_action_pnl_delta": float(replay_metrics.get("replay_net_action_pnl_delta", np.nan)),
        "defensive_success": float(replay_metrics.get("replay_defensive_success", np.nan)),
    }


def _render_report(
    *,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    by_head: pd.DataFrame,
    overlap: pd.DataFrame,
    swap: pd.DataFrame,
) -> str:
    lines = [
        "# Market-State Priority Capacity-Reallocation Shadow Replay",
        "",
        "This replay keeps T1 scores, ranks, thresholds, sizing, costs and global capacity unchanged.",
        "It only tests row-level per-strategy capacity overrides around a frozen market-state priority schedule.",
        "",
        "## Contract",
        "",
        f"- Cap-sweep dir: `{manifest['inputs']['cap_sweep_dir']}`",
        f"- Source priority arm: `{manifest['inputs']['source_priority_arm']}`",
        f"- Arm selector: `{manifest['params']['resolved_arm_contains']}`",
        f"- Arm selector source: `{manifest['params']['arm_selector_source']}`",
        "- Active production remains static T1.",
        "- This is a shadow allocation-mechanics ablation, not the primary threshold-controller track.",
        "",
        "## Metrics",
        "",
    ]
    view_cols = [
        "arm",
        "bar_uplift",
        "concurrent_uplift",
        "reduce_disfavored",
        "trade_count",
        "net_pnl",
        "delta_net_pnl",
        "full_sl_rate",
        "delta_full_sl_rate",
        "timeout_rate",
        "delta_timeout_rate",
        "accepted_jaccard",
        "entrants",
        "removed",
        "net_replacement_pnl",
        "defensive_success",
    ]
    lines.append(metrics[[c for c in view_cols if c in metrics.columns]].to_markdown(index=False))
    lines.extend(["", "## By Head", ""])
    lines.append(by_head.to_markdown(index=False) if not by_head.empty else "_No by-head rows._")
    lines.extend(["", "## Accepted Overlap", ""])
    lines.append(overlap.to_markdown(index=False) if not overlap.empty else "_No overlap rows._")
    lines.extend(["", "## Accepted Swap Utility", ""])
    lines.append(swap.to_markdown(index=False) if not swap.empty else "_No swap rows._")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-sweep-dir", type=Path, default=DEFAULT_CAP_SWEEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arm-contains", default="cap_0p15_zge_0p5")
    parser.add_argument("--use-selected-challenger", action="store_true")
    parser.add_argument("--bar-uplifts", default="0,1")
    parser.add_argument("--concurrent-uplifts", default="0,1")
    parser.add_argument("--no-reduce-disfavored", action="store_true")
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _read_cap_sweep_inputs(
        args.cap_sweep_dir,
        str(args.arm_contains),
        use_selected_challenger=bool(args.use_selected_challenger),
    )
    source_arm = str(inputs["arm"])
    candidates = _load_candidates(inputs["candidates_path"])
    train_deployable = _load_candidates(inputs["train_deployable_path"])
    params, policy_payload = mstc._load_policy_params(
        inputs["policy_manifest_path"],
        str(args.policy_variant),
    )
    base_strategy_bar_cap = int(
        params.max_new_entries_per_strategy_per_bar
        if params.max_new_entries_per_strategy_per_bar is not None
        else params.max_new_entries_per_bar
    )
    base_strategy_concurrent_cap = (
        int(params.max_concurrent_per_strategy)
        if params.max_concurrent_per_strategy is not None
        else None
    )
    reduce_disfavored = not bool(args.no_reduce_disfavored)

    accepted_by_arm: dict[str, pd.DataFrame] = {}
    summary_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    base_candidates = candidates.copy()
    base_candidates["portfolio_priority_adjustment"] = 0.0
    for col in ["portfolio_max_new_entries_per_strategy_per_bar", "portfolio_max_concurrent_per_strategy"]:
        if col in base_candidates.columns:
            base_candidates = base_candidates.drop(columns=[col])
    base_decisions, base_equity, base_accepted, base_summary, base_by_head = _replay_arm(
        arm=BASELINE_ARM,
        candidates=base_candidates,
        train_deployable=train_deployable,
        params=params,
        market_mode=str(args.market_mode),
    )
    accepted_by_arm[BASELINE_ARM] = base_accepted
    summary_frames.append(base_summary)
    by_head_frames.append(base_by_head)
    metric_rows.append(
        _metric_row(
            arm=BASELINE_ARM,
            summary_part=base_summary,
            accepted=base_accepted,
            base_summary=base_summary,
            base_accepted=base_accepted,
            bar_uplift=None,
            concurrent_uplift=None,
            reduce_disfavored=None,
        )
    )
    base_decisions.to_parquet(args.output_dir / f"{BASELINE_ARM}_decisions.parquet", index=False)
    base_equity.to_parquet(args.output_dir / f"{BASELINE_ARM}_equity.parquet", index=False)
    base_accepted.to_parquet(args.output_dir / f"{BASELINE_ARM}_accepted_trades.parquet", index=False)

    priority_decisions, priority_equity, priority_accepted, priority_summary, priority_by_head = _replay_arm(
        arm=source_arm,
        candidates=candidates,
        train_deployable=train_deployable,
        params=params,
        market_mode=str(args.market_mode),
    )
    accepted_by_arm[source_arm] = priority_accepted
    summary_frames.append(priority_summary)
    by_head_frames.append(priority_by_head)
    metric_rows.append(
        _metric_row(
            arm=source_arm,
            summary_part=priority_summary,
            accepted=priority_accepted,
            base_summary=base_summary,
            base_accepted=base_accepted,
            bar_uplift=0,
            concurrent_uplift=0,
            reduce_disfavored=reduce_disfavored,
        )
    )
    priority_decisions.to_parquet(args.output_dir / f"{source_arm}_decisions.parquet", index=False)
    priority_equity.to_parquet(args.output_dir / f"{source_arm}_equity.parquet", index=False)
    priority_accepted.to_parquet(args.output_dir / f"{source_arm}_accepted_trades.parquet", index=False)

    for bar_uplift in _parse_int_list(args.bar_uplifts):
        for concurrent_uplift in _parse_int_list(args.concurrent_uplifts):
            if bar_uplift == 0 and concurrent_uplift == 0:
                continue
            arm = _arm_name(
                source_arm,
                bar_uplift=int(bar_uplift),
                concurrent_uplift=int(concurrent_uplift),
                reduce_disfavored=reduce_disfavored,
            )
            arm_candidates = apply_capacity_reallocation(
                candidates,
                base_strategy_bar_cap=base_strategy_bar_cap,
                base_strategy_concurrent_cap=base_strategy_concurrent_cap,
                bar_uplift=int(bar_uplift),
                concurrent_uplift=int(concurrent_uplift),
                reduce_disfavored=reduce_disfavored,
            )
            decisions, equity, accepted, summary_part, by_head_part = _replay_arm(
                arm=arm,
                candidates=arm_candidates,
                train_deployable=train_deployable,
                params=params,
                market_mode=str(args.market_mode),
            )
            accepted_by_arm[arm] = accepted
            summary_frames.append(summary_part)
            by_head_frames.append(by_head_part)
            metric_rows.append(
                _metric_row(
                    arm=arm,
                    summary_part=summary_part,
                    accepted=accepted,
                    base_summary=base_summary,
                    base_accepted=base_accepted,
                    bar_uplift=int(bar_uplift),
                    concurrent_uplift=int(concurrent_uplift),
                    reduce_disfavored=reduce_disfavored,
                )
            )
            arm_candidates.to_parquet(args.output_dir / f"{arm}_candidates.parquet", index=False)
            decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
            equity.to_parquet(args.output_dir / f"{arm}_equity.parquet", index=False)
            accepted.to_parquet(args.output_dir / f"{arm}_accepted_trades.parquet", index=False)

    summary = pd.concat(summary_frames, ignore_index=True)
    by_head = pd.concat(by_head_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    overlap = _accepted_overlap(accepted_by_arm)
    accepted_all = pd.concat(list(accepted_by_arm.values()), ignore_index=True)
    swap = mstc._threshold_action_utility(accepted_all, BASELINE_ARM)

    summary.to_csv(args.output_dir / "market_state_priority_capacity_reallocation_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "market_state_priority_capacity_reallocation_by_head.csv", index=False)
    metrics.to_csv(args.output_dir / "market_state_priority_capacity_reallocation_metrics.csv", index=False)
    overlap.to_csv(args.output_dir / "market_state_priority_capacity_reallocation_overlap.csv", index=False)
    swap.to_csv(args.output_dir / "market_state_priority_capacity_reallocation_swap_utility.csv", index=False)

    out_manifest = {
        "generated_by": "replay_market_state_priority_capacity_reallocation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "shadow_market_state_priority_capacity_reallocation",
        "contract": {
            "active_baseline": "static_T1",
            "shadow_only": True,
            "changes_scores_or_ranks": False,
            "changes_thresholds": False,
            "changes_sizing": False,
            "changes_global_capacity": False,
            "changes_per_strategy_capacity": True,
            "qfail_active": False,
            "head_health_active": False,
        },
        "params": {
            "arm_contains": str(args.arm_contains),
            "use_selected_challenger": bool(args.use_selected_challenger),
            "resolved_arm_contains": str(inputs["resolved_arm_contains"]),
            "arm_selector_source": str(inputs["arm_selector_source"]),
            "bar_uplifts": _parse_int_list(args.bar_uplifts),
            "concurrent_uplifts": _parse_int_list(args.concurrent_uplifts),
            "reduce_disfavored": reduce_disfavored,
            "base_strategy_bar_cap": base_strategy_bar_cap,
            "base_strategy_concurrent_cap": base_strategy_concurrent_cap,
            "policy_variant": str(args.policy_variant),
            "market_mode": str(args.market_mode),
        },
        "inputs": {
            "cap_sweep_dir": str(args.cap_sweep_dir),
            "cap_sweep_manifest_sha256": _sha256(inputs["manifest_path"]),
            "source_priority_arm": source_arm,
            "candidates": str(inputs["candidates_path"]),
            "candidates_sha256": _sha256(inputs["candidates_path"]),
            "train_deployable_candidates": str(inputs["train_deployable_path"]),
            "train_deployable_candidates_sha256": _sha256(inputs["train_deployable_path"]),
            "policy_manifest": str(inputs["policy_manifest_path"]),
            "policy_manifest_sha256": _sha256(inputs["policy_manifest_path"]),
            "policy_manifest_run_id": policy_payload.get("run_id"),
        },
        "summary": metrics.to_dict("records"),
        "outputs": {
            "metrics": str(args.output_dir / "market_state_priority_capacity_reallocation_metrics.csv"),
            "summary": str(args.output_dir / "market_state_priority_capacity_reallocation_summary.csv"),
            "by_head": str(args.output_dir / "market_state_priority_capacity_reallocation_by_head.csv"),
            "overlap": str(args.output_dir / "market_state_priority_capacity_reallocation_overlap.csv"),
            "swap": str(args.output_dir / "market_state_priority_capacity_reallocation_swap_utility.csv"),
            "report": str(args.output_dir / "market_state_priority_capacity_reallocation_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(out_manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_priority_capacity_reallocation_report.md").write_text(
        _render_report(
            manifest=out_manifest,
            metrics=metrics,
            by_head=by_head,
            overlap=overlap,
            swap=swap,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "metrics": metrics.to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()
