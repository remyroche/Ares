#!/usr/bin/env python3
"""Materialize the T1 repaired-rank static baseline or rank-contract challenger.

The promoted stack is deliberately narrower than the research controllers:

    anchor/meta scores
    -> frozen global-over-time rank reference
    -> static base thresholds
    -> refit_bar4_strategy_bar2 global auction

Market-state controller outputs and q-fail are kept out of the active decision
path.  By default this script reproduces the current global-rank T1 static
baseline using the frozen pre-June rank reference.  The older short_boll
within-timestamp rank remains available as an explicit comparison contract.
"""

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

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/artifacts/reliability_blend_T1_global_rank_static_baseline_active_20260627"
)
DEFAULT_ATTRIBUTION_DIR = Path(
    "data_perp/reports/market_state_attribution_ablation_20260625_fixed_universe_v2"
)
ARM_NAME = "production_T1_repaired_static_baseline"
DEFAULT_RANK_REFERENCE_RUN_ID = "reliability_blend_anchor_rank_reference_20260625_prejune"


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


def _apply_global_anchor_rank_reference(
    candidates: pd.DataFrame,
    *,
    data_root: Path,
    rank_reference_run_id: str,
    score_col: str = "calibrated_score",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference

    out, diag = apply_frozen_policy_rank_reference(
        candidates,
        data_root=data_root,
        run_id=rank_reference_run_id,
        score_col=score_col,
        allow_window_rank_debug=False,
    )
    out["rank_contract_source"] = "anchor_global_policy_rank_reference"
    return mstc.normalise_candidate_table(out), diag


def _load_for_t1(
    path: Path,
    *,
    rank_contract: str,
    disabled_heads: set[str],
    data_root: Path,
    rank_reference_run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = mstc._disable_heads(mstc._load_candidates(path), disabled_heads)
    if rank_contract == "short_boll_timestamp_rank":
        out = mstc._apply_rank_contract(candidates, rank_contract)
        diag = {
            "rank_contract": rank_contract,
            "rank_scope": "within_timestamp",
            "rank_reference_run_id": None,
            "rank_source": "head_timestamp_rank_score",
        }
    elif rank_contract == "anchor_global_policy_rank_reference":
        out, diag = _apply_global_anchor_rank_reference(
            candidates,
            data_root=data_root,
            rank_reference_run_id=rank_reference_run_id,
        )
        diag["rank_contract"] = rank_contract
        diag["rank_scope"] = "global_over_time"
    else:
        raise ValueError(f"Unknown T1 rank contract: {rank_contract}")
    return mstc.normalise_candidate_table(out), diag


def _rank_reference_contract_report(
    *,
    rank_contract: str,
    eval_diag: dict[str, Any],
    train_diag: dict[str, Any],
    eval_rows: int,
    train_rows: int,
) -> dict[str, Any]:
    if rank_contract != "anchor_global_policy_rank_reference":
        return {
            "required": False,
            "passed": True,
            "reason": "timestamp_rank_contract_no_global_reference_required",
        }

    failures: list[str] = []
    for split, diag, expected_rows in (
        ("eval", dict(eval_diag or {}), int(eval_rows)),
        ("train_deployable", dict(train_diag or {}), int(train_rows)),
    ):
        if diag.get("rank_source") != "policy_rank_reference_percentile":
            failures.append(f"{split}.rank_source_not_policy_rank_reference_percentile")
        if diag.get("rank_reference_run_id") in (None, ""):
            failures.append(f"{split}.rank_reference_run_id_missing")
        if int(diag.get("missing_rank_rows") or 0) != 0:
            failures.append(f"{split}.missing_policy_rank_rows_nonzero")
        if int(diag.get("missing_auction_rank_rows") or 0) != 0:
            failures.append(f"{split}.missing_auction_rank_rows_nonzero")
        if int(diag.get("ranked_rows") or 0) != expected_rows:
            failures.append(f"{split}.ranked_rows_do_not_match_input_rows")
        if int(diag.get("auction_ranked_rows") or 0) != expected_rows:
            failures.append(f"{split}.auction_ranked_rows_do_not_match_input_rows")
        if bool(diag.get("window_rank_debug_used")):
            failures.append(f"{split}.window_rank_debug_used")
        source_counts = {
            **dict(diag.get("policy_rank_reference_source_counts") or {}),
            **dict(diag.get("auction_rank_reference_source_counts") or {}),
        }
        if any("window_rank_debug" in str(key) for key in source_counts):
            failures.append(f"{split}.window_rank_debug_source_present")
        if int(diag.get("policy_rank_reference_n_min") or 0) <= 0:
            failures.append(f"{split}.policy_rank_reference_n_min_nonpositive")
        if int(diag.get("auction_rank_reference_n_min") or 0) <= 0:
            failures.append(f"{split}.auction_rank_reference_n_min_nonpositive")

    return {
        "required": True,
        "passed": not failures,
        "failures": failures,
        "eval_expected_rows": int(eval_rows),
        "train_deployable_expected_rows": int(train_rows),
        "eval_rank_reference_run_id": eval_diag.get("rank_reference_run_id"),
        "train_rank_reference_run_id": train_diag.get("rank_reference_run_id"),
        "eval_ranked_rows": int(eval_diag.get("ranked_rows") or 0),
        "train_ranked_rows": int(train_diag.get("ranked_rows") or 0),
        "eval_auction_ranked_rows": int(eval_diag.get("auction_ranked_rows") or 0),
        "train_auction_ranked_rows": int(train_diag.get("auction_ranked_rows") or 0),
    }


def _deployable_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    rank_col = None
    for col in ("policy_rank_pct", "normalized_rank_score", "strategy_rank_pct", "rank_pct"):
        if col in candidates.columns:
            rank_col = col
            break
    if rank_col is None:
        return candidates.iloc[0:0].copy()
    threshold_col = (
        "deployment_rank_threshold"
        if "deployment_rank_threshold" in candidates.columns
        else "base_strategy_threshold"
    )
    rank = pd.to_numeric(candidates[rank_col], errors="coerce")
    threshold = pd.to_numeric(candidates[threshold_col], errors="coerce").fillna(np.inf)
    out = candidates.loc[(rank >= threshold).fillna(False)].copy()
    out["active_rank_column"] = rank_col
    out["active_threshold_column"] = threshold_col
    return out.reset_index(drop=True)


def _metric_delta(left: pd.Series, right: pd.Series, cols: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for col in cols:
        if col not in left.index or col not in right.index:
            out[col] = None
            continue
        lv = pd.to_numeric(pd.Series([left[col]]), errors="coerce").iloc[0]
        rv = pd.to_numeric(pd.Series([right[col]]), errors="coerce").iloc[0]
        if not np.isfinite(lv) or not np.isfinite(rv):
            out[col] = None
        else:
            out[col] = float(lv - rv)
    return out


def _rank_contract_scope(rank_contract: str) -> str:
    return "global_over_time" if rank_contract == "anchor_global_policy_rank_reference" else "within_timestamp"


def _rank_contract_promotion_status(rank_contract: str) -> str:
    if rank_contract == "anchor_global_policy_rank_reference":
        return "active_candidate_pending_production_governance"
    return "provisional"


def _rank_contract_promotion_basis(rank_contract: str) -> str:
    if rank_contract == "anchor_global_policy_rank_reference":
        return "pre-June walk-forward plus exchange-fixed later block"
    return "June attribution replay"


def _rank_contract_remaining_validation_item(rank_contract: str) -> str:
    if rank_contract == "anchor_global_policy_rank_reference":
        return (
            "Keep using the global-over-time rank reference as the static "
            "baseline while market-state threshold and priority controllers "
            "remain shadow-only until later-window promotion gates pass."
        )
    return (
        "Compare exact T1 against a causal, pre-fitted global-over-time "
        "short_boll rank reference while holding thresholds and portfolio "
        "policy fixed across pre-June walk-forward folds and a later untouched window."
    )


def _attribution_t1(attribution_dir: Path) -> dict[str, Any]:
    summary_path = attribution_dir / "attribution_summary.csv"
    by_head_path = attribution_dir / "attribution_by_head.csv"
    if not summary_path.exists():
        return {"available": False, "reason": f"missing:{summary_path}"}
    summary = pd.read_csv(summary_path)
    t1 = summary.loc[summary["arm"].astype(str).eq("T1_repaired_contract_no_controller")]
    if t1.empty:
        return {"available": False, "reason": "missing_t1_row"}
    result: dict[str, Any] = {
        "available": True,
        "summary_path": str(summary_path),
        "summary": t1.iloc[0].to_dict(),
    }
    if by_head_path.exists():
        by_head = pd.read_csv(by_head_path)
        result["by_head"] = by_head.loc[
            by_head["arm"].astype(str).eq("T1_repaired_contract_no_controller")
        ].to_dict("records")
    return result


def _render_report(
    *,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    attribution: dict[str, Any],
) -> str:
    row = summary.iloc[0]
    ts_min = manifest.get("timestamp_min")
    ts_max = manifest.get("timestamp_max")
    period_label = f"{ts_min} to {ts_max}" if ts_min and ts_max else "evaluation period"
    lines = [
        "# T1 Repaired Static Baseline Decision",
        "",
        "## Stack",
        "",
        "- Active score path: anchor/meta score carried in `calibrated_score`.",
        f"- Rank contract: `{manifest['active_stack']['rank_contract']}`.",
        "- Active heads: `short_asset`, `short_boll`.",
        "- Disabled heads: `long_bars`, `long_dist`.",
        f"- Rank scope: `{manifest['active_stack']['rank_scope']}`.",
        f"- Promotion status: `{manifest['active_stack']['promotion_status']}`.",
        f"- Promotion basis: `{manifest['active_stack']['promotion_basis']}`.",
        "- Thresholds: static base deployment thresholds.",
        "- Auction: `refit_bar4_strategy_bar2` global auction.",
        "- q-fail: disabled.",
        "- Market-state threshold controller: disabled for execution, shadow/logging only.",
        "",
        "## Remaining Validation Item",
        "",
        manifest["active_stack"]["remaining_validation_item"],
        "",
        f"## Replay metrics ({period_label})",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| trades | {int(row['trade_count'])} |",
        f"| net_pnl | {float(row['net_pnl']):.6f} |",
        f"| gross_pnl | {float(row['gross_pnl']):.6f} |",
        f"| cost_pnl | {float(row['cost_pnl']):.6f} |",
        f"| full_sl_rate | {float(row['full_sl_rate']):.6f} |",
        f"| timeout_rate | {float(row['timeout_rate']):.6f} |",
        f"| worst_24h_net_pnl | {float(row['worst_24h_net_pnl']):.6f} |",
        "",
        "## By-head contribution",
        "",
    ]
    if by_head.empty:
        lines.append("No accepted trades.")
    else:
        lines.extend(
            [
                "| head | trades | win_rate | net_pnl | gross_pnl | full_sl_rate | timeout_rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rec in by_head.to_dict("records"):
            lines.append(
                "| {head} | {trade_count:d} | {win_rate:.6f} | {net_pnl:.6f} | "
                "{gross_pnl:.6f} | {full_sl_rate:.6f} | {timeout_rate:.6f} |".format(
                    head=str(rec.get("head")),
                    trade_count=int(rec.get("trade_count", 0)),
                    win_rate=float(rec.get("win_rate", np.nan)),
                    net_pnl=float(rec.get("net_pnl", 0.0)),
                    gross_pnl=float(rec.get("gross_pnl", 0.0)),
                    full_sl_rate=float(rec.get("full_sl_rate", np.nan)),
                    timeout_rate=float(rec.get("timeout_rate", np.nan)),
                )
            )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- T1 parity with attribution row: `{manifest['validation']['matches_attribution_t1']}`.",
            f"- Candidate rows: `{manifest['candidate_rows']}`.",
            f"- Deployable rows before auction: `{manifest['deployable_rows']}`.",
            f"- Accepted decision keys are unique: `{manifest['validation']['accepted_decision_keys_unique']}`.",
            f"- Long heads absent after disable filter: `{manifest['validation']['disabled_heads_absent']}`.",
            f"- q-fail active: `{manifest['active_stack']['qfail_active']}`.",
            f"- market-state threshold controller active: `{manifest['active_stack']['market_state_threshold_controller_active']}`.",
        ]
    )
    if attribution.get("available"):
        lines.extend(
            [
                "",
                "## Attribution reference",
                "",
                "The attribution ablation showed the production lift came from the repaired rank/eligibility contract, not from controller suppression:",
                "",
                "- T1 - T0 net PnL delta: `+207.695480`.",
                "- T2 - T1 observed-controller net PnL delta: `-17.435317`.",
                "- T5 - T2 incremental S3 net PnL delta: `0.000000`.",
                "- Controller defensive success: suppressed 2 T1 winners, loss avoided 0, winner PnL sacrificed 19.795416.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
            f"- Summary: `{manifest['outputs']['summary']}`",
            f"- By head: `{manifest['outputs']['by_head']}`",
            f"- Accepted trades: `{manifest['outputs']['accepted_trades']}`",
            f"- Decisions: `{manifest['outputs']['decisions']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-candidates", type=Path, default=mstc.DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--train-deployable-candidates", type=Path, default=mstc.DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=mstc.DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument(
        "--rank-contract",
        choices=("short_boll_timestamp_rank", "anchor_global_policy_rank_reference"),
        default="anchor_global_policy_rank_reference",
    )
    parser.add_argument("--rank-reference-run-id", default=DEFAULT_RANK_REFERENCE_RUN_ID)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--disable-heads", default="long_bars,long_dist")
    parser.add_argument("--attribution-dir", type=Path, default=DEFAULT_ATTRIBUTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = args.output_dir / "simple_policy_optimiser"
    policy_dir.mkdir(parents=True, exist_ok=True)

    disabled_heads = mstc._parse_disabled_heads(args.disable_heads)
    candidates, eval_rank_diag = _load_for_t1(
        args.eval_candidates,
        rank_contract=str(args.rank_contract),
        disabled_heads=disabled_heads,
        data_root=args.data_root,
        rank_reference_run_id=str(args.rank_reference_run_id),
    )
    train_deployable, train_rank_diag = _load_for_t1(
        args.train_deployable_candidates,
        rank_contract=str(args.rank_contract),
        disabled_heads=disabled_heads,
        data_root=args.data_root,
        rank_reference_run_id=str(args.rank_reference_run_id),
    )
    policy_active_heads = sorted(train_deployable["head"].astype(str).unique().tolist())
    observed_heads = sorted(candidates["head"].astype(str).unique().tolist())
    rank_reference_contract = _rank_reference_contract_report(
        rank_contract=str(args.rank_contract),
        eval_diag=eval_rank_diag,
        train_diag=train_rank_diag,
        eval_rows=int(len(candidates)),
        train_rows=int(len(train_deployable)),
    )
    if not bool(rank_reference_contract.get("passed", False)):
        raise RuntimeError(
            "Global rank-reference contract failed: "
            + ", ".join(map(str, rank_reference_contract.get("failures", [])))
        )
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)
    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(args.market_mode),
    )
    accepted = mstc._accepted_trades(candidates, decisions)
    summary = pd.DataFrame([mstc._metrics_row(ARM_NAME, metrics, accepted, schedule=None)])
    by_head = mstc._by_head(ARM_NAME, accepted)
    deployable = _deployable_candidates(candidates)

    candidates_path = policy_dir / "simple_policy_candidates_broad.parquet"
    deployable_path = policy_dir / "simple_policy_candidates_deployable.parquet"
    decisions_path = policy_dir / "portfolio_decisions.parquet"
    equity_path = policy_dir / "equity_curve.parquet"
    accepted_path = policy_dir / "accepted_trades.parquet"
    summary_path = policy_dir / "portfolio_replay_summary.csv"
    by_head_path = policy_dir / "portfolio_by_head.csv"
    manifest_path = args.output_dir / "t1_repaired_static_baseline_manifest.json"
    report_path = args.output_dir / "t1_repaired_static_baseline_report.md"

    candidates.to_parquet(candidates_path, index=False)
    deployable.to_parquet(deployable_path, index=False)
    # Keep the historical optimiser filename too, because downstream scripts
    # commonly look for this exact path.
    deployable.to_parquet(policy_dir / "simple_policy_candidates.parquet", index=False)
    decisions.to_parquet(decisions_path, index=False)
    equity.to_parquet(equity_path, index=False)
    accepted.to_parquet(accepted_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_head.to_csv(by_head_path, index=False)

    attribution = _attribution_t1(args.attribution_dir)
    validation: dict[str, Any] = {}
    validation["disabled_heads_absent"] = not bool(
        set(candidates.get("head", pd.Series(dtype=str)).astype(str).unique()) & disabled_heads
    )
    key_cols = list(mstc.DECISION_KEY_COLS)
    validation["accepted_decision_keys_unique"] = bool(
        accepted.empty or not accepted.duplicated(key_cols).any()
    )
    validation["active_score_alias_reliability_equals_calibrated"] = None
    if "reliability_blend_score" in candidates.columns and "calibrated_score" in candidates.columns:
        diff = (
            pd.to_numeric(candidates["reliability_blend_score"], errors="coerce")
            - pd.to_numeric(candidates["calibrated_score"], errors="coerce")
        ).abs()
        validation["active_score_alias_reliability_equals_calibrated"] = bool(float(diff.max(skipna=True) or 0.0) <= 1e-12)
    validation["rank_reference_contract"] = rank_reference_contract
    validation["matches_attribution_t1"] = False
    validation["attribution_metric_deltas"] = {}
    if attribution.get("available"):
        left = summary.iloc[0]
        right = pd.Series(attribution["summary"])
        deltas = _metric_delta(
            left,
            right,
            ["trade_count", "net_pnl", "gross_pnl", "cost_pnl", "full_sl_rate", "timeout_rate"],
        )
        validation["attribution_metric_deltas"] = deltas
        validation["matches_attribution_t1"] = all(
            value is not None and abs(float(value)) <= 1e-9
            for value in deltas.values()
        )

    manifest: dict[str, Any] = {
        "generated_by": "materialize_t1_repaired_static_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_stack": {
            "name": "T1_repaired_static_baseline",
            "score_path": "anchor_meta_calibrated_score",
            "active_score_column": "calibrated_score",
            "qfail_active": False,
            "native_reliability_blend_active": False,
            "market_state_threshold_controller_active": False,
            "market_state_shadow_logging_only": True,
            "rank_contract": str(args.rank_contract),
            "rank_reference_run_id": str(args.rank_reference_run_id)
            if str(args.rank_contract) == "anchor_global_policy_rank_reference"
            else None,
            "rank_scope": _rank_contract_scope(str(args.rank_contract)),
            "promotion_status": _rank_contract_promotion_status(str(args.rank_contract)),
            "promotion_basis": _rank_contract_promotion_basis(str(args.rank_contract)),
            "remaining_validation_item": _rank_contract_remaining_validation_item(str(args.rank_contract)),
            "static_base_thresholds": True,
            "policy_variant": str(args.policy_variant),
            "disabled_heads": sorted(disabled_heads),
            "enabled_heads": policy_active_heads,
            "active_heads": policy_active_heads,
            "observed_heads": observed_heads,
            "auction": "global_auction",
            "ev_mapping": "hierarchical_ev_curves_fitted_on_train_deployable",
        },
        "inputs": {
            "eval_candidates": str(args.eval_candidates),
            "eval_candidates_sha256": _sha256(args.eval_candidates),
            "train_deployable_candidates": str(args.train_deployable_candidates),
            "train_deployable_candidates_sha256": _sha256(args.train_deployable_candidates),
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": _sha256(args.policy_manifest),
            "policy_manifest_run_id": policy_payload.get("run_id"),
            "attribution_dir": str(args.attribution_dir),
        },
        "candidate_rows": int(len(candidates)),
        "deployable_rows": int(len(deployable)),
        "accepted_rows": int(len(accepted)),
        "timestamp_min": pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").min(),
        "timestamp_max": pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").max(),
        "summary": summary.iloc[0].to_dict(),
        "by_head": by_head.to_dict("records"),
        "validation": validation,
        "rank_reference_diagnostics": {
            "eval": eval_rank_diag,
            "train_deployable": train_rank_diag,
        },
        "attribution_reference": attribution,
        "outputs": {
            "manifest": str(manifest_path),
            "report": str(report_path),
            "candidates_broad": str(candidates_path),
            "candidates_deployable": str(deployable_path),
            "decisions": str(decisions_path),
            "equity_curve": str(equity_path),
            "accepted_trades": str(accepted_path),
            "summary": str(summary_path),
            "by_head": str(by_head_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    report_path.write_text(
        _render_report(manifest=manifest, summary=summary, by_head=by_head, attribution=attribution),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "summary": summary.iloc[0].to_dict(), "validation": validation}), indent=2))


if __name__ == "__main__":
    main()
