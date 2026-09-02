#!/usr/bin/env python3
"""Compare fixed-policy T1 rank-contract artifacts.

The intended use is to compare the measured T1 timestamp-rank contract against
the causal global-over-time rank-reference challenger while keeping the score,
thresholds, disabled heads, cost model, EV mapping, and auction policy fixed.
This is an artifact-only report; it does not rerun replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUMMARY_COLS = (
    "trade_count",
    "net_pnl",
    "gross_pnl",
    "cost_pnl",
    "full_sl_rate",
    "timeout_rate",
    "worst_24h_net_pnl",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(root: Path) -> dict[str, Any]:
    return _read_json(root / "t1_repaired_static_baseline_manifest.json")


def _policy_dir(root: Path) -> Path:
    return root / "simple_policy_optimiser"


def _summary(root: Path, name: str) -> pd.DataFrame:
    path = _policy_dir(root) / "portfolio_replay_summary.csv"
    df = pd.read_csv(path)
    df.insert(0, "contract_name", name)
    return df


def _by_head(root: Path, name: str) -> pd.DataFrame:
    path = _policy_dir(root) / "portfolio_by_head.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "contract_name", name)
    return df


def _accepted(root: Path) -> pd.DataFrame:
    path = _policy_dir(root) / "accepted_trades.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _deployable(root: Path, name: str) -> pd.DataFrame:
    path = _policy_dir(root) / "simple_policy_candidates_deployable.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "head" in df.columns:
        rows = []
        for head, group in df.groupby("head", sort=True):
            rows.append(
                {
                    "contract_name": name,
                    "head": str(head),
                    "deployable_rows": int(len(group)),
                    "timestamp_count": int(
                        pd.to_datetime(group["timestamp"], utc=True, errors="coerce").nunique()
                    )
                    if "timestamp" in group.columns
                    else None,
                    "mean_rank": float(
                        pd.to_numeric(group.get("normalized_rank_score"), errors="coerce").mean()
                    )
                    if "normalized_rank_score" in group.columns
                    else np.nan,
                    "min_rank": float(
                        pd.to_numeric(group.get("normalized_rank_score"), errors="coerce").min()
                    )
                    if "normalized_rank_score" in group.columns
                    else np.nan,
                    "max_rank": float(
                        pd.to_numeric(group.get("normalized_rank_score"), errors="coerce").max()
                    )
                    if "normalized_rank_score" in group.columns
                    else np.nan,
                }
            )
        return pd.DataFrame(rows)
    return pd.DataFrame(
        [{"contract_name": name, "head": "all", "deployable_rows": int(len(df))}]
    )


def _candidate_universe(root: Path, name: str) -> tuple[pd.DataFrame, set[str]]:
    path = _policy_dir(root) / "simple_policy_candidates_broad.parquet"
    if not path.exists():
        return (
            pd.DataFrame(
                [
                    {
                        "contract_name": name,
                        "path": str(path),
                        "exists": False,
                        "rows": 0,
                        "unique_decision_keys": 0,
                        "duplicate_decision_keys": 0,
                        "timestamp_count": 0,
                        "heads": "",
                        "sha256": "",
                    }
                ]
            ),
            set(),
        )
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    keys = set(_decision_key(df)) if not df.empty else set()
    duplicate_count = int(len(df) - len(keys))
    heads = ",".join(sorted(df.get("head", pd.Series(dtype=object)).dropna().astype(str).unique()))
    timestamp_count = (
        int(pd.to_datetime(df["timestamp"], utc=True, errors="coerce").nunique())
        if "timestamp" in df.columns
        else 0
    )
    by_head_rows: dict[str, int] = {}
    if "head" in df.columns:
        by_head_rows = {str(k): int(v) for k, v in df.groupby("head", sort=True).size().items()}
    return (
        pd.DataFrame(
            [
                {
                    "contract_name": name,
                    "path": str(path),
                    "exists": True,
                    "rows": int(len(df)),
                    "unique_decision_keys": int(len(keys)),
                    "duplicate_decision_keys": duplicate_count,
                    "timestamp_count": timestamp_count,
                    "heads": heads,
                    "by_head_rows_json": json.dumps(by_head_rows, sort_keys=True),
                    "sha256": _sha256(path),
                }
            ]
        ),
        keys,
    )


def _decision_key(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in ("timestamp", "symbol", "strategy_id", "side", "head") if col in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df)), index=df.index).astype(str)
    values = []
    for col in cols:
        if col == "timestamp":
            value = pd.to_datetime(df[col], utc=True, errors="coerce").astype(str)
        else:
            value = df[col].astype(str)
        values.append(value)
    out = values[0]
    for value in values[1:]:
        out = out.str.cat(value, sep="|")
    return out


def _overlap(base: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, Any]:
    if base.empty and challenger.empty:
        return {
            "base_accepted": 0,
            "challenger_accepted": 0,
            "intersection": 0,
            "union": 0,
            "jaccard": 1.0,
            "base_only": 0,
            "challenger_only": 0,
        }
    base_keys = set(_decision_key(base))
    challenger_keys = set(_decision_key(challenger))
    union = base_keys | challenger_keys
    inter = base_keys & challenger_keys
    return {
        "base_accepted": int(len(base_keys)),
        "challenger_accepted": int(len(challenger_keys)),
        "intersection": int(len(inter)),
        "union": int(len(union)),
        "jaccard": float(len(inter) / len(union)) if union else 1.0,
        "base_only": int(len(base_keys - challenger_keys)),
        "challenger_only": int(len(challenger_keys - base_keys)),
    }


def _removed_added_pnl(base: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, float]:
    if base.empty and challenger.empty:
        return {
            "removed_count": 0.0,
            "added_count": 0.0,
            "removed_net_pnl": 0.0,
            "added_net_pnl": 0.0,
            "removed_winner_pnl": 0.0,
            "removed_loser_loss": 0.0,
            "added_winner_pnl": 0.0,
            "added_loser_loss": 0.0,
        }
    base = base.copy()
    challenger = challenger.copy()
    base["_key"] = _decision_key(base)
    challenger["_key"] = _decision_key(challenger)
    challenger_keys = set(challenger["_key"])
    base_keys = set(base["_key"])
    removed = base.loc[~base["_key"].isin(challenger_keys)]
    added = challenger.loc[~challenger["_key"].isin(base_keys)]

    def _pnl(frame: pd.DataFrame) -> pd.Series:
        if "net_pnl" in frame.columns:
            return pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0.0)
        if "net_return" in frame.columns:
            return pd.to_numeric(frame["net_return"], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=frame.index)

    removed_pnl = _pnl(removed)
    added_pnl = _pnl(added)
    return {
        "removed_count": float(len(removed)),
        "added_count": float(len(added)),
        "removed_net_pnl": float(removed_pnl.sum()),
        "added_net_pnl": float(added_pnl.sum()),
        "removed_winner_pnl": float(removed_pnl.clip(lower=0.0).sum()),
        "removed_loser_loss": float((-removed_pnl.clip(upper=0.0)).sum()),
        "added_winner_pnl": float(added_pnl.clip(lower=0.0).sum()),
        "added_loser_loss": float((-added_pnl.clip(upper=0.0)).sum()),
    }


def _delta(base_summary: pd.DataFrame, challenger_summary: pd.DataFrame) -> pd.DataFrame:
    base = base_summary.iloc[0]
    challenger = challenger_summary.iloc[0]
    rows = []
    for col in SUMMARY_COLS:
        if col not in base.index or col not in challenger.index:
            continue
        left = pd.to_numeric(pd.Series([challenger[col]]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([base[col]]), errors="coerce").iloc[0]
        rows.append({"metric": col, "challenger_minus_base": float(left - right)})
    return pd.DataFrame(rows)


def _head_delta(base_by_head: pd.DataFrame, challenger_by_head: pd.DataFrame) -> pd.DataFrame:
    if base_by_head.empty and challenger_by_head.empty:
        return pd.DataFrame()
    cols = [col for col in SUMMARY_COLS if col in set(base_by_head.columns) | set(challenger_by_head.columns)]
    base = base_by_head.set_index("head") if "head" in base_by_head.columns else pd.DataFrame()
    challenger = challenger_by_head.set_index("head") if "head" in challenger_by_head.columns else pd.DataFrame()
    heads = sorted(set(base.index.astype(str)) | set(challenger.index.astype(str)))
    rows = []
    for head in heads:
        rec: dict[str, Any] = {"head": head}
        for col in cols:
            b = pd.to_numeric(pd.Series([base[col].get(head, 0.0) if col in base.columns and head in base.index else 0.0]), errors="coerce").iloc[0]
            c = pd.to_numeric(pd.Series([challenger[col].get(head, 0.0) if col in challenger.columns and head in challenger.index else 0.0]), errors="coerce").iloc[0]
            rec[f"delta_{col}"] = float(c - b)
        rows.append(rec)
    return pd.DataFrame(rows)


def _stack_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    stack = dict(manifest.get("active_stack") or {})
    return {
        "score_path": stack.get("score_path"),
        "active_score_column": stack.get("active_score_column"),
        "static_base_thresholds": bool(stack.get("static_base_thresholds") is True),
        "policy_variant": stack.get("policy_variant"),
        "active_heads": list(stack.get("enabled_heads") or stack.get("active_heads") or []),
        "disabled_heads": list(stack.get("disabled_heads") or []),
        "auction": stack.get("auction"),
        "ev_mapping": stack.get("ev_mapping"),
        "qfail_active": bool(stack.get("qfail_active") is True),
        "native_reliability_blend_active": bool(stack.get("native_reliability_blend_active") is True),
        "market_state_threshold_controller_active": bool(stack.get("market_state_threshold_controller_active") is True),
        "market_state_shadow_logging_only": bool(stack.get("market_state_shadow_logging_only") is True),
    }


def _rank_reference_contract_failures(manifest: dict[str, Any], *, prefix: str) -> list[str]:
    stack = dict(manifest.get("active_stack") or {})
    if stack.get("rank_contract") != "anchor_global_policy_rank_reference":
        return []
    validation = dict(manifest.get("validation") or {})
    contract = validation.get("rank_reference_contract")
    failures: list[str] = []
    if not isinstance(contract, dict):
        return [f"{prefix}.rank_reference_contract_missing"]
    if contract.get("required") is not True:
        failures.append(f"{prefix}.rank_reference_contract_not_required")
    if contract.get("passed") is not True:
        failures.append(f"{prefix}.rank_reference_contract_not_passed")
    if contract.get("failures") not in ([], None):
        failures.append(f"{prefix}.rank_reference_contract_failures_not_empty")
    diagnostics = manifest.get("rank_reference_diagnostics")
    if not isinstance(diagnostics, dict):
        failures.append(f"{prefix}.rank_reference_diagnostics_missing")
    else:
        for split in ("eval", "train_deployable"):
            diag = diagnostics.get(split)
            if not isinstance(diag, dict):
                failures.append(f"{prefix}.{split}_rank_reference_diag_missing")
                continue
            if diag.get("rank_source") != "policy_rank_reference_percentile":
                failures.append(f"{prefix}.{split}_rank_source_not_policy_reference")
            if int(diag.get("missing_rank_rows") or 0) != 0:
                failures.append(f"{prefix}.{split}_missing_policy_rank_rows_nonzero")
            if int(diag.get("missing_auction_rank_rows") or 0) != 0:
                failures.append(f"{prefix}.{split}_missing_auction_rank_rows_nonzero")
            if bool(diag.get("window_rank_debug_used")):
                failures.append(f"{prefix}.{split}_window_rank_debug_used")
    return failures


def _comparison_contract(
    *,
    base_dir: Path,
    challenger_dir: Path,
    base_name: str,
    challenger_name: str,
    base_manifest: dict[str, Any],
    challenger_manifest: dict[str, Any],
    candidate_universe: pd.DataFrame,
    candidate_universe_overlap: dict[str, Any],
) -> dict[str, Any]:
    base_stack = dict(base_manifest.get("active_stack") or {})
    challenger_stack = dict(challenger_manifest.get("active_stack") or {})
    base_contract = _stack_contract(base_manifest)
    challenger_contract = _stack_contract(challenger_manifest)
    failures: list[str] = []
    if base_stack.get("rank_contract") != "short_boll_timestamp_rank":
        failures.append("base_rank_contract_not_short_boll_timestamp_rank")
    if base_stack.get("rank_scope") != "within_timestamp":
        failures.append("base_rank_scope_not_within_timestamp")
    if challenger_stack.get("rank_contract") != "anchor_global_policy_rank_reference":
        failures.append("challenger_rank_contract_not_anchor_global_policy_rank_reference")
    if challenger_stack.get("rank_scope") != "global_over_time":
        failures.append("challenger_rank_scope_not_global_over_time")
    failures.extend(_rank_reference_contract_failures(challenger_manifest, prefix="challenger"))
    comparable_keys = [
        "score_path",
        "active_score_column",
        "static_base_thresholds",
        "policy_variant",
        "active_heads",
        "disabled_heads",
        "auction",
        "ev_mapping",
        "qfail_active",
        "native_reliability_blend_active",
        "market_state_threshold_controller_active",
    ]
    for key in comparable_keys:
        left = base_contract.get(key)
        right = challenger_contract.get(key)
        if isinstance(left, list):
            left = sorted(map(str, left))
        if isinstance(right, list):
            right = sorted(map(str, right))
        if left != right:
            failures.append(f"fixed_policy_contract_mismatch_{key}")
    expected_stack = {
        "score_path": "anchor_meta_calibrated_score",
        "active_score_column": "calibrated_score",
        "static_base_thresholds": True,
        "policy_variant": "refit_bar4_strategy_bar2",
        "active_heads": ["short_asset", "short_boll"],
        "disabled_heads": ["long_bars", "long_dist"],
        "auction": "global_auction",
        "ev_mapping": "hierarchical_ev_curves_fitted_on_train_deployable",
        "qfail_active": False,
        "native_reliability_blend_active": False,
        "market_state_threshold_controller_active": False,
    }
    for key, expected_value in expected_stack.items():
        actual = base_contract.get(key)
        if isinstance(actual, list):
            actual = sorted(map(str, actual))
            expected_value = sorted(map(str, expected_value)) if isinstance(expected_value, list) else expected_value
        if actual != expected_value:
            failures.append(f"fixed_policy_contract_unexpected_{key}")
    if not bool(candidate_universe_overlap.get("identical")):
        failures.append("candidate_universe_not_identical")
    if int(candidate_universe_overlap.get("base_duplicate_decision_keys") or 0) != 0:
        failures.append("base_candidate_universe_duplicate_decision_keys")
    if int(candidate_universe_overlap.get("challenger_duplicate_decision_keys") or 0) != 0:
        failures.append("challenger_candidate_universe_duplicate_decision_keys")
    fixed_policy_contract = dict(expected_stack)
    fixed_policy_contract["rank_contract_is_the_only_arm_difference"] = True
    fixed_policy_contract["candidate_universe_fixed"] = bool(candidate_universe_overlap.get("identical"))
    return {
        "generated_by": "compare_t1_rank_contracts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "later_fixed_contract_timestamp_vs_global_rank_validation",
        "base": {
            "name": base_name,
            "dir": str(base_dir),
            "manifest": str(base_dir / "t1_repaired_static_baseline_manifest.json"),
            "manifest_sha256": _sha256(base_dir / "t1_repaired_static_baseline_manifest.json"),
            "rank_contract": base_stack.get("rank_contract"),
            "rank_scope": base_stack.get("rank_scope"),
            "promotion_status": base_stack.get("promotion_status"),
            "stack_contract": base_contract,
        },
        "challenger": {
            "name": challenger_name,
            "dir": str(challenger_dir),
            "manifest": str(challenger_dir / "t1_repaired_static_baseline_manifest.json"),
            "manifest_sha256": _sha256(challenger_dir / "t1_repaired_static_baseline_manifest.json"),
            "rank_contract": challenger_stack.get("rank_contract"),
            "rank_scope": challenger_stack.get("rank_scope"),
            "promotion_status": challenger_stack.get("promotion_status"),
            "rank_reference_run_id": challenger_stack.get("rank_reference_run_id"),
            "rank_reference_contract": (
                dict(challenger_manifest.get("validation") or {}).get("rank_reference_contract")
            ),
            "rank_reference_diagnostics": challenger_manifest.get("rank_reference_diagnostics"),
            "stack_contract": challenger_contract,
        },
        "fixed_policy_contract": fixed_policy_contract,
        "candidate_universe": {
            "rows": candidate_universe.to_dict("records"),
            "overlap": candidate_universe_overlap,
        },
        "validation": {
            "passed": not failures,
            "failures": failures,
            "rank_contract_is_the_only_arm_difference": not failures,
            "score_threshold_auction_ev_costs_fixed": not failures,
            "candidate_universe_identical": bool(candidate_universe_overlap.get("identical")),
            "controller_and_qfail_disabled": (
                base_contract.get("qfail_active") is False
                and challenger_contract.get("qfail_active") is False
                and base_contract.get("market_state_threshold_controller_active") is False
                and challenger_contract.get("market_state_threshold_controller_active") is False
            ),
        },
    }


def _render_report(
    *,
    output_dir: Path,
    base_name: str,
    challenger_name: str,
    base_manifest: dict[str, Any],
    challenger_manifest: dict[str, Any],
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    by_head: pd.DataFrame,
    head_delta: pd.DataFrame,
    deployable: pd.DataFrame,
    candidate_universe: pd.DataFrame,
    overlap: dict[str, Any],
    swap: dict[str, float],
    comparison_manifest: dict[str, Any],
) -> str:
    lines = [
        "# T1 Rank-Contract Comparison",
        "",
        "This report compares saved fixed-policy T1 artifacts. It does not rerun replay or change the active baseline.",
        "",
        "## Contracts",
        "",
        f"- Base `{base_name}`: `{base_manifest['active_stack']['rank_contract']}` / `{base_manifest['active_stack']['rank_scope']}` / status `{base_manifest['active_stack']['promotion_status']}`.",
        f"- Challenger `{challenger_name}`: `{challenger_manifest['active_stack']['rank_contract']}` / `{challenger_manifest['active_stack']['rank_scope']}` / status `{challenger_manifest['active_stack']['promotion_status']}`.",
        f"- Fixed-policy contract passed: `{comparison_manifest['validation']['passed']}`.",
        "",
        "## Summary Metrics",
        "",
        "| contract | trades | net_pnl | gross_pnl | cost_pnl | full_sl_rate | timeout_rate | worst_24h_net_pnl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['contract_name']} | {int(row['trade_count'])} | "
            f"{float(row['net_pnl']):.6f} | {float(row['gross_pnl']):.6f} | "
            f"{float(row['cost_pnl']):.6f} | {float(row['full_sl_rate']):.6f} | "
            f"{float(row['timeout_rate']):.6f} | {float(row['worst_24h_net_pnl']):.6f} |"
        )
    lines.extend(["", "## Challenger Minus Base", "", "| metric | delta |", "|---|---:|"])
    for _, row in delta.iterrows():
        lines.append(f"| {row['metric']} | {float(row['challenger_minus_base']):.6f} |")
    lines.extend(
        [
            "",
            "## Accepted-Set Change",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| base accepted | {overlap['base_accepted']} |",
            f"| challenger accepted | {overlap['challenger_accepted']} |",
            f"| intersection | {overlap['intersection']} |",
            f"| union | {overlap['union']} |",
            f"| Jaccard | {overlap['jaccard']:.6f} |",
            f"| base-only trades | {overlap['base_only']} |",
            f"| challenger-only trades | {overlap['challenger_only']} |",
            f"| removed net PnL | {swap['removed_net_pnl']:.6f} |",
            f"| added net PnL | {swap['added_net_pnl']:.6f} |",
            f"| removed winner PnL | {swap['removed_winner_pnl']:.6f} |",
            f"| removed loser loss | {swap['removed_loser_loss']:.6f} |",
            f"| added winner PnL | {swap['added_winner_pnl']:.6f} |",
            f"| added loser loss | {swap['added_loser_loss']:.6f} |",
            "",
            "## By-Head Metrics",
            "",
            "| contract | head | trades | win_rate | net_pnl | full_sl_rate | timeout_rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_head.iterrows():
        lines.append(
            f"| {row['contract_name']} | {row['head']} | {int(row['trade_count'])} | "
            f"{float(row['win_rate']):.6f} | {float(row['net_pnl']):.6f} | "
            f"{float(row['full_sl_rate']):.6f} | {float(row['timeout_rate']):.6f} |"
        )
    lines.extend(["", "## By-Head Deltas", ""])
    if head_delta.empty:
        lines.append("No by-head delta rows.")
    else:
        lines.extend(["| head | delta_trades | delta_net_pnl | delta_full_sl_rate |", "|---|---:|---:|---:|"])
        for _, row in head_delta.iterrows():
            lines.append(
                f"| {row['head']} | {float(row.get('delta_trade_count', 0.0)):.6f} | "
                f"{float(row.get('delta_net_pnl', 0.0)):.6f} | "
                f"{float(row.get('delta_full_sl_rate', 0.0)):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Deployable Rows",
            "",
            "| contract | head | deployable_rows | timestamp_count | mean_rank | min_rank | max_rank |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in deployable.iterrows():
        lines.append(
            f"| {row['contract_name']} | {row['head']} | {int(row['deployable_rows'])} | "
            f"{'' if pd.isna(row.get('timestamp_count')) else int(row['timestamp_count'])} | "
            f"{float(row.get('mean_rank', np.nan)):.6f} | "
            f"{float(row.get('min_rank', np.nan)):.6f} | "
            f"{float(row.get('max_rank', np.nan)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Universe",
            "",
            "| contract | broad rows | unique keys | duplicate keys | timestamps | heads |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in candidate_universe.iterrows():
        lines.append(
            f"| {row['contract_name']} | {int(row['rows'])} | "
            f"{int(row['unique_decision_keys'])} | {int(row['duplicate_decision_keys'])} | "
            f"{int(row['timestamp_count'])} | {row.get('heads', '')} |"
        )
    universe_overlap = comparison_manifest.get("candidate_universe", {}).get("overlap", {})
    lines.extend(
        [
            "",
            "| universe metric | value |",
            "|---|---:|",
            f"| identical | `{universe_overlap.get('identical')}` |",
            f"| base-only keys | {int(universe_overlap.get('base_only_keys') or 0)} |",
            f"| challenger-only keys | {int(universe_overlap.get('challenger_only_keys') or 0)} |",
            f"| key Jaccard | {float(universe_overlap.get('jaccard') or 0.0):.6f} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The global-over-time rank challenger is causal and fitted from the pre-June reference, but this June replay alone is not promotion evidence. Use it as a fixed-contract challenger; promotion still requires pre-June walk-forward validation and a later untouched matured period.",
            "",
            "Generated files:",
            f"- `{output_dir / 'rank_contract_summary.csv'}`",
            f"- `{output_dir / 'rank_contract_delta.csv'}`",
            f"- `{output_dir / 'rank_contract_by_head.csv'}`",
            f"- `{output_dir / 'rank_contract_by_head_delta.csv'}`",
            f"- `{output_dir / 'rank_contract_deployable_rows.csv'}`",
            f"- `{output_dir / 'rank_contract_candidate_universe.csv'}`",
            f"- `{output_dir / 'rank_contract_accepted_overlap.json'}`",
            f"- `{output_dir / 'rank_contract_comparison_manifest.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_rank_contracts(
    *,
    base_dir: Path,
    challenger_dir: Path,
    output_dir: Path,
    base_name: str,
    challenger_name: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = _manifest(base_dir)
    challenger_manifest = _manifest(challenger_dir)
    base_summary = _summary(base_dir, base_name)
    challenger_summary = _summary(challenger_dir, challenger_name)
    summary = pd.concat([base_summary, challenger_summary], ignore_index=True)
    delta = _delta(base_summary, challenger_summary)
    base_by_head = _by_head(base_dir, base_name)
    challenger_by_head = _by_head(challenger_dir, challenger_name)
    by_head = pd.concat([base_by_head, challenger_by_head], ignore_index=True)
    head_delta = _head_delta(base_by_head, challenger_by_head)
    deployable = pd.concat(
        [_deployable(base_dir, base_name), _deployable(challenger_dir, challenger_name)],
        ignore_index=True,
    )
    base_universe, base_universe_keys = _candidate_universe(base_dir, base_name)
    challenger_universe, challenger_universe_keys = _candidate_universe(challenger_dir, challenger_name)
    candidate_universe = pd.concat([base_universe, challenger_universe], ignore_index=True)
    universe_union = base_universe_keys | challenger_universe_keys
    universe_intersection = base_universe_keys & challenger_universe_keys
    base_dup = int(base_universe["duplicate_decision_keys"].iloc[0]) if not base_universe.empty else 0
    challenger_dup = (
        int(challenger_universe["duplicate_decision_keys"].iloc[0])
        if not challenger_universe.empty
        else 0
    )
    candidate_universe_overlap = {
        "base_keys": int(len(base_universe_keys)),
        "challenger_keys": int(len(challenger_universe_keys)),
        "intersection_keys": int(len(universe_intersection)),
        "union_keys": int(len(universe_union)),
        "base_only_keys": int(len(base_universe_keys - challenger_universe_keys)),
        "challenger_only_keys": int(len(challenger_universe_keys - base_universe_keys)),
        "jaccard": float(len(universe_intersection) / len(universe_union)) if universe_union else 1.0,
        "base_duplicate_decision_keys": base_dup,
        "challenger_duplicate_decision_keys": challenger_dup,
        "identical": (
            base_universe_keys == challenger_universe_keys
            and base_dup == 0
            and challenger_dup == 0
            and bool(base_universe["exists"].iloc[0])
            and bool(challenger_universe["exists"].iloc[0])
        ),
    }
    base_accepted = _accepted(base_dir)
    challenger_accepted = _accepted(challenger_dir)
    overlap = _overlap(base_accepted, challenger_accepted)
    swap = _removed_added_pnl(base_accepted, challenger_accepted)
    comparison_manifest = _comparison_contract(
        base_dir=base_dir,
        challenger_dir=challenger_dir,
        base_name=base_name,
        challenger_name=challenger_name,
        base_manifest=base_manifest,
        challenger_manifest=challenger_manifest,
        candidate_universe=candidate_universe,
        candidate_universe_overlap=candidate_universe_overlap,
    )

    paths = {
        "manifest": output_dir / "rank_contract_comparison_manifest.json",
        "summary": output_dir / "rank_contract_summary.csv",
        "delta": output_dir / "rank_contract_delta.csv",
        "by_head": output_dir / "rank_contract_by_head.csv",
        "head_delta": output_dir / "rank_contract_by_head_delta.csv",
        "deployable": output_dir / "rank_contract_deployable_rows.csv",
        "candidate_universe": output_dir / "rank_contract_candidate_universe.csv",
        "overlap": output_dir / "rank_contract_accepted_overlap.json",
        "report": output_dir / "t1_rank_contract_comparison_report.md",
    }
    summary.to_csv(paths["summary"], index=False)
    delta.to_csv(paths["delta"], index=False)
    by_head.to_csv(paths["by_head"], index=False)
    head_delta.to_csv(paths["head_delta"], index=False)
    deployable.to_csv(paths["deployable"], index=False)
    candidate_universe.to_csv(paths["candidate_universe"], index=False)
    paths["overlap"].write_text(
        json.dumps({"overlap": overlap, "swap_pnl": swap}, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["manifest"].write_text(json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(
        _render_report(
            output_dir=output_dir,
            base_name=base_name,
            challenger_name=challenger_name,
            base_manifest=base_manifest,
            challenger_manifest=challenger_manifest,
            summary=summary,
            delta=delta,
            by_head=by_head,
            head_delta=head_delta,
            deployable=deployable,
            candidate_universe=candidate_universe,
            overlap=overlap,
            swap=swap,
            comparison_manifest=comparison_manifest,
        ),
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--challenger-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-name", default="timestamp_rank_t1")
    parser.add_argument("--challenger-name", default="global_rank_challenger")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = compare_rank_contracts(
        base_dir=args.base_dir,
        challenger_dir=args.challenger_dir,
        output_dir=args.output_dir,
        base_name=args.base_name,
        challenger_name=args.challenger_name,
    )
    print(f"Wrote rank-contract comparison report: {paths['report']}")


if __name__ == "__main__":
    main()
