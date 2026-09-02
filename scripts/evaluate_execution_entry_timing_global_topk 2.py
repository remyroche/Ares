#!/usr/bin/env python3
"""Evaluate timing heads and actions after one pooled causal EV-map top-k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _auc(target: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    if valid.sum() < 2 or np.unique(target[valid]).size < 2:
        return np.nan
    return float(roc_auc_score(target[valid], score[valid]))


def _brier(target: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    if not valid.any():
        return np.nan
    return float(brier_score_loss(target[valid], np.clip(score[valid], 0.0, 1.0)))


def _spearman(target: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    if valid.sum() < 3 or np.unique(target[valid]).size < 2 or np.unique(score[valid]).size < 2:
        return np.nan
    return float(spearmanr(target[valid], score[valid]).statistic)


def _head_metrics(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
        for action_id, part in actions.groupby("action_id", sort=True):
            filled = part["filled"].astype(bool).to_numpy()
            fill_target = part["fill_indicator"].to_numpy(dtype=float)
            fill_score = _finite(part[f"oof_{arm}_fill_probability"])
            adverse_target = part.loc[filled, "adverse_first"].astype(float).to_numpy()
            adverse_score = _finite(
                part.loc[filled, f"oof_{arm}_adverse_first_probability"]
            )
            delta_target = _finite(part.loc[filled, "filled_delta_ev_vs_now"])
            delta_score = _finite(part.loc[filled, f"oof_{arm}_filled_delta_ev"])
            utility_target = _finite(part["action_realized_utility"])
            utility_score = _finite(part[f"oof_{arm}_expected_action_ev"])
            valid_utility = np.isfinite(utility_target) & np.isfinite(utility_score)
            valid_delta = np.isfinite(delta_target) & np.isfinite(delta_score)
            rows.append(
                {
                    "arm": arm,
                    "action_id": action_id,
                    "rows": int(len(part)),
                    "realized_fill_rate": float(fill_target.mean()),
                    "fill_auc": _auc(fill_target, fill_score),
                    "fill_brier": _brier(fill_target, fill_score),
                    "filled_rows": int(filled.sum()),
                    "adverse_rate_if_filled": float(np.mean(adverse_target))
                    if len(adverse_target)
                    else np.nan,
                    "adverse_auc_if_filled": _auc(adverse_target, adverse_score),
                    "adverse_brier_if_filled": _brier(adverse_target, adverse_score),
                    "delta_ev_spearman_if_filled": _spearman(delta_target, delta_score),
                    "delta_ev_mae_if_filled": float(
                        np.mean(np.abs(delta_target[valid_delta] - delta_score[valid_delta]))
                    )
                    if valid_delta.any()
                    else np.nan,
                    "utility_spearman": _spearman(utility_target, utility_score),
                    "utility_mae": float(
                        np.mean(
                            np.abs(
                                utility_target[valid_utility]
                                - utility_score[valid_utility]
                            )
                        )
                    )
                    if valid_utility.any()
                    else np.nan,
                    "utility_bias": float(
                        np.mean(utility_score[valid_utility] - utility_target[valid_utility])
                    )
                    if valid_utility.any()
                    else np.nan,
                    "realized_utility_bps": float(10_000.0 * np.mean(utility_target)),
                    "enter_now_ev_bps": float(
                        10_000.0 * np.mean(_finite(part["enter_now_net_ev"]))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _policy_metrics(
    frame: pd.DataFrame, *, mapping_col: str, top_k_fraction: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    scopes.extend(
        ("month", str(month), part)
        for month, part in frame.groupby("month", sort=True)
    )
    scopes.extend(
        ("side", str(side), part) for side, part in frame.groupby("side", sort=True)
    )
    for scope, value, part in scopes:
        selected_count = max(1, int(np.ceil(top_k_fraction * len(part))))
        admitted = part.nlargest(selected_count, mapping_col)
        for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
            action = _finite(admitted[f"oof_{arm}_realized_action_utility"])
            enter = _finite(admitted[f"oof_{arm}_enter_now_ev"])
            valid = np.isfinite(action) & np.isfinite(enter)
            chosen = admitted[f"oof_{arm}_recommended_action_id"].astype(str)
            rows.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "scope_value": value,
                    "eligible_rows": int(len(part)),
                    "admitted_rows": int(len(admitted)),
                    "ranking_scope": "one_pooled_global_top_k_within_reported_scope",
                    "per_timestamp_quota": False,
                    "mapping": mapping_col,
                    "action_ev_bps": float(10_000.0 * np.mean(action[valid])),
                    "enter_now_ev_bps": float(10_000.0 * np.mean(enter[valid])),
                    "delta_vs_enter_now_bps": float(
                        10_000.0 * np.mean(action[valid] - enter[valid])
                    ),
                    "fill_rate": float(
                        admitted[f"oof_{arm}_realized_fill"].astype(bool).mean()
                    ),
                    "missed_profitable_trade_rate": float(
                        admitted[f"oof_{arm}_missed_profitable_trade"]
                        .astype(bool)
                        .mean()
                    ),
                    "adverse_first_rate_if_filled": float(
                        admitted.loc[
                            admitted[f"oof_{arm}_realized_fill"].astype(bool),
                            f"oof_{arm}_realized_adverse_first",
                        ]
                        .astype(bool)
                        .mean()
                    ),
                    "enter_now_share": float(chosen.eq("enter_now").mean()),
                    "wait_market_60m_share": float(
                        chosen.eq("wait_market_60m").mean()
                    ),
                    "wait_market_180m_share": float(
                        chosen.eq("wait_market_180m").mean()
                    ),
                    "adverse_limit_share": float(
                        chosen.str.startswith("adverse_limit_").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ValueError("output directory already exists")
    handoff = pd.read_parquet(args.handoff)
    recommendations = pd.read_parquet(args.recommendations)
    actions = pd.read_parquet(args.actions)
    if len(handoff) != len(recommendations):
        raise ValueError("handoff and recommendations must have identical row counts")
    identity = handoff.loc[:, IDENTITY].reset_index(drop=True)
    if not identity["side_name"].astype(str).eq(
        recommendations["side"].astype(str).reset_index(drop=True)
    ).all():
        raise ValueError("recommendation row order does not match the handoff")
    rec = pd.concat(
        [identity, recommendations.reset_index(drop=True)], axis=1
    )
    mapped = pd.read_parquet(args.mapped_oof)
    if mapped.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("mapped OOF artifact has duplicate identities")
    required_map = [*IDENTITY, args.mapping_col]
    missing = sorted(set(required_map).difference(mapped.columns))
    if missing:
        raise ValueError(f"mapped OOF artifact is missing {missing}")
    joined = rec.merge(
        mapped.loc[:, required_map],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    joined[args.mapping_col] = pd.to_numeric(joined[args.mapping_col], errors="coerce")
    joined = joined.loc[joined[args.mapping_col].notna()].copy()
    if joined.empty:
        raise ValueError("no timing recommendations have a causal mapped EV")
    head = _head_metrics(actions)
    policy = _policy_metrics(
        joined, mapping_col=args.mapping_col, top_k_fraction=args.top_k_fraction
    )
    args.output_dir.mkdir(parents=True)
    head_path = args.output_dir / "per_action_head_metrics.csv"
    policy_path = args.output_dir / "mapped_global_topk_policy_metrics.csv"
    joined_path = args.output_dir / "mapped_timing_recommendations.parquet"
    head.to_csv(head_path, index=False)
    policy.to_csv(policy_path, index=False)
    joined.to_parquet(joined_path, index=False, compression="zstd")
    report = {
        "schema": "execution_entry_timing_global_topk_evaluation_v1",
        "rows": {
            "timing_oof": int(len(rec)),
            "causal_mapped_intersection": int(len(joined)),
        },
        "contract": {
            "mapping": args.mapping_col,
            "top_k_fraction": float(args.top_k_fraction),
            "ranking": "pooled globally across timestamps and sides; never per timestamp",
            "action_layer": "applied only after mapped EV admission; never reranks EV",
        },
        "overall": policy.query(
            "scope == 'overall' and scope_value == 'all'"
        ).to_dict("records"),
        "files": {
            "head_metrics": head_path.name,
            "policy_metrics": policy_path.name,
            "recommendations": joined_path.name,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "report": report_path,
        "head_metrics": head_path,
        "policy_metrics": policy_path,
        "recommendations": joined_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--mapped-oof", type=Path, required=True)
    parser.add_argument("--mapping-col", default="causal_recent_side_isotonic_ev")
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    try:
        paths = run(_parser().parse_args())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"timing global-top-k evaluation failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
