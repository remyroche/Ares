#!/usr/bin/env python3
"""Ablate frozen conservative timing gates on the mapped global top-k book."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


GATES: tuple[dict[str, Any], ...] = (
    {"name": "enter_now_only", "kinds": ("enter_now",)},
    {
        "name": "unrestricted",
        "kinds": ("enter_now", "wait_market", "adverse_limit"),
    },
    {"name": "market_wait_only", "kinds": ("enter_now", "wait_market")},
    {"name": "limit_only", "kinds": ("enter_now", "adverse_limit")},
    {
        "name": "sixty_minute_only",
        "kinds": ("enter_now", "wait_market", "adverse_limit"),
        "max_wait": 60,
    },
    {
        "name": "conservative_25bps_fill70",
        "kinds": ("enter_now", "wait_market", "adverse_limit"),
        "min_delta": 0.0025,
        "min_fill": 0.70,
        "max_expected_missed": 0.0025,
    },
    {
        "name": "conservative_50bps_fill80",
        "kinds": ("enter_now", "wait_market", "adverse_limit"),
        "min_delta": 0.0050,
        "min_fill": 0.80,
        "max_expected_missed": 0.0015,
    },
    {
        "name": "conservative_100bps_fill90",
        "kinds": ("enter_now", "wait_market", "adverse_limit"),
        "min_delta": 0.0100,
        "min_fill": 0.90,
        "max_expected_missed": 0.0010,
    },
)


def _choose(part: pd.DataFrame, *, arm: str, gate: dict[str, Any]) -> pd.Series:
    expected_col = f"oof_{arm}_expected_action_ev"
    fill_col = f"oof_{arm}_fill_probability"
    missed_col = f"oof_{arm}_expected_missed_ev"
    allowed = part["action_kind"].isin(gate["kinds"])
    if "max_wait" in gate:
        allowed &= part["wait_minutes"].le(int(gate["max_wait"]))
    candidates = part.loc[allowed].copy()
    now = candidates.loc[candidates["action_kind"].eq("enter_now")].iloc[0]
    selected = candidates.loc[candidates[expected_col].idxmax()]
    if selected["action_kind"] != "enter_now":
        predicted_delta = float(selected[expected_col]) - float(now[expected_col])
        if (
            predicted_delta < float(gate.get("min_delta", -np.inf))
            or float(selected[fill_col]) < float(gate.get("min_fill", -np.inf))
            or float(selected[missed_col])
            > float(gate.get("max_expected_missed", np.inf))
        ):
            selected = now
    return selected


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ValueError("output directory already exists")
    handoff = pd.read_parquet(args.handoff)
    actions = pd.read_parquet(args.actions)
    mapped = pd.read_parquet(
        args.mapped_oof, columns=[*IDENTITY, args.mapping_col]
    )
    if actions["base_position"].min() < 0 or actions["base_position"].max() >= len(handoff):
        raise ValueError("action base positions do not index the timing handoff")
    identity = handoff.loc[:, IDENTITY].reset_index(drop=True)
    identity["base_position"] = np.arange(len(identity), dtype=int)
    eligible = identity.merge(
        mapped,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    eligible[args.mapping_col] = pd.to_numeric(
        eligible[args.mapping_col], errors="coerce"
    )
    eligible = eligible.loc[eligible[args.mapping_col].notna()].copy()
    admitted_count = max(1, int(np.ceil(args.top_k_fraction * len(eligible))))
    admitted = eligible.nlargest(admitted_count, args.mapping_col)
    action_book = actions.loc[
        actions["base_position"].isin(admitted["base_position"])
    ].copy()

    decisions: list[dict[str, Any]] = []
    for arm in ("lgbm", "fixed_grid", "ridge_logistic"):
        for gate in GATES:
            for position, part in action_book.groupby("base_position", sort=False):
                selected = _choose(part, arm=arm, gate=gate)
                decisions.append(
                    {
                        "arm": arm,
                        "gate": gate["name"],
                        "base_position": int(position),
                        "action_id": selected["action_id"],
                        "action_kind": selected["action_kind"],
                        "filled": bool(selected["filled"]),
                        "missed_profitable_trade": bool(
                            (not bool(selected["filled"]))
                            and float(selected["missed_opportunity_ev"]) > 0.0
                        ),
                        "adverse_first": bool(selected["adverse_first"]),
                        "action_utility": float(selected["action_realized_utility"]),
                        "enter_now_ev": float(selected["enter_now_net_ev"]),
                    }
                )
    decision = pd.DataFrame(decisions).merge(
        admitted.loc[:, ["base_position", "__ts__", "side_name"]],
        on="base_position",
        how="left",
        validate="many_to_one",
    )
    decision["month"] = pd.to_datetime(decision["__ts__"], utc=True).dt.strftime(
        "%Y-%m"
    )
    rows: list[dict[str, Any]] = []
    for (arm, gate), base in decision.groupby(["arm", "gate"], sort=True):
        scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", base)]
        scopes.extend(
            ("month", str(value), part)
            for value, part in base.groupby("month", sort=True)
        )
        scopes.extend(
            ("side", str(value), part)
            for value, part in base.groupby("side_name", sort=True)
        )
        for scope, value, part in scopes:
            filled = part["filled"].astype(bool)
            rows.append(
                {
                    "arm": arm,
                    "gate": gate,
                    "scope": scope,
                    "scope_value": value,
                    "rows": int(len(part)),
                    "action_ev_bps": float(10_000.0 * part["action_utility"].mean()),
                    "enter_now_ev_bps": float(10_000.0 * part["enter_now_ev"].mean()),
                    "delta_vs_enter_now_bps": float(
                        10_000.0
                        * (part["action_utility"] - part["enter_now_ev"]).mean()
                    ),
                    "fill_rate": float(filled.mean()),
                    "missed_profitable_trade_rate": float(
                        part["missed_profitable_trade"].mean()
                    ),
                    "adverse_first_rate_if_filled": float(
                        part.loc[filled, "adverse_first"].mean()
                    ),
                    "enter_now_share": float(
                        part["action_kind"].eq("enter_now").mean()
                    ),
                    "wait_market_share": float(
                        part["action_kind"].eq("wait_market").mean()
                    ),
                    "adverse_limit_share": float(
                        part["action_kind"].eq("adverse_limit").mean()
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "gate_metrics.csv"
    decision_path = args.output_dir / "gate_decisions.parquet"
    metrics.to_csv(metrics_path, index=False)
    decision.to_parquet(decision_path, index=False, compression="zstd")
    overall = metrics.query("scope == 'overall'").sort_values(
        ["delta_vs_enter_now_bps", "action_ev_bps"], ascending=False
    )
    report = {
        "schema": "execution_entry_timing_action_gate_ablation_v1",
        "contract": {
            "admission": "one pooled global top-k after causal recent EV mapping",
            "per_timestamp_quota": False,
            "timing_reranks_ev": False,
            "gates": list(GATES),
        },
        "rows": {
            "mapped_eligible": int(len(eligible)),
            "globally_admitted": int(len(admitted)),
        },
        "overall": overall.to_dict("records"),
        "decision": (
            "retain_enter_now_only"
            if not overall.loc[overall["gate"].ne("enter_now_only"), "delta_vs_enter_now_bps"].gt(0).any()
            else "review_positive_fixed_gate_on_independent_slices"
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"report": report_path, "metrics": metrics_path, "decisions": decision_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
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
        raise SystemExit(f"timing action-gate ablation failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
