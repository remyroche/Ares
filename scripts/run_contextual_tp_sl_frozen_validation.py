#!/usr/bin/env python3
"""Run a fixed contextual TP/SL validation package.

This is a thin orchestration layer over:

1. `materialize_contextual_tp_sl_combo_replay.py`
2. `compare_materialized_contextual_tp_sl_replays.py`
3. `audit_contextual_tp_sl_promotion_gate.py`

It exists so forward validation of frozen candidate combos can be repeated with
one command on each newly accumulated candidate window.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_COMBOS = {
    "static": "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S",
    "wf_recent": "long_bars:S_long_dist:R_short_asset:R_short_bollinger:J",
}
OPTIONAL_COMBOS = {
    "best_net": "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R",
    "best_balanced": "long_bars:J_long_dist:R_short_asset:I_short_bollinger:R",
    "performance_probe": "long_bars:P_long_dist:P_short_asset:P_short_bollinger:P",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run(cmd: List[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _combo_specs(include_challengers: bool, extra_combo: List[str]) -> Dict[str, str]:
    combos = dict(DEFAULT_COMBOS)
    if include_challengers:
        combos.update(OPTIONAL_COMBOS)
    for raw in extra_combo:
        if "=" not in raw:
            raise ValueError(f"Invalid --combo {raw!r}; expected label=combo_id")
        label, combo_id = raw.split("=", 1)
        if not label.strip() or not combo_id.strip():
            raise ValueError(f"Invalid --combo {raw!r}; label and combo_id are required")
        combos[label.strip()] = combo_id.strip()
    return combos


def _read_global(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "comparison" / "materialized_replay_global_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    (out_dir / "frozen_validation_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    global_df = _read_global(out_dir)
    lines = [
        "# Contextual TP/SL Frozen Validation",
        "",
        f"Source directory: `{payload['source_dir']}`",
        f"Validation role: `{payload['validation_role']}`",
        f"Start filter: `{payload.get('start') or ''}`",
        f"End filter: `{payload.get('end') or ''}`",
        "",
        "## Combos",
        "",
        pd.DataFrame(
            [{"label": label, "combo_id": combo} for label, combo in payload["combos"].items()]
        ).to_markdown(index=False),
        "",
        "## Global Comparison",
        "",
        global_df.to_markdown(index=False) if not global_df.empty else "_No comparison rows._",
        "",
        "## Reports",
        "",
        "- `comparison/materialized_replay_comparison_report.md`",
        "- `promotion_gate/contextual_tp_sl_promotion_gate_report.md`",
    ]
    (out_dir / "frozen_validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--validation-role", default="forward")
    parser.add_argument("--baseline-label", default="static")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--include-challengers", action="store_true")
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="Additional label=combo_id. Repeatable.",
    )
    parser.add_argument("--min-candidate-rows", type=int, default=1000)
    parser.add_argument("--min-trade-count", type=int, default=500)
    parser.add_argument("--min-weeks", type=int, default=4)
    parser.add_argument("--min-active-heads", type=int, default=3)
    parser.add_argument("--min-positive-week-share", type=float, default=0.60)
    parser.add_argument("--min-positive-week-delta-share", type=float, default=0.60)
    parser.add_argument("--min-delta-net-pnl", type=float, default=0.0)
    parser.add_argument("--max-delta-full-sl-rate", type=float, default=0.0)
    parser.add_argument("--min-delta-max-drawdown", type=float, default=0.0)
    parser.add_argument("--min-delta-week-q10-pnl", type=float, default=-1000.0)
    parser.add_argument("--min-delta-week-q20-pnl", type=float, default=0.0)
    parser.add_argument("--min-delta-worst-week-pnl", type=float, default=-1.0e18)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combos = _combo_specs(bool(args.include_challengers), list(args.combo))

    materialized_runs: List[Tuple[str, Path]] = []
    for label, combo_id in combos.items():
        run_dir = args.out_dir / "materialized" / label
        cmd = [
            sys.executable,
            "-u",
            "scripts/materialize_contextual_tp_sl_combo_replay.py",
            "--source-dir",
            str(args.source_dir),
            "--combo-id",
            combo_id,
            "--out-dir",
            str(run_dir),
            "--market-mode",
            str(args.market_mode),
        ]
        if args.start:
            cmd.extend(["--start", str(args.start)])
        if args.end:
            cmd.extend(["--end", str(args.end)])
        _run(cmd, root)
        materialized_runs.append((label, run_dir))

    comparison_dir = args.out_dir / "comparison"
    compare_cmd = [
        sys.executable,
        "-u",
        "scripts/compare_materialized_contextual_tp_sl_replays.py",
        "--out-dir",
        str(comparison_dir),
        "--baseline-label",
        str(args.baseline_label),
    ]
    for label, run_dir in materialized_runs:
        compare_cmd.extend(["--run", f"{label}={run_dir}"])
    _run(compare_cmd, root)

    gate_dir = args.out_dir / "promotion_gate"
    gate_cmd = [
        sys.executable,
        "-u",
        "scripts/audit_contextual_tp_sl_promotion_gate.py",
        "--comparison-dir",
        str(comparison_dir),
        "--out-dir",
        str(gate_dir),
        "--validation-role",
        str(args.validation_role),
        "--baseline-label",
        str(args.baseline_label),
        "--min-candidate-rows",
        str(args.min_candidate_rows),
        "--min-trade-count",
        str(args.min_trade_count),
        "--min-weeks",
        str(args.min_weeks),
        "--min-active-heads",
        str(args.min_active_heads),
        "--min-positive-week-share",
        str(args.min_positive_week_share),
        "--min-positive-week-delta-share",
        str(args.min_positive_week_delta_share),
        "--min-delta-net-pnl",
        str(args.min_delta_net_pnl),
        "--max-delta-full-sl-rate",
        str(args.max_delta_full_sl_rate),
        "--min-delta-max-drawdown",
        str(args.min_delta_max_drawdown),
        "--min-delta-week-q10-pnl",
        str(args.min_delta_week_q10_pnl),
        "--min-delta-week-q20-pnl",
        str(args.min_delta_week_q20_pnl),
        f"--min-delta-worst-week-pnl={args.min_delta_worst_week_pnl}",
    ]
    _run(gate_cmd, root)

    payload: Dict[str, Any] = {
        "generated_by": "run_contextual_tp_sl_frozen_validation",
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "market_mode": str(args.market_mode),
        "validation_role": str(args.validation_role),
        "baseline_label": str(args.baseline_label),
        "start": str(args.start),
        "end": str(args.end),
        "combos": combos,
        "comparison_dir": str(comparison_dir),
        "promotion_gate_dir": str(gate_dir),
    }
    _write_summary(args.out_dir, payload)
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "runs": len(materialized_runs),
                    "comparison_dir": str(comparison_dir),
                    "promotion_gate_dir": str(gate_dir),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
