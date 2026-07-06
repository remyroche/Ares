#!/usr/bin/env python3
"""Run a bounded multiplier grid for the selected contextual TP/SL overlay rules."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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


def _parse_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _label(lb_mult: float, ld_mult: float, sa_mult: float) -> str:
    return (
        f"grid_lb{int(round(lb_mult * 1000)):03d}"
        f"_ld{int(round(ld_mult * 1000)):03d}"
        f"_sa{int(round(sa_mult * 1000)):03d}"
    )


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data_perp/reports/contextual_tp_sl_ablation_q35w07_q20w03_6mo_diagperf_20260630")
    parser.add_argument("--combo-id", default="long_bars:S_long_dist:R_short_asset:R_short_bollinger:J")
    parser.add_argument("--out-dir", default="data_perp/reports/contextual_tp_sl_overlay_multiplier_grid_6mo_20260701")
    parser.add_argument("--multipliers", default="0.25,0.50,0.75")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mults = _parse_floats(args.multipliers)
    rows: List[Dict[str, Any]] = []
    for lb_mult, ld_mult, sa_mult in product(mults, mults, mults):
        label = _label(lb_mult, ld_mult, sa_mult)
        run_dir = out_dir / "multi_overlay" / "materialized" / label
        manifest_path = run_dir / "combo_replay_manifest.json"
        if args.skip_existing and manifest_path.exists():
            status = "skipped_existing"
        else:
            cmd = [
                sys.executable,
                "scripts/run_contextual_tp_sl_multi_head_diagnostic_overlay.py",
                "--source-dir",
                str(args.source_dir),
                "--combo-id",
                str(args.combo_id),
                "--out-dir",
                str(out_dir),
                "--rule",
                f"long_bars:composite:0.55:{lb_mult:g}",
                "--rule",
                f"long_dist:composite:0.70:{ld_mult:g}",
                "--rule",
                f"short_asset:recent_hr_surprise:0.85:{sa_mult:g}",
                "--groups",
                "uncertainty,drift,ood,recent_hr_surprise",
                "--market-mode",
                str(args.market_mode),
                "--label",
                label,
            ]
            print(f"RUN {label}", flush=True)
            result = _run(cmd)
            (run_dir / "grid_command_stdout.txt").write_text(result.stdout, encoding="utf-8")
            (run_dir / "grid_command_stderr.txt").write_text(result.stderr, encoding="utf-8")
            status = "completed"
        metrics: Dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics = manifest.get("metrics", {})
        rows.append(
            {
                "label": label,
                "status": status,
                "long_bars_multiplier": lb_mult,
                "long_dist_multiplier": ld_mult,
                "short_asset_multiplier": sa_mult,
                "out_dir": str(run_dir),
                "net_pnl": metrics.get("net_pnl"),
                "gross_pnl": metrics.get("gross_pnl"),
                "trade_count": metrics.get("trade_count"),
                "full_sl_rate": metrics.get("full_sl_rate"),
                "timeout_rate": metrics.get("timeout_rate"),
                "max_drawdown": metrics.get("max_drawdown"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "overlay_multiplier_grid_summary.csv", index=False)
    manifest = {
        "generated_by": "run_contextual_tp_sl_overlay_multiplier_grid",
        "source_dir": str(args.source_dir),
        "combo_id": str(args.combo_id),
        "out_dir": str(out_dir),
        "multipliers": mults,
        "run_count": int(len(summary)),
        "completed_count": int(summary["status"].eq("completed").sum()),
        "skipped_existing_count": int(summary["status"].eq("skipped_existing").sum()),
        "outputs": ["overlay_multiplier_grid_summary.csv"],
    }
    (out_dir / "overlay_multiplier_grid_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
