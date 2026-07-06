#!/usr/bin/env python3
"""Run frozen contextual TP/SL forward validation if a ready source exists.

This script scans local candidate sources, selects the best post-cutoff source
that satisfies coverage requirements, and launches the frozen validation runner.
If no source is ready, it writes a readiness report and exits successfully
without running an underpowered replay.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if not isinstance(value, (dict, list, tuple)):
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
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


def _select_source(scan_csv: Path) -> Optional[pd.Series]:
    if not scan_csv.exists():
        return None
    frame = pd.read_csv(scan_csv)
    if frame.empty or "usable_post_cutoff" not in frame.columns:
        return None
    usable = frame.loc[frame["usable_post_cutoff"].astype(bool)].copy()
    if usable.empty:
        return None
    usable["post_cutoff_rows_num"] = pd.to_numeric(usable["post_cutoff_rows"], errors="coerce").fillna(0)
    usable["post_cutoff_timestamps_num"] = pd.to_numeric(
        usable["post_cutoff_timestamps"], errors="coerce"
    ).fillna(0)
    if "has_required_diagnostic_groups" in usable.columns:
        usable["diagnostic_ready"] = usable["has_required_diagnostic_groups"].astype(str).str.lower().eq("true")
    else:
        usable["diagnostic_ready"] = True
    usable = usable.sort_values(
        ["diagnostic_ready", "post_cutoff_rows_num", "post_cutoff_timestamps_num", "candidate_end"],
        ascending=[False, False, False, False],
    )
    return usable.iloc[0]


def _nearest_source(scan_csv: Path) -> Optional[pd.Series]:
    if not scan_csv.exists():
        return None
    frame = pd.read_csv(scan_csv)
    if frame.empty:
        return None
    frame["post_cutoff_rows_num"] = pd.to_numeric(frame.get("post_cutoff_rows", 0), errors="coerce").fillna(0)
    frame["post_cutoff_timestamps_num"] = pd.to_numeric(
        frame.get("post_cutoff_timestamps", 0), errors="coerce"
    ).fillna(0)
    frame["post_cutoff_active_heads_num"] = pd.to_numeric(
        frame.get("post_cutoff_active_heads", 0), errors="coerce"
    ).fillna(0)
    if "has_required_diagnostic_groups" in frame.columns:
        frame["diagnostic_ready"] = frame["has_required_diagnostic_groups"].astype(str).str.lower().eq("true")
    else:
        frame["diagnostic_ready"] = True
    frame = frame.sort_values(
        [
            "diagnostic_ready",
            "post_cutoff_rows_num",
            "post_cutoff_timestamps_num",
            "post_cutoff_active_heads_num",
            "candidate_end",
        ],
        ascending=[False, False, False, False, False],
    )
    return frame.iloc[0]


def _source_slug(source_dir: str) -> str:
    text = str(source_dir).strip().strip("/").replace("/", "__").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)[-180:]


def _backfill_candidate_source(scan_csv: Path) -> Optional[pd.Series]:
    if not scan_csv.exists():
        return None
    frame = pd.read_csv(scan_csv)
    if frame.empty:
        return None
    if "missing_required_diagnostic_groups" not in frame.columns:
        return None
    missing = frame["missing_required_diagnostic_groups"].astype(str)
    candidates = frame.loc[missing.str.contains("performance", case=False, na=False)].copy()
    if candidates.empty:
        return None
    for col in ("has_required_arms", "has_required_columns"):
        if col in candidates.columns:
            candidates = candidates.loc[candidates[col].astype(str).str.lower().eq("true")].copy()
    if candidates.empty:
        return None
    candidates["post_cutoff_rows_num"] = pd.to_numeric(candidates.get("post_cutoff_rows", 0), errors="coerce").fillna(0)
    candidates["post_cutoff_timestamps_num"] = pd.to_numeric(
        candidates.get("post_cutoff_timestamps", 0), errors="coerce"
    ).fillna(0)
    candidates["post_cutoff_active_heads_num"] = pd.to_numeric(
        candidates.get("post_cutoff_active_heads", 0), errors="coerce"
    ).fillna(0)
    candidates = candidates.sort_values(
        ["post_cutoff_rows_num", "post_cutoff_timestamps_num", "post_cutoff_active_heads_num", "candidate_end"],
        ascending=[False, False, False, False],
    )
    return candidates.iloc[0]


def _write_readiness(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    scan_dir: Path,
    selected: Optional[pd.Series],
    nearest: Optional[pd.Series],
    validation_dir: Optional[Path],
) -> None:
    payload: Dict[str, Any] = {
        "generated_by": "run_latest_contextual_tp_sl_forward_validation_if_ready",
        "cutoff": str(args.cutoff),
        "out_dir": str(out_dir),
        "scan_dir": str(scan_dir),
        "validation_dir": str(validation_dir) if validation_dir is not None else None,
        "ready": selected is not None,
        "selected_source": selected.to_dict() if selected is not None else None,
        "nearest_source": nearest.to_dict() if nearest is not None else None,
        "coverage_requirements": {
            "min_post_cutoff_rows": int(args.min_post_cutoff_rows),
            "min_post_cutoff_timestamps": int(args.min_post_cutoff_timestamps),
            "min_post_cutoff_active_heads": int(args.min_post_cutoff_active_heads),
            "required_diagnostic_groups": list(args.required_diagnostic_group or []),
            "min_diagnostic_group_features": int(args.min_diagnostic_group_features),
            "min_trade_count": int(args.min_trade_count),
            "min_weeks": int(args.min_weeks),
            "min_active_heads": int(args.min_active_heads),
        },
    }
    (out_dir / "latest_forward_validation_readiness.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Latest Contextual TP/SL Forward Validation Readiness",
        "",
        f"Cutoff: `{args.cutoff}`",
        f"Ready: `{selected is not None}`",
        f"Scan directory: `{scan_dir}`",
        f"Validation directory: `{validation_dir or ''}`",
        "",
    ]
    if selected is None:
        lines.extend(
            [
                "No local candidate source currently satisfies the post-cutoff coverage requirements.",
                "",
                f"Minimum post-cutoff rows: `{args.min_post_cutoff_rows}`",
                f"Minimum post-cutoff timestamps: `{args.min_post_cutoff_timestamps}`",
                f"Minimum post-cutoff active heads: `{args.min_post_cutoff_active_heads}`",
                f"Required diagnostic groups: `{', '.join(args.required_diagnostic_group or []) or 'none'}`",
                f"Minimum diagnostic columns per group/arm: `{args.min_diagnostic_group_features}`",
                f"Minimum trades for gate: `{args.min_trade_count}`",
                f"Minimum weeks for gate: `{args.min_weeks}`",
                f"Minimum active heads for gate: `{args.min_active_heads}`",
            ]
        )
        if nearest is not None:
            lines.extend(
                [
                    "",
                    "Closest source:",
                    "",
                    pd.DataFrame([nearest.to_dict()]).to_markdown(index=False),
                ]
            )
    else:
        lines.extend(
            [
                "Selected source:",
                "",
                pd.DataFrame([selected.to_dict()]).to_markdown(index=False),
            ]
        )
    (out_dir / "latest_forward_validation_readiness.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root", action="append", default=None)
    parser.add_argument("--cutoff", default="2026-06-26T14:00:00Z")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--min-post-cutoff-rows", type=int, default=1000)
    parser.add_argument("--min-post-cutoff-timestamps", type=int, default=20)
    parser.add_argument("--min-post-cutoff-active-heads", type=int, default=3)
    parser.add_argument(
        "--required-diagnostic-group",
        action="append",
        default=["uncertainty", "drift", "ood", "performance"],
        choices=["uncertainty", "drift", "ood", "performance"],
        help=(
            "Diagnostic group required in every required candidate arm. "
            "Defaults to all four groups used by the contextual TP/SL ablation."
        ),
    )
    parser.add_argument("--min-diagnostic-group-features", type=int, default=1)
    parser.add_argument("--min-trade-count", type=int, default=500)
    parser.add_argument("--min-weeks", type=int, default=4)
    parser.add_argument("--min-active-heads", type=int, default=3)
    parser.add_argument("--include-challengers", action="store_true")
    parser.add_argument(
        "--auto-materialize-performance-features",
        action="store_true",
        help=(
            "If no source is ready only because recent-performance diagnostics are missing, "
            "copy the best source into this output directory, backfill those diagnostics, and rescan."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Run validation on best source even if not ready.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = args.out_dir / "source_scan"
    root_args = args.root or ["data_perp/reports", "data_perp/artifacts"]
    scan_cmd = [
        sys.executable,
        "-u",
        "scripts/scan_contextual_tp_sl_candidate_sources.py",
        "--out-dir",
        str(scan_dir),
        "--cutoff",
        str(args.cutoff),
        "--min-post-cutoff-rows",
        str(args.min_post_cutoff_rows),
        "--min-post-cutoff-timestamps",
        str(args.min_post_cutoff_timestamps),
        "--min-post-cutoff-active-heads",
        str(args.min_post_cutoff_active_heads),
        "--min-diagnostic-group-features",
        str(args.min_diagnostic_group_features),
    ]
    for group in args.required_diagnostic_group or []:
        scan_cmd.extend(["--required-diagnostic-group", str(group)])
    for item in root_args:
        scan_cmd.extend(["--root", str(item)])
    _run(scan_cmd, root)

    scan_csv = scan_dir / "contextual_tp_sl_candidate_source_scan.csv"
    selected = _select_source(scan_csv)
    nearest = _nearest_source(scan_csv)
    materialized_source: Optional[Path] = None
    if selected is None and bool(args.auto_materialize_performance_features):
        backfill = _backfill_candidate_source(scan_csv)
        if backfill is not None:
            materialized_source = args.out_dir / "performance_feature_sources" / _source_slug(str(backfill["source_dir"]))
            materialize_cmd = [
                sys.executable,
                "-u",
                "scripts/materialize_contextual_tp_sl_performance_features.py",
                "--source-dir",
                str(backfill["source_dir"]),
                "--out-dir",
                str(materialized_source),
                "--copy-sidecars",
            ]
            _run(materialize_cmd, root)
            scan_dir = args.out_dir / "source_scan_after_performance_backfill"
            rescan_cmd = list(scan_cmd)
            out_pos = rescan_cmd.index("--out-dir") + 1
            rescan_cmd[out_pos] = str(scan_dir)
            rescan_cmd.extend(["--root", str(materialized_source)])
            _run(rescan_cmd, root)
            scan_csv = scan_dir / "contextual_tp_sl_candidate_source_scan.csv"
            selected = _select_source(scan_csv)
            nearest = _nearest_source(scan_csv)
    if selected is None and args.force and scan_csv.exists():
        frame = pd.read_csv(scan_csv)
        if not frame.empty:
            frame["post_cutoff_rows_num"] = pd.to_numeric(
                frame.get("post_cutoff_rows", 0), errors="coerce"
            ).fillna(0)
            frame = frame.sort_values(["post_cutoff_rows_num", "candidate_end"], ascending=[False, False])
            selected = frame.iloc[0]

    validation_dir: Optional[Path] = None
    if selected is not None:
        validation_dir = args.out_dir / "frozen_validation"
        validation_cmd = [
            sys.executable,
            "-u",
            "scripts/run_contextual_tp_sl_frozen_validation.py",
            "--source-dir",
            str(selected["source_dir"]),
            "--out-dir",
            str(validation_dir),
            "--validation-role",
            "forward",
            "--baseline-label",
            "static",
            "--market-mode",
            str(args.market_mode),
            "--start",
            str(args.cutoff),
            "--include-challengers",
            "--min-candidate-rows",
            str(args.min_post_cutoff_rows),
            "--min-trade-count",
            str(args.min_trade_count),
            "--min-weeks",
            str(args.min_weeks),
            "--min-active-heads",
            str(args.min_active_heads),
        ]
        if not bool(args.include_challengers):
            validation_cmd.remove("--include-challengers")
        _run(validation_cmd, root)

    _write_readiness(
        args.out_dir,
        args=args,
        scan_dir=scan_dir,
        selected=selected,
        nearest=nearest,
        validation_dir=validation_dir,
    )
    if materialized_source is not None:
        readiness_path = args.out_dir / "latest_forward_validation_readiness.json"
        if readiness_path.exists():
            payload = json.loads(readiness_path.read_text(encoding="utf-8"))
            payload["materialized_performance_source"] = str(materialized_source)
            readiness_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "ready": selected is not None and not bool(args.force),
                    "ran_validation": validation_dir is not None,
                    "selected_source": str(selected["source_dir"]) if selected is not None else None,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
