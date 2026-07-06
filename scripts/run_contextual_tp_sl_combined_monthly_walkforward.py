#!/usr/bin/env python3
"""Monthly walk-forward replay for contextual TP/SL component challengers.

Each month is replayed from a fresh portfolio state.  Diagnostic percentile
references are fitted only on rows before the monthly replay start.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_COMBO = "long_bars:S_long_dist:R_short_asset:R_short_bollinger:J"
VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "shortasset_uncertainty_only": {
        "gate_head": "short_asset",
        "risk_column": "diagnostic_uncertainty_risk",
        "threshold": 0.95,
        "weekly_gate": False,
    },
    "shortasset_drift_only": {
        "gate_head": "short_asset",
        "risk_column": "diagnostic_drift_risk",
        "threshold": 0.75,
        "weekly_gate": False,
    },
    "shortasset_ood_only": {
        "gate_head": "short_asset",
        "risk_column": "diagnostic_ood_risk",
        "threshold": 0.95,
        "weekly_gate": False,
    },
    "longbars_uncertainty_only": {
        "gate_head": "long_bars",
        "risk_column": "diagnostic_uncertainty_risk",
        "threshold": 0.75,
        "weekly_gate": False,
    },
    "longbars_drift_only": {
        "gate_head": "long_bars",
        "risk_column": "diagnostic_drift_risk",
        "threshold": 0.85,
        "weekly_gate": False,
    },
    "longbars_ood_only": {
        "gate_head": "long_bars",
        "risk_column": "diagnostic_ood_risk",
        "threshold": 0.85,
        "weekly_gate": False,
    },
    "longbars_weekgate_only": {
        "gate_head": "short_asset",
        "risk_column": "diagnostic_uncertainty_risk",
        "threshold": 2.0,
        "weekly_gate": True,
    },
    "combined": {
        "gate_head": "short_asset",
        "risk_column": "diagnostic_uncertainty_risk",
        "threshold": 0.95,
        "weekly_gate": True,
    },
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run(cmd: List[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _month_windows(start_month: str, end_month: str) -> List[tuple[str, str, str]]:
    months = pd.period_range(start=start_month, end=end_month, freq="M")
    windows: List[tuple[str, str, str]] = []
    for month in months:
        start = month.start_time.tz_localize("UTC")
        end = month.end_time.tz_localize("UTC")
        windows.append((str(month), start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")))
    return windows


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads((path / "combo_replay_manifest.json").read_text(encoding="utf-8"))


def _diagnostic_parent_manifest(path: Path) -> Dict[str, Any]:
    parent = path.parent.parent
    manifest_path = parent / "diagnostic_head_gate_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _accepted(path: Path) -> pd.DataFrame:
    decisions = pd.read_parquet(path / "combo_replay_decisions.parquet")
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    out = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if out.empty:
        return out
    text = out["strategy_id"].astype(str)
    out["head"] = text.str.extract(r"^(short_bollinger|long_bars|long_dist|short_asset)", expand=False)
    net = pd.to_numeric(out.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(out.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    size = pd.to_numeric(out.get("position_size", 0.0), errors="coerce").fillna(0.0)
    out["net_pnl_amount"] = size * net
    out["gross_pnl_amount"] = size * gross
    out["is_win"] = net > 0.0
    reason = out.get("position_exit_reason", pd.Series("", index=out.index)).astype(str)
    out["is_full_sl"] = reason.str.contains("full_sl|sl", case=False, na=False)
    out["is_timeout"] = reason.str.contains("timeout", case=False, na=False)
    return out


def _global_record(month: str, label: str, path: Path) -> Dict[str, Any]:
    manifest = _load_manifest(path)
    diag_manifest = _diagnostic_parent_manifest(path)
    metrics = manifest.get("metrics", {})
    return {
        "month": month,
        "label": label,
        "path": str(path),
        "candidate_rows": manifest.get("candidate_rows"),
        "candidate_start": manifest.get("candidate_start"),
        "candidate_end": manifest.get("candidate_end"),
        "risk_reference_end": manifest.get("risk_reference_end", diag_manifest.get("risk_reference_end")),
        "risk_reference_rows": manifest.get("risk_reference_rows", diag_manifest.get("risk_reference_rows")),
        "net_pnl": metrics.get("net_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "trade_count": metrics.get("trade_count"),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "strategy_concentration": metrics.get("strategy_concentration"),
        "side_concentration": metrics.get("side_concentration"),
    }


def _head_records(month: str, label: str, path: Path) -> List[Dict[str, Any]]:
    accepted = _accepted(path)
    if accepted.empty:
        return []
    grouped = (
        accepted.groupby("head", dropna=False)
        .agg(
            net_pnl=("net_pnl_amount", "sum"),
            gross_pnl=("gross_pnl_amount", "sum"),
            trades=("accepted", "size"),
            hit_rate=("is_win", "mean"),
            full_sl_rate=("is_full_sl", "mean"),
            timeout_rate=("is_timeout", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "label", label)
    grouped.insert(0, "month", month)
    return grouped.to_dict(orient="records")


def _add_deltas(frame: pd.DataFrame, *, baseline_label: str, keys: List[str]) -> pd.DataFrame:
    metrics = [
        c
        for c in frame.columns
        if c not in {"label", "path", "candidate_start", "candidate_end", "risk_reference_end", *keys}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    base = frame.loc[frame["label"].eq(baseline_label), [*keys, *metrics]].copy()
    base = base.rename(columns={c: f"{c}_baseline" for c in metrics})
    out = frame.merge(base, on=keys, how="left")
    for col in metrics:
        out[f"delta_{col}"] = out[col] - out[f"{col}_baseline"]
    return out


def _summary_by_label(global_df: pd.DataFrame) -> pd.DataFrame:
    challengers = global_df.loc[~global_df["label"].eq("wf_recent")].copy()
    rows: List[Dict[str, Any]] = []
    for label, frame in challengers.groupby("label", sort=False):
        rows.append(
            {
                "label": label,
                "months": int(frame["month"].nunique()),
                "sum_delta_net_pnl": float(frame["delta_net_pnl"].sum()),
                "median_delta_net_pnl": float(frame["delta_net_pnl"].median()),
                "positive_month_share": float(frame["delta_net_pnl"].gt(0).mean()),
                "mean_delta_full_sl_rate": float(frame["delta_full_sl_rate"].mean()),
                "mean_delta_max_drawdown": float(frame["delta_max_drawdown"].mean()),
                "sum_delta_trades": float(frame["delta_trade_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _risk_label(risk_column: str, threshold: float, *, weekly_gate: bool) -> str:
    base = str(risk_column).replace("diagnostic_", "").replace("_risk", "")
    label = f"{base}_gte{int(round(float(threshold) * 1000))}_drop"
    if weekly_gate:
        label = f"{label}__long_bars_net_lt_1000_lb2"
    return label


def _write_report(out_dir: Path, global_df: pd.DataFrame, head_df: pd.DataFrame, payload: Dict[str, Any]) -> None:
    summary_df = _summary_by_label(global_df)
    summary = summary_df.to_dict(orient="records")
    payload["summary"] = summary
    (out_dir / "monthly_walkforward_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Component Monthly Walk-Forward",
        "",
        "This is a development walk-forward replay, not untouched OOS.",
        "Each month starts from a fresh portfolio state. Challenger diagnostic percentile references use only prior rows.",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "_No challenger rows._",
        "",
        "## Global By Month",
        "",
        global_df.to_markdown(index=False),
        "",
        "## Per Head By Month",
        "",
        head_df.to_markdown(index=False) if not head_df.empty else "_No head rows._",
    ]
    (out_dir / "monthly_walkforward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-month", default="2026-01")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--combo-id", default=DEFAULT_COMBO)
    parser.add_argument("--weekly-gate-path", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--variants",
        default="shortasset_uncertainty_only,longbars_weekgate_only,combined",
        help=(
            "Comma-separated challenger variants. Supported: "
            + ",".join(sorted(VARIANT_CONFIGS))
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = _month_windows(str(args.start_month), str(args.end_month))
    global_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    variants = [part.strip() for part in str(args.variants).split(",") if part.strip()]
    supported = set(VARIANT_CONFIGS)
    unknown = sorted(set(variants) - supported)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Supported variants: {sorted(supported)}")

    for month, start, end in windows:
        month_dir = args.out_dir / "months" / month
        baseline_dir = month_dir / "wf_recent"
        if not args.summarize_only:
            _run(
                [
                    sys.executable,
                    "-u",
                    "scripts/materialize_contextual_tp_sl_combo_replay.py",
                    "--source-dir",
                    str(args.source_dir),
                    "--combo-id",
                    str(args.combo_id),
                    "--out-dir",
                    str(baseline_dir),
                    "--market-mode",
                    str(args.market_mode),
                    "--start",
                    start,
                    "--end",
                    end,
                ],
                root,
            )
            for variant in variants:
                config = VARIANT_CONFIGS[variant]
                variant_dir = month_dir / f"{variant}_runs"
                cmd = [
                    sys.executable,
                    "-u",
                    "scripts/ablate_contextual_tp_sl_diagnostic_head_gate.py",
                    "--source-dir",
                    str(args.source_dir),
                    "--combo-id",
                    str(args.combo_id),
                    "--out-dir",
                    str(variant_dir),
                    "--gate-head",
                    str(config["gate_head"]),
                    "--risk-columns",
                    str(config["risk_column"]),
                    "--actions",
                    "drop",
                    "--market-mode",
                    str(args.market_mode),
                    "--start",
                    start,
                    "--end",
                    end,
                    "--risk-reference-end",
                    start,
                    "--thresholds",
                    str(config["threshold"]),
                ]
                if config["weekly_gate"]:
                    cmd.extend(
                        [
                            "--weekly-gate-path",
                            str(args.weekly_gate_path),
                            "--weekly-gate-head",
                            "long_bars",
                        ]
                    )
                _run(cmd, root)
        if not (baseline_dir / "combo_replay_manifest.json").exists():
            raise FileNotFoundError(f"Missing monthly baseline replay: {baseline_dir}")
        global_rows.append(_global_record(month, "wf_recent", baseline_dir))
        head_rows.extend(_head_records(month, "wf_recent", baseline_dir))
        for variant in variants:
            config = VARIANT_CONFIGS[variant]
            run_dir = (
                month_dir
                / f"{variant}_runs"
                / "materialized"
                / _risk_label(
                    str(config["risk_column"]),
                    float(config["threshold"]),
                    weekly_gate=bool(config["weekly_gate"]),
                )
            )
            if not (run_dir / "combo_replay_manifest.json").exists():
                raise FileNotFoundError(f"Missing monthly {variant} replay: {run_dir}")
            global_rows.append(_global_record(month, variant, run_dir))
            head_rows.extend(_head_records(month, variant, run_dir))

    global_df = _add_deltas(pd.DataFrame(global_rows), baseline_label="wf_recent", keys=["month"])
    head_df = _add_deltas(pd.DataFrame(head_rows), baseline_label="wf_recent", keys=["month", "head"])
    global_df.to_csv(args.out_dir / "monthly_walkforward_global.csv", index=False)
    head_df.to_csv(args.out_dir / "monthly_walkforward_head.csv", index=False)
    payload: Dict[str, Any] = {
        "generated_by": "run_contextual_tp_sl_combined_monthly_walkforward",
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "combo_id": str(args.combo_id),
        "weekly_gate_path": str(args.weekly_gate_path),
        "market_mode": str(args.market_mode),
        "variants": variants,
        "windows": [{"month": m, "start": s, "end": e} for m, s, e in windows],
    }
    _write_report(args.out_dir, global_df, head_df, payload)
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "months": len(windows)}), indent=2))


if __name__ == "__main__":
    main()
