#!/usr/bin/env python3
"""Gate 3 readiness audit for materialized S52 meta handoff variants.

The materialized handoff contains two intentionally separate files:

* a clean decision-time candidate file, which must not contain realized outcomes;
* an offline evaluation file, which carries realized path labels for audit only.

This script compares multiple materialized variants and reports where they pass
or fail the current Gate 3 bars, with side/source/regime breakdowns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HANDOFF_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_v1"
)
DEFAULT_SCORED_LEDGER = DEFAULT_HANDOFF_ROOT / "s52_trailing_regime_scored_ledger.parquet"
DEFAULT_VARIANTS = (
    "s52_meta_threshold_top10_handoff_v1",
    "s52_meta_threshold_top10_handoff_sidecap90_v1",
    "s52_meta_threshold_top10_handoff_sidecap85_v1",
    "s52_meta_threshold_top10_handoff_sidecap80_v1",
    "s52_meta_threshold_top10_handoff_sidecap75_v1",
    "s52_meta_threshold_top10_handoff_sidecap70_v1",
    "s52_meta_threshold_top10_handoff_sidecap65_v1",
    "s52_meta_threshold_top10_handoff_sidecap60_v1",
)
OUTCOME_COLUMNS = {
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "clean_exec",
    "dirty_positive",
    "u_policy_net",
    "ret_net",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_mae_norm",
    "first_touch_full_path_mae_norm",
    "inferred_decision_stop_touch",
    "inferred_full_path_stop_touch",
    "inferred_raw_path_stop_touch",
    "underwater_bars_before_mfe_1r",
}
GROUPING_SPECS = {
    "side": ("side_name",),
    "source": ("source_semantic_family",),
    "side_source": ("side_name", "source_semantic_family"),
    "side_aegmm": ("side_name", "aegmm_cluster"),
    "side_side_aegmm": ("side_name", "side_aegmm_cluster"),
    "side_clean_score_bin": ("side_name", "regime_clean_exec_score_bin"),
    "side_dirty_score_bin": ("side_name", "regime_dirty_positive_score_bin"),
    "side_bad_mae_score_bin": ("side_name", "regime_first_touch_bad_mae_score_bin"),
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


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _max_month_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or "month" not in frame.columns or column not in frame.columns:
        return float("nan")
    vals = frame.groupby("month", dropna=False)[column].apply(_rate)
    return float(vals.max()) if len(vals) else float("nan")


def _worst_month_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or "month" not in frame.columns or column not in frame.columns:
        return float("nan")
    vals = frame.groupby("month", dropna=False)[column].apply(_mean)
    return float(vals.min()) if len(vals) else float("nan")


def _sum(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.sum()) if len(arr) else float("nan")


def _positive_margin_rate(frame: pd.DataFrame) -> float:
    return _rate(_num(frame.get("exec_margin"), index=frame.index).gt(0.0))


def _dominant_side_share(frame: pd.DataFrame) -> float:
    if frame.empty or "side_name" not in frame.columns:
        return float("nan")
    vc = frame["side_name"].astype(str).str.lower().value_counts(normalize=True)
    return float(vc.iloc[0]) if len(vc) else float("nan")


def _variant_label(path: Path) -> str:
    name = path.name
    return name.replace("s52_meta_threshold_top10_handoff_", "").replace("s52_meta_threshold_top10_handoff", "uncapped")


def _read_variant(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clean_path = path / "s52_meta_threshold_guarded_candidates.parquet"
    offline_path = path / "s52_meta_threshold_guarded_offline_eval_candidates.parquet"
    audit_path = path / "s52_meta_threshold_guarded_leakage_audit.json"
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)
    if not offline_path.exists():
        raise FileNotFoundError(offline_path)
    clean = pd.read_parquet(clean_path)
    offline = pd.read_parquet(offline_path)
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return clean, offline, audit


def _enrich_stop_touch_metrics(
    offline: pd.DataFrame,
    *,
    scored_ledger: pd.DataFrame | None,
    stop_mult: float,
) -> pd.DataFrame:
    if scored_ledger is None or scored_ledger.empty:
        return offline
    keys = ["__ts__", "__symbol__", "side_name"]
    if not all(col in offline.columns for col in keys) or not all(col in scored_ledger.columns for col in keys):
        return offline
    metric_cols = [
        "first_touch_mae_norm",
        "first_touch_full_path_mae_norm",
        "full_path_mae_norm",
        "mae_norm",
    ]
    cols = keys + [col for col in metric_cols if col in scored_ledger.columns and col not in offline.columns]
    if len(cols) <= len(keys):
        return offline
    out = offline.merge(scored_ledger[cols], on=keys, how="left", validate="one_to_one")
    decision_mae = _num(out.get("first_touch_mae_norm"), index=out.index)
    full_path_mae = _num(out.get("first_touch_full_path_mae_norm"), index=out.index)
    if full_path_mae.isna().all():
        full_path_mae = _num(out.get("full_path_mae_norm"), index=out.index)
    raw_path_mae = _num(out.get("mae_norm"), index=out.index)
    out["inferred_decision_stop_touch"] = decision_mae.ge(float(stop_mult)).astype(float)
    out["inferred_full_path_stop_touch"] = full_path_mae.ge(float(stop_mult)).astype(float)
    out["inferred_raw_path_stop_touch"] = raw_path_mae.ge(float(stop_mult)).astype(float)
    return out


def _gate_summary(
    variant: str,
    variant_dir: Path,
    clean: pd.DataFrame,
    offline: pd.DataFrame,
    audit: dict[str, Any],
    *,
    max_side_share: float,
    min_rows: int,
    min_symbols: int,
    allow_bad_mae_pnl_override: bool,
    min_override_mean_ret_net: float,
    min_override_worst_month_ret_net: float,
) -> dict[str, Any]:
    clean_forbidden = sorted(col for col in clean.columns if col in OUTCOME_COLUMNS)
    audit_forbidden = sorted(audit.get("clean_handoff_forbidden_columns", []) or [])
    rows = {
        "variant": variant,
        "variant_dir": str(variant_dir),
        "rows": int(len(offline)),
        "clean_rows": int(len(clean)),
        "symbols": int(offline.get("__symbol__", offline.get("symbol", pd.Series(dtype=str))).nunique()),
        "months": int(offline.get("month", pd.Series(dtype=str)).nunique()),
        "mean_exec_margin": _mean(offline.get("exec_margin")),
        "worst_month_exec_margin": _worst_month_mean(offline, "exec_margin"),
        "mean_ret_net": _mean(offline.get("ret_net")),
        "sum_ret_net": _sum(offline.get("ret_net")),
        "worst_month_ret_net": _worst_month_mean(offline, "ret_net"),
        "mean_u_policy_net": _mean(offline.get("u_policy_net")),
        "full_path_bad_mae": _rate(offline.get("full_path_bad_mae_1r")),
        "max_month_full_path_bad_mae": _max_month_rate(offline, "full_path_bad_mae_1r"),
        "timeout": _rate(offline.get("timeout")),
        "max_month_timeout": _max_month_rate(offline, "timeout"),
        "inferred_decision_stop_touch": _rate(offline.get("inferred_decision_stop_touch")),
        "inferred_full_path_stop_touch": _rate(offline.get("inferred_full_path_stop_touch")),
        "inferred_raw_path_stop_touch": _rate(offline.get("inferred_raw_path_stop_touch")),
        "clean_exec_precision": _rate(offline.get("clean_exec")),
        "dirty_positive_rate": _rate(offline.get("dirty_positive")),
        "positive_margin_rate": _positive_margin_rate(offline),
        "mfe_before_mae_rate": _rate(offline.get("mfe_before_mae_1r")),
        "mae_before_mfe_rate": _rate(offline.get("mae_before_mfe_1r")),
        "dominant_side_share": _dominant_side_share(offline),
        "short_share": float(offline["side_name"].astype(str).str.lower().eq("short").mean())
        if "side_name" in offline.columns and len(offline)
        else float("nan"),
        "clean_handoff_forbidden_columns": ",".join(clean_forbidden),
        "audit_forbidden_columns": ",".join(audit_forbidden),
        "duplicate_decision_key_rows": int(audit.get("duplicate_decision_key_rows", -1)),
    }
    checks = {
        "clean_handoff_no_outcomes": not clean_forbidden and not audit_forbidden,
        "duplicate_free": int(rows["duplicate_decision_key_rows"]) == 0,
        "enough_rows": int(rows["rows"]) >= int(min_rows),
        "enough_symbols": int(rows["symbols"]) >= int(min_symbols),
        "mean_exec_margin_positive": float(rows["mean_exec_margin"]) > 0.0,
        "worst_month_exec_margin_positive": float(rows["worst_month_exec_margin"]) > 0.0,
        "row_bad_mae_bar": float(rows["full_path_bad_mae"]) <= 0.50,
        "month_bad_mae_bar": float(rows["max_month_full_path_bad_mae"]) <= 0.50,
        "timeout_bar": float(rows["timeout"]) <= 0.12,
        "side_share_bar": float(rows["dominant_side_share"]) <= float(max_side_share),
    }
    rows.update({f"pass_{key}": bool(value) for key, value in checks.items()})
    rows["failed_checks"] = ",".join(key for key, value in checks.items() if not value)
    bad_mae_failures = [
        key
        for key, value in checks.items()
        if not value and key in {"row_bad_mae_bar", "month_bad_mae_bar"}
    ]
    non_bad_failures = [
        key
        for key, value in checks.items()
        if not value and key not in {"row_bad_mae_bar", "month_bad_mae_bar"}
    ]
    override_mean_ret_net_pass = float(rows["mean_ret_net"]) >= float(min_override_mean_ret_net)
    override_worst_month_ret_net_pass = float(rows["worst_month_ret_net"]) >= float(min_override_worst_month_ret_net)
    pnl_override = (
        bool(allow_bad_mae_pnl_override)
        and not non_bad_failures
        and bool(bad_mae_failures)
        and override_mean_ret_net_pass
        and override_worst_month_ret_net_pass
    )
    rows["bad_mae_pnl_override_enabled"] = bool(allow_bad_mae_pnl_override)
    rows["bad_mae_only_failures"] = ",".join(bad_mae_failures) if bad_mae_failures and not non_bad_failures else ""
    rows["pass_override_mean_ret_net_bar"] = bool(override_mean_ret_net_pass)
    rows["pass_override_worst_month_ret_net_bar"] = bool(override_worst_month_ret_net_pass)
    rows["bad_mae_accepted_by_pnl_override"] = bool(pnl_override)
    if not bad_mae_failures:
        rows["path_risk_status"] = "bad_mae_within_bar"
    elif pnl_override:
        rows["path_risk_status"] = "bad_mae_accepted_by_pnl_override"
    elif non_bad_failures:
        rows["path_risk_status"] = "bad_mae_and_other_failures"
    else:
        rows["path_risk_status"] = "bad_mae_blocked_insufficient_pnl"
    rows["pnl_override_candidate"] = bool(pnl_override)
    if all(checks.values()):
        rows["gate3_status"] = "strict_pass_candidate"
    elif pnl_override:
        rows["gate3_status"] = "pnl_override_candidate"
    else:
        rows["gate3_status"] = "blocked"
    return rows


def _breakdown_metrics(variant: str, frame: pd.DataFrame, grouping_name: str, cols: tuple[str, ...]) -> list[dict[str, Any]]:
    if frame.empty or not all(col in frame.columns for col in cols):
        return []
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(list(cols), dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec: dict[str, Any] = {
            "variant": variant,
            "grouping": grouping_name,
            "rows": int(len(group)),
            "symbols": int(group.get("__symbol__", group.get("symbol", pd.Series(dtype=str))).nunique()),
            "months": int(group.get("month", pd.Series(dtype=str)).nunique()),
            "mean_exec_margin": _mean(group.get("exec_margin")),
            "mean_ret_net": _mean(group.get("ret_net")),
            "mean_u_policy_net": _mean(group.get("u_policy_net")),
            "full_path_bad_mae": _rate(group.get("full_path_bad_mae_1r")),
            "timeout": _rate(group.get("timeout")),
            "inferred_decision_stop_touch": _rate(group.get("inferred_decision_stop_touch")),
            "inferred_full_path_stop_touch": _rate(group.get("inferred_full_path_stop_touch")),
            "inferred_raw_path_stop_touch": _rate(group.get("inferred_raw_path_stop_touch")),
            "clean_exec_precision": _rate(group.get("clean_exec")),
            "dirty_positive_rate": _rate(group.get("dirty_positive")),
            "positive_margin_rate": _positive_margin_rate(group),
            "mfe_before_mae_rate": _rate(group.get("mfe_before_mae_1r")),
            "mae_before_mfe_rate": _rate(group.get("mae_before_mfe_1r")),
        }
        for col, value in zip(cols, key, strict=False):
            rec[col] = value
        rows.append(rec)
    return rows


def _failure_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, rec in summary.iterrows():
        failed = [item for item in str(rec.get("failed_checks", "")).split(",") if item]
        for item in failed:
            rows.append(
                {
                    "variant": rec["variant"],
                    "failed_check": item,
                    "rows": rec["rows"],
                    "symbols": rec["symbols"],
                    "mean_exec_margin": rec["mean_exec_margin"],
                    "mean_ret_net": rec.get("mean_ret_net", np.nan),
                    "worst_month_ret_net": rec.get("worst_month_ret_net", np.nan),
                    "full_path_bad_mae": rec["full_path_bad_mae"],
                    "max_month_full_path_bad_mae": rec["max_month_full_path_bad_mae"],
                    "timeout": rec["timeout"],
                    "dominant_side_share": rec["dominant_side_share"],
                }
            )
    return pd.DataFrame(rows)


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "nan"
    if not math.isfinite(val):
        return "nan"
    return f"{val * 100:.2f}%"


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    breakdown: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    max_side_share: float,
    allow_bad_mae_pnl_override: bool,
    min_override_mean_ret_net: float,
    min_override_worst_month_ret_net: float,
) -> None:
    best = summary.sort_values(
        ["pass_month_bad_mae_bar", "mean_exec_margin", "max_month_full_path_bad_mae"],
        ascending=[False, False, True],
    ).head(1)
    lines = [
        "# S52 Meta Handoff Gate 3 Readiness",
        "",
        "## Scope",
        "",
        "Compares materialized S52 meta-threshold handoff variants using offline eval outcomes, while auditing clean handoff files for realized-outcome leakage columns.",
        "",
        "## Gate Bars",
        "",
        "- mean executable margin > 0",
        "- worst-month executable margin > 0",
        "- row-weighted full-path bad-MAE <= 50%",
        "- max-month full-path bad-MAE <= 50%",
        "- in explicit PnL-override mode, bad-MAE above 50% is a path-risk warning rather than a veto when it is the only failed check and net-return floors pass",
        "- timeout <= 12%",
        f"- dominant side share <= {_fmt_pct(max_side_share)}",
        "- clean handoff has no realized outcome columns",
        "- duplicate decision keys = 0",
        "",
        "## Variant Summary",
        "",
        summary[
            [
                "variant",
                "rows",
                "symbols",
                "mean_exec_margin",
                "mean_ret_net",
                "worst_month_ret_net",
                "worst_month_exec_margin",
                "full_path_bad_mae",
                "max_month_full_path_bad_mae",
                "timeout",
                "inferred_decision_stop_touch",
                "inferred_full_path_stop_touch",
                "dominant_side_share",
                "path_risk_status",
                "bad_mae_accepted_by_pnl_override",
                "gate3_status",
                "failed_checks",
            ]
        ].to_markdown(index=False),
        "",
    ]
    if not best.empty:
        rec = best.iloc[0]
        lines += [
            "## Best Diagnostic Variant",
            "",
            f"- variant: `{rec['variant']}`",
            f"- rows: `{int(rec['rows'])}`",
            f"- symbols: `{int(rec['symbols'])}`",
            f"- mean executable margin: `{_fmt_pct(rec['mean_exec_margin'])}`",
            f"- mean net return: `{_fmt_pct(rec.get('mean_ret_net'))}`",
            f"- worst-month net return: `{_fmt_pct(rec.get('worst_month_ret_net'))}`",
            f"- worst-month executable margin: `{_fmt_pct(rec['worst_month_exec_margin'])}`",
            f"- full-path bad-MAE: `{_fmt_pct(rec['full_path_bad_mae'])}`",
            f"- max-month full-path bad-MAE: `{_fmt_pct(rec['max_month_full_path_bad_mae'])}`",
            f"- timeout: `{_fmt_pct(rec['timeout'])}`",
            f"- inferred decision-window stop touch: `{_fmt_pct(rec.get('inferred_decision_stop_touch'))}`",
            f"- inferred full-path stop touch: `{_fmt_pct(rec.get('inferred_full_path_stop_touch'))}`",
            f"- dominant side share: `{_fmt_pct(rec['dominant_side_share'])}`",
            f"- status: `{rec['gate3_status']}`",
            "",
        ]
    lines += [
        "## PnL Override",
        "",
        f"- enabled: `{bool(allow_bad_mae_pnl_override)}`",
        f"- minimum mean net return: `{_fmt_pct(min_override_mean_ret_net)}`",
        f"- minimum worst-month net return: `{_fmt_pct(min_override_worst_month_ret_net)}`",
        "- override applies only to bad-MAE failures; leakage, support, timeout, side-share, and positive-economics checks must still pass.",
        "",
        "## Failure Matrix",
        "",
        failures.to_markdown(index=False) if not failures.empty else "_No failed checks._",
        "",
        "## Largest Side/Source Buckets",
        "",
    ]
    if not breakdown.empty:
        side_source = breakdown[breakdown["grouping"].eq("side_source")].sort_values(["variant", "rows"], ascending=[True, False])
        lines.append(side_source.head(40).to_markdown(index=False) if not side_source.empty else "_No side/source breakdown._")
    else:
        lines.append("_No breakdown rows._")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run_audit(
    *,
    handoff_root: Path,
    variants: tuple[str, ...],
    out_dir: Path,
    max_side_share: float,
    min_rows: int,
    min_symbols: int,
    scored_ledger_path: Path | None = None,
    stop_mult: float = 0.50,
    allow_bad_mae_pnl_override: bool = False,
    min_override_mean_ret_net: float = 0.010,
    min_override_worst_month_ret_net: float = 0.005,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_ledger: pd.DataFrame | None = None
    if scored_ledger_path is not None and scored_ledger_path.exists():
        scored_ledger = pd.read_parquet(scored_ledger_path)
    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for variant_name in variants:
        variant_dir = handoff_root / variant_name
        if not variant_dir.exists():
            raise FileNotFoundError(variant_dir)
        clean, offline, audit = _read_variant(variant_dir)
        offline = _enrich_stop_touch_metrics(offline, scored_ledger=scored_ledger, stop_mult=float(stop_mult))
        variant = _variant_label(variant_dir)
        summary_rows.append(
            _gate_summary(
                variant,
                variant_dir,
                clean,
                offline,
                audit,
                max_side_share=max_side_share,
                min_rows=min_rows,
                min_symbols=min_symbols,
                allow_bad_mae_pnl_override=allow_bad_mae_pnl_override,
                min_override_mean_ret_net=min_override_mean_ret_net,
                min_override_worst_month_ret_net=min_override_worst_month_ret_net,
            )
        )
        for grouping_name, cols in GROUPING_SPECS.items():
            breakdown_rows.extend(_breakdown_metrics(variant, offline, grouping_name, cols))
    summary = pd.DataFrame(summary_rows).sort_values(
        ["gate3_status", "mean_exec_margin", "max_month_full_path_bad_mae"],
        ascending=[True, False, True],
    )
    breakdown = pd.DataFrame(breakdown_rows)
    failures = _failure_rows(summary)
    paths = {
        "summary": out_dir / "s52_meta_handoff_gate3_readiness_summary.csv",
        "breakdown": out_dir / "s52_meta_handoff_gate3_readiness_breakdown.csv",
        "failures": out_dir / "s52_meta_handoff_gate3_readiness_failures.csv",
        "report": out_dir / "s52_meta_handoff_gate3_readiness.md",
        "manifest": out_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    breakdown.to_csv(paths["breakdown"], index=False)
    failures.to_csv(paths["failures"], index=False)
    _write_report(
        paths["report"],
        summary,
        breakdown,
        failures,
        max_side_share=max_side_share,
        allow_bad_mae_pnl_override=allow_bad_mae_pnl_override,
        min_override_mean_ret_net=min_override_mean_ret_net,
        min_override_worst_month_ret_net=min_override_worst_month_ret_net,
    )
    manifest = {
        "generated_by": "report_s52_meta_handoff_gate3_readiness",
        "handoff_root": str(handoff_root),
        "variants": list(variants),
        "gate_bars": {
            "mean_exec_margin_gt": 0.0,
            "worst_month_exec_margin_gt": 0.0,
            "full_path_bad_mae_lte": 0.50,
            "max_month_full_path_bad_mae_lte": 0.50,
            "timeout_lte": 0.12,
            "dominant_side_share_lte": float(max_side_share),
            "min_rows": int(min_rows),
            "min_symbols": int(min_symbols),
            "inferred_stop_touch_threshold_r": float(stop_mult),
            "scored_ledger_path": str(scored_ledger_path) if scored_ledger_path is not None else None,
            "allow_bad_mae_pnl_override": bool(allow_bad_mae_pnl_override),
            "min_override_mean_ret_net": float(min_override_mean_ret_net),
            "min_override_worst_month_ret_net": float(min_override_worst_month_ret_net),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "strict_pass_candidates": summary.loc[summary["gate3_status"].eq("strict_pass_candidate"), "variant"].astype(str).tolist(),
        "pnl_override_candidates": summary.loc[summary["gate3_status"].eq("pnl_override_candidate"), "variant"].astype(str).tolist(),
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_HANDOFF_ROOT / "s52_meta_handoff_gate3_readiness_v1")
    parser.add_argument("--variant", action="append", dest="variants", help="Variant directory name under --handoff-root. Repeatable.")
    parser.add_argument("--max-side-share", type=float, default=0.80)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-symbols", type=int, default=50)
    parser.add_argument("--scored-ledger", type=Path, default=DEFAULT_SCORED_LEDGER)
    parser.add_argument("--stop-mult", type=float, default=0.50)
    parser.add_argument("--allow-bad-mae-pnl-override", action="store_true")
    parser.add_argument("--min-override-mean-ret-net", type=float, default=0.010)
    parser.add_argument("--min-override-worst-month-ret-net", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variants = tuple(args.variants) if args.variants else DEFAULT_VARIANTS
    manifest = run_audit(
        handoff_root=args.handoff_root,
        variants=variants,
        out_dir=args.out_dir,
        max_side_share=float(args.max_side_share),
        min_rows=int(args.min_rows),
        min_symbols=int(args.min_symbols),
        scored_ledger_path=args.scored_ledger,
        stop_mult=float(args.stop_mult),
        allow_bad_mae_pnl_override=bool(args.allow_bad_mae_pnl_override),
        min_override_mean_ret_net=float(args.min_override_mean_ret_net),
        min_override_worst_month_ret_net=float(args.min_override_worst_month_ret_net),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
