#!/usr/bin/env python3
"""Audit shadow market-state head-priority modulation across replay windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_priority_shadow_promotion_audit_20260626")


def _accepted_jaccard_failure(min_accepted_jaccard: float) -> str:
    pct = int(round(float(min_accepted_jaccard) * 100.0))
    return f"accepted_jaccard_below_required_{pct}pct"


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _window_from_candidates(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "candidate_rows": 0,
            "timestamp_count": 0,
            "window_start": None,
            "window_end": None,
        }
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except Exception:
        frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        return {
            "candidate_rows": int(len(frame)),
            "timestamp_count": 0,
            "window_start": None,
            "window_end": None,
        }
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    ts = ts.dropna()
    return {
        "candidate_rows": int(len(frame)),
        "timestamp_count": int(ts.nunique()),
        "window_start": ts.min().isoformat() if not ts.empty else None,
        "window_end": ts.max().isoformat() if not ts.empty else None,
    }


def _read_candidate_path(cap_dir: Path, manifest: dict[str, Any]) -> Path | None:
    inputs = dict(manifest.get("inputs") or {})
    raw = inputs.get("candidates")
    if raw:
        return Path(str(raw))
    priority_manifest_path = cap_dir.parent / "manifest.json"
    priority_dir = inputs.get("priority_dir")
    if priority_dir:
        priority_manifest_path = Path(str(priority_dir)) / "manifest.json"
    if priority_manifest_path.exists():
        priority_manifest = _load_json(priority_manifest_path)
        priority_inputs = dict(priority_manifest.get("inputs") or {})
        raw = priority_inputs.get("candidates")
        if raw:
            return Path(str(raw))
    return None


def _select_arm(metrics: pd.DataFrame, arm_contains: str) -> pd.Series:
    if metrics.empty or "arm" not in metrics.columns:
        raise ValueError("metrics file is missing arm rows")
    mask = metrics["arm"].astype(str).str.contains(str(arm_contains), regex=False, na=False)
    selected = metrics.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"no metrics arm contains {arm_contains!r}")
    if len(selected) > 1:
        selected = selected.sort_values(["max_adjustment", "min_abs_z"], ascending=[True, True])
    return selected.iloc[0]


def _portable_arm_selector(arm: str) -> str:
    """Convert a run-specific arm name into a portable cap/z selector."""
    text = str(arm or "")
    marker = "_cap_"
    if marker not in text:
        return text
    return f"cap_{text.split(marker, 1)[1]}"


def selected_challenger_selector(cap_dir: Path) -> str | None:
    """Read a cap-sweep selected challenger and return a portable selector."""
    path = cap_dir / "selected_shadow_challenger.json"
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except Exception:
        return None
    if not bool(payload.get("selected")):
        return None
    arm = payload.get("arm")
    if not arm:
        row = payload.get("selected_row")
        if isinstance(row, dict):
            arm = row.get("arm")
    if not arm:
        return None
    selector = _portable_arm_selector(str(arm))
    return selector or None


def resolve_arm_selector(
    cap_dirs: list[Path],
    *,
    arm_contains: str,
    use_selected_challenger: bool,
) -> tuple[str, str]:
    if not use_selected_challenger:
        return str(arm_contains), "explicit_arm_contains"
    for cap_dir in cap_dirs:
        selector = selected_challenger_selector(cap_dir)
        if selector:
            return selector, f"selected_shadow_challenger:{cap_dir}"
    return str(arm_contains), "fallback_explicit_arm_contains"


def select_recurrent_challenger_selector(
    cap_dirs: list[Path],
    *,
    min_window_count: int = 3,
    min_positive_delta_share: float = 0.50,
    min_action_windows: int = 2,
    min_positive_action_windows: int = 2,
    min_accepted_jaccard: float = 0.95,
    max_full_sl_delta: float = 0.005,
    max_timeout_delta: float = 0.0,
) -> dict[str, Any]:
    """Select a portable cap selector using recurrent multi-window action."""

    rows: list[dict[str, Any]] = []
    for window_idx, cap_dir in enumerate(cap_dirs):
        metrics_path = Path(cap_dir) / "head_priority_cap_sweep_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        if metrics.empty or "arm" not in metrics.columns:
            continue
        for _, row in metrics.iterrows():
            rec = row.to_dict()
            arm = str(rec.get("arm") or "")
            if not arm or arm == "P0_static_priority":
                continue
            rec["source_dir"] = str(cap_dir)
            rec["window_index"] = int(window_idx)
            rec["portable_selector"] = _portable_arm_selector(arm)
            rows.append(rec)
    if not rows:
        return {
            "selected": False,
            "reason": "no_cap_sweep_metric_rows",
            "arm_selector": None,
            "candidates": [],
        }

    frame = pd.DataFrame(rows)
    numeric_cols = [
        "delta_net_pnl",
        "accepted_jaccard",
        "coverage",
        "entrants",
        "removed",
        "delta_full_sl_rate",
        "delta_timeout_rate",
        "net_replacement_pnl",
        "net_action_pnl_delta",
        "max_adjustment",
        "min_abs_z",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    candidates: list[dict[str, Any]] = []
    for selector, group in frame.groupby("portable_selector", sort=True, observed=True):
        delta = pd.to_numeric(group.get("delta_net_pnl"), errors="coerce")
        entrants = pd.to_numeric(group.get("entrants"), errors="coerce").fillna(0.0)
        removed = pd.to_numeric(group.get("removed"), errors="coerce").fillna(0.0)
        action = (entrants + removed) > 0.0
        replacement = pd.to_numeric(group.get("net_replacement_pnl"), errors="coerce")
        action_delta = pd.to_numeric(group.get("net_action_pnl_delta"), errors="coerce")
        jaccard = pd.to_numeric(group.get("accepted_jaccard"), errors="coerce")
        full_sl = pd.to_numeric(group.get("delta_full_sl_rate"), errors="coerce")
        timeout = pd.to_numeric(group.get("delta_timeout_rate"), errors="coerce")
        coverage = pd.to_numeric(group.get("coverage"), errors="coerce")

        failures: list[str] = []
        window_count = int(group["window_index"].nunique())
        action_window_count = int(action.sum())
        positive_action_window_count = int(((delta > 0.0) & action).sum())
        positive_share = float((delta > 0.0).mean()) if len(group) else 0.0
        if window_count < int(min_window_count):
            failures.append("fewer_than_required_windows")
        if float(delta.median()) <= 0.0:
            failures.append("median_delta_net_pnl_not_positive")
        if float(delta.quantile(0.25)) < 0.0:
            failures.append("q25_delta_net_pnl_negative")
        if positive_share < float(min_positive_delta_share):
            failures.append("positive_delta_window_share_below_gate")
        if action_window_count < int(min_action_windows):
            failures.append("fewer_than_required_action_windows")
        if positive_action_window_count < int(min_positive_action_windows):
            failures.append("fewer_than_required_positive_action_windows")
        if jaccard.notna().any() and float(jaccard.min()) < float(min_accepted_jaccard):
            failures.append(_accepted_jaccard_failure(min_accepted_jaccard))
        if coverage.notna().any() and bool((coverage < 0.999).any()):
            failures.append("schedule_coverage_below_99p9pct")
        if full_sl.notna().any() and bool((full_sl > float(max_full_sl_delta)).any()):
            failures.append("full_sl_worsened_in_a_window")
        if timeout.notna().any() and bool((timeout > float(max_timeout_delta)).any()):
            failures.append("timeout_worsened_in_a_window")
        if bool(action.any()):
            if float(replacement[action].median()) <= 0.0:
                failures.append("action_window_replacement_pnl_not_positive")
            if float(action_delta[action].median()) <= 0.0:
                failures.append("action_window_net_action_pnl_not_positive")
        else:
            failures.append("no_action_windows")

        max_adjustment = (
            float(pd.to_numeric(group.get("max_adjustment"), errors="coerce").median())
            if "max_adjustment" in group
            else np.nan
        )
        min_abs_z = (
            float(pd.to_numeric(group.get("min_abs_z"), errors="coerce").median())
            if "min_abs_z" in group
            else np.nan
        )
        median_action_delta = float(action_delta[action].median()) if bool(action.any()) else 0.0
        median_replacement = float(replacement[action].median()) if bool(action.any()) else 0.0
        score = (
            float(delta.median())
            + 0.50 * median_action_delta
            + 0.25 * median_replacement
            - 100.0 * max(float(full_sl.max()) if full_sl.notna().any() else 0.0, 0.0)
            - 50.0 * max(float(timeout.max()) if timeout.notna().any() else 0.0, 0.0)
        )
        candidates.append(
            {
                "arm_selector": str(selector),
                "selected": False,
                "gate_passed": not failures,
                "fail_reasons": ";".join(failures),
                "selection_score": score,
                "window_count": window_count,
                "action_window_count": action_window_count,
                "positive_action_window_count": positive_action_window_count,
                "median_delta_net_pnl": float(delta.median()),
                "q25_delta_net_pnl": float(delta.quantile(0.25)),
                "positive_delta_window_share": positive_share,
                "min_accepted_jaccard": float(jaccard.min()) if jaccard.notna().any() else None,
                "max_full_sl_delta": float(full_sl.max()) if full_sl.notna().any() else None,
                "max_timeout_delta": float(timeout.max()) if timeout.notna().any() else None,
                "median_replacement_pnl_action_windows": median_replacement if bool(action.any()) else None,
                "median_net_action_pnl_delta_action_windows": (
                    median_action_delta if bool(action.any()) else None
                ),
                "max_adjustment": max_adjustment if np.isfinite(max_adjustment) else None,
                "min_abs_z": min_abs_z if np.isfinite(min_abs_z) else None,
                "source_dirs": sorted(group["source_dir"].astype(str).unique()),
            }
        )

    candidates_frame = pd.DataFrame(candidates)
    passing = candidates_frame.loc[candidates_frame["gate_passed"].astype(bool)].copy()
    policy = {
        "min_window_count": int(min_window_count),
        "min_positive_delta_share": float(min_positive_delta_share),
        "min_action_windows": int(min_action_windows),
        "min_positive_action_windows": int(min_positive_action_windows),
        "min_accepted_jaccard": float(min_accepted_jaccard),
        "max_full_sl_delta": float(max_full_sl_delta),
        "max_timeout_delta": float(max_timeout_delta),
    }
    if passing.empty:
        best = (
            candidates_frame.sort_values(
                ["selection_score", "positive_delta_window_share", "action_window_count"],
                ascending=[False, False, False],
                na_position="last",
            )
            .iloc[0]
            .to_dict()
            if not candidates_frame.empty
            else {}
        )
        return {
            "selected": False,
            "reason": "no_recurrent_gate_passing_arm",
            "arm_selector": None,
            "best_candidate": _json_safe(best),
            "candidates": _json_safe(candidates),
            "selection_policy": policy,
        }

    passing = passing.sort_values(
        ["selection_score", "max_adjustment", "min_abs_z"],
        ascending=[False, True, True],
        na_position="last",
    )
    selected = passing.iloc[0].to_dict()
    selector = str(selected.get("arm_selector"))
    for candidate in candidates:
        if candidate.get("arm_selector") == selector:
            candidate["selected"] = True
    return {
        "selected": True,
        "reason": "selected_recurrent_gate_passing_priority_arm",
        "arm_selector": selector,
        "selected_row": _json_safe(selected),
        "passing_count": int(len(passing)),
        "candidates": _json_safe(candidates),
        "selection_policy": policy,
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def resolve_gate_tolerances(
    cap_dirs: list[Path],
    *,
    min_accepted_jaccard: float | None = None,
    max_full_sl_delta: float | None = None,
    max_timeout_delta: float | None = None,
) -> dict[str, float]:
    """Resolve promotion-gate safety tolerances.

    Direct function calls keep strict defaults.  CLI audits can inherit the
    safety surface recorded by the cap-sweep selector so that selection and
    promotion reports are not using inconsistent full-SL/timeout rules.
    """
    defaults = {
        "min_accepted_jaccard": 0.90,
        "max_full_sl_delta": 0.0,
        "max_timeout_delta": 0.0,
    }
    resolved = dict(defaults)
    for cap_dir in cap_dirs:
        manifest_path = cap_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            continue
        params = dict(manifest.get("params") or {})
        if "selection_min_accepted_jaccard" in params:
            resolved["min_accepted_jaccard"] = _safe_float(
                params.get("selection_min_accepted_jaccard"),
                resolved["min_accepted_jaccard"],
            )
        if "selection_max_full_sl_delta" in params:
            resolved["max_full_sl_delta"] = _safe_float(
                params.get("selection_max_full_sl_delta"),
                resolved["max_full_sl_delta"],
            )
        if "selection_max_timeout_delta" in params:
            resolved["max_timeout_delta"] = _safe_float(
                params.get("selection_max_timeout_delta"),
                resolved["max_timeout_delta"],
            )
        break
    if min_accepted_jaccard is not None:
        resolved["min_accepted_jaccard"] = float(min_accepted_jaccard)
    if max_full_sl_delta is not None:
        resolved["max_full_sl_delta"] = float(max_full_sl_delta)
    if max_timeout_delta is not None:
        resolved["max_timeout_delta"] = float(max_timeout_delta)
    return resolved


def _select_baseline_by_head(by_head: pd.DataFrame) -> pd.DataFrame:
    if by_head.empty or "arm" not in by_head.columns:
        return pd.DataFrame()
    return by_head.loc[by_head["arm"].astype(str).eq("P0_static_priority")].copy()


def _select_arm_by_head(by_head: pd.DataFrame, arm_contains: str) -> pd.DataFrame:
    if by_head.empty or "arm" not in by_head.columns:
        return pd.DataFrame()
    return by_head.loc[by_head["arm"].astype(str).str.contains(str(arm_contains), regex=False, na=False)].copy()


def _select_exact_arm_by_head(by_head: pd.DataFrame, arm: str) -> pd.DataFrame:
    if by_head.empty or "arm" not in by_head.columns:
        return pd.DataFrame()
    return by_head.loc[by_head["arm"].astype(str).eq(str(arm))].copy()


def _window_label(row: dict[str, Any], fallback: str) -> str:
    start = row.get("window_start")
    end = row.get("window_end")
    if start and end:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.date() == end_ts.date():
            return f"{start_ts.strftime('%Y-%m-%d %H:%M')}-{end_ts.strftime('%H:%M')} UTC"
        return f"{start_ts.strftime('%Y-%m-%d %H:%M')}-{end_ts.strftime('%Y-%m-%d %H:%M')} UTC"
    return fallback


def load_window(cap_dir: Path, *, arm_contains: str, label: str | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = cap_dir / "manifest.json"
    metrics_path = cap_dir / "head_priority_cap_sweep_metrics.csv"
    by_head_path = cap_dir / "head_priority_cap_sweep_by_head.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    manifest = _load_json(manifest_path)
    metrics = pd.read_csv(metrics_path)
    selected = _select_arm(metrics, arm_contains)
    candidate_path = _read_candidate_path(cap_dir, manifest)
    window = _window_from_candidates(candidate_path)
    row = {
        "source_dir": str(cap_dir),
        "source_manifest_sha256": _sha256(manifest_path),
        "metrics_sha256": _sha256(metrics_path),
        "candidate_path": str(candidate_path) if candidate_path is not None else None,
        **window,
    }
    row["window_label"] = label or _window_label(row, cap_dir.name)
    for col in selected.index:
        value = selected[col]
        if isinstance(value, np.generic):
            value = value.item()
        row[str(col)] = value
    selected_arm = str(row.get("arm") or "")
    by_head_out = pd.DataFrame()
    if by_head_path.exists():
        by_head = pd.read_csv(by_head_path)
        base = _select_baseline_by_head(by_head)
        arm = _select_exact_arm_by_head(by_head, selected_arm)
        if not arm.empty:
            base = base.rename(columns={c: f"baseline_{c}" for c in base.columns if c not in {"head"}})
            arm = arm.rename(columns={c: f"shadow_{c}" for c in arm.columns if c not in {"head"}})
            by_head_out = arm.merge(base, on="head", how="left")
            by_head_out.insert(0, "window_label", row["window_label"])
            by_head_out.insert(0, "source_dir", str(cap_dir))
            for metric in ["net_pnl", "trade_count", "full_sl_rate", "timeout_rate"]:
                shadow_col = f"shadow_{metric}"
                base_col = f"baseline_{metric}"
                if shadow_col in by_head_out.columns and base_col in by_head_out.columns:
                    by_head_out[f"delta_{metric}"] = (
                        pd.to_numeric(by_head_out[shadow_col], errors="coerce")
                        - pd.to_numeric(by_head_out[base_col], errors="coerce")
                    )
    return row, by_head_out


def head_mix_metrics(by_head: pd.DataFrame) -> pd.DataFrame:
    """Summarize accepted-trade head concentration by replay window.

    Priority modulation can help by changing which head receives scarce global
    auction capacity.  These diagnostics are deliberately head-agnostic: they
    measure concentration and accepted-mix movement without rewarding a named
    strategy such as short_boll or penalizing short_asset specifically.
    """

    if by_head.empty or "window_label" not in by_head.columns:
        return pd.DataFrame()
    baseline_col = (
        "baseline_trade_count"
        if "baseline_trade_count" in by_head.columns
        else "baseline_trade_count"
    )
    shadow_col = (
        "shadow_trade_count"
        if "shadow_trade_count" in by_head.columns
        else "trade_count"
        if "trade_count" in by_head.columns
        else None
    )
    if baseline_col not in by_head.columns or shadow_col is None or shadow_col not in by_head.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for window_label, group in by_head.groupby("window_label", sort=True, observed=True):
        baseline = (
            pd.to_numeric(group[baseline_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        shadow = (
            pd.to_numeric(group[shadow_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        baseline_total = float(baseline.sum())
        shadow_total = float(shadow.sum())
        baseline_share = baseline / baseline_total if baseline_total > 0.0 else baseline * 0.0
        shadow_share = shadow / shadow_total if shadow_total > 0.0 else shadow * 0.0

        def _entropy(share: pd.Series) -> float:
            values = share.to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.size <= 1:
                return 0.0
            entropy = -float(np.sum(values * np.log(values)))
            return float(entropy / np.log(values.size))

        l1_delta = 0.5 * float(np.abs(shadow_share - baseline_share).sum())
        max_abs_delta = float(np.abs(shadow_share - baseline_share).max()) if len(group) else 0.0
        baseline_dominant = float(baseline_share.max()) if len(group) and baseline_total > 0.0 else 0.0
        shadow_dominant = float(shadow_share.max()) if len(group) and shadow_total > 0.0 else 0.0
        baseline_active = int((baseline > 0.0).sum())
        shadow_active = int((shadow > 0.0).sum())
        rows.append(
            {
                "window_label": window_label,
                "baseline_active_head_count": baseline_active,
                "shadow_active_head_count": shadow_active,
                "active_head_count_delta": shadow_active - baseline_active,
                "baseline_dominant_head_share": baseline_dominant,
                "shadow_dominant_head_share": shadow_dominant,
                "dominant_head_share_delta": shadow_dominant - baseline_dominant,
                "head_trade_share_l1_delta": l1_delta,
                "max_head_trade_share_abs_delta": max_abs_delta,
                "baseline_head_mix_entropy": _entropy(baseline_share),
                "shadow_head_mix_entropy": _entropy(shadow_share),
                "head_mix_entropy_delta": _entropy(shadow_share) - _entropy(baseline_share),
                "starved_head_count": int(((baseline > 0.0) & (shadow <= 0.0)).sum()),
                "activated_head_count": int(((baseline <= 0.0) & (shadow > 0.0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def promotion_gate(
    summary: pd.DataFrame,
    *,
    min_accepted_jaccard: float = 0.90,
    max_full_sl_delta: float = 0.0,
    max_timeout_delta: float = 0.0,
    min_shadow_active_head_count: int | None = None,
    max_shadow_dominant_head_share: float | None = None,
    max_head_trade_share_l1_delta: float | None = None,
) -> dict[str, Any]:
    if summary.empty:
        return {"passed": False, "failures": ["empty_summary"]}
    delta = pd.to_numeric(summary["delta_net_pnl"], errors="coerce")
    jaccard = pd.to_numeric(summary["accepted_jaccard"], errors="coerce")
    coverage = pd.to_numeric(summary.get("coverage", pd.Series(np.nan, index=summary.index)), errors="coerce")
    entrants = pd.to_numeric(summary["entrants"], errors="coerce").fillna(0)
    removed = pd.to_numeric(summary["removed"], errors="coerce").fillna(0)
    action = (entrants + removed) > 0
    full_sl_delta = pd.to_numeric(summary["delta_full_sl_rate"], errors="coerce")
    timeout_delta = pd.to_numeric(summary["delta_timeout_rate"], errors="coerce")
    defensive = pd.to_numeric(summary["defensive_success"], errors="coerce")
    shadow_active_heads = pd.to_numeric(
        summary.get("shadow_active_head_count", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    shadow_dominant = pd.to_numeric(
        summary.get("shadow_dominant_head_share", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    head_l1 = pd.to_numeric(
        summary.get("head_trade_share_l1_delta", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    failures: list[str] = []
    if len(summary) < 3:
        failures.append("fewer_than_3_windows")
    if float(delta.median()) <= 0.0:
        failures.append("median_delta_net_pnl_not_positive")
    if float(delta.quantile(0.25)) < 0.0:
        failures.append("q25_delta_net_pnl_negative")
    if float((delta > 0.0).mean()) < 0.50:
        failures.append("positive_delta_window_share_below_50pct")
    if int(action.sum()) < 2:
        failures.append("fewer_than_2_action_windows")
    if int(((delta > 0.0) & action).sum()) < 2:
        failures.append("fewer_than_2_positive_action_windows")
    if float(jaccard.min()) < float(min_accepted_jaccard):
        failures.append(_accepted_jaccard_failure(min_accepted_jaccard))
    if bool((coverage < 0.999).any()):
        failures.append("schedule_coverage_below_99p9pct")
    if bool((full_sl_delta > float(max_full_sl_delta)).any()):
        failures.append("full_sl_worsened_in_a_window")
    if bool((timeout_delta > float(max_timeout_delta)).any()):
        failures.append("timeout_worsened_in_a_window")
    if float(defensive[action].median()) <= 0.0 if bool(action.any()) else True:
        failures.append("action_window_defensive_success_not_positive")
    if min_shadow_active_head_count is not None and shadow_active_heads.notna().any():
        if bool((shadow_active_heads < int(min_shadow_active_head_count)).any()):
            failures.append("shadow_active_head_count_below_gate")
    if max_shadow_dominant_head_share is not None and shadow_dominant.notna().any():
        if bool((shadow_dominant > float(max_shadow_dominant_head_share)).any()):
            failures.append("shadow_dominant_head_share_above_gate")
    if max_head_trade_share_l1_delta is not None and head_l1.notna().any():
        if bool((head_l1 > float(max_head_trade_share_l1_delta)).any()):
            failures.append("head_trade_share_l1_delta_above_gate")
    return {
        "passed": not failures,
        "failures": failures,
        "window_count": int(len(summary)),
        "action_window_count": int(action.sum()),
        "positive_action_window_count": int(((delta > 0.0) & action).sum()),
        "median_delta_net_pnl": float(delta.median()),
        "q25_delta_net_pnl": float(delta.quantile(0.25)),
        "positive_delta_window_share": float((delta > 0.0).mean()),
        "nonnegative_delta_window_share": float((delta >= 0.0).mean()),
        "min_accepted_jaccard": float(jaccard.min()),
        "required_min_accepted_jaccard": float(min_accepted_jaccard),
        "min_coverage": float(coverage.min()) if coverage.notna().any() else None,
        "max_full_sl_delta": float(full_sl_delta.max()),
        "allowed_max_full_sl_delta": float(max_full_sl_delta),
        "max_timeout_delta": float(timeout_delta.max()),
        "allowed_max_timeout_delta": float(max_timeout_delta),
        "median_defensive_success_action_windows": (
            float(defensive[action].median()) if bool(action.any()) else None
        ),
        "min_shadow_active_head_count": (
            int(shadow_active_heads.min()) if shadow_active_heads.notna().any() else None
        ),
        "max_shadow_dominant_head_share": (
            float(shadow_dominant.max()) if shadow_dominant.notna().any() else None
        ),
        "median_head_trade_share_l1_delta": (
            float(head_l1.median()) if head_l1.notna().any() else None
        ),
        "max_head_trade_share_l1_delta": (
            float(head_l1.max()) if head_l1.notna().any() else None
        ),
        "required_min_shadow_active_head_count": (
            int(min_shadow_active_head_count) if min_shadow_active_head_count is not None else None
        ),
        "allowed_max_shadow_dominant_head_share": (
            float(max_shadow_dominant_head_share)
            if max_shadow_dominant_head_share is not None
            else None
        ),
        "allowed_max_head_trade_share_l1_delta": (
            float(max_head_trade_share_l1_delta)
            if max_head_trade_share_l1_delta is not None
            else None
        ),
    }


def opportunity_routing_gate(
    summary: pd.DataFrame,
    *,
    min_accepted_jaccard: float = 0.90,
    max_full_sl_delta: float = 0.0,
    max_timeout_delta: float = 0.0,
    min_shadow_active_head_count: int | None = None,
    max_shadow_dominant_head_share: float | None = None,
    max_head_trade_share_l1_delta: float | None = None,
) -> dict[str, Any]:
    """Gate market-state priority as opportunity allocation, not suppression.

    Priority modulation can improve the portfolio by replacing one accepted
    trade with a better accepted trade from another head.  That is different
    from threshold control, where the decisive metric is defensive success
    from suppressing bad trades.  This gate therefore requires recurrent
    positive replacement/action economics while keeping the same safety
    constraints on overlap, coverage, full-SL and timeout.
    """
    if summary.empty:
        return {"passed": False, "failures": ["empty_summary"]}
    delta = pd.to_numeric(summary["delta_net_pnl"], errors="coerce")
    jaccard = pd.to_numeric(summary["accepted_jaccard"], errors="coerce")
    coverage = pd.to_numeric(summary.get("coverage", pd.Series(np.nan, index=summary.index)), errors="coerce")
    entrants = pd.to_numeric(summary["entrants"], errors="coerce").fillna(0)
    removed = pd.to_numeric(summary["removed"], errors="coerce").fillna(0)
    action = (entrants + removed) > 0
    full_sl_delta = pd.to_numeric(summary["delta_full_sl_rate"], errors="coerce")
    timeout_delta = pd.to_numeric(summary["delta_timeout_rate"], errors="coerce")
    replacement = pd.to_numeric(summary.get("net_replacement_pnl"), errors="coerce")
    action_delta = pd.to_numeric(summary.get("net_action_pnl_delta"), errors="coerce")
    shadow_active_heads = pd.to_numeric(
        summary.get("shadow_active_head_count", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    shadow_dominant = pd.to_numeric(
        summary.get("shadow_dominant_head_share", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    head_l1 = pd.to_numeric(
        summary.get("head_trade_share_l1_delta", pd.Series(np.nan, index=summary.index)),
        errors="coerce",
    )
    failures: list[str] = []
    if len(summary) < 3:
        failures.append("fewer_than_3_windows")
    if float(delta.median()) <= 0.0:
        failures.append("median_delta_net_pnl_not_positive")
    if float(delta.quantile(0.25)) < 0.0:
        failures.append("q25_delta_net_pnl_negative")
    if float((delta > 0.0).mean()) < 0.50:
        failures.append("positive_delta_window_share_below_50pct")
    if int(action.sum()) < 2:
        failures.append("fewer_than_2_action_windows")
    if int(((delta > 0.0) & action).sum()) < 2:
        failures.append("fewer_than_2_positive_action_windows")
    if float(jaccard.min()) < float(min_accepted_jaccard):
        failures.append(_accepted_jaccard_failure(min_accepted_jaccard))
    if bool((coverage < 0.999).any()):
        failures.append("schedule_coverage_below_99p9pct")
    if bool((full_sl_delta > float(max_full_sl_delta)).any()):
        failures.append("full_sl_worsened_in_a_window")
    if bool((timeout_delta > float(max_timeout_delta)).any()):
        failures.append("timeout_worsened_in_a_window")
    if bool(action.any()):
        if float(replacement[action].median()) <= 0.0:
            failures.append("action_window_replacement_pnl_not_positive")
        if float(action_delta[action].median()) <= 0.0:
            failures.append("action_window_net_action_pnl_not_positive")
        if float((replacement[action] > 0.0).mean()) < 0.50:
            failures.append("positive_replacement_window_share_below_50pct")
        if float((action_delta[action] > 0.0).mean()) < 0.50:
            failures.append("positive_action_pnl_window_share_below_50pct")
    else:
        failures.append("no_action_windows")
    if min_shadow_active_head_count is not None and shadow_active_heads.notna().any():
        if bool((shadow_active_heads < int(min_shadow_active_head_count)).any()):
            failures.append("shadow_active_head_count_below_gate")
    if max_shadow_dominant_head_share is not None and shadow_dominant.notna().any():
        if bool((shadow_dominant > float(max_shadow_dominant_head_share)).any()):
            failures.append("shadow_dominant_head_share_above_gate")
    if max_head_trade_share_l1_delta is not None and head_l1.notna().any():
        if bool((head_l1 > float(max_head_trade_share_l1_delta)).any()):
            failures.append("head_trade_share_l1_delta_above_gate")
    return {
        "passed": not failures,
        "failures": failures,
        "window_count": int(len(summary)),
        "action_window_count": int(action.sum()),
        "positive_action_window_count": int(((delta > 0.0) & action).sum()),
        "median_delta_net_pnl": float(delta.median()),
        "q25_delta_net_pnl": float(delta.quantile(0.25)),
        "positive_delta_window_share": float((delta > 0.0).mean()),
        "nonnegative_delta_window_share": float((delta >= 0.0).mean()),
        "min_accepted_jaccard": float(jaccard.min()),
        "required_min_accepted_jaccard": float(min_accepted_jaccard),
        "min_coverage": float(coverage.min()) if coverage.notna().any() else None,
        "max_full_sl_delta": float(full_sl_delta.max()),
        "allowed_max_full_sl_delta": float(max_full_sl_delta),
        "max_timeout_delta": float(timeout_delta.max()),
        "allowed_max_timeout_delta": float(max_timeout_delta),
        "median_replacement_pnl_action_windows": (
            float(replacement[action].median()) if bool(action.any()) else None
        ),
        "median_net_action_pnl_delta_action_windows": (
            float(action_delta[action].median()) if bool(action.any()) else None
        ),
        "positive_replacement_window_share": (
            float((replacement[action] > 0.0).mean()) if bool(action.any()) else None
        ),
        "positive_action_pnl_window_share": (
            float((action_delta[action] > 0.0).mean()) if bool(action.any()) else None
        ),
        "min_shadow_active_head_count": (
            int(shadow_active_heads.min()) if shadow_active_heads.notna().any() else None
        ),
        "max_shadow_dominant_head_share": (
            float(shadow_dominant.max()) if shadow_dominant.notna().any() else None
        ),
        "median_head_trade_share_l1_delta": (
            float(head_l1.median()) if head_l1.notna().any() else None
        ),
        "max_head_trade_share_l1_delta": (
            float(head_l1.max()) if head_l1.notna().any() else None
        ),
        "required_min_shadow_active_head_count": (
            int(min_shadow_active_head_count) if min_shadow_active_head_count is not None else None
        ),
        "allowed_max_shadow_dominant_head_share": (
            float(max_shadow_dominant_head_share)
            if max_shadow_dominant_head_share is not None
            else None
        ),
        "allowed_max_head_trade_share_l1_delta": (
            float(max_head_trade_share_l1_delta)
            if max_head_trade_share_l1_delta is not None
            else None
        ),
    }


def _render_report(summary: pd.DataFrame, by_head: pd.DataFrame, gate: dict[str, Any], manifest: dict[str, Any]) -> str:
    opportunity_gate = gate.get("opportunity_routing_gate") or {}
    defensive_gate = gate.get("defensive_suppression_gate") or gate
    lines = [
        "# Market-State Priority Shadow Promotion Audit",
        "",
        "This audit aggregates fixed-universe replays for the bounded market-state head-priority shadow challenger.",
        "",
        "## Contract",
        "",
        f"- Arm selector: `{manifest['params']['resolved_arm_contains']}`",
        f"- Arm selector source: `{manifest['params']['arm_selector_source']}`",
        "- Active production remains static T1 unless the promotion gate passes.",
        "- q-fail, HeadHealth, threshold control, model scores, thresholds and sizing remain unchanged.",
        "- The action being audited is only `portfolio_priority_adjustment` in the global auction.",
        "",
        "## Window Summary",
        "",
    ]
    view_cols = [
        "window_label",
        "timestamp_count",
        "candidate_rows",
        "active_schedule_share",
        "trade_count",
        "net_pnl",
        "delta_net_pnl",
        "delta_full_sl_rate",
        "delta_timeout_rate",
        "accepted_jaccard",
        "baseline_active_head_count",
        "shadow_active_head_count",
        "shadow_dominant_head_share",
        "head_trade_share_l1_delta",
        "entrants",
        "removed",
        "net_replacement_pnl",
        "defensive_success",
        "gate_passed",
    ]
    lines.append(summary[[c for c in view_cols if c in summary.columns]].to_markdown(index=False))
    lines.extend(["", "## Opportunity-Routing Gate", ""])
    opportunity_view = {k: v for k, v in opportunity_gate.items() if k != "failures"}
    lines.append(pd.DataFrame([opportunity_view]).to_markdown(index=False))
    lines.extend(["", "Opportunity failures:", ""])
    if opportunity_gate.get("failures"):
        lines.extend([f"- `{item}`" for item in opportunity_gate["failures"]])
    else:
        lines.append("- none")
    lines.extend(["", "## Defensive-Suppression Gate", ""])
    defensive_view = {k: v for k, v in defensive_gate.items() if k not in {"failures", "opportunity_routing_gate", "defensive_suppression_gate"}}
    lines.append(pd.DataFrame([defensive_view]).to_markdown(index=False))
    lines.extend(["", "Defensive failures:", ""])
    if defensive_gate.get("failures"):
        lines.extend([f"- `{item}`" for item in defensive_gate["failures"]])
    else:
        lines.append("- none")
    lines.extend(["", "## By Head", ""])
    if by_head.empty:
        lines.append("_No by-head rows._")
    else:
        keep = [
            "window_label",
            "head",
            "baseline_net_pnl",
            "shadow_net_pnl",
            "delta_net_pnl",
            "baseline_trade_count",
            "shadow_trade_count",
            "delta_trade_count",
            "baseline_full_sl_rate",
            "shadow_full_sl_rate",
            "delta_full_sl_rate",
            "baseline_timeout_rate",
            "shadow_timeout_rate",
            "delta_timeout_rate",
        ]
        lines.append(by_head[[c for c in keep if c in by_head.columns]].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The challenger is promotable only if it shows repeated positive action, preserves accepted-set "
                "overlap, and does not worsen full-SL or timeout risk. The opportunity-routing gate evaluates "
                "whether accepted replacements add value. The defensive-suppression gate is retained only as a "
                "separate diagnostic because head-priority modulation is not a threshold-suppression controller."
            ),
            "",
        ]
    )
    if opportunity_gate.get("passed"):
        lines.append("Decision: opportunity-routing gate passed. Review fixed-contract and untouched-window evidence before activation.")
    else:
        lines.append("Decision: opportunity-routing gate failed. Keep the market-state priority arm shadow-only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-sweep-dir", action="append", type=Path, required=True)
    parser.add_argument("--window-label", action="append", default=[])
    parser.add_argument("--arm-contains", default="cap_0p15_zge_0p5")
    parser.add_argument("--use-selected-challenger", action="store_true")
    parser.add_argument(
        "--select-recurrent-challenger",
        action="store_true",
        help="Select the cap/z selector from all supplied cap-sweep windows using recurrent action gates.",
    )
    parser.add_argument("--recurrent-min-window-count", type=int, default=3)
    parser.add_argument("--recurrent-min-positive-delta-share", type=float, default=0.50)
    parser.add_argument("--recurrent-min-action-windows", type=int, default=2)
    parser.add_argument("--recurrent-min-positive-action-windows", type=int, default=2)
    parser.add_argument("--min-accepted-jaccard", type=float, default=None)
    parser.add_argument("--max-full-sl-delta", type=float, default=None)
    parser.add_argument("--max-timeout-delta", type=float, default=None)
    parser.add_argument("--min-shadow-active-head-count", type=int, default=None)
    parser.add_argument("--max-shadow-dominant-head-share", type=float, default=None)
    parser.add_argument("--max-head-trade-share-l1-delta", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap_dirs = list(args.cap_sweep_dir or [])
    gate_tolerances = resolve_gate_tolerances(
        cap_dirs,
        min_accepted_jaccard=args.min_accepted_jaccard,
        max_full_sl_delta=args.max_full_sl_delta,
        max_timeout_delta=args.max_timeout_delta,
    )
    recurrent_selection: dict[str, Any] | None = None
    if bool(args.select_recurrent_challenger):
        recurrent_selection = select_recurrent_challenger_selector(
            cap_dirs,
            min_window_count=int(args.recurrent_min_window_count),
            min_positive_delta_share=float(args.recurrent_min_positive_delta_share),
            min_action_windows=int(args.recurrent_min_action_windows),
            min_positive_action_windows=int(args.recurrent_min_positive_action_windows),
            min_accepted_jaccard=float(gate_tolerances["min_accepted_jaccard"]),
            max_full_sl_delta=float(gate_tolerances["max_full_sl_delta"]),
            max_timeout_delta=float(gate_tolerances["max_timeout_delta"]),
        )
        (args.output_dir / "recurrent_shadow_challenger.json").write_text(
            json.dumps(_json_safe(recurrent_selection), indent=2) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(recurrent_selection.get("candidates") or []).to_csv(
            args.output_dir / "recurrent_shadow_challenger_candidates.csv",
            index=False,
        )
    if recurrent_selection and bool(recurrent_selection.get("selected")):
        resolved_arm_contains = str(recurrent_selection.get("arm_selector"))
        arm_selector_source = "recurrent_shadow_challenger"
    else:
        resolved_arm_contains, arm_selector_source = resolve_arm_selector(
            cap_dirs,
            arm_contains=str(args.arm_contains),
            use_selected_challenger=bool(args.use_selected_challenger),
        )
    labels = list(args.window_label or [])
    rows: list[dict[str, Any]] = []
    by_head_frames: list[pd.DataFrame] = []
    for idx, cap_dir in enumerate(cap_dirs):
        label = labels[idx] if idx < len(labels) else None
        row, by_head = load_window(cap_dir, arm_contains=resolved_arm_contains, label=label)
        rows.append(row)
        if not by_head.empty:
            by_head_frames.append(by_head)
    summary = pd.DataFrame(rows)
    summary["window_start_sort"] = pd.to_datetime(summary["window_start"], utc=True, errors="coerce")
    summary = summary.sort_values(["window_start_sort", "source_dir"], na_position="last").drop(
        columns=["window_start_sort"]
    )
    by_head = pd.concat(by_head_frames, ignore_index=True) if by_head_frames else pd.DataFrame()
    mix = head_mix_metrics(by_head)
    if not mix.empty:
        summary = summary.merge(mix, on="window_label", how="left", validate="one_to_one")
    head_mix_gate_kwargs = {
        "min_shadow_active_head_count": args.min_shadow_active_head_count,
        "max_shadow_dominant_head_share": args.max_shadow_dominant_head_share,
        "max_head_trade_share_l1_delta": args.max_head_trade_share_l1_delta,
    }
    defensive_gate = promotion_gate(summary, **gate_tolerances, **head_mix_gate_kwargs)
    opportunity_gate = opportunity_routing_gate(summary, **gate_tolerances, **head_mix_gate_kwargs)
    gate = {
        **defensive_gate,
        "gate_type": "shadow_priority_with_defensive_and_opportunity_gates",
        "defensive_suppression_gate": defensive_gate,
        "opportunity_routing_gate": opportunity_gate,
        "opportunity_routing_passed": bool(opportunity_gate.get("passed")),
        "opportunity_should_remain_shadow": not bool(opportunity_gate.get("passed")),
    }
    manifest = {
        "generated_by": "audit_market_state_priority_shadow_promotion",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "arm_contains": str(args.arm_contains),
            "use_selected_challenger": bool(args.use_selected_challenger),
            "resolved_arm_contains": resolved_arm_contains,
            "arm_selector_source": arm_selector_source,
            "cap_sweep_dirs": [str(p) for p in cap_dirs],
            "window_labels": labels,
            "gate_tolerances": gate_tolerances,
            "select_recurrent_challenger": bool(args.select_recurrent_challenger),
            "recurrent_selector": recurrent_selection,
            "head_mix_gate": {
                "min_shadow_active_head_count": args.min_shadow_active_head_count,
                "max_shadow_dominant_head_share": args.max_shadow_dominant_head_share,
                "max_head_trade_share_l1_delta": args.max_head_trade_share_l1_delta,
            },
        },
        "outputs": {
            "window_summary": str(args.output_dir / "market_state_priority_shadow_window_summary.csv"),
            "by_head": str(args.output_dir / "market_state_priority_shadow_by_head.csv"),
            "promotion_gate": str(args.output_dir / "market_state_priority_shadow_promotion_gate.json"),
            "report": str(args.output_dir / "market_state_priority_shadow_promotion_report.md"),
        },
    }
    summary.to_csv(args.output_dir / "market_state_priority_shadow_window_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "market_state_priority_shadow_by_head.csv", index=False)
    (args.output_dir / "market_state_priority_shadow_promotion_gate.json").write_text(
        json.dumps(_json_safe(gate), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(summary, by_head, gate, manifest)
    (args.output_dir / "market_state_priority_shadow_promotion_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "gate": gate}), indent=2))


if __name__ == "__main__":
    main()
