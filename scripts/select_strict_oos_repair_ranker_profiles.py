#!/usr/bin/env python3
"""Freeze strict OOS repair-ranker candidates from one month and read out holdout.

This is a post-hoc diagnostic guardrail around
``run_strict_oos_repair_ranker_ablation.py``. It does not fit models. It reads
the month-forward repair-ranker ledger, selects candidate profiles using one
selection month only, and reports the same profiles on a later holdout month.

The purpose is to reduce the bias from ranking profiles on the full May+June
aggregate before deciding which repair setup is worth validating next.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _json_safe, _safe_mean  # noqa: E402


DEFAULT_INPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_ablation"
)
DEFAULT_MONTHLY = DEFAULT_INPUT_DIR / "strict_oos_repair_ranker_monthly.csv"
DEFAULT_SOURCE_MANIFEST = DEFAULT_INPUT_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_candidate_freeze"
)
PROFILE_KEYS = ["source_bucket", "proxy_col", "top_frac", "feature_mode", "selection_method"]
MONTHLY_METRIC_COLS = [
    "train_events",
    "train_positive_events",
    "train_negative_events",
    "scope_rows",
    "selected_rows",
    "repair_mean_u",
    "proxy_mean_u",
    "oracle_mean_u",
    "scope_mean_u",
    "repair_delta_mean_u_vs_proxy",
    "repair_delta_mean_u_vs_scope",
    "repair_hit_u",
    "proxy_hit_u",
    "repair_bad_mae_1r_rate",
    "proxy_bad_mae_1r_rate",
    "repair_timeout_or_slow_holding_rate",
    "proxy_timeout_or_slow_holding_rate",
    "repair_economic_capture_rate",
    "proxy_economic_capture_rate",
    "repair_recoverable_rate",
    "proxy_recoverable_rate",
    "repair_oracle_capture_at_k",
    "proxy_oracle_capture_at_k",
    "repair_delta_oracle_capture_at_k",
    "repair_proxy_overlap_at_k",
]


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _proxy_family(proxy_col: str) -> str:
    if proxy_col in {"oof_pred", "oof_meta_clf"}:
        return "oof_meta_pair"
    if proxy_col in {"oof_base_clf", "base_rank_pct", "pred_H10_pred_mean", "base_H10_pred_mean"}:
        return "base_rank_family"
    return str(proxy_col)


def _load_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(["period", *PROFILE_KEYS]).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["period"] = frame["period"].astype(str)
    frame["top_frac"] = _safe_numeric(frame["top_frac"]).round(6)
    for col in MONTHLY_METRIC_COLS:
        if col in frame.columns:
            frame[col] = _safe_numeric(frame[col])
    return frame


def _selection_score(frame: pd.DataFrame) -> pd.Series:
    delta = _safe_numeric(frame["repair_delta_mean_u_vs_proxy"]).fillna(-1.0)
    mean_u = _safe_numeric(frame["repair_mean_u"]).fillna(-1.0)
    oracle_delta = _safe_numeric(frame["repair_delta_oracle_capture_at_k"]).fillna(0.0)
    bad_mae_excess = (
        _safe_numeric(frame["repair_bad_mae_1r_rate"])
        - _safe_numeric(frame["proxy_bad_mae_1r_rate"])
    ).clip(lower=0.0).fillna(0.0)
    timeout_excess = (
        _safe_numeric(frame["repair_timeout_or_slow_holding_rate"])
        - _safe_numeric(frame["proxy_timeout_or_slow_holding_rate"])
    ).clip(lower=0.0).fillna(0.0)
    selected_rows = _safe_numeric(frame["selected_rows"]).fillna(0.0)
    row_scale = np.log1p(selected_rows) / np.log(31.0)
    return delta + (0.25 * mean_u) + (0.02 * oracle_delta) + (0.005 * row_scale) - (0.01 * bad_mae_excess) - (0.01 * timeout_excess)


def _candidate_pool(
    monthly: pd.DataFrame,
    *,
    selection_month: str,
    min_selected_rows: int,
    min_train_class_rows: int,
    min_selection_delta: float,
    min_selection_mean_u: float,
    min_oracle_capture_delta: float,
    max_bad_mae_rate: float,
    max_bad_mae_excess: float,
    max_timeout_excess: float,
) -> pd.DataFrame:
    selection = monthly[monthly["period"].eq(selection_month)].copy()
    if selection.empty:
        return selection

    selection["bad_mae_excess"] = (
        _safe_numeric(selection["repair_bad_mae_1r_rate"])
        - _safe_numeric(selection["proxy_bad_mae_1r_rate"])
    )
    selection["timeout_excess"] = (
        _safe_numeric(selection["repair_timeout_or_slow_holding_rate"])
        - _safe_numeric(selection["proxy_timeout_or_slow_holding_rate"])
    )
    mask = _safe_numeric(selection["selected_rows"]).ge(min_selected_rows)
    mask &= _safe_numeric(selection["train_positive_events"]).ge(min_train_class_rows)
    mask &= _safe_numeric(selection["train_negative_events"]).ge(min_train_class_rows)
    mask &= _safe_numeric(selection["repair_delta_mean_u_vs_proxy"]).ge(min_selection_delta)
    mask &= _safe_numeric(selection["repair_mean_u"]).ge(min_selection_mean_u)
    mask &= _safe_numeric(selection["repair_delta_oracle_capture_at_k"]).ge(min_oracle_capture_delta)
    mask &= _safe_numeric(selection["repair_bad_mae_1r_rate"]).le(max_bad_mae_rate)
    mask &= selection["bad_mae_excess"].fillna(0.0).le(max_bad_mae_excess)
    mask &= selection["timeout_excess"].fillna(0.0).le(max_timeout_excess)
    pool = selection.loc[mask].copy()
    if pool.empty:
        return pool
    pool["proxy_family"] = pool["proxy_col"].map(_proxy_family)
    pool["selection_score"] = _selection_score(pool)
    pool = pool.sort_values(
        ["selection_score", "repair_delta_mean_u_vs_proxy", "repair_mean_u"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return pool


def _dedupe_candidates(pool: pd.DataFrame, *, max_profiles: int) -> pd.DataFrame:
    if pool.empty:
        return pool
    dedupe_cols = ["source_bucket", "top_frac", "feature_mode", "selection_method", "proxy_family"]
    ranked = pool.drop_duplicates(dedupe_cols, keep="first").copy()
    # A second pass forces source-bucket diversity before filling remaining slots.
    diverse = ranked.drop_duplicates(["source_bucket"], keep="first").head(max_profiles).copy()
    if len(diverse) < max_profiles:
        remaining = ranked[~ranked.index.isin(diverse.index)].head(max_profiles - len(diverse))
        diverse = pd.concat([diverse, remaining], ignore_index=False)
    return diverse.sort_values("selection_score", ascending=False, kind="mergesort").head(max_profiles).reset_index(drop=True)


def _prefixed(row: pd.Series, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in MONTHLY_METRIC_COLS:
        if col in row.index:
            out[f"{prefix}_{col}"] = row[col]
    return out


def _build_profile_summary(
    selected: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    selection_month: str,
    holdout_month: str,
    min_holdout_oracle_capture: float,
    max_bad_mae_excess: float,
    max_timeout_excess: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    holdout = monthly[monthly["period"].eq(holdout_month)].copy()
    for rank, row in selected.reset_index(drop=True).iterrows():
        key_mask = pd.Series(True, index=holdout.index)
        for key in PROFILE_KEYS:
            key_mask &= holdout[key].eq(row[key])
        hold = holdout.loc[key_mask]
        base = {
            "profile_rank": int(rank + 1),
            "selection_month": selection_month,
            "holdout_month": holdout_month,
            **{key: row[key] for key in PROFILE_KEYS},
            "proxy_family": row.get("proxy_family"),
            "selection_score": row.get("selection_score"),
            **_prefixed(row, "selection"),
        }
        if hold.empty:
            base.update({"holdout_status": "missing_holdout_row"})
            rows.append(base)
            continue
        hold_row = hold.iloc[0]
        holdout_delta = float(hold_row["repair_delta_mean_u_vs_proxy"])
        holdout_mean = float(hold_row["repair_mean_u"])
        holdout_bad_excess = float(
            hold_row["repair_bad_mae_1r_rate"] - hold_row["proxy_bad_mae_1r_rate"]
        )
        holdout_timeout_excess = float(
            hold_row["repair_timeout_or_slow_holding_rate"]
            - hold_row["proxy_timeout_or_slow_holding_rate"]
        )
        holdout_oracle_delta = float(hold_row["repair_delta_oracle_capture_at_k"])
        holdout_oracle_capture = float(hold_row["repair_oracle_capture_at_k"])
        failure_reasons: list[str] = []
        if not (math.isfinite(holdout_mean) and holdout_mean > 0.0):
            failure_reasons.append("non_positive_holdout_mean")
        if not (math.isfinite(holdout_delta) and holdout_delta > 0.0):
            failure_reasons.append("does_not_beat_proxy")
        if not (math.isfinite(holdout_oracle_delta) and holdout_oracle_delta >= 0.0):
            failure_reasons.append("loses_oracle_capture_vs_proxy")
        if not (math.isfinite(holdout_oracle_capture) and holdout_oracle_capture >= min_holdout_oracle_capture):
            failure_reasons.append("insufficient_oracle_capture")
        if math.isfinite(holdout_bad_excess) and holdout_bad_excess > max_bad_mae_excess:
            failure_reasons.append("bad_mae_excess")
        if math.isfinite(holdout_timeout_excess) and holdout_timeout_excess > max_timeout_excess:
            failure_reasons.append("timeout_excess")
        survives = (
            math.isfinite(holdout_mean)
            and holdout_mean > 0.0
            and math.isfinite(holdout_delta)
            and holdout_delta > 0.0
            and math.isfinite(holdout_oracle_delta)
            and holdout_oracle_delta >= 0.0
            and math.isfinite(holdout_oracle_capture)
            and holdout_oracle_capture >= min_holdout_oracle_capture
            and (not math.isfinite(holdout_bad_excess) or holdout_bad_excess <= max_bad_mae_excess)
            and (not math.isfinite(holdout_timeout_excess) or holdout_timeout_excess <= max_timeout_excess)
        )
        if survives:
            status = "survives_holdout"
        elif math.isfinite(holdout_delta) and holdout_delta > 0.0:
            status = "beats_proxy_but_fails_guard"
        else:
            status = "fails_holdout_delta"
        base.update(
            {
                **_prefixed(hold_row, "holdout"),
                "holdout_bad_mae_excess": holdout_bad_excess,
                "holdout_timeout_excess": holdout_timeout_excess,
                "holdout_failure_reasons": ",".join(failure_reasons),
                "holdout_status": status,
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    candidate_pool: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_repair_ranker_candidate_freeze_report.md"
    pool_cols = [
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature_mode",
        "selection_method",
        "selected_rows",
        "repair_mean_u",
        "proxy_mean_u",
        "repair_delta_mean_u_vs_proxy",
        "repair_delta_oracle_capture_at_k",
        "repair_bad_mae_1r_rate",
        "proxy_bad_mae_1r_rate",
        "selection_score",
    ]
    summary_cols = [
        "profile_rank",
        "holdout_status",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature_mode",
        "selection_method",
        "selection_repair_mean_u",
        "selection_proxy_mean_u",
        "selection_repair_delta_mean_u_vs_proxy",
        "holdout_repair_mean_u",
        "holdout_proxy_mean_u",
        "holdout_repair_delta_mean_u_vs_proxy",
        "holdout_repair_oracle_capture_at_k",
        "holdout_proxy_oracle_capture_at_k",
        "holdout_repair_bad_mae_1r_rate",
        "holdout_proxy_bad_mae_1r_rate",
        "holdout_bad_mae_excess",
        "holdout_failure_reasons",
    ]
    lines = [
        "# Strict OOS Repair Ranker Candidate Freeze",
        "",
        "Post-hoc diagnostic candidate freeze. Profiles are selected using only the selection month, then read out on the later holdout month.",
        "",
        "## Protocol",
        "",
        f"- Selection month: `{manifest['selection_month']}`",
        f"- Holdout month: `{manifest['holdout_month']}`",
        f"- Input monthly ledger: `{manifest['monthly_path']}`",
        f"- Candidate pool rows: `{manifest['candidate_pool_rows']}`",
        f"- Selected profiles: `{manifest['selected_profiles']}`",
        "",
        "This reduces the May+June aggregate-selection bias, but it is still post-hoc because the repair grid was already generated before this freeze report.",
        "",
        "## Holdout Readout",
        "",
        _table(summary, summary_cols, limit=None),
        "",
        "## Candidate Pool Top Rows",
        "",
        _table(candidate_pool, pool_cols, limit=30),
        "",
        "## Interpretation",
        "",
        "- `survives_holdout` means the profile beat the same proxy in June, stayed positive, did not lose oracle capture, and did not exceed configured risk-slack guards.",
        "- A surviving profile is a candidate for a later untouched validation period, not a production gate.",
        "- If no profile survives, the repair-ranker signal is not robust enough for training integration.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    monthly_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    selection_month: str,
    holdout_month: str,
    max_profiles: int,
    min_selected_rows: int,
    min_train_class_rows: int,
    min_selection_delta: float,
    min_selection_mean_u: float,
    min_oracle_capture_delta: float,
    max_bad_mae_rate: float,
    min_holdout_oracle_capture: float,
    max_bad_mae_excess: float,
    max_timeout_excess: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = _load_monthly(monthly_path)
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    pool = _candidate_pool(
        monthly,
        selection_month=selection_month,
        min_selected_rows=min_selected_rows,
        min_train_class_rows=min_train_class_rows,
        min_selection_delta=min_selection_delta,
        min_selection_mean_u=min_selection_mean_u,
        min_oracle_capture_delta=min_oracle_capture_delta,
        max_bad_mae_rate=max_bad_mae_rate,
        max_bad_mae_excess=max_bad_mae_excess,
        max_timeout_excess=max_timeout_excess,
    )
    selected = _dedupe_candidates(pool, max_profiles=max_profiles)
    summary = _build_profile_summary(
        selected,
        monthly,
        selection_month=selection_month,
        holdout_month=holdout_month,
        min_holdout_oracle_capture=min_holdout_oracle_capture,
        max_bad_mae_excess=max_bad_mae_excess,
        max_timeout_excess=max_timeout_excess,
    )

    paths = {
        "candidate_pool": output_dir / "strict_oos_repair_ranker_candidate_pool.csv",
        "selected_profiles": output_dir / "strict_oos_repair_ranker_selected_profiles.csv",
        "profile_summary": output_dir / "strict_oos_repair_ranker_candidate_holdout_summary.csv",
        "selected_profiles_json": output_dir / "strict_oos_repair_ranker_selected_profiles.json",
        "manifest": output_dir / "manifest.json",
    }
    pool.to_csv(paths["candidate_pool"], index=False)
    selected.to_csv(paths["selected_profiles"], index=False)
    summary.to_csv(paths["profile_summary"], index=False)
    selected_records = selected[PROFILE_KEYS + ["proxy_family", "selection_score"]].to_dict("records") if not selected.empty else []
    paths["selected_profiles_json"].write_text(
        json.dumps(_json_safe(selected_records), indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "scope": "strict_oos_repair_ranker_candidate_freeze",
        "monthly_path": str(monthly_path),
        "source_manifest_path": str(source_manifest_path),
        "output_dir": str(output_dir),
        "selection_month": selection_month,
        "holdout_month": holdout_month,
        "max_profiles": int(max_profiles),
        "selection_rules": {
            "min_selected_rows": int(min_selected_rows),
            "min_train_class_rows": int(min_train_class_rows),
            "min_selection_delta": float(min_selection_delta),
            "min_selection_mean_u": float(min_selection_mean_u),
            "min_oracle_capture_delta": float(min_oracle_capture_delta),
            "max_bad_mae_rate": float(max_bad_mae_rate),
            "min_holdout_oracle_capture": float(min_holdout_oracle_capture),
            "max_bad_mae_excess": float(max_bad_mae_excess),
            "max_timeout_excess": float(max_timeout_excess),
        },
        "source_manifest_scope": source_manifest.get("scope"),
        "source_validation_months": source_manifest.get("validation_months"),
        "candidate_pool_rows": int(len(pool)),
        "selected_profiles": int(len(selected)),
        "holdout_status_counts": summary["holdout_status"].value_counts().to_dict()
        if not summary.empty and "holdout_status" in summary.columns
        else {},
        "holdout_mean_delta": _safe_mean(summary.get("holdout_repair_delta_mean_u_vs_proxy")),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        candidate_pool=pool,
        selected=selected,
        summary=summary,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-path", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--source-manifest-path", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-month", type=str, default="2026-05")
    parser.add_argument("--holdout-month", type=str, default="2026-06")
    parser.add_argument("--max-profiles", type=int, default=3)
    parser.add_argument("--min-selected-rows", type=int, default=5)
    parser.add_argument("--min-train-class-rows", type=int, default=10)
    parser.add_argument("--min-selection-delta", type=float, default=0.0)
    parser.add_argument("--min-selection-mean-u", type=float, default=0.0)
    parser.add_argument("--min-oracle-capture-delta", type=float, default=0.0)
    parser.add_argument("--max-bad-mae-rate", type=float, default=0.75)
    parser.add_argument("--min-holdout-oracle-capture", type=float, default=0.05)
    parser.add_argument("--max-bad-mae-excess", type=float, default=0.15)
    parser.add_argument("--max-timeout-excess", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        monthly_path=args.monthly_path,
        source_manifest_path=args.source_manifest_path,
        output_dir=args.output_dir,
        selection_month=args.selection_month,
        holdout_month=args.holdout_month,
        max_profiles=args.max_profiles,
        min_selected_rows=args.min_selected_rows,
        min_train_class_rows=args.min_train_class_rows,
        min_selection_delta=args.min_selection_delta,
        min_selection_mean_u=args.min_selection_mean_u,
        min_oracle_capture_delta=args.min_oracle_capture_delta,
        max_bad_mae_rate=args.max_bad_mae_rate,
        min_holdout_oracle_capture=args.min_holdout_oracle_capture,
        max_bad_mae_excess=args.max_bad_mae_excess,
        max_timeout_excess=args.max_timeout_excess,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
