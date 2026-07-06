#!/usr/bin/env python3
"""Validate pre-registered strict-OOS repair-ranker profiles.

This script is diagnostic-only. It does not select profiles, fit repair models,
or integrate anything into training. It reads a frozen profile manifest and a
month-forward repair-ranker monthly ledger, then applies hard promotion guards
to the specified validation periods.
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

from scripts.run_label_quality_proxy_diagnostics import _json_safe, _safe_mean, _safe_quantile  # noqa: E402


DEFAULT_PROFILE_MANIFEST = Path("configs/strict_oos_repair_ranker_frozen_profiles.json")
DEFAULT_MONTHLY = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_ablation/strict_oos_repair_ranker_monthly.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_frozen_validation"
)

PROFILE_KEYS = ["source_bucket", "proxy_col", "top_frac", "feature_mode", "selection_method"]
MONTHLY_NUMERIC_COLS = [
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
MONTHLY_EVAL_COLS = [
    "profile_name",
    "period",
    *PROFILE_KEYS,
    "selected_rows",
    "repair_mean_u",
    "proxy_mean_u",
    "repair_delta_mean_u_vs_proxy",
    "repair_oracle_capture_at_k",
    "proxy_oracle_capture_at_k",
    "repair_delta_oracle_capture_at_k",
    "repair_bad_mae_1r_rate",
    "proxy_bad_mae_1r_rate",
    "bad_mae_excess",
    "repair_timeout_or_slow_holding_rate",
    "proxy_timeout_or_slow_holding_rate",
    "timeout_excess",
    "period_status",
    "period_failure_reasons",
]

DEFAULT_GUARDS = {
    "min_months": 1,
    "min_selected_rows": 5,
    "require_positive_each_period": True,
    "require_delta_positive_each_period": True,
    "min_mean_repair_u": 0.0,
    "min_worst_month_repair_u": 0.0,
    "min_mean_delta_u_vs_proxy": 0.0,
    "min_worst_month_delta_u_vs_proxy": 0.0,
    "min_oracle_capture": 0.05,
    "min_oracle_capture_delta": 0.0,
    "max_bad_mae_excess": 0.15,
    "max_timeout_excess": 0.15,
    "max_repair_bad_mae_rate": math.inf,
    "max_repair_timeout_or_slow_holding_rate": math.inf,
}


def _parse_csv(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def load_profile_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"{path} must contain a non-empty profiles list")
    for idx, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ValueError(f"profile #{idx + 1} must be an object")
        missing = sorted(set(PROFILE_KEYS).difference(profile))
        if missing:
            raise ValueError(f"profile {profile.get('name', idx + 1)} is missing keys: {missing}")
        profile["top_frac"] = round(float(profile["top_frac"]), 6)
        profile.setdefault("name", "|".join(str(profile[key]) for key in PROFILE_KEYS))
    guards = {**DEFAULT_GUARDS, **data.get("guards", {})}
    data["guards"] = guards
    data["validation_periods"] = [str(v) for v in data.get("validation_periods", [])]
    data["non_promotion_periods"] = [str(v) for v in data.get("non_promotion_periods", [])]
    return data


def load_monthly_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(["period", *PROFILE_KEYS]).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["period"] = frame["period"].astype(str)
    frame["top_frac"] = _safe_numeric(frame["top_frac"]).round(6)
    for col in MONTHLY_NUMERIC_COLS:
        if col in frame.columns:
            frame[col] = _safe_numeric(frame[col])
    return frame


def _profile_mask(frame: pd.DataFrame, profile: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for key in PROFILE_KEYS:
        if key == "top_frac":
            mask &= _safe_numeric(frame[key]).round(6).eq(round(float(profile[key]), 6))
        else:
            mask &= frame[key].astype(str).eq(str(profile[key]))
    return mask


def _period_failure_reasons(row: pd.Series, guards: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    selected_rows = float(row.get("selected_rows", float("nan")))
    repair_mean = float(row.get("repair_mean_u", float("nan")))
    delta = float(row.get("repair_delta_mean_u_vs_proxy", float("nan")))
    oracle_capture = float(row.get("repair_oracle_capture_at_k", float("nan")))
    oracle_delta = float(row.get("repair_delta_oracle_capture_at_k", float("nan")))
    repair_bad = float(row.get("repair_bad_mae_1r_rate", float("nan")))
    proxy_bad = float(row.get("proxy_bad_mae_1r_rate", float("nan")))
    repair_timeout = float(row.get("repair_timeout_or_slow_holding_rate", float("nan")))
    proxy_timeout = float(row.get("proxy_timeout_or_slow_holding_rate", float("nan")))
    bad_excess = repair_bad - proxy_bad if _finite(repair_bad) and _finite(proxy_bad) else float("nan")
    timeout_excess = repair_timeout - proxy_timeout if _finite(repair_timeout) and _finite(proxy_timeout) else float("nan")

    if not (_finite(selected_rows) and selected_rows >= float(guards["min_selected_rows"])):
        reasons.append("selected_rows_below_min")
    if bool(guards["require_positive_each_period"]) and not (_finite(repair_mean) and repair_mean > 0.0):
        reasons.append("non_positive_repair_mean")
    if bool(guards["require_delta_positive_each_period"]) and not (_finite(delta) and delta > 0.0):
        reasons.append("does_not_beat_proxy")
    if not (_finite(oracle_capture) and oracle_capture >= float(guards["min_oracle_capture"])):
        reasons.append("insufficient_oracle_capture")
    if not (_finite(oracle_delta) and oracle_delta >= float(guards["min_oracle_capture_delta"])):
        reasons.append("loses_oracle_capture_vs_proxy")
    if _finite(bad_excess) and bad_excess > float(guards["max_bad_mae_excess"]):
        reasons.append("bad_mae_excess")
    if _finite(timeout_excess) and timeout_excess > float(guards["max_timeout_excess"]):
        reasons.append("timeout_excess")
    if _finite(repair_bad) and repair_bad > float(guards["max_repair_bad_mae_rate"]):
        reasons.append("repair_bad_mae_rate_too_high")
    if _finite(repair_timeout) and repair_timeout > float(guards["max_repair_timeout_or_slow_holding_rate"]):
        reasons.append("repair_timeout_rate_too_high")
    return reasons


def evaluate_profile(
    monthly: pd.DataFrame,
    profile: dict[str, Any],
    *,
    validation_periods: list[str],
    guards: dict[str, Any],
    non_promotion_periods: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    non_promotion_periods = non_promotion_periods or set()
    expected_periods = [str(period) for period in validation_periods]
    profile_rows = monthly.loc[_profile_mask(monthly, profile)].copy()
    profile_rows = profile_rows[profile_rows["period"].isin(expected_periods)].copy()
    profile_rows = profile_rows.sort_values("period", kind="mergesort").drop_duplicates(
        ["period", *PROFILE_KEYS], keep="first"
    )
    observed_periods = profile_rows["period"].astype(str).tolist()
    missing_periods = sorted(set(expected_periods).difference(observed_periods))

    monthly_rows: list[dict[str, Any]] = []
    for _, row in profile_rows.iterrows():
        bad_excess = row.get("repair_bad_mae_1r_rate", np.nan) - row.get("proxy_bad_mae_1r_rate", np.nan)
        timeout_excess = row.get("repair_timeout_or_slow_holding_rate", np.nan) - row.get(
            "proxy_timeout_or_slow_holding_rate", np.nan
        )
        reasons = _period_failure_reasons(row, guards)
        monthly_rows.append(
            {
                "profile_name": profile["name"],
                "period": row["period"],
                **{key: row[key] for key in PROFILE_KEYS},
                "selected_rows": row.get("selected_rows", np.nan),
                "repair_mean_u": row.get("repair_mean_u", np.nan),
                "proxy_mean_u": row.get("proxy_mean_u", np.nan),
                "repair_delta_mean_u_vs_proxy": row.get("repair_delta_mean_u_vs_proxy", np.nan),
                "repair_oracle_capture_at_k": row.get("repair_oracle_capture_at_k", np.nan),
                "proxy_oracle_capture_at_k": row.get("proxy_oracle_capture_at_k", np.nan),
                "repair_delta_oracle_capture_at_k": row.get("repair_delta_oracle_capture_at_k", np.nan),
                "repair_bad_mae_1r_rate": row.get("repair_bad_mae_1r_rate", np.nan),
                "proxy_bad_mae_1r_rate": row.get("proxy_bad_mae_1r_rate", np.nan),
                "bad_mae_excess": bad_excess,
                "repair_timeout_or_slow_holding_rate": row.get("repair_timeout_or_slow_holding_rate", np.nan),
                "proxy_timeout_or_slow_holding_rate": row.get("proxy_timeout_or_slow_holding_rate", np.nan),
                "timeout_excess": timeout_excess,
                "period_status": "passes_period_guards" if not reasons else "fails_period_guards",
                "period_failure_reasons": ",".join(reasons),
            }
        )
    monthly_eval = pd.DataFrame(monthly_rows, columns=MONTHLY_EVAL_COLS)

    repair_mean = _safe_numeric(monthly_eval.get("repair_mean_u", pd.Series(dtype=float)))
    delta = _safe_numeric(monthly_eval.get("repair_delta_mean_u_vs_proxy", pd.Series(dtype=float)))
    oracle_capture = _safe_numeric(monthly_eval.get("repair_oracle_capture_at_k", pd.Series(dtype=float)))
    oracle_delta = _safe_numeric(monthly_eval.get("repair_delta_oracle_capture_at_k", pd.Series(dtype=float)))
    bad_excess = _safe_numeric(monthly_eval.get("bad_mae_excess", pd.Series(dtype=float)))
    timeout_excess = _safe_numeric(monthly_eval.get("timeout_excess", pd.Series(dtype=float)))
    repair_bad = _safe_numeric(monthly_eval.get("repair_bad_mae_1r_rate", pd.Series(dtype=float)))
    repair_timeout = _safe_numeric(monthly_eval.get("repair_timeout_or_slow_holding_rate", pd.Series(dtype=float)))
    selected_rows = _safe_numeric(monthly_eval.get("selected_rows", pd.Series(dtype=float)))

    failure_reasons: list[str] = []
    if missing_periods:
        failure_reasons.append("missing_validation_periods")
    if len(monthly_eval) < int(guards["min_months"]):
        failure_reasons.append("insufficient_validation_months")
    if len(monthly_eval) and selected_rows.lt(float(guards["min_selected_rows"])).any():
        failure_reasons.append("selected_rows_below_min")
    if bool(guards["require_positive_each_period"]) and len(monthly_eval) and repair_mean.le(0.0).any():
        failure_reasons.append("non_positive_repair_month")
    if bool(guards["require_delta_positive_each_period"]) and len(monthly_eval) and delta.le(0.0).any():
        failure_reasons.append("non_positive_delta_month")
    if _finite(_safe_mean(repair_mean)) and _safe_mean(repair_mean) < float(guards["min_mean_repair_u"]):
        failure_reasons.append("mean_repair_u_below_min")
    if len(repair_mean.dropna()) and _safe_quantile(repair_mean, 0.0) < float(guards["min_worst_month_repair_u"]):
        failure_reasons.append("worst_month_repair_u_below_min")
    if _finite(_safe_mean(delta)) and _safe_mean(delta) < float(guards["min_mean_delta_u_vs_proxy"]):
        failure_reasons.append("mean_delta_below_min")
    if len(delta.dropna()) and _safe_quantile(delta, 0.0) < float(guards["min_worst_month_delta_u_vs_proxy"]):
        failure_reasons.append("worst_month_delta_below_min")
    if len(oracle_capture.dropna()) and _safe_quantile(oracle_capture, 0.0) < float(guards["min_oracle_capture"]):
        failure_reasons.append("oracle_capture_below_min")
    if len(oracle_delta.dropna()) and _safe_quantile(oracle_delta, 0.0) < float(guards["min_oracle_capture_delta"]):
        failure_reasons.append("oracle_capture_delta_below_min")
    if len(bad_excess.dropna()) and _safe_quantile(bad_excess, 1.0) > float(guards["max_bad_mae_excess"]):
        failure_reasons.append("bad_mae_excess")
    if len(timeout_excess.dropna()) and _safe_quantile(timeout_excess, 1.0) > float(guards["max_timeout_excess"]):
        failure_reasons.append("timeout_excess")
    if len(repair_bad.dropna()) and _safe_quantile(repair_bad, 1.0) > float(guards["max_repair_bad_mae_rate"]):
        failure_reasons.append("repair_bad_mae_rate_too_high")
    if len(repair_timeout.dropna()) and _safe_quantile(repair_timeout, 1.0) > float(
        guards["max_repair_timeout_or_slow_holding_rate"]
    ):
        failure_reasons.append("repair_timeout_rate_too_high")

    retrospective_only = bool(expected_periods) and set(expected_periods).issubset(non_promotion_periods)
    promotion_allowed = not retrospective_only and not failure_reasons
    if failure_reasons:
        validation_status = "fails_frozen_validation"
    elif retrospective_only:
        validation_status = "passes_guards_but_retrospective_only"
    else:
        validation_status = "passes_frozen_validation"

    aggregate = {
        "profile_name": profile["name"],
        **{key: profile[key] for key in PROFILE_KEYS},
        "expected_periods": ",".join(expected_periods),
        "observed_periods": ",".join(observed_periods),
        "missing_periods": ",".join(missing_periods),
        "months_observed": int(len(monthly_eval)),
        "repair_positive_months": int(repair_mean.gt(0.0).sum()),
        "delta_positive_months": int(delta.gt(0.0).sum()),
        "mean_selected_rows": _safe_mean(selected_rows),
        "min_selected_rows": _safe_quantile(selected_rows, 0.0),
        "mean_repair_u": _safe_mean(repair_mean),
        "worst_month_repair_u": _safe_quantile(repair_mean, 0.0),
        "mean_proxy_u": _safe_mean(monthly_eval.get("proxy_mean_u", pd.Series(dtype=float))),
        "mean_delta_u_vs_proxy": _safe_mean(delta),
        "worst_month_delta_u_vs_proxy": _safe_quantile(delta, 0.0),
        "mean_oracle_capture": _safe_mean(oracle_capture),
        "min_oracle_capture": _safe_quantile(oracle_capture, 0.0),
        "mean_oracle_capture_delta": _safe_mean(oracle_delta),
        "min_oracle_capture_delta": _safe_quantile(oracle_delta, 0.0),
        "max_bad_mae_excess": _safe_quantile(bad_excess, 1.0),
        "mean_bad_mae_excess": _safe_mean(bad_excess),
        "max_timeout_excess": _safe_quantile(timeout_excess, 1.0),
        "mean_timeout_excess": _safe_mean(timeout_excess),
        "max_repair_bad_mae_rate": _safe_quantile(repair_bad, 1.0),
        "max_repair_timeout_or_slow_holding_rate": _safe_quantile(repair_timeout, 1.0),
        "retrospective_only": retrospective_only,
        "promotion_allowed": promotion_allowed,
        "validation_status": validation_status,
        "failure_reasons": ",".join(dict.fromkeys(failure_reasons)),
    }
    return monthly_eval, aggregate


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_repair_ranker_frozen_validation_report.md"
    aggregate_cols = [
        "validation_status",
        "profile_name",
        "expected_periods",
        "observed_periods",
        "missing_periods",
        "mean_repair_u",
        "mean_proxy_u",
        "mean_delta_u_vs_proxy",
        "worst_month_repair_u",
        "min_oracle_capture",
        "max_bad_mae_excess",
        "max_timeout_excess",
        "promotion_allowed",
        "failure_reasons",
    ]
    monthly_cols = [
        "profile_name",
        "period",
        "selected_rows",
        "repair_mean_u",
        "proxy_mean_u",
        "repair_delta_mean_u_vs_proxy",
        "repair_oracle_capture_at_k",
        "bad_mae_excess",
        "timeout_excess",
        "period_status",
        "period_failure_reasons",
    ]
    lines = [
        "# Strict OOS Repair Ranker Frozen Validation",
        "",
        "Diagnostic-only validation of pre-registered repair-ranker profiles. This report does not select profiles or fit models.",
        "",
        "## Scope",
        "",
        f"- Profile manifest: `{manifest.get('profile_manifest_path')}`",
        f"- Monthly ledger: `{manifest.get('monthly_path')}`",
        f"- Validation periods: `{', '.join(manifest.get('validation_periods', []))}`",
        f"- Non-promotion periods: `{', '.join(manifest.get('non_promotion_periods', []))}`",
        "",
        "## Aggregate Gate Results",
        "",
        _table(aggregate, aggregate_cols),
        "",
        "## Monthly Readout",
        "",
        _table(monthly, monthly_cols),
        "",
        "## Interpretation",
        "",
        "- `passes_frozen_validation` is required before considering training integration.",
        "- `passes_guards_but_retrospective_only` is still not promotion evidence because the period was already used during discovery.",
        "- Missing validation periods mean the profile is frozen and waiting for a later strict-OOS ledger.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_validation(
    *,
    profile_manifest_path: Path,
    monthly_path: Path,
    output_dir: Path,
    validation_periods: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_manifest = load_profile_manifest(profile_manifest_path)
    monthly_ledger = load_monthly_ledger(monthly_path)
    periods = validation_periods or profile_manifest["validation_periods"]
    periods = [str(period) for period in periods]
    if not periods:
        raise ValueError("No validation periods supplied in manifest or --validation-periods")
    guards = profile_manifest["guards"]
    non_promotion_periods = set(profile_manifest.get("non_promotion_periods", []))

    monthly_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict[str, Any]] = []
    for profile in profile_manifest["profiles"]:
        monthly_eval, aggregate = evaluate_profile(
            monthly_ledger,
            profile,
            validation_periods=periods,
            guards=guards,
            non_promotion_periods=non_promotion_periods,
        )
        monthly_frames.append(monthly_eval)
        aggregate_rows.append(aggregate)

    monthly_out = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    aggregate_out = pd.DataFrame(aggregate_rows)
    paths = {
        "monthly": output_dir / "strict_oos_repair_ranker_frozen_validation_monthly.csv",
        "aggregate": output_dir / "strict_oos_repair_ranker_frozen_validation_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly_out.to_csv(paths["monthly"], index=False)
    aggregate_out.to_csv(paths["aggregate"], index=False)
    manifest = {
        "scope": "strict_oos_repair_ranker_frozen_validation",
        "profile_manifest_path": str(profile_manifest_path),
        "monthly_path": str(monthly_path),
        "output_dir": str(output_dir),
        "validation_periods": periods,
        "non_promotion_periods": sorted(non_promotion_periods),
        "profile_count": int(len(profile_manifest["profiles"])),
        "guard_config": guards,
        "status_counts": aggregate_out["validation_status"].value_counts().to_dict()
        if not aggregate_out.empty and "validation_status" in aggregate_out.columns
        else {},
        "promotion_allowed_count": int(aggregate_out.get("promotion_allowed", pd.Series(dtype=bool)).fillna(False).sum())
        if not aggregate_out.empty
        else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    report = _write_report(output_dir=output_dir, manifest=manifest, aggregate=aggregate_out, monthly=monthly_out)
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, default=DEFAULT_PROFILE_MANIFEST)
    parser.add_argument("--monthly-path", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validation-periods",
        type=str,
        default=None,
        help="Comma-separated override. If omitted, use validation_periods from the profile manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_validation(
        profile_manifest_path=args.profile_manifest,
        monthly_path=args.monthly_path,
        output_dir=args.output_dir,
        validation_periods=_parse_csv(args.validation_periods) or None,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
