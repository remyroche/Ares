#!/usr/bin/env python3
"""Post-score diagnostic for causal F72 SHAP top-ten residual-state inputs.

This opens policy outcomes only after verifying that both the frozen SHAP
ledger and the top-ten state panels are target-free.  It reports whether each
recent same-band state is related to the subsequently realised strict-
prequential Base residual.  It never fits or alters a Meta, MC1, portfolio,
or live component.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import materialize_strict_r3_p8u_meta_base_state_v1 as base_state
import run_strict_r3_p8u_meta_target_query_grid_v1 as meta


IDENTITY = tuple(meta.IDENTITY)
SCHEMA = "strict_r3_p8u_f72_shap_top10_residual_state_audit_v1"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or tuple(sorted(result)) != result:
        raise ValueError("months must be chronological unique YYYY-MM values")
    return result


def _end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _target_free(frame: pd.DataFrame) -> None:
    forbidden = {"policy_net_bps", "policy_path_valid", "policy_label_available_ts", "residual_bps"}
    leaked = forbidden.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"target-free input leaks {sorted(leaked)}")


def _conditional_ic(frame: pd.DataFrame, feature: str) -> tuple[float, float, int]:
    ics: list[float] = []
    effects: list[float] = []
    for _band, part in frame.groupby("band", sort=False):
        part = part.loc[np.isfinite(part[feature]) & np.isfinite(part.residual_bps)]
        if len(part) < 100 or part[feature].nunique() < 5 or part.residual_bps.nunique() < 5:
            continue
        value = float(spearmanr(part[feature], part.residual_bps).statistic)
        if np.isfinite(value):
            ics.append(value)
        low, high = part[feature].quantile(.25), part[feature].quantile(.75)
        lower = part.loc[part[feature].le(low), "residual_bps"]
        upper = part.loc[part[feature].ge(high), "residual_bps"]
        if len(lower) and len(upper):
            effects.append(float(upper.mean() - lower.mean()))
    return (
        float(np.mean(ics)) if ics else float("nan"),
        float(np.mean(effects)) if effects else float("nan"),
        len(ics),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--shap-root", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--months", default="2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    months = _months(args.months)
    manifest = json.loads((args.state_root / "run_manifest.json").read_text())
    if manifest.get("schema") != "strict_r3_p8u_f72_shap_top10_residual_state_v1":
        raise AssertionError("wrong top-ten residual-state source")
    history_months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in manifest["months"])
    parts: list[pd.DataFrame] = []
    state_parts: list[pd.DataFrame] = []
    for month in history_months:
        shap_path = args.shap_root / f"month={month:%Y-%m}.parquet"
        shap = pd.read_parquet(shap_path, columns=[*IDENTITY, "base_score", "base_rank_ts"])
        shap["__decision_ts__"] = pd.to_datetime(shap["__decision_ts__"], utc=True, errors="raise")
        _target_free(shap)
        parts.append(shap)
        if month in months:
            state = pd.read_parquet(args.state_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet")
            state["__decision_ts__"] = pd.to_datetime(state["__decision_ts__"], utc=True, errors="raise")
            _target_free(state)
            state_parts.append(state.loc[:, list(IDENTITY) + [column for column in state if column.startswith("shap_top")]])
    history = pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    policy = meta._read_policy(args.policy_labels)
    events = base_state._policy_events(history, policy)
    residual = events.loc[:, ["candidate_id", "residual_bps", "band", "available"]]
    rows: list[dict[str, object]] = []
    for month, state in zip(months, state_parts):
        frame = state.merge(residual, on="candidate_id", how="left", validate="one_to_one")
        valid = frame.residual_bps.notna()
        for feature in [column for column in state if column.startswith("shap_top")]:
            ic, effect, used_bands = _conditional_ic(frame.loc[valid].copy(), feature)
            rows.append({
                "month": f"{month:%Y-%m}", "feature": feature, "rows": int(len(frame)),
                "valid_residual_rows": int(valid.sum()), "feature_coverage": float(frame[feature].notna().mean()),
                "conditional_residual_ic": ic, "same_band_upper_minus_lower_residual_bps": effect,
                "base_bands_used": used_bands,
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("feature", sort=True).agg(
        months=("month", "nunique"), coverage_min=("feature_coverage", "min"),
        residual_ic_mean=("conditional_residual_ic", "mean"), residual_ic_positive_months=("conditional_residual_ic", lambda x: int((x > 0).sum())),
        upper_minus_lower_mean_bps=("same_band_upper_minus_lower_residual_bps", "mean"),
        positive_effect_months=("same_band_upper_minus_lower_residual_bps", lambda x: int((x > 0).sum())),
    ).reset_index()
    top = pd.read_parquet(args.state_root / "timestamp_top10_contributor_summary.parquet")
    top["month"] = pd.to_datetime(top.__decision_ts__, utc=True).dt.strftime("%Y-%m")
    frequency = top.groupby(["month", "contributor_feature"], sort=True).agg(
        top10_hours=("contributor_rank", "size"), top1_hours=("contributor_rank", lambda x: int((x == 1).sum())),
        mean_abs_mass=("aggregate_abs_shap_mass", "mean"), mean_candidate_fraction=("candidate_fraction_top10", "mean"),
    ).reset_index()
    args.out.mkdir(parents=True)
    detail.to_parquet(args.out / "state_residual_diagnostics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "state_residual_summary.parquet", index=False, compression="zstd")
    frequency.to_parquet(args.out / "timestamp_top10_contributor_frequency.parquet", index=False, compression="zstd")
    _once(args.out / "correctness_report.json", {
        "source_state_panels_target_free_before_outcome_join": True,
        "source_shap_panels_target_free_before_outcome_join": True,
        "residual_target_is_strict_prequential_base_anchor": True,
        "conditional_analysis_uses_same_base_rank_band": True,
        "diagnostic_only_no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline post-score diagnostic only",
        "state_root": str(args.state_root.resolve()), "shap_root": str(args.shap_root.resolve()),
        "policy_labels": str(args.policy_labels.resolve()), "months": [f"{month:%Y-%m}" for month in months],
    })


if __name__ == "__main__":
    main()
