#!/usr/bin/env python3
"""Screen additive V2 market-state fields for *Base reliability*, not alpha.

The current P8U route, F72 Base score, Under F120 score, dual MC1 maps, and
portfolio auction are immutable parents.  This utility asks the narrower
question: conditional on a Base-tail score band, do frozen target-free market
transitions identify unusually large Base-to-policy conversion error?

It deliberately gives each timestamp unit mass within each Base-tail segment,
then applies target-free frozen-episode balancing.  This prevents a busy hour
or a large candidate cross-section from becoming pseudo-independent evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_state_reliability_v2_screen"
SELECTION_END = pd.Timestamp("2026-01-01", tz="UTC")
TAILS: dict[str, tuple[float, float]] = {
    "base_top2": (0.98, 1.000001),
    "base_top5": (0.95, 1.000001),
    "base_5to10": (0.90, 0.95),
    "base_top10": (0.90, 1.000001),
}
CONTROL_LEVELS = (
    "volatility_level", "execution_spread", "liquidity_depth",
    "funding_dispersion", "correlation",
)
CONTROL_DIRECT = (
    "return_iqr", "breadth", "volatility_level", "execution_spread",
    "funding_dispersion", "correlation",
)
GEOMETRY_SUFFIXES = (
    "rms", "mahalanobis", "abs1_breadth", "abs2_breadth",
    "positive_breadth", "negative_breadth", "sign_coherence", "iqr",
    "mad", "max_abs", "top3_abs_mean",
)
EPISODE_FIELDS = (
    "v2_regime_distance", "v2_regime_second_distance",
    "v2_regime_assignment_margin", "v2_regime_transition_flag",
    "v2_time_since_regime_change_hours",
)
THRESHOLDS = (0.60, 0.70, 0.80, 0.90)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member.relative_to(ROOT)).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _feature_contract(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    fields: list[str] = []
    # 46 deviation fields: innovation and fast-vs-slow transition for every
    # semantic state.  These form the V2 primary family.
    fields.extend(sorted(name for name in available if name.startswith("v2_innovation_z__")))
    fields.extend(sorted(name for name in available if name.startswith("v2_transition_z__")))
    # Eight uncertainty controls, 22 generic geometry features, eight explicit
    # contrasts, five episode diagnostics, and 11 pure-level/direct controls.
    fields.extend(f"v2_uncertainty__{name}" for name in (*CONTROL_LEVELS, "return_iqr", "breadth", "oi_effective_rank"))
    fields.extend(f"v2_{kind}_{suffix}" for kind in ("innovation", "transition") for suffix in GEOMETRY_SUFFIXES)
    fields.extend(sorted(name for name in available if name.startswith("v2_contrast__")))
    fields.extend(EPISODE_FIELDS)
    fields.extend(f"v2_level_control__{name}" for name in CONTROL_LEVELS)
    fields.extend(f"v2_direct_delta_control__{name}" for name in CONTROL_DIRECT)
    # Preserve order, reject an accidental contract widening or missing field.
    result = list(dict.fromkeys(fields))
    missing = [name for name in result if name not in available]
    if missing:
        raise AssertionError(f"missing V2 screen fields: {missing}")
    if len(result) != 100:
        raise AssertionError(f"expected a 100-field deviation-first contract, got {len(result)}")
    return result


def _weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if int(valid.sum()) < 20 or np.unique(x[valid]).size < 4:
        return np.nan
    xx, yy, ww = rankdata(x[valid]), rankdata(y[valid]), w[valid]
    ww = ww / ww.sum()
    mx, my = float(np.dot(ww, xx)), float(np.dot(ww, yy))
    vx, vy = float(np.dot(ww, (xx - mx) ** 2)), float(np.dot(ww, (yy - my) ** 2))
    if vx <= 0.0 or vy <= 0.0:
        return np.nan
    return float(np.dot(ww, (xx - mx) * (yy - my)) / np.sqrt(vx * vy))


def _weighted_cmi(x: np.ndarray, y: np.ndarray, condition: np.ndarray, w: np.ndarray, *, xbins: int = 8, ybins: int = 6) -> float:
    """Weighted binned I(X;Y|Base fine-band), calculated without repetition."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if int(valid.sum()) < 100 or np.unique(x[valid]).size < 4:
        return np.nan
    x, y, condition, w = x[valid], y[valid], condition[valid], w[valid]
    score = 0.0
    mass = 0.0
    for token in np.unique(condition):
        mask = condition == token
        if int(mask.sum()) < 20:
            continue
        xx, yy, ww = x[mask], y[mask], w[mask]
        xr = rankdata(xx) / (len(xx) + 1.0)
        if np.unique(yy).size <= 2:
            yb = yy.astype(np.int16)
            ny = int(yb.max()) + 1
        else:
            yr = rankdata(yy) / (len(yy) + 1.0)
            yb = np.minimum(ybins - 1, np.floor(yr * ybins)).astype(np.int16)
            ny = ybins
        xb = np.minimum(xbins - 1, np.floor(xr * xbins)).astype(np.int16)
        table = np.zeros((xbins, ny), dtype=float)
        np.add.at(table, (xb, yb), ww)
        local_mass = float(table.sum())
        if local_mass <= 0.0:
            continue
        joint = table / local_mass
        px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
        nonzero = joint > 0.0
        value = float((joint[nonzero] * np.log(joint[nonzero] / (px @ py)[nonzero])).sum())
        score += value * local_mass
        mass += local_mass
    return score / mass if mass else np.nan


def _weights(frame: pd.DataFrame, regime_reference: pd.Series) -> pd.DataFrame:
    output = frame.copy()
    frequency = regime_reference.value_counts(dropna=False).astype(float)
    median = float(frequency.median()) if len(frequency) else 1.0
    regime_weight = np.sqrt(median / output.v2_regime_id.map(frequency).fillna(median).clip(lower=1.0))
    output["v2_regime_weight"] = regime_weight.clip(.5, 2.0).astype(np.float32)
    for tail, (lower, upper) in TAILS.items():
        mask = output.base_rank_ts.ge(lower) & output.base_rank_ts.lt(upper)
        count = output.loc[mask].groupby("__decision_ts__")["candidate_id"].transform("size")
        output.loc[mask, f"v2_weight__{tail}"] = (output.loc[mask, "v2_regime_weight"] / count.clip(lower=1)).astype(np.float32)
    return output


def _fine_band(rank: np.ndarray) -> np.ndarray:
    return np.minimum(9, np.maximum(0, np.floor((rank - .90) / .01))).astype(np.int8)


def _historical_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach month-level large-error cutoffs fitted only to prior resolved rows."""
    output = frame.copy()
    output["month_start"] = output.__decision_ts__.dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    values = output.loc[:, ["month_start", "available", "base_abs_residual_bps"]].copy()
    for q in THRESHOLDS:
        column = f"large_error_q{int(q * 100):02d}"
        output[column] = np.nan
        for month in sorted(output.month_start.unique()):
            prior = values.loc[(values.month_start < month) & (values.available < month), "base_abs_residual_bps"]
            if len(prior) >= 1_000:
                output.loc[output.month_start.eq(month), column] = float(prior.quantile(q))
        output[f"target_{column}"] = output.base_abs_residual_bps.gt(output[column]).astype("float32")
        output.loc[output[column].isna(), f"target_{column}"] = np.nan
    return output


def _metric_row(part: pd.DataFrame, feature: str, tail: str, *, period: str, token: str) -> dict[str, object]:
    w = part[f"v2_weight__{tail}"].to_numpy(float)
    x = pd.to_numeric(part[feature], errors="coerce").to_numpy(float)
    abs_y = part.base_abs_residual_bps.to_numpy(float)
    signed_y = part.residual_bps.to_numpy(float)
    condition = part.v2_base_fine_band.to_numpy(np.int8)
    row: dict[str, object] = {
        "period_kind": period, "period": token, "tail": tail, "feature": feature,
        "timestamps": int(part.__decision_ts__.nunique()), "rows": int(len(part)),
        "effective_weight": float(np.nansum(w)),
        "weighted_spearman_abs_residual": _weighted_spearman(x, abs_y, w),
        "weighted_spearman_signed_residual": _weighted_spearman(x, signed_y, w),
        "weighted_cmi_abs_residual": _weighted_cmi(x, abs_y, condition, w),
    }
    for q in THRESHOLDS:
        target = part[f"target_large_error_q{int(q * 100):02d}"].to_numpy(float)
        row[f"weighted_cmi_large_error_q{int(q * 100):02d}"] = _weighted_cmi(x, target, condition, w, ybins=2)
    return row


def _evaluate(frame: pd.DataFrame, fields: list[str], *, period: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for token, source in frame.groupby(period, sort=True):
        for tail, (lower, upper) in TAILS.items():
            part = source.loc[source.base_rank_ts.ge(lower) & source.base_rank_ts.lt(upper)]
            if part.__decision_ts__.nunique() < 100:
                continue
            for feature in fields:
                rows.append(_metric_row(part, feature, tail, period=period, token=str(token)))
    return pd.DataFrame(rows)


def _macro_regime_metrics(frame: pd.DataFrame, fields: list[str], *, era: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tail, (lower, upper) in TAILS.items():
        tail_frame = frame.loc[frame.base_rank_ts.ge(lower) & frame.base_rank_ts.lt(upper)]
        for feature in fields:
            values: list[float] = []
            supports: list[int] = []
            for _, part in tail_frame.groupby("v2_regime_id", sort=True):
                if part.__decision_ts__.nunique() < 100 or part.__decision_ts__.dt.to_period("M").nunique() < 3:
                    continue
                value = _weighted_cmi(
                    pd.to_numeric(part[feature], errors="coerce").to_numpy(float),
                    part.base_abs_residual_bps.to_numpy(float), part.v2_base_fine_band.to_numpy(np.int8),
                    part[f"v2_weight__{tail}"].to_numpy(float),
                )
                if np.isfinite(value):
                    values.append(float(value)); supports.append(int(part.__decision_ts__.nunique()))
            rows.append({"era": era, "tail": tail, "feature": feature, "macro_regime_cmi": float(np.mean(values)) if values else np.nan, "supported_regimes": len(values), "min_regime_timestamps": min(supports) if supports else 0})
    return pd.DataFrame(rows)


def _summary(monthly: pd.DataFrame, macro: pd.DataFrame, *, era: str) -> pd.DataFrame:
    abs_columns = ["weighted_cmi_abs_residual", *[f"weighted_cmi_large_error_q{int(q * 100):02d}" for q in THRESHOLDS]]
    grouped = monthly.groupby(["tail", "feature"], sort=True).agg(
        observed_months=("period", "nunique"),
        mean_abs_residual_ic=("weighted_spearman_abs_residual", "mean"),
        abs_residual_ic_sign_months=("weighted_spearman_abs_residual", lambda x: int((x > 0).sum())),
        mean_signed_residual_ic=("weighted_spearman_signed_residual", "mean"),
        min_abs_residual_ic=("weighted_spearman_abs_residual", "min"),
        mean_cmi_abs=("weighted_cmi_abs_residual", "mean"),
    ).reset_index()
    for column in abs_columns[1:]:
        extra = monthly.groupby(["tail", "feature"], sort=True)[column].mean().rename(f"mean_{column}").reset_index()
        grouped = grouped.merge(extra, on=["tail", "feature"], how="left", validate="one_to_one")
    grouped["abs_residual_ic_consistency"] = grouped.abs_residual_ic_sign_months / grouped.observed_months.clip(lower=1)
    grouped = grouped.merge(macro, on=["tail", "feature"], how="left", validate="one_to_one")
    grouped["era"] = era
    # Selection is intentionally conservative but permits conditional fields
    # that are recurrent rather than universally active.
    grouped["eligible"] = (
        grouped.observed_months.ge(6)
        & grouped.abs_residual_ic_consistency.ge(.60)
        & grouped.mean_cmi_abs.gt(0.0)
        & grouped.supported_regimes.ge(2)
    )
    cmi_scale = grouped.mean_cmi_abs.median() or 1e-9
    ic_scale = grouped.mean_abs_residual_ic.abs().median() or 1e-9
    macro_scale = grouped.macro_regime_cmi.median() or 1e-9
    grouped["selection_score"] = (
        grouped.mean_cmi_abs / cmi_scale
        + grouped.mean_abs_residual_ic / ic_scale
        + grouped.macro_regime_cmi.fillna(0.0) / macro_scale
    ) * grouped.abs_residual_ic_consistency
    return grouped.sort_values(["tail", "eligible", "selection_score", "feature"], ascending=[True, False, False, True], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state_root, out = ROOT / args.state_root, ROOT / args.out
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads((state_root / "correctness_report.json").read_text())
    if not all(value is True or key == "schema" for key, value in receipt.items()):
        raise AssertionError("state/reliability substrate receipt is not clean")
    state = pd.read_parquet(state_root / "target_free_state_episode_hourly.parquet")
    events = pd.read_parquet(state_root / "labelled_base_top10_residual_events.parquet")
    state["__decision_ts__"] = pd.to_datetime(state["__decision_ts__"], utc=True, errors="raise")
    events["__decision_ts__"] = pd.to_datetime(events["__decision_ts__"], utc=True, errors="raise")
    fields = _feature_contract(state.columns)
    frame = events.merge(state.loc[:, ["__decision_ts__", "v2_regime_id", *fields]], on="__decision_ts__", how="left", validate="many_to_one")
    if len(frame) != len(events) or frame[fields].isna().all(axis=None):
        raise AssertionError("target-free state join failed")
    frame["v2_base_fine_band"] = _fine_band(frame.base_rank_ts.to_numpy(float))
    selection_mask = frame.__decision_ts__.lt(SELECTION_END)
    if int(selection_mask.sum()) < 10_000:
        raise AssertionError("insufficient 2025 selection support")
    frame = _weights(frame, frame.loc[selection_mask, "v2_regime_id"])
    frame = _historical_thresholds(frame)
    frame["month"] = frame.__decision_ts__.dt.strftime("%Y-%m")
    frame["week"] = frame.__decision_ts__.dt.to_period("W-SUN").astype(str)
    selection = frame.loc[selection_mask].copy()
    confirmation = frame.loc[~selection_mask].copy()
    monthly = pd.concat([_evaluate(selection, fields, period="month"), _evaluate(confirmation, fields, period="month")], ignore_index=True)
    weekly = _evaluate(confirmation, fields, period="week")
    macro_selection = _macro_regime_metrics(selection, fields, era="selection_2025")
    macro_confirmation = _macro_regime_metrics(confirmation, fields, era="confirmation_2026")
    summary_selection = _summary(monthly.loc[monthly.period.str[:4].eq("2025")], macro_selection, era="selection_2025")
    summary_confirmation = _summary(monthly.loc[monthly.period.str[:4].eq("2026")], macro_confirmation, era="confirmation_2026")
    selected = summary_selection.loc[summary_selection.eligible].copy()
    out.mkdir(parents=True)
    monthly.to_parquet(out / "monthly_weighted_reliability_screen.parquet", index=False)
    weekly.to_parquet(out / "weekly_weighted_reliability_screen_2026.parquet", index=False)
    macro_selection.to_parquet(out / "macro_regime_cmi_selection_2025.parquet", index=False)
    macro_confirmation.to_parquet(out / "macro_regime_cmi_confirmation_2026.parquet", index=False)
    summary_selection.to_parquet(out / "feature_summary_selection_2025.parquet", index=False)
    summary_confirmation.to_parquet(out / "feature_summary_confirmation_2026.parquet", index=False)
    selected.to_parquet(out / "selected_reliability_candidates_2025.parquet", index=False)
    _once(out / "feature_contract.json", {"schema": SCHEMA, "features": fields, "count": len(fields), "design": "deviation-first 100 fields; level/direct controls only"})
    correctness = {
        "parent_p8u_stack_unchanged": True,
        "new_fields_are_additive_reliability_inputs_only": True,
        "state_features_target_free": True,
        "episodes_frozen_before_labels": True,
        "base_residuals_strict_prequential": True,
        "large_error_thresholds_fitted_on_prior_resolved_rows_only": True,
        "regime_weights_use_target_free_episode_ids": True,
        "candidate_weight_is_timestamp_normalised": True,
        "no_meta_mc1_admission_portfolio_or_live_mutation": True,
    }
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {"schema": SCHEMA, "scope": "offline additive Base-reliability screen", "state_root": str(state_root.relative_to(ROOT)), "state_root_sha256": _sha(state_root), "selection_end": str(SELECTION_END), "features": len(fields), "correctness": correctness})
    print(json.dumps({"out": str(out), "events": len(frame), "fields": len(fields), "selection_candidates": int(len(selected))}, sort_keys=True))


if __name__ == "__main__":
    main()
