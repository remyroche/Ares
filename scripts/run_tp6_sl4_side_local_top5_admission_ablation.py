#!/usr/bin/env python3
"""Causal side-local top-5% admission-map comparison for frozen meta scores.

For every evaluation day and side, the map is fitted only on resolved earlier
scores in the requested trailing period.  Crucially, only that reference side's
top 5% is used: a map means "expected net EV conditional on our intended
side-local admission policy", not EV for the entire candidate population.

The two sides are merged only after their maps pass the same temporal
comparability gate.  A failed side is explicitly ineligible rather than being
given a pooled/global score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


TOP_FRACTIONS = (.005, .01, .02, .03, .05, .10)
SIDE_TOP_FRACTION = .05
MIN_REFERENCE_TOP5 = 100
MIN_VALIDATION_TOP5 = 50
SHRINKAGE_ROWS = 500


def _score_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    """Side/day relative score; one is best and uses no outcome information."""
    out = frame.copy()
    ordered = out.sort_values(
        ["side_name", "__ts__", "side_calibrated_score_bps", "meta_raw_score", "candidate_id"],
        ascending=[True, True, False, False, True], kind="mergesort",
    )
    rank = ordered.groupby(["side_name", "__ts__"], sort=False).cumcount() + 1
    count = ordered.groupby(["side_name", "__ts__"], sort=False)["candidate_id"].transform("size")
    ordered["side_score_percentile"] = 1. - (rank - 1.) / np.maximum(count - 1., 1.)
    return ordered.sort_index()


def _top5(frame: pd.DataFrame) -> pd.DataFrame:
    n = max(1, int(np.ceil(len(frame) * SIDE_TOP_FRACTION)))
    return frame.sort_values(
        ["side_score_percentile", "meta_raw_score", "candidate_id"],
        ascending=[False, False, True], kind="mergesort",
    ).head(n)


def _fit_map(reference: pd.DataFrame) -> tuple[IsotonicRegression, pd.DataFrame]:
    selected = _top5(reference)
    if len(selected) < MIN_REFERENCE_TOP5 or selected.side_score_percentile.nunique() < 2:
        raise ValueError("insufficient top-5% reference support")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(selected.side_score_percentile, selected.t4_tp6_sl4_net_bps)
    return model, selected


def _gate(reference: pd.DataFrame) -> dict[str, float | bool | str]:
    """Chronological validation of a side's own top-5% map.

    The gate has no global/other-side fallback.  The calibration-error bound is
    a two-standard-error test with a 25-bps numerical floor, so a side is not
    excluded merely because an economically noisy small sample is imprecise.
    """
    ordered_days = np.array(sorted(reference["__ts__"].dt.normalize().unique()))
    if len(ordered_days) < 4:
        return {"passed": False, "reason": "fewer than four reference days"}
    split = max(1, int(np.floor(len(ordered_days) * .75)))
    split = min(split, len(ordered_days) - 1)
    cutoff = pd.Timestamp(ordered_days[split])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    fit = reference[reference["__ts__"].lt(cutoff)]
    validate = reference[reference["__ts__"].ge(cutoff)]
    try:
        model, fit_top = _fit_map(fit)
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}
    validation_top = _top5(validate)
    if len(validation_top) < MIN_VALIDATION_TOP5:
        return {"passed": False, "reason": "insufficient held-out top-5% support"}
    predicted = model.predict(validation_top.side_score_percentile).mean()
    realised = validation_top.t4_tp6_sl4_net_bps.mean()
    se = validation_top.t4_tp6_sl4_net_bps.std(ddof=1) / np.sqrt(len(validation_top))
    tolerance = max(25., 2. * float(se))
    return {
        "passed": bool(abs(predicted - realised) <= tolerance),
        "reason": "passed" if abs(predicted - realised) <= tolerance else "held-out top-5% calibration error exceeds two-SE bound",
        "fit_top5_n": int(len(fit_top)),
        "validation_top5_n": int(len(validation_top)),
        "validation_predicted_net_bps": float(predicted),
        "validation_realised_net_bps": float(realised),
        "validation_error_bps": float(predicted - realised),
        "validation_tolerance_bps": float(tolerance),
    }


def _apply_window(scored: pd.DataFrame, history: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    score = _score_percentile(scored)
    hist = _score_percentile(history)
    output = score.copy()
    output["admission_mapped_mean_net_bps"] = np.nan
    output["admission_eligible"] = False
    audits: list[dict[str, object]] = []
    all_rows = pd.concat([hist.assign(__source__="calibration"), score.assign(__source__="evaluation")], ignore_index=True)
    for day in sorted(score["__ts__"].dt.normalize().unique()):
        asof = pd.Timestamp(day)
        if asof.tzinfo is None:
            asof = asof.tz_localize("UTC")
        current_idx = output.index[output["__ts__"].dt.normalize().eq(asof)]
        # Outcomes may be used only after H12 has resolved, and never from this
        # day.  Prior evaluation rows become available causally on later days.
        reference_all = all_rows[
            all_rows["__ts__"].lt(asof)
            & all_rows["__label_available_at__"].lt(asof)
            & all_rows["__label_available_at__"].ge(asof - pd.Timedelta(days=days))
        ]
        for side in ("long", "short"):
            current = output.loc[current_idx][lambda x: x.side_name.eq(side)]
            reference = reference_all[reference_all.side_name.eq(side)]
            audit: dict[str, object] = {"map_days": days, "asof_day": str(asof), "side_name": side, "reference_rows": int(len(reference)), "current_rows": int(len(current))}
            # A 60-day arm is not a 30-day arm merely because the available
            # ledger happens to be short.  Require resolved observations from
            # the beginning of the requested interval before admitting it.
            if reference.empty or reference["__label_available_at__"].min() > asof - pd.Timedelta(days=days) + pd.Timedelta(hours=24):
                audit.update({"passed": False, "reason": "insufficient full calendar-window warm-up", "reference_top5_n": 0, "shrinkage_weight": 0.})
                audits.append(audit)
                continue
            try:
                model, selected_reference = _fit_map(reference)
                gate = _gate(reference)
                audit.update(gate)
                audit["reference_top5_n"] = int(len(selected_reference))
                audit["reference_top5_realised_net_bps"] = float(selected_reference.t4_tp6_sl4_net_bps.mean())
                if bool(gate["passed"]):
                    # Conservative *within-side* shrinkage: a successful but
                    # small map retains its own level, while its local score
                    # spread is compressed.  No other-side/global outcome is
                    # used to manufacture cross-side comparability.
                    side_mean = float(selected_reference.t4_tp6_sl4_net_bps.mean())
                    shrink = len(selected_reference) / (len(selected_reference) + SHRINKAGE_ROWS)
                    mapped = model.predict(current.side_score_percentile)
                    output.loc[current.index, "admission_mapped_mean_net_bps"] = shrink * mapped + (1. - shrink) * side_mean
                    output.loc[current.index, "admission_eligible"] = True
                    audit["shrinkage_weight"] = float(shrink)
                else:
                    audit["shrinkage_weight"] = 0.
            except ValueError as exc:
                audit.update({"passed": False, "reason": str(exc), "reference_top5_n": 0, "shrinkage_weight": 0.})
            audits.append(audit)
    audit_frame = pd.DataFrame(audits)
    # Explicit cross-side comparability gate.  Independent maps may each be
    # locally calibrated yet disagree in their *relative* error, which would
    # let the noisier side win the global ranking merely through scale.  When
    # both sides are available, require their held-out biases to agree within
    # the combined two-SE tolerances.  A lone valid side is still tradable: it
    # is not being numerically compared with an unvalidated other side.
    audit_frame["side_comparability_passed"] = False
    audit_frame["side_comparability_reason"] = "side map failed local validation"
    for day, positions in audit_frame.groupby("asof_day", sort=False).groups.items():
        ix = list(positions)
        local = audit_frame.loc[ix].set_index("side_name")
        passed = local[local.passed.fillna(False)]
        if len(passed) == 1:
            side = passed.index[0]
            pos = audit_frame.index[(audit_frame.asof_day.eq(day)) & (audit_frame.side_name.eq(side))]
            audit_frame.loc[pos, ["side_comparability_passed", "side_comparability_reason"]] = [True, "only one locally validated side"]
            continue
        if len(passed) != 2:
            continue
        errors = passed.validation_error_bps.to_numpy(float)
        tolerances = passed.validation_tolerance_bps.to_numpy(float)
        comparable = bool(abs(errors[0] - errors[1]) <= np.hypot(tolerances[0], tolerances[1]))
        reason = "paired held-out calibration biases agree" if comparable else "paired held-out calibration biases disagree"
        audit_frame.loc[ix, ["side_comparability_passed", "side_comparability_reason"]] = [comparable, reason]
        if not comparable:
            stamp = pd.Timestamp(day)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            output.loc[output["__ts__"].dt.normalize().eq(stamp), ["admission_eligible", "admission_mapped_mean_net_bps"]] = [False, np.nan]
    return output, audit_frame


def _metrics(scored: pd.DataFrame) -> list[dict[str, object]]:
    eligible = scored[scored.admission_eligible].sort_values(
        ["admission_mapped_mean_net_bps", "meta_raw_score", "candidate_id"], ascending=[False, False, True], kind="mergesort"
    )
    rows: list[dict[str, object]] = []
    for fraction in TOP_FRACTIONS:
        # The trading policy remains global top-k over the complete candidate
        # population.  The gate may leave fewer eligible rows than k; that is
        # an explicit no-trade decision, never a hidden change in allocation.
        requested = int(np.ceil(len(scored) * fraction))
        selected = eligible.head(requested)
        for view, values in (("global_after_side_gate", selected), ("long", selected[selected.side_name.eq("long")]), ("short", selected[selected.side_name.eq("short")])):
            rows.append({"allocation": view, "top_fraction": fraction, "requested_n": requested, "n": int(len(values)), "fill_rate": float(len(selected) / requested) if requested else np.nan, "gross_bps": float(values.t4_tp6_sl4_gross_bps.mean()) if len(values) else np.nan, "net_bps": float(values.t4_tp6_sl4_net_bps.mean()) if len(values) else np.nan})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--calibration-history", type=Path, nargs="+", required=True, help="one or more strictly earlier score ledgers (calibration or prior evaluation)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--periods", type=int, nargs="+", default=[14, 21, 30, 60])
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    scored = pd.read_parquet(a.predictions)
    history = pd.concat([pd.read_parquet(path) for path in a.calibration_history], ignore_index=True)
    required = {"candidate_id", "__ts__", "__label_available_at__", "side_name", "meta_raw_score", "side_calibrated_score_bps", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_gross_bps"}
    for name, frame in (("predictions", scored), ("calibration history", history)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} lacks {sorted(missing)}")
        for field in ("__ts__", "__label_available_at__"):
            frame[field] = pd.to_datetime(frame[field], utc=True)
    # A score can appear in an older calibration ledger and in its subsequent
    # final-fit evaluation ledger.  Retain the latter only when both describe
    # the same candidate, rather than duplicating an outcome in the map.
    history = history.sort_values(["candidate_id", "__ts__"], kind="mergesort").drop_duplicates(["candidate_id", "__ts__"], keep="last")
    a.out.mkdir(parents=True)
    manifest: dict[str, object] = {"schema": "tp6_sl4_side_local_top5_admission_v1", "method": "each side isotonic is fit only on its resolved trailing top-5% score population; side maps are merged only after the temporal comparability gate", "periods": a.periods, "results": {}}
    for days in a.periods:
        mapped, audit = _apply_window(scored, history, days)
        metrics = _metrics(mapped)
        mapped.to_parquet(a.out / f"mapped_{days}d.parquet", index=False)
        audit.to_parquet(a.out / f"gate_audit_{days}d.parquet", index=False)
        pd.DataFrame(metrics).to_parquet(a.out / f"metrics_{days}d.parquet", index=False)
        manifest["results"][str(days)] = {"metrics": metrics, "local_gate_side_days_passed": int(audit.passed.fillna(False).sum()), "side_comparability_side_days_passed": int(audit.side_comparability_passed.fillna(False).sum()), "gate_side_days_total": int(len(audit)), "eligible_rows": int(mapped.admission_eligible.sum()), "first_eligible_day": str(mapped.loc[mapped.admission_eligible, "__ts__"].min()) if mapped.admission_eligible.any() else None}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
