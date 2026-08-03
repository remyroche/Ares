#!/usr/bin/env python3
"""Bounded pre-2026 regime-only downside-risk broadcast ablation.

This is deliberately a diagnostic, not a new model.  The sealed regime OOF
failure head is reduced to one causal timestamp-level downside estimate and
broadcast to every candidate in that hour.  Therefore a penalty can only move
hours relative to one another; it cannot change candidate order within an hour.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
CALIBRATOR = ART / "pre2026_hourly_book_risk_calibrator_20260730_v2_r1"
OUT = ART / "pre2026_regime_only_downside_risk_broadcast_20260730_v3"
# Expected downside is already a return-scale probability × severity product.
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
TOP_FRACTION = 0.10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed(root: Path, *names: str) -> None:
    if not root.is_dir() or not all((root / name).is_file() for name in names):
        raise RuntimeError(f"missing sealed input: {root}")
    if sha256(root / "manifest.json") != (root / "manifest.sha256").read_text().split()[0]:
        raise RuntimeError(f"manifest seal mismatch: {root}")


def global_top10(frame: pd.DataFrame, score: str) -> pd.Series:
    """One deterministic pooled global top-k, never timestamp-local."""
    count = max(1, int(np.ceil(len(frame) * TOP_FRACTION)))
    order = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    return frame.candidate_id.isin(order.head(count).candidate_id)


def expected_downside_hourly(predictions: pd.DataFrame) -> pd.DataFrame:
    """Exact opportunity × conditional failure × conditional severity in return units."""
    need = {
        "book_opportunity",
        "book_failure_rate_if_selected",
        "book_downside_severity_if_selected",
    }
    predictions = predictions.loc[predictions.arm.eq("regime")].copy()
    if set(predictions.target.unique()) & need != need:
        raise RuntimeError("missing authoritative expected-downside head")
    pivot = predictions.loc[predictions.target.isin(need)].pivot(
        index=["era", "__ts__"], columns=["kind", "target"], values="prediction"
    )
    required = [(kind, target) for kind in ("score_only", "context") for target in need]
    if pivot[required].isna().any().any():
        raise RuntimeError("authoritative hourly head pairing incomplete")
    out = pivot.reset_index()
    for kind in ("score_only", "context"):
        out[f"{kind}_expected_downside"] = (
            out[(kind, "book_opportunity")].clip(0, 1)
            * out[(kind, "book_failure_rate_if_selected")].clip(0, 1)
            * out[(kind, "book_downside_severity_if_selected")].clip(lower=0)
        )
    out.columns = [x[0] if isinstance(x, tuple) and x[1] == "" else "__".join(x) if isinstance(x, tuple) else x for x in out.columns]
    return out[["era", "__ts__", "score_only_expected_downside", "context_expected_downside"]]


def period_summary(selected: pd.DataFrame, period: str) -> pd.DataFrame:
    key = selected["__ts__"].dt.tz_localize(None).dt.to_period(period).astype(str)
    return (
        selected.assign(period=key)
        .groupby(["era", "period"], as_index=False)
        .agg(selected_rows=("candidate_id", "size"), net_ev=("execution_net_ev_12h", "mean"))
    )


def q(values: pd.Series, fraction: float) -> float:
    return float(values.quantile(fraction)) if len(values) else float("nan")


def evaluate_arm(frame: pd.DataFrame, score: str, basis: str, penalty: float) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    era_rows: list[dict[str, object]] = []
    side_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    for era, era_frame in frame.groupby("era", sort=True):
        chosen = global_top10(era_frame, score)
        chosen_frame = era_frame.loc[chosen].copy()
        rows.append(chosen_frame)
        era_rows.append(
            {
                "basis": basis,
                "lambda": penalty,
                "era": era,
                "candidate_rows": len(era_frame),
                "selected_rows": len(chosen_frame),
                "top10_net_ev": chosen_frame.execution_net_ev_12h.mean(),
            }
        )
        for side, side_frame in chosen_frame.groupby("side_name", sort=True):
            side_rows.append(
                {
                    "basis": basis,
                    "lambda": penalty,
                    "era": era,
                    "side_name": side,
                    "selected_rows": len(side_frame),
                    "top10_net_ev": side_frame.execution_net_ev_12h.mean(),
                }
            )
        for period, code in (("week", "W"), ("month", "M")):
            summary = period_summary(chosen_frame, code)
            for item in summary.to_dict("records"):
                period_rows.append({"basis": basis, "lambda": penalty, "period_kind": period, **item})
    selected = pd.concat(rows, ignore_index=True)
    return selected, {"basis": basis, "lambda": penalty}, pd.DataFrame(era_rows), pd.DataFrame(side_rows), pd.DataFrame(period_rows)


def gate(control: pd.DataFrame, selected: pd.DataFrame, eras: pd.DataFrame, sides: pd.DataFrame, periods: pd.DataFrame, basis: str, penalty: float) -> dict[str, object]:
    control_net = control.execution_net_ev_12h.mean()
    aggregate_net = selected.execution_net_ev_12h.mean()
    control_era = control.groupby("era").execution_net_ev_12h.mean()
    arm_era = eras.set_index("era").top10_net_ev
    era_delta = arm_era - control_era
    control_side = control.groupby("side_name").execution_net_ev_12h.mean()
    arm_side = selected.groupby("side_name").execution_net_ev_12h.mean()
    side_delta = arm_side - control_side
    period_deltas: dict[str, float] = {}
    for kind in ("week", "month"):
        arm = periods[periods.period_kind.eq(kind)].set_index(["era", "period"]).net_ev
        base = period_summary(control, "W" if kind == "week" else "M").set_index(["era", "period"]).net_ev
        delta = arm.sub(base, fill_value=np.nan).dropna()
        period_deltas[f"{kind}_q10_delta"] = q(delta, 0.10)
        period_deltas[f"{kind}_q50_delta"] = q(delta, 0.50)
    eligible = bool(
        penalty > 0
        and aggregate_net > control_net
        and era_delta.min() >= 0
        and side_delta.reindex(["long", "short"]).notna().all()
        and (side_delta.reindex(["long", "short"]) >= 0).all()
        and all(period_deltas[name] >= 0 for name in period_deltas)
    )
    return {
        "basis": basis,
        "lambda": penalty,
        "aggregate_top10_net_ev": aggregate_net,
        "aggregate_delta_vs_residual": aggregate_net - control_net,
        "minimum_era_delta_vs_residual": era_delta.min(),
        "long_delta_vs_residual": side_delta.get("long", np.nan),
        "short_delta_vs_residual": side_delta.get("short", np.nan),
        **period_deltas,
        "eligible": eligible,
    }


def run() -> Path:
    if OUT.exists():
        raise FileExistsError(f"immutable output exists: {OUT}")
    sealed(CALIBRATOR, "manifest.json", "hourly_oof_predictions.parquet", "candidate_oof_broadcast_scores.parquet", "hourly_design_integrity_audit.csv")
    predictions = pd.read_parquet(CALIBRATOR / "hourly_oof_predictions.parquet")
    hourly = expected_downside_hourly(predictions)
    supported_eras = set(hourly.era)
    source = pd.read_parquet(CALIBRATOR / "candidate_oof_broadcast_scores.parquet")
    source = source.loc[source.arm.eq("regime")].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source = source.loc[
        source.era.isin(supported_eras)
        & source.residual_score.notna()
        & source.execution_net_ev_12h.notna()
    ].copy()
    if source.empty or source.candidate_id.duplicated().any() or source["__ts__"].dt.minute.ne(0).any():
        raise RuntimeError("invalid supported hourly source universe")
    frame = source.merge(hourly, on=["era", "__ts__"], how="left", validate="many_to_one")
    for col in ("score_only_expected_downside", "context_expected_downside"):
        frame[f"{col}_available"] = frame[col].notna()
        frame[col] = pd.to_numeric(frame[col], errors="raise").fillna(0.0)  # explicit zero fallback only
    frame["residual_control_score"] = frame.residual_score
    control_selected, _, control_eras, control_sides, control_periods = evaluate_arm(
        frame, "residual_control_score", "residual_control", 0.0
    )

    all_selected = [control_selected.assign(basis="residual_control", lambda_=0.0)]
    all_eras = [control_eras]
    all_sides = [control_sides]
    all_periods = [control_periods]
    order_audits: list[dict[str, object]] = [
        {"basis": "residual_control", "lambda": 0.0, "max_within_hour_penalty_span": 0.0, "within_hour_order_preserved": True}
    ]
    gates: list[dict[str, object]] = []
    for basis in ("score_only_expected_downside", "context_expected_downside"):
        for penalty in LAMBDAS:
            if penalty == 0:
                continue
            score = f"{basis}_lambda_{penalty:g}"
            frame[score] = frame.residual_score - penalty * frame[basis]
            span = (
                (frame[score] - frame.residual_score)
                .groupby([frame["era"], frame["__ts__"]])
                .agg(lambda x: float(x.max() - x.min()))
            )
            if not span.le(1e-15).all():
                raise RuntimeError("broadcast penalty changes within-hour order")
            order_audits.append(
                {"basis": basis, "lambda": penalty, "max_within_hour_penalty_span": span.max(), "within_hour_order_preserved": True}
            )
            selected, _, eras, sides, periods = evaluate_arm(frame, score, basis, penalty)
            all_selected.append(selected.assign(basis=basis, lambda_=penalty))
            all_eras.append(eras)
            all_sides.append(sides)
            all_periods.append(periods)
            gates.append(gate(control_selected, selected, eras, sides, periods, basis, penalty))
    eligibility = pd.DataFrame(gates)
    selected = pd.concat(all_selected, ignore_index=True)
    eras = pd.concat(all_eras, ignore_index=True)
    sides = pd.concat(all_sides, ignore_index=True)
    periods = pd.concat(all_periods, ignore_index=True)
    order_audit = pd.DataFrame(order_audits)
    selected_gamma = None
    if eligibility.eligible.any():
        selected_gamma = eligibility.loc[eligibility.eligible].sort_values(
            ["minimum_era_delta_vs_residual", "aggregate_delta_vs_residual", "lambda"],
            ascending=[False, False, True], kind="stable"
        ).iloc[0].to_dict()

    stage = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    try:
        hourly.to_parquet(stage / "hourly_broadcasts.parquet", index=False)
        frame.to_parquet(stage / "scored_candidates.parquet", index=False)
        selected.to_parquet(stage / "selected_global_top10.parquet", index=False)
        eras.to_csv(stage / "era_top10_metrics.csv", index=False)
        sides.to_csv(stage / "side_top10_metrics.csv", index=False)
        periods.to_csv(stage / "weekly_monthly_top10_metrics.csv", index=False)
        eligibility.to_csv(stage / "eligibility.csv", index=False)
        order_audit.to_csv(stage / "within_hour_order_audit.csv", index=False)
        coverage = frame.groupby("era", as_index=False).agg(
            candidate_rows=("candidate_id", "size"),
            broadcast_hours=("__ts__", "nunique"),
            score_only_fallback_rows=("score_only_expected_downside_available", lambda x: int((~x).sum())),
            context_fallback_rows=("context_expected_downside_available", lambda x: int((~x).sum())),
        )
        coverage.to_csv(stage / "coverage_and_zero_fallback.csv", index=False)
        contract = {
            "schema": "pre2026_regime_only_downside_risk_broadcast_v3",
            "status": "SEALED_PRE2026_REGIME_EXPECTED_DOWNSIDE_BROADCAST_NON_PROMOTION",
            "decision_cadence": "1h",
            "exact_replay_bar_cadence": "1m_labels_only",
            "supported_eras": sorted(supported_eras),
            "risk_sources": ["score_only_expected_downside", "context_expected_downside"],
            "risk_scalar": "clip(book_opportunity,0,1)*clip(book_failure_rate_if_selected,0,1)*clip(book_downside_severity_if_selected,0,+inf)",
            "broadcast": "authoritative hourly expected-downside scalar; same penalty for every candidate in an hour, asserted by within_hour_order_audit; zero for unavailable timestamp",
            "lambdas": list(LAMBDAS),
            "selection": "one deterministic pooled global top10 within each supported held era; weekly/monthly tables decompose fixed membership",
            "eligibility_gate": "strict aggregate and every-era improvement versus absolute residual control; non-negative long/short deltas; non-negative weekly/monthly Q10/Q50 deltas",
            "selected_for_2026": selected_gamma,
            "authorized_for_2026": False,
            "no_2026": True,
            "implementation_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__))},
        }
        (stage / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {
            "schema": contract["schema"],
            "status": contract["status"],
            "promotion_eligible": selected_gamma is not None,
            "contract": contract,
            "inputs_sha256": {
                str((CALIBRATOR / "manifest.json").resolve()): sha256(CALIBRATOR / "manifest.json"),
                str((CALIBRATOR / "hourly_oof_predictions.parquet").resolve()): sha256(CALIBRATOR / "hourly_oof_predictions.parquet"),
                str((CALIBRATOR / "candidate_oof_broadcast_scores.parquet").resolve()): sha256(CALIBRATOR / "candidate_oof_broadcast_scores.parquet"),
            },
            "outputs_sha256": {path.name: sha256(path) for path in files},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, OUT)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return OUT


if __name__ == "__main__":
    print(run())
