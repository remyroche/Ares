#!/usr/bin/env python3
"""Sealed pre-registered hourly residual x context follow-on.

The only discovery input is the corrected final-v3 diagnostic v2.  It fixes
one continuous regime and one transition interaction per permitted side before
the untouched 2026 ledger is read.  It is deliberately a small Ridge learner:
the question here is whether the discovered *interaction* transfers, not
whether a new high-capacity context model can be found on the assessment year.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, validate_candidate_identity
from scripts import run_final_identical_row_regime_stack_gam_ablation as base

SCHEMA = "final_v3_preregistered_residual_interactions_v1"
DIAGNOSTIC_SCHEMA = "final_v3_context_interaction_diagnostics_v2"
DIAGNOSTIC = ROOT / "data_perp/artifacts/final_v3_context_interaction_diagnostics_20260730_v2"
V3_CONTROL = ROOT / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3"
OUT = ROOT / "data_perp/artifacts/final_v3_preregistered_residual_interactions_20260730_v1"


@dataclass(frozen=True)
class InteractionArm:
    name: str
    active_side: str
    context_fields: tuple[str, ...]


# This is the complete preregistration.  It was frozen from pre-2026 rows in
# corrected diagnostic v2: leading residual x regime_state_age_hours and
# residual x transition_lgbm_probability SHAP interaction.  No 2026 result can
# alter it.  In particular there is intentionally no short regime-only arm.
ARMS = (
    InteractionArm("long_residual_x_regime_state_age", "long", ("regime_state_age_hours",)),
    InteractionArm("long_residual_x_transition_probability", "long", ("transition_lgbm_probability",)),
    InteractionArm("long_residual_x_combined", "long", ("regime_state_age_hours", "transition_lgbm_probability")),
    InteractionArm("short_residual_x_transition_probability", "short", ("transition_lgbm_probability",)),
    InteractionArm("short_residual_x_combined", "short", ("regime_state_age_hours", "transition_lgbm_probability")),
)
CONTROL_NAMES = ("baseline", "gam_regime_only", "gam_transition_only", "gam_combined")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _sealed_manifest(path: Path, schema: str) -> dict[str, Any]:
    manifest = path / "manifest.json"
    marker = path / "manifest.sha256"
    if not manifest.is_file() or not marker.is_file() or marker.read_text().split(maxsplit=1)[0] != base.sha(manifest):
        raise RegimeOOFStackError(f"sealed manifest invalid: {path}")
    value = json.loads(manifest.read_text())
    if value.get("schema") != schema or not str(value.get("status", "")).startswith("SEALED"):
        raise RegimeOOFStackError(f"wrong sealed schema/status: {path}")
    return value


def _cadence(frame: pd.DataFrame, *, name: str) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame.__ts__, utc=True, errors="raise")
    hourly = timestamps.astype("int64") % pd.Timedelta(hours=1).value == 0
    if not hourly.all():
        raise RegimeOOFStackError(f"{name} has non-hourly rows")
    return {"table": name, "rows": int(len(frame)), "unique_timestamps": int(timestamps.nunique()), "duplicate_candidate_identity_rows": int(frame.duplicated(list(IDENTITY_COLUMNS)).sum()), "non_hourly_rows": int((~hourly).sum()), "cadence": "1h"}


def _features(frame: pd.DataFrame, arm: InteractionArm) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    names = [base.RESIDUAL, *arm.context_fields]
    for field in arm.context_fields:
        name = f"{base.RESIDUAL}__x__{field}"
        out[name] = pd.to_numeric(out[base.RESIDUAL], errors="coerce") * pd.to_numeric(out[field], errors="coerce")
        names.append(name)
    return out, names


def _predict(train: pd.DataFrame, test: pd.DataFrame, arm: InteractionArm) -> tuple[np.ndarray, dict[str, Any]]:
    fit = train.loc[pd.to_numeric(train[base.TARGET], errors="coerce").notna()].copy()
    if len(fit) < 8:
        raise RegimeOOFStackError(f"insufficient targets for {arm.name}")
    fit, fields = _features(fit, arm)
    view, _ = _features(test, arm)
    x, z = base._matrix(fit, view, fields)
    y = pd.to_numeric(fit[base.TARGET], errors="raise").to_numpy(float)
    # Fixed low-capacity, regularised interaction learner; no HPO in this
    # follow-on.  Standardisation makes the interaction penalty comparable.
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=80.0))]).fit(x, y)
    return np.asarray(model.predict(z), float), {"family": "fixed_regularized_ridge_interaction", "features": fields, "ridge_alpha": 80.0, "non_null_target_fit_rows": len(fit), "excluded_null_target_fit_rows": len(train) - len(fit)}


def _oof(history: pd.DataFrame, arm: InteractionArm, *, start: pd.Timestamp, min_train: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    blocks = pd.date_range(start.normalize(), history.__ts__.max().normalize() + pd.Timedelta(days=1), freq="3MS", tz="UTC")
    for number, block in enumerate(blocks):
        evaluation = history.loc[(history.__ts__ >= block) & (history.__ts__ < block + pd.DateOffset(months=3))].copy()
        training = history.loc[history.execution_label_end_utc < block].copy()
        if evaluation.empty or len(training) < min_train:
            continue
        active_eval = evaluation.loc[evaluation.side_name.eq(arm.active_side)].copy()
        active_fit = training.loc[training.side_name.eq(arm.active_side)].copy()
        if len(active_fit) < min_train // 3:
            continue
        raw, model = _predict(active_fit, active_eval, arm)
        # Preserve every candidate in the global universe: the non-target side
        # is the frozen residual baseline, never dropped or re-tuned.
        evaluation["raw_score"] = evaluation[base.RESIDUAL].to_numpy(float)
        evaluation.loc[active_eval.index, "raw_score"] = raw
        rows.append(evaluation.loc[:, [*IDENTITY_COLUMNS, "execution_label_end_utc", base.TARGET, "raw_score"]].assign(arm=arm.name, oof_block_start_utc=block))
        audit.append({"arm": arm.name, "active_side": arm.active_side, "oof_block_start_utc": block, "train_rows": len(active_fit), "evaluation_rows": len(evaluation), "active_evaluation_rows": len(active_eval), "train_label_end_max": active_fit.execution_label_end_utc.max(), **model})
    if not rows:
        raise RegimeOOFStackError(f"no prior OOF support for {arm.name}")
    return validate_candidate_identity(pd.concat(rows, ignore_index=True)), pd.DataFrame(audit)


def _forward(history: pd.DataFrame, current: pd.DataFrame, arm: InteractionArm, mapper, oof: pd.DataFrame) -> pd.DataFrame:
    result = current.copy()
    result["raw_score"] = result[base.RESIDUAL].to_numpy(float)
    fit = history.loc[history.side_name.eq(arm.active_side)]
    target = current.loc[current.side_name.eq(arm.active_side)]
    raw, _ = _predict(fit, target, arm)
    result.loc[target.index, "raw_score"] = raw
    result["mapped_score"] = mapper(result.raw_score.to_numpy(float))
    result["arm"] = arm.name
    result["map_source_last_label_end_utc"] = pd.to_datetime(oof.execution_label_end_utc, utc=True).max()
    result["map_age_days"] = (result.__ts__.min() - result.map_source_last_label_end_utc.iloc[0]).days
    return result


def _controls(current: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    _sealed_manifest(V3_CONTROL, base.SCHEMA)
    frozen = pd.read_parquet(V3_CONTROL / "frozen_2026_candidate_scores.parquet")
    current_ids = current.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    frames=[]; summaries=[]; periods=[]; sides=[]; calibration=[]
    for name in CONTROL_NAMES:
        view = frozen.loc[frozen.arm.eq(name)].copy()
        ids = view.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        if not ids.equals(current_ids):
            raise RegimeOOFStackError(f"v3 control {name} does not have the required identical hourly universe")
        frames.append(view)
        summaries.append(pd.read_csv(V3_CONTROL / "metrics_summary.csv").query("arm == @name"))
        periods.append(pd.read_parquet(V3_CONTROL / "period_metrics.parquet").query("arm == @name"))
        sides.append(pd.read_parquet(V3_CONTROL / "side_metrics.parquet").query("arm == @name"))
        calibration.append(pd.read_parquet(V3_CONTROL / "calibration_deciles.parquet").query("arm == @name"))
    return frames, summaries, periods, sides, calibration


def run(*, sidecar_manifest: Path, historical_scores: Path, current_scores: Path, output: Path = OUT, oof_start: str = "2023-01-01T00:00:00Z", min_train_rows: int = 12000, max_map_age_days: int = 365) -> Path:
    output = Path(output)
    if output.exists():
        raise RegimeOOFStackError(f"refusing to overwrite {output}")
    diagnostic = _sealed_manifest(DIAGNOSTIC, DIAGNOSTIC_SCHEMA)
    sidecar, regime_path, transition_path = base._load_manifest(Path(sidecar_manifest))
    context = base._hourly_context(regime_path, transition_path)
    history_ledger = base._verified_scores(Path(historical_scores), role="historical")
    current_ledger = base._verified_scores(Path(current_scores), role="forward")
    history = base._join(history_ledger, context, role="historical")
    current = base._join(current_ledger, context, role="forward")
    cadence = [_cadence(history_ledger, name="historical_score_ledger"), _cadence(current_ledger, name="forward_score_ledger"), _cadence(history, name="historical_fit_oof_rows"), _cadence(current, name="forward_assessment_rows")]
    if len(current) != 127777:
        raise RegimeOOFStackError(f"expected exact 127777-row 2026 universe, got {len(current)}")
    c_frames, c_summary, c_periods, c_sides, c_calibration = _controls(current)
    all_oof=[]; all_audit=[]; frames=[*c_frames]; summaries=[*c_summary]; periods=[*c_periods]; sides=[*c_sides]; calibration=[*c_calibration]
    for arm in ARMS:
        oof, audit = _oof(history, arm, start=pd.Timestamp(oof_start), min_train=min_train_rows)
        mapper = base._mapper(oof.raw_score.to_numpy(float), oof[base.TARGET].to_numpy(float))
        oof["mapped_score"] = mapper(oof.raw_score.to_numpy(float))
        result = _forward(history, current, arm, mapper, oof)
        if result.map_age_days.iloc[0] > max_map_age_days:
            raise RegimeOOFStackError(f"{arm.name} causal map is stale")
        row, period, side, _, cal = base._evaluate(result, base.Arm(arm.name, "side_local_residual_interaction", "pre_registered", base.TARGET, "ridge"))
        row["map_age_days"] = result.map_age_days.iloc[0]
        frames.append(result); summaries.append(pd.DataFrame([row])); periods.append(period); sides.append(side); calibration.append(cal); all_oof.append(oof); all_audit.append(audit)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.concat(frames, ignore_index=True).to_parquet(temporary / "frozen_2026_candidate_scores.parquet", index=False)
        pd.concat(summaries, ignore_index=True).to_csv(temporary / "metrics_summary.csv", index=False)
        pd.concat(periods, ignore_index=True).to_parquet(temporary / "period_metrics.parquet", index=False)
        pd.concat(sides, ignore_index=True).to_parquet(temporary / "side_metrics.parquet", index=False)
        pd.concat(calibration, ignore_index=True).to_parquet(temporary / "calibration_deciles.parquet", index=False)
        pd.concat(all_oof, ignore_index=True).to_parquet(temporary / "historical_oof_scores.parquet", index=False)
        pd.concat(all_audit, ignore_index=True).to_parquet(temporary / "oof_fit_audit.parquet", index=False)
        pd.DataFrame(cadence).to_csv(temporary / "row_cadence_audit.csv", index=False)
        contract = {"candidate_cadence": "1h", "fit_oof_mapping_assessment_cadence": "all rows are 1h; each table has zero non-hourly rows", "minute_data": "1m is permitted only inside existing nested exact 12h label/path/replay inputs; it never forms a fit, OOF, mapping, or assessment row", "discovery": "only corrected final-v3 diagnostic v2 pre-2026 evidence; selected fields are frozen in code", "arms": [arm.__dict__ for arm in ARMS], "excluded": "short regime-only; raw state IDs, GMM posterior, morphology; no BOCPD standalone score, gate, quota, or promotion", "learner": "side-local fixed StandardScaler+Ridge(alpha=80); selected side only, opposite side is frozen residual baseline", "split": "strict pre-2026 expanding 3-month OOF; labels resolve before each fold; 2026 is untouched", "mapping": "pooled monotone increasing pre-2026 OOF isotonic EV map", "selection": "one pooled global top10 across both sides after mapping; raw score only resolves exact mapped ties", "controls": "sealed v3 baseline and GAM controls, identity-checked against the same 127777 hourly forward candidates"}
        _write_json(temporary / "contract.json", contract)
        files = [p for p in temporary.iterdir() if p.is_file()]
        manifest = {"schema": SCHEMA, "status": "SEALED_STRICT_PRE_REGISTERED_PRE2026_DISCOVERY_UNTOUCHED2026_ASSESSMENT_NON_PROMOTION", "promotion_eligible": False, "diagnostic_manifest_sha256": base.sha(DIAGNOSTIC / "manifest.json"), "sidecar_manifest_sha256": base.sha(Path(sidecar_manifest)), "inputs": {str(Path(p).resolve()): base.sha(Path(p)) for p in (sidecar_manifest, historical_scores, current_scores, regime_path, transition_path, DIAGNOSTIC / "manifest.json", V3_CONTROL / "manifest.json")}, "row_cadence_audit": cadence, "contract": contract, "outputs_sha256": {p.name: base.sha(p) for p in files}}
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(f"{base.sha(temporary / 'manifest.json')}  manifest.json\n")
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-manifest", type=Path, required=True)
    parser.add_argument("--historical-scores", type=Path, required=True)
    parser.add_argument("--current-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--oof-start", default="2023-01-01T00:00:00Z")
    parser.add_argument("--min-train-rows", type=int, default=12000)
    # The frozen v3 controls themselves have a 304-day causal-map age.  Keep
    # the same explicit bound rather than incorrectly rejecting the identical
    # historical/forward split used by those controls.
    parser.add_argument("--max-map-age-days", type=int, default=365)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(run(**vars(parse_args())))
