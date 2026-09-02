#!/usr/bin/env python3
"""Materialise the final 12 December 2025 causal context hours.

The canonical sidecar excludes these rows because their transition labels
resolve after the 2026 boundary.  This separate extension does not read those
labels: it reconstructs the frozen July-2025 fold models from earlier resolved
transition labels and applies them only to decision-time raw catalogue fields.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_changepoint import CHANGEPOINT_INPUT_COLUMNS
from scripts.materialize_authoritative_soft_regime_transition_sidecars import (
    REGIME_CONTEXT, _entropy_margin, assemble_sidecars, sha256,
)
from scripts.run_strict_bocpd_regime_transition_challenger import (
    HEADS, _features as bocpd_features, _fit as bocpd_fit, _signal_context,
)
from scripts.run_strict_forward_transition_challenger_v2 import FOLDS, family_features, model
from scripts.run_strict_forward_transition_evaluation import label_available

ART = ROOT / "data_perp/artifacts"
CATALOGUE = ART / "transition_pattern_catalogue_20260730_v6/adaptive_phase_labels.parquet"
LGBM_ROOT = ART / "strict_forward_transition_challenger_20260730_v2"
BOCPD_ROOT = ART / "strict_bocpd_regime_transition_challenger_20260730_v2"
AUTHORITATIVE = ART / "authoritative_soft_regime_transition_sidecars_20260730_v1"
OUT = ART / "dec2025_final12h_frozen_predec_regime_transition_context_extension_20260730_v1"
FOLD_START = pd.Timestamp("2025-07-01T00:00:00Z")
SCORE_START = pd.Timestamp("2025-12-31T12:00:00Z")
SCORE_END = pd.Timestamp("2026-01-01T00:00:00Z")


class ExtensionError(RuntimeError):
    pass


def _dump(path: Path, value: object) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def _sealed(root: Path, schema: str, status_prefix: str | None = None) -> dict:
    manifest = root / "manifest.json"
    marker = root / "manifest.sha256"
    if not manifest.is_file() or not marker.is_file() or marker.read_text().split()[0] != sha256(manifest):
        raise ExtensionError(f"unsealed source: {root}")
    value = json.loads(manifest.read_text())
    if value.get("schema") != schema or (status_prefix and not str(value.get("status", "")).startswith(status_prefix)):
        raise ExtensionError(f"wrong source contract: {root}")
    return value


def _causal_numeric_fields() -> list[str]:
    schema = pq.ParquetFile(CATALOGUE).schema_arrow
    excluded = {"source_utc", "execution_decision_utc", "calendar_segment_id", "source_segment_id", "segment_id"}
    result = []
    for field in schema:
        if field.name in excluded or field.name.startswith(("target__", "state_context__", "source_artifact")):
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type):
            result.append(field.name)
    if not set(CHANGEPOINT_INPUT_COLUMNS).issubset(result):
        raise ExtensionError("causal BOCPD raw fields are absent from catalogue schema")
    return result


def _raw_history() -> pd.DataFrame:
    columns = ["source_utc", "source_segment_id", *_causal_numeric_fields()]
    raw = pd.read_parquet(CATALOGUE, columns=columns)
    raw["source_utc"] = pd.to_datetime(raw["source_utc"], utc=True, errors="raise")
    raw = raw.loc[raw["source_utc"].lt(SCORE_END)].sort_values("source_utc", kind="stable").reset_index(drop=True)
    if raw["source_utc"].duplicated().any() or raw["source_utc"].min() > pd.Timestamp("2022-08-30T00:00:00Z"):
        raise ExtensionError("raw causal source is not complete hourly history")
    requested = raw.loc[raw["source_utc"].ge(SCORE_START), "source_utc"].to_numpy()
    if len(requested) != 12 or not np.array_equal(requested, pd.date_range(SCORE_START, periods=12, freq="h", tz="UTC").to_numpy()):
        raise ExtensionError("the final twelve raw decision-time timestamps are not complete")
    return raw


def _training_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [target for _, target in HEADS]
    columns = ["source_utc", "target__available_utc", "target__pattern_phase_available_utc", *targets, *_causal_numeric_fields()]
    labels = pd.read_parquet(CATALOGUE, columns=columns, filters=[("source_utc", "<", FOLD_START)])
    labels["source_utc"] = pd.to_datetime(labels["source_utc"], utc=True, errors="raise")
    resolved = label_available(labels)
    labels = labels.loc[resolved.lt(SCORE_END)].copy()
    labels["label_resolution_utc"] = resolved.loc[labels.index].to_numpy()
    for target in targets:
        labels[target] = pd.to_numeric(labels[target], errors="coerce").fillna(0).astype(int)
    labels = labels.sort_values("source_utc", kind="stable").reset_index(drop=True)
    fit = labels.loc[labels["label_resolution_utc"].lt(FOLD_START)].copy().reset_index(drop=True)
    if (fit.empty or fit["source_utc"].max() >= FOLD_START
            or fit["label_resolution_utc"].max() >= FOLD_START):
        raise ExtensionError("training labels violate frozen July boundary")
    return labels, fit


def _lgbm_probability(raw: pd.DataFrame, labels: pd.DataFrame, features: list[str], winner: dict, *, score_start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    # Recreate the fold-0..2 raw ledger solely to reconstruct the frozen
    # fold-3 Platt calibrator.  All these labels resolve before the July fit.
    oof: list[pd.DataFrame] = []
    for fold, start in enumerate(FOLDS[:3]):
        stop = start + pd.DateOffset(months=6)
        fit = labels.loc[labels["source_utc"].lt(start)].copy()
        score = labels.loc[labels["source_utc"].ge(start) & labels["source_utc"].lt(stop)].copy()
        if fit.empty or score.empty or fit["target__transition_active"].nunique() != 2:
            raise ExtensionError(f"insufficient frozen calibration support for LGBM fold {fold}")
        imputer = SimpleImputer(strategy="median")
        x, z = imputer.fit_transform(fit[features]), imputer.transform(score[features])
        y = fit["target__transition_active"].to_numpy(int)
        weight = float(winner["positive_weight"])
        classifier = model(str(winner["model"]), multiclass=False, seed=20260730 + fold).fit(x, y, sample_weight=np.where(y == 1, weight, 1.0))
        probability = classifier.predict_proba(z)[:, list(classifier.classes_).index(1)]
        oof.append(pd.DataFrame({"raw": probability, "y": score["target__transition_active"].to_numpy(int)}))
    prior = pd.concat(oof, ignore_index=True)
    calibrator = LogisticRegression(C=1.0, max_iter=200, random_state=20260730).fit(prior[["raw"]], prior["y"])
    imputer = SimpleImputer(strategy="median")
    fit = labels.loc[labels["label_resolution_utc"].lt(FOLD_START)].copy()
    x = imputer.fit_transform(fit[features])
    y = fit["target__transition_active"].to_numpy(int)
    classifier = model(str(winner["model"]), multiclass=False, seed=20260733).fit(x, y, sample_weight=np.where(y == 1, float(winner["positive_weight"]), 1.0))
    score = raw.loc[raw["source_utc"].ge(score_start), ["source_utc", *features]].copy()
    z = imputer.transform(score[features])
    raw_probability = classifier.predict_proba(z)[:, list(classifier.classes_).index(1)]
    score["lgbm_transition_probability"] = calibrator.predict_proba(pd.DataFrame({"raw": raw_probability}))[:, 1]
    score["lgbm_entropy"], score["lgbm_margin"] = _entropy_margin(score["lgbm_transition_probability"])
    score["lgbm_transition_available"] = True
    score["lgbm_ood_available"] = False
    score["lgbm_ood_score"] = np.nan
    score["provenance_partition"] = "frozen_predec_score_only_extension"
    score["train_end_exclusive_utc"] = FOLD_START
    score["fit_label_resolution_max_utc"] = fit["label_resolution_utc"].max()
    audit = {"features": features, "fold_start_utc": FOLD_START, "fit_rows": int(len(fit)),
             "fit_label_resolution_max_utc": fit["label_resolution_utc"].max(), "calibration_rows": int(len(prior)),
             "no_december_label_read": True}
    return score, audit


def _bocpd_context(raw: pd.DataFrame) -> pd.DataFrame:
    reference = raw.loc[raw["source_utc"].lt(FOLD_START)].copy()
    pieces = [_signal_context(reference, raw, signal=signal, horizon=24) for signal in CHANGEPOINT_INPUT_COLUMNS]
    joined = pieces[0]
    for item in pieces[1:]:
        joined = joined.merge(item, on="source_utc", how="inner", validate="one_to_one")
    probabilities = joined[[f"bocpd__{signal}__change_probability" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    run_mean = joined[[f"bocpd__{signal}__run_length_mean" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    run_q05 = joined[[f"bocpd__{signal}__run_length_q05" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    entropy = joined[[f"bocpd__{signal}__run_length_entropy" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    joined["bocpd__change_probability_mean"] = probabilities.mean(axis=1)
    joined["bocpd__change_probability_max"] = probabilities.max(axis=1)
    joined["bocpd__run_length_mean"] = run_mean.mean(axis=1)
    joined["bocpd__run_length_q05"] = run_q05.mean(axis=1)
    joined["bocpd__run_length_entropy"] = entropy.mean(axis=1)
    joined["bocpd__signal_count"] = float(len(CHANGEPOINT_INPUT_COLUMNS))
    context, threshold = bocpd_features(joined, len(reference))
    if context["source_utc"].max() != SCORE_END - pd.Timedelta(hours=1):
        raise ExtensionError("BOCPD causal raw history stops before requested final hour")
    return context, {"reference_rows": int(len(reference)), "threshold": float(threshold), "horizon": 24}


def _bocpd_rows(context: pd.DataFrame, labels: pd.DataFrame, winners: pd.DataFrame, *, score_start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    train = labels.loc[:, ["source_utc", *[target for _, target in HEADS], "label_resolution_utc"]].merge(context, on="source_utc", how="inner", validate="one_to_one")
    if len(train) != len(labels):
        raise ExtensionError("BOCPD training label/raw identity mismatch")
    score = context.loc[context["source_utc"].ge(score_start)].copy()
    audit: dict[str, dict] = {}
    for head, target in HEADS:
        winner = winners.loc[winners["head"].eq(head)]
        if len(winner) != 1 or int(winner.iloc[0]["expected_run_hours"]) != 24:
            raise ExtensionError(f"unexpected frozen BOCPD winner: {head}")
        row = winner.iloc[0]
        probability = bocpd_fit(train, score, target=target, c=float(row["logistic_c"]))
        field = f"bocpd_{head}_probability"
        score[field] = probability
        score[f"bocpd_{head}_available"] = True
        score[f"bocpd_{head}_entropy"], score[f"bocpd_{head}_margin"] = _entropy_margin(score[field])
        audit[head] = {"fit_rows": int(len(train)), "fit_label_resolution_max_utc": train["label_resolution_utc"].max(),
                       "logistic_c": float(row["logistic_c"]), "no_december_label_read": True}
    score["bocpd_regime_available"] = score.loc[:, list(REGIME_CONTEXT)].notna().all(axis=1)
    score["bocpd_ood_available"] = False
    score["bocpd_ood_score"] = np.nan
    score["provenance_partition"] = "frozen_predec_score_only_extension"
    score["train_end_exclusive_utc"] = FOLD_START
    score["fit_label_resolution_max_utc"] = train["label_resolution_utc"].max()
    return score, audit


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    lgbm_manifest = _sealed(LGBM_ROOT, "strict_forward_transition_challenger_v2")
    _sealed(BOCPD_ROOT, "strict_bocpd_regime_transition_challenger_v2", "SEALED_STRICT_RESUMABLE_BOCPD")
    _sealed(AUTHORITATIVE, "authoritative_soft_regime_transition_sidecars_v1", "SEALED")
    winner = lgbm_manifest["winner"]["active"]
    raw = _raw_history()
    calibration_labels, labels = _training_labels()
    fit_raw = raw.merge(labels.loc[:, ["source_utc"]], on="source_utc", how="inner", validate="one_to_one")
    features = family_features(raw, fit_raw, str(winner["family"]))
    if not features:
        raise ExtensionError("frozen fold-03 causal feature selection is empty")
    # Score a 24-hour overlap-plus-missing window so the preceding twelve
    # timestamps can be compared to the canonical sidecar exactly.
    score_window = raw.loc[raw["source_utc"].ge(SCORE_START - pd.Timedelta(hours=12))].copy()
    lgbm, lgbm_audit = _lgbm_probability(score_window, calibration_labels, features, winner, score_start=SCORE_START - pd.Timedelta(hours=12))
    context, context_audit = _bocpd_context(raw)
    winners = pd.read_csv(BOCPD_ROOT / "frozen_bocpd_winners.csv")
    bocpd_all, bocpd_audit = _bocpd_rows(context, labels, winners, score_start=SCORE_START - pd.Timedelta(hours=12))
    bocpd = bocpd_all.copy()
    regime, transition = assemble_sidecars(lgbm, bocpd)
    old_regime = pd.read_parquet(AUTHORITATIVE / "soft_regime_hourly.parquet")
    old_transition = pd.read_parquet(AUTHORITATIVE / "soft_transition_hourly.parquet")
    for frame in (old_regime, old_transition):
        frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    overlap = regime["source_utc"].lt(SCORE_START)
    old_regime = old_regime.loc[old_regime["source_utc"].isin(regime.loc[overlap, "source_utc"])]
    old_transition = old_transition.loc[old_transition["source_utc"].isin(transition.loc[overlap, "source_utc"])]
    def diff(left: pd.DataFrame, right: pd.DataFrame) -> dict:
        joined = left.merge(right, on="source_utc", suffixes=("_new", "_old"), validate="one_to_one")
        metrics = {}
        for column in left.columns:
            if column == "source_utc" or column not in right.columns or not pd.api.types.is_numeric_dtype(left[column]):
                continue
            if pd.api.types.is_bool_dtype(left[column]):
                metrics[column] = float((joined[f"{column}_new"].astype(bool) != joined[f"{column}_old"].astype(bool)).sum())
            else:
                delta = (pd.to_numeric(joined[f"{column}_new"], errors="coerce") - pd.to_numeric(joined[f"{column}_old"], errors="coerce")).abs()
                metrics[column] = float(delta.max())
        return {"rows": int(len(joined)), "max_abs_by_numeric_field": metrics, "max_abs": float(max(metrics.values(), default=0.0))}
    validation = {"regime": diff(regime.loc[overlap], old_regime), "transition": diff(transition.loc[overlap], old_transition)}
    if validation["regime"]["rows"] != 12 or validation["transition"]["rows"] != 12 or max(validation["regime"]["max_abs"], validation["transition"]["max_abs"]) > 1e-8:
        stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
        try:
            report = {"status": "FROZEN_RECONSTRUCTION_DOES_NOT_REPRODUCE_CANONICAL_OVERLAP", "promotion_eligible": False,
                      "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                      "raw_input_availability": "all 12 final decision-time raw catalogue timestamps are present",
                      "blocked_reason": "the score-only reconstruction fails exact Dec31 00:00--11:00 canonical sidecar reproduction; final12 values are withheld",
                      "overlap_validation": validation,
                      "safe_remediation": "locate serialized frozen fold-03 LGBM and BOCPD head/imputer/calibrator state, or reproduce the canonical sidecar builder's exact persisted checkpoints; do not fill, forward-fill, or use these unvalidated predictions"}
            _dump(stage / "readiness_report.json", report)
            manifest = {"schema": "dec2025_final12h_frozen_predec_regime_transition_context_extension_v1",
                        "status": "SEALED_FAIL_CLOSED_FROZEN_CONTEXT_REPRODUCTION_MISMATCH", "promotion_eligible": False,
                        "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                        "inputs_sha256": {"catalogue": sha256(CATALOGUE), "lgbm_manifest": sha256(LGBM_ROOT / "manifest.json"), "bocpd_manifest": sha256(BOCPD_ROOT / "manifest.json"), "canonical_sidecar_manifest": sha256(AUTHORITATIVE / "manifest.json")},
                        "outputs_sha256": {"readiness_report.json": sha256(stage / "readiness_report.json")}}
            _dump(stage / "manifest.json", manifest)
            (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n")
            os.replace(stage, output)
            return output
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    regime = regime.loc[regime["source_utc"].ge(SCORE_START)].copy()
    transition = transition.loc[transition["source_utc"].ge(SCORE_START)].copy()
    if len(regime) != 12 or len(transition) != 12 or not regime["bocpd_regime_available"].all() or not transition["lgbm_transition_available"].all():
        raise ExtensionError("final extension rows are not complete")
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        regime.to_parquet(stage / "soft_regime_extension.parquet", index=False, compression="zstd")
        transition.to_parquet(stage / "soft_transition_extension.parquet", index=False, compression="zstd")
        _dump(stage / "frozen_score_audit.json", {"lgbm": lgbm_audit, "bocpd_context": context_audit, "bocpd_heads": bocpd_audit, "overlap_validation": validation})
        contract = {"decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                    "scope": "2025-12-31 12:00 through 23:00 UTC only",
                    "reason": "canonical historical OOF sidecar excluded rows whose transition labels resolve after 2026-01-01",
                    "method": "reconstruct frozen 2025-07 fold LGBM and BOCPD heads using only transition labels resolved before 2025-07-01; apply only decision-time causal raw catalogue inputs",
                    "provenance_partition": "frozen_predec_score_only_extension",
                    "forbidden": "no December transition target labels, no execution labels, no 2026 inputs, no HPO, no feature selection, no retraining on post-cutoff labels",
                    "validation": "reproduces the immediately preceding twelve canonical hourly sidecar rows to max absolute tolerance 1e-8"}
        _dump(stage / "contract.json", contract)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": "dec2025_final12h_frozen_predec_regime_transition_context_extension_v1",
                    "status": "SEALED_FROZEN_PREDEC_CAUSAL_CONTEXT_EXTENSION_NON_PROMOTION", "promotion_eligible": False,
                    "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                    "historical_contract": "canonical sidecar remains blocked-OOF; this exact final12 append is separately identified frozen pre-December score-only context",
                    "forward_contract": "untouched 2026 provenance remains in the unmodified canonical sidecar and is not read here",
                    "inputs_sha256": {"catalogue": sha256(CATALOGUE), "lgbm_manifest": sha256(LGBM_ROOT / "manifest.json"), "bocpd_manifest": sha256(BOCPD_ROOT / "manifest.json"), "canonical_sidecar_manifest": sha256(AUTHORITATIVE / "manifest.json")},
                    "contract": contract, "outputs_sha256": {path.name: sha256(path) for path in files}}
        _dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
