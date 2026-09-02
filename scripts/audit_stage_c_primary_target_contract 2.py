#!/usr/bin/env python3
"""Fail-closed audit of Stage-C's sole ``retain_h0_given_clear`` target.

This is a contract/readiness audit only.  It never fits a retention model,
changes a target, or promotes a Stage-C result.  The audit recomputes the
conditional H0 target from frozen exact-H12 labels and verifies every v4 OOF
prediction row, candidate-ID comparison ledger and chronological fold record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
PANEL = ART / "stage_c_continuation_feature_panel_20260731_v2/stage_c_candidate_population.parquet"
PERSISTENCE = ART / "historical_exact_h12_postcost_persistence_labels_20260731_v1/postcost_persistence_labels.parquet"
EVENTS = ART / "historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
ALIGNMENT = ART / "historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
STAGE_C_V4 = ART / "stage_c_conditional_retention_ablation_20260731_v4"
DEFAULT_OUTPUT = ART / "stage_c_primary_target_contract_audit_20260801_v1"

PRIMARY_TARGET = "retain_h0_given_clear"
EXPECTED_ARMS = ("C0", "C1", "C2", "C3", "C6", "C8")
BLOCKED_ARMS = ("C4", "C5", "C7")
TARGET_LEAKAGE_TOKENS = (
    "retain_h0_given_clear", "retain_h25_given_clear", "continuous_net_given_clear",
    "postcost_h0_", "postcost_h25_",
)


class StageCTargetContractError(ValueError):
    """An input does not expose the fields needed to audit the contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_hash(values: Iterable[object]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _utc(frame: pd.DataFrame, column: str, *, source: str) -> pd.Series:
    if column not in frame:
        raise StageCTargetContractError(f"{source} is missing {column!r}")
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise StageCTargetContractError(f"{source}.{column} contains an invalid timestamp")
    return value


def _bool(frame: pd.DataFrame, column: str, *, source: str) -> pd.Series:
    if column not in frame:
        raise StageCTargetContractError(f"{source} is missing {column!r}")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise StageCTargetContractError(f"{source}.{column} must be a finite 0/1 field")
    return numeric.astype(bool)


def _required(frame: pd.DataFrame, names: Iterable[str], *, source: str) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise StageCTargetContractError(f"{source} is missing required columns: {missing}")


def _check(name: str, passed: bool, detail: str, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if passed else "BLOCKED",
        "detail": detail,
        "rows": int(rows) if rows is not None else None,
    }


def _same_timestamp(left: pd.Series, right: pd.Series) -> bool:
    return _utc(pd.DataFrame({"value": left}), "value", source="left").equals(
        _utc(pd.DataFrame({"value": right}), "value", source="right")
    )


def _json_feature_tokens(stability: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in ("base_features", "incremental_selected", "model_features"):
        if column not in stability:
            continue
        for value in stability[column].dropna():
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, list):
                names.extend(str(item) for item in parsed)
    return names


def audit_frames(
    *,
    panel: pd.DataFrame,
    persistence: pd.DataFrame,
    events: pd.DataFrame,
    alignment: pd.DataFrame,
    predictions: pd.DataFrame,
    evaluation_ids: pd.DataFrame,
    stability: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return check rows and per-fold coverage; block on any mismatch.

    The target is recomputed rather than trusting a materialised target column:
    H0 clear-first from the frozen event pack, then ``exact_h12_net_bps > 0``.
    """
    _required(panel, (
        "candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts",
        "target_id", "execution_policy_id", "cost_model_id", PRIMARY_TARGET,
        f"{PRIMARY_TARGET}__valid", f"{PRIMARY_TARGET}__condition_met",
        f"{PRIMARY_TARGET}__support_side", f"{PRIMARY_TARGET}__support_month",
    ), source="Stage-C panel")
    _required(persistence, (
        "candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts",
        "exact_h12_net_bps", "postcost_h0_clear_first",
        "postcost_h0_persistence_target_valid", "postcost_h0_retained_net",
    ), source="frozen persistence labels")
    _required(events, ("candidate_id", "postcost_h0_event"), source="frozen post-cost events")
    _required(alignment, (
        "candidate_id", "target_id", "execution_policy_id", "cost_model_id", "exact_h12_net_bps",
    ), source="frozen exact-H12 alignment")
    checks: list[dict[str, Any]] = []
    for name, frame in (("panel", panel), ("persistence", persistence), ("events", events), ("alignment", alignment)):
        checks.append(_check(
            f"candidate_id_one_to_one__{name}",
            frame.candidate_id.notna().all() and not frame.candidate_id.astype(str).duplicated().any(),
            f"{name} candidate IDs are non-null and one-to-one", rows=len(frame),
        ))
    panel_keys = panel.loc[:, [
        "candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts",
        "target_id", "execution_policy_id", "cost_model_id", PRIMARY_TARGET,
        f"{PRIMARY_TARGET}__valid", f"{PRIMARY_TARGET}__condition_met",
        f"{PRIMARY_TARGET}__support_side", f"{PRIMARY_TARGET}__support_month",
    ]].copy()
    frozen = persistence.merge(events, on="candidate_id", how="inner", validate="one_to_one")
    frozen = frozen.merge(alignment, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "__alignment"))
    checks.append(_check(
        "frozen_label_join_complete",
        len(frozen) == len(persistence) == len(events) == len(alignment),
        "event, persistence and exact-H12 alignment sources have the same candidate universe",
        rows=len(frozen),
    ))
    joined = panel_keys.merge(frozen, on="candidate_id", how="left", validate="one_to_one", suffixes=("__panel", "__frozen"))
    checks.append(_check(
        "stage_c_panel_is_frozen_label_subset",
        len(joined) == len(panel_keys) and joined.postcost_h0_event.notna().all(),
        "every Stage-C row resolves to one frozen event/persistence/alignment row",
        rows=len(joined),
    ))
    for column in ("side", "decision_ts", "label_end_ts", "label_available_ts"):
        left, right = f"{column}__panel", f"{column}__frozen"
        same = joined[left].eq(joined[right]).all() if column == "side" else _same_timestamp(joined[left], joined[right])
        checks.append(_check(
            f"frozen_identity_match__{column}", same,
            f"Stage-C {column} exactly matches frozen persistence labels", rows=len(joined),
        ))
    for column in ("target_id", "execution_policy_id", "cost_model_id"):
        panel_column = f"{column}__panel"
        frozen_column = f"{column}__frozen"
        checks.append(_check(
            f"frozen_contract_id_match__{column}",
            joined[panel_column].notna().all()
            and joined[panel_column].nunique(dropna=False) == 1
            and joined[panel_column].eq(joined[frozen_column]).all(),
            f"Stage-C {column} is one frozen value and equals exact-H12 alignment", rows=len(joined),
        ))
    decision = _utc(joined, "decision_ts__panel", source="panel")
    label_end = _utc(joined, "label_end_ts__panel", source="panel")
    label_available = _utc(joined, "label_available_ts__panel", source="panel")
    checks.append(_check(
        "exact_h12_label_availability",
        label_end.eq(decision + pd.Timedelta(hours=12)).all() and label_available.eq(label_end).all(),
        "all Stage-C labels resolve exactly at decision_ts + 12h and become available then", rows=len(joined),
    ))
    clear_event = joined.postcost_h0_event.eq("clear_cost_first")
    clear_label = _bool(joined, "postcost_h0_clear_first", source="frozen persistence labels")
    persistence_valid = _bool(joined, "postcost_h0_persistence_target_valid", source="frozen persistence labels")
    retained_frozen = _bool(joined, "postcost_h0_retained_net", source="frozen persistence labels")
    panel_valid = _bool(joined, f"{PRIMARY_TARGET}__valid", source="Stage-C panel")
    panel_condition = _bool(joined, f"{PRIMARY_TARGET}__condition_met", source="Stage-C panel")
    checks.append(_check(
        "h0_clear_first_exact_event_contract",
        clear_event.eq(clear_label).all() and clear_event.eq(persistence_valid).all(),
        "H0 clear-first support is exactly frozen postcost_h0_event == clear_cost_first", rows=int(clear_event.sum()),
    ))
    checks.append(_check(
        "retention_validity_equals_h0_clear_first",
        panel_valid.eq(clear_event).all() and panel_condition.eq(clear_event).all(),
        "valid and condition_met equal H0 clear-first; non-clear rows are never negative examples", rows=int(panel_valid.sum()),
    ))
    frozen_net = pd.to_numeric(joined.exact_h12_net_bps, errors="coerce")
    expected = clear_event & frozen_net.gt(0.0)
    target = pd.to_numeric(joined[PRIMARY_TARGET], errors="coerce")
    checks.append(_check(
        "retain_h0_formula_exact_net_positive_on_clear_support",
        target.loc[panel_valid].notna().all()
        and target.loc[panel_valid].astype(int).eq(expected.loc[panel_valid].astype(int)).all()
        and retained_frozen.eq(expected).all(),
        "retain_h0_given_clear equals (exact_h12_net_bps > 0) only on H0 clear-first support", rows=int(panel_valid.sum()),
    ))
    checks.append(_check(
        "non_clear_retention_label_is_null",
        target.loc[~panel_valid].isna().all(),
        "non-clear candidates carry null target values, not a negative retention label", rows=int((~panel_valid).sum()),
    ))
    support_side = joined[f"{PRIMARY_TARGET}__support_side"]
    support_month = joined[f"{PRIMARY_TARGET}__support_month"]
    checks.append(_check(
        "support_side_and_month_contract",
        support_side.loc[panel_valid].eq(joined.loc[panel_valid, "side__panel"]).all()
        and support_side.loc[~panel_valid].isna().all()
        and support_month.loc[panel_valid].eq(decision.loc[panel_valid].dt.strftime("%Y-%m")).all()
        and support_month.loc[~panel_valid].isna().all(),
        "support side/month are populated only on the H0 clear-first support", rows=int(panel_valid.sum()),
    ))

    _required(predictions, (
        "candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "month", "label",
        "exact_h12_net_bps", "arm", "split", "fold", "prediction",
    ), source="Stage-C v4 OOF predictions")
    prediction_keys = ["candidate_id", "arm", "split", "fold"]
    checks.append(_check(
        "oof_prediction_identity_one_to_one",
        predictions.candidate_id.notna().all() and not predictions.duplicated(prediction_keys).any(),
        "one prediction per candidate/arm/split/fold", rows=len(predictions),
    ))
    oof = predictions.merge(
        joined.loc[:, [
            "candidate_id", "side__panel", "decision_ts__panel", "label_end_ts__panel", "label_available_ts__panel",
            PRIMARY_TARGET, f"{PRIMARY_TARGET}__valid", "exact_h12_net_bps",
        ]],
        on="candidate_id", how="left", validate="many_to_one", suffixes=("", "__frozen"),
    )
    checks.append(_check(
        "oof_rows_join_frozen_primary_target",
        len(oof) == len(predictions) and oof[f"{PRIMARY_TARGET}__valid"].notna().all(),
        "every OOF prediction candidate resolves to the audited Stage-C target panel", rows=len(oof),
    ))
    oof_valid = _bool(oof, f"{PRIMARY_TARGET}__valid", source="joined OOF panel")
    oof_label = pd.to_numeric(oof.label, errors="coerce")
    oof_prediction = pd.to_numeric(oof.prediction, errors="coerce")
    checks.extend([
        _check("oof_rows_are_h0_clear_first_only", oof_valid.all(), "all prediction rows are on the conditional H0 clear-first support", rows=len(oof)),
        _check("oof_labels_equal_recomputed_h0_target", oof_label.notna().all() and oof_label.astype(int).eq(pd.to_numeric(oof[PRIMARY_TARGET], errors="raise").astype(int)).all(), "every OOF label equals the recomputed frozen H0 retention target", rows=len(oof)),
        _check("oof_prediction_range", np.isfinite(oof_prediction).all() and oof_prediction.between(0.0, 1.0).all(), "retention-head predictions are finite probabilities", rows=len(oof)),
        _check("oof_exact_h12_net_match", np.allclose(pd.to_numeric(oof.exact_h12_net_bps, errors="raise"), pd.to_numeric(oof.exact_h12_net_bps__frozen, errors="raise"), rtol=0.0, atol=1e-9), "reported OOF exact H12 net equals frozen exact-H12 net", rows=len(oof)),
        _check("oof_identity_timestamps_and_side_match", oof.side.eq(oof.side__panel).all() and _same_timestamp(oof.decision_ts, oof.decision_ts__panel) and _same_timestamp(oof.label_end_ts, oof.label_end_ts__panel) and _same_timestamp(oof.label_available_ts, oof.label_available_ts__panel), "OOF side and timestamps exactly match frozen labels", rows=len(oof)),
        _check("oof_month_is_decision_month", oof.month.astype(str).eq(_utc(oof, "decision_ts", source="OOF predictions").dt.strftime("%Y-%m")).all(), "OOF month is derived from the decision timestamp", rows=len(oof)),
    ])
    observed_arms = tuple(sorted(oof.arm.astype(str).unique()))
    checks.append(_check(
        "no_unapproved_or_blocked_stage1_arms",
        observed_arms == EXPECTED_ARMS and not set(BLOCKED_ARMS).intersection(observed_arms),
        f"observed prediction arms={observed_arms}; expected only={EXPECTED_ARMS}", rows=len(oof),
    ))
    checks.append(_check(
        "no_h25_or_continuous_target_sweep_in_oof_ledger",
        not any(any(token in column.lower() for token in ("retain_h25", "continuous_net", "postcost_h25")) for column in oof.columns),
        "OOF ledger has no H25 or continuous target field", rows=len(oof),
    ))

    _required(evaluation_ids, ("split", "fold", "arm", "rows", "candidate_id_sha256", "identical_to_c0"), source="candidate identity ledger")
    observed_identity: list[dict[str, Any]] = []
    for (split, fold, arm), part in predictions.groupby(["split", "fold", "arm"], sort=True):
        observed_identity.append({"split": split, "fold": fold, "arm": arm, "rows": len(part), "candidate_id_sha256": _id_hash(part.candidate_id.astype(str)), "identical_to_c0": True})
    observed_identity_frame = pd.DataFrame(observed_identity)
    identity_compare = evaluation_ids.merge(observed_identity_frame, on=["split", "fold", "arm"], how="outer", suffixes=("__stored", "__actual"), indicator=True)
    checks.append(_check(
        "stored_candidate_id_ledger_matches_oof_rows",
        identity_compare._merge.eq("both").all()
        and identity_compare.rows__stored.eq(identity_compare.rows__actual).all()
        and identity_compare.candidate_id_sha256__stored.eq(identity_compare.candidate_id_sha256__actual).all()
        and identity_compare.identical_to_c0__stored.astype(bool).all(),
        "stored per-arm/fold candidate hashes and row counts exactly match OOF predictions", rows=len(identity_compare),
    ))
    per_fold_arms = predictions.groupby(["split", "fold"], sort=True).arm.agg(lambda values: tuple(sorted(values.unique()))).reset_index(name="arms")
    per_fold_arms["identical_candidate_order"] = per_fold_arms.apply(lambda row: all(
        tuple(predictions.loc[(predictions.split.eq(row.split)) & (predictions.fold.eq(row.fold)) & (predictions.arm.eq(arm)), "candidate_id"].astype(str))
        == tuple(predictions.loc[(predictions.split.eq(row.split)) & (predictions.fold.eq(row.fold)) & (predictions.arm.eq("C0")), "candidate_id"].astype(str))
        for arm in EXPECTED_ARMS
    ), axis=1)
    checks.append(_check(
        "all_compared_arms_use_identical_candidate_ids",
        per_fold_arms.arms.map(lambda value: tuple(value) == EXPECTED_ARMS).all() and per_fold_arms.identical_candidate_order.all(),
        "C0/C1/C2/C3/C6/C8 candidate order and IDs are equal within every split/fold", rows=len(per_fold_arms),
    ))

    _required(stability, (
        "arm", "side", "split", "fold", "fold_start_utc", "purge_embargo_hours", "train_decision_ts_max",
        "train_label_available_ts_max", "final_oos_labels_used", "base_features", "incremental_selected", "model_features",
    ), source="Stage-C fold stability")
    fold_start = pd.to_datetime(stability.fold_start_utc, utc=True, errors="coerce")
    train_decision = pd.to_datetime(stability.train_decision_ts_max, utc=True, errors="coerce")
    train_available = pd.to_datetime(stability.train_label_available_ts_max, utc=True, errors="coerce")
    stable_oof = (
        fold_start.notna().all() and train_decision.notna().all() and train_available.notna().all()
        and pd.to_numeric(stability.purge_embargo_hours, errors="coerce").eq(12).all()
        and train_decision.lt(fold_start - pd.Timedelta(hours=12)).all()
        and train_available.lt(fold_start).all()
        and ~stability.final_oos_labels_used.astype(bool).any()
    )
    checks.append(_check(
        "folds_have_strict_h12_purge_and_no_final_selection",
        stable_oof,
        "every stored fit ends before fold_start−12h; labels resolve before fold_start; final OOS labels were not selected on", rows=len(stability),
    ))
    feature_names = _json_feature_tokens(stability)
    leakage = sorted({name for name in feature_names if any(token in name.lower() for token in TARGET_LEAKAGE_TOKENS)})
    checks.append(_check(
        "target_columns_never_enter_feature_matrix",
        not leakage,
        f"target-like feature names={leakage}", rows=len(feature_names),
    ))
    manifest_target_ok = manifest.get("target") == PRIMARY_TARGET and manifest.get("population") == "exact H0 clear-first support only"
    checks.append(_check(
        "manifest_declares_only_primary_h0_conditional_target",
        manifest_target_ok,
        "manifest must declare retain_h0_given_clear on exact H0 clear-first support only", rows=None,
    ))

    coverage = oof.groupby(["split", "fold", "arm", "side", "month"], sort=True).agg(
        rows=("candidate_id", "size"), retention_prevalence=("label", "mean"), exact_h12_net_bps=("exact_h12_net_bps", "mean")
    ).reset_index()
    readiness = pd.DataFrame(checks)
    return readiness, coverage


def _verify_manifest_hashes(manifest: dict[str, Any], *, paths: dict[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    inputs = manifest.get("inputs", {})
    for label, path in paths.items():
        matching = [value for key, value in inputs.items() if str(key).endswith(str(path.relative_to(ROOT))) or Path(str(key)).name == path.name]
        declared = matching[0] if len(matching) == 1 else None
        records.append(_check(
            f"manifest_input_hash__{label}", declared == _sha256(path),
            f"declared={declared}; actual={_sha256(path)}", rows=None,
        ))
    for name, declared in manifest.get("outputs", {}).items():
        path = STAGE_C_V4 / name
        records.append(_check(
            f"manifest_output_hash__{name}", path.is_file() and declared == _sha256(path),
            f"declared={declared}; actual={_sha256(path) if path.is_file() else None}", rows=None,
        ))
    return records


def run(*, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Create a new, immutable Stage-C primary-target readiness artifact."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    manifest_path = STAGE_C_V4 / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness, coverage = audit_frames(
        panel=pd.read_parquet(PANEL),
        persistence=pd.read_parquet(PERSISTENCE),
        events=pd.read_parquet(EVENTS),
        alignment=pd.read_parquet(ALIGNMENT),
        predictions=pd.read_parquet(STAGE_C_V4 / "retention_conditional_oof_predictions.parquet"),
        evaluation_ids=pd.read_parquet(STAGE_C_V4 / "retention_evaluation_candidate_ids.parquet"),
        stability=pd.read_parquet(STAGE_C_V4 / "retention_feature_stability.parquet"),
        manifest=manifest,
    )
    hashes = _verify_manifest_hashes(manifest, paths={"panel": PANEL, "persistence": PERSISTENCE, "events": EVENTS, "alignment": ALIGNMENT})
    readiness = pd.concat([readiness, pd.DataFrame(hashes)], ignore_index=True)
    status = "STAGE_C_V4_PRIMARY_TARGET_CONTRACT_VERIFIED" if readiness.status.eq("PASS").all() else "BLOCKED_STAGE_C_V4_PRIMARY_TARGET_CONTRACT"
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        readiness.to_parquet(stage / "primary_target_contract_readiness.parquet", index=False, compression="zstd")
        coverage.to_parquet(stage / "primary_target_oof_fold_coverage.parquet", index=False, compression="zstd")
        report = [
            "# Stage-C v4 primary target contract audit", "",
            f"Terminal status: **{status}**.", "",
            "The audit recomputes `retain_h0_given_clear` from frozen H0 clear-first events and exact H12 net. It is a readiness audit only: no model, target, policy or promotion is changed.", "",
            f"- Checks passed: {int(readiness.status.eq('PASS').sum())} / {len(readiness)}",
            f"- OOF rows checked: {int(coverage.rows.sum()):,}",
            f"- OOF arms: {', '.join(EXPECTED_ARMS)}",
        ]
        (stage / "primary_target_contract_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        outputs = {name: _sha256(stage / name) for name in ("primary_target_contract_readiness.parquet", "primary_target_oof_fold_coverage.parquet", "primary_target_contract_audit.md")}
        artifact_manifest = {
            "schema": "stage_c_primary_target_contract_audit_v1",
            "status": status,
            "promotion_eligible": False,
            "primary_target": PRIMARY_TARGET,
            "inputs": {
                "stage_c_panel": {"path": str(PANEL), "sha256": _sha256(PANEL)},
                "frozen_persistence": {"path": str(PERSISTENCE), "sha256": _sha256(PERSISTENCE)},
                "frozen_events": {"path": str(EVENTS), "sha256": _sha256(EVENTS)},
                "frozen_alignment": {"path": str(ALIGNMENT), "sha256": _sha256(ALIGNMENT)},
                "stage_c_v4_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            },
            "checks": {"total": int(len(readiness)), "passed": int(readiness.status.eq("PASS").sum()), "blocked": int(readiness.status.ne("PASS").sum())},
            "outputs": outputs,
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        }
        (stage / "run_manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return artifact_manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
