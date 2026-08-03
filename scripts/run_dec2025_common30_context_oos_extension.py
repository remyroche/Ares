#!/usr/bin/env python3
"""Fixed pre-August regime/transition context scoring on December common30.

The test ledger carries exact labels solely for post-score economics.  Each
context arm is fitted side-locally from the historical blocked-OOF panel plus
July blocked OOF rows whose labels resolve before 2025-08-01.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_augnov2025_common30_context_oos_extension as preaug
from scripts import run_final_identical_row_regime_stack_gam_ablation as final

IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
TARGET = final.TARGET
RESIDUAL = final.RESIDUAL
CUT = pd.Timestamp("2025-08-01", tz="UTC")
SIDECARES = ROOT / "data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1/manifest.json"
HISTORY = ROOT / "data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet"
JULY = ROOT / "data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1"
DECEMBER = ROOT / "data_perp/artifacts/dec2025_common30_frozen_august_base_residual_oos_bridge_20260730_v1"
OUT = ROOT / "data_perp/artifacts/dec2025_common30_fixed_preaug_context_oos_extension_20260730_v1"


class ContextError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _sealed(root: Path, schema: str, status: str) -> dict:
    manifest = root / "manifest.json"
    marker = root / "manifest.sha256"
    if not manifest.is_file() or not marker.is_file() or marker.read_text().split()[0] != _sha(manifest):
        raise ContextError(f"unsealed source: {root}")
    value = json.loads(manifest.read_text())
    if value.get("schema") != schema or value.get("status") != status:
        raise ContextError(f"wrong sealed source: {root}")
    return value


def run(*, sidecars: Path = SIDECARES, historical: Path = HISTORY, july_root: Path = JULY,
        december_root: Path = DECEMBER, output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise ContextError(f"refusing to overwrite {output}")
    _, regime, transition = final._load_manifest(Path(sidecars))
    context = final._hourly_context(regime, transition)
    historical_panel = final._join(final._verified_scores(Path(historical), role="historical"), context, role="historical")
    july_manifest = _sealed(Path(july_root), "july2025_common30_final_base_residual_oof_bridge_v1", "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION")
    july_path = Path(july_root) / "oof_predictions.parquet"
    july_contract = json.loads((Path(july_root) / "bridge_contract.json").read_text())
    if (july_contract.get("schema") != "july2025_common30_final_base_residual_oof_bridge_v1"
            or july_contract.get("status") != "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION"
            or july_contract.get("outputs", {}).get(july_path.name) != _sha(july_path)):
        raise ContextError("July bridge companion contract is detached")
    july = pd.read_parquet(july_path, columns=[*IDENTITY, TARGET, RESIDUAL, "execution_label_end_utc", "residual_is_oof"])
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True, errors="raise")
    july["execution_label_end_utc"] = pd.to_datetime(july["execution_label_end_utc"], utc=True, errors="raise")
    if len(july) != 44_640 or not july["residual_is_oof"].all():
        raise ContextError("July OOF bridge is incomplete")
    july = preaug.join_context(july, context, "July frozen context training")
    december_manifest = _sealed(Path(december_root), "dec2025_common30_frozen_august_base_residual_oos_bridge_v1", "SEALED_COMMON30_FROZEN_PRE_DECEMBER_BASE_RESIDUAL_OOS_SCORE_BRIDGE_NON_PROMOTION")
    december_path = Path(december_root) / "oos_predictions.parquet"
    if december_manifest.get("outputs_sha256", {}).get(december_path.name) != _sha(december_path):
        raise ContextError("December bridge checksum is detached")
    test_all = pd.read_parquet(december_path)
    test_all["__ts__"] = pd.to_datetime(test_all["__ts__"], utc=True, errors="raise")
    test_all["execution_label_end_utc"] = pd.to_datetime(test_all["execution_label_end_utc"], utc=True, errors="raise")
    if (len(test_all) != 44_640 or test_all.candidate_id.duplicated().any() or not test_all["residual_is_oos"].all()
            or not test_all["__ts__"].dt.strftime("%Y-%m").eq("2025-12").all()):
        raise ContextError("December frozen score bridge is incomplete")
    context_hours = set(pd.to_datetime(context["source_utc"], utc=True, errors="raise"))
    absent = test_all.loc[~test_all["__ts__"].isin(context_hours), list(IDENTITY)].copy()
    absent["absence_reason"] = "authoritative_regime_transition_hourly_context_not_materialised"
    absent["context_action"] = "excluded_from_context_sensitivity_without_imputation"
    test = test_all.loc[test_all["__ts__"].isin(context_hours)].copy()
    if len(absent) != 720 or absent["__ts__"].nunique() != 12 or len(test) != 43_920:
        raise ContextError("unexpected December context-coverage boundary; do not infer or fill it")
    test = preaug.join_context(test, context, "December frozen context assessment common subset")
    train = pd.concat([historical_panel, july], ignore_index=True, sort=False)
    train = train.loc[pd.to_datetime(train["execution_label_end_utc"], utc=True).lt(CUT)].copy()
    if train.empty or not pd.to_datetime(train["execution_label_end_utc"], utc=True).lt(CUT).all() or not train["__ts__"].lt(CUT).all():
        raise ContextError("context fit includes unresolved post-cutoff outcomes")
    arms = [final.Arm("frozen_residual", "baseline", "none", TARGET, "raw")]
    for family, placement in (("lgbm", "residual_trust"), ("gam", "additive_bounded_gam")):
        for context_name in ("regime", "transition", "combined"):
            arms.append(final.Arm(f"{placement}_{context_name}", placement, context_name, TARGET, family))
    summaries: list[dict] = []
    periods: list[pd.DataFrame] = []
    sides: list[pd.DataFrame] = []
    scores: list[pd.DataFrame] = []
    audits: list[dict] = []
    for ordinal, arm in enumerate(arms):
        predicted = []
        for side, local in test.groupby("side_name", observed=True, sort=True):
            fit = train.loc[train.side_name.eq(side)].copy()
            raw, metadata = final._predict(fit, local, arm, 20251200 + ordinal * 31 + (side == "short"))
            predicted.append(local.assign(arm=arm.name, raw_score=raw))
            audits.append({"arm": arm.name, "side_name": side, "fit_rows": int(len(fit)),
                           "fit_label_end_max": pd.to_datetime(fit["execution_label_end_utc"], utc=True).max(),
                           "fit_labels_before_aug": bool(pd.to_datetime(fit["execution_label_end_utc"], utc=True).lt(CUT).all()),
                           "evaluation_rows": int(len(local)), "evaluation_start": local["__ts__"].min(),
                           "evaluation_end": local["__ts__"].max(), "december_outcomes_not_in_fit": True, **metadata})
        scored = pd.concat(predicted, ignore_index=True)
        summary, period, side_table, scored_book = preaug.evaluate(scored, arm.name)
        summaries.append(summary); periods.append(period); sides.append(side_table)
        scores.append(scored_book.loc[:, [*IDENTITY, "execution_label_end_utc", TARGET, "execution_gross_ev_12h", "execution_cost_return", "arm", "raw_score", "selected_global_top10"]])
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.DataFrame(summaries).to_csv(stage / "metrics_summary.csv", index=False)
        pd.concat(periods, ignore_index=True).to_parquet(stage / "period_metrics.parquet", index=False)
        pd.concat(sides, ignore_index=True).to_parquet(stage / "side_metrics.parquet", index=False)
        pd.concat(scores, ignore_index=True).to_parquet(stage / "december_raw_context_scores.parquet", index=False, compression="zstd")
        absent.to_parquet(stage / "context_unavailable_candidates.parquet", index=False, compression="zstd")
        _dump(stage / "fit_audit.json", audits)
        contract = {
            "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
            "arms": [arm.__dict__ for arm in arms],
            "training": "side-local fixed context architectures trained only on compatible historical blocked-OOF plus July common30 blocked-OOF rows with execution labels resolved strictly before 2025-08-01",
            "assessment": "sealed December common30 frozen pre-December base+residual scores restricted to one exact 43,920-row common-context subset; one pooled global raw-score top10 per arm; baseline is re-evaluated on this same subset; period tables only decompose fixed membership",
            "assessment_label_boundary": "December labels, including 2026-01-01 12:00 UTC availability, are used only after context scores are fixed for economics",
            "no_hpo_or_feature_selection": True, "no_mapping_or_promotion": True,
            "context_coverage": {"input_rows": 44_640, "context_scored_rows": 43_920, "excluded_rows": 720, "excluded_hourly_timestamps": 12, "missingness_policy": "no fill/no forward fill/no reroute; excluded candidate IDs are emitted explicitly"},
            "scope_limitation": "common30 only; conservative source models are frozen pre-August, not an expanding-through-November refit; no full-December context arm claim is allowed until the 12 missing hourly sidecar timestamps are materialised",
        }
        _dump(stage / "contract.json", contract)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": "dec2025_common30_fixed_preaug_context_oos_extension_v1",
                    "status": "SEALED_FIXED_PREDEC_CONTEXT_OOS_EXTENSION_NON_PROMOTION", "promotion_eligible": False,
                    "inputs": {str(Path(path).resolve()): _sha(Path(path)) for path in (sidecars, historical, Path(july_root) / "manifest.json", Path(july_root) / "bridge_contract.json", Path(december_root) / "manifest.json")},
                    "contract": contract, "outputs_sha256": {path.name: _sha(path) for path in files}}
        _dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
