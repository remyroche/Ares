#!/usr/bin/env python3
"""Strict July-2025 common30 regime-context raw-score extension.

The sealed July bridge supplies the *only* evaluation cohort.  Each side-local
context model is fit on the final v3 blocked-OOF ledger using outcomes whose
execution labels resolved before 2025-07-01.  Regime and transition values are
the sealed causal hourly sidecars; one-minute data remains only inside the
already-materialised execution labels.  This is a raw-score diagnostic, not a
replacement EV map and not a promotion experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_final_identical_row_regime_stack_gam_ablation as final

SCHEMA = "july2025_common30_regime_context_raw_score_extension_v1"
SIDECARES = ROOT / "data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1/manifest.json"
HISTORY = ROOT / "data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet"
JULY = ROOT / "data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1"
OUT = ROOT / "data_perp/artifacts/july2025_common30_regime_context_raw_score_extension_20260730_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
TARGET, ALPHA, RESIDUAL = final.TARGET, final.ALPHA, final.RESIDUAL
START = pd.Timestamp("2025-07-01T00:00:00Z")
END = pd.Timestamp("2025-08-01T00:00:00Z")
TOP = .10


class ExtensionError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _sealed_july(root: Path) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    manifest_path, marker, contract_path, prediction = root / "manifest.json", root / "manifest.sha256", root / "bridge_contract.json", root / "oof_predictions.parquet"
    if not all(path.is_file() for path in (manifest_path, marker, contract_path, prediction)):
        raise ExtensionError("sealed July bridge files are missing")
    if marker.read_text().split(maxsplit=1)[0] != sha(manifest_path):
        raise ExtensionError("July bridge manifest checksum is invalid")
    manifest, contract = json.loads(manifest_path.read_text()), json.loads(contract_path.read_text())
    if manifest.get("schema") != "july2025_common30_final_base_residual_oof_bridge_v1" or not str(manifest.get("manifest_status", "")).startswith("SEALED") or contract.get("status") != "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION":
        raise ExtensionError("requires the sealed common30 July blocked-OOF bridge")
    # The original envelope records a superseded prediction checksum, whereas
    # its immutable bridge contract records the completed residual-stage file.
    # Bind to that exact contract checksum rather than silently accepting an
    # inconsistent envelope hash; retain both source hashes in our manifest.
    expected = contract.get("outputs", {}).get(prediction.name)
    if expected != sha(prediction):
        raise ExtensionError("July bridge prediction checksum mismatch")
    columns = [*IDENTITY, TARGET, ALPHA, RESIDUAL, "execution_label_end_utc", "residual_is_oof"]
    frame = pd.read_parquet(prediction, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["execution_label_end_utc"] = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
    if len(frame) != 44_640 or frame.duplicated(IDENTITY).any() or not frame.residual_is_oof.all():
        raise ExtensionError("July bridge does not prove exact residual OOF common30 coverage")
    if not frame.__ts__.between(START, END - pd.Timedelta(hours=1)).all() or not frame.execution_label_end_utc.gt(frame.__ts__).all():
        raise ExtensionError("July bridge dates or label availability are invalid")
    if (frame.__ts__.astype("int64") % pd.Timedelta(hours=1).value != 0).any() or frame[[TARGET, ALPHA, RESIDUAL]].isna().any().any():
        raise ExtensionError("July bridge is not a complete hourly labelled score panel")
    return frame, manifest, prediction, contract_path


def _join_july(scores: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    before = scores.loc[:, IDENTITY].sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    result = scores.merge(context, left_on="__ts__", right_on="source_utc", how="left", validate="many_to_one", sort=False)
    after = result.loc[:, IDENTITY].sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if len(result) != len(scores) or not before.equals(after) or result.source_utc.isna().any():
        raise ExtensionError("hourly context join changed or lost July candidate identity")
    required = [*final.REGIME, *final.TRANSITION]
    available = result[required].notna().all(axis=1) & result.bocpd_regime_available.astype(bool) & result.lgbm_transition_available.astype(bool)
    if not available.all():
        raise ExtensionError("July bridge has unavailable regime/transition context")
    for suffix in ("bocpd", "lgbm"):
        if not result[f"provenance_partition_{suffix}"].eq("blocked_oof_2022_2025").all():
            raise ExtensionError(f"July {suffix} context is not blocked OOF")
        train_end = pd.to_datetime(result[f"train_end_exclusive_utc_{suffix}"], utc=True, errors="raise")
        label_end = pd.to_datetime(result[f"fit_label_resolution_max_utc_{suffix}"], utc=True, errors="raise")
        if train_end.isna().any() or label_end.isna().any() or not label_end.lt(train_end).all() or not train_end.eq(START).all():
            raise ExtensionError(f"July {suffix} context is not a strict pre-July fit")
    return result.drop(columns="source_utc")


def _rank(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    return float(a.loc[valid].corr(b.loc[valid], method="spearman")) if valid.sum() >= 3 else float("nan")


def _score_metrics(frame: pd.DataFrame, arm: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["raw_score", "candidate_id"], ascending=[False, True], kind="stable")
    chosen = set(ordered.head(max(1, math.ceil(len(ordered) * TOP))).candidate_id)
    work = frame.copy()
    work["selected_global_top10_raw"] = work.candidate_id.isin(chosen)
    picked = work.loc[work.selected_global_top10_raw]
    summary = {
        "arm": arm, "candidate_rows": int(len(work)), "top10_rows": int(len(picked)),
        "raw_execution_rank_ic": _rank(work.raw_score, work[TARGET]),
        "raw_alpha_rank_ic": _rank(work.raw_score, work[ALPHA]),
        "top10_net_ev": float(picked[TARGET].mean()),
        "top10_hit_rate": float(picked[TARGET].gt(0).mean()),
        "top10_long_share": float(picked.side_name.eq("long").mean()),
    }
    periods = []
    for kind, key in (("week", work.__ts__.dt.strftime("%G-W%V")), ("month", work.__ts__.dt.strftime("%Y-%m"))):
        for period, local in work.groupby(key, observed=True, sort=True):
            selection = local.loc[local.selected_global_top10_raw]
            periods.append({"arm": arm, "period_type": kind, "period": period, "candidate_rows": int(len(local)), "global_selected_rows": int(len(selection)), "raw_execution_rank_ic": _rank(local.raw_score, local[TARGET]), "raw_alpha_rank_ic": _rank(local.raw_score, local[ALPHA]), "mean_net_ev": float(selection[TARGET].mean()), "hit_rate": float(selection[TARGET].gt(0).mean())})
    sides = []
    for side, local in work.groupby("side_name", observed=True, sort=True):
        selection = local.loc[local.selected_global_top10_raw]
        sides.append({"arm": arm, "side_name": side, "candidate_rows": int(len(local)), "global_selected_rows": int(len(selection)), "raw_execution_rank_ic": _rank(local.raw_score, local[TARGET]), "raw_alpha_rank_ic": _rank(local.raw_score, local[ALPHA]), "top10_net_ev": float(selection[TARGET].mean()), "top10_hit_rate": float(selection[TARGET].gt(0).mean())})
    return summary, pd.DataFrame(periods), pd.DataFrame(sides), work


def run(*, sidecar_manifest: Path = SIDECARES, historical_scores: Path = HISTORY, july_root: Path = JULY, output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise ExtensionError(f"immutable output already exists: {output}")
    sidecar_manifest = Path(sidecar_manifest)
    _, regime_path, transition_path = final._load_manifest(sidecar_manifest)
    context = final._hourly_context(regime_path, transition_path)
    historical = final._verified_scores(Path(historical_scores), role="historical")
    history = final._join(historical, context, role="historical")
    train = history.loc[history.execution_label_end_utc.lt(START)].copy()
    if train.empty or not train.execution_label_end_utc.lt(START).all():
        raise ExtensionError("context-model training labels are not strictly pre-July")
    july, july_manifest, july_prediction, july_contract = _sealed_july(Path(july_root))
    test = _join_july(july, context)
    # The fixed six arms are exactly the residual/GAM context placements from
    # final v3, plus a frozen residual raw-score control.  No July row enters a
    # fit, parameter choice, feature choice, or EV mapping.
    arms = [final.Arm("baseline_raw_residual", "baseline", "none", TARGET, "raw")]
    for family, placement in (("lgbm", "residual_trust"), ("gam", "additive_bounded_gam")):
        for kind in ("regime", "transition", "combined"):
            arms.append(final.Arm(f"{placement}_{kind}_raw", placement, kind, TARGET, family))
    all_scores: list[pd.DataFrame] = []
    fit_audit: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    periods: list[pd.DataFrame] = []
    sides: list[pd.DataFrame] = []
    for number, arm in enumerate(arms):
        pieces: list[pd.DataFrame] = []
        for side, local_test in test.groupby("side_name", observed=True, sort=True):
            local_train = train.loc[train.side_name.eq(side)].copy()
            if len(local_train) < 8:
                raise ExtensionError(f"{arm.name}/{side}: inadequate strict pre-July fit rows")
            raw, model = final._predict(local_train, local_test, arm, 20250700 + number * 31 + int(side == "short"))
            pieces.append(local_test.assign(arm=arm.name, raw_score=raw))
            fit_audit.append({"arm": arm.name, "placement": arm.placement, "context": arm.context, "family": arm.family, "side_name": side, "fit_rows": int(len(local_train)), "fit_label_resolution_max_utc": local_train.execution_label_end_utc.max(), "fit_labels_strictly_before_july_1": bool(local_train.execution_label_end_utc.lt(START).all()), "evaluation_rows": int(len(local_test)), "evaluation_start_utc": local_test.__ts__.min(), "evaluation_end_utc": local_test.__ts__.max(), **model})
        scored = pd.concat(pieces, ignore_index=True)
        summary, per, side, selected = _score_metrics(scored, arm.name)
        results.append(summary); periods.append(per); sides.append(side)
        all_scores.append(selected.loc[:, [*IDENTITY, "execution_label_end_utc", TARGET, ALPHA, RESIDUAL, "arm", "raw_score", "selected_global_top10_raw"]])
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.concat(all_scores, ignore_index=True).to_parquet(stage / "july_raw_context_scores.parquet", index=False)
        pd.DataFrame(results).to_csv(stage / "metrics_summary.csv", index=False)
        pd.concat(periods, ignore_index=True).to_parquet(stage / "period_metrics.parquet", index=False)
        pd.concat(sides, ignore_index=True).to_parquet(stage / "side_metrics.parquet", index=False)
        write_json(stage / "fit_audit.json", fit_audit)
        contract = {
            "candidate_cadence": "1h", "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
            "evaluation": "sealed exact July-2025 30-asset common-universe bridge; both base and residual scores are blocked OOF",
            "training": "side-local only; all execution labels resolve strictly before 2025-07-01T00:00:00Z",
            "context": "continuous semantic BOCPD regime and LGBM/BOCPD transition probabilities joined many-to-one at hourly decision time; raw state identities/posterior axes are prohibited",
            "arms": [arm.__dict__ for arm in arms],
            "selection": "global pooled top10 of raw scores solely for July diagnostic; no EV map is fit, refreshed, selected, or promoted",
            "scope": "common30 July extension; not population-identical to the wider 2026 forward ledger; no 2026 labels, outcomes, HPO, feature selection, or model selection are used",
            "source_sidecar_manifest_sha256": sha(sidecar_manifest), "source_historical_scores_sha256": sha(Path(historical_scores)), "source_july_manifest_sha256": sha(Path(july_root) / "manifest.json"), "source_july_bridge_contract_sha256": sha(july_contract), "source_july_prediction_sha256": sha(july_prediction),
        }
        write_json(stage / "contract.json", contract)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": SCHEMA, "status": "SEALED_STRICT_PREJULY_TRAINED_JULY_COMMON30_RAW_CONTEXT_EXTENSION_NON_PROMOTION", "promotion_eligible": False, "inputs": {str(sidecar_manifest.resolve()): sha(sidecar_manifest), str(Path(historical_scores).resolve()): sha(Path(historical_scores)), str((Path(july_root) / "manifest.json").resolve()): sha(Path(july_root) / "manifest.json"), str(july_contract.resolve()): sha(july_contract), str(july_prediction.resolve()): sha(july_prediction)}, "contract": contract, "outputs_sha256": {path.name: sha(path) for path in files}}
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
