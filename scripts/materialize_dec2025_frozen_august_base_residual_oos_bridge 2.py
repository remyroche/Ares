#!/usr/bin/env python3
"""Score December common30 with immutable pre-August base/residual fits.

This is a deliberately conservative frozen-pre-December bridge.  It reuses
the sealed base/residual pair fit before 2025-08-01, rather than refitting a
new base whose residual score distribution would no longer match its residual
learner.  December exact 1m labels are joined only after all scores exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_febapr2025_canonical_base_oof import IDENTITY, _load_contracts, _materialize_features, _sha256
from scripts.run_mayjun2025_canonical_residual_continuation import _feature_matrix

SOURCE = ROOT / "data_perp/artifacts/augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1"
CANDIDATES = ROOT / "data_perp/artifacts/dec2025_common30_exact1m_backfill_inputs_20260730_v1/candidates.parquet"
LABELS = ROOT / "data_perp/artifacts/dec2025_execution_ev_common30_exact1m_labels_20260730_v1/labels.parquet"
PROMOTION = ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
AE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
STORE = ROOT / "data_perp/features/20260711_070000"
OUT = ROOT / "data_perp/artifacts/dec2025_common30_frozen_august_base_residual_oos_bridge_20260730_v1"
CUT = pd.Timestamp("2025-08-01", tz="UTC")


class BridgeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return _sha256(path)


def _dump(path: Path, value: dict) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _candidates(side: str) -> pd.DataFrame:
    frame = pd.read_parquet(CANDIDATES)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame = frame.loc[frame.side_name.eq(side)].sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(frame) != 22_320 or frame.candidate_id.duplicated().any() or not frame["__ts__"].dt.strftime("%Y-%m").eq("2025-12").all():
        raise BridgeError(f"{side}: December hourly candidate scope is invalid")
    return frame


def _source_model(side: str, name: str) -> Path:
    path = SOURCE / "models" / side / name
    if not path.is_file():
        raise BridgeError(f"missing sealed source model: {path}")
    return path


def score_base(output: Path, side: str) -> None:
    destination = output / side / "base_oos_predictions.parquet"
    if destination.exists():
        return
    model = _load_pickle(_source_model(side, "base.pkl"))
    route = _load_contracts(PROMOTION, AE)[side]
    frame = _candidates(side)
    frame["__feature_symbol__"] = frame.candidate_id.astype(str).str.split("|", n=1).str[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    features, coverage = _materialize_features(frame, route, STORE, destination.parent / "base_oos_features.parquet")
    if coverage["exact_key_fraction"] != 1.0:
        raise BridgeError(f"{side}: exact PIT base feature coverage failure")
    out = frame.copy()
    out["base_oos_score"] = model.predict(features.loc[:, list(route["features"])])
    out["score_base_alpha"] = out["base_oos_score"]
    out["base_rank_timestamp_side"] = out.groupby("__ts__")["base_oos_score"].rank(method="first", ascending=False).astype(int)
    out["base_group_rows"] = out.groupby("__ts__")["candidate_id"].transform("size").astype(int)
    out["base_rank_pct_timestamp_side"] = out["base_rank_timestamp_side"] / out["base_group_rows"]
    out["base_score_fit_cutoff_utc"] = CUT
    out["base_is_oos"] = True
    out.to_parquet(destination, index=False, compression="zstd")
    _dump(destination.with_suffix(".json"), {"side": side, "rows": int(len(out)), "feature_coverage": coverage,
        "base_model_source": str(_source_model(side, "base.pkl")), "base_model_sha256": _sha(_source_model(side, "base.pkl")),
        "fit_label_cutoff": "native label resolution < 2025-08-01T00:00:00Z; therefore strictly before December",
        "no_december_target_or_execution_outcome_read": True})


def score_residual(output: Path, side: str) -> None:
    destination = output / side / "oos_predictions.parquet"
    if destination.exists():
        return
    base = pd.read_parquet(output / side / "base_oos_predictions.parquet")
    base["base_oof_score"] = base["base_oos_score"]
    pack = _load_pickle(_source_model(side, "residual.pkl"))
    features, coverage = _feature_matrix(base, side, STORE)
    out = base.copy()
    out["base_expected_ev"] = pack["iso"].predict(out["base_oos_score"])
    out["residual_delta_ev"] = pack["model"].predict(features)
    out["residual_expected_ev"] = out["base_expected_ev"] + float(pack["alpha"]) * out["residual_delta_ev"]
    out["score_residual_expected_ev"] = out["residual_expected_ev"]
    out["residual_score_fit_cutoff_utc"] = CUT
    out["residual_is_oos"] = True
    out.to_parquet(destination, index=False, compression="zstd")
    _dump(destination.with_suffix(".json"), {"side": side, "rows": int(len(out)), "feature_coverage": coverage,
        "residual_model_source": str(_source_model(side, "residual.pkl")), "residual_model_sha256": _sha(_source_model(side, "residual.pkl")),
        "fit_label_cutoff": "native label resolution < 2025-08-01T00:00:00Z; therefore strictly before December",
        "no_december_target_or_execution_outcome_read": True})


def seal(output: Path) -> None:
    scored = pd.concat([pd.read_parquet(output / side / "oos_predictions.parquet") for side in ("long", "short")], ignore_index=True)
    labels = pd.read_parquet(LABELS)
    for frame in (scored, labels):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    value_columns = [column for column in labels.columns if column not in IDENTITY]
    out = scored.merge(labels.loc[:, list(IDENTITY) + value_columns], on=list(IDENTITY), how="left", validate="one_to_one")
    if (len(out) != 44_640 or out.candidate_id.duplicated().any() or not out.base_is_oos.all() or not out.residual_is_oos.all()
            or out["execution_net_ev_12h"].isna().any()):
        raise BridgeError("sealed December score/label identity is incomplete")
    if not pd.to_datetime(out["execution_label_available_at"], utc=True).gt(CUT).all():
        raise BridgeError("December assessment label boundary is invalid")
    out = out.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    out.to_parquet(output / "oos_predictions.parquet", index=False, compression="zstd")
    source_audits = {f"{side}_{name}": json.loads((_source_model(side, name + ".json")).read_text())
                     for side in ("long", "short") for name in ("base_fit", "residual_fit")}
    for name, audit in source_audits.items():
        maximum = pd.Timestamp(audit["train_label_resolution_max_utc"], tz="UTC") if pd.Timestamp(audit["train_label_resolution_max_utc"]).tzinfo is None else pd.Timestamp(audit["train_label_resolution_max_utc"]).tz_convert("UTC")
        if not maximum < pd.Timestamp("2025-12-01", tz="UTC"):
            raise BridgeError(f"source audit violates pre-December boundary: {name}")
    manifest = {
        "schema": "dec2025_common30_frozen_august_base_residual_oos_bridge_v1",
        "status": "SEALED_COMMON30_FROZEN_PRE_DECEMBER_BASE_RESIDUAL_OOS_SCORE_BRIDGE_NON_PROMOTION",
        "promotion_eligible": False,
        "scope": "December 2025 exact common30 candidate population; strict 1h scoring with 1m nested execution labels only",
        "decision_cadence": "1h",
        "exact_replay_bar_cadence": "1m_labels_only",
        "score_fit": "immutable side-local base/residual pair originally fit using only native labels resolved before 2025-08-01; this is strictly pre-December but not an expanding-through-November refit",
        "assessment_only_labels": "December execution labels (including 2026-01-01 12:00 UTC resolution) are joined only after all frozen scores exist; they never enter fitting, mapping, calibration, HPO, or feature selection",
        "rows": int(len(out)), "by_side": out.side_name.value_counts().sort_index().to_dict(),
        "fit_cutoff_utc": str(CUT), "source_fit_audits": source_audits,
        "inputs_sha256": {"candidates": _sha(CANDIDATES), "labels_assessment_only": _sha(LABELS), "source_bridge_manifest": _sha(SOURCE / "manifest.json")},
        "outputs_sha256": {"oos_predictions.parquet": _sha(output / "oos_predictions.parquet")},
    }
    _dump(output / "bridge_contract.json", manifest)
    _dump(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(f"{_sha(output / 'manifest.json')}  manifest.json\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stage", required=True, choices=("score_base", "score_residual", "seal"))
    parser.add_argument("--side", choices=("long", "short"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "seal":
        if args.side:
            parser.error("--side is not valid for seal")
        seal(args.output)
    elif not args.side:
        parser.error("--side is required for scoring")
    elif args.stage == "score_base":
        score_base(args.output, args.side)
    else:
        score_residual(args.output, args.side)


if __name__ == "__main__":
    main()
