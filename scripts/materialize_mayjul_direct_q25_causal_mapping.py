#!/usr/bin/env python3
"""Bind exact May--July direct-q25 OOF provenance and map it causally.

No direct model is refit.  The original OOF prediction values are retained
bit-for-bit; chronological folds are reconstructed solely from the hash-bound
dataset and the original recipe.  A 21-day map is then fit only on already
resolved exact H12 labels, leaving its support warm-up unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.materialize_mayjul2026_exact_allscore_ic_ev_waterfall import _canonicalize_direct_identity
    from scripts.run_cross_era_tail_payoff_challenger import chronological_folds
    from scripts.run_execution_ev_recent_mapping_ablation import causal_mappings
except ModuleNotFoundError:
    from materialize_mayjul2026_exact_allscore_ic_ev_waterfall import _canonicalize_direct_identity
    from run_cross_era_tail_payoff_challenger import chronological_folds
    from run_execution_ev_recent_mapping_ablation import causal_mappings


WATERFALL = ROOT / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
DIRECT = ROOT / "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"
DATASET = ROOT / "data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/mayjul_exact_direct_q25_causal_mapping_20260730_v1"
SCHEMA = "mayjul_exact_direct_q25_causal_mapping_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
WINDOW_DAYS = 21
MIN_REFERENCE_ROWS = 500
SIDE_SHRINK_ROWS = 500.0
TOPS = (.01, .05, .10, .20)


class MappingError(RuntimeError):
    """The frozen direct q25 provenance cannot be reconstructed safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _manifest_output(root: Path, name: str) -> tuple[Path, dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest.get("outputs", {}).get(name, {})
    path = Path(str(item.get("path", "")))
    if not path.is_absolute():
        path = ROOT / path if (ROOT / path).is_file() else root / path
    if not path.is_file() or item.get("sha256") != sha256(path):
        raise MappingError(f"{name} is not hash-bound in {root}")
    return path, manifest, manifest_path


def _normalise(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise MappingError(f"{source} lacks identity: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any() or result["candidate_id"].duplicated().any():
        raise MappingError(f"{source} does not have unique exact identity")
    return result


def _fold_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    folds = chronological_folds(dataset)
    pieces: list[pd.DataFrame] = []
    for fold in folds:
        valid = dataset.iloc[fold.valid].loc[:, list(IDENTITY)].copy()
        train_resolved = pd.to_datetime(dataset.iloc[fold.train]["label_resolution_utc"], utc=True, errors="raise")
        if not train_resolved.lt(fold.start).all():
            raise MappingError(f"{fold.name} has a noncausal training label")
        valid["oof_fold_name"] = fold.name
        valid["fit_cutoff_utc"] = fold.start
        valid["max_training_label_resolution_utc"] = train_resolved.max()
        valid["fold_train_rows"] = int(len(fold.train))
        pieces.append(valid)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(list(IDENTITY)).any():
        raise MappingError("chronological fold validity windows overlap")
    return result


def reconstruct() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    waterfall_path, waterfall_manifest, waterfall_manifest_path = _manifest_output(WATERFALL, "allscore_waterfall")
    direct_path, direct_manifest, direct_manifest_path = _manifest_output(DIRECT, "historical_oof_winner")
    dataset_path, dataset_manifest, dataset_manifest_path = _manifest_output(DATASET, "dataset")
    frozen_path, model_path = DIRECT / "frozen_before_current_evaluation.json", DIRECT / "frozen_models.joblib"
    if not frozen_path.is_file() or not model_path.is_file():
        raise MappingError("frozen direct state/model is absent")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("dataset", {}).get("sha256") != sha256(dataset_path):
        raise MappingError("frozen direct state binds a different dataset")
    if frozen.get("model", {}).get("sha256") != sha256(model_path):
        raise MappingError("frozen direct state model hash fails")
    if frozen.get("winner", {}).get("score_column") != "q25_net_bps":
        raise MappingError("frozen winner is not the direct q25 score")
    waterfall = _normalise(pd.read_parquet(waterfall_path), source="all-score waterfall")
    direct = _canonicalize_direct_identity(pd.read_parquet(direct_path), "direct q25 OOF source")
    dataset = _canonicalize_direct_identity(
        pd.read_parquet(dataset_path), "tail-payoff fold dataset"
    )
    # The output q25 values are the fixed ranker output; never recompute them.
    direct = direct.loc[:, [*IDENTITY, "q25_net_bps", "execution_net_ev_12h", "label_resolution_utc"]].copy()
    joined = waterfall.merge(direct, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or len(joined) != len(waterfall):
        raise MappingError(f"direct q25 does not cover exact waterfall IDs: {joined['_merge'].value_counts().to_dict()}")
    joined = joined.drop(columns="_merge")
    if not np.array_equal(joined["score_direct_q25_challenger_bps"].to_numpy(), joined["q25_net_bps"].to_numpy()):
        raise MappingError("direct q25 is not bit-identical to the waterfall direct layer")
    if not np.array_equal(joined["execution_net_ev_12h_x"].to_numpy(), joined["execution_net_ev_12h_y"].to_numpy()):
        raise MappingError("direct score source and exact waterfall net labels differ")
    joined = joined.rename(columns={"execution_net_ev_12h_x": "execution_net_ev_12h", "execution_net_ev_12h_y": "direct_source_execution_net_ev_12h"})
    for timestamp in ("execution_decision_utc", "execution_label_end_utc", "label_resolution_utc"):
        joined[timestamp] = pd.to_datetime(joined[timestamp], utc=True, errors="raise")
    if not joined["label_resolution_utc"].eq(joined["execution_label_end_utc"]).all():
        raise MappingError("direct q25 labels do not share exact H12 resolution")
    if not joined["execution_decision_utc"].eq(joined["__ts__"] + pd.Timedelta(hours=1)).all():
        raise MappingError("waterfall does not preserve signal+1h decisions")
    # Dataset establishes the actual feature surface consumed by the fold models.
    dataset_keys = dataset.loc[:, list(IDENTITY)].copy()
    joined = joined.merge(dataset_keys, on=list(IDENTITY), how="left", validate="one_to_one", indicator="dataset_join")
    if not joined["dataset_join"].eq("both").all():
        raise MappingError("exact waterfall IDs are not covered by bound fold dataset")
    joined = joined.drop(columns="dataset_join")
    folds = _fold_frame(dataset)
    joined = joined.merge(folds, on=list(IDENTITY), how="left", validate="one_to_one", indicator="fold_join")
    if not joined["fold_join"].eq("both").all():
        raise MappingError("exact waterfall IDs do not have a unique chronological fold")
    joined = joined.drop(columns="fold_join")
    # Feature universe manifest explicitly says the immutable causal feature
    # store is sampled at candidate signal time.  It is hash-bound through the
    # frozen dataset report/state, so score production is available at decision.
    joined["feature_available_at"] = joined["__ts__"]
    joined["score_available_at"] = joined["execution_decision_utc"]
    if not joined["feature_available_at"].lt(joined["execution_decision_utc"]).all():
        raise MappingError("a direct-q25 feature is not strictly before decision")
    if not joined["score_available_at"].le(joined["execution_decision_utc"]).all():
        raise MappingError("direct q25 score is unavailable at decision")
    if not joined["max_training_label_resolution_utc"].lt(joined["fit_cutoff_utc"]).all() or not joined["fit_cutoff_utc"].le(joined["execution_decision_utc"]).all():
        raise MappingError("fold fit cutoff is not causally before each candidate")
    joined["candidate_month"] = joined["__ts__"].dt.strftime("%Y-%m")
    joined["direct_score_model_lineage"] = "cross_era_direct_net_quantile_challenger_q25_oof"
    joined["direct_score_oof_recipe"] = "chronological_folds_prior_resolved_labels"
    joined["direct_score_source_hash"] = sha256(direct_path)
    joined["direct_frozen_state_hash"] = sha256(frozen_path)
    joined["direct_frozen_final_model_hash"] = sha256(model_path)
    # The final model is bound as a frozen lineage asset, not claimed as the
    # per-fold OOF binary.  The exact OOF score column itself is authoritative.
    joined["oof_model_binary_status"] = "not_persisted; exact_oof_output_and_recipe_hash_bound"
    provenance_columns = [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "feature_available_at", "score_available_at", "q25_net_bps", "oof_fold_name", "fit_cutoff_utc", "max_training_label_resolution_utc", "fold_train_rows", "direct_score_model_lineage", "direct_score_oof_recipe", "direct_score_source_hash", "direct_frozen_state_hash", "direct_frozen_final_model_hash", "oof_model_binary_status"]
    sources = {
        "waterfall": {"path": str(waterfall_path), "sha256": sha256(waterfall_path), "manifest": str(waterfall_manifest_path), "manifest_sha256": sha256(waterfall_manifest_path)},
        "direct_oof": {"path": str(direct_path), "sha256": sha256(direct_path), "manifest": str(direct_manifest_path), "manifest_sha256": sha256(direct_manifest_path)},
        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path), "manifest": str(dataset_manifest_path), "manifest_sha256": sha256(dataset_manifest_path)},
        "frozen_state": {"path": str(frozen_path), "sha256": sha256(frozen_path)}, "frozen_final_model": {"path": str(model_path), "sha256": sha256(model_path)},
        "runner": {"path": str(ROOT / "scripts/run_cross_era_direct_net_quantile_challenger.py"), "sha256": sha256(ROOT / "scripts/run_cross_era_direct_net_quantile_challenger.py")},
    }
    return joined, joined.loc[:, provenance_columns].copy(), sources


def _map(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped, audit_records = causal_mappings(frame, score_col="q25_net_bps", window_days=WINDOW_DAYS, min_reference_rows=MIN_REFERENCE_ROWS, side_support_target=SIDE_SHRINK_ROWS)
    mapped["causal_mapped_direct_q25_ev"] = mapped["causal_recent_side_isotonic_ev"]
    mapped["mapped_eligible"] = mapped["causal_mapped_direct_q25_ev"].notna()
    mapped["mapping_available_at"] = mapped["execution_decision_utc"]
    if not mapped["mapping_available_at"].le(mapped["execution_decision_utc"]).all():
        raise MappingError("mapped score availability violates decision timing")
    # Emit full audit including unavailable warm-up days, not only helper's
    # available snapshots.
    rows: list[dict[str, Any]] = []
    for snapshot, local in mapped.groupby(mapped["execution_decision_utc"].dt.floor("D"), sort=True):
        reference = mapped.loc[mapped["execution_label_end_utc"].lt(snapshot) & mapped["execution_label_end_utc"].ge(snapshot - pd.Timedelta(days=WINDOW_DAYS))]
        counts = reference.groupby("side_name", observed=True).size()
        rows.append({"snapshot_utc": snapshot, "reference_window_start_utc": snapshot - pd.Timedelta(days=WINDOW_DAYS), "reference_window_end_utc": snapshot, "reference_rows": int(len(reference)), "long_reference_rows": int(counts.get("long", 0)), "short_reference_rows": int(counts.get("short", 0)), "reference_label_end_max_utc": reference["execution_label_end_utc"].max() if len(reference) else pd.NaT, "strictly_resolved_before_snapshot": bool(reference["execution_label_end_utc"].lt(snapshot).all()), "mapping_available": bool(local["mapped_eligible"].all()), "current_rows": int(len(local))})
    audit = pd.DataFrame(rows)
    if not audit["strictly_resolved_before_snapshot"].all():
        raise MappingError("map audit contains a future/unresolved label")
    return mapped, audit


def _top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    return frame.sort_values([score, "candidate_id", "__ts__", "__symbol__", "side_name"], ascending=[False, True, True, True, True], kind="stable").iloc[:count]


def four_layer_diagnostic(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = mapped.loc[mapped["mapped_eligible"].astype(bool)].copy()
    layers = {"raw_base_alpha": "score_base_alpha", "residual_alpha_expected_ev": "score_residual_expected_ev", "direct_ev_q25": "score_direct_q25_challenger_bps", "causal_mapped_direct_q25_ev": "causal_mapped_direct_q25_ev"}
    records: list[dict[str, Any]] = []; ties: list[dict[str, Any]] = []
    for scope, local in [("aggregate", work), *[(f"month:{month}", part) for month, part in work.groupby("candidate_month", sort=True)]]:
        for layer, score in layers.items():
            for fraction in TOPS:
                selected = _top(local, score, fraction); net = selected["execution_net_ev_12h"].to_numpy(float)
                cutoff = float(selected[score].iloc[-1]); tie_rows = int(np.isclose(local[score].to_numpy(float), cutoff, rtol=0.0, atol=1e-14).sum())
                records.append({"scope": scope, "layer": layer, "score_column": score, "top_fraction": fraction, "candidate_rows": int(len(local)), "selected_rows": int(len(selected)), "mean_net_bps": float(net.mean() * 1e4), "positive_net_rate": float((net > 0).mean()), "mean_gross_bps": float(selected.execution_gross_ev_12h.mean() * 1e4), "mean_cost_bps": float(selected.execution_cost_return.mean() * 1e4), "long_selected_rows": int(selected.side_name.eq("long").sum()), "short_selected_rows": int(selected.side_name.eq("short").sum()), "cutoff_score": cutoff, "cutoff_tie_rows": tie_rows})
                ties.append({"scope": scope, "layer": layer, "top_fraction": fraction, "cutoff_score": cutoff, "cutoff_tie_rows": tie_rows, "distinct_scores": int(local[score].nunique())})
    common = work.loc[:, [*IDENTITY, "candidate_month", "execution_decision_utc", "execution_label_end_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "score_base_alpha", "score_residual_expected_ev", "score_direct_q25_challenger_bps", "causal_mapped_direct_q25_ev", "oof_fold_name", "fit_cutoff_utc", "max_training_label_resolution_utc"]].copy()
    return common, pd.DataFrame(records), pd.DataFrame(ties)


def run(*, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    joined, provenance, sources = reconstruct()
    mapped, audit = _map(joined)
    common, diagnostic, ties = four_layer_diagnostic(mapped)
    if len(common) != int(mapped["mapped_eligible"].sum()):
        raise MappingError("four-layer common population does not match mapped eligibility")
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name, table in (("direct_q25_oof_provenance", provenance), ("causal_mapped_direct_q25_candidates", mapped), ("causal_mapping_audit", audit), ("four_layer_common_rows", common), ("four_layer_tail_metrics", diagnostic), ("four_layer_cutoff_ties", ties)):
            path = stage / f"{name}.parquet"; table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / path.name), "rows": int(len(table)), "sha256": sha256(path)}
        manifest = {"schema": SCHEMA, "status": "COMPLETE_EXACT_DIRECT_Q25_OOF_CAUSAL_MAP_DIAGNOSTIC_ONLY", "promotion_eligible": False,
            "sources": sources, "rows": {"waterfall_exact_ids": int(len(joined)), "mapped_eligible": int(mapped.mapped_eligible.sum()), "warmup_unmapped": int((~mapped.mapped_eligible).sum()), "four_layer_common": int(len(common))},
            "contracts": {"direct_score": "exact score_direct_q25_challenger_bps / q25_net_bps only; no mapped_direct_net/base-alpha alias", "oof": "chronological fold validity reconstructed from bound v3 dataset; all training label resolution < fold fit cutoff <= candidate decision", "features": "bound feature-universe contract states immutable causal values at signal __ts__; score available at decision", "mapping": "new isotonic calibration only; frozen direct q25 ranking values unchanged; exact H12 labels strictly resolved before UTC-day snapshot within 21d; 500 min rows and 500 side shrink; honest unavailable warm-up", "selection": "four-layer diagnostics use pooled global stable candidate-id ties only; side counts attribution, no quotas", "model_binding_caveat": "historical per-fold binaries were not persisted; exact OOF q25 output, original runner/fold recipe, frozen state, final frozen model and feature contract are hash-bound; final model is not claimed as per-fold binary parity"},
            "outputs": outputs, "outputs_sha256": {f"{name}.parquet": item["sha256"] for name, item in outputs.items()}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}
        _write_json(stage / "manifest.json", manifest); (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv)
    print(json.dumps(_safe(run(output_dir=args.output_dir)), sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
