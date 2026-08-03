#!/usr/bin/env python3
"""Materialise the March--April exact-H12 common score ledger under the current map.

The three score arms are pre-existing frozen outputs: canonical base OOF,
canonical residual OOF, and the direct-q25 challenger OOF.  This runner never
refits them.  It reconstructs the direct score's chronological provenance from
the hash-bound fold dataset, then applies the current admission mapping:
21 UTC days, exact label resolution before each snapshot, at least 500 pooled
reference rows, and side isotonic shrinkage toward the pooled map at 500 rows.

This is intentionally a reused-March/April diagnostic.  It is not a promotion
test, a policy replay, or a model-selection exercise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_mayjul_direct_q25_causal_mapping import _fold_frame
from scripts.run_execution_ev_recent_mapping_ablation import causal_mappings


SCHEMA = "marapr2025_exact_h12_current_mapping_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
NET = "execution_net_ev_12h"
WINDOW_DAYS = 21
MIN_REFERENCE_ROWS = 500
SIDE_SUPPORT_TARGET = 500.0
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
SCORE_COLUMNS: Mapping[str, str] = {
    "base_alpha_raw": "score_base_alpha",
    "residual_expected_ev": "score_residual_expected_ev",
    "direct_q25_exact_h12": "score_direct_q25_ev_bps",
}
LEDGER_ROOT = ROOT / "data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1"
BASE_LEDGER = LEDGER_ROOT / "ledgers/canonical_base_exact1m_current_spread_cf.parquet"
RESIDUAL_LEDGER = LEDGER_ROOT / "ledgers/canonical_residual_exact1m_current_spread_cf.parquet"
DIRECT_ROOT = ROOT / "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"
DIRECT_OOF = DIRECT_ROOT / "historical_oof_winner.parquet"
DIRECT_FROZEN_STATE = DIRECT_ROOT / "frozen_before_current_evaluation.json"
DIRECT_FINAL_MODEL = DIRECT_ROOT / "frozen_models.joblib"
FOLD_DATASET_ROOT = ROOT / "data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"
FOLD_DATASET = FOLD_DATASET_ROOT / "cross_era_tail_payoff_dataset.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/marapr2025_exact_h12_current_mapping_20260730_v1"


class MaterializationError(RuntimeError):
    """The frozen historical inputs do not prove a safe common score ledger."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise MaterializationError(f"missing bound input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _normalise(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise MaterializationError(f"{name} lacks identity fields: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any() or result["candidate_id"].duplicated().any():
        raise MaterializationError(f"{name} has duplicate candidate identity")
    return result


def _ledger_record(manifest: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    resolved = str(path.resolve())
    for record in manifest.get("ledgers", []):
        if str(Path(str(record.get("path", ""))).resolve()) == resolved:
            if str(record.get("sha256")) != sha256(path):
                raise MaterializationError(f"ledger manifest hash changed: {path.name}")
            return record
    raise MaterializationError(f"ledger is absent from historical manifest: {path.name}")


def _output_record(root: Path, name: str) -> tuple[Path, Mapping[str, Any], Path]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("outputs", {}).get(name, {})
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or str(record.get("sha256")) != sha256(path):
        raise MaterializationError(f"output is not hash-bound: {root.name}/{name}")
    return path, manifest, manifest_path


def _require_exact_h12(frame: pd.DataFrame, *, name: str) -> None:
    required = {
        "execution_decision_utc", "execution_label_end_utc", "execution_gross_ev_12h",
        "execution_cost_return", NET, "exact_policy_parity", "label_horizon_hours",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MaterializationError(f"{name} lacks exact-H12 fields: {missing}")
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame["exact_policy_parity"].astype(bool).all() or not frame["label_horizon_hours"].eq(12).all():
        raise MaterializationError(f"{name} is not exact current-policy H12")
    if not frame["execution_decision_utc"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise MaterializationError(f"{name} does not preserve signal-plus-one-hour decision")
    if not frame["execution_label_end_utc"].eq(frame["execution_decision_utc"] + pd.Timedelta(hours=12)).all():
        raise MaterializationError(f"{name} does not preserve exact H12 resolution")
    if not np.allclose(
        frame["execution_gross_ev_12h"].to_numpy(float) - frame["execution_cost_return"].to_numpy(float),
        frame[NET].to_numpy(float), rtol=0.0, atol=1e-10,
    ):
        raise MaterializationError(f"{name} gross-cost-net reconciliation failed")


def reconstruct() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledger_manifest_path = LEDGER_ROOT / "manifest.json"
    ledger_manifest = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))
    if ledger_manifest.get("schema") != "historical_score_economics_conversion_ledgers_v1":
        raise MaterializationError("unexpected historical ledger schema")
    _ledger_record(ledger_manifest, BASE_LEDGER)
    _ledger_record(ledger_manifest, RESIDUAL_LEDGER)
    base = _normalise(pd.read_parquet(BASE_LEDGER), name="base ledger")
    residual = _normalise(pd.read_parquet(RESIDUAL_LEDGER), name="residual ledger")
    _require_exact_h12(base, name="base ledger")
    _require_exact_h12(residual, name="residual ledger")
    residual = residual.loc[residual["__ts__"].dt.strftime("%Y-%m").isin(("2025-03", "2025-04"))].copy()
    base = base.loc[base["candidate_id"].isin(residual["candidate_id"])].copy()
    if len(base) != len(residual) or len(residual) != 140_682:
        raise MaterializationError("base/residual common March-April coverage changed")
    base_small = base.loc[:, [*IDENTITY, NET, "execution_gross_ev_12h", "execution_cost_return", "score_base_alpha"]]
    common = residual.merge(base_small, on=list(IDENTITY), how="inner", validate="one_to_one", suffixes=("", "__base"))
    for column in (NET, "execution_gross_ev_12h", "execution_cost_return"):
        if not np.array_equal(common[column].to_numpy(), common[f"{column}__base"].to_numpy()):
            raise MaterializationError(f"base/residual exact label mismatch: {column}")
        common = common.drop(columns=f"{column}__base")
    direct_path, direct_manifest, direct_manifest_path = _output_record(DIRECT_ROOT, "historical_oof_winner")
    direct = _normalise(pd.read_parquet(direct_path), name="direct q25 OOF")
    direct = direct.loc[direct["era"].astype(str).eq("2025_feb_apr")].copy()
    direct["label_resolution_utc"] = pd.to_datetime(direct["label_resolution_utc"], utc=True, errors="raise")
    direct_small = direct.loc[:, [*IDENTITY, "q25_net_bps", NET, "label_resolution_utc"]]
    common = common.merge(direct_small, on=list(IDENTITY), how="inner", validate="one_to_one", suffixes=("", "__direct"))
    if len(common) != 140_682 or not np.array_equal(common[NET].to_numpy(), common[f"{NET}__direct"].to_numpy()):
        raise MaterializationError("direct q25 common identity or label parity changed")
    common = common.drop(columns=f"{NET}__direct")
    if not common["label_resolution_utc"].eq(common["execution_label_end_utc"]).all():
        raise MaterializationError("direct q25 label resolution is not exact H12")
    if not np.isfinite(common[["score_base_alpha", "score_residual_expected_ev", "q25_net_bps"]].to_numpy(float)).all():
        raise MaterializationError("one or more frozen scores are non-finite")
    common["score_direct_q25_ev_bps"] = common["q25_net_bps"].astype(float)

    frozen = json.loads(DIRECT_FROZEN_STATE.read_text(encoding="utf-8"))
    if frozen.get("winner", {}).get("score_column") != "q25_net_bps":
        raise MaterializationError("direct frozen winner is not q25")
    dataset_path, dataset_manifest, dataset_manifest_path = _output_record(FOLD_DATASET_ROOT, "dataset")
    if frozen.get("dataset", {}).get("sha256") != sha256(dataset_path) or frozen.get("model", {}).get("sha256") != sha256(DIRECT_FINAL_MODEL):
        raise MaterializationError("direct frozen state no longer binds its dataset/model")
    dataset = _normalise(pd.read_parquet(dataset_path), name="direct fold dataset")
    folds = _fold_frame(dataset)
    common = common.merge(folds, on=list(IDENTITY), how="left", validate="one_to_one", indicator="fold_join")
    if not common["fold_join"].eq("both").all():
        raise MaterializationError("direct common rows lack reconstructed chronological fold")
    common = common.drop(columns="fold_join")
    if not common["oof_fold_name"].isin(("old_march", "old_april")).all():
        raise MaterializationError("unexpected direct chronological fold for March-April")
    if not common["max_training_label_resolution_utc"].lt(common["fit_cutoff_utc"]).all() or not common["fit_cutoff_utc"].le(common["execution_decision_utc"]).all():
        raise MaterializationError("direct fold chronology is non-causal")
    common["feature_available_at"] = common["__ts__"]
    common["score_available_at"] = common["execution_decision_utc"]
    if not common["feature_available_at"].lt(common["execution_decision_utc"]).all() or not common["score_available_at"].le(common["execution_decision_utc"]).all():
        raise MaterializationError("score availability violates decision timing")
    common["candidate_month"] = common["__ts__"].dt.strftime("%Y-%m")
    common["direct_score_model_lineage"] = "cross_era_direct_net_quantile_challenger_q25_oof"
    common["direct_score_oof_recipe"] = "chronological_folds_prior_resolved_labels"
    common["direct_score_source_hash"] = sha256(direct_path)
    common["direct_frozen_state_hash"] = sha256(DIRECT_FROZEN_STATE)
    common["direct_frozen_final_model_hash"] = sha256(DIRECT_FINAL_MODEL)
    common["oof_model_binary_status"] = "not_persisted; exact_oof_output_and_recipe_hash_bound"
    provenance_columns = [
        *IDENTITY, "execution_decision_utc", "execution_label_end_utc", "feature_available_at",
        "score_available_at", "q25_net_bps", "oof_fold_name", "fit_cutoff_utc",
        "max_training_label_resolution_utc", "fold_train_rows", "direct_score_model_lineage",
        "direct_score_oof_recipe", "direct_score_source_hash", "direct_frozen_state_hash",
        "direct_frozen_final_model_hash", "oof_model_binary_status",
    ]
    evidence = {
        "base_ledger": binding(BASE_LEDGER), "residual_ledger": binding(RESIDUAL_LEDGER),
        "ledger_manifest": binding(ledger_manifest_path), "direct_oof": binding(direct_path),
        "direct_manifest": binding(direct_manifest_path), "direct_frozen_state": binding(DIRECT_FROZEN_STATE),
        "direct_final_model": binding(DIRECT_FINAL_MODEL), "direct_fold_dataset": binding(dataset_path),
        "direct_fold_dataset_manifest": binding(dataset_manifest_path),
        "direct_recipe_runner": binding(ROOT / "scripts/run_cross_era_tail_payoff_challenger.py"),
    }
    return common.sort_values(["execution_decision_utc", "candidate_id"], kind="stable").reset_index(drop=True), common.loc[:, provenance_columns].copy(), evidence


def _map_score(frame: pd.DataFrame, score_name: str, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped, _ = causal_mappings(frame, score_col=score_column, window_days=WINDOW_DAYS, min_reference_rows=MIN_REFERENCE_ROWS, side_support_target=SIDE_SUPPORT_TARGET)
    value_column = f"mapped_{score_name}_ev"
    mapped[value_column] = mapped["causal_recent_side_isotonic_ev"]
    mapped = mapped.drop(columns=["causal_recent_percentile", "causal_recent_robust_z", "causal_recent_isotonic_ev", "causal_recent_side_isotonic_ev"])
    audits: list[dict[str, Any]] = []
    for snapshot, local in mapped.groupby(mapped["execution_decision_utc"].dt.floor("D"), sort=True):
        reference = mapped.loc[
            mapped["execution_label_end_utc"].lt(snapshot)
            & mapped["execution_label_end_utc"].ge(snapshot - pd.Timedelta(days=WINDOW_DAYS))
        ]
        counts = reference.groupby("side_name", observed=True).size()
        audits.append({
            "score_name": score_name, "snapshot_utc": snapshot,
            "reference_window_start_utc": snapshot - pd.Timedelta(days=WINDOW_DAYS),
            "reference_window_end_utc": snapshot, "reference_rows": int(len(reference)),
            "long_reference_rows": int(counts.get("long", 0)), "short_reference_rows": int(counts.get("short", 0)),
            "reference_label_end_max_utc": reference["execution_label_end_utc"].max() if len(reference) else pd.NaT,
            "strictly_resolved_before_snapshot": bool(reference["execution_label_end_utc"].lt(snapshot).all()),
            "mapping_available": bool(local[value_column].notna().all()), "current_rows": int(len(local)),
        })
    audit = pd.DataFrame(audits)
    if not audit["strictly_resolved_before_snapshot"].all():
        raise MaterializationError(f"{score_name} map uses unresolved labels")
    return mapped.loc[:, [*IDENTITY, value_column]], audit


def map_current_policy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    audits: list[pd.DataFrame] = []
    for score_name, score_column in SCORE_COLUMNS.items():
        mapped, audit = _map_score(output, score_name, score_column)
        output = output.merge(mapped, on=list(IDENTITY), how="left", validate="one_to_one")
        audits.append(audit)
    mapped_columns = [f"mapped_{name}_ev" for name in SCORE_COLUMNS]
    output["common_mapping_eligible"] = output[mapped_columns].notna().all(axis=1)
    output["mapping_available_at"] = output["execution_decision_utc"]
    if not output["mapping_available_at"].le(output["execution_decision_utc"]).all():
        raise MaterializationError("mapped score availability violates decision timing")
    return output, pd.concat(audits, ignore_index=True)


def _stable_top(frame: pd.DataFrame, score: str, fraction: float, secondary: str | None = None) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    columns = [score]
    ascending = [False]
    if secondary is not None:
        columns.append(secondary)
        ascending.append(False)
    columns.extend(["candidate_id", "__ts__", "__symbol__", "side_name"])
    ascending.extend([True, True, True, True])
    return frame.sort_values(columns, ascending=ascending, kind="stable").iloc[:count].copy()


def evaluate(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = mapped.loc[mapped["common_mapping_eligible"].astype(bool)].copy()
    metrics: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    ties: list[dict[str, Any]] = []
    for month, local in work.groupby("candidate_month", sort=True):
        for score_name, raw_column in SCORE_COLUMNS.items():
            mapped_column = f"mapped_{score_name}_ev"
            for mode, score_column, secondary in (("raw_common", raw_column, None), ("current_recent_ev_mapped", mapped_column, raw_column)):
                record: dict[str, Any] = {"month": str(month), "score_name": score_name, "mode": mode, "score_column": score_column, "eligible_rows": len(local)}
                for fraction in TOP_FRACTIONS:
                    selected = _stable_top(local, score_column, fraction, secondary)
                    label = f"top{int(fraction * 100):02d}"
                    record.update({
                        f"{label}_rows": len(selected),
                        f"{label}_gross_bps": float(selected["execution_gross_ev_12h"].mean() * 1e4),
                        f"{label}_cost_bps": float(selected["execution_cost_return"].mean() * 1e4),
                        f"{label}_net_bps": float(selected[NET].mean() * 1e4),
                        f"{label}_positive_net_rate": float(selected[NET].gt(0.0).mean()),
                    })
                    if fraction == 0.10:
                        book = selected.loc[:, [*IDENTITY, "candidate_month", "execution_gross_ev_12h", "execution_cost_return", NET]].copy()
                        book["score_name"] = score_name
                        book["mode"] = mode
                        book["score_value"] = selected[score_column].to_numpy(float)
                        books.append(book)
                        cutoff = float(selected[score_column].iloc[-1])
                        ties.append({"month": str(month), "score_name": score_name, "mode": mode, "cutoff": cutoff, "cutoff_tie_rows": int(np.isclose(local[score_column].to_numpy(float), cutoff, rtol=0.0, atol=1e-14).sum()), "distinct_scores": int(local[score_column].nunique()), "raw_secondary_used": bool(secondary)})
                metrics.append(record)
    book_frame = pd.concat(books, ignore_index=True)
    side_rows: list[dict[str, Any]] = []
    for (month, score_name, mode), local in book_frame.groupby(["candidate_month", "score_name", "mode"], sort=True):
        for side in ("long", "short"):
            chosen = local.loc[local["side_name"].eq(side)]
            side_rows.append({"month": str(month), "score_name": score_name, "mode": mode, "side_name": side, "rows": len(chosen), "share": float(len(chosen) / len(local)), "conditional_net_bps": float(chosen[NET].mean() * 1e4) if len(chosen) else np.nan, "contribution_net_bps": float(chosen[NET].sum() * 1e4 / len(local)) if len(chosen) else 0.0})
    return pd.DataFrame(metrics), book_frame, pd.DataFrame(ties), pd.DataFrame(side_rows)


def run(*, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    source, provenance, evidence = reconstruct()
    mapped, audit = map_current_policy(source)
    metrics, books, ties, side = evaluate(mapped)
    stage = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    stage.mkdir(parents=True)
    try:
        tables = {
            "common_candidates.parquet": source, "direct_q25_oof_provenance.parquet": provenance,
            "mapped_candidates.parquet": mapped, "mapping_audit.parquet": audit,
            "period_metrics.parquet": metrics, "selection_books.parquet": books,
            "cutoff_ties.parquet": ties, "global_book_side_attribution.parquet": side,
        }
        outputs: dict[str, Any] = {}
        for name, table in tables.items():
            path = stage / name
            table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {"path": str((output_dir / name).resolve()), "rows": len(table), "sha256": sha256(path)}
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_REUSED_MARAPR_EXACT_H12_CURRENT_MAPPING_DIAGNOSTIC_NO_PROMOTION_NO_REPLAY",
            "promotion_eligible": False, "portfolio_replay_authorized": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": evidence,
            "contract": {
                "scores": "frozen base/residual/direct-q25 OOF outputs; no model refit or score alteration",
                "direct_provenance": "chronological direct-q25 fold validity reconstructed from the bound tail-payoff dataset; per-fold binaries unavailable but recipe/state/model/output hashes are bound",
                "mapping": "21 UTC days; exact H12 label_end < snapshot; minimum 500 pooled rows; side isotonic shrunk toward pooled at 500 support; unavailable warm-up retained",
                "selection": "one pooled-global calendar-month top-k after mapped EV; raw score then candidate ID only resolve plateaus; no quotas",
                "policy_status": "March-April was used in historical direct model selection; all outputs are diagnostic/reused evidence only",
                "actions": "timing, MAE, target-price and wait layers excluded",
            },
            "constants": {"window_days": WINDOW_DAYS, "minimum_reference_rows": MIN_REFERENCE_ROWS, "side_support_target": SIDE_SUPPORT_TARGET},
            "rows": {"source": len(mapped), "common_mapping_eligible": int(mapped["common_mapping_eligible"].sum()), "warmup_unmapped": int((~mapped["common_mapping_eligible"]).sum())},
            "score_registry": dict(SCORE_COLUMNS), "outputs": outputs, "runner": binding(Path(__file__)),
        }
        write_json(stage / "manifest.json", manifest)
        digest = sha256(stage / "manifest.json")
        (stage / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="utf-8")
        write_json(stage / "seal.json", {"schema": SCHEMA, "manifest_sha256": digest, "files_sha256": {path.relative_to(stage).as_posix(): sha256(path) for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "seal.json"}})
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(safe(run(output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
