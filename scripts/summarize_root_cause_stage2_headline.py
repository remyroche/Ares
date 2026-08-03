#!/usr/bin/env python3
"""Create a deterministic, read-only headline summary of canonical Stage-2.

This is an evidence presentation layer only: it reads the sealed Stage-2 v4
artefacts and never fits, selects, ranks, or changes any trading model/policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"
DEFAULT_INPUT = ART / "root_cause_feature_information_20260731_v4"
DEFAULT_PARQUET = "stage2_headline_evidence_summary.parquet"
DEFAULT_JSON = "stage2_headline_evidence_summary.json"
SCHEMA = "root_cause_stage2_headline_evidence_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _record(section: str, metric: str, *, value: Any = None, **fields: Any) -> dict[str, Any]:
    return {"section": section, "metric": metric, "value": _finite(value), **fields}


def _require(input_dir: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"canonical Stage-2 artefact is incomplete: {missing}")


def _best_features(results: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metrics = (("transported_ic", "transported_ic_mean"), ("top_bottom_decile_spread_bps", "top_bottom_decile_spread_mean_bps"))
    for side in sorted(results.side.astype(str).unique()):
        local = results.loc[results.side.astype(str).eq(side)]
        for label, column in metrics:
            candidates = local.dropna(subset=[column]).sort_values([column, "feature_name"], ascending=[False, True], kind="stable")
            if candidates.empty:
                records.append(_record("best_transported_feature", label, side=side, status="NOT_AVAILABLE"))
                continue
            row = candidates.iloc[0]
            records.append(_record(
                "best_transported_feature", label, value=row[column], side=side, feature_name=str(row.feature_name), rank=1,
                support_rows=int(row.evaluated_rows), folds=int(row.folds),
                complementary_ic=float(row.transported_ic_mean),
                complementary_spread_bps=float(row.top_bottom_decile_spread_mean_bps), status="OK",
            ))
    return records


def _mechanism_means(mechanisms: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (side, group), part in mechanisms.groupby(["side", "mechanism_group"], sort=True, observed=True):
        ok = part.loc[part.status.eq("OK")]
        kwargs = {
            "side": str(side), "mechanism_group": str(group), "folds_ok": int(ok.fold.nunique()),
            "support_rows": int(ok.test_rows.sum()) if not ok.empty else 0,
            "ok_rows": int(len(ok)), "not_run_rows": int(len(part) - len(ok)),
            "status": "OK" if not ok.empty else "NOT_RUN",
        }
        for metric in ("spearman_ic", "top_bottom_decile_spread_bps", "oof_mae_bps"):
            records.append(_record("mechanism_group_mean", metric, value=ok[metric].mean() if not ok.empty else None, **kwargs))
    return records


def _residual_probe_means(probes: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouping = ["head", "probe_family", "side"]
    for keys, part in probes.groupby(grouping, sort=True, observed=True):
        head, family, side = map(str, keys)
        ok = part.loc[part.status.eq("OK")]
        kwargs = {
            "head": head, "probe_family": family, "side": side,
            "folds_ok": int(ok.fold.nunique()), "support_rows": int(ok.test_rows.sum()) if not ok.empty else 0,
            "ok_rows": int(len(ok)), "not_run_rows": int(len(part) - len(ok)),
            "status": "OK" if not ok.empty else "NOT_RUN",
        }
        records.append(_record("residual_probe_mean", "residual_probe_oof_ic", value=ok.residual_probe_oof_ic.mean() if not ok.empty else None, **kwargs))
        records.append(_record("residual_probe_mean", "residual_probe_oof_mae_bps", value=ok.residual_probe_oof_mae_bps.mean() if not ok.empty else None, **kwargs))
    return records


def _drift_ranges(drift: pd.DataFrame, cohort: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for side, part in drift.groupby("side", sort=True, observed=True):
        for metric in ("psi", "jensen_shannon", "wasserstein", "missingness_delta"):
            series = pd.to_numeric(part[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            records.extend([
                _record("drift_range", f"{metric}_min", value=series.min() if not series.empty else None, side=str(side), observations=int(len(series)), status="OK" if not series.empty else "NOT_AVAILABLE"),
                _record("drift_range", f"{metric}_max", value=series.max() if not series.empty else None, side=str(side), observations=int(len(series)), status="OK" if not series.empty else "NOT_AVAILABLE"),
            ])
    monthly = cohort.loc[cohort.scope_type.eq("side_month")]
    for side, part in monthly.groupby("side", sort=True, observed=True):
        series = pd.to_numeric(part.adversarial_auc, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        records.extend([
            _record("adversarial_drift_range", "adversarial_auc_min", value=series.min() if not series.empty else None, side=str(side), observations=int(len(series)), status="OK" if not series.empty else "NOT_AVAILABLE"),
            _record("adversarial_drift_range", "adversarial_auc_max", value=series.max() if not series.empty else None, side=str(side), observations=int(len(series)), status="OK" if not series.empty else "NOT_AVAILABLE"),
        ])
    return records


def _support_counts(named: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source, frame in sorted(named.items()):
        if "status" not in frame.columns:
            records.append(_record("support_status", "rows", value=len(frame), source_artifact=source, status="NO_STATUS_COLUMN"))
            continue
        for status, part in frame.groupby("status", dropna=False, sort=True, observed=True):
            label = "NULL" if pd.isna(status) else str(status)
            records.append(_record("support_status", "rows", value=len(part), source_artifact=source, status=label))
    return records


def _availability(inventory: pd.DataFrame) -> list[dict[str, Any]]:
    total = len(inventory)
    research = int(inventory.research_causal_probe_eligible.fillna(False).astype(bool).sum())
    production = int(inventory.production_live_reuse_eligible.fillna(False).astype(bool).sum())
    records = [
        _record("availability", "features_total", value=total, status="OK"),
        _record("availability", "research_causal_cutoff_verified", value=research, status="SEALED_RESEARCH_ONLY"),
        _record("availability", "production_live_reuse_verified", value=production, status="REQUIRES_PER_FEATURE_LIVE_AND_STALENESS_EVIDENCE"),
    ]
    for status, part in inventory.groupby("live_reproducibility_status", dropna=False, sort=True, observed=True):
        records.append(_record("availability", "live_reproducibility_status_count", value=len(part), availability_status="NULL" if pd.isna(status) else str(status), status="OK"))
    for status, part in inventory.groupby("staleness_status", dropna=False, sort=True, observed=True):
        records.append(_record("availability", "staleness_status_count", value=len(part), availability_status="NULL" if pd.isna(status) else str(status), status="OK"))
    return records


def build_headline(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "results": "feature_information_results.parquet",
        "mechanisms": "feature_information_mechanism_oof.parquet",
        "probes": "feature_information_residual_probes.parquet",
        "drift": "feature_information_drift.parquet",
        "cohort": "feature_information_cohort_drift.parquet",
        "inventory": "feature_information_inventory.parquet",
        "gross_mapping": "feature_information_fold_local_gross_mapping.parquet",
        "netmap": "feature_information_current_netmap_diagnostics.parquet",
        "stage2_manifest": "run_manifest.json",
    }
    _require(input_dir, required.values())
    loaded = {name: pd.read_parquet(input_dir / path) for name, path in required.items() if path.endswith(".parquet")}
    stage2_manifest = json.loads((input_dir / required["stage2_manifest"]).read_text(encoding="utf-8"))
    records = [
        *_best_features(loaded["results"]),
        *_mechanism_means(loaded["mechanisms"]),
        *_residual_probe_means(loaded["probes"]),
        *_drift_ranges(loaded["drift"], loaded["cohort"]),
        *_support_counts({
            "mechanism_oof": loaded["mechanisms"], "residual_probes": loaded["probes"],
            "fold_local_gross_mapping": loaded["gross_mapping"], "current_netmap": loaded["netmap"],
        }),
        *_availability(loaded["inventory"]),
    ]
    frame = pd.DataFrame(records)
    stable_columns = [
        "section", "metric", "side", "head", "probe_family", "mechanism_group", "feature_name", "source_artifact",
        "availability_status", "rank", "folds", "folds_ok", "support_rows", "ok_rows", "not_run_rows", "observations",
        "complementary_ic", "complementary_spread_bps", "value", "status",
    ]
    frame = frame.reindex(columns=[*stable_columns, *sorted(set(frame.columns) - set(stable_columns))])
    sort_columns = [name for name in ("section", "metric", "side", "head", "probe_family", "mechanism_group", "feature_name", "source_artifact", "availability_status") if name in frame]
    frame = frame.sort_values(sort_columns, kind="stable", na_position="last").reset_index(drop=True)
    payload = {
        "schema": SCHEMA,
        "status": "READ_ONLY_DIAGNOSTIC_SUMMARY_NO_FITTING_OR_POLICY_CHANGE",
        "canonical_stage2_dir": str(input_dir),
        "stage2_manifest_sha256": sha256(input_dir / required["stage2_manifest"]),
        "stage2_input_sha256": stage2_manifest.get("inputs_sha256", {}),
        "stage2_output_sha256": stage2_manifest.get("outputs_sha256", {}),
        "records": frame.where(pd.notna(frame), None).to_dict("records"),
        "availability_interpretation": "research_causal_cutoff_verified is sealed raw-panel/cutoff evidence only. production_live_reuse_verified requires explicit per-feature live reproducibility and staleness evidence.",
    }
    return frame, payload


def run(*, input_dir: Path = DEFAULT_INPUT, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or input_dir
    parquet_path, json_path = output_dir / DEFAULT_PARQUET, output_dir / DEFAULT_JSON
    if parquet_path.exists() or json_path.exists():
        raise FileExistsError("headline summary output already exists")
    frame, payload = build_headline(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".stage2_headline.", suffix=".parquet", dir=output_dir)
    os.close(fd)
    temp_parquet = Path(temp_name)
    temp_json = output_dir / f".{DEFAULT_JSON}.tmp"
    try:
        frame.to_parquet(temp_parquet, index=False, compression="zstd")
        payload["summary_parquet_sha256"] = sha256(temp_parquet)
        temp_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temp_parquet, parquet_path)
        os.replace(temp_json, json_path)
    except Exception:
        temp_parquet.unlink(missing_ok=True)
        temp_json.unlink(missing_ok=True)
        raise
    return {"rows": int(len(frame)), "parquet": str(parquet_path), "json": str(json_path), "parquet_sha256": sha256(parquet_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(input_dir=args.input, output_dir=args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
