#!/usr/bin/env python3
"""Build a hash-bound exact-ID research panel for the 2022 inverse lineage.

This is deliberately a *source-separated research* adapter.  It binds the
authoritative causal stage, complete exact-path coverage proof, and 12-hour
multitask labels, while keeping the panel explicitly non-OOF and
non-promotable.  The only model inputs it exposes are the 69 causal fields
from the staged candidate population; all policy/path fields from the label
artifact are retained as outcomes, never features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / (
    "data_perp/artifacts/jan_jul_2022_inverse_pi_causal_exact1m_stage_"
    "20260730_v1"
)
DEFAULT_LABELS = ROOT / (
    "data_perp/artifacts/jan_jul_2022_inverse_pi_causal_multitask_labels_"
    "20260730_v2"
)
DEFAULT_COVERAGE = ROOT / (
    "data_perp/artifacts/jan_jul_2022_inverse_pi_causal_candidate_coverage_"
    "20260730_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/jan_jul_2022_inverse_pi_exact_id_research_panel_"
    "20260730_v1"
)

SCHEMA = "inverse_exact_id_research_panel_v1"
STAGE_SCHEMA = "historical_backcast_exact1m_request_stage_v2"
COVERAGE_SCHEMA = "historical_exact1m_candidate_coverage_v1"
LINEAGE = "historical_inverse_pi_market_grid_exact1m_research_only"
POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
EVIDENCE_SCOPE = "inverse_pi_market_grid_causal_features_research_not_oof"
ECONOMICS_CONTRACT = "inverse_quote_notional_current_spread_counterfactual_only"
ECONOMICS_LABELS = "inverse_quote_notional_current_spread_counterfactual"
PRODUCT_LINEAGE = "kraken_inverse_pi_exact_product_binding_v1"
IDENTITY_STAGE = ("candidate_id", "signal_timestamp", "symbol", "side_name")
IDENTITY_LABELS = ("candidate_id", "__ts__", "__symbol__", "side_name")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")


def _load_manifest(root: Path, *, name: str) -> tuple[Path, dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{name} manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} manifest is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} manifest must be an object: {path}")
    return path, value


def _require(value: Mapping[str, Any], key: str, expected: Any, *, name: str) -> None:
    actual = value.get(key)
    if actual != expected:
        raise ValueError(
            f"{name} contract mismatch for {key}: expected {expected!r}, got {actual!r}"
        )


def _manifest_output(
    manifest: Mapping[str, Any], root: Path, key: str, *, name: str
) -> Path:
    record = manifest.get("outputs", {}).get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} manifest lacks output record for {key}")
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{name} output record for {key} is incomplete")
    path = Path(raw_path)
    # A copied artifact can retain its original absolute path in its manifest;
    # accept the local basename only when the recorded file is absent.
    if not path.is_file():
        path = root / Path(raw_path).name
    if not path.is_file():
        raise FileNotFoundError(f"{name} output is missing for {key}: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{name} output checksum fails for {key}: {path}")
    return path


def _normalize_stage(rows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_STAGE).difference(rows.columns))
    if missing:
        raise ValueError(f"staged candidates lack identity fields: {missing}")
    work = rows.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["signal_timestamp"] = pd.to_datetime(
        work["signal_timestamp"], utc=True, errors="coerce"
    )
    work["symbol"] = work["symbol"].astype(str)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if work["signal_timestamp"].isna().any() or not work["side_name"].isin(("long", "short")).all():
        raise ValueError("staged candidates contain invalid exact identity values")
    if work["candidate_id"].duplicated(keep=False).any():
        raise ValueError("staged candidates have duplicate candidate_id values")
    if work.duplicated(list(IDENTITY_STAGE), keep=False).any():
        raise ValueError("staged candidates have duplicate exact identities")
    decision = pd.to_datetime(work.get("decision_timestamp"), utc=True, errors="coerce")
    if decision.isna().any() or not decision.equals(work["signal_timestamp"] + pd.Timedelta(hours=1)):
        raise ValueError("staged decision_timestamp is not signal_timestamp + 1 hour")
    return work


def _normalize_labels(rows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_LABELS).difference(rows.columns))
    if missing:
        raise ValueError(f"joined labels lack identity fields: {missing}")
    work = rows.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce")
    work["__symbol__"] = work["__symbol__"].astype(str)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if work["__ts__"].isna().any() or not work["side_name"].isin(("long", "short")).all():
        raise ValueError("joined labels contain invalid exact identity values")
    if work["candidate_id"].duplicated(keep=False).any():
        raise ValueError("joined labels have duplicate candidate_id values")
    if work.duplicated(list(IDENTITY_LABELS), keep=False).any():
        raise ValueError("joined labels have duplicate exact identities")
    return work


def causal_feature_contract(columns: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Return the exact, allow-listed 69 pre-entry feature inputs."""

    asset = [
        column
        for column in columns
        if column.startswith(("ret_", "rv_", "downside_rv_", "atr_", "range_", "drawdown_", "recovery_", "trend_", "path_efficiency_", "volume_", "jump_"))
    ]
    market = [
        column
        for column in columns
        if column.startswith("market_") or column == "btc_minus_alt_median_ret_24h"
    ]
    transition = [column for column in columns if column.startswith("transition_raw__")]
    families = {"asset": asset, "market": market, "transition": transition}
    features = [column for family in families.values() for column in family]
    if len(features) != len(set(features)):
        raise ValueError("causal feature families overlap")
    forbidden = ("target", "future", "execution", "label", "outcome", "path_arch")
    bad = [
        column for column in features if any(token in column.lower() for token in forbidden)
    ]
    if bad:
        raise ValueError(f"future/outcome fields attempted as features: {bad}")
    if len(features) != 69 or tuple(map(len, families.values())) != (29, 15, 25):
        found = {name: len(values) for name, values in families.items()}
        raise ValueError(f"expected 69 causal fields (29/15/25), found {found}")
    return features, families


def _validate_contracts(
    stage: Mapping[str, Any], labels: Mapping[str, Any], coverage: Mapping[str, Any]
) -> None:
    for name, manifest in (("stage", stage), ("labels", labels), ("coverage", coverage)):
        _require(manifest, "lineage", LINEAGE, name=name)
        _require(manifest, "candidate_population_lineage", POPULATION_LINEAGE, name=name)
        _require(manifest, "evidence_scope", EVIDENCE_SCOPE, name=name)
        _require(manifest, "product_lineage", PRODUCT_LINEAGE, name=name)
        _require(manifest, "execution_parity_claim", False, name=name)
        _require(manifest, "promotion_eligible", False, name=name)
    _require(stage, "schema", STAGE_SCHEMA, name="stage")
    _require(stage, "economics_contract", ECONOMICS_CONTRACT, name="stage")
    _require(stage, "return_unit", "quote_notional_price_return_not_inverse_collateral_roe", name="stage")
    _require(stage, "path_horizon_minutes", 720, name="stage")
    _require(stage, "signal_to_decision_hours", 1, name="stage")
    _require(labels, "economics", ECONOMICS_LABELS, name="labels")
    _require(labels, "oof_status", "not_oof", name="labels")
    _require(coverage, "schema", COVERAGE_SCHEMA, name="coverage")
    _require(coverage, "status", "complete", name="coverage")
    _require(coverage, "candidate_coverage_fraction", 1.0, name="coverage")
    _require(coverage, "incomplete_candidates", 0, name="coverage")
    _require(coverage, "required_minutes_per_candidate", 720, name="coverage")


def build_panel(
    stage_rows: pd.DataFrame, label_rows: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str]]:
    stage = _normalize_stage(stage_rows)
    labels = _normalize_labels(label_rows)
    features, families = causal_feature_contract(stage.columns)
    label_rename = {
        "__ts__": "__label_signal_timestamp__",
        "__symbol__": "__label_symbol__",
        "side_name": "__label_side_name__",
    }
    labels_for_join = labels.rename(columns=label_rename)
    merged = stage.merge(
        labels_for_join,
        on="candidate_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "__label"),
    )
    missing_stage = int(merged["_merge"].eq("right_only").sum())
    missing_labels = int(merged["_merge"].eq("left_only").sum())
    if missing_stage or missing_labels:
        raise ValueError(
            "exact-ID join is incomplete: "
            f"stage_missing={missing_stage}, labels_missing={missing_labels}"
        )
    mismatch = (
        ~merged["signal_timestamp"].eq(merged["__label_signal_timestamp__"])
        | ~merged["symbol"].eq(merged["__label_symbol__"])
        | ~merged["side_name"].eq(merged["__label_side_name__"])
    )
    if mismatch.any():
        raise ValueError(
            f"exact-ID join has signal/symbol/side mismatch for {int(mismatch.sum())} rows"
        )
    for label_column, stage_column in (
        ("__decision_ts__", "decision_timestamp"),
        ("__label_end_ts__", "path_end_exclusive"),
    ):
        if label_column in merged and stage_column in merged:
            actual = pd.to_datetime(merged[label_column], utc=True, errors="coerce")
            expected = pd.to_datetime(merged[stage_column], utc=True, errors="coerce")
            if actual.isna().any() or not actual.equals(expected):
                raise ValueError(f"label {label_column} does not match staged {stage_column}")
    if "__label_available_at__" in merged:
        available = pd.to_datetime(merged["__label_available_at__"], utc=True, errors="coerce")
        expected = merged["decision_timestamp"] + pd.Timedelta(hours=12)
        if available.isna().any() or not available.equals(expected):
            raise ValueError("labels are not explicitly available only at decision + 12 hours")
    merged = merged.drop(columns=["_merge"])
    # The label-side identity witnesses are retained to make every downstream
    # notebook capable of re-checking the equality without reloads.
    identity = [
        "candidate_id", "signal_timestamp", "decision_timestamp", "path_end_exclusive",
        "symbol", "side_name", "__label_signal_timestamp__", "__label_symbol__",
        "__label_side_name__",
    ]
    label_columns = [
        column
        for column in merged.columns
        if column not in stage.columns or column.endswith("__label")
    ]
    return merged, features, families, [*identity, *label_columns]


def run(args: argparse.Namespace) -> dict[str, Path]:
    stage_root = Path(args.stage_root).resolve()
    labels_root = Path(args.labels_root).resolve()
    coverage_root = Path(args.coverage_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    stage_manifest_path, stage_manifest = _load_manifest(stage_root, name="stage")
    labels_manifest_path, labels_manifest = _load_manifest(labels_root, name="labels")
    coverage_manifest_path, coverage_manifest = _load_manifest(coverage_root, name="coverage")
    _validate_contracts(stage_manifest, labels_manifest, coverage_manifest)
    stage_path = _manifest_output(stage_manifest, stage_root, "staged_candidates", name="stage")
    labels_path = _manifest_output(labels_manifest, labels_root, "joined_multitask_labels", name="labels")
    coverage_path = _manifest_output(coverage_manifest, coverage_root, "candidate_coverage", name="coverage")
    # The coverage proof must bind precisely the stage manifest and the label
    # artifact must bind precisely that same coverage proof.
    coverage_stage = coverage_manifest.get("stage_manifest", {})
    if coverage_stage.get("sha256") != sha256(stage_manifest_path):
        raise ValueError("coverage manifest is not bound to the supplied stage manifest")
    label_coverage = labels_manifest.get("sources", {}).get("candidate_coverage_manifest", {})
    if label_coverage.get("sha256") != sha256(coverage_manifest_path):
        raise ValueError("labels manifest is not bound to the supplied coverage manifest")
    stage_rows = _normalize_stage(pd.read_parquet(stage_path))
    labels_rows = _normalize_labels(pd.read_parquet(labels_path))
    coverage_rows = pd.read_parquet(coverage_path)
    if len(stage_rows) != int(stage_manifest.get("selected_rows", -1)):
        raise ValueError("stage row count disagrees with stage manifest")
    if len(labels_rows) != int(labels_manifest.get("rows", -1)):
        raise ValueError("label row count disagrees with labels manifest")
    if len(coverage_rows) != len(stage_rows) or len(coverage_rows) != int(coverage_manifest.get("complete_candidates", -1)):
        raise ValueError("coverage row count does not prove all staged candidates complete")
    if "candidate_id" not in coverage_rows or coverage_rows["candidate_id"].astype(str).duplicated().any():
        raise ValueError("coverage table lacks unique candidate_id")
    if set(coverage_rows["candidate_id"].astype(str)) != set(stage_rows["candidate_id"]):
        raise ValueError("coverage table candidate population differs from stage")
    panel, features, families, identity_and_labels = build_panel(stage_rows, labels_rows)
    if len(panel) != len(stage_rows):
        raise ValueError("panel row count is not exact stage population")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temp_name:
        temp_dir = Path(temp_name)
        panel_path = temp_dir / "inverse_exact_id_research_panel.parquet"
        feature_contract_path = temp_dir / "feature_contract.json"
        manifest_path = temp_dir / "manifest.json"
        panel.to_parquet(panel_path, index=False)
        feature_contract = {
            "feature_columns": features,
            "feature_families": families,
            "feature_count": len(features),
            "input_timing": "strictly causal at signal_timestamp; no path or policy outcomes",
            "forbidden_feature_tokens": ["target", "future", "execution", "label", "outcome", "path_arch"],
        }
        _write_json(feature_contract_path, feature_contract)
        manifest = {
            "schema": SCHEMA,
            "status": "research_panel_ready_not_oof_not_promotable",
            "lineage": LINEAGE,
            "candidate_population_lineage": POPULATION_LINEAGE,
            "evidence_scope": EVIDENCE_SCOPE,
            "economics_contract": ECONOMICS_CONTRACT,
            "economics_label_contract": ECONOMICS_LABELS,
            "product_lineage": PRODUCT_LINEAGE,
            "return_unit": "quote_notional_price_return_not_inverse_collateral_roe",
            "execution_parity_claim": False,
            "promotion_eligible": False,
            "oof_status": "not_oof",
            "rows": int(len(panel)),
            "identity_columns": list(IDENTITY_STAGE),
            "identity_witness_columns": identity_and_labels,
            "feature_columns": features,
            "feature_families": families,
            "label_columns": [column for column in panel.columns if column not in stage_rows.columns],
            "input_hashes": {
                "stage_manifest": {"path": str(stage_manifest_path), "sha256": sha256(stage_manifest_path)},
                "staged_candidates": {"path": str(stage_path), "sha256": sha256(stage_path)},
                "labels_manifest": {"path": str(labels_manifest_path), "sha256": sha256(labels_manifest_path)},
                "joined_multitask_labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
                "coverage_manifest": {"path": str(coverage_manifest_path), "sha256": sha256(coverage_manifest_path)},
                "candidate_coverage": {"path": str(coverage_path), "sha256": sha256(coverage_path)},
            },
            "coverage": {
                "candidate_coverage_fraction": 1.0,
                "complete_candidates": int(len(coverage_rows)),
                "required_minutes_per_candidate": 720,
            },
            "outputs": {
                "inverse_exact_id_research_panel": {"path": str(output_dir / panel_path.name), "rows": int(len(panel)), "sha256": sha256(panel_path)},
                "feature_contract": {"path": str(output_dir / feature_contract_path.name), "sha256": sha256(feature_contract_path)},
            },
        }
        _write_json(manifest_path, manifest)
        (temp_dir / "manifest.sha256").write_text(sha256(manifest_path) + "  manifest.json\n")
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite immutable panel artifact: {output_dir}")
        os.replace(temp_dir, output_dir)
    return {
        "panel": output_dir / "inverse_exact_id_research_panel.parquet",
        "feature_contract": output_dir / "feature_contract.json",
        "manifest": output_dir / "manifest.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--coverage-root", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run(parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
