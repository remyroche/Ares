#!/usr/bin/env python3
"""Audit evidence-backed lineage for the long-only base-to-portfolio chain.

The auditor deliberately does not infer research claims from directory names.  A
missing manifest field is reported as missing evidence, while a path asserted by
a manifest but absent on disk is a failed validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASENAME = "long_chain_lineage"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_paths(directory: Path | None) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    preferred = (
        "manifest.json",
        "training_live_parity_contract.json",
        "policy_optimisation.json",
        "portfolio_policy_replay_report.json",
        "optimized_portfolio_policy_config.json",
        "best_policy_params.json",
    )
    named_artifacts = {
        "policy_optimisation.json",
        "policy_optimisation_oos_metrics.json",
        "portfolio_policy_replay_report.json",
        "optimized_portfolio_policy_config.json",
        "best_policy_params.json",
        "best_policy_params_perps.json",
        "strategy_for_inference.json",
    }
    found = {
        path.resolve()
        for path in directory.rglob("*.json")
        if path.name in named_artifacts or path.name.endswith("manifest.json") or path.name.endswith("_contract.json")
    }
    ordered: list[Path] = []
    for name in preferred:
        ordered.extend(sorted(path for path in found if path.name == name))
    ordered.extend(sorted(found - set(ordered)))
    return ordered


@dataclass(frozen=True)
class Document:
    path: Path
    data: dict[str, Any]


def _documents(directory: Path | None) -> list[Document]:
    documents = [
        Document(path=path, data=data)
        for path in _manifest_paths(directory)
        if (data := _read_json(path)) is not None
    ]
    referenced: list[Document] = []

    def visit(value: Any, key: str, document: Document) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key, document)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, document)
        elif isinstance(value, str) and value.endswith(".json") and _is_path_value(key, value):
            path = _resolve_path(value, document.path, directory)
            if path not in known and (payload := _read_json(path)) is not None:
                known.add(path)
                referenced.append(Document(path=path, data=payload))

    known = {document.path for document in documents}
    for document in documents:
        visit(document.data, "", document)
    return documents + referenced


def _find_key(value: Any, key: str) -> Any | None:
    if isinstance(value, dict):
        if key in value and value[key] is not None:
            return value[key]
        for child in value.values():
            found = _find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_key(child, key)
            if found is not None:
                return found
    return None


def _pick(documents: Iterable[Document], *keys: str) -> tuple[Any | None, str | None]:
    for document in documents:
        for key in keys:
            value = _find_key(document.data, key)
            if value is not None:
                return value, str(document.path)
    return None, None


def _is_path_value(key: str, value: str) -> bool:
    lowered = key.lower()
    # Manifests contain many strings which happen to look file-like but are not
    # artifact locations: source partition names, ISO timestamps, timeframe
    # values and content hashes.  Only validate fields whose *key* explicitly
    # declares filesystem semantics.  This keeps the audit evidence-first
    # without manufacturing path failures from unrelated metadata.
    if any(token in lowered for token in ("hash", "sha256", "checksum")):
        return False
    exact_keys = {
        "file",
        "filename",
        "directory",
        "manifest",
    }
    path_suffixes = (
        "_path",
        "_file",
        "_dir",
        "_directory",
        "_manifest",
        "_artifact",
    )
    return lowered in exact_keys or lowered.endswith(path_suffixes)


def _resolve_path(raw: str, manifest: Path, input_dir: Path | None) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (manifest.parent, input_dir, ROOT):
        if base is not None:
            resolved = base / candidate
            if resolved.exists():
                return resolved
    return (input_dir or manifest.parent) / candidate


def _path_record(raw: str, manifest: Path, input_dir: Path | None) -> dict[str, Any]:
    resolved = _resolve_path(raw, manifest, input_dir)
    return {
        "declared_path": raw,
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "sha256": _sha256(resolved),
        "source_manifest": str(manifest),
    }


def _referenced_paths(documents: Iterable[Document], input_dir: Path | None) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def visit(value: Any, key: str, document: Document) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key, document)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, document)
        elif isinstance(value, str) and _is_path_value(key, value):
            record = _path_record(value, document.path, input_dir)
            records[(record["declared_path"], record["source_manifest"])] = record

    for document in documents:
        records[(str(document.path), str(document.path))] = {
            "declared_path": str(document.path),
            "resolved_path": str(document.path),
            "exists": True,
            "sha256": _sha256(document.path),
            "source_manifest": str(document.path),
        }
        visit(document.data, "", document)
    return sorted(records.values(), key=lambda item: (item["declared_path"], item["source_manifest"]))


def _feature_hash(documents: Iterable[Document]) -> tuple[str | None, str | None]:
    value, source = _pick(
        documents,
        "feature_contract_hash",
        "feature_hash",
        "selected_features_hash",
        "external_feature_sidecar_sha256",
    )
    if isinstance(value, str):
        return value, source
    features, source = _pick(documents, "selected_feature_union", "selected_features", "feature_names")
    if isinstance(features, list):
        return hashlib.sha256(_json(features).encode("utf-8")).hexdigest(), source
    return None, None


def _params_hash(documents: Iterable[Document], input_dir: Path | None) -> tuple[str | None, str | None]:
    value, source = _pick(documents, "hpo_params_hash", "params_hash", "fixed_params_json")
    if not isinstance(value, str):
        params, source = _pick(documents, "best_params", "fixed_params", "selected_params")
        return (hashlib.sha256(_json(params).encode("utf-8")).hexdigest(), source) if params is not None else (None, None)
    source_path = Path(source) if source else ROOT
    return _sha256(_resolve_path(value, source_path, input_dir)), source


def _explicit_leakage_status(documents: Iterable[Document]) -> tuple[str, Any | None, str | None]:
    value, source = _pick(documents, "leakage_status", "leakage_passed", "leakage_safe")
    if value is True or (isinstance(value, str) and value.strip().lower() in {"pass", "passed", "ok"}):
        return "PASS", value, source
    if value is False or (isinstance(value, str) and value.strip().lower() in {"fail", "failed"}):
        return "FAIL", value, source
    contract, contract_source = _pick(documents, "leakage_contract")
    return "MISSING_EVIDENCE", contract, contract_source


def _value(documents: Iterable[Document], *keys: str) -> dict[str, Any]:
    found, source = _pick(documents, *keys)
    return {"value": found, "source_manifest": source}


def _state_path(documents: Iterable[Document], input_dir: Path | None) -> dict[str, Any]:
    value, source = _pick(
        documents,
        "ae_gmm_state_path",
        "ae_gmm_state_reference_state_path",
        "fixed_ae_gmm_state_pkl",
        "ae_gmm_state_pkl",
    )
    if not isinstance(value, str) or source is None:
        return {"value": None, "source_manifest": source, "artifact": None}
    return {
        "value": value,
        "source_manifest": source,
        "artifact": _path_record(value, Path(source), input_dir),
    }


def _row(
    layer: str,
    documents: list[Document],
    input_dir: Path | None,
    required_fields: tuple[str, ...],
) -> dict[str, Any]:
    fields = {
        "training_window_splits": _value(documents, "training_window", "train_window", "fold_windows", "validation_windowing"),
        "oos_scope": _value(documents, "oos_scope", "single_fit_oos_window", "oos_model_age_contract", "folds"),
        "target_labels": _value(documents, "target_modes", "target_mode", "target", "labels_path"),
        "feature_selection_recipe": _value(documents, "base_feature_selection_recipe", "feature_selection_recipe", "feature_selection_method"),
        "feature_count": _value(documents, "fixed_selected_features_count", "feature_count", "meta_feature_count"),
        "hpo_scope": _value(documents, "hpo_scope", "hpo_calibration_fold", "hpo"),
        "hpo_trials": _value(documents, "n_trials_requested", "hpo_trials", "n_trials"),
        "cost_contract": _value(documents, "cost_contract", "fee_contract", "spread_contract", "round_trip_cost_pct"),
        "final_refit_exclusion": _value(documents, "final_refit_exclusion", "final_refit_excluded", "final_refit_contract"),
        "geometry": _value(documents, "side_archetype_expected_ev_policy", "strategies", "geometry"),
    }
    feature_hash, feature_hash_source = _feature_hash(documents)
    hpo_hash, hpo_hash_source = _params_hash(documents, input_dir)
    leakage_status, leakage_evidence, leakage_source = _explicit_leakage_status(documents)
    fields["feature_hash"] = {"value": feature_hash, "source_manifest": feature_hash_source}
    fields["hpo_params_hash"] = {"value": hpo_hash, "source_manifest": hpo_hash_source}
    fields["ae_gmm_state"] = _state_path(documents, input_dir)
    fields["leakage"] = {"value": leakage_evidence, "source_manifest": leakage_source}

    artifacts = _referenced_paths(documents, input_dir)
    missing = [name for name in required_fields if fields[name]["value"] is None]
    if not documents:
        missing.append("authoritative_manifest")
    invalid_paths = [item for item in artifacts if not item["exists"]]
    if fields["ae_gmm_state"]["artifact"] and not fields["ae_gmm_state"]["artifact"]["exists"]:
        invalid_paths.append(fields["ae_gmm_state"]["artifact"])
    if "leakage" in required_fields and leakage_status != "PASS":
        missing.append("explicit_leakage_pass")
    status = "PASS" if not missing and not invalid_paths else "FAILED_VALIDATION" if invalid_paths else "MISSING_EVIDENCE"
    return {
        "layer": layer,
        "status": status,
        "leakage_status": leakage_status,
        "source_manifests": [str(document.path) for document in documents],
        "artifacts": artifacts,
        "invalid_paths": invalid_paths,
        "missing_evidence": sorted(set(missing)),
        **fields,
    }


def audit_long_chain(
    base_report_dir: Path,
    meta_report_dir: Path,
    simple_policy_dir: Path,
    portfolio_replay_dir: Path | None = None,
) -> dict[str, Any]:
    base = _documents(base_report_dir)
    meta = _documents(meta_report_dir)
    policy = _documents(simple_policy_dir)
    portfolio = _documents(portfolio_replay_dir)
    model_required = (
        "training_window_splits", "oos_scope", "target_labels", "feature_selection_recipe",
        "feature_count", "feature_hash", "hpo_scope", "hpo_trials", "hpo_params_hash",
        "ae_gmm_state", "cost_contract", "final_refit_exclusion", "leakage",
    )
    return {
        "schema_version": "long_chain_lineage_v1",
        "inputs": {
            "base_report_dir": str(base_report_dir),
            "meta_report_dir": str(meta_report_dir),
            "simple_policy_dir": str(simple_policy_dir),
            "portfolio_replay_dir": str(portfolio_replay_dir) if portfolio_replay_dir else None,
        },
        "rows": [
            _row("base", base, base_report_dir, model_required),
            _row("meta_long_residual_expert", meta, meta_report_dir, model_required),
            _row("side_archetype_ev_map", policy, simple_policy_dir, ("training_window_splits", "oos_scope", "target_labels", "cost_contract", "leakage")),
            _row("simple_policy_geometry", policy, simple_policy_dir, ("training_window_splits", "oos_scope", "cost_contract", "final_refit_exclusion", "geometry", "leakage")),
            _row("portfolio_policy", portfolio, portfolio_replay_dir, ("oos_scope", "cost_contract", "geometry", "leakage")),
        ],
    }


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    columns = [
        "layer", "status", "leakage_status", "source_manifests", "artifacts", "invalid_paths", "missing_evidence",
        "training_window_splits", "oos_scope", "target_labels", "feature_selection_recipe", "feature_count",
        "feature_hash", "hpo_scope", "hpo_trials", "hpo_params_hash", "ae_gmm_state", "cost_contract",
        "final_refit_exclusion", "geometry",
    ]
    return [{column: _json(row[column]) if isinstance(row.get(column), (dict, list)) else str(row.get(column, "")) for column in columns} for row in rows]


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Long-only Chain Lineage Audit", "", "| Layer | Status | Leakage | Missing evidence | Invalid paths |", "|---|---|---|---:|---:|"]
    for row in report["rows"]:
        lines.append(f"| {row['layer']} | {row['status']} | {row['leakage_status']} | {len(row['missing_evidence'])} | {len(row['invalid_paths'])} |")
    lines.extend(["", "## Evidence", ""])
    for row in report["rows"]:
        lines.append(f"### {row['layer']}")
        lines.append(f"- Source manifests: {', '.join(row['source_manifests']) or 'missing'}")
        lines.append(f"- Missing evidence: {', '.join(row['missing_evidence']) or 'none'}")
        if row["invalid_paths"]:
            lines.append("- Invalid referenced paths: " + ", ".join(item["declared_path"] for item in row["invalid_paths"]))
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{OUTPUT_BASENAME}.json"
    csv_path = output_dir / f"{OUTPUT_BASENAME}.csv"
    markdown_path = output_dir / f"{OUTPUT_BASENAME}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = _csv_rows(report["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(_markdown(report) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report-dir", type=Path, required=True)
    parser.add_argument("--meta-report-dir", type=Path, required=True)
    parser.add_argument("--simple-policy-dir", type=Path, required=True)
    parser.add_argument("--portfolio-replay-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_long_chain(
        args.base_report_dir, args.meta_report_dir, args.simple_policy_dir, args.portfolio_replay_dir
    )
    outputs = write_report(report, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
