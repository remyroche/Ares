#!/usr/bin/env python3
"""Prove deployed-policy replay parity on the current exact-label overlap.

Historical labels may only be accepted when the exact simulator contract has
first reproduced the current May--July candidate labels on the same candidate
identities.  This runner replays a deterministic, side/month-stratified sample
from the immutable current label artifact and fails closed on any categorical
or numerical discrepancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
NUMERIC_COLUMNS = (
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_entry_price",
    "execution_exit_price",
    "execution_expected_spread_bps",
    "execution_entry_half_spread_bps",
    "execution_exit_half_spread_bps",
)
CATEGORICAL_COLUMNS = (
    "policy_archetype",
    "execution_geometry_key",
    "execution_geometry_source",
    "execution_exit_reason",
)
DEFAULT_REFERENCE_MANIFEST = ROOT / (
    "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
    "manifest.json"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/deployed_policy_label_parity_20260727_v1"


def _sha256(path: Path) -> str:
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
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def select_stratified_sample(frame: pd.DataFrame, *, per_side_month: int) -> pd.DataFrame:
    """Choose an order-stable sample from every represented side/month stratum."""

    if per_side_month < 1:
        raise ValueError("per_side_month must be positive")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["candidate_month"] = work["__ts__"].dt.strftime("%Y-%m")
    selected: list[pd.DataFrame] = []
    for _, group in work.groupby(["candidate_month", "side_name"], sort=True):
        ordered = group.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
        positions = np.unique(
            np.linspace(0, len(ordered) - 1, min(len(ordered), per_side_month), dtype=int)
        )
        selected.append(ordered.iloc[positions])
    return pd.concat(selected, ignore_index=True).drop(columns="candidate_month")


def compare_label_frames(
    replayed: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    atol: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join exact candidate labels and return field-level difference evidence."""

    joined = replayed.merge(
        reference,
        on=list(IDENTITY),
        suffixes=("__replayed", "__reference"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    diagnostics: list[dict[str, Any]] = []
    for column in NUMERIC_COLUMNS:
        left = pd.to_numeric(joined[f"{column}__replayed"], errors="coerce")
        right = pd.to_numeric(joined[f"{column}__reference"], errors="coerce")
        delta = (left - right).abs()
        diagnostics.append(
            {
                "field": column,
                "kind": "numeric",
                "compared_rows": int((left.notna() & right.notna()).sum()),
                "mismatch_rows": int(delta.gt(float(atol)).sum()),
                "max_abs_delta": float(delta.max()) if delta.notna().any() else np.nan,
            }
        )
    for column in CATEGORICAL_COLUMNS:
        left = joined[f"{column}__replayed"].astype("string")
        right = joined[f"{column}__reference"].astype("string")
        diagnostics.append(
            {
                "field": column,
                "kind": "categorical",
                "compared_rows": int((left.notna() & right.notna()).sum()),
                "mismatch_rows": int((left != right).fillna(True).sum()),
                "max_abs_delta": np.nan,
            }
        )
    comparison = pd.DataFrame(diagnostics)
    accounting = (
        pd.to_numeric(replayed["execution_gross_ev_12h"], errors="raise")
        - pd.to_numeric(replayed["execution_cost_return"], errors="raise")
        - pd.to_numeric(replayed["execution_net_ev_12h"], errors="raise")
    ).abs()
    summary = {
        "replayed_rows": int(len(replayed)),
        "reference_rows": int(len(reference)),
        "identity_inner_rows": int(joined["_merge"].eq("both").sum()),
        "identity_mismatch_rows": int((~joined["_merge"].eq("both")).sum()),
        "field_mismatch_rows": int(comparison["mismatch_rows"].sum()),
        "max_numeric_abs_delta": float(
            pd.to_numeric(comparison.loc[comparison["kind"].eq("numeric"), "max_abs_delta"], errors="coerce").max()
        ),
        "replayed_accounting_max_abs_error": float(accounting.max()),
    }
    summary["parity_pass"] = bool(
        summary["identity_mismatch_rows"] == 0
        and summary["field_mismatch_rows"] == 0
        and summary["replayed_accounting_max_abs_error"] <= float(atol)
    )
    return comparison, summary


def _reference_contract(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Path]:
    source = manifest.get("source", {})
    output = manifest.get("output", {})
    accounting = manifest.get("accounting", {})
    geometry = manifest.get("geometry", {})
    required = ("candidates", "context", "path_targets", "policy")
    missing = [name for name in required if not source.get(name)]
    if missing or not output.get("path"):
        raise ValueError(f"reference manifest lacks source contract paths: {missing}")
    if accounting.get("simulator") != (
        "extreme_price_movements.simple_policy_optimiser.simulate_and_score"
    ):
        raise ValueError("reference manifest does not use deployed policy simulator")
    if float(geometry.get("fallback_rate", -1.0)) != 1.0 or int(
        geometry.get("side_archetype_rows", -1)
    ) != 0:
        raise ValueError("reference does not prove the current universal fallback geometry")
    paths = {name: ROOT / str(source[name]) for name in required}
    paths["labels"] = ROOT / str(output["path"])
    paths["spread"] = ROOT / str(accounting.get("spread_baseline", ""))
    if not paths["spread"].exists():
        raise ValueError("reference spread baseline path is unavailable")
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"reference {name} path is unavailable: {path}")
    if str(source.get("policy_sha256", "")) != _sha256(paths["policy"]):
        raise ValueError("reference policy hash does not match its policy file")
    return paths


def _replay_sample(
    sample: pd.DataFrame,
    *,
    paths: Mapping[str, Path],
    horizon_minutes: int,
    data_root: Path,
) -> pd.DataFrame:
    # The optimiser reads this setting at replay time.  Set it before importing
    # the shared materializer so parity cannot silently use another baseline.
    os.environ["EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH"] = str(paths["spread"].resolve())
    from scripts import materialize_execution_ev_policy_labels as labels  # noqa: PLC0415

    policy, exit_contract = labels._policy_contract(
        paths["policy"], horizon_minutes_override=horizon_minutes
    )
    candidate_rows = labels._load_candidates(
        paths["candidates"],
        paths["context"],
        paths["path_targets"],
        decision_delay_minutes=60,
        allow_subset=True,
    )
    candidate_rows, _ = labels._resolved_geometry(candidate_rows, policy)
    selected = candidate_rows.merge(
        sample.loc[:, list(IDENTITY)], on=list(IDENTITY), how="inner", validate="one_to_one"
    ).sort_values(list(IDENTITY), kind="stable")
    if len(selected) != len(sample):
        raise ValueError("sample identities are absent from replay inputs")
    strategy_by_key = {
        str(item.get("canonical_strategy_id")): item
        for item in policy["strategies"]
        if isinstance(item, Mapping) and item.get("selected", True)
    }
    parts: list[pd.DataFrame] = []
    for symbol, indices in selected.groupby("__symbol__", sort=True).groups.items():
        rows = selected.loc[list(indices)].copy().reset_index(drop=True)
        start = rows["__decision_ts__"].min()
        end = rows["__decision_ts__"].max() + pd.Timedelta(minutes=horizon_minutes)
        bars = labels._load_symbol_bars(data_root, str(symbol), start, end)
        grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
        values = bars.reindex(grid).loc[:, list(labels.PATH_COLUMNS)].to_numpy(dtype=np.float32)
        offsets = ((rows["__decision_ts__"] - start) / pd.Timedelta(minutes=1)).astype(np.int64).to_numpy()
        matrices = tuple(
            np.stack([values[offset : offset + horizon_minutes, column] for offset in offsets])
            for column in range(len(labels.PATH_COLUMNS))
        )
        for key, local_indices in rows.groupby("execution_geometry_key", sort=True).groups.items():
            positions = np.asarray(list(local_indices), dtype=np.int64)
            strategy = strategy_by_key.get(str(key))
            if strategy is None:
                raise ValueError(f"reference geometry key is no longer present: {key}")
            simulated = labels._simulate_batch(
                rows.iloc[positions].reset_index(drop=True),
                tuple(matrix[positions] for matrix in matrices),
                strategy,
            )
            source = rows.iloc[positions].reset_index(drop=True)
            parts.append(
                pd.concat(
                    [
                        source.loc[:, [*IDENTITY, "policy_archetype", "execution_geometry_key", "execution_geometry_source"]],
                        simulated,
                    ],
                    axis=1,
                )
            )
    replayed = pd.concat(parts, ignore_index=True).sort_values(list(IDENTITY), kind="stable")
    if int(exit_contract["horizon_minutes"]) != int(horizon_minutes):
        raise ValueError("replay horizon differs from requested reference horizon")
    return replayed.reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    paths = _reference_contract(manifest, manifest_path=args.reference_manifest)
    reference = pd.read_parquet(paths["labels"], columns=[*IDENTITY, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS])
    sample = select_stratified_sample(reference, per_side_month=args.per_side_month)
    replayed = _replay_sample(
        sample,
        paths=paths,
        horizon_minutes=args.horizon_minutes,
        data_root=args.data_root,
    )
    comparison, summary = compare_label_frames(replayed, sample, atol=args.atol)
    args.output_dir.mkdir(parents=True)
    comparison_path = args.output_dir / "field_comparison.csv"
    sample_path = args.output_dir / "sample_identities.parquet"
    evidence_path = args.output_dir / "evidence_gate.json"
    comparison.to_csv(comparison_path, index=False)
    sample.loc[:, list(IDENTITY)].to_parquet(sample_path, index=False, compression="zstd")
    evidence = {
        "schema": "deployed_policy_current_overlap_parity_gate_v1",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": _sha256(args.reference_manifest),
        "reference_policy_sha256": _sha256(paths["policy"]),
        "reference_spread_baseline_sha256": _sha256(paths["spread"]),
        "geometry": {
            "current_side_archetype_rows": int(manifest["geometry"]["side_archetype_rows"]),
            "current_side_parent_fallback_rows": int(manifest["geometry"]["side_parent_fallback_rows"]),
            "current_fallback_rate": float(manifest["geometry"]["fallback_rate"]),
            "conclusion": "current observable candidates resolve to the side-parent fallback",
        },
        "sample": {
            "per_side_month": int(args.per_side_month),
            "rows": int(len(sample)),
            "by_side_month": sample.assign(month=pd.to_datetime(sample["__ts__"], utc=True).dt.strftime("%Y-%m")).groupby(["month", "side_name"], sort=True).size().to_dict(),
        },
        "comparison": summary,
        "artifacts": {
            "field_comparison": str(comparison_path),
            "sample_identities": str(sample_path),
        },
        "historical_acceptance_prerequisite_passed": bool(summary["parity_pass"]),
    }
    _write_json(evidence_path, evidence)
    if not summary["parity_pass"]:
        raise ValueError(f"deployed-policy current-overlap parity failed; inspect {evidence_path}")
    return {"evidence": evidence_path, "comparison": comparison_path, "sample": sample_path}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    result.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    result.add_argument("--horizon-minutes", type=int, default=720)
    result.add_argument("--per-side-month", type=int, default=16)
    result.add_argument("--atol", type=float, default=1e-10)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    if options.horizon_minutes <= 0 or options.per_side_month < 1 or options.atol < 0.0:
        raise ValueError("horizon, sample count and tolerance must be nonnegative/positive")
    outputs = run(options)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
