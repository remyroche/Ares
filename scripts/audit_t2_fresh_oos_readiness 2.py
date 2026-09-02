#!/usr/bin/env python3
"""Fail-closed readiness audit for the untouched Feb--Apr 2025 T2 evaluation.

The original configuration replay cannot be used as a fresh result.  This
audit establishes whether a later cohort can be replayed *without changing*
the frozen 361-field causal feature contract.  It intentionally does not fit
or score a model, and it rejects a reduced-feature substitute.

The audit has three independent gates:

* accepted exact-policy candidate identities and H12 economics;
* complete immutable one-minute native paths at decision through H12;
* availability of every frozen raw causal feature in the point-in-time
  historical feature ledgers.

The last gate is deliberately schema-level here.  A subsequent materializer
must also emit the per-feature availability and dependency lineage registry
before an OOS result can be called promotion-eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_POPULATION = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v1/population.parquet"
DEFAULT_PATH_COMPLETION = ROOT / "data_perp/artifacts/febapr2025_native_first_touch_full_12h_paths_20260729_v2/completion.json"
DEFAULT_FEATURES = ROOT / "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/frozen_raw_causal_features.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/t2_fresh_oos_readiness_20260801_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_population(path: Path) -> pd.DataFrame:
    needed = [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "execution_label_available_at_utc", "feature_source_ledger"]
    frame = pd.read_parquet(path, columns=needed)
    if frame.empty or frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("frozen population must be nonempty with unique exact identities")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    decision = pd.to_datetime(frame["execution_decision_utc"], utc=True, errors="raise")
    end = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
    available = pd.to_datetime(frame["execution_label_available_at_utc"], utc=True, errors="raise")
    if not decision.eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("fresh population violates the next-hour entry contract")
    if not end.eq(decision + pd.Timedelta(hours=12)).all() or not available.eq(end).all():
        raise ValueError("fresh population violates the H12 label-resolution contract")
    return frame


def _ledger_audit(population: pd.DataFrame, frozen: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledgers = sorted({Path(item) for item in population["feature_source_ledger"].astype(str)})
    schemas: list[dict[str, Any]] = []
    identities: list[pd.DataFrame] = []
    available: set[str] = set()
    for ledger in ledgers:
        parquet = pq.ParquetFile(ledger)
        columns = list(parquet.schema.names)
        available.update(columns)
        missing_identity = sorted(set(IDENTITY).difference(columns))
        if missing_identity:
            raise ValueError(f"feature ledger has no exact identity: {ledger}: {missing_identity}")
        identity = pd.read_parquet(ledger, columns=list(IDENTITY))
        identity["__ts__"] = pd.to_datetime(identity["__ts__"], utc=True, errors="raise")
        if identity.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"feature ledger has duplicate exact identities: {ledger}")
        identities.append(identity)
        schemas.append({
            "feature_source_ledger": str(ledger),
            "rows": int(parquet.metadata.num_rows),
            "columns": len(columns),
            "frozen_features_present": int(sum(name in columns for name in frozen)),
            "sha256": _sha256(ledger),
        })
    joined = pd.concat(identities, ignore_index=True)
    # The old feature ledgers use slash-form symbols (BTC/USD:USD), whereas
    # the exact-policy population intentionally uses the exchange-store form
    # (BTC_USD:USD).  Candidate ID is generated with the former and is the
    # stable join key.  Treat the schema convention as a declared normalising
    # transform, never as a permissive partial join.
    if joined["candidate_id"].duplicated().any():
        raise ValueError("historical feature ledgers have duplicate candidate IDs")
    population_key = population.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
    joined_key = joined.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
    matched = population_key.merge(joined_key, on="candidate_id", how="left", suffixes=("_population", "_ledger"), validate="one_to_one")
    normalised_ledger_symbol = matched["__symbol___ledger"].astype("string").str.replace("/", "_", regex=False)
    exact_identity = (
        len(matched) == len(population_key)
        and matched["__ts___ledger"].notna().all()
        and matched["__ts___population"].eq(matched["__ts___ledger"]).all()
        and matched["side_name_population"].astype(str).str.lower().eq(matched["side_name_ledger"].astype(str).str.lower()).all()
        and matched["__symbol___population"].astype(str).eq(normalised_ledger_symbol.astype(str)).all()
    )
    if not exact_identity:
        raise ValueError("historical feature ledgers fail the candidate-ID/time/side/normalised-symbol join")
    contract = pd.DataFrame({"feature": frozen})
    contract["available_in_archived_2025_ledger"] = contract["feature"].isin(available)
    contract["next_action"] = contract["available_in_archived_2025_ledger"].map({
        True: "read_from_point_in_time_ledger_and_prove_per_feature_lineage",
        False: "recompute_from_historical_sources_then_prove_parity_and_lineage",
    })
    return contract, pd.DataFrame(schemas), {
        "ledger_count": len(ledgers),
        "source_rows": int(len(joined_key)),
        "source_rows_not_in_fresh_population": int(len(joined_key) - len(population_key)),
        "identity_exact": bool(exact_identity),
        "frozen_feature_count": len(frozen),
        "frozen_features_available": int(contract["available_in_archived_2025_ledger"].sum()),
        "frozen_features_missing": int((~contract["available_in_archived_2025_ledger"]).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit artifact: {args.output_dir}")
    population = _read_population(args.population)
    completion = json.loads(args.path_completion.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE" or not completion.get("all_windows_complete"):
        raise ValueError("full exact one-minute path completion gate has not passed")
    path_identity = completion.get("identity_contract", {})
    if int(path_identity.get("candidate_inputs_rows", -1)) != len(population) or not path_identity.get("no_missing_or_extra"):
        raise ValueError("complete path artifact identity does not match fresh population")
    frozen_payload = json.loads(args.features_json.read_text(encoding="utf-8"))
    frozen = [str(name) for name in frozen_payload.get("raw_feature_columns", [])]
    if len(frozen) != 361 or len(set(frozen)) != len(frozen):
        raise ValueError("expected exactly 361 unique frozen raw causal features")
    contract, schemas, ledger = _ledger_audit(population, frozen)
    args.output_dir.mkdir(parents=True)
    contract.to_csv(args.output_dir / "frozen_feature_contract_coverage.csv", index=False)
    schemas.to_csv(args.output_dir / "source_ledger_schema.csv", index=False)
    all_features = contract["available_in_archived_2025_ledger"].all()
    result = {
        "schema": "t2_fresh_oos_readiness_v1",
        "status": (
            "READY_FOR_361_FEATURE_MATERIALISATION_AND_LINEAGE_AUDIT"
            if all_features
            else "BLOCKED_FROZEN_361_FEATURE_CONTRACT_NOT_MATERIALISED"
        ),
        "purpose": "preflight only; no model has been trained or evaluated",
        "population": {
            "path": str(args.population), "sha256": _sha256(args.population), "rows": len(population),
            "timing": "feature cutoff at signal close; decision/entry = signal + 1h; labels available = decision + 12h",
        },
        "exact_1m_paths": {
            "completion": str(args.path_completion), "sha256": _sha256(args.path_completion),
            "rows": int(completion["rows"]), "shards": int(completion["shards"]),
            "all_windows_complete": True,
        },
        "frozen_feature_contract": {
            "path": str(args.features_json), "sha256": _sha256(args.features_json),
            **ledger,
            "coverage_csv": str(args.output_dir / "frozen_feature_contract_coverage.csv"),
            "schema_csv": str(args.output_dir / "source_ledger_schema.csv"),
        },
        "hard_rules": [
            "Do not evaluate the frozen 361-feature model on an available-only reduced panel.",
            "Do not admit execution_cost_return or any realised target/path field as an inference input.",
            "A recomputed missing feature needs an entry-time availability and dependency-lineage record before use.",
            "A fresh evaluation remains unopened until all 361 fields are materialised under one exact feature contract.",
        ],
        "next_step": (
            "Materialise the missing frozen features from their historical causal sources, then run feature-value parity and per-feature lineage audits."
            if not all_features else
            "Run the per-feature lineage audit, then build the exact 361-field fresh evaluation ledger."
        ),
    }
    _write_json(args.output_dir / "readiness.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--path-completion", type=Path, default=DEFAULT_PATH_COMPLETION)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
