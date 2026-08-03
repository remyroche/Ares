#!/usr/bin/env python3
"""Materialise exact current-policy enter-now and Wait10 labels for Feb--Apr.

The output is a training ledger, not a selected trading book.  It keeps the
canonical residual-top40 population unchanged, loads only exact point-in-time
pre-entry features, and uses the deployed simple-policy simulator for both
actions.  Work is checkpointed by exact-one-minute path row group so a long
historical replay can be resumed without accepting stale partial output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frozen_preentry_wait_action_ablation import (
    PARITY_FIELDS,
    _simulate,
    parse_paths,
)

POPULATION = (
    ROOT
    / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1"
    / "population.parquet"
)
PATH_ROOT = ROOT / "data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1"
POLICY_INPUT_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1"
)
LABEL_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1"
)
CONTEXT_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_historical_path_head_context_20260727_v1"
)
POLICY_PATH = (
    ROOT
    / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
    / "production_staging/best_policy_params.json"
)
OUT = (
    ROOT
    / "data_perp/artifacts/febapr2025_current_policy_wait10_action_20260730_v1"
)

SCHEMA = "febapr2025_current_policy_wait10_action_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
BASE_FEATURES = (
    "base_oof_score",
    "base_rank_timestamp_side",
    "base_group_rows",
    "base_rank_pct_timestamp_side",
)
STATE_FEATURES = (
    "hour_sin",
    "hour_cos",
    "range_12h_pct",
    "range_24h_pct",
    "volatility_zscore",
    "jump_intensity",
    "chop_score",
    "regime_stability_24h",
    "regime_transition_entropy_12h",
    "regime_transition_entropy_48h",
    "mkt_atr_expansion_1h",
    "breadth_accel_1h",
    "breadth_chg_1h",
    "cross_asset_corr_chg_1h",
    "correlation_breakdown_dispersion",
    "correlation_heterogeneity_dispersion",
    "cs_dispersion_ret_4h",
    "cs_dispersion_ret_24h",
    "leverage_build_score",
    "liquidation_onset_score",
    "liquidation_climax_score",
    "fragile_leverage_rebuild",
    "mark_perp_dislocation",
    "median_spread_bps",
    "spread_proxy_abs_return_bps_robust_z",
    "spread_proxy_body_bps_robust_z",
    "shock_12h",
    "shock_vol_ratio",
    "entropy_jump_24h",
    "complexity_regime_24h",
)
MODEL_FEATURES = (*BASE_FEATURES, *STATE_FEATURES)
WAIT_MINUTES = 10


class ContractError(RuntimeError):
    pass


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalize_symbol(value: pd.Series) -> pd.Series:
    return value.astype(str).str.replace("/", "_", regex=False)


def identity_digest(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True).astype(str)
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    digest = hashlib.sha256()
    for row in values.itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_source_manifests(
    *,
    population: Path,
    path_root: Path,
    policy_input_root: Path,
    label_root: Path,
    context_root: Path,
) -> dict[str, Any]:
    path_manifest = json.loads((path_root / "manifest.json").read_text())
    policy_manifest = json.loads((policy_input_root / "manifest.json").read_text())
    label_manifest = json.loads((label_root / "manifest.json").read_text())
    context_manifest = json.loads((context_root / "manifest.json").read_text())
    if (
        path_manifest.get("schema") != "execution_entry_timing_1m_paths_v1"
        or int(path_manifest.get("rows", {}).get("output", -1)) != 205_194
    ):
        raise ContractError("exact 720x1m path manifest mismatch")
    if policy_manifest.get("schema") != "historical_execution_ev_deployed_policy_inputs_v1":
        raise ContractError("deployed policy-input manifest mismatch")
    for record in policy_manifest.get("outputs", {}).values():
        path = ROOT / str(record["path"])
        if sha256(path) != record["sha256"]:
            raise ContractError(f"policy-input hash mismatch: {path}")
    label_record = label_manifest.get("output", {})
    if (
        label_manifest.get("schema") != "execution_ev_deployed_policy_1m_labels_v1"
        or sha256(label_root / "labels.parquet") != label_record.get("sha256")
    ):
        raise ContractError("current-policy label hash mismatch")
    index_record = context_manifest.get("context_index", {})
    if (
        context_manifest.get("schema")
        != "febapr2025_historical_path_head_context_v2_partitioned"
        or sha256(context_root / "context_index.parquet")
        != index_record.get("sha256")
    ):
        raise ContractError("historical PIT context index mismatch")
    return {
        "population_sha256": sha256(population),
        "paths_manifest_sha256": sha256(path_root / "manifest.json"),
        "paths_sha256": sha256(path_root / "paths.parquet"),
        "policy_inputs_manifest_sha256": sha256(policy_input_root / "manifest.json"),
        "labels_manifest_sha256": sha256(label_root / "manifest.json"),
        "labels_sha256": label_record["sha256"],
        "context_manifest_sha256": sha256(context_root / "manifest.json"),
        "context_index_sha256": index_record["sha256"],
    }


def load_context_features(context_root: Path) -> pd.DataFrame:
    index = pd.read_parquet(context_root / "context_index.parquet")
    required_index = {*IDENTITY, "shard_manifest"}
    if not required_index.issubset(index.columns):
        raise ContractError("context index is missing identity or shard lineage")
    pieces: list[pd.DataFrame] = []
    for manifest_name in sorted(index["shard_manifest"].astype(str).unique()):
        manifest_path = Path(manifest_name)
        manifest = json.loads(manifest_path.read_text())
        data_path = Path(manifest["output_path"])
        if (
            manifest.get("schema")
            != "febapr2025_historical_path_head_context_v2_partitioned"
            or sha256(data_path) != manifest.get("output_sha256")
        ):
            raise ContractError(f"context shard does not verify: {manifest_path}")
        schema = set(pq.ParquetFile(data_path).schema_arrow.names)
        missing = set((*IDENTITY, *MODEL_FEATURES)).difference(schema)
        if missing:
            raise ContractError(
                f"context shard lacks action features {sorted(missing)}: {data_path}"
            )
        pieces.append(
            pd.read_parquet(data_path, columns=[*IDENTITY, *MODEL_FEATURES])
        )
    features = pd.concat(pieces, ignore_index=True)
    features["__ts__"] = pd.to_datetime(features["__ts__"], utc=True)
    features["side_name"] = features["side_name"].astype(str).str.lower()
    if (
        len(features) != 205_194
        or features.duplicated(list(IDENTITY), keep=False).any()
        or identity_digest(features) != identity_digest(index)
    ):
        raise ContractError("historical context identity coverage changed")
    return features


def _join_by_candidate(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    candidate = right.loc[:, ["candidate_id", *columns]].copy()
    if candidate["candidate_id"].duplicated().any():
        raise ContractError(f"{name} candidate IDs are not unique")
    result = left.merge(candidate, on="candidate_id", how="left", validate="one_to_one")
    if result.loc[:, list(columns)].isna().all(axis=1).any():
        raise ContractError(f"{name} has incomplete population coverage")
    return result


def load_population_contract(
    *,
    population_path: Path,
    policy_input_root: Path,
    label_root: Path,
    context_root: Path,
) -> pd.DataFrame:
    population = pd.read_parquet(
        population_path,
        columns=[*IDENTITY, "__decision_ts__"],
    )
    population["__ts__"] = pd.to_datetime(population["__ts__"], utc=True)
    population["__decision_ts__"] = pd.to_datetime(
        population["__decision_ts__"], utc=True
    )
    population["side_name"] = population["side_name"].astype(str).str.lower()
    if (
        len(population) != 205_194
        or population.duplicated(list(IDENTITY), keep=False).any()
        or not population["__decision_ts__"].eq(
            population["__ts__"] + pd.Timedelta(hours=1)
        ).all()
    ):
        raise ContractError("canonical residual-top40 population changed")

    context = pd.read_parquet(
        policy_input_root / "context.parquet",
        columns=[*IDENTITY, "policy_archetype"],
    )
    targets = pd.read_parquet(
        policy_input_root / "path_targets.parquet",
        columns=[
            *IDENTITY,
            "__barrier_pct__",
            "__path_auxiliary_atr_fraction__",
        ],
    )
    reference_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_geometry_key",
        *PARITY_FIELDS,
        "execution_cost_return",
        "execution_exit_reason",
        "execution_label_end_utc",
        "execution_label_available_at",
    ]
    labels = pd.read_parquet(
        label_root / "labels.parquet", columns=reference_columns
    )
    features = load_context_features(context_root)
    result = _join_by_candidate(
        population, context, ["policy_archetype"], name="policy context"
    )
    result = _join_by_candidate(
        result,
        targets,
        ["__barrier_pct__", "__path_auxiliary_atr_fraction__"],
        name="policy target",
    )
    result = _join_by_candidate(
        result,
        labels,
        [name for name in reference_columns if name not in IDENTITY],
        name="current-policy labels",
    )
    result = result.merge(
        features,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if result.loc[:, list(MODEL_FEATURES)].isna().all(axis=1).any():
        raise ContractError("PIT features are missing for at least one candidate")
    for suffix in ("execution_decision_utc", "execution_label_end_utc", "execution_label_available_at"):
        result[suffix] = pd.to_datetime(result[suffix], utc=True)
    if not result["execution_decision_utc"].eq(result["__decision_ts__"]).all():
        raise ContractError("current-policy decision time changed")
    if not result["execution_label_end_utc"].eq(
        result["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ContractError("current-policy labels do not use the exact 12h deadline")
    return result


def simulate_path_batch(
    path_rows: pd.DataFrame,
    contract: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = path_rows.merge(
        contract,
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("_path", ""),
    )
    if rows["policy_archetype"].isna().any():
        raise ContractError("path batch is outside the canonical population")
    for name in ("side_name", "__ts__"):
        path_name = f"{name}_path"
        if name == "__ts__":
            equal = pd.to_datetime(rows[path_name], utc=True).eq(rows[name])
        else:
            equal = rows[path_name].astype(str).str.lower().eq(rows[name].astype(str))
        if not equal.all():
            raise ContractError(f"path batch {name} identity mismatch")
    if not normalize_symbol(rows["__symbol___path"]).eq(
        normalize_symbol(rows["__symbol__"])
    ).all():
        raise ContractError("path batch normalized symbol mismatch")
    rows["path_symbol"] = rows["__symbol___path"].astype(str)
    timestamps, arrays = parse_paths(rows["execution_future_path"])
    expected = rows["execution_decision_utc"].astype("int64").to_numpy()
    if not np.array_equal(timestamps[:, 0], expected):
        raise ContractError("path does not begin at the first executable minute")
    enter = _simulate(rows, arrays, policy, wait_minutes=0)
    wait = _simulate(rows, arrays, policy, wait_minutes=WAIT_MINUTES)

    parity: list[dict[str, Any]] = []
    for field in PARITY_FIELDS:
        delta = np.abs(
            enter[field].to_numpy(dtype=float)
            - rows[field].to_numpy(dtype=float)
        )
        parity.append(
            {
                "field": field,
                "rows": int(len(delta)),
                "mismatch_rows": int((delta > 1e-12).sum()),
                "max_abs_delta": float(delta.max(initial=0.0)),
            }
        )
    for field in ("execution_exit_reason", "execution_geometry_key"):
        mismatch = enter[field].astype(str).to_numpy() != rows[field].astype(str).to_numpy()
        parity.append(
            {
                "field": field,
                "rows": int(len(mismatch)),
                "mismatch_rows": int(mismatch.sum()),
                "max_abs_delta": np.nan,
            }
        )
    parity_frame = pd.DataFrame(parity)
    if parity_frame["mismatch_rows"].sum() != 0:
        raise ContractError(f"enter-now parity failed:\n{parity_frame}")

    output = rows.loc[:, list(IDENTITY)].copy()
    output["execution_decision_utc"] = rows["execution_decision_utc"].to_numpy()
    output["execution_label_end_utc"] = rows["execution_label_end_utc"].to_numpy()
    output["candidate_month"] = rows["__ts__"].dt.strftime("%Y-%m")
    for prefix, replay in (("enter_now", enter), ("wait10", wait)):
        output[f"{prefix}_gross"] = replay["execution_gross_ev_12h"].to_numpy(dtype=float)
        output[f"{prefix}_cost"] = replay["execution_cost_return"].to_numpy(dtype=float)
        output[f"{prefix}_net"] = replay["execution_net_ev_12h"].to_numpy(dtype=float)
        output[f"{prefix}_exit_reason"] = replay["execution_exit_reason"].astype(str).to_numpy()
        output[f"{prefix}_exit_hour"] = replay["execution_exit_hour"].to_numpy(dtype=float)
        output[f"{prefix}_mfe"] = replay["execution_mfe_return_12h"].to_numpy(dtype=float)
        output[f"{prefix}_mae"] = replay["execution_mae_return_12h"].to_numpy(dtype=float)
        reconciliation = np.abs(
            output[f"{prefix}_gross"]
            - output[f"{prefix}_cost"]
            - output[f"{prefix}_net"]
        )
        if (reconciliation > 1e-12).any():
            raise ContractError(f"{prefix} cost is not deducted exactly once")
    output["wait_delta"] = output["wait10_net"] - output["enter_now_net"]
    output["wait_better"] = output["wait_delta"].gt(0.0)
    output["wait_action_entry_utc"] = (
        output["execution_decision_utc"] + pd.Timedelta(minutes=WAIT_MINUTES)
    )
    return output, parity_frame


def completed_batch(data_path: Path, manifest_path: Path, source_hash: str) -> bool:
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(
        manifest.get("schema") == SCHEMA
        and manifest.get("source_identity_sha256") == source_hash
        and manifest.get("output_sha256") == sha256(data_path)
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.with_name(f".{args.output.name}.work")
    work.mkdir(parents=True, exist_ok=True)
    (work / "batches").mkdir(exist_ok=True)
    provenance = verify_source_manifests(
        population=args.population,
        path_root=args.path_root,
        policy_input_root=args.policy_input_root,
        label_root=args.label_root,
        context_root=args.context_root,
    )
    contract = load_population_contract(
        population_path=args.population,
        policy_input_root=args.policy_input_root,
        label_root=args.label_root,
        context_root=args.context_root,
    )
    contract_by_id = contract.set_index("candidate_id", drop=False)
    policy = json.loads(args.policy_path.read_text())
    path_file = pq.ParquetFile(args.path_root / "paths.parquet")
    batch_files: list[Path] = []
    parity_files: list[Path] = []
    for row_group in range(path_file.num_row_groups):
        path_rows = path_file.read_row_group(row_group).to_pandas()
        source_hash = identity_digest(
            path_rows.assign(
                __symbol__=normalize_symbol(path_rows["__symbol__"]),
                __ts__=pd.to_datetime(path_rows["__ts__"], utc=True),
                side_name=path_rows["side_name"].astype(str).str.lower(),
            )
        )
        data_path = work / "batches" / f"{row_group:04d}.parquet"
        parity_path = work / "batches" / f"{row_group:04d}.parity.csv"
        manifest_path = work / "batches" / f"{row_group:04d}.manifest.json"
        if not completed_batch(data_path, manifest_path, source_hash):
            ids = path_rows["candidate_id"].astype(str)
            if not ids.isin(contract_by_id.index).all():
                raise ContractError(f"row group {row_group} contains unknown candidates")
            local_contract = contract_by_id.loc[ids].reset_index(drop=True)
            labels, parity = simulate_path_batch(path_rows, local_contract, policy)
            temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
            labels.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, data_path)
            parity.to_csv(parity_path, index=False)
            write_json(
                manifest_path,
                {
                    "schema": SCHEMA,
                    "row_group": row_group,
                    "rows": int(len(labels)),
                    "source_identity_sha256": source_hash,
                    "output_sha256": sha256(data_path),
                    "parity_sha256": sha256(parity_path),
                },
            )
        batch_files.append(data_path)
        parity_files.append(parity_path)

    labels = pd.concat(
        [pd.read_parquet(path) for path in batch_files], ignore_index=True
    ).sort_values(["execution_decision_utc", "candidate_id"], kind="stable")
    features = contract.loc[:, [*IDENTITY, *MODEL_FEATURES]].copy()
    features = features.sort_values(["__ts__", "candidate_id"], kind="stable")
    if (
        len(labels) != len(contract)
        or labels.duplicated(list(IDENTITY), keep=False).any()
        or identity_digest(labels) != identity_digest(contract)
        or identity_digest(features) != identity_digest(contract)
    ):
        raise ContractError("final label/feature identity coverage changed")
    parity = pd.concat(
        [pd.read_csv(path) for path in parity_files], ignore_index=True
    )
    parity_summary = (
        parity.groupby("field", sort=True)
        .agg(
            rows=("rows", "sum"),
            mismatch_rows=("mismatch_rows", "sum"),
            max_abs_delta=("max_abs_delta", "max"),
        )
        .reset_index()
    )
    if parity_summary["mismatch_rows"].sum() != 0:
        raise ContractError("full-population current-policy parity failed")

    final = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        labels.to_parquet(final / "action_labels.parquet", index=False, compression="zstd")
        features.to_parquet(final / "preentry_features.parquet", index=False, compression="zstd")
        parity_summary.to_csv(final / "control_parity.csv", index=False)
        roles = {
            "schema": SCHEMA,
            "model_inputs": list(MODEL_FEATURES),
            "base_context": list(BASE_FEATURES),
            "market_and_transition_context": list(STATE_FEATURES),
            "target_only": [
                "enter_now_gross",
                "enter_now_cost",
                "enter_now_net",
                "wait10_gross",
                "wait10_cost",
                "wait10_net",
                "wait_delta",
                "wait_better",
                "execution_label_end_utc",
            ],
            "forbidden_model_inputs": [
                "execution_future_path",
                "__barrier_pct__",
                "__path_auxiliary_atr_fraction__",
                "policy_archetype",
                "candidate_month",
                "side_name",
                "candidate_id",
                "__symbol__",
                "__ts__",
            ],
        }
        write_json(final / "feature_roles.json", roles)
        outputs = {
            name: sha256(final / name)
            for name in (
                "action_labels.parquet",
                "preentry_features.parquet",
                "control_parity.csv",
                "feature_roles.json",
            )
        }
        by_month_side = (
            labels.groupby(["candidate_month", "side_name"], sort=True)
            .agg(
                rows=("candidate_id", "size"),
                wait_better_rate=("wait_better", "mean"),
                mean_wait_delta=("wait_delta", "mean"),
            )
            .reset_index()
            .to_dict("records")
        )
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_TRAINING_LEDGER_CURRENT_POLICY_EXACT_PIT_NO_SELECTION",
            "rows": int(len(labels)),
            "identity_sha256": identity_digest(labels),
            "rows_by_month_side": by_month_side,
            "contract": {
                "population": "unchanged canonical residual-top40 rows; this is not a selected global trading book",
                "features": "exact signal-time PIT fields only; no future paths, targets, policy geometry or calendar month are model inputs",
                "enter_now": "exact deployed simple-policy replay with full-row parity to sealed current-policy labels",
                "wait10": "no position for minutes 0--9; entry at minute-10 open; same barrier/strategy; remaining 710 minutes to the original absolute 12h deadline; action costs recomputed once",
                "use": "older-data action-head training and transfer diagnosis only; frozen March/April selected-book evaluation remains separate",
            },
            "feature_count": len(MODEL_FEATURES),
            "input_provenance": {
                **provenance,
                "policy_sha256": sha256(args.policy_path),
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(final / "manifest.json", manifest)
        (final / "manifest.sha256").write_text(
            f"{sha256(final / 'manifest.json')}  manifest.json\n"
        )
        os.replace(final, args.output)
    except Exception:
        shutil.rmtree(final, ignore_errors=True)
        raise
    shutil.rmtree(work)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--population", type=Path, default=POPULATION)
    result.add_argument("--path-root", type=Path, default=PATH_ROOT)
    result.add_argument("--policy-input-root", type=Path, default=POLICY_INPUT_ROOT)
    result.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    result.add_argument("--context-root", type=Path, default=CONTEXT_ROOT)
    result.add_argument("--policy-path", type=Path, default=POLICY_PATH)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(safe(run(args)), indent=2))


if __name__ == "__main__":
    main()
