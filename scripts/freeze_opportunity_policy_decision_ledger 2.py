#!/usr/bin/env python3
"""Freeze a content-addressed policy descriptor and immutable decision ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / (
    "configs/research_execution_policy_decision_contract_20260729_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            safe(payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def verify_sources(specification: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, source in specification["sources"].items():
        path = (ROOT / source["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256(path) != source["sha256"]:
            raise ValueError(f"policy-contract source hash changed: {key}")
        paths[key] = path
    return paths


def build_decision_ledger(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    policy_id: str,
) -> pd.DataFrame:
    required_candidates = {
        "candidate_id",
        "timestamp",
        "symbol",
        "side",
        "normalized_rank_score",
        "source_score_timestamp",
        "oof_fold",
        "net_return",
        "gross_return",
        "exit_timestamp",
        "simple_policy_exit_reason",
    }
    required_decisions = {
        "candidate_index",
        "accepted",
        "rejection_reason",
        "position_size",
        "open_positions_before",
        "open_positions_after",
        "wallet_before",
        "wallet_after",
        "position_exit_timestamp",
        "position_net_return",
        "position_gross_return",
        "position_exit_reason",
    }
    missing_candidates = sorted(required_candidates.difference(candidates.columns))
    missing_decisions = sorted(required_decisions.difference(decisions.columns))
    if missing_candidates or missing_decisions:
        raise ValueError(
            "decision-ledger contract missing "
            f"candidates={missing_candidates}, decisions={missing_decisions}"
        )
    work = candidates.copy().reset_index(drop=True)
    if work["candidate_id"].astype(str).duplicated().any():
        raise ValueError("decision-ledger candidate IDs must be unique")
    local_decisions = decisions.copy()
    index = pd.to_numeric(
        local_decisions["candidate_index"], errors="raise"
    ).astype(int)
    if index.duplicated().any() or set(index) != set(range(len(work))):
        raise ValueError("portfolio decisions must cover every candidate index once")
    local_decisions = local_decisions.assign(candidate_index=index).sort_values(
        "candidate_index", kind="stable"
    )
    for column in ("timestamp", "source_score_timestamp", "exit_timestamp"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    for column in ("position_exit_timestamp",):
        local_decisions[column] = pd.to_datetime(
            local_decisions[column], utc=True, errors="raise"
        )
    result = work.loc[
        :,
        [
            "candidate_id",
            "timestamp",
            "source_score_timestamp",
            "symbol",
            "side",
            "normalized_rank_score",
            "oof_fold",
            "net_return",
            "gross_return",
            "exit_timestamp",
            "simple_policy_exit_reason",
        ],
    ].copy()
    result = result.rename(
        columns={
            "timestamp": "decision_utc",
            "source_score_timestamp": "signal_utc",
            "symbol": "asset",
            "side": "side_name",
            "normalized_rank_score": "mapped_rank_score",
            "oof_fold": "evaluation_fold",
            "net_return": "candidate_net_return",
            "gross_return": "candidate_gross_return",
            "exit_timestamp": "candidate_exit_utc",
            "simple_policy_exit_reason": "candidate_exit_reason",
        }
    )
    decision_columns = [
        "accepted",
        "rejection_reason",
        "position_size",
        "open_positions_before",
        "open_positions_after",
        "wallet_before",
        "wallet_after",
        "position_exit_timestamp",
        "position_net_return",
        "position_gross_return",
        "position_exit_reason",
    ]
    for column in decision_columns:
        result[f"portfolio_{column}"] = local_decisions[column].to_numpy()
    result.insert(0, "policy_id", policy_id)
    result.insert(
        1,
        "decision_id",
        [
            hashlib.sha256(
                (
                    f"{policy_id}|{candidate_id}|{stamp.isoformat()}"
                ).encode("utf-8")
            ).hexdigest()
            for candidate_id, stamp in zip(
                result["candidate_id"].astype(str),
                result["decision_utc"],
                strict=True,
            )
        ],
    )
    result["global_top10_selected"] = True
    result["evaluation_role"] = "retrospective_model_oof"
    if result["decision_id"].duplicated().any():
        raise ValueError("decision IDs must be unique")
    if not result["decision_utc"].eq(
        result["signal_utc"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError("decision ledger violates signal+1h timing")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    specification_path = Path(args.specification)
    specification = json.loads(
        specification_path.read_text(encoding="utf-8")
    )
    if specification.get("status") not in {
        "RESEARCH_SNAPSHOT_NOT_INCUMBENT",
        "FROZEN_PROSPECTIVE_INCUMBENT",
    }:
        raise ValueError("unknown policy-contract status")
    if (
        specification["status"] == "FROZEN_PROSPECTIVE_INCUMBENT"
        and not specification.get("promotion_eligible", False)
    ):
        raise ValueError("prospective incumbent contract must declare eligibility")
    sources = verify_sources(specification)
    candidates = pd.read_parquet(sources["global_topk_candidates"])
    decisions = pd.read_parquet(sources["baseline_portfolio_decisions"])
    candidate_time = pd.to_datetime(candidates["timestamp"], utc=True, errors="raise")
    contract = {
        **{
            key: value
            for key, value in specification.items()
            if key != "sources"
        },
        "source_contracts": specification["sources"],
        "materialized_universe": sorted(candidates["symbol"].astype(str).unique()),
        "decision_calendar": {
            "start": candidate_time.min(),
            "end": candidate_time.max(),
            "candidate_rows": int(len(candidates)),
        },
    }
    policy_id = canonical_hash(contract)
    contract["policy_id"] = policy_id
    ledger = build_decision_ledger(candidates, decisions, policy_id)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    contract_path = output / "policy_contract.json"
    ledger_path = output / "immutable_decision_ledger.parquet"
    contract_path.write_text(
        json.dumps(safe(contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    manifest = {
        "schema": "opportunity_policy_decision_ledger_v1",
        "status": specification["status"],
        "promotion_eligible": bool(specification["promotion_eligible"]),
        "policy_id": policy_id,
        "decision_rows": int(len(ledger)),
        "portfolio_accepted_rows": int(
            ledger["portfolio_accepted"].astype(bool).sum()
        ),
        "prospective_rows": int(
            ledger["evaluation_role"].eq("prospective_forward_oos").sum()
        ),
        "calendar": {
            "start": ledger["decision_utc"].min(),
            "end": ledger["decision_utc"].max(),
        },
        "outputs": {
            "policy_contract": {
                "path": str(contract_path.resolve()),
                "sha256": sha256(contract_path),
            },
            "decision_ledger": {
                "path": str(ledger_path.resolve()),
                "sha256": sha256(ledger_path),
            },
        },
        "specification": {
            "path": str(specification_path.resolve()),
            "sha256": sha256(specification_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.sha256").write_text(
        sha256(manifest_path) + "\n", encoding="utf-8"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--specification", type=Path, default=DEFAULT_SPEC)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    print(json.dumps(safe(run(parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
