#!/usr/bin/env python3
"""Replace mixed/superseded outcomes with one canonical exact 1m policy ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "canonical_exact_policy_regime_input_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
OUTCOME_COLUMNS = (
    "execution_decision_utc",
    "execution_label_end_utc",
    "execution_net_ev_12h",
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_exit_reason",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def replace_outcomes(
    features: pd.DataFrame,
    exact_policy: pd.DataFrame,
    atr: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    identity = list(IDENTITY)
    for name, frame in (
        ("features", features),
        ("exact policy", exact_policy),
        ("ATR", atr),
    ):
        missing_identity = sorted(set(identity) - set(frame.columns))
        if missing_identity:
            raise ValueError(f"{name} missing identity: {missing_identity}")
        if frame.duplicated(identity).any():
            raise ValueError(f"{name} contains duplicate identities")
    missing_outcomes = sorted(set(OUTCOME_COLUMNS) - set(exact_policy.columns))
    if missing_outcomes:
        raise ValueError("exact policy missing outcomes: " + ", ".join(missing_outcomes))
    atr_column = next(
        (
            column
            for column in (
                "oof_entry_atr_fraction",
                "__path_auxiliary_atr_fraction__",
            )
            if column in atr
        ),
        None,
    )
    if atr_column is None:
        raise ValueError("ATR source is missing a causal entry ATR fraction")
    old_target = (
        features.loc[:, [*identity, "execution_net_ev_12h"]].copy()
        if "execution_net_ev_12h" in features
        else None
    )
    feature_columns = [
        column for column in features.columns if column not in OUTCOME_COLUMNS
    ]
    joined = features.loc[:, feature_columns].merge(
        exact_policy.loc[:, [*identity, *OUTCOME_COLUMNS]],
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    if "oof_entry_atr_fraction" not in joined:
        atr_values = atr.loc[:, [*identity, atr_column]].rename(
            columns={atr_column: "oof_entry_atr_fraction"}
        )
        joined = joined.merge(
            atr_values,
            on=identity,
            how="left",
            validate="one_to_one",
        )
    accounting_delta = (
        joined["execution_gross_ev_12h"].to_numpy(dtype=float)
        - joined["execution_cost_return"].to_numpy(dtype=float)
        - joined["execution_net_ev_12h"].to_numpy(dtype=float)
    )
    max_accounting_delta = float(np.max(np.abs(accounting_delta)))
    if max_accounting_delta > 1e-7:
        raise ValueError(f"exact policy accounting mismatch: {max_accounting_delta}")
    target_change = {}
    if old_target is not None:
        paired = old_target.merge(
            joined.loc[:, [*identity, "execution_net_ev_12h"]],
            on=identity,
            how="inner",
            suffixes=("_old", "_canonical"),
            validate="one_to_one",
        )
        delta = (
            paired["execution_net_ev_12h_canonical"].to_numpy(dtype=float)
            - paired["execution_net_ev_12h_old"].to_numpy(dtype=float)
        )
        target_change = {
            "paired_rows": int(len(paired)),
            "exact_match_rows": int((np.abs(delta) <= 1e-7).sum()),
            "changed_rows": int((np.abs(delta) > 1e-7).sum()),
            "mean_canonical_minus_old_bps": float(delta.mean() * 10_000.0),
            "max_abs_delta_bps": float(np.abs(delta).max() * 10_000.0),
        }
    audit = {
        "feature_rows": int(len(features)),
        "exact_policy_rows": int(len(exact_policy)),
        "joined_rows": int(len(joined)),
        "feature_coverage": float(len(joined) / len(features)),
        "atr_rows": int(joined["oof_entry_atr_fraction"].notna().sum()),
        "atr_coverage": float(joined["oof_entry_atr_fraction"].notna().mean()),
        "max_gross_cost_net_delta": max_accounting_delta,
        "target_change": target_change,
    }
    return joined.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True), audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    features = pd.read_parquet(args.features)
    exact_policy = pd.read_parquet(args.exact_policy)
    atr = pd.read_parquet(args.atr_source)
    joined, audit = replace_outcomes(features, exact_policy, atr)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "joined.parquet"
    joined.to_parquet(output, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_input_not_model_evidence",
        "contract": {
            "identity": list(IDENTITY),
            "feature_policy": "preserve decision-time features; remove every prior realized outcome before exact join",
            "target_policy": "one exact 1m deployed-policy replay source for all dates",
            "accounting": "gross - cost = net; spread already embedded in gross and not deducted again",
            "atr": "causal pre-entry ATR retained for supporting-label materialization only",
        },
        "audit": audit,
        "inputs": {
            "features": {"path": str(args.features), "sha256": _sha256(args.features)},
            "exact_policy": {
                "path": str(args.exact_policy),
                "sha256": _sha256(args.exact_policy),
            },
            "atr_source": {
                "path": str(args.atr_source),
                "sha256": _sha256(args.atr_source),
            },
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "rows": int(len(joined)),
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "execution_ev_context_clean_regime_input_forward_july19_20260726_v1/"
            "joined.parquet"
        ),
    )
    parser.add_argument(
        "--exact-policy",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
            "execution_ev_policy_labels.parquet"
        ),
    )
    parser.add_argument(
        "--atr-source",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "path_archetype_labels_july20_20260726_v1/"
            "path_archetype_labels.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
