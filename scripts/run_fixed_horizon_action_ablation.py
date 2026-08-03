#!/usr/bin/env python3
"""Paired fixed-horizon exit actions on unchanged residual-selected books."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SELECTED_ROOT = ROOT / "data_perp/artifacts/residual_selected_exit_opportunity_counterfactual_20260730_v3"
TARGET_ROOT = ROOT / "data_perp/artifacts/execution_action_target_pack_20260730_v2"
OUT = ROOT / "data_perp/artifacts/fixed_horizon_action_ablation_20260730_v2"
TOPS = (0.01, 0.05, 0.10, 0.20)
HORIZONS = (1, 2, 4, 8, 12)
ARMS = ("deployed", *[f"fixed_{hours}h" for hours in HORIZONS])
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")


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
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_seal(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise ContractError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise ContractError(f"schema mismatch: {root}")
    for name, expected in manifest["outputs_sha256"].items():
        if sha256(root / name) != expected:
            raise ContractError(f"output hash mismatch: {root / name}")
    return manifest


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    denominator = float(frame[weight].sum())
    if denominator <= 0:
        raise ContractError("empty weighted scope")
    return float((frame[weight] * frame[value]).sum() / denominator)


def metric_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            active = month_rows.loc[month_rows[weight].gt(0)]
            total = float(active[weight].sum())
            for scope, local in [
                ("global", active),
                *[
                    (f"side_{side}", side_rows)
                    for side, side_rows in active.groupby("side_name", sort=True)
                ],
            ]:
                deployed = weighted_mean(local, "net__deployed", weight)
                for arm in ARMS:
                    net = weighted_mean(local, f"net__{arm}", weight)
                    gross = weighted_mean(local, f"gross__{arm}", weight)
                    cost = weighted_mean(local, f"cost__{arm}", weight)
                    rows.append(
                        {
                            "candidate_month": month,
                            "top_fraction": fraction,
                            "scope": scope,
                            "arm": arm,
                            "expected_selected_rows": float(local[weight].sum()),
                            "global_expected_selected_rows": total,
                            "net_bps": net * 10_000.0,
                            "gross_bps": gross * 10_000.0,
                            "cost_bps": cost * 10_000.0,
                            "paired_delta_vs_deployed_bps": (
                                net - deployed
                            )
                            * 10_000.0,
                            "positive_rate": weighted_mean(
                                local,
                                f"positive__{arm}",
                                weight,
                            ),
                        }
                    )
    return rows


def bootstrap_rows(
    frame: pd.DataFrame, *, draws: int = 2_000
) -> list[dict[str, Any]]:
    rows = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            active = month_rows.loc[month_rows[weight].gt(0)].copy()
            for scope, local in [
                ("global", active),
                *[
                    (f"side_{side}", side_rows.copy())
                    for side, side_rows in active.groupby("side_name", sort=True)
                ],
            ]:
                local["day"] = pd.to_datetime(
                    local.execution_decision_utc, utc=True
                ).dt.floor("D")
                days = sorted(local.day.unique())
                rng = np.random.default_rng(
                    20260730
                    + int(fraction * 100)
                    + (0 if scope == "global" else sum(map(ord, scope)))
                )
                index = rng.integers(0, len(days), size=(draws, len(days)))
                for arm in ARMS:
                    local["_absolute"] = local[weight] * local[f"net__{arm}"]
                    local["_delta"] = local[weight] * (
                        local[f"net__{arm}"] - local["net__deployed"]
                    )
                    local["_den"] = local[weight]
                    daily = (
                        local.groupby("day", sort=True)[
                            ["_absolute", "_delta", "_den"]
                        ]
                        .sum()
                        .reindex(days, fill_value=0.0)
                    )
                    denominator = daily._den.to_numpy()[index].sum(axis=1)
                    absolute = (
                        daily._absolute.to_numpy()[index].sum(axis=1) / denominator
                    )
                    delta = daily._delta.to_numpy()[index].sum(axis=1) / denominator
                    rows.append(
                        {
                            "candidate_month": month,
                            "top_fraction": fraction,
                            "scope": scope,
                            "arm": arm,
                            "days": len(days),
                            "draws": draws,
                            "net_bps": weighted_mean(
                                local, f"net__{arm}", weight
                            )
                            * 10_000.0,
                            "net_ci_low_bps": float(
                                np.quantile(absolute, 0.025) * 10_000.0
                            ),
                            "net_ci_high_bps": float(
                                np.quantile(absolute, 0.975) * 10_000.0
                            ),
                            "paired_delta_bps": weighted_mean(
                                local.assign(
                                    _paired=local[f"net__{arm}"]
                                    - local["net__deployed"]
                                ),
                                "_paired",
                                weight,
                            )
                            * 10_000.0,
                            "paired_delta_ci_low_bps": float(
                                np.quantile(delta, 0.025) * 10_000.0
                            ),
                            "paired_delta_ci_high_bps": float(
                                np.quantile(delta, 0.975) * 10_000.0
                            ),
                        }
                    )
    return rows


def run(output: Path = OUT, *, draws: int = 2_000) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    selected_manifest = verify_seal(
        SELECTED_ROOT, "residual_selected_exit_opportunity_counterfactual_v3"
    )
    target_manifest = verify_seal(
        TARGET_ROOT, "execution_action_target_pack_v2"
    )
    selected = pd.read_parquet(SELECTED_ROOT / "selected_counterfactuals.parquet")
    target_columns = [
        *IDENTITY,
        "canonical_cost_return",
        *[
            f"target_fixed_{hours}h_gross_return"
            for hours in HORIZONS
        ],
        *[
            f"target_fixed_{hours}h_net_return"
            for hours in HORIZONS
        ],
    ]
    targets = pd.read_parquet(TARGET_ROOT / "labels.parquet", columns=target_columns)
    frame = selected.merge(
        targets,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(selected):
        raise ContractError("action targets do not cover selected identities")
    if not np.allclose(
        frame.canonical_cost_return, frame.cost, atol=1e-12
    ):
        raise ContractError("canonical cost changed between artifacts")
    frame["gross__deployed"] = frame.deployed_gross
    frame["net__deployed"] = frame.deployed_net
    frame["cost__deployed"] = frame.cost
    frame["positive__deployed"] = frame.deployed_net.gt(0).astype(int)
    for hours in HORIZONS:
        arm = f"fixed_{hours}h"
        frame[f"gross__{arm}"] = frame[
            f"target_fixed_{hours}h_gross_return"
        ]
        frame[f"net__{arm}"] = frame[
            f"target_fixed_{hours}h_net_return"
        ]
        frame[f"cost__{arm}"] = frame.canonical_cost_return
        frame[f"positive__{arm}"] = frame[f"net__{arm}"].gt(0).astype(int)
        if not np.allclose(
            frame[f"gross__{arm}"] - frame[f"cost__{arm}"],
            frame[f"net__{arm}"],
            atol=1e-12,
        ):
            raise ContractError(f"{arm} accounting failed")
    source_metrics = pd.read_csv(SELECTED_ROOT / "metrics.csv")
    for month, local in frame.groupby("candidate_month", sort=True):
        for fraction in TOPS:
            weight = f"weight_top_{int(fraction * 100):02d}"
            parent = source_metrics.loc[
                source_metrics.candidate_month.eq(month)
                & source_metrics.top_fraction.eq(fraction)
                & source_metrics.scope.eq("global")
            ]
            if len(parent) != 1:
                raise ContractError(
                    f"parent global selection row missing: {month}/{fraction}"
                )
            expected = float(parent.expected_selected_rows.iloc[0])
            if not np.isclose(local[weight].sum(), expected):
                raise ContractError(f"global book weight changed: {month}/{fraction}")
    metrics = pd.DataFrame(metric_rows(frame))
    bootstrap = pd.DataFrame(bootstrap_rows(frame, draws=draws))
    parity = []
    for month in sorted(frame.candidate_month.unique()):
        for fraction in TOPS:
            expected = source_metrics.loc[
                source_metrics.candidate_month.eq(month)
                & source_metrics.top_fraction.eq(fraction)
                & source_metrics.scope.eq("global")
            ].iloc[0]
            actual = metrics.loc[
                metrics.candidate_month.eq(month)
                & metrics.top_fraction.eq(fraction)
                & metrics.scope.eq("global")
                & metrics.arm.eq("deployed")
            ].iloc[0]
            delta = float(actual.net_bps - expected.deployed_net_bps)
            if abs(delta) > 1e-9:
                raise ContractError("deployed control parity failed")
            parity.append(
                {
                    "candidate_month": month,
                    "top_fraction": fraction,
                    "net_bps_delta": delta,
                    "passed": True,
                }
            )
    primary = metrics.loc[
        metrics.top_fraction.eq(0.10) & metrics.scope.eq("global")
    ]
    diagnosis = []
    for month, local in primary.groupby("candidate_month", sort=True):
        fixed = local.loc[local.arm.ne("deployed")]
        best = fixed.sort_values(
            ["net_bps", "arm"], ascending=[False, True]
        ).iloc[0]
        diagnosis.append(
            {
                "candidate_month": month,
                "best_fixed_horizon_diagnostic_only": best.arm,
                "best_fixed_net_bps": best.net_bps,
                "deployed_net_bps": float(
                    local.loc[local.arm.eq("deployed"), "net_bps"].iloc[0]
                ),
                "no_fixed_horizon_is_a_promoted_policy": True,
            }
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        outputs: dict[str, Any] = {
            "paired_candidates.parquet": frame,
            "metrics.csv": metrics,
            "daily_bootstrap_ci.csv": bootstrap,
            "control_parity.csv": pd.DataFrame(parity),
            "diagnosis.json": {
                "schema": "fixed_horizon_action_diagnosis_v2",
                "status": "DIAGNOSTIC_ONLY_NO_ARM_SELECTION_NO_PROMOTION",
                "rows": diagnosis,
            },
        }
        for name, value in outputs.items():
            if name.endswith(".parquet"):
                value.to_parquet(stage / name, index=False, compression="zstd")
            elif name.endswith(".csv"):
                value.to_csv(stage / name, index=False)
            else:
                write_json(stage / name, value)
        manifest = {
            "schema": "fixed_horizon_action_ablation_v2",
            "status": "SEALED_DIAGNOSTIC_ONLY_UNCHANGED_BOOKS_NO_PROMOTION",
            "promotion_eligible": False,
            "arms": list(ARMS),
            "contract": {
                "selection": "exact v3 candidate IDs and fractional global monthly weights; no reranking",
                "actions": "deployed control plus forced minute-60/120/240/480/720 close",
                "cost": "same canonical row round-trip cost once for every arm",
                "uncertainty": f"{draws} UTC-day clustered draws on paired outcomes after freezing weights",
                "selection_of_arm": "forbidden on reused March/April diagnostics; best-arm fields are hindsight descriptions only",
            },
            "input_provenance": {
                "selected_manifest_sha256": sha256(SELECTED_ROOT / "manifest.json"),
                "selected_sha256": selected_manifest["outputs_sha256"][
                    "selected_counterfactuals.parquet"
                ],
                "target_manifest_sha256": sha256(TARGET_ROOT / "manifest.json"),
                "target_sha256": target_manifest["outputs_sha256"]["labels.parquet"],
            },
            "outputs_sha256": {
                name: sha256(stage / name) for name in outputs
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "Forced closes use raw exact path closes and canonical cost; no additional market-impact claim is made.",
                "March and April are reused diagnostics and cannot select a deployable horizon.",
                "This phase does not replay deployed stop/trail state under a shorter time stop.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()
    print(json.dumps(safe(run(args.output, draws=args.bootstrap_draws)), indent=2))


if __name__ == "__main__":
    main()
