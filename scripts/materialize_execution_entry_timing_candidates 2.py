#!/usr/bin/env python3
"""Build the exact cost/ATR candidate contract for entry-timing path labels."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SCHEMA = "execution_entry_timing_candidates_v1"
HISTORICAL_COUNTERFACTUAL_ECONOMICS = {
    "current_frozen_spread_counterfactual",
    "inverse_quote_notional_current_spread_counterfactual",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_target_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {
        "execution_ev_12h_hourly_policy_labels_v2",
        "execution_ev_deployed_policy_1m_labels_v1",
    }:
        raise ValueError("target manifest uses an unsupported execution-EV schema")
    if payload.get("prediction_role") != "execution_ev_12h_labels":
        raise ValueError("target manifest has the wrong prediction role")
    signed = payload.get("prediction_role_manifest_sha256")
    if not isinstance(signed, str) or not hmac.compare_digest(
        signed, _manifest_hash(payload)
    ):
        raise ValueError("target manifest signature does not verify")
    timing = payload.get("timing", {})
    accounting = payload.get("accounting", {})
    if schema == "execution_ev_12h_hourly_policy_labels_v2":
        if (
            timing.get("signal_timestamp") != "__ts__"
            or timing.get("first_path_timestamp") != "__decision_ts__"
            or float(timing.get("decision_delay_hours", -1)) != 1.0
            or float(timing.get("horizon_hours", -1)) != 12.0
        ):
            raise ValueError(
                "target manifest does not use canonical 1h decision / 12h timing"
            )
        if accounting.get("cost_contract") != "explicit_fee_plus_full_p90_spread":
            raise ValueError("target manifest must decompose fee and full p90 spread")
    else:
        lineage = payload.get("historical_lineage") or {}
        if (
            int(timing.get("signal_to_decision_minutes", -1)) != 60
            or int(timing.get("horizon_minutes", -1)) != 720
            or timing.get("label_available_at")
            != "decision + full replay horizon"
        ):
            raise ValueError(
                "deployed-policy target does not use canonical 1h decision / 12h timing"
            )
        if (
            lineage.get("oof_status") != "not_oof"
            or bool(lineage.get("execution_parity_claim"))
            or bool(lineage.get("promotion_eligible"))
            or lineage.get("economics")
            not in HISTORICAL_COUNTERFACTUAL_ECONOMICS
        ):
            raise ValueError(
                "deployed-policy target lacks an allowed historical "
                "counterfactual lineage"
            )
    return payload


def _canonical_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame))
    if missing:
        raise ValueError(f"{source} is missing identity columns {missing}")
    output = frame.copy()
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="coerce")
    if output["__ts__"].isna().any():
        raise ValueError(f"{source} has invalid UTC signal timestamps")
    for column in ("__symbol__", "side_name", "candidate_id"):
        output[column] = output[column].astype("string").str.strip()
        if output[column].isna().any() or output[column].eq("").any():
            raise ValueError(f"{source} has blank {column}")
    output["side_name"] = output["side_name"].str.lower()
    if not output["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{source} side_name must be canonical long/short")
    if output.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError(f"{source} has duplicate exact candidate identities")
    return output


def materialize(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output
    manifest_path = args.manifest or output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise ValueError("refusing to overwrite timing candidates or manifest")
    target = _load_target_manifest(args.execution_ev_target_manifest)
    source = target.get("source", {})
    deployed_1m = (
        target.get("schema") == "execution_ev_deployed_policy_1m_labels_v1"
    )
    expected_candidate_sha = (
        source.get("path_targets_sha256")
        if deployed_1m
        else source.get("sha256")
    )
    if expected_candidate_sha != _sha256(args.candidates):
        raise ValueError("candidate artifact hash does not match target manifest")
    if target.get("source_artifact_sha256") != _sha256(args.execution_ev_labels):
        raise ValueError("execution-EV label artifact hash does not match target manifest")

    candidate_columns = [
        *IDENTITY,
        args.atr_fraction_col,
    ]
    candidates = _canonical_identity(
        pd.read_parquet(args.candidates, columns=candidate_columns),
        source="target candidates",
    )
    candidates["atr_fraction"] = pd.to_numeric(
        candidates[args.atr_fraction_col], errors="coerce"
    )

    label_columns = (
        [
            *IDENTITY,
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_cost_return",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
        ]
        if deployed_1m
        else [
            *IDENTITY,
            "__decision_ts__",
            "execution_label_end_utc",
            "execution_fee_return",
            "execution_spread_return",
            "execution_cost_return",
        ]
    )
    labels = _canonical_identity(
        pd.read_parquet(args.execution_ev_labels, columns=label_columns),
        source="execution-EV labels",
    )
    if deployed_1m:
        labels = labels.rename(
            columns={"execution_decision_utc": "__decision_ts__"}
        )
    universe_path = getattr(args, "universe", None)
    if universe_path is not None:
        universe = _canonical_identity(
            pd.read_parquet(universe_path, columns=list(IDENTITY)),
            source="downstream OOF universe",
        )
        labels = universe.merge(
            labels,
            on=list(IDENTITY),
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not labels["_merge"].eq("both").all():
            raise ValueError(
                "downstream OOF universe is not fully covered by execution-EV labels"
            )
        labels = labels.drop(columns="_merge")
    labels["__decision_ts__"] = pd.to_datetime(
        labels["__decision_ts__"], utc=True, errors="coerce"
    )
    labels["execution_label_end_utc"] = pd.to_datetime(
        labels["execution_label_end_utc"], utc=True, errors="coerce"
    )
    if labels[["__decision_ts__", "execution_label_end_utc"]].isna().any().any():
        raise ValueError("execution-EV labels have invalid decision/label-end timestamps")
    if not (
        labels["__decision_ts__"] == labels["__ts__"] + pd.Timedelta(hours=1)
    ).all() or not (
        labels["execution_label_end_utc"]
        == labels["__decision_ts__"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("execution-EV labels violate canonical timing")
    numeric_cost_columns = (
        (
            "execution_cost_return",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
        )
        if deployed_1m
        else (
            "execution_fee_return",
            "execution_spread_return",
            "execution_cost_return",
        )
    )
    for column in numeric_cost_columns:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
        if labels[column].isna().any() or (labels[column] < 0.0).any():
            raise ValueError(f"execution-EV labels have invalid {column}")
    if not deployed_1m and not np.allclose(
        labels["execution_fee_return"] + labels["execution_spread_return"],
        labels["execution_cost_return"],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("execution-EV fee plus spread does not equal total cost")

    joined = labels.merge(
        candidates.loc[:, [*IDENTITY, "atr_fraction"]],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or joined["atr_fraction"].isna().any():
        raise ValueError("target label universe is not fully covered by ATR candidates")
    joined = joined.drop(columns="_merge")
    if (
        joined["atr_fraction"].isna().any()
        or (joined["atr_fraction"] <= 0.0).any()
        or (joined["atr_fraction"] > 1.0).any()
    ):
        raise ValueError("target-universe ATR fraction must be finite in (0, 1]")
    fee = (
        joined["execution_cost_return"]
        if deployed_1m
        else joined["execution_fee_return"]
    )
    entry_spread = (
        joined["execution_entry_half_spread_bps"]
        if deployed_1m
        else joined["execution_spread_return"] * 5_000.0
    )
    exit_spread = (
        joined["execution_exit_half_spread_bps"]
        if deployed_1m
        else joined["execution_spread_return"] * 5_000.0
    )
    result = pd.DataFrame(
        {
            **{column: joined[column] for column in IDENTITY},
            "__decision_ts__": joined["__decision_ts__"],
            "execution_label_end_utc": joined["execution_label_end_utc"],
            "atr_fraction": joined["atr_fraction"].astype(np.float32),
            "fee": fee.astype(np.float32),
            "entry_spread": entry_spread.astype(np.float32),
            "exit_spread": exit_spread.astype(np.float32),
        }
    ).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if not deployed_1m:
        recomposed = (
            result["fee"]
            + (result["entry_spread"] + result["exit_spread"]) / 10_000.0
        )
        if not np.allclose(
            recomposed.to_numpy(dtype=np.float64),
            joined.sort_values(list(IDENTITY), kind="stable")[
                "execution_cost_return"
            ].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("prepared timing costs do not recompose to target costs")

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "prediction_role": "execution_entry_timing_candidates",
        "source_artifact_sha256": _sha256(output),
        "sources": {
            "candidates": {"path": str(args.candidates), "sha256": _sha256(args.candidates)},
            "execution_ev_labels": {
                "path": str(args.execution_ev_labels),
                "sha256": _sha256(args.execution_ev_labels),
            },
            "execution_ev_target_manifest": {
                "path": str(args.execution_ev_target_manifest),
                "sha256": _sha256(args.execution_ev_target_manifest),
                "signed_manifest_sha256": target[
                    "prediction_role_manifest_sha256"
                ],
            },
            **(
                {
                    "downstream_oof_universe": {
                        "path": str(universe_path),
                        "sha256": _sha256(universe_path),
                        "contract": "exact_identity_filter_before_path_materialization",
                    }
                }
                if universe_path is not None
                else {}
            ),
        },
        "rows": int(len(result)),
        "identity": list(IDENTITY),
        "timing": {
            "decision": "__ts__ + 1h",
            "label_end": "__decision_ts__ + 12h",
        },
        "atr": {
            "input_column": args.atr_fraction_col,
            "output_column": "atr_fraction",
            "absolute_atr_derivation": "decision_price_times_atr_fraction",
        },
        "cost_accounting": {
            "fee": (
                "deployed execution_cost_return charged once"
                if deployed_1m
                else "execution_fee_return charged once"
            ),
            "entry_spread": (
                "deployed policy entry half-spread in bps"
                if deployed_1m
                else "one half of full p90 spread in bps"
            ),
            "exit_spread": (
                "deployed policy exit half-spread in bps"
                if deployed_1m
                else "one half of full p90 spread in bps"
            ),
            "recomposition": (
                "spread is embedded in executable gross return; fee remains separate"
                if deployed_1m
                else "fee + (entry_spread + exit_spread) / 10000"
            ),
        },
        "historical_lineage": target.get("historical_lineage"),
    }
    manifest["prediction_role_manifest_sha256"] = _manifest_hash(manifest)
    _write_json(manifest_path, manifest)
    return {"candidates": output, "manifest": manifest_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--execution-ev-labels", type=Path, required=True)
    parser.add_argument("--execution-ev-target-manifest", type=Path, required=True)
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Optional exact downstream OOF identity universe; every row must be covered.",
    )
    parser.add_argument(
        "--atr-fraction-col", default="__path_auxiliary_atr_fraction__"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main() -> None:
    try:
        paths = materialize(_parser().parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"entry-timing candidate materialization failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
