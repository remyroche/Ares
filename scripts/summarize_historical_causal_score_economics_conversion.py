#!/usr/bin/env python3
"""Verify and summarize source-separated causal conversion experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, item in payload["outputs"].items():
        output = Path(item["path"])
        if not output.is_absolute():
            output = Path.cwd() / output
        if not output.is_file():
            raise FileNotFoundError(f"{path}: missing {name} output {output}")
        if _sha256(output) != item["sha256"]:
            raise ValueError(f"{path}: hash mismatch for {name}")
    audit_path = Path(payload["outputs"]["audit"]["path"])
    if not audit_path.is_absolute():
        audit_path = Path.cwd() / audit_path
    audit = pd.read_parquet(audit_path)
    used = audit.loc[audit["reference_rows"].gt(0)]
    if len(used):
        maximum = pd.to_datetime(
            used["reference_label_end_max_utc"], utc=True, errors="raise"
        )
        snapshot = pd.to_datetime(used["snapshot_utc"], utc=True, errors="raise")
        if not maximum.lt(snapshot).all():
            raise ValueError(f"{path}: noncausal reference row detected")
    return payload


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    all_manifests = sorted(args.mapping_root.glob("*/manifest.json"))
    invalidated = [
        path
        for path in all_manifests
        if (path.parent / "DUPLICATE_EXPERIMENT_INVALIDATION.json").is_file()
    ]
    manifests = [path for path in all_manifests if path not in invalidated]
    if not manifests:
        raise FileNotFoundError("no completed mapping manifests found")
    comparison_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = _verify_manifest(manifest_path)
        economics_path = Path(manifest["outputs"]["economics"]["path"])
        calibration_path = Path(manifest["outputs"]["calibration"]["path"])
        if not economics_path.is_absolute():
            economics_path = Path.cwd() / economics_path
        if not calibration_path.is_absolute():
            calibration_path = Path.cwd() / calibration_path
        economics = pd.read_csv(economics_path)
        calibration = pd.read_csv(calibration_path)
        top10 = economics.loc[np.isclose(economics["top_k_fraction"], 0.10)]
        pooled = top10.loc[
            top10["scope"].eq("pooled")
            & top10["mapping"].isin(("score_raw", "mapped_direct_net"))
        ].set_index("mapping")
        if set(pooled.index) != {"score_raw", "mapped_direct_net"}:
            raise ValueError(f"{manifest_path}: missing pooled raw/direct top10")
        pooled_calibration = calibration.loc[calibration["scope"].eq("pooled")]
        if len(pooled_calibration) != 1:
            raise ValueError(f"{manifest_path}: pooled calibration is not unique")
        raw = pooled.loc["score_raw"]
        mapped = pooled.loc["mapped_direct_net"]
        cal = pooled_calibration.iloc[0]
        month = top10.loc[
            top10["scope"].str.startswith("month_")
            & top10["mapping"].isin(("score_raw", "mapped_direct_net"))
        ].copy()
        month["experiment"] = manifest_path.parent.name
        month["source_family"] = manifest["source_family"]
        month["score_column"] = manifest["score_column"]
        month_rows.extend(month.to_dict(orient="records"))
        raw_month = month.loc[month["mapping"].eq("score_raw")].set_index("scope")
        mapped_month = month.loc[
            month["mapping"].eq("mapped_direct_net")
        ].set_index("scope")
        shared = raw_month.index.intersection(mapped_month.index)
        month_delta = (
            mapped_month.loc[shared, "mean_net_bps"]
            - raw_month.loc[shared, "mean_net_bps"]
        )
        comparison_rows.append(
            {
                "experiment": manifest_path.parent.name,
                "source_family": manifest["source_family"],
                "evidence_tier": manifest["evidence_tier"],
                "score_column": manifest["score_column"],
                "promotion_eligible": bool(
                    manifest["promotion_boundary"]["source_promotion_eligible"]
                ),
                "input_rows": int(manifest["rows"]["input"]),
                "mapped_rows": int(manifest["rows"]["mapped_eligible"]),
                "raw_pooled_top10_net_bps": float(raw["mean_net_bps"]),
                "mapped_pooled_top10_net_bps": float(mapped["mean_net_bps"]),
                "pooled_mapping_delta_bps": float(
                    mapped["mean_net_bps"] - raw["mean_net_bps"]
                ),
                "raw_positive_months": int(
                    (raw_month["mean_net_bps"] > 0.0).sum()
                ),
                "mapped_positive_months": int(
                    (mapped_month["mean_net_bps"] > 0.0).sum()
                ),
                "months_mapping_improves": int((month_delta > 0.0).sum()),
                "months_mapping_degrades": int((month_delta < 0.0).sum()),
                "worst_mapped_month_net_bps": float(
                    mapped_month["mean_net_bps"].min()
                ),
                "maximum_mapped_month_side_share": float(
                    np.maximum(
                        mapped_month["long_rows"], mapped_month["short_rows"]
                    ).div(mapped_month["selected_rows"]).max()
                ),
                "opportunity_auc": float(cal["opportunity_auc"]),
                "opportunity_brier": float(cal["opportunity_brier"]),
                "mapped_direct_net_spearman": float(
                    cal["direct_net_spearman"]
                ),
                "q50_coverage": float(cal["opportunity_q50_coverage"]),
                "q80_coverage": float(cal["opportunity_q80_coverage"]),
            }
        )
        verified.append(
            {
                "manifest": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "strict_causal_audit_pass": True,
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("experiment")
    months = pd.DataFrame(month_rows).sort_values(
        ["experiment", "scope", "mapping"]
    )
    promotion = comparison.loc[comparison["promotion_eligible"]]
    eligible = promotion.loc[
        promotion["mapped_pooled_top10_net_bps"].gt(0.0)
        & promotion["worst_mapped_month_net_bps"].gt(0.0)
        & promotion["maximum_mapped_month_side_share"].lt(0.95)
        & promotion["opportunity_auc"].gt(0.55)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_path = args.output_dir / "experiment_comparison.csv"
    months_path = args.output_dir / "month_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    months.to_csv(months_path, index=False)
    manifest = {
        "schema": "historical_causal_score_economics_conversion_summary_v1",
        "status": (
            "NO_PROMOTION_ELIGIBLE_CONVERSION_MAP"
            if eligible.empty
            else "PROMOTION_GATE_SURVIVOR_REQUIRES_PORTFOLIO_REPLAY"
        ),
        "mapping_root": str(args.mapping_root),
        "completed_experiments": int(len(comparison)),
        "ignored_invalidated_duplicates": [str(path.parent) for path in invalidated],
        "verified_manifests": verified,
        "promotion_gate": {
            "canonical_source_only": True,
            "pooled_top10_net_positive": True,
            "every_month_net_positive": True,
            "maximum_month_side_share": 0.95,
            "opportunity_auc_above": 0.55,
            "survivors": eligible["experiment"].tolist(),
        },
        "interpretation": (
            "Aggregate improvement is insufficient. Source families remain "
            "separate; month/latest-period economics and cross-side balance are "
            "mandatory. Quantile and exit-mixture rankings remain diagnostic."
        ),
        "outputs": {
            "comparison": {
                "path": str(comparison_path),
                "sha256": _sha256(comparison_path),
            },
            "months": {"path": str(months_path), "sha256": _sha256(months_path)},
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "comparison": comparison_path,
        "months": months_path,
        "manifest": manifest_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    outputs = run(_parser().parse_args())
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
