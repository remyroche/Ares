#!/usr/bin/env python3
"""Audit the chronological full-base joint-economics ablation.

This post-fit audit preserves the experiment's pooled-global ranking contract.
It reports overall, fold, week, side-contribution, component-attribution, and
promotion-gate evidence without fitting or selecting another model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_full_base_joint_economics_decomposition_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_full_base_joint_economics_summary_20260729_v1"
)
SCHEMA = "canonical_full_base_joint_economics_summary_v1"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
SCORES = {
    "direct_primary": "direct_primary_score",
    "opportunity": "prediction__opportunity_score",
    "exit_mixture": "prediction__exit_mixture_score",
    "joint": "joint_score",
}
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stable_global_top_mask(
    frame: pd.DataFrame, score: Iterable[float], fraction: float
) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    if not np.isfinite(values).all() or len(values) != len(frame):
        raise ValueError("ranking score must be finite and row-aligned")
    count = max(1, int(np.ceil(len(frame) * float(fraction))))
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -values))
    selected = np.zeros(len(frame), dtype=bool)
    selected[order[:count]] = True
    return selected


def validate_source(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "canonical_full_base_joint_economics_decomposition_v1":
        raise ValueError("unexpected source schema")
    if sidecar.read_text().split()[0] != sha256(manifest_path):
        raise ValueError("source manifest detached hash mismatch")
    outputs = manifest.get("outputs_sha256", {})
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in (
        ("development", "development_strict_expanding_oof_predictions.parquet"),
        ("april", "april_reused_diagnostic_predictions.parquet"),
    ):
        path = root / filename
        if outputs.get(filename) != sha256(path):
            raise ValueError(f"source hash mismatch: {filename}")
        frame = pd.read_parquet(path)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        if frame.duplicated([*IDENTITY, "arm"]).any():
            raise ValueError(f"{name} predictions are not unique by arm")
        frames[name] = frame
    return frames["development"], frames["april"], manifest


def selected_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    side = rows["side_name"].value_counts()
    return {
        "selected_rows": int(len(rows)),
        "mean_gross_bps": float(rows["execution_gross_ev_12h"].mean() * 10_000.0),
        "mean_cost_bps": float(rows["execution_cost_return"].mean() * 10_000.0),
        "mean_net_bps": float(rows["execution_net_ev_12h"].mean() * 10_000.0),
        "sum_net": float(rows["execution_net_ev_12h"].sum()),
        "positive_net_precision": float(rows["execution_net_ev_12h"].gt(0.0).mean()),
        "opportunity_0bps_precision": float(
            rows["opportunity_gross_above_cost_0bps"].mean()
        ),
        "opportunity_25bps_precision": float(
            rows["opportunity_gross_above_cost_25bps"].mean()
        ),
        "long_share": float(side.get("long", 0) / max(len(rows), 1)),
        "short_share": float(side.get("short", 0) / max(len(rows), 1)),
        "symbols": int(rows["__symbol__"].nunique()),
    }


def tail_rows(
    frame: pd.DataFrame,
    *,
    split: str,
    slice_kind: str,
    slice_value: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm, arm_rows in frame.groupby("arm", sort=True):
        arm_rows = arm_rows.reset_index(drop=True)
        for score_name, column in SCORES.items():
            for fraction in FRACTIONS:
                mask = stable_global_top_mask(arm_rows, arm_rows[column], fraction)
                records.append(
                    {
                        "split": split,
                        "slice_kind": slice_kind,
                        "slice_value": slice_value,
                        "arm": arm,
                        "score_name": score_name,
                        "fraction": fraction,
                        **selected_metrics(arm_rows.loc[mask]),
                    }
                )
    return records


def build_tail_audit(development: pd.DataFrame, april: pd.DataFrame) -> pd.DataFrame:
    records = tail_rows(
        development, split="development_strict_oof", slice_kind="overall", slice_value="all"
    )
    records += tail_rows(
        april, split="april_reused_diagnostic", slice_kind="overall", slice_value="all"
    )
    for fold, rows in development.groupby("fold_id", sort=True):
        records += tail_rows(
            rows,
            split="development_strict_oof",
            slice_kind="fold",
            slice_value=str(int(fold)),
        )
    week = april["__ts__"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    for week_start, rows in april.assign(__week__=week).groupby("__week__", sort=True):
        records += tail_rows(
            rows,
            split="april_reused_diagnostic",
            slice_kind="week",
            slice_value=week_start.isoformat(),
        )
    latest_start = pd.Timestamp("2025-04-24", tz="UTC")
    records += tail_rows(
        april.loc[april["__ts__"].ge(latest_start)],
        split="april_reused_diagnostic",
        slice_kind="latest_week",
        slice_value=latest_start.isoformat(),
    )
    return pd.DataFrame.from_records(records)


def side_contributions(
    frame: pd.DataFrame, *, split: str, fraction: float = 0.10
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for arm, arm_rows in frame.groupby("arm", sort=True):
        arm_rows = arm_rows.reset_index(drop=True)
        for score_name, column in SCORES.items():
            mask = stable_global_top_mask(arm_rows, arm_rows[column], fraction)
            for side, rows in arm_rows.loc[mask].groupby("side_name", sort=True):
                records.append(
                    {
                        "split": split,
                        "arm": arm,
                        "score_name": score_name,
                        "fraction": fraction,
                        "side_name": side,
                        **selected_metrics(rows),
                    }
                )
    return pd.DataFrame.from_records(records)


def component_attribution(
    frame: pd.DataFrame, *, split: str, fraction: float = 0.10
) -> pd.DataFrame:
    component_columns = [
        name
        for name in frame.columns
        if name.startswith("prediction__") or name in {"direct_primary_score", "joint_score"}
    ]
    records: list[dict[str, Any]] = []
    for arm, arm_rows in frame.groupby("arm", sort=True):
        arm_rows = arm_rows.reset_index(drop=True)
        for score_name, score_column in SCORES.items():
            selected = stable_global_top_mask(arm_rows, arm_rows[score_column], fraction)
            rows = arm_rows.loc[selected]
            record: dict[str, Any] = {
                "split": split,
                "arm": arm,
                "score_name": score_name,
                "fraction": fraction,
                **selected_metrics(rows),
            }
            for column in component_columns:
                record[f"mean__{column}"] = float(rows[column].mean())
            records.append(record)
    return pd.DataFrame.from_records(records)


def promotion_gates(tails: pd.DataFrame) -> list[dict[str, Any]]:
    april = tails.loc[
        tails["split"].eq("april_reused_diagnostic")
        & tails["slice_kind"].eq("overall")
        & np.isclose(tails["fraction"], 0.10)
    ]
    latest = tails.loc[
        tails["split"].eq("april_reused_diagnostic")
        & tails["slice_kind"].eq("latest_week")
        & np.isclose(tails["fraction"], 0.10)
    ]
    records: list[dict[str, Any]] = []
    for row in april.itertuples(index=False):
        recent = latest.loc[
            latest["arm"].eq(row.arm) & latest["score_name"].eq(row.score_name)
        ]
        direct = april.loc[
            april["arm"].eq(row.arm) & april["score_name"].eq("direct_primary")
        ]
        latest_net = float(recent.iloc[0]["mean_net_bps"]) if len(recent) == 1 else np.nan
        direct_net = float(direct.iloc[0]["mean_net_bps"]) if len(direct) == 1 else np.nan
        checks = {
            "global_top10_positive": float(row.mean_net_bps) > 0.0,
            "latest_week_top10_positive": latest_net > 0.0,
            "max_side_share_below_95pct": max(float(row.long_share), float(row.short_share))
            < 0.95,
            "beats_same_arm_direct_top10": (
                row.score_name == "direct_primary"
                or float(row.mean_net_bps) > direct_net
            ),
        }
        records.append(
            {
                "arm": row.arm,
                "score_name": row.score_name,
                "april_top10_net_bps": float(row.mean_net_bps),
                "latest_week_top10_net_bps": latest_net,
                "same_arm_direct_top10_net_bps": direct_net,
                **checks,
                "eligible_for_portfolio_replay": bool(all(checks.values())),
            }
        )
    return records


def write_artifact(
    output: Path,
    *,
    source: Path,
    source_manifest: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    gates: list[dict[str, Any]],
) -> None:
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        for filename, table in tables.items():
            table.to_parquet(temporary / filename, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETE_NO_PORTFOLIO_REPLAY"
            if not any(item["eligible_for_portfolio_replay"] for item in gates)
            else "COMPLETE_REPLAY_CANDIDATE_EXISTS",
            "source": {
                "root": str(source),
                "manifest_sha256": sha256(source / "manifest.json"),
                "schema": source_manifest["schema"],
            },
            "ranking": {
                "scope": "one pooled global book",
                "fractions": list(FRACTIONS),
                "tie_break": "candidate_id ascending",
                "never_per_timestamp_or_side": True,
            },
            "april_status": "reused diagnostic, not untouched promotion evidence",
            "promotion_gates": gates,
            "portfolio_replay": {
                "performed": False,
                "reason": "requires every predeclared economic, latest-period, balance, and beat-control gate",
            },
            "outputs_sha256": {
                filename: sha256(temporary / filename) for filename in sorted(tables)
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (temporary / "manifest.sha256").write_text(
            f"{sha256(manifest_path)}  manifest.json\n"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(source: Path, output: Path) -> Path:
    development, april, manifest = validate_source(source)
    tails = build_tail_audit(development, april)
    gates = promotion_gates(tails)
    tables = {
        "tail_slices.parquet": tails,
        "side_contributions_top10.parquet": pd.concat(
            [
                side_contributions(development, split="development_strict_oof"),
                side_contributions(april, split="april_reused_diagnostic"),
            ],
            ignore_index=True,
        ),
        "component_attribution_top10.parquet": pd.concat(
            [
                component_attribution(development, split="development_strict_oof"),
                component_attribution(april, split="april_reused_diagnostic"),
            ],
            ignore_index=True,
        ),
        "promotion_gates.parquet": pd.DataFrame.from_records(gates),
    }
    write_artifact(
        output, source=source, source_manifest=manifest, tables=tables, gates=gates
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.output)
