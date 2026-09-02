#!/usr/bin/env python3
"""Build the frozen bad-episode registry used by contextual meta diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path("data_perp/reports/contextual_meta_episode_registry_20260622")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _episode_bounds(episode: str) -> tuple[str, str]:
    start = pd.Timestamp(episode, tz="UTC")
    end = start + pd.Timedelta(days=7)
    return start.isoformat(), end.isoformat()


def _rows_from_canonical_episode_effects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "is_bad_episode" not in df.columns:
        return []
    bad = df.loc[df["is_bad_episode"].astype(bool)].copy()
    if bad.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (head, episode), group in bad.groupby(["head", "episode"], sort=True):
        start, end = _episode_bounds(str(episode))
        deltas = pd.to_numeric(group.get("delta_log_loss_improvement"), errors="coerce")
        tail = pd.to_numeric(group.get("delta_tail_loss_10pct"), errors="coerce")
        severity = float(np.nanmean(np.abs(deltas.fillna(0.0))) + 0.25 * np.nanmean(np.abs(tail.fillna(0.0))))
        rows.append(
            {
                "episode_id": str(episode),
                "definition": "canonical_context_retrain_bad_episode",
                "target": "diagnostic",
                "start": start,
                "end": end,
                "severity": severity,
                "eligible_heads": str(head),
                "reason_for_inclusion": (
                    "canonical_context_retrain_episode_effects.csv marked is_bad_episode=True; "
                    f"diagnostic_targets={','.join(sorted(set(group['target'].astype(str))))}"
                ),
                "reason_for_exclusion": "",
                "source_artifact": str(path),
                "source_rows": int(len(group)),
            }
        )
    return rows


def _rows_from_short_asset_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for episode in data.get("bad_episodes", []) or []:
        start, end = _episode_bounds(str(episode))
        rows.append(
            {
                "episode_id": str(episode),
                "definition": "short_asset_context_economic_bad_episode",
                "target": "diagnostic",
                "start": start,
                "end": end,
                "severity": 1.0,
                "eligible_heads": "short_asset",
                "reason_for_inclusion": (
                    "short_asset_context_economic_manifest.json bad_episodes; "
                    f"diagnostic_target={data.get('target', '')}"
                ),
                "reason_for_exclusion": "",
                "source_artifact": str(path),
                "source_rows": 1,
            }
        )
    return rows


def _merge_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "definition",
                "target",
                "start",
                "end",
                "severity",
                "eligible_heads",
                "reason_for_inclusion",
                "reason_for_exclusion",
                "source_artifact",
                "source_rows",
            ]
        )
    df = pd.DataFrame(rows)
    merged: list[dict[str, Any]] = []
    for (head, episode), group in df.groupby(["eligible_heads", "episode_id"], sort=True):
        merged.append(
            {
                "episode_id": str(episode),
                "definition": "|".join(sorted(set(group["definition"].astype(str)))),
                "target": "diagnostic",
                "start": group["start"].iloc[0],
                "end": group["end"].iloc[0],
                "severity": float(pd.to_numeric(group["severity"], errors="coerce").fillna(0.0).max()),
                "eligible_heads": str(head),
                "reason_for_inclusion": " | ".join(sorted(set(group["reason_for_inclusion"].astype(str)))),
                "reason_for_exclusion": "",
                "source_artifact": " | ".join(sorted(set(group["source_artifact"].astype(str)))),
                "source_rows": int(pd.to_numeric(group["source_rows"], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(merged).sort_values(["eligible_heads", "episode_id"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    out_dir = _ensure_dir(Path(args.output_dir))
    rows = []
    rows.extend(_rows_from_canonical_episode_effects(Path(args.canonical_episode_effects)))
    rows.extend(_rows_from_short_asset_manifest(Path(args.short_asset_manifest)))
    registry = _merge_rows(rows)
    registry_path = out_dir / "frozen_bad_episode_registry.csv"
    registry.to_csv(registry_path, index=False)
    manifest = {
        "status": "completed",
        "registry": str(registry_path),
        "rows": int(len(registry)),
        "heads": sorted(set(registry.get("eligible_heads", pd.Series(dtype=str)).astype(str))),
        "episodes": sorted(set(registry.get("episode_id", pd.Series(dtype=str)).astype(str))),
        "inputs": {
            "canonical_episode_effects": str(args.canonical_episode_effects),
            "short_asset_manifest": str(args.short_asset_manifest),
        },
        "contract": {
            "training_target": "unchanged y_bin in one-head retrain scripts",
            "registry_target": "diagnostic only; not a training label",
        },
    }
    (out_dir / "frozen_bad_episode_registry_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default)
    )
    print(f"[episode_registry] wrote {registry_path} rows={len(registry)}", flush=True)
    return registry_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--canonical-episode-effects",
        default="data_perp/reports/canonical_context_retrain_experiment_20260622/canonical_context_retrain_episode_effects.csv",
    )
    parser.add_argument(
        "--short-asset-manifest",
        default="data_perp/reports/short_asset_context_economic_diagnostic_20260622/short_asset_context_economic_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
