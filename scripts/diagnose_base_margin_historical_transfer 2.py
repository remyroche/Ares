#!/usr/bin/env python3
"""Reverse-transfer audit for the frozen base-margin false-positive screen.

The screen feature, sign, and threshold are read from the current-panel
diagnostic artifact and applied unchanged to the independent long common-30
strict OOF lineage.  Historical outcomes are used only for assessment.  This
is recurrence evidence, not chronological promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "base_margin_historical_transfer_v1"
DEFAULT_INPUT = ROOT / (
    "data_perp/artifacts/"
    "feb2025_jul2026_execution_ev_common30_transfer_oof_20260727_v4/"
    "two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_SCREEN = ROOT / (
    "data_perp/artifacts/"
    "execution_ev_false_positive_feature_diagnosis_20260727_v2/"
    "frozen_screens.csv"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/base_margin_historical_transfer_20260727_v1"
)
TARGET = "execution_net_ev_12h"
SCORE = "historical_direct_ev_oof"
FEATURE = "base_margin_to_cutoff_z"


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
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def global_top_fraction(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if frame.empty:
        return frame.copy()
    score = pd.to_numeric(frame[SCORE], errors="raise").to_numpy(np.float64)
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    rows = int(math.ceil(len(frame) * float(fraction)))
    selected = np.argsort(-score, kind="mergesort")[:rows]
    return frame.iloc[selected].copy()


def screen_metrics(
    selected: pd.DataFrame,
    *,
    threshold: float,
    direction: float,
    scope: str,
) -> dict[str, Any]:
    directional = (
        pd.to_numeric(selected[FEATURE], errors="raise").to_numpy(np.float64)
        * float(direction)
    )
    keep = directional >= float(threshold) * float(direction)
    net = pd.to_numeric(selected[TARGET], errors="raise").to_numpy(np.float64)
    high = net >= 0.005

    def mean_bps(mask: np.ndarray) -> float:
        return float(net[mask].mean() * 10_000.0) if mask.any() else float("nan")

    return {
        "scope": scope,
        "selected_rows": int(len(selected)),
        "keep_rows": int(keep.sum()),
        "drop_rows": int((~keep).sum()),
        "keep_fraction": float(keep.mean()) if len(keep) else float("nan"),
        "all_net_bps": mean_bps(np.ones(len(keep), dtype=bool)),
        "keep_net_bps": mean_bps(keep),
        "drop_net_bps": mean_bps(~keep),
        "keep_minus_drop_net_bps": (
            mean_bps(keep) - mean_bps(~keep)
            if keep.any() and (~keep).any()
            else float("nan")
        ),
        "keep_high_surplus_rate": float(high[keep].mean()) if keep.any() else float("nan"),
        "drop_high_surplus_rate": (
            float(high[~keep].mean()) if (~keep).any() else float("nan")
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    screen = pd.read_csv(args.screen)
    chosen = screen.loc[screen["feature"].eq(FEATURE)]
    if len(chosen) != 1:
        raise ValueError(f"expected exactly one frozen {FEATURE} screen")
    direction = float(chosen.iloc[0]["direction_tp_over_fp"])
    threshold = float(chosen.iloc[0]["frozen_selected_book_median"])
    if direction not in (-1.0, 1.0) or not np.isfinite(threshold):
        raise ValueError("invalid frozen direction or threshold")

    frame = pd.read_parquet(
        args.input,
        columns=[
            "__ts__",
            "__symbol__",
            "side_name",
            "candidate_id",
            "candidate_month",
            TARGET,
            SCORE,
            FEATURE,
        ],
    )
    if frame.duplicated(["__ts__", "__symbol__", "side_name", "candidate_id"]).any():
        raise ValueError("historical strict-OOF identity is not unique")
    required = frame[[TARGET, SCORE, FEATURE]].to_numpy(np.float64)
    if not np.isfinite(required).all():
        raise ValueError("historical assessment fields contain non-finite values")

    records: list[dict[str, Any]] = []
    overall = global_top_fraction(frame, float(args.top_fraction))
    records.append(
        screen_metrics(
            overall, threshold=threshold, direction=direction, scope="all_months_global"
        )
    )
    for month, group in frame.groupby("candidate_month", sort=True):
        selected = global_top_fraction(group, float(args.top_fraction))
        records.append(
            screen_metrics(
                selected,
                threshold=threshold,
                direction=direction,
                scope=str(month),
            )
        )
    metrics = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "status": "completed_reverse_transfer_diagnostic_not_promotion_evidence",
            "contract": {
                "screen_source": "current June-discovered, July-locked assessment artifact",
                "screen_feature": FEATURE,
                "direction": direction,
                "threshold": threshold,
                "screen_fit": "none; feature/sign/threshold applied unchanged",
                "ranking": "one pooled global top10 across all months plus month-local diagnostics; no timestamp/side/asset quota",
                "interpretation": "reverse-time recurrence diagnostic only; not causal promotion evidence",
            },
            "inputs": {
                "strict_oof": {
                    "path": str(args.input.resolve().relative_to(ROOT)),
                    "sha256": _sha256(args.input),
                },
                "frozen_screen": {
                    "path": str(args.screen.resolve().relative_to(ROOT)),
                    "sha256": _sha256(args.screen),
                },
            },
            "outputs": {"metrics": "metrics.csv"},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
