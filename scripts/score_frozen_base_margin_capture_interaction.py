#!/usr/bin/env python3
"""Score the pinned base-margin/capture interaction on already-used OOS blocks.

This is deliberately a *diagnostic* scorer.  It reuses the strict side-local
OOF heads and causal direct-EV mapping emitted by the frozen v8 capture-support
run; it does not refit a model, choose a weight, or re-map on outcomes.  The
available June/later-July blocks were already used in diagnosis, so this output
cannot promote the challenger.  The next genuinely new block must be scored by
the integrated OOF runner under a successor source lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exact_policy_capture_support_ablation import (
    FROZEN_BASE_MARGIN_SCREEN,
    FROZEN_INTERACTION_INPUT_HASHES,
    IDENTITY_COLUMNS,
    TARGET_COLUMN,
    load_frozen_base_margin_interaction,
    margin_capture_soft_interaction,
)


SCHEMA = "frozen_base_margin_capture_interaction_diagnostic_v1"
SOURCE_DIR = ROOT / "data_perp/artifacts/exact_policy_capture_support_ablation_20260727_v8"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
SOURCE_MANIFEST_SHA256 = "baf292c9774b84d63c0618b0bc76c8b01fb211748af4292e1f60299c30c682fc"
DEFAULT_INPUT = ROOT / "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/frozen_base_margin_capture_interaction_diagnostic_20260727_v1"
DIRECT_ARM = "direct_net"
DIRECT_STAGE = "canonical_recent_ev_mapping"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
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


def verify_frozen_source(manifest_path: Path) -> dict[str, Any]:
    """Verify every reused prediction/source hash before loading data."""
    if _sha256(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("capture-support source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "exact_policy_capture_support_ablation_v1":
        raise ValueError("unexpected capture-support source schema")
    outputs = manifest.get("outputs", {})
    for name in ("predictions", "head_predictions"):
        record = outputs.get(name, {})
        path = ROOT / str(record.get("path", ""))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"capture-support {name} hash mismatch")
    source_inputs = manifest.get("inputs", {})
    for name, expected in FROZEN_INTERACTION_INPUT_HASHES.items():
        if source_inputs.get(name, {}).get("sha256") != expected:
            raise ValueError(f"capture-support {name} lineage hash mismatch")
    return manifest


def _diagnostic_interaction_scores(group: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    """Apply a frozen rank perturbation while retaining the causal direct map.

    The raw head standardization is side/window-local solely to put the already
    generated direct and probability heads on a common confidence scale.  It is
    an output diagnostic, not a live calibration.  The direct EV map itself is
    the causal mapping emitted by the source run and is never refit here.
    """
    work = group.copy()
    raw_interaction, _, report = margin_capture_soft_interaction(
        work["direct_net"].to_numpy(float),
        work["capture_probability"].to_numpy(float),
        work["base_margin_to_cutoff_z"].to_numpy(float),
        work["direct_net"].to_numpy(float),
        work["capture_probability"].to_numpy(float),
        work["base_margin_to_cutoff_z"].to_numpy(float),
        contract=contract,
    )
    raw_direct = work["direct_net"].to_numpy(float)
    direct_center = float(raw_direct.mean())
    direct_scale = max(float(raw_direct.std()), 1e-8)
    direct_z = (raw_direct - direct_center) / direct_scale
    # Retain the source's causal EV map.  The bounded rank perturbation is
    # converted using only the mapped-score cross-sectional scale of the block.
    mapped = work["canonical_recent_ev_score"].to_numpy(float)
    mapped_scale = max(float(mapped.std()), 1e-8)
    work["direct_capture_margin_soft_interaction_score"] = (
        mapped + (raw_interaction - direct_z) * mapped_scale
    )
    work["interaction_rank_delta"] = raw_interaction - direct_z
    work["interaction_report"] = json.dumps(report, sort_keys=True)
    return work


def _top10_metrics(frame: pd.DataFrame, score: str, *, scope: str) -> dict[str, Any]:
    count = max(1, int(math.ceil(0.10 * len(frame))))
    selected = frame.iloc[
        np.argsort(-frame[score].to_numpy(float), kind="mergesort")[:count]
    ]
    target = selected[TARGET_COLUMN].to_numpy(float)
    return {
        "scope": scope,
        "score": score,
        "rows": int(len(frame)),
        "selected_rows": count,
        "top10_net_bps": float(target.mean() * 10_000.0),
        "top10_positive_rate": float((target > 0.0).mean()),
        "top10_margin_mean": float(selected["base_margin_to_cutoff_z"].mean()),
        "top10_capture_probability_mean": float(selected["capture_probability"].mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source_manifest = verify_frozen_source(args.source_manifest)
    if _sha256(args.input) != FROZEN_INTERACTION_INPUT_HASHES["data"]:
        raise ValueError("joined input hash mismatch; scorer is pinned to reused-OOS blocks")
    contract = load_frozen_base_margin_interaction(args.base_margin_screen)
    head_path = ROOT / source_manifest["outputs"]["head_predictions"]["path"]
    prediction_path = ROOT / source_manifest["outputs"]["predictions"]["path"]
    heads = pd.read_parquet(head_path)
    direct = pd.read_parquet(prediction_path)
    direct = direct.loc[
        direct["arm"].eq(DIRECT_ARM) & direct["mapping_stage"].eq(DIRECT_STAGE),
        [*IDENTITY_COLUMNS, "window", "canonical_recent_ev_score"],
    ]
    if direct.duplicated([*IDENTITY_COLUMNS, "window"]).any() or heads.duplicated([*IDENTITY_COLUMNS, "window"]).any():
        raise ValueError("source predictions do not have unique window identities")
    source = pd.read_parquet(args.input, columns=[*IDENTITY_COLUMNS, TARGET_COLUMN, "base_margin_to_cutoff_z"])
    if source.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("joined source does not have unique identities")
    scored = heads.merge(direct, on=[*IDENTITY_COLUMNS, "window"], validate="one_to_one")
    scored = scored.merge(source, on=list(IDENTITY_COLUMNS), validate="many_to_one")
    if scored[["direct_net", "capture_probability", "canonical_recent_ev_score", TARGET_COLUMN, "base_margin_to_cutoff_z"]].isna().any().any():
        raise ValueError("diagnostic scorer has missing head, map, target, or margin inputs")
    parts = []
    for _, group in scored.groupby(["window", "side_name"], sort=True):
        parts.append(_diagnostic_interaction_scores(group, contract))
    scored = pd.concat(parts, ignore_index=True)
    metrics: list[dict[str, Any]] = []
    for window, group in scored.groupby("window", sort=True):
        baseline = _top10_metrics(group, "canonical_recent_ev_score", scope=str(window))
        challenger = _top10_metrics(group, "direct_capture_margin_soft_interaction_score", scope=str(window))
        baseline["arm"] = DIRECT_ARM
        challenger["arm"] = str(contract["name"])
        challenger["delta_vs_direct_bps"] = challenger["top10_net_bps"] - baseline["top10_net_bps"]
        metrics.extend([baseline, challenger])
    for score, arm in (("canonical_recent_ev_score", DIRECT_ARM), ("direct_capture_margin_soft_interaction_score", contract["name"])):
        row = _top10_metrics(scored, score, scope="all_reused_blocks_global")
        row["arm"] = arm
        metrics.append(row)
    metric_frame = pd.DataFrame(metrics)
    global_rows = metric_frame.loc[metric_frame["scope"].eq("all_reused_blocks_global")]
    baseline_global = float(global_rows.loc[global_rows["arm"].eq(DIRECT_ARM), "top10_net_bps"].iloc[0])
    global_challenger = metric_frame["scope"].eq("all_reused_blocks_global") & metric_frame["arm"].eq(contract["name"])
    metric_frame.loc[global_challenger, "delta_vs_direct_bps"] = (
        metric_frame.loc[global_challenger, "top10_net_bps"] - baseline_global
    )
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.csv"
    predictions_path = args.output_dir / "predictions.parquet"
    metric_frame.to_csv(metrics_path, index=False)
    scored.to_parquet(predictions_path, index=False)
    output = {
        "schema": SCHEMA,
        "status": "completed_reused_oos_diagnostic_not_promotion_evidence",
        "contract": {
            "interaction": contract,
            "heads": "strict side-local temporal OOF heads reused byte-for-byte from v8",
            "mapping": "source direct canonical recent-EV map reused byte-for-byte; no outcome remapping in this scorer",
            "ranking": "one pooled global top10 per window and across reused blocks; no timestamp, side, or asset quota",
            "decision_evidence": "the next genuinely new forward block only, under a separately frozen successor source lock",
            "limitation": "current blocks and their output-wide confidence standardization are reused diagnostic evidence only",
        },
        "inputs": {
            "source_manifest": {"path": str(args.source_manifest), "sha256": _sha256(args.source_manifest)},
            "joined_input": {"path": str(args.input), "sha256": _sha256(args.input)},
            "base_margin_screen": {"path": str(args.base_margin_screen), "sha256": _sha256(args.base_margin_screen)},
        },
        "outputs": {
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "predictions": {"path": str(predictions_path), "sha256": _sha256(predictions_path)},
        },
    }
    _write_json(args.output_dir / "manifest.json", output)
    return output


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--base-margin-screen", type=Path, default=FROZEN_BASE_MARGIN_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(_parser()), indent=2, default=str))
