#!/usr/bin/env python3
"""Assessment-only full-December raw base/residual economics for the frozen bridge."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_augnov2025_common30_context_oos_extension as common

BRIDGE = ROOT / "data_perp/artifacts/dec2025_common30_frozen_august_base_residual_oos_bridge_20260730_v1"
OUT = ROOT / "data_perp/artifacts/dec2025_common30_frozen_base_residual_raw_economics_20260730_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise RuntimeError(output)
    manifest_path = BRIDGE / "manifest.json"
    bridge = json.loads(manifest_path.read_text())
    if (not (BRIDGE / "manifest.sha256").is_file() or (BRIDGE / "manifest.sha256").read_text().split()[0] != _sha(manifest_path)
            or bridge.get("status") != "SEALED_COMMON30_FROZEN_PRE_DECEMBER_BASE_RESIDUAL_OOS_SCORE_BRIDGE_NON_PROMOTION"):
        raise RuntimeError("December bridge is not sealed")
    source = BRIDGE / "oos_predictions.parquet"
    if bridge.get("outputs_sha256", {}).get(source.name) != _sha(source):
        raise RuntimeError("December bridge output hash detached")
    frame = pd.read_parquet(source)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if len(frame) != 44_640 or frame.candidate_id.duplicated().any() or not frame.residual_is_oos.all() or not frame.base_is_oos.all():
        raise RuntimeError("invalid full December score ledger")
    definitions = {"frozen_base_alpha_raw": "score_base_alpha", "frozen_residual_expected_ev_raw": "score_residual_expected_ev"}
    summaries, periods, sides, ledgers = [], [], [], []
    for arm, score in definitions.items():
        scored = frame.copy(); scored["raw_score"] = pd.to_numeric(scored[score], errors="raise")
        summary, period, side, book = common.evaluate(scored, arm)
        summary["score_column"] = score
        summaries.append(summary); periods.append(period); sides.append(side)
        ledgers.append(book.assign(score_column=score))
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.DataFrame(summaries).to_csv(stage / "metrics_summary.csv", index=False)
        pd.concat(periods, ignore_index=True).to_parquet(stage / "period_metrics.parquet", index=False)
        pd.concat(sides, ignore_index=True).to_parquet(stage / "side_metrics.parquet", index=False)
        pd.concat(ledgers, ignore_index=True).to_parquet(stage / "raw_score_books.parquet", index=False, compression="zstd")
        contract = {"decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                    "selection": "one pooled global top 10 percent per raw score across both sides and all timestamps; never per timestamp, side or asset",
                    "scores": definitions, "assessment_only": "December exact labels (some resolving 2026-01-01 12:00Z) are read only after frozen score materialisation",
                    "no_map_calibration_tuning_or_promotion": True}
        _dump(stage / "contract.json", contract)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": "dec2025_frozen_base_residual_raw_economics_v1", "status": "SEALED_ASSESSMENT_ONLY_RAW_ECONOMICS_NON_PROMOTION",
                    "promotion_eligible": False, "contract": contract, "inputs": {str(source.resolve()): _sha(source), str(manifest_path.resolve()): _sha(manifest_path)},
                    "outputs_sha256": {path.name: _sha(path) for path in files}}
        _dump(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(run())
