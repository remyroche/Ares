#!/usr/bin/env python3
"""Audit causal-map plateaus without refitting a target or a mapper.

The target-purity runner ranks mapped expected net with a stable order.  A
daily causal isotonic map can have equal mapped values, so its global-tail
boundary may otherwise fall on candidate ordering rather than model evidence.
This diagnostic compares that existing stable order with a deterministic
secondary order of *raw decision-time score*, then candidate ID.  It changes
neither fitted map values nor the causal threshold policy.

Pooled-global tails are diagnostic only; this does not create a deployable
period-wide top-k rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v10"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/exact_h12_score_map_tie_audit_20260731_v1"
TOPS = (0.01, 0.05, 0.10, 0.20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _selection(frame: pd.DataFrame, fraction: float, *, raw_tiebreak: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    n = max(1, int(np.ceil(len(frame) * fraction)))
    if raw_tiebreak:
        ordered = frame.sort_values(
            ["calibrated_expected_net_bps", "raw_score", "candidate_id"],
            ascending=[False, False, True], kind="stable",
        )
    else:
        # Mirrors the current evaluator's stable argsort on the already
        # canonical decision_ts/candidate_id ordered artifact.
        ordered = frame.sort_values("calibrated_expected_net_bps", ascending=False, kind="stable")
    selected = ordered.head(n).copy()
    cutoff = float(selected.calibrated_expected_net_bps.iloc[-1])
    tie = frame.loc[frame.calibrated_expected_net_bps.eq(cutoff)]
    above = int(frame.calibrated_expected_net_bps.gt(cutoff).sum())
    return selected, {
        "selected_rows": n,
        "cutoff_mapped_expected_net_bps": cutoff,
        "cutoff_tie_rows": int(len(tie)),
        "cutoff_tie_rows_needed": n - above,
        "cutoff_tie_raw_score_unique": int(tie.raw_score.nunique()),
        "cutoff_tie_realised_net_mean_bps": float(tie.exact_h12_net_bps.mean()),
        "cutoff_tie_realised_net_min_bps": float(tie.exact_h12_net_bps.min()),
        "cutoff_tie_realised_net_max_bps": float(tie.exact_h12_net_bps.max()),
    }


def _book_rows(frame: pd.DataFrame, arm: str, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction in TOPS:
        book, tie = _selection(frame, fraction, raw_tiebreak=method == "mapped_then_raw_tiebreak")
        rows.append({
            "arm": arm, "method": method, "fraction": fraction, **tie,
            "gross_bps": float(book.exact_h12_gross_bps.mean()),
            "cost_bps": float(book.row_cost_bps.mean()),
            "net_bps": float(book.exact_h12_net_bps.mean()),
            "positive_net_rate": float(book.exact_h12_net_bps.gt(0.0).mean()),
            "long_share": float(book.side.eq("long").mean()),
        })
        for side, part in book.groupby("side", sort=True):
            rows.append({
                "arm": arm, "method": method, "scope": "selected_side", "fraction": fraction,
                "side": side, "selected_rows": len(part), "net_bps": float(part.exact_h12_net_bps.mean()),
            })
    return rows


def _bootstrap(frame: pd.DataFrame, arm: str, *, seed: int, replicates: int) -> dict[str, Any]:
    frame = frame.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    day_codes, days = pd.factorize(frame.decision_ts.dt.floor("D"), sort=True)
    day_rows = [np.flatnonzero(day_codes == value) for value in range(len(days))]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        positions = np.concatenate([day_rows[value] for value in rng.integers(0, len(day_rows), size=len(day_rows))])
        sampled = frame.iloc[positions].reset_index(drop=True)
        baseline, _ = _selection(sampled, 0.10, raw_tiebreak=False)
        tiebreak, _ = _selection(sampled, 0.10, raw_tiebreak=True)
        deltas[replicate] = float(tiebreak.exact_h12_net_bps.mean() - baseline.exact_h12_net_bps.mean())
    baseline, _ = _selection(frame, 0.10, raw_tiebreak=False)
    tiebreak, _ = _selection(frame, 0.10, raw_tiebreak=True)
    return {
        "arm": arm, "fraction": 0.10, "day_blocks": len(day_rows), "replicates": replicates,
        "delta_net_bps_full_sample": float(tiebreak.exact_h12_net_bps.mean() - baseline.exact_h12_net_bps.mean()),
        "delta_net_bps_bootstrap_mean": float(deltas.mean()),
        "delta_net_bps_p05": float(np.quantile(deltas, 0.05)),
        "delta_net_bps_p95": float(np.quantile(deltas, 0.95)),
        "probability_improves": float((deltas > 0.0).mean()),
    }


def run(input_dir: Path, output_dir: Path, *, seed: int = 20260731, replicates: int = 400) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    results_path = input_dir / "target_ablation_results.parquet"
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exact_h12_target_purity_ablation_v10":
        raise ValueError("sealed v10 target-ablation output is required")
    columns = ["arm", "candidate_id", "side", "decision_ts", "raw_score", "calibrated_expected_net_bps", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps"]
    data = pd.read_parquet(results_path, columns=columns)
    data["decision_ts"] = pd.to_datetime(data.decision_ts, utc=True, errors="raise")
    if data.duplicated(["arm", "candidate_id"]).any() or not np.isfinite(data.raw_score).all() or not np.isfinite(data.calibrated_expected_net_bps).all():
        raise ValueError("target result coverage is incomplete")
    counts = data.groupby("arm", sort=True).candidate_id.agg(["count", "nunique"])
    if not counts["count"].eq(counts["nunique"]).all() or counts["count"].nunique() != 1:
        raise ValueError("arms do not have identical complete candidate identities")
    candidate_ids = None
    for arm, part in data.groupby("arm", sort=True):
        ids = tuple(part.sort_values(["decision_ts", "candidate_id"], kind="stable").candidate_id)
        if candidate_ids is None:
            candidate_ids = ids
        elif ids != candidate_ids:
            raise ValueError(f"{arm} candidate identities differ")
    books, bootstraps = [], []
    for index, (arm, part) in enumerate(data.groupby("arm", sort=True)):
        part = part.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
        books.extend(_book_rows(part, arm, "mapped_stable"))
        books.extend(_book_rows(part, arm, "mapped_then_raw_tiebreak"))
        bootstraps.append(_bootstrap(part, arm, seed=seed + index, replicates=replicates))
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        pd.DataFrame(books).to_csv(stage / "tie_resolution_books.csv", index=False)
        pd.DataFrame(bootstraps).to_csv(stage / "tie_resolution_day_bootstrap.csv", index=False)
        summary = {
            "schema": "exact_h12_score_map_tie_audit_v1",
            "status": "COMPLETED_RESEARCH_ONLY_NO_PROMOTION",
            "input": {"artifact": str(input_dir), "manifest_sha256": _sha256(manifest_path), "results_sha256": _sha256(results_path)},
            "contract": {
                "change": "only deterministic secondary raw-score ordering inside equal mapped-score plateaus",
                "unchanged": ["candidate IDs", "raw scores", "mapped values", "policy", "costs", "target", "causal threshold"],
                "selection": "pooled-global top-k diagnostic only; no side/timestamp quota",
            },
            "rows_per_arm": int(counts["count"].iloc[0]),
            "arms": sorted(data.arm.unique().tolist()),
            "outputs": {path.name: _sha256(path) for path in stage.iterdir() if path.is_file()},
        }
        _write_json(stage / "manifest.json", summary)
        os.replace(stage, output_dir)
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--replicates", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, seed=args.seed, replicates=args.replicates), indent=2))


if __name__ == "__main__":
    main()
