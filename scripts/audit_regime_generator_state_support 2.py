#!/usr/bin/env python3
"""Label-free support, episode, and K-split audit for primary regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_regime_execution_plan import _candidate_states, _read_generator_bindings


SCHEMA = "regime_generator_state_support_audit_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episodes(labels: np.ndarray, *, minimum_hours: int = 6) -> dict[int, int]:
    result: dict[int, int] = {}
    if not len(labels):
        return result
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            if end - start >= int(minimum_hours):
                state = int(labels[start])
                result[state] = result.get(state, 0) + 1
            start = end
    return result


def _diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads((path.parent / "parameter_diagnostics.json").read_text())
    return {str(row["fold_id"]): row for row in data if row.get("system") == "primary"}


def support_rows(generator: str, path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = _candidate_states(path)
    details = _diagnostics(path)
    rows: list[dict[str, Any]] = []
    posterior = [f"market_regime__state_p_{state}" for state in range(5)]
    for fold, local in frame.groupby("regime_fold_id", observed=True, sort=True):
        hourly = local.sort_values("__ts__", kind="stable").drop_duplicates("__ts__", keep="last")
        state_count = int(round(float(hourly["market_regime__state_count"].iloc[0])))
        active = posterior[:state_count]
        probability = hourly.loc[:, active].to_numpy(float)
        labels = np.argmax(probability, axis=1)
        episodes = _episodes(labels)
        item = details[str(fold)]
        alignment = item.get("primary_fold_alignment", {})
        for state, name in enumerate(active):
            soft_occupancy = float(probability[:, state].mean())
            rows.append({
                "generator": generator, "fold_id": fold, "state": state,
                "hourly_rows": int(len(hourly)), "effective_k": state_count,
                "soft_occupancy": soft_occupancy,
                "effective_support_rows": float(probability[:, state].sum()),
                "hard_occupancy": float((labels == state).mean()),
                "independent_episodes_ge_6h": int(episodes.get(state, 0)),
                "fold_alignment_gate": bool(alignment.get("passed", False)),
                "fold_alignment_status": alignment.get("status"),
                "fold_alignment_distance": alignment.get("mean_matched_centroid_distance"),
                "centroid": item.get("effective_state_centroids", [])[state],
            })
    return pd.DataFrame(rows), details


def split_rows(support: pd.DataFrame, diagnostics: dict[str, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = ("G2_k3", "G1_k4", "G0_k5")
    for lower_name, higher_name in zip(ordered[:-1], ordered[1:]):
        lower_diag, higher_diag = diagnostics[lower_name], diagnostics[higher_name]
        for fold in sorted(set(lower_diag) & set(higher_diag)):
            low = np.asarray(lower_diag[fold]["effective_state_centroids"], dtype=float)
            high = np.asarray(higher_diag[fold]["effective_state_centroids"], dtype=float)
            if len(high) != len(low) + 1 or low.shape[1] != high.shape[1]:
                rows.append({"lower_generator": lower_name, "higher_generator": higher_name, "fold_id": fold, "status": "not_a_k_plus_one_pair"})
                continue
            candidates: list[dict[str, Any]] = []
            for child_a, child_b in combinations(range(len(high)), 2):
                merged = 0.5 * (high[child_a] + high[child_b])
                distances = np.sqrt(((low - merged) ** 2).sum(axis=1))
                parent = int(np.argmin(distances))
                high_support = support.loc[(support.generator == higher_name) & (support.fold_id == fold)].set_index("state")
                candidates.append({
                    "lower_generator": lower_name, "higher_generator": higher_name, "fold_id": fold,
                    "status": "candidate_split", "parent_state": parent,
                    "child_a": child_a, "child_b": child_b,
                    "merged_to_parent_centroid_distance": float(distances[parent]),
                    "child_centroid_distance": float(np.sqrt(((high[child_a] - high[child_b]) ** 2).sum())),
                    "child_a_soft_occupancy": float(high_support.loc[child_a, "soft_occupancy"]),
                    "child_b_soft_occupancy": float(high_support.loc[child_b, "soft_occupancy"]),
                    "child_a_effective_support_rows": float(high_support.loc[child_a, "effective_support_rows"]),
                    "child_b_effective_support_rows": float(high_support.loc[child_b, "effective_support_rows"]),
                    "child_a_episodes": int(high_support.loc[child_a, "independent_episodes_ge_6h"]),
                    "child_b_episodes": int(high_support.loc[child_b, "independent_episodes_ge_6h"]),
                })
            best = min(candidates, key=lambda item: (item["merged_to_parent_centroid_distance"], item["child_a"], item["child_b"]))
            for item in candidates:
                item["nearest_parent_split"] = bool(item is best)
            rows.extend(candidates)
    return pd.DataFrame(rows)


def rare_exception(support: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    low = support.loc[support.soft_occupancy.lt(0.02)].copy()
    for (generator, state), local in low.groupby(["generator", "state"], observed=True, sort=True):
        eras = pd.Series(local.fold_id.astype(str).str.extract(r"(20\d{2})", expand=False)).dropna().astype(str).nunique()
        records.append({
            "generator": generator, "state": int(state), "rare_fold_count": int(len(local)),
            "era_count": int(eras), "minimum_episodes_ge_6h": int(local.independent_episodes_ge_6h.min()),
            "all_alignment_gates": bool(local.fold_alignment_gate.all()),
            "qualifies_explicit_rare_state_exception": bool(len(local) >= 3 and eras >= 2 and local.independent_episodes_ge_6h.min() >= 3 and local.fold_alignment_gate.all()),
        })
    return pd.DataFrame(records)


def run(*, generators: dict[str, Path], output_dir: Path) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    support_parts: list[pd.DataFrame] = []
    diag: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in generators.items():
        table, diagnostic = support_rows(name, path)
        support_parts.append(table); diag[name] = diagnostic
    support = pd.concat(support_parts, ignore_index=True)
    splits = split_rows(support, diag)
    exception = rare_exception(support)
    output.mkdir(parents=True)
    support.drop(columns=["centroid"]).to_csv(output / "state_support.csv", index=False)
    splits.to_csv(output / "k_split_diagnostics.csv", index=False)
    exception.to_csv(output / "rare_state_exception_audit.csv", index=False)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETED_LABEL_FREE_SUPPORT_AUDIT",
        "contract": {"episodes": "distinct hard-state runs lasting at least 6 hours on deduplicated hourly causal states", "rare_exception": "at least three rare folds, two calendar eras, three independent episodes in every rare fold, and causal fold alignment", "economic_labels_used": False},
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha(path)} for name, path in generators.items()},
        "outputs": {name: _sha(output / name) for name in ("state_support.csv", "k_split_diagnostics.csv", "rare_state_exception_audit.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    values = _args()
    print(run(generators=_read_generator_bindings(values.generator), output_dir=values.output_dir))
