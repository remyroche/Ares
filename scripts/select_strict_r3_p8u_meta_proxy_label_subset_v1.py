#!/usr/bin/env python3
"""Select a deliberately diverse MC1-label subset for the P8u Meta proxy.

This is not an HPO selector.  Its only job is to choose 30--60 trials whose
expensive downstream MC1 labels will cover target families, losses, feature
contracts, HPO shapes, and poor/medium/strong cheap diagnostics.  The later
learned surrogate, not this sampler, ranks trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_p8u_meta_proxy_label_subset_v1"
DIAGNOSTICS = (
    "residual_ic", "conditional_mi_given_base", "probe_delta_top2_ev",
    "weekly_q10", "candidate_minus_control_top2_ev",
)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def _zscore(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(raw)
    if not finite.any():
        return np.zeros(len(raw), dtype=float)
    median = float(np.nanmedian(raw[finite]))
    scale = float(np.nanmedian(np.abs(raw[finite] - median))) * 1.4826
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(raw[finite]))
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros(len(raw), dtype=float)
    return np.nan_to_num((raw - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _diagnostic_strata(frame: pd.DataFrame) -> pd.Series:
    # A balanced, equal-weight *sampling* coordinate.  It intentionally does
    # not become a trial score or a promotion rule.
    values = np.mean(np.column_stack([_zscore(frame[column]) for column in DIAGNOSTICS]), axis=1)
    q1, q2 = np.quantile(values, [.33, .67])
    return pd.Series(np.where(values <= q1, "weak", np.where(values <= q2, "middle", "strong")), index=frame.index)


def _farthest_fill(frame: pd.DataFrame, selected: set[int], count: int) -> list[int]:
    if count <= 0:
        return []
    matrix = np.column_stack([_zscore(frame[column]) for column in DIAGNOSTICS])
    candidates = [index for index in frame.index if index not in selected]
    if not candidates:
        return []
    result: list[int] = []
    if selected:
        selected_matrix = matrix[list(selected)]
    else:
        # Start from the most central row only to initialise a pure diversity
        # walk; this is not an economic preference.
        center = np.square(matrix).sum(axis=1)
        first = min(candidates, key=lambda index: (center[index], _stable_hash(str(frame.loc[index, "trial"]))))
        result.append(first); selected_matrix = matrix[[first]]; candidates.remove(first)
    while candidates and len(result) < count:
        distances = []
        reference = np.vstack([selected_matrix, matrix[result]]) if result else selected_matrix
        for index in candidates:
            distances.append((float(np.min(np.square(reference - matrix[index]).sum(axis=1))), index))
        _, winner = max(distances, key=lambda item: (item[0], -_stable_hash(str(frame.loc[item[1], "trial"]))))
        result.append(winner); candidates.remove(winner)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=36)
    args = parser.parse_args()
    if not 1 <= args.n <= 60:
        raise ValueError("--n must be between 1 and 60")
    if args.out.exists():
        raise FileExistsError(args.out)
    parts: list[pd.DataFrame] = []
    # Append-contract research may use multiple score roots.  A source-root
    # name plus trial name is the immutable identity; a bare trial name is
    # ambiguous across independently generated contracts.
    source_trial_records: dict[tuple[str, str], dict[str, Any]] = {}
    source_roots_by_name: dict[str, Path] = {}
    for root in (path.resolve() for path in args.descriptor_root):
        path = root / "trial_descriptor_summary.parquet"
        if not path.exists() or not (root / "correctness_report.json").exists():
            raise AssertionError(f"incomplete descriptor root: {root}")
        part = pd.read_parquet(path)
        if part.trial.duplicated().any():
            raise AssertionError(f"duplicate trial inside {root}")
        part["descriptor_root"] = root.name
        parts.append(part)
        descriptor_manifest = json.loads((root / "run_manifest.json").read_text())
        score_roots = [Path(value).resolve() for value in descriptor_manifest.get("score_roots", [])]
        if not score_roots or len(score_roots) != len(set(score_roots)):
            raise AssertionError(f"{root}: invalid source score roots")
        for score_root in score_roots:
            existing = source_roots_by_name.get(score_root.name)
            if existing is not None and existing != score_root:
                raise AssertionError(f"ambiguous score-root name {score_root.name}")
            source_roots_by_name[score_root.name] = score_root
            score_manifest = json.loads((score_root / "run_manifest.json").read_text())
            for record in score_manifest.get("trials", []):
                name = str(record["name"])
                key = (score_root.name, name)
                if key in source_trial_records:
                    raise AssertionError(f"duplicate source trial record {key}")
                source_trial_records[key] = dict(record)
    frame = pd.concat(parts, ignore_index=True)
    if frame.trial.duplicated().any():
        raise AssertionError("a trial appears in more than one descriptor root")
    missing = sorted(set(DIAGNOSTICS).difference(frame.columns))
    if missing:
        raise AssertionError(f"descriptor metrics missing: {missing}")
    frame["diagnostic_stratum"] = _diagnostic_strata(frame)
    frame["sampling_family"] = (
        frame.target_family.astype(str) + "|" + frame.loss.astype(str) + "|" + frame.feature_contract.astype(str)
    )
    frame["selection_reason"] = ""

    selected: set[int] = set()
    # First guarantee each observed target/loss/feature family contributes one
    # example.  Across families the requested stratum rotates weak/middle/
    # strong, so the labelled set is contrastive without spending three slots
    # per family or collapsing coverage when the budget is 30--60.
    for family, group in frame.groupby("sampling_family", sort=True):
        requested = ("weak", "middle", "strong")[_stable_hash(family) % 3]
        local = group.loc[group.diagnostic_stratum.eq(requested)]
        if local.empty:
            local = group
        candidate = min(local.index, key=lambda index: _stable_hash(str(frame.loc[index, "trial"])))
        selected.add(int(candidate))
        frame.loc[candidate, "selection_reason"] = f"family_coverage:{family}:{requested}"
        if len(selected) >= args.n:
            break
    # If family coverage exceeds the requested budget, retain deterministic
    # breadth rather than selecting the strongest direct diagnostics.
    if len(selected) > args.n:
        selected = set(sorted(selected, key=lambda index: _stable_hash(str(frame.loc[index, "trial"])))[:args.n])
        frame.loc[~frame.index.isin(selected), "selection_reason"] = ""
    # Fill the remaining budget using max-min diversity in cheap-descriptor
    # space.  This intentionally includes mediocre and poor trials so the
    # learned downstream proxy has contrastive labels.
    additions = _farthest_fill(frame, selected, args.n - len(selected))
    for index in additions:
        selected.add(index)
        frame.loc[index, "selection_reason"] = "descriptor_diversity_fill"

    result = frame.loc[sorted(selected)].copy().sort_values(
        ["target_family", "loss", "feature_contract", "diagnostic_stratum", "score_root", "trial"], kind="stable"
    ).reset_index(drop=True)
    args.out.mkdir(parents=True)
    result.to_parquet(args.out / "selected_trials.parquet", index=False, compression="zstd")
    result.loc[:, ["trial", "descriptor_root", "score_root", "target", "arm_name", "target_family", "loss", "feature_family", "feature_contract", "feature_count", "selection_reason"]].to_json(
        args.out / "selected_trials.json", orient="records", indent=2
    )
    plan: list[dict[str, Any]] = []
    for record in result.to_dict("records"):
        trial = str(record["trial"])
        score_root_name = str(record["score_root"])
        key = (score_root_name, trial)
        if key not in source_trial_records or score_root_name not in source_roots_by_name:
            raise AssertionError(f"selected trial lacks source record: {key}")
        source_root = source_roots_by_name[score_root_name]
        source_manifest = json.loads((source_root / "run_manifest.json").read_text())
        actual_contract = str(source_manifest.get("meta_feature_contract"))
        if actual_contract != str(record["feature_contract"]):
            raise AssertionError(f"{key}: descriptor/source feature-contract mismatch")
        plan.append({
            "trial": trial,
            "source_score_root": str(source_root),
            "source_feature_contract": actual_contract,
            "trial_config": source_trial_records[key],
            "selection_reason": record["selection_reason"],
            "diagnostic_stratum": record["diagnostic_stratum"],
        })
    (args.out / "selected_trial_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n")
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "representative expensive-MC1 label sampler only; no HPO ranking, no promotion, no model/live mutation",
        "descriptor_roots": [str(path.resolve()) for path in args.descriptor_root],
        "requested_trials": int(args.n), "selected_trials": int(len(result)),
        "diagnostics_used_only_for_diversity": list(DIAGNOSTICS),
        "strata": ["target family", "loss", "actual feature contract", "weak/middle/strong cheap diagnostics"],
        "source_root_trial_identity_is_preserved": True,
        "selection_authority": "none; downstream labels train/falsify the later learned proxy",
        "selected_trial_plan": "complete source trial configurations and their exact feature-contract/score-root lineage are persisted for strict prehistory score extension",
    })
    print(args.out)


if __name__ == "__main__":
    main()
