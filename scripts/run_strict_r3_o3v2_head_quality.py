#!/usr/bin/env python3
"""Offline O3-v2 consensus-head quality and complementarity audit.

It reads immutable target-free score receipts, joins outcomes only after the
receipt audit, and reports standalone, residualized, double-fault, semantic,
leave-one-out, and greedy-combination diagnostics.  It does not refit MC1 or
touch any canonical/live artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


SCHEMA = "strict_r3_o3v2_head_quality_v2"
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_label_available_ts",
    "semantic_path_valid", "semantic_policy_net_bps", "semantic_archetype", "semantic_tbm_event",
})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in paths:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _tail(outcome: np.ndarray, score: np.ndarray, tail: float) -> float:
    valid = np.isfinite(outcome) & np.isfinite(score)
    if valid.sum() < 20:
        return np.nan
    threshold = np.quantile(score[valid], 1.0 - tail, method="higher")
    chosen = outcome[valid & (score >= threshold)]
    return float(np.mean(chosen)) if len(chosen) else np.nan


def _residualize(score: np.ndarray, base: np.ndarray) -> np.ndarray:
    valid = np.isfinite(score) & np.isfinite(base)
    output = np.full(len(score), np.nan, dtype=float)
    if valid.sum() < 50:
        return output
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(base[valid], score[valid])
    output[valid] = score[valid] - fit.predict(base[valid])
    return output


def _policy_rank(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(pct=True, method="average").to_numpy(float)


def _score_path(root: Path, arm: str, token: str) -> Path:
    """Support both flat and partitioned immutable score receipts."""
    flat = root / "target_free_scores" / arm / f"month={token}.parquet"
    partitioned = root / "target_free_scores" / arm / f"month={token}" / "scores.parquet"
    found = [path for path in (flat, partitioned) if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"{arm} {token}: expected one immutable score receipt, found {found}")
    return found[0]


def _head_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Discover this architecture's frozen head names from its own receipt.

    O3-v2 H1 uses six semantic-family heads rather than the incumbent five
    cap heads.  A fixed incumbent name list would silently audit the wrong
    architecture.  The receipt itself is the contract authority.
    """
    heads = tuple(sorted(column for column in frame.columns if column.endswith("__rank")))
    if not heads:
        raise AssertionError("score receipt has no specialist rank fields")
    return heads


def _load(root: Path, arm: str, months: tuple[str, ...], policy: pd.DataFrame, semantic_root: Path) -> pd.DataFrame:
    pieces = []
    for token in months:
        path = _score_path(root, arm, token)
        raw = pd.read_parquet(path)
        if leak := PROHIBITED.intersection(raw.columns):
            raise AssertionError(f"{path}: outcome field in target-free receipt: {sorted(leak)}")
        # The specialist receipt retains the parent base rank under its F1
        # provenance name.  Normalize the diagnostic alias only in memory;
        # it neither modifies the immutable receipt nor changes a score.
        if "base_rank_ts" not in raw.columns and "f1_base_rank_ts" in raw.columns:
            raw["base_rank_ts"] = raw["f1_base_rank_ts"]
        if "base_rank_ts" not in raw.columns:
            raise AssertionError(f"{path}: missing parent base rank for residual diagnostics")
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        semantic_path = semantic_root / "parts" / f"month={token}" / "semantics.parquet"
        semantic = pd.read_parquet(semantic_path, columns=("candidate_id", "semantic_archetype"))
        joined = raw.merge(policy, on="candidate_id", how="left", validate="one_to_one").merge(semantic, on="candidate_id", how="left", validate="one_to_one")
        valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
        pieces.append(joined.loc[valid].copy())
    return pd.concat(pieces, ignore_index=True)


def _analyse(
    frame: pd.DataFrame, arm: str, heads: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    base = pd.to_numeric(frame["base_rank_ts"], errors="coerce").to_numpy(float)
    policy_rank = _policy_rank(frame)
    base_error = np.abs(policy_rank - base) >= .25
    scores = {head: pd.to_numeric(frame[head], errors="coerce").to_numpy(float) for head in heads}
    rows = []
    residuals = {}
    for head, score in scores.items():
        residual = _residualize(score, base)
        residuals[head] = residual
        head_error = np.abs(policy_rank - score) >= .25
        valid = np.isfinite(score) & np.isfinite(outcome)
        rows.append({
            "arm": arm, "head": head, "standalone_ic": float(spearmanr(score[valid], outcome[valid]).statistic),
            "residual_ic": float(spearmanr(residual[valid], outcome[valid]).statistic),
            "top1_ev": _tail(outcome, score, .01), "top2_ev": _tail(outcome, score, .02), "top5_ev": _tail(outcome, score, .05),
            "double_fault_rate": float(np.mean(base_error & head_error)), "base_error_coverage": float(np.mean(base_error)),
        })
    head_metrics = pd.DataFrame(rows)
    corr_rows = []
    for index, left in enumerate(heads):
        for right in heads[index + 1:]:
            valid = np.isfinite(residuals[left]) & np.isfinite(residuals[right])
            corr_rows.append({"arm": arm, "left": left, "right": right, "residual_spearman": float(spearmanr(residuals[left][valid], residuals[right][valid]).statistic)})
    correlations = pd.DataFrame(corr_rows)
    # Leave-one-out uses the canonical 75/25 blend, always in timestamp-local
    # rank space.  This is an economic diagnostic, not an admission simulation.
    loo_rows = []
    for omit in (None, *heads):
        keep = [head for head in heads if head != omit]
        consensus = np.nanmean(np.column_stack([scores[head] for head in keep]), axis=1)
        blend = .75 * base + .25 * consensus
        loo_rows.append({"arm": arm, "omitted_head": omit or "none", "heads": ",".join(keep), "top1_ev": _tail(outcome, blend, .01), "top2_ev": _tail(outcome, blend, .02), "top5_ev": _tail(outcome, blend, .05)})
    loo = pd.DataFrame(loo_rows)
    # Greedy best-1..5 selection: each step maximises mean top-1/2/5 economic
    # score.  It prevents a forced five-head production conclusion.
    selected: list[str] = []
    greedy_rows = []
    remaining = set(heads)
    while remaining:
        candidates = []
        for head in sorted(remaining):
            trial = selected + [head]
            consensus = np.nanmean(np.column_stack([scores[item] for item in trial]), axis=1)
            blend = .75 * base + .25 * consensus
            evs = [_tail(outcome, blend, tail) for tail in (.01, .02, .05)]
            candidates.append((float(np.nanmean(evs)), head, evs))
        score, winner, evs = max(candidates, key=lambda item: item[0])
        selected.append(winner)
        remaining.remove(winner)
        greedy_rows.append({"arm": arm, "n_heads": len(selected), "heads": ",".join(selected), "selection_score": score, "top1_ev": evs[0], "top2_ev": evs[1], "top5_ev": evs[2]})
    semantic_rows = []
    for head, score in scores.items():
        error = np.abs(policy_rank - score)
        for archetype, group in frame.assign(__error__=error).groupby("semantic_archetype", dropna=False):
            semantic_rows.append({"arm": arm, "head": head, "archetype": str(archetype), "rows": len(group), "mean_rank_error": float(group["__error__"].mean()), "policy_net_bps": float(group["policy_net_bps"].mean())})
    return head_metrics, correlations, loo, pd.DataFrame(greedy_rows), pd.DataFrame(semantic_rows)


def run(*, root: Path, semantic_root: Path, policy_path: Path, out: Path, arms: tuple[str, ...], months: tuple[str, ...]) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps"))
    all_parts = []
    for arm in arms:
        frame = _load(root, arm, months, policy, semantic_root)
        parts = _analyse(frame, arm, _head_columns(frame))
        all_parts.append(parts)
    names = ("head_metrics", "head_residual_correlation", "head_leave_one_out", "head_greedy_selection", "head_semantic_correctness")
    for index, name in enumerate(names):
        pd.concat([part[index] for part in all_parts], ignore_index=True).to_parquet(out / f"{name}.parquet", index=False, compression="zstd")
    manifest = {"schema": SCHEMA, "scope": "offline head-quality research only; no model refit, MC1, or live mutation", "arms": list(arms), "months": list(months), "head_contract": "rank fields discovered from each immutable O3-v2 score receipt", "causality": "target-free score files audited before outcomes/semantic labels are joined", "source_hashes": {"root": _hash(root), "semantic": _hash(semantic_root), "policy": _hash(policy_path)}}
    fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--months", required=True)
    args = parser.parse_args()
    run(root=args.root, semantic_root=args.semantic_root, policy_path=args.policy_path, out=args.out, arms=tuple(args.arms.split(",")), months=tuple(args.months.split(",")))


if __name__ == "__main__":
    main()
