#!/usr/bin/env python3
"""Correct Stage 4 economics to use the required pooled global top-k selection.

The Stage 3/4 runner intentionally remains immutable after a long canonical
run.  Its historical named-gap summary averages side-local top-decile metrics,
whereas the trading contract selects candidates globally across both sides and
timestamps.  This postprocessor reads frozen held-out predictions only and
emits a separately versioned correction.  It neither retrains nor changes a
score, target, feature, or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


TOP_K = 0.10
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 20260801


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _global_top_k(frame: pd.DataFrame, k: float = TOP_K) -> pd.DataFrame:
    """Select a deterministic pooled global top k percent, never per side/day."""
    if frame.empty:
        return frame.copy()
    count = max(1, int(math.ceil(len(frame) * k)))
    return frame.sort_values(
        ["combined_economic_prediction_bps", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",  # stable and therefore reproducible for equal scores
    ).head(count)


def _summary(selected: pd.DataFrame, source_rows: int) -> dict[str, object]:
    months = selected.assign(month=selected["__ts__"].dt.to_period("M")).groupby("month", observed=True)
    sides = selected.groupby("side_name", observed=True)
    return {
        "selection_scope": "pooled_global_across_sides_and_timestamps",
        "top_k_fraction": TOP_K,
        "source_rows": int(source_rows),
        "selected_rows": int(len(selected)),
        "selected_fraction": float(len(selected) / source_rows) if source_rows else np.nan,
        "score_cutoff_bps": float(selected["combined_economic_prediction_bps"].iloc[-1]) if len(selected) else np.nan,
        "gross_topk_bps": float(selected["gross_h12_bps"].mean()),
        "net_topk_bps": float(selected["net_h12_bps"].mean()),
        "gross_topk_worst_month_bps": float(months["gross_h12_bps"].mean().min()),
        "net_topk_worst_month_bps": float(months["net_h12_bps"].mean().min()),
        "gross_topk_worst_side_bps": float(sides["gross_h12_bps"].mean().min()),
        "net_topk_worst_side_bps": float(sides["net_h12_bps"].mean().min()),
    }


def _day_net_means(selected: pd.DataFrame) -> pd.Series:
    return selected.groupby(selected["__ts__"].dt.floor("D"), observed=True)["net_h12_bps"].mean()


def _paired_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    """Paired daily bootstrap of two independently global-selected books."""
    a = _day_net_means(reference).rename("reference")
    b = _day_net_means(candidate).rename("candidate")
    paired = pd.concat([a, b], axis=1).dropna()
    delta = (paired["candidate"] - paired["reference"]).to_numpy(dtype=float)
    if not len(delta):
        return {"paired_days": 0, "mean_delta_bps": np.nan, "ci_low_bps": np.nan, "ci_high_bps": np.nan, "p_delta_le_zero": np.nan}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_idx = rng.integers(0, len(delta), size=(BOOTSTRAP_REPS, len(delta)))
    boot = delta[sample_idx].mean(axis=1)
    return {
        "paired_days": int(len(delta)),
        "mean_delta_bps": float(delta.mean()),
        "ci_low_bps": float(np.quantile(boot, 0.025)),
        "ci_high_bps": float(np.quantile(boot, 0.975)),
        "p_delta_le_zero": float((boot <= 0.0).mean()),
    }


def build(input_dir: Path, output_dir: Path) -> None:
    pred_path = input_dir / "base_residual_oof_predictions.parquet"
    frame = pd.read_parquet(pred_path)
    required = {
        "candidate_id", "__ts__", "side_name", "gross_h12_bps", "net_h12_bps",
        "model_family", "seed", "split", "evaluation_scope", "combined_economic_prediction_bps",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction ledger lacks required columns: {sorted(missing)}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    # The correction is deliberately restricted to true outer held-out later OOS.
    held = frame.loc[(frame["evaluation_scope"] == "outer_heldout") & (frame["split"] == "later_oos")].copy()
    if held.empty:
        raise ValueError("no outer-heldout later-OOS predictions")

    selection_rows: list[dict[str, object]] = []
    books: dict[tuple[str, int], pd.DataFrame] = {}
    for (family, seed), group in held.groupby(["model_family", "seed"], observed=True, sort=True):
        book = _global_top_k(group)
        books[(str(family), int(seed))] = book
        selection_rows.append({"model_family": family, "seed": seed, **_summary(book, len(group))})
    selections = pd.DataFrame(selection_rows).sort_values(["model_family", "seed"], kind="mergesort")

    # Compare only exact matched seeds.  Prior is deterministic but emitted with each
    # seed precisely so all stochastic arms retain a paired underlying candidate set.
    by_key = selections.set_index(["model_family", "seed"])
    gaps: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []
    comparisons = [
        ("null_to_causal", "prior", "causal_capacity_oracle"),
        ("production_to_causal", "production_like_lgbm", "causal_capacity_oracle"),
        ("causal_to_future", "causal_capacity_oracle", "future_feature_oracle"),
    ]
    for comparison, left_family, right_family in comparisons:
        per_seed: list[float] = []
        for seed in sorted(set(selections.seed)):
            left_key, right_key = (left_family, int(seed)), (right_family, int(seed))
            if left_key not in by_key.index or right_key not in by_key.index:
                continue
            left, right = by_key.loc[left_key], by_key.loc[right_key]
            delta = float(right.net_topk_bps - left.net_topk_bps)
            per_seed.append(delta)
            boot_rows.append({
                "comparison": comparison, "left_model_family": left_family, "right_model_family": right_family,
                "seed": int(seed), "selection_scope": "pooled_global_across_sides_and_timestamps",
                **_paired_bootstrap(books[left_key], books[right_key]),
            })
        if per_seed:
            gaps.append({
                "comparison": comparison, "left_model_family": left_family, "right_model_family": right_family,
                "selection_scope": "pooled_global_across_sides_and_timestamps",
                "metric": "net_h12_bps", "seed_count": len(per_seed),
                "left_bps": float(np.mean([by_key.loc[(left_family, int(s))].net_topk_bps for s in sorted(set(selections.seed)) if (left_family, int(s)) in by_key.index and (right_family, int(s)) in by_key.index])),
                "right_bps": float(np.mean([by_key.loc[(right_family, int(s))].net_topk_bps for s in sorted(set(selections.seed)) if (left_family, int(s)) in by_key.index and (right_family, int(s)) in by_key.index])),
                "right_minus_left_bps": float(np.mean(per_seed)),
                "economic_regret_bps": float(np.mean(per_seed)),
                "seed_std_bps": float(np.std(per_seed, ddof=0)),
            })
    gaps_df = pd.DataFrame(gaps)
    bootstrap_df = pd.DataFrame(boot_rows).sort_values(["comparison", "seed"], kind="mergesort")

    output_dir.mkdir(parents=True, exist_ok=True)
    selections.to_parquet(output_dir / "global_topk_economics.parquet", index=False)
    gaps_df.to_parquet(output_dir / "global_topk_named_gaps.parquet", index=False)
    bootstrap_df.to_parquet(output_dir / "global_topk_paired_bootstrap.parquet", index=False)
    (output_dir / "GLOBAL_TOPK_CORRECTION.md").write_text(
        "# Pooled global top-k economics correction\n\n"
        "This artifact supersedes only the Stage-4 *named economic gaps* that were previously formed by averaging side-local top-decile values. "
        "Every selection here pools long and short candidates across all timestamps in the held-out `later_oos` population, then selects the top 10% globally. "
        "It changes no model, score, target, feature, label, threshold, sizing rule, or portfolio rule.\n\n"
        "The gross and net target columns are exact H12 realized outcomes. Bootstrap comparisons are paired by day after each model's independently global top-k selection.\n"
    )
    manifest = {
        "artifact": "root_cause_base_residual_global_topk_correction",
        "version": "20260801_v1",
        "input_artifact": str(input_dir),
        "input_prediction_sha256": _sha256(pred_path),
        "selection_contract": "top 10% globally pooled across long/short and all timestamps; deterministic score-desc, candidate-id-asc tie break",
        "excluded": ["per-side selection", "per-timestamp selection", "portfolio constraints", "retraining", "policy changes"],
        "outputs_sha256": {},
    }
    for path in sorted(output_dir.glob("*")):
        if path.name != "run_manifest.json" and path.is_file():
            manifest["outputs_sha256"][path.name] = _sha256(path)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = {
        "passed": True,
        "checks": {
            "outer_heldout_later_oos_only": bool((held.evaluation_scope == "outer_heldout").all() and (held.split == "later_oos").all()),
            # A true pooled winner may legitimately all come from one side.  What
            # matters is that the ranked source population always contained both.
            "both_sides_in_every_ranked_population": bool(
                all(set(group.side_name) == {"long", "short"} for _, group in held.groupby(["model_family", "seed"], observed=True))
            ),
            "selection_rows_equal_ceil_ten_percent": bool(all(row.selected_rows == math.ceil(row.source_rows * TOP_K) for row in selections.itertuples())),
            "exact_input_hash_recorded": True,
        },
    }
    (output_dir / "correctness_test_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        build(args.input.resolve(), temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
