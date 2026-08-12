#!/usr/bin/env python3
"""Three-fold mechanism-neutral feature selection for canonical residual meta.

Selection is fit only on the pre-2025 history and is therefore not allowed to
use the 2025 ablation results.  Each block is screened with the same shallow
LambdaRank model, then highly correlated substitutes are collapsed at |rho| >=
.90.  Features recurring in at least two of three chronological screens are
kept, with a small top-gain fallback when a block has insufficient recurrence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_canonical_residual_meta_block_ablation import (  # noqa: E402
    ARM_BLOCKS,
    BLOCKS,
    DEFAULT_HEADS,
    _feature_frame,
    _groups,
    _load,
    _map_canonical,
)

DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1"
SELECTION_MONTHS = ("2024-07", "2024-09", "2024-11")


def _screen(train: pd.DataFrame, held: pd.DataFrame, xtr: pd.DataFrame, xte: pd.DataFrame, features: list[str], target: np.ndarray, *, seed: int) -> tuple[pd.Series, float]:
    order, groups = _groups(train)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
        n_estimators=80, learning_rate=0.05, max_depth=3, num_leaves=8,
        min_child_samples=max(80, int(0.03 * len(train))), feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=1.0, lambda_l2=10.0,
        max_bin=63, label_gain=[0.0, 0.25, 1.0, 3.0, 7.0],
        random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(xtr[features].iloc[order], target[order], group=groups)
    raw = model.predict(xte[features])
    held_score = pd.Series(raw, index=held.index)
    ic = float(pd.concat([held_score.rename("score"), held.exact_net_bps], axis=1).corr(method="spearman").iloc[0, 1])
    return pd.Series(model.feature_importances_, index=features, dtype=float), ic


def _correlation_prune(frame: pd.DataFrame, features: list[str], gains: pd.Series, threshold: float = 0.90) -> list[str]:
    if not features:
        return []
    order = gains.reindex(features).fillna(0.0).sort_values(ascending=False).index.tolist()
    corr = frame[order].corr().abs().fillna(0.0)
    keep: list[str] = []
    for feature in order:
        if not keep or float(corr.loc[feature, keep].max()) < threshold:
            keep.append(feature)
    return keep


def run(*, head_path: Path = DEFAULT_HEADS, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    panel, context, context_hash = _load()
    heads = pd.read_parquet(head_path)
    panel = panel.merge(heads, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_head"))
    audits: list[dict[str, object]] = []
    gains_by_block: dict[str, list[pd.Series]] = {b: [] for b in BLOCKS}
    rec_by_block: dict[str, list[set[str]]] = {b: [] for b in BLOCKS}
    for fold_idx, month in enumerate(SELECTION_MONTHS):
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        train = panel.loc[
            (panel.__ts__ < pd.Timestamp(month, tz="UTC"))
            & (panel.label_available_ts < pd.Timestamp(month, tz="UTC"))
        ].copy()
        if len(train) < 500 or held.empty:
            continue
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy()
            te = held.loc[held.side_name.eq(side)].copy()
            if len(tr) < 250 or te.empty:
                continue
            tr_expected, te_expected = _map_canonical(tr, te)
            tr["canonical_expected_net_bps"] = tr_expected
            te["canonical_expected_net_bps"] = te_expected
            residual = tr.exact_net_bps.to_numpy(float) - tr_expected
            grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
            xtr, xte, _ = _feature_frame(tr, te, context)
            xtr["canonical_expected_net_bps"] = tr_expected
            xte["canonical_expected_net_bps"] = te_expected
            xtr["base_plus_consensus25"] = tr.base_plus_consensus25.to_numpy(float)
            xte["base_plus_consensus25"] = te.base_plus_consensus25.to_numpy(float)
            for block, block_features in BLOCKS.items():
                features = [c for c in block_features if c in xtr.columns]
                features = ["canonical_expected_net_bps", "base_plus_consensus25", *features]
                features = list(dict.fromkeys(features))
                gains, ic = _screen(tr, te, xtr, xte, features, grade, seed=20260808 + 100 * fold_idx + (0 if side == "long" else 1))
                gains_by_block[block].append(gains)
                positive = gains[gains > 0].sort_values(ascending=False).head(max(8, int(np.ceil(len(features) * 0.4))).__int__()).index
                rec_by_block[block].append(set(positive))
                for f, g in gains.items():
                    audits.append({"fold": fold_idx, "selection_month": month, "side": side, "block": block, "feature": f, "gain_importance": float(g), "held_rank_ic": ic, "positive_gain": bool(g > 0)})
    selected: dict[str, list[str]] = {}
    recurrence: dict[str, dict[str, int]] = {}
    for block, gains_list in gains_by_block.items():
        if not gains_list:
            selected[block] = list(BLOCKS[block]); recurrence[block] = {}
            continue
        all_features = sorted(set().union(*(g.index.tolist() for g in gains_list)))
        mean_gain = pd.concat(gains_list, axis=1).reindex(all_features).fillna(0.0).mean(axis=1)
        recurrence_counts = {f: sum(f in r for r in rec_by_block[block]) for f in all_features}
        recurrence[block] = recurrence_counts
        recurring = [f for f in all_features if recurrence_counts[f] >= 2 and mean_gain[f] > 0]
        # Keep the mechanism anchors even if their gain is weak in one screen.
        anchors = ["canonical_expected_net_bps", "base_plus_consensus25"]
        recurring = [*anchors, *[f for f in recurring if f not in anchors]]
        # Correlation pruning uses the last selection fold's train substrate;
        # it is only a redundancy veto, never an economic selection signal.
        prune_features = [f for f in recurring if f in xtr.columns] if 'xtr' in locals() else recurring
        if 'xtr' in locals() and prune_features:
            pruned = _correlation_prune(xtr, prune_features, mean_gain)
        else:
            pruned = recurring
        if len(pruned) < 3:
            pruned = [*anchors, *mean_gain.drop(labels=anchors, errors="ignore").sort_values(ascending=False).head(6).index]
        selected[block] = list(dict.fromkeys(pruned))
    output_dir.mkdir(parents=True)
    pd.DataFrame(audits).to_parquet(output_dir / "selection_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_canonical_residual_feature_selection_v1",
        "status": "COMPLETE", "selection_months": list(SELECTION_MONTHS),
        "selection_history_only": True, "correlation_threshold": 0.90,
        "recurrence_rule": "positive gain in at least two of three chronological screens",
        "selected_features": selected, "recurrence": recurrence,
        "context_sha256": context_hash,
    }
    (output_dir / "selected_features.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "run_manifest.json").write_text(json.dumps({"schema": manifest["schema"], "status": "COMPLETE", "selection_months": list(SELECTION_MONTHS), "artifacts": ["selection_audit.parquet", "selected_features.json", "run_manifest.json"]}, indent=2) + "\n")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=Path, default=DEFAULT_HEADS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(head_path=args.heads, output_dir=args.output_dir))
