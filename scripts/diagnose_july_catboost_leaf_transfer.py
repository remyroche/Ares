#!/usr/bin/env python3
"""Bounded, non-promotional July CatBoost residual leaf-transfer diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import load_execution_ev_model_ablation_bundle  # noqa: E402

ID = ["__ts__", "__symbol__", "side_name", "candidate_id"]
DEFAULT_OLD = ROOT / "data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet"
DEFAULT_FORWARD = ROOT / "data_perp/artifacts/execution_ev_context_head_clean_forward_july19_20260726_v2/strict_forward_winner_inputs_and_raw_scores.parquet"
DEFAULT_BUNDLE = ROOT / "data_perp/artifacts/execution_ev_context_head_clean_20260726_v1/execution_ev_model_ablation_bundle.joblib"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/july_catboost_residual_leaf_transfer_20260726_v2"


def js_divergence(train: np.ndarray, evaluation: np.ndarray) -> float:
    """Jensen-Shannon divergence over aligned non-negative count vectors."""
    p = np.asarray(train, dtype=float); q = np.asarray(evaluation, dtype=float)
    p = p / max(p.sum(), 1.0); q = q / max(q.sum(), 1.0); m = (p + q) / 2.0
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)), 0.0))
    return float((kl(p, m) + kl(q, m)) / 2.0)


def leaf_support_features(train_leaves: np.ndarray, eval_leaves: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-row causal support from early-July leaf occupancy only."""
    train = np.asarray(train_leaves, dtype=np.int64); evaluation = np.asarray(eval_leaves, dtype=np.int64)
    supports = np.zeros_like(evaluation, dtype=np.int32); js: list[float] = []
    unseen: list[float] = []
    for tree in range(train.shape[1]):
        counts = np.bincount(train[:, tree])
        leaf = evaluation[:, tree]
        in_range = leaf < len(counts)
        supports[:, tree] = 0
        supports[in_range, tree] = counts[leaf[in_range]]
        all_leaves = np.arange(max(len(counts), int(leaf.max(initial=0)) + 1))
        test_counts = np.bincount(leaf, minlength=len(all_leaves))
        train_counts = np.bincount(train[:, tree], minlength=len(all_leaves))
        js.append(js_divergence(train_counts, test_counts))
        unseen.append(float((supports[:, tree] == 0).mean()))
    out = pd.DataFrame({
        "leaf_support_mean": supports.mean(axis=1),
        "leaf_support_q10": np.quantile(supports, 0.10, axis=1),
        "leaf_support_min": supports.min(axis=1),
        "leaf_unseen_tree_fraction": (supports == 0).mean(axis=1),
        "leaf_low_support_tree_fraction": (supports < 5).mean(axis=1),
    })
    return out, {"mean_tree_js": float(np.mean(js)), "max_tree_js": float(np.max(js)), "mean_tree_unseen_fraction": float(np.mean(unseen))}


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    return frame


def _period_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Outcome and rank metrics for a fixed model in one genuinely held-out period.

    The top-decile is by the locally fit raw EV score only: it intentionally does
    not claim performance after the production 21-day admission calibrator.
    """
    if frame.empty:
        return {"rows": 0}
    net = frame["execution_net_ev_12h"]
    score = frame["july_model_raw_ev"]
    positive = (net > 0).astype(int)
    n_top = max(1, int(np.ceil(len(frame) * 0.10)))
    top = frame.nlargest(n_top, "july_model_raw_ev")
    auc: float | None = None
    if positive.nunique() == 2:
        # Mann-Whitney AUC of raw score against economically positive outcome.
        rank = score.rank(method="average")
        n_pos = int(positive.sum()); n_neg = int(len(positive) - n_pos)
        auc = float((rank[positive.eq(1)].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    unconditional = float(net.mean())
    top_net = float(top["execution_net_ev_12h"].mean())
    return {
        "rows": int(len(frame)),
        "mean_net_ev": unconditional,
        "mean_residual": float(frame["residual_target"].mean()),
        "positive_rate": float(positive.mean()),
        "mean_prediction": float(score.mean()),
        "rank_spearman_prediction_vs_net_ev": float(score.corr(net, method="spearman")),
        "auc_positive_net_ev": auc,
        "raw_model_top10_rows": int(len(top)),
        "raw_model_top10_net_ev": top_net,
        "raw_model_top10_positive_rate": float((top["execution_net_ev_12h"] > 0).mean()),
        "raw_model_top10_lift_vs_unconditional_net_ev": float(top_net - unconditional),
    }


def _leaf_occupancy_table(
    train_leaves: np.ndarray,
    train_net: np.ndarray,
    eval_leaves: np.ndarray,
    eval_net: np.ndarray,
    side: str,
    period: str,
) -> pd.DataFrame:
    """Per-tree leaf support and realised economics, without collapsing signatures."""
    rows: list[pd.DataFrame] = []
    for tree in range(train_leaves.shape[1]):
        train_df = pd.DataFrame({"leaf_id": train_leaves[:, tree], "train_net_ev": train_net})
        eval_df = pd.DataFrame({"leaf_id": eval_leaves[:, tree], "evaluation_net_ev": eval_net})
        train_group = train_df.groupby("leaf_id", as_index=False).agg(train_count=("leaf_id", "size"), train_mean_net_ev=("train_net_ev", "mean"))
        eval_group = eval_df.groupby("leaf_id", as_index=False).agg(evaluation_count=("leaf_id", "size"), evaluation_mean_net_ev=("evaluation_net_ev", "mean"))
        merged = train_group.merge(eval_group, on="leaf_id", how="outer").fillna({"train_count": 0, "evaluation_count": 0})
        merged["side"] = side; merged["period"] = period; merged["tree_index"] = tree
        merged["unseen_in_early_july_train"] = merged["train_count"].eq(0)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, default=DEFAULT_OLD)
    p.add_argument("--forward", type=Path, default=DEFAULT_FORWARD)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--train-start", default="2026-07-01T00:00:00Z")
    p.add_argument("--eval-start", default="2026-07-08T00:00:00Z")
    p.add_argument("--early-eval-end", default="2026-07-11T00:00:00Z")
    p.add_argument("--clusters", type=int, default=4)
    return p


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    bundle = load_execution_ev_model_ablation_bundle(args.bundle)
    features = list(bundle.raw_feature_columns)
    base_cols = [*ID, "execution_decision_utc", "execution_label_end_utc", "execution_net_ev_12h", "existing_alpha_ev", *features]
    old = _read(args.old, list(dict.fromkeys(base_cols)))
    new = _read(args.forward, list(dict.fromkeys(base_cols)))
    data = pd.concat([old, new], ignore_index=True).drop_duplicates(ID, keep="last")
    for c in ["execution_decision_utc", "execution_label_end_utc"]: data[c] = pd.to_datetime(data[c], utc=True, errors="raise")
    data["residual_target"] = data["execution_net_ev_12h"] - data["existing_alpha_ev"]
    finite = np.isfinite(data.loc[:, features].to_numpy(dtype=float)).all(axis=1) & np.isfinite(data["residual_target"])
    data = data.loc[finite].copy()
    train_start = pd.Timestamp(args.train_start); eval_start = pd.Timestamp(args.eval_start); early_end = pd.Timestamp(args.early_eval_end)
    # Strict 12h purge plus resolved-label condition prevents pre-evaluation
    # July outcomes leaking into this deliberately local diagnostic fit.
    train = data.loc[(data["__ts__"] >= train_start) & (data["__ts__"] < eval_start - pd.Timedelta(hours=12)) & (data["execution_label_end_utc"] < eval_start)].copy()
    periods = {"early_july_holdout": data.loc[(data["__ts__"] >= eval_start) & (data["__ts__"] < early_end)].copy(), "late_july_forward": data.loc[data["__ts__"] >= early_end].copy(), "may_reverse_diagnostic": data.loc[(data["__ts__"] >= pd.Timestamp("2026-05-01", tz="UTC")) & (data["__ts__"] < pd.Timestamp("2026-06-01", tz="UTC"))].copy(), "june_reverse_diagnostic": data.loc[(data["__ts__"] >= pd.Timestamp("2026-06-01", tz="UTC")) & (data["__ts__"] < pd.Timestamp("2026-07-01", tz="UTC"))].copy()}
    if len(train) < 100: raise ValueError("too few purged early-July rows")
    reports: dict[str, object] = {}; rows: list[pd.DataFrame] = []; occupancies: list[pd.DataFrame] = []
    for side in ("long", "short"):
        tr = train.loc[train.side_name.eq(side)].copy()
        if len(tr) < 50: continue
        model = CatBoostRegressor(loss_function="MAE", iterations=250, learning_rate=0.03, depth=6, l2_leaf_reg=6.0, random_seed=42, thread_count=1, verbose=False, allow_writing_files=False, random_strength=0.5, bagging_temperature=1.0, bootstrap_type="Bayesian")
        model.fit(tr[features], tr["residual_target"])
        train_leaf = np.asarray(model.calc_leaf_indexes(tr[features]), dtype=np.int64)
        clusterer = MiniBatchKMeans(n_clusters=min(int(args.clusters), len(tr)), random_state=42, n_init=5, batch_size=min(1024, len(tr))).fit(train_leaf.astype(np.float32))
        reports[side] = {"train_rows": int(len(tr)), "trees": int(train_leaf.shape[1]), "leaf_signature_clusters": int(clusterer.n_clusters), "periods": {}}
        for period, raw in periods.items():
            ev = raw.loc[raw.side_name.eq(side)].copy()
            if ev.empty: continue
            leaves = np.asarray(model.calc_leaf_indexes(ev[features]), dtype=np.int64)
            support, drift = leaf_support_features(train_leaf, leaves)
            ev = ev.reset_index(drop=True); ev["july_model_raw_ev"] = ev["existing_alpha_ev"].to_numpy() + model.predict(ev[features]); ev = pd.concat([ev, support], axis=1); ev["leaf_signature_cluster"] = clusterer.predict(leaves.astype(np.float32))
            reports[side]["periods"][period] = {**_period_metrics(ev), **drift, "cluster_net_ev": ev.groupby("leaf_signature_cluster")["execution_net_ev_12h"].mean().to_dict()}
            occupancies.append(_leaf_occupancy_table(train_leaf, tr["execution_net_ev_12h"].to_numpy(), leaves, ev["execution_net_ev_12h"].to_numpy(), side, period))
            ev["transfer_side"] = side; ev["transfer_period"] = period; rows.append(ev.loc[:, [*ID, "transfer_side", "transfer_period", "execution_net_ev_12h", "residual_target", "july_model_raw_ev", "leaf_support_mean", "leaf_support_q10", "leaf_support_min", "leaf_unseen_tree_fraction", "leaf_low_support_tree_fraction", "leaf_signature_cluster"]])
    output = pd.concat(rows, ignore_index=True)
    global_periods = {period: _period_metrics(output.loc[output["transfer_period"].eq(period)]) for period in periods}
    occupancy = pd.concat(occupancies, ignore_index=True)
    args.output_dir.mkdir(parents=True)
    output.to_parquet(args.output_dir / "row_leaf_transfer_diagnostics.parquet", index=False)
    occupancy.to_parquet(args.output_dir / "per_tree_leaf_occupancy_and_economics.parquet", index=False)
    payload = {
        "schema": "july_catboost_residual_leaf_transfer_v2",
        "contract": "July-only model fit on resolved/purged early July; May/June are reverse-transfer diagnostics only and not promotion evidence. Top-decile metrics are raw local model ranking only, before the 21-day admission calibrator.",
        "train_rows": int(len(train)),
        "reports": reports,
        "global_periods": global_periods,
        "outputs": {"rows": str(args.output_dir / "row_leaf_transfer_diagnostics.parquet"), "per_tree_leaf_occupancy": str(args.output_dir / "per_tree_leaf_occupancy_and_economics.parquet")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload


if __name__ == "__main__": print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
