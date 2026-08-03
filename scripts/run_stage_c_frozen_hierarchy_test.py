#!/usr/bin/env python3
"""Frozen E15/E16 hierarchy replay with a Stage-C retention matrix only.

This is deliberately the sole code path allowed to turn a conditional feature
result into a Stage-B economic diagnostic.  Clear/adverse heads, state-net
estimators, costs, policy, IDs and global common-bps ranking stay frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_exact_h12_target_purity_ablation as v11
from scripts import run_stage_c_conditional_retention_ablation as conditional

FEATURE_PANEL = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v1/stage_c_candidate_population.parquet"
CONDITIONAL_RESULTS = ROOT / "data_perp/artifacts/stage_c_conditional_retention_ablation_20260731_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_c_frozen_hierarchy_test_20260731_v1"
SIDES = ("long", "short")


def _matrix(frame: pd.DataFrame, base_raw: list[str], selected_new: list[str]) -> pd.DataFrame:
    """Fold-local transformed retention matrix; base policy inputs are unchanged."""
    raw = v11._matrix(frame, base_raw).reset_index(drop=True)
    policy = v11._policy_features(frame).reset_index(drop=True)
    new = frame.loc[:, selected_new].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True) if selected_new else pd.DataFrame(index=raw.index)
    return pd.concat([raw, policy, new], axis=1)


def _winner(predictions: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    """Use development OOF only; never look at Aug--Nov labels to choose fields."""
    dev = predictions.loc[predictions.month.lt("2024-08")].copy()
    summary = dev.groupby("arm").agg(auc=("prediction", lambda s: np.nan), rows=("label", "size"))
    # Derive comparison directly from development prediction rows for a stable,
    # transparent predeclared gate.
    from sklearn.metrics import brier_score_loss, roc_auc_score
    metrics = []
    for arm, local in dev.groupby("arm"):
        if local.label.nunique() == 2:
            metrics.append({"arm": arm, "auc": float(roc_auc_score(local.label, local.prediction)), "brier": float(brier_score_loss(local.label, local.prediction)), "months": int(local.month.nunique())})
    score = pd.DataFrame(metrics).set_index("arm")
    control = score.loc["C0"]
    candidates = score.drop(index=[name for name in ("C0", "C4", "C5") if name in score.index])
    survivors = candidates.loc[(candidates.auc > control.auc + 0.002) & (candidates.brier <= control.brier + 0.002) & (candidates.months >= 2)].sort_values("auc", ascending=False)
    return survivors.index.tolist()[:2], {"development_metrics": score.reset_index().to_dict("records"), "survivors": survivors.reset_index().to_dict("records")}


def _score_period(train: pd.DataFrame, test: pd.DataFrame, base_by_side: dict[str, list[str]], candidate_groups: dict[str, list[str]], *, seed: int, trees: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts, selected_manifest = [], {}
    for side_index, side in enumerate(SIDES):
        tr = train.loc[train.side.eq(side)].reset_index(drop=True)
        te = test.loc[test.side.eq(side)].reset_index(drop=True)
        clear = np.isin(v11._persistence_event(tr, "h0"), ("retained", "giveback"))
        y = (v11._persistence_event(tr.loc[clear].reset_index(drop=True), "h0") == "retained").astype(int)
        raw_new = [name for values in candidate_groups.values() for name in values]
        selected_new, selector = conditional._select_incremental(tr.loc[clear].reset_index(drop=True), y, raw_new, seed=seed + side_index, trees=trees)
        base_x_train = v11._features_for(tr, base_by_side[side])
        base_x_test = v11._features_for(te, base_by_side[side])
        retain_unclipped_train = _matrix(tr, base_by_side[side], selected_new)
        retain_unclipped_test = _matrix(te, base_by_side[side], selected_new)
        retain_x_train, retain_x_test, transformer = conditional._fit_transform(retain_unclipped_train, retain_unclipped_test, retain_unclipped_train.columns.tolist())
        common = dict(seed=seed + side_index * 100, trees=trees, token="h0", return_components=True)
        control = v11._hierarchical_persistence_expected_net(tr, te, base_x_train, base_x_test, **common)
        changed = v11._hierarchical_persistence_expected_net(tr, te, base_x_train, base_x_test, retention_x_train=retain_x_train, retention_x_test=retain_x_test, **common)
        if not np.array_equal(control["p_clear_cost_before_adverse"], changed["p_clear_cost_before_adverse"]) or not np.array_equal(control["p_adverse_given_not_clear"], changed["p_adverse_given_not_clear"]):
            raise AssertionError("Stage-C hierarchy changed a frozen non-retention head")
        base = te.loc[:, ["candidate_id", "side", "decision_ts", "label_available_ts", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps"]].copy()
        for arm, component in (("H_control", control), ("H_new", changed)):
            current = base.copy()
            current["raw_score"] = component["raw_score"]
            current["arm"] = arm
            parts.append(current)
        selected_manifest[side] = {"selected_new": selected_new, "selector": selector, "transformer": transformer}
    return pd.concat(parts, ignore_index=True), selected_manifest


def run(*, feature_panel: Path, conditional_output: Path, output: Path, smoke: bool = False, seed: int = 20260731) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    compatible = pd.read_parquet(feature_panel)
    conditional_predictions = pd.read_parquet(conditional_output / "retention_conditional_oof_predictions.parquet")
    survivor_arms, decision = _winner(conditional_predictions)
    frame, raw = v11._read(v11.PANEL, v11.ALIGNMENT, v11.FEATURE_CONTRACT, v11.POSTCOST_EVENTS, v11.PERSISTENCE_LABELS, smoke=False)
    frame = frame.merge(compatible.drop(columns=[name for name in ("side", "decision_ts", "label_available_ts", "exact_h12_net_bps") if name in compatible]), on="candidate_id", how="inner", validate="one_to_one")
    if smoke:
        frame = frame.groupby([frame.decision_ts.dt.strftime("%Y-%m"), "side"], group_keys=False).head(1000).reset_index(drop=True)
    if not frame.feature_available_ts.le(frame.decision_ts).all():
        raise AssertionError("new retention input is not decision-time available")
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        if not survivor_arms:
            pd.DataFrame(columns=["arm", "scope", "net_bps"]).to_parquet(stage / "stage_b_incremental_retention_results.parquet", index=False)
            (stage / "stage_b_incremental_retention_summary.md").write_text("# Frozen Stage-B incremental retention test\n\nNo Stage-1 mechanism passed the development-only conditional retention gate; Stage B was not run.\n", encoding="utf-8")
            (stage / "feature_disposition.yaml").write_text("terminal_decision: CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION\n", encoding="utf-8")
            (stage / "retention_compact_feature_manifest.json").write_text(json.dumps({"survivors": [], "decision": decision}, indent=2) + "\n", encoding="utf-8")
            (stage / "run_manifest.json").write_text(json.dumps({"status": "NO_STAGE1_SURVIVOR", "decision": decision}, indent=2) + "\n", encoding="utf-8")
            os.replace(stage, output)
            return {"status": "NO_STAGE1_SURVIVOR", **decision}
        all_groups = conditional._group_features(frame)
        candidate_groups = {arm: all_groups[arm] for arm in survivor_arms}
        masks = v11._calendar(frame)
        base_full, _frozen, _raw = conditional._frozen_e15_features()
        base_by_side = {side: [name for name in base_full[side] if name not in {"estimated_spread_bps", "entry_half_spread_bps", "barrier_pct", "entry_price_log"}] for side in SIDES}
        # Frozen base OOS score is recreated before the compatible population is
        # applied, preserving its original base-training contract.
        full_v11, full_raw = v11._read(v11.PANEL, v11.ALIGNMENT, v11.FEATURE_CONTRACT, v11.POSTCOST_EVENTS, v11.PERSISTENCE_LABELS, smoke=False)
        full_selected = v11._select_base_features(full_v11.loc[v11._calendar(full_v11)["base_train"]].reset_index(drop=True), full_raw, seed=seed, trees=60 if smoke else 180)
        base_scores = v11._base_scores(full_v11, full_selected, full_raw, v11._calendar(full_v11), seed=seed + 10, trees=60 if smoke else 180)
        frame = frame.merge(base_scores.loc[:, ["candidate_id", "calibrated_expected_net_bps"]].rename(columns={"calibrated_expected_net_bps": "base_expected_net_bps"}), on="candidate_id", how="inner", validate="one_to_one")
        meta = frame.loc[masks["meta_train"] & frame.base_expected_net_bps.notna()].copy()
        evaluate = frame.loc[masks["eval"] & frame.base_expected_net_bps.notna()].copy()
        # Prequential history is generated prior to each month, so causal maps
        # never use unresolved/future targets.
        history_parts: list[pd.DataFrame] = []
        for month in sorted(meta.decision_ts.dt.strftime("%Y-%m").unique()):
            test = meta.loc[meta.decision_ts.dt.strftime("%Y-%m").eq(month)].copy()
            train = meta.loc[meta.decision_ts.dt.strftime("%Y-%m").lt(month) & meta.label_available_ts.lt(pd.Timestamp(f"{month}-01", tz="UTC"))].copy()
            if len(train) < 1000 or len(test) == 0:
                continue
            scored, _ = _score_period(train, test, base_by_side, candidate_groups, seed=seed + int(month[-2:]), trees=60 if smoke else 180)
            history_parts.append(scored)
        if not history_parts:
            raise ValueError("insufficient prequential Stage-C history for frozen causal map")
        raw_eval, selected_manifest = _score_period(meta, evaluate, base_by_side, candidate_groups, seed=seed + 500, trees=60 if smoke else 180)
        outputs = []
        for arm, side_bridge in (("H_control", False), ("H_new", False), ("H_new_bridge", True)):
            current = raw_eval.loc[raw_eval.arm.eq("H_new" if arm == "H_new_bridge" else arm)].drop(columns="arm").copy()
            history = pd.concat(history_parts, ignore_index=True).loc[lambda d: d.arm.eq("H_new" if arm == "H_new_bridge" else arm)].drop(columns="arm")
            mapped = v11._causal_map(history, current, side_specific=side_bridge)
            mapped["arm"] = arm
            outputs.append(v11._causal_threshold(mapped))
        results = pd.concat(outputs, ignore_index=True)
        books = [v11._book_records(results.loc[results.arm.eq(arm)], arm) for arm in ("H_control", "H_new", "H_new_bridge")]
        flat_books = [item for group in books for item in group]
        bootstrap = v11._paired_day_bootstrap(results, control_arm="H_control", seed=seed + 88, replicates=50 if smoke else v11.BOOTSTRAP_REPLICATES)
        results.to_parquet(stage / "stage_b_incremental_retention_results.parquet", index=False, compression="zstd")
        pd.DataFrame(flat_books).to_parquet(stage / "stage_b_incremental_retention_metrics.parquet", index=False, compression="zstd")
        bootstrap.to_parquet(stage / "stage_b_incremental_retention_bootstrap.parquet", index=False, compression="zstd")
        (stage / "retention_compact_feature_manifest.json").write_text(json.dumps({"survivor_arms": survivor_arms, "selected_by_side": selected_manifest, "decision": decision}, indent=2, default=str) + "\n", encoding="utf-8")
        summary = ["# Frozen Stage-B incremental retention test", "", "Only `P(retain | clear)` differs between H_control and H_new. H_new_bridge changes only the existing side bridge after raw hierarchy scoring.", "", "```csv", pd.DataFrame(flat_books).query("scope == 'pooled_global_top' and fraction == 0.10").to_csv(index=False).rstrip(), "```"]
        (stage / "stage_b_incremental_retention_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        (stage / "feature_disposition.yaml").write_text("terminal_decision: RETENTION_FEATURES_DIAGNOSTIC_ONLY\nreason: Stage-B economics require manual predeclared-gate review; no promotion is performed by this runner.\n", encoding="utf-8")
        manifest = {"status": "COMPLETED_RESEARCH_ONLY_NO_PROMOTION", "survivor_arms": survivor_arms, "frozen": {"clear_head": "E15", "adverse_head": "E15", "cost_policy": "v11"}, "selection": decision}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-panel", type=Path, default=FEATURE_PANEL)
    parser.add_argument("--conditional-output", type=Path, default=CONDITIONAL_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(feature_panel=args.feature_panel, conditional_output=args.conditional_output, output=args.output, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
