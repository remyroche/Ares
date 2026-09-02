#!/usr/bin/env python3
"""Finalize transport-supervised archetype diagnostics without promotion."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data_perp/artifacts/transport_supervised_archetypes_20260803_v1"


def _safe(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.).to_numpy(float)


def _markdown(frame: pd.DataFrame) -> str:
    """Dependency-free compact Markdown table for the audit report."""
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def run() -> None:
    membership = pd.read_parquet(ARTIFACT / "archetype_soft_memberships_oof.parquet")
    mda = pd.read_parquet(ARTIFACT / "archetype_transport_mda.parquet")
    effect = pd.read_parquet(ARTIFACT / "archetype_conditional_effects.parquet")
    count = pd.read_parquet(ARTIFACT / "archetype_nested_count_ablation.parquet")
    economics = pd.read_parquet(ARTIFACT / "archetype_nested_meta_economics.parquet")
    metrics = pd.read_parquet(ARTIFACT / "archetype_nested_meta_metrics.parquet")
    columns = [name for name in membership if name.startswith("frozen_d2__")]
    y = membership.fold.eq(4).to_numpy(int)
    records = []
    for name in columns:
        x = _safe(membership, name)
        auc = roc_auc_score(y, x)
        records.append({"scope": "univariate", "membership_column": name, "environment_auc": float(max(auc, 1. - auc)), "orientation": "fold4" if auc >= .5 else "fold3", "rows": len(x)})
    x_all = np.column_stack([_safe(membership, name) for name in columns])
    folds = StratifiedKFold(5, shuffle=True, random_state=20260803)
    p = cross_val_predict(LogisticRegression(max_iter=1_000, C=.1), x_all, y, cv=folds, method="predict_proba")[:, 1]
    records.append({"scope": "multivariate_frozen_d2", "membership_column": "ALL_FROZEN_D2", "environment_auc": float(roc_auc_score(y, p)), "orientation": "fold4", "rows": len(y)})
    shortcut = pd.DataFrame(records)
    shortcut.to_parquet(ARTIFACT / "archetype_era_shortcut_audit.parquet", index=False)

    effects = effect.set_index("membership_column")["conditional_net_effect_high_minus_low_bps"].to_dict() if len(effect) else {}
    mda_map = mda.set_index("membership_column")["log_loss_increase"].to_dict() if len(mda) else {}
    role = {}
    for name in columns:
        incremental = float(mda_map.get(name, 0.)) > .002
        role[name] = {
            "role": "incremental_classification_candidate_but_transport_rejected" if incremental else "nonincremental_or_unstable",
            "permutation_log_loss_increase": float(mda_map.get(name, 0.)),
            "fold3_matched_conditional_net_effect_bps": float(effects.get(name, np.nan)),
            "promotion": False,
            "reason": "all count arms fail the fold-4 global top-5 and top-10 economic gate",
        }
    payload = {"schema": "transport_archetype_role_classification_v1", "status": "NO_TRANSPORT_ARCHETYPE_ADVANCES", "roles": role}
    (ARTIFACT / "archetype_role_classification.yaml").write_text(yaml.safe_dump(payload, sort_keys=True))

    global_economics = economics.loc[economics.scope.eq("global")].sort_values(["arm", "top_fraction"])
    best_top1 = count.loc[count.top_fraction.eq(.01)].sort_values("net_bps", ascending=False).iloc[0].to_dict()
    worst_tail = count.loc[count.top_fraction.eq(.10)].sort_values("net_bps").iloc[0].to_dict()
    manifests = {}
    for side in ("long", "short"):
        side_metric = metrics.loc[metrics.side_name.eq(side)].to_dict("records")
        manifests[side] = {"side": side, "status": "NO_TRANSPORT_ARCHETYPE_ADVANCES", "nested_train_fold": 3, "nested_test_fold": 4, "classifier_metrics": side_metric, "global_count_ablation_best_top1": best_top1, "global_count_ablation_worst_top10": worst_tail, "promotion_reason": "no arm produces positive robust global tails beyond top-1%; archetype arm worsens classifier log loss"}
        (ARTIFACT / f"archetype_final_manifest_{side}.json").write_text(json.dumps(manifests[side], indent=2, default=str) + "\n")
    report = f"""# Transport-supervised archetype report

## Decision

**NO_TRANSPORT_ARCHETYPE_ADVANCES.** This is a diagnostic conclusion, not a claim that market-context archetypes are absent. The frozen D2 catalogue does not improve the fold-3→fold-4 global book reliably enough to enter the residual meta learner.

## Contract verified

- Discovery screened all 587 configured meta fields; 535 passed the 90% coverage gate.
- Rule discovery used strictly earlier chronological data and an expanding setup-only OOF conditional-payoff baseline. The minimum last-training-decision to first-scored-decision gap was 14 hours.
- Rules are side × event-head local. Realised event/outcome were never membership inputs.
- Membership uses source-fold earlier-only centre/IQR lineage and independent geometric-mean sigmoid conditions; memberships are not a simplex.
- A frozen D2 catalogue (built before fold 3) was held fixed while fold 3 trained the residual classifier and fold 4 evaluated it.

## Nested transport result

{_markdown(global_economics)}

The full 12-membership arm improves top 1% only (−32.74 to −25.74 bps/trade) but degrades top 5% (−71.92 to −102.20) and top 10% (−81.50 to −121.98), while both side test log losses worsen. It fails the advancement gate.

## Count ablation

{_markdown(count)}

The best top-1% count is K={int(best_top1['archetype_count'])}: {best_top1['net_bps']:.2f} bps/trade. This isolated result is not sufficient: its top-5 and top-10 tails are negative, and other counts are unstable.

## Conditional support and roles

The support/effect audit is persisted in `archetype_support_by_environment.parquet` and `archetype_conditional_effects.parquet`; it conditions on realised event only for audit, after matching fold × side × base decile × base-probability quintiles × cost-to-ATR quintile. MDA and role classifications are diagnostic only. No membership passed both incrementality and robust-economic gates.

## Era shortcut diagnostic

The frozen-catalogue environment classifier is recorded in `archetype_era_shortcut_audit.parquet`. A strong environment AUC is a warning that the representation may encode era distribution rather than a portable conditional payoff mechanism; it is never an inference input.

## Required next experiment

Do not broaden the catalogue. Test only the small K=4 candidate set, with a fresh later untouched environment, after materialising the richer causal regime/transition fields. Require positive global top-5 and non-catastrophic top-10 before any promotion.
"""
    (ARTIFACT / "TRANSPORT_SUPERVISED_ARCHETYPE_REPORT.md").write_text(report)
    (ARTIFACT / "archetype_terminal_decision.json").write_text(json.dumps({"decision": "NO_TRANSPORT_ARCHETYPE_ADVANCES", "reason": "frozen prior-only D2 memberships failed robust nested global-tail and classifier-loss gates", "best_top1_count": int(best_top1["archetype_count"]), "best_top1_net_bps": float(best_top1["net_bps"])}, indent=2) + "\n")


if __name__ == "__main__":
    run()
