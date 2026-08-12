#!/usr/bin/env python3
"""One-block recent-health additions to the matched Support+OOD reliability arm.

This is intentionally a *pure feature* ablation.  The Support+OOD LGBM
parameters, training rows, score transform, seed schedule, base+consensus
control, and alpha=1.0 multiplier are all fixed from the preceding compact
path experiment.  Each challenger adds exactly one causal state block:

* covariance/correlation break;
* global recent base correctness;
* cross-output/model-state correctness;
* recurrent prototype-path correctness;
* K=9 soft-archetype correctness; or
* exact recurrent leaf-rule correctness.

All health states contain only outcomes whose ``label_available_ts`` is
strictly earlier than a row's decision timestamp.  Exact leaf health groups by
the frozen, threshold-banded ``rule_signature``—never a month-local leaf token.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_tp6_sl4_compact_path_joint_hpo_2025 as compact  # noqa: E402
from scripts import run_tp6_sl4_rule_state_reliability_ablation_2025 as state  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS  # noqa: E402


SEED = compact.SEED
TAILS = compact.TAILS
BASELINE_ARTIFACT = compact.OUT
OUT = ROOT / "data_perp/artifacts/tp6_sl4_support_ood_health_blocks_20260809_v1"
MIN_SUPPORT = state.MIN_PATH_EFFECTIVE_SUPPORT


def _metrics(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    expected = pd.to_numeric(panel["frozen_base_expected_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    net = pd.to_numeric(panel["net_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    residual = net - expected
    win = net > 0.0
    return {
        "directional_correct": ((panel.base_score.to_numpy(float) > 0.0) == win).astype(float),
        "approximately_correct": (np.abs(residual) <= 50.0).astype(float),
        "adverse_residual_rate": (residual <= -50.0).astype(float),
        "strong_adverse_residual_rate": (residual <= -100.0).astype(float),
    }


def _weighted_entity_health(
    panel: pd.DataFrame,
    weights: np.ndarray,
    entity_names: Sequence[str],
    *,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate, active entity-health fields and a support audit.

    ``weights`` are observable at decision time.  The outcome rollup is made
    by ``label_available_ts`` then mapped strictly before decision.  Fields are
    multiplied by their live membership/exposure and set to zero below support.
    """
    metrics = _metrics(panel)
    fields: dict[str, np.ndarray] = {}
    audits: list[dict[str, object]] = []
    for idx, entity in enumerate(entity_names):
        rolling = state._rolling_rates(panel.label_available_ts, weights[:, idx], metrics, prefix="")
        mapped = state._asof_features(panel, rolling)
        for days in (3, 7, 14):
            support = mapped[f"support_{days}d"].to_numpy(float)
            active = weights[:, idx] * (support >= MIN_SUPPORT)
            for metric in metrics:
                name = f"{prefix}__{entity}__{metric}__{days}d"
                fields[name] = active * mapped[f"{metric}_{days}d"].to_numpy(float)
            audits.append({
                "entity": entity, "window": f"{days}d", "median_prior_support": float(np.median(support)),
                "active_adequate_fraction": float(np.mean(active > 0.0)),
            })
    return pd.DataFrame(fields, index=panel.index).fillna(0.0).astype("float32"), pd.DataFrame(audits)


def _aggregate_path_recent(panel: pd.DataFrame) -> list[str]:
    """Already materialised per-prototype correctness, contribution-weighted."""
    return [field for field in panel if field.startswith("path_recent_")]


def _global_recent(panel: pd.DataFrame) -> list[str]:
    return [field for field in panel if field.startswith("model_recent_") and "cross_" not in field]


def _cross_model_recent(panel: pd.DataFrame) -> list[str]:
    return [field for field in panel if field.startswith("model_recent_cross_")]


def _rule_assignments(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each active leaf to a stable threshold-banded rule signature."""
    pieces: list[pd.DataFrame] = []
    catalog_audit: list[dict[str, object]] = []
    wanted = set(panel.month.astype(str))
    for folder in sorted(state.RAW.glob("month=*")):
        month = folder.name.split("=", 1)[1]
        if month not in wanted:
            continue
        leaves = pd.read_parquet(folder / "leaf_assignments.parquet")
        leaves = leaves.loc[leaves.side_name.eq("long")].copy()
        tree_fields = [field for field in leaves if field.startswith("leaf_assignment__")]
        long = pd.DataFrame({
            "candidate_id": np.repeat(leaves.candidate_id.to_numpy(), len(tree_fields)),
            "head_tree_slot": np.tile(np.asarray([int(field.rsplit("_", 1)[-1]) for field in tree_fields], dtype=np.int16), len(leaves)),
            "leaf_token": leaves.loc[:, tree_fields].to_numpy().reshape(-1),
        })
        long["leaf_token"] = pd.to_numeric(long.leaf_token, errors="coerce").fillna(0).astype("uint64")
        catalog = pd.read_parquet(folder / "leaf_rule_catalog.parquet")
        catalog = catalog.loc[catalog.side_name.eq("long") & catalog.head_name.eq("canonical_residual"), ["head_tree_slot", "leaf_token", "rule_signature", "ensemble_tree_contribution"]].copy()
        catalog["head_tree_slot"] = pd.to_numeric(catalog.head_tree_slot, errors="coerce").astype("int16")
        catalog["leaf_token"] = pd.to_numeric(catalog.leaf_token, errors="coerce").fillna(0).astype("uint64")
        catalog["weight"] = pd.to_numeric(catalog.ensemble_tree_contribution, errors="coerce").abs().fillna(0.0)
        catalog = catalog.drop(columns="ensemble_tree_contribution").drop_duplicates(["head_tree_slot", "leaf_token"])
        mapped = long.merge(catalog, on=["head_tree_slot", "leaf_token"], how="left", validate="many_to_one")
        mapped["month"] = month
        pieces.append(mapped.loc[mapped.rule_signature.notna(), ["candidate_id", "rule_signature", "weight", "month"]])
        catalog_audit.append({"month": month, "candidate_leaf_rows": len(long), "mapped_rule_rows": int(mapped.rule_signature.notna().sum()), "tree_slots": len(tree_fields)})
    output = pd.concat(pieces, ignore_index=True)
    if output.empty:
        raise RuntimeError("no active leaf rules mapped")
    return output, pd.DataFrame(catalog_audit)


def _leaf_rule_recent(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Contribution-weighted correctness of active, recurrent exact leaf rules.

    This uses rule signatures with at least 30 historical activations over the
    panel.  The support requirement is applied once more to each 3/7/14-day
    causal window, so a rule may be recurrent globally but inactive until it
    accumulates enough previously resolved rows.
    """
    assignments, catalog_audit = _rule_assignments(panel)
    base = panel[["candidate_id", "__ts__", "label_available_ts"]].copy()
    for key, value in _metrics(panel).items():
        base[key] = value.astype("float32")
    long = assignments.merge(base, on="candidate_id", how="inner", validate="many_to_one")
    total = long.groupby("rule_signature", observed=True).size()
    recurrent = set(total.index[total.ge(MIN_SUPPORT)])
    long = long.loc[long.rule_signature.isin(recurrent)].copy()
    if long.empty:
        raise RuntimeError("no leaf-rule signatures meet recurrent support")
    metric_names = list(_metrics(panel))
    event = long[["rule_signature", "label_available_ts", "weight"]].copy()
    event = event.rename(columns={"label_available_ts": "_available_ts", "weight": "_w"})
    for metric in metric_names:
        event[metric] = event["_w"] * long[metric].to_numpy(float)
    bucket = event.groupby(["rule_signature", "_available_ts"], observed=True, sort=True).sum().reset_index()

    def roll(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("_available_ts", kind="stable").set_index("_available_ts")
        prior = group[["_w", *metric_names]].shift(1).fillna(0.0)
        out: dict[str, pd.Series] = {}
        for days in (3, 7, 14):
            value = prior.rolling(f"{days}D", min_periods=1).sum()
            out[f"support_{days}d"] = value["_w"]
            for metric in metric_names:
                out[f"{metric}_{days}d"] = value[metric] / value["_w"].replace(0.0, np.nan)
        result = pd.DataFrame(out, index=group.index).reset_index()
        result["rule_signature"] = str(group["rule_signature"].iloc[0])
        return result

    state_rows = pd.concat([roll(group) for _, group in bucket.groupby("rule_signature", observed=True, sort=False)], ignore_index=True)
    left = long[["candidate_id", "rule_signature", "__ts__", "weight"]].copy()
    left["_row"] = np.arange(len(left), dtype=np.int64)
    merged = pd.merge_asof(
        left.sort_values(["__ts__", "rule_signature"], kind="stable"),
        state_rows.sort_values(["_available_ts", "rule_signature"], kind="stable"),
        left_on="__ts__", right_on="_available_ts", by="rule_signature", direction="backward", allow_exact_matches=False,
    ).sort_values("_row", kind="stable")
    candidate_weight = merged.groupby("candidate_id", observed=True)["weight"].sum().replace(0.0, np.nan)
    values: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, object]] = []
    for days in (3, 7, 14):
        support = pd.to_numeric(merged[f"support_{days}d"], errors="coerce").fillna(0.0)
        active = merged.weight.to_numpy(float) * (support.to_numpy(float) >= MIN_SUPPORT)
        denom = pd.Series(active, index=merged.candidate_id).groupby(level=0, observed=True).sum().replace(0.0, np.nan)
        values[f"leaf_recent_support_{days}d"] = (pd.Series(merged.weight.to_numpy(float) * support.to_numpy(float), index=merged.candidate_id).groupby(level=0, observed=True).sum() / candidate_weight)
        values[f"leaf_recent_adequate_mass_{days}d"] = denom / candidate_weight
        for metric in metric_names:
            rate = pd.to_numeric(merged[f"{metric}_{days}d"], errors="coerce").fillna(0.0).to_numpy(float)
            values[f"leaf_recent_{metric}_{days}d"] = pd.Series(active * rate, index=merged.candidate_id).groupby(level=0, observed=True).sum() / denom
        audit_rows.append({"window": f"{days}d", "recurrent_rules": len(recurrent), "active_leaf_rows": int((active > 0.0).sum()), "adequate_weight_fraction": float(active.sum() / max(merged.weight.sum(), 1e-12))})
    output = pd.DataFrame({name: value.reindex(panel.candidate_id).fillna(0.0).to_numpy(float) for name, value in values.items()}, index=panel.index).astype("float32")
    rule_audit = pd.DataFrame({"rule_signature": list(total.index), "panel_activations": total.to_numpy(), "recurrent": total.index.isin(recurrent)})
    del long, assignments, event, bucket, state_rows, merged
    gc.collect()
    return output, pd.concat([catalog_audit.assign(audit="catalog"), pd.DataFrame(audit_rows).assign(audit="health")], ignore_index=True, sort=False), rule_audit


def _baseline_params() -> dict[str, object]:
    winner = pd.read_parquet(BASELINE_ARTIFACT / "hpo_winners.parquet")
    values = winner.loc[winner.arm.eq("support_ood"), "params_json"]
    if len(values) != 1:
        raise RuntimeError("missing frozen Support+OOD HPO winner")
    return json.loads(str(values.iloc[0]))


def _run_fold(train: pd.DataFrame, held: pd.DataFrame, configs: dict[str, list[str]], params: dict[str, object], month_no: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr_anchor, te_anchor = compact._map_base(train, held)
    train = train.copy(); held = held.copy()
    train["base_anchor"] = tr_anchor; held["base_anchor"] = te_anchor
    target = (train.net_bps.to_numpy(float) - tr_anchor > 0.0).astype(np.int8)
    out = held[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_plus_consensus25"]].copy()
    out["canonical_control"] = held.base_plus_consensus25.to_numpy(float)
    usage: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for arm, fields in configs.items():
        fields = [field for field in dict.fromkeys(fields) if field in train.columns]
        probability, info, model, _ = compact._fit_probability(
            train, held, fields, target, params,
            seed=SEED + month_no * 10000 + (zlib.adler32(arm.encode()) % 100000), return_model=True,
        )
        score = out.canonical_control.to_numpy(float) * np.clip(1.0 + (probability - 0.5), 0.5, 1.5)
        out[f"multiply__{arm}__a100"] = pd.Series(score).rank(pct=True, method="average").to_numpy("float32")
        audit.append({"month": str(held.month.iloc[0]), "arm": arm, "features": len(fields), **info})
        if model is not None:
            usage.append(pd.DataFrame({"month": str(held.month.iloc[0]), "arm": arm, "field": fields, "gain": model.booster_.feature_importance(importance_type="gain")}))
            del model
            gc.collect()
    return out, pd.DataFrame(audit), pd.concat(usage, ignore_index=True)


def run(*, out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    panel, blocks, lineage, state_audit = compact._build_panel()
    print("MATERIALIZE archetype correctness", flush=True)
    membership = [field for field in panel if field.startswith("k09__cluster__") and field.endswith("__membership")]
    archetype, archetype_audit = _weighted_entity_health(panel, panel.loc[:, membership].to_numpy(float), membership, prefix="archetype_recent")
    print("MATERIALIZE exact leaf-rule correctness", flush=True)
    leaf_recent, leaf_audit, rule_audit = _leaf_rule_recent(panel)
    panel = pd.concat([panel, archetype, leaf_recent], axis=1)
    core = blocks["market_context"]
    baseline = [*core, *blocks["soft_membership"], *blocks["activated_leaf_support"], *blocks["rule_path_ood_drift"]]
    configs = {
        "support_ood": baseline,
        "support_ood_plus_covariance": [*baseline, *blocks["covariance_correlation_break"]],
        "support_ood_plus_global_recent": [*baseline, *_global_recent(panel)],
        "support_ood_plus_cross_model_recent": [*baseline, *_cross_model_recent(panel)],
        "support_ood_plus_path_recent": [*baseline, *_aggregate_path_recent(panel)],
        "support_ood_plus_archetype_recent": [*baseline, *archetype.columns],
        "support_ood_plus_leaf_recent": [*baseline, *leaf_recent.columns],
    }
    configs = {name: list(dict.fromkeys(fields)) for name, fields in configs.items()}
    params = _baseline_params()
    parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    usage_parts: list[pd.DataFrame] = []
    for month_no, month in enumerate(MONTHS):
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[panel.__ts__.lt(cutoff) & panel.label_available_ts.lt(cutoff)].copy()
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        print(f"FOLD {month}", flush=True)
        score, audit, usage = _run_fold(train, held, configs, params, month_no)
        parts.append(score); audit_parts.append(audit); usage_parts.append(usage)
    prediction = pd.concat(parts, ignore_index=True)
    arms = ["canonical_control", *[field for field in prediction if field.startswith("multiply__")]]
    glob, monthly, stability = compact._metric_table(prediction, arms)
    baseline_old = pd.read_parquet(BASELINE_ARTIFACT / "metrics_global.parquet")
    old = baseline_old.loc[(baseline_old.arm.eq("multiply__support_ood__a100")) & baseline_old["tail"].eq(0.05), "net_bps_per_trade"]
    current = glob.loc[(glob.arm.eq("multiply__support_ood__a100")) & glob["tail"].eq(0.05), "net_bps_per_trade"]
    parity = float(current.iloc[0] - old.iloc[0]) if len(current) == len(old) == 1 else float("nan")
    usage = pd.concat(usage_parts, ignore_index=True)
    usage["block"] = np.where(usage.field.str.startswith("archetype_recent"), "archetype_recent", np.where(usage.field.str.startswith("leaf_recent"), "leaf_recent", np.where(usage.field.str.startswith("path_recent"), "path_recent", np.where(usage.field.str.startswith("model_recent"), "model_recent", np.where(usage.field.isin(blocks["covariance_correlation_break"]), "covariance", "baseline")))))
    summary = glob.loc[glob["tail"].eq(0.05)].sort_values("net_bps_per_trade", ascending=False)
    prediction.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(out / "metrics_global.parquet", index=False); monthly.to_parquet(out / "metrics_monthly.parquet", index=False); stability.to_parquet(out / "metrics_stability.parquet", index=False)
    pd.concat(audit_parts, ignore_index=True).to_parquet(out / "model_audit.parquet", index=False)
    usage.to_parquet(out / "feature_usage_by_fold.parquet", index=False)
    usage.groupby(["arm", "block", "field"], observed=True).agg(mean_gain=("gain", "mean"), used_months=("gain", lambda v: int((v > 0).sum()))).reset_index().to_parquet(out / "feature_usage_summary.parquet", index=False)
    archetype_audit.to_parquet(out / "archetype_recent_support_audit.parquet", index=False); leaf_audit.to_parquet(out / "leaf_recent_support_audit.parquet", index=False); rule_audit.to_parquet(out / "leaf_rule_recurrence_audit.parquet", index=False)
    lineage.to_parquet(out / "lineage.parquet", index=False); state_audit.to_parquet(out / "inherited_state_audit.parquet", index=False)
    correctness = {
        "baseline_hpo_frozen_from": str(BASELINE_ARTIFACT / "hpo_winners.parquet"),
        "baseline_alpha": 1.0,
        "baseline_parity_top5_bps": parity,
        "one_added_block_per_challenger": True,
        "outcome_states_asof_label_available_ts": True,
        "archetype_health_uses_frozen_k9_soft_memberships": True,
        "leaf_health_uses_recurrent_rule_signature_not_leaf_token": True,
        "leaf_health_support_gated": True,
        "all_scores_finite": bool(np.isfinite(prediction[arms].to_numpy(float)).all()),
        "scope": "matched long-only 2025 TP6/SL4/H12 development replay; no HPO or modulation selection is performed here",
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {"schema": "tp6_sl4_support_ood_health_blocks_20260809_v1", "status": "COMPLETE", "rows": len(prediction), "months": list(MONTHS), "arms": list(configs), "params": params, "artifacts": sorted(path.name for path in out.iterdir())}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ["# Support+OOD health blocks — fixed alpha=1.0", "", "Each challenger adds exactly one causal block to the frozen Support+OOD reliability model.  No per-arm HPO or transform selection occurs in this experiment.", "", "## Global Top-5", "", summary.round(3).to_string(index=False), "", "## Top-5 stability", "", stability.loc[stability["tail"].eq(0.05)].sort_values("mean_net_bps", ascending=False).round(3).to_string(index=False), "", "## Correctness", "", json.dumps(correctness, indent=2)]
    (out / "SUPPORT_OOD_HEALTH_BLOCKS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "top5": summary.head(7)[["arm", "net_bps_per_trade"]].to_dict("records"), "parity": parity}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(out=args.out)
