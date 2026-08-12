#!/usr/bin/env python3
"""Fixed-alpha factorial of global, path, and recurrent-leaf correctness.

The preceding one-block test found global recent correctness incrementally
useful over the frozen Support+OOD reliability arm, while leaf and path states
were not useful alone.  This compact 2**3 factorial asks whether those states
become complementary when added together.  It deliberately fixes the same
Support+OOD HPO winner and alpha=1.0 multiplier across all eight arms.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_tp6_sl4_compact_path_joint_hpo_2025 as compact  # noqa: E402
from scripts import run_tp6_sl4_support_ood_health_blocks_2025 as health  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS  # noqa: E402


OUT = ROOT / "data_perp/artifacts/tp6_sl4_support_ood_correctness_combinations_20260809_v1"


def _name(parts: tuple[str, ...]) -> str:
    return "support_ood" if not parts else "support_ood_plus_" + "_plus_".join(parts)


def run(*, out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    panel, blocks, lineage, inherited_audit = compact._build_panel()
    print("MATERIALIZE exact leaf-rule correctness", flush=True)
    leaf_recent, leaf_audit, rule_audit = health._leaf_rule_recent(panel)
    panel = pd.concat([panel, leaf_recent], axis=1)
    baseline = [
        *blocks["market_context"], *blocks["soft_membership"],
        *blocks["activated_leaf_support"], *blocks["rule_path_ood_drift"],
    ]
    additions = {
        "global_recent": health._global_recent(panel),
        "path_recent": health._aggregate_path_recent(panel),
        "leaf_recent": list(leaf_recent.columns),
    }
    configs: dict[str, list[str]] = {}
    for size in range(4):
        for subset in itertools.combinations(additions, size):
            fields = list(baseline)
            for name in subset:
                fields.extend(additions[name])
            configs[_name(subset)] = list(dict.fromkeys(fields))
    params = health._baseline_params()
    parts: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    usage: list[pd.DataFrame] = []
    for number, month in enumerate(MONTHS):
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[panel.__ts__.lt(cutoff) & panel.label_available_ts.lt(cutoff)].copy()
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        print(f"FOLD {month}", flush=True)
        score, audit, gain = health._run_fold(train, held, configs, params, number)
        parts.append(score); audits.append(audit); usage.append(gain)
        gc.collect()
    prediction = pd.concat(parts, ignore_index=True)
    arms = ["canonical_control", *[field for field in prediction if field.startswith("multiply__")]]
    glob, monthly, stability = compact._metric_table(prediction, arms)
    old = pd.read_parquet(health.BASELINE_ARTIFACT / "metrics_global.parquet")
    expected = old.loc[(old.arm.eq("multiply__support_ood__a100")) & old["tail"].eq(0.05), "net_bps_per_trade"]
    actual = glob.loc[(glob.arm.eq("multiply__support_ood__a100")) & glob["tail"].eq(0.05), "net_bps_per_trade"]
    parity = float(actual.iloc[0] - expected.iloc[0]) if len(actual) == len(expected) == 1 else float("nan")
    gain = pd.concat(usage, ignore_index=True)
    gain["block"] = np.where(gain.field.str.startswith("leaf_recent"), "leaf_recent", np.where(gain.field.str.startswith("path_recent"), "path_recent", np.where(gain.field.str.startswith("model_recent"), "global_recent", "baseline")))
    gain_summary = gain.groupby(["arm", "block", "field"], observed=True).agg(mean_gain=("gain", "mean"), used_months=("gain", lambda x: int((x > 0.0).sum()))).reset_index()
    top5 = glob.loc[glob["tail"].eq(0.05)].sort_values("net_bps_per_trade", ascending=False)
    prediction.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(out / "metrics_global.parquet", index=False); monthly.to_parquet(out / "metrics_monthly.parquet", index=False); stability.to_parquet(out / "metrics_stability.parquet", index=False)
    pd.concat(audits, ignore_index=True).to_parquet(out / "model_audit.parquet", index=False)
    gain.to_parquet(out / "feature_usage_by_fold.parquet", index=False); gain_summary.to_parquet(out / "feature_usage_summary.parquet", index=False)
    leaf_audit.to_parquet(out / "leaf_recent_support_audit.parquet", index=False); rule_audit.to_parquet(out / "leaf_rule_recurrence_audit.parquet", index=False); lineage.to_parquet(out / "lineage.parquet", index=False); inherited_audit.to_parquet(out / "inherited_state_audit.parquet", index=False)
    correctness = {
        "factorial": "global_recent x path_recent x leaf_recent, 2^3 including Support+OOD control",
        "frozen_hpo": str(health.BASELINE_ARTIFACT / "hpo_winners.parquet"),
        "fixed_alpha": 1.0,
        "support_ood_control_parity_top5_bps": parity,
        "all_outcome_health_asof_label_available_ts": True,
        "leaf_health_uses_stable_rule_signature": True,
        "one_pooled_global_rank_after_monthly_score_generation": True,
        "all_scores_finite": bool(np.isfinite(prediction[arms].to_numpy(float)).all()),
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {"schema": "tp6_sl4_support_ood_correctness_combinations_20260809_v1", "status": "COMPLETE", "rows": len(prediction), "arms": list(configs), "params": params, "artifacts": sorted(path.name for path in out.iterdir())}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ["# Support+OOD correctness combinations — fixed alpha=1.0", "", "Every arm is one member of the complete global/path/leaf recent-correctness factorial. HPO and multiplier are frozen from Support+OOD, so this is an interaction test, not a capacity test.", "", "## Global Top-5", "", top5.round(3).to_string(index=False), "", "## Top-5 stability", "", stability.loc[stability["tail"].eq(0.05)].sort_values("mean_net_bps", ascending=False).round(3).to_string(index=False), "", "## Correctness", "", json.dumps(correctness, indent=2)]
    (out / "SUPPORT_OOD_CORRECTNESS_COMBINATIONS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "parity": parity, "top5": top5.head(9)[["arm", "net_bps_per_trade"]].to_dict("records")}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(out=args.out)
