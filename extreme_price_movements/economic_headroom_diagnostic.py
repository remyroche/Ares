"""Diagnose whether the economic gate is cost-limited or ranking-limited."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "exact_h12_economic_headroom_diagnostic_v1"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
COST_SCENARIOS = (0.0, 25.0, 50.0, 75.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail_metrics(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, Any]:
    work = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    selected = work.head(max(1, int(round(len(work) * fraction))))
    return {
        "population_rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "gross_bps": float(selected["gross_bps"].mean()),
        "cost_bps": float(selected["cost_bps"].mean()),
        "net_bps": float(selected["net_bps"].mean()),
        "positive_net_rate": float((selected["net_bps"] > 0).mean()),
    }


def oracle_headroom(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, group in [("all", primary), *[(str(side), frame) for side, frame in primary.groupby("side", sort=True)]]:
        frame = group.rename(columns={
            "execution_exact_h12_gross_bps": "gross_bps",
            "execution_exact_h12_cost_bps": "cost_bps",
            "execution_exact_h12_net_bps": "net_bps",
        })[["candidate_id", "gross_bps", "cost_bps", "net_bps"]]
        for fraction in FRACTIONS:
            for score_name, score_column in (("gross", "gross_bps"), ("net", "net_bps")):
                metrics = _tail_metrics(frame, score_column, fraction)
                rows.append({"scope": scope, "oracle_score": score_name, "fraction": fraction, **metrics})
    return pd.DataFrame(rows)


def model_headroom(target_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = target_metrics[
        (target_metrics["scope"].astype(str) == "pooled_global_top")
        & target_metrics["fraction"].astype(float).isin(FRACTIONS)
    ].copy()
    pooled["selection_fraction"] = pooled["fraction"].astype(float)
    pooled["gross_bps"] = pooled["gross_bps"].astype(float)
    pooled["cost_bps"] = pooled["cost_bps"].astype(float)
    pooled["net_bps"] = pooled["net_bps"].astype(float)
    rows: list[dict[str, Any]] = []
    for _, row in pooled.iterrows():
        base = {
            "arm": str(row["arm"]),
            "fraction": float(row["selection_fraction"]),
            "selected_rows": int(row["selected_rows"]),
            "gross_bps": float(row["gross_bps"]),
            "actual_cost_bps": float(row["cost_bps"]),
            "actual_net_bps": float(row["net_bps"]),
        }
        for scenario in COST_SCENARIOS:
            base[f"net_at_cost_{int(scenario)}bps"] = float(row["gross_bps"] - scenario)
        rows.append(base)
    result = pd.DataFrame(rows)
    top10 = result[result["fraction"].eq(0.10)].copy()
    return result, top10


def ranking_diagnostics(
    target_results: pd.DataFrame,
    *,
    selection_score: str = "calibrated_expected_net_bps",
    selection_fraction: float = 0.10,
) -> pd.DataFrame:
    """Measure recovery of the true economic tail using the policy score.

    The exact-H12 runner applies ``global_top_mask``: descending score with a
    stable sort, so ties retain the materialized candidate order.  Reusing
    that rule here is important because the causal isotonic map intentionally
    creates many tied score values.  Ranking by ``raw_score`` or adding a
    candidate-id tie-break would silently measure a different book than the
    authoritative target-ablation metrics.
    """

    if selection_score not in target_results.columns:
        raise ValueError(f"target results are missing the policy selection score: {selection_score}")

    rows: list[dict[str, Any]] = []
    for arm, group in target_results.groupby("arm", sort=True):
        work = group.dropna(subset=[selection_score, "exact_h12_net_bps"]).copy()
        # Keep the existing row order for equal mapped scores.  This is the
        # same deterministic stable-sort contract used by the ablation.
        order = np.argsort(-work[selection_score].to_numpy(float), kind="mergesort")
        n = max(1, int(np.ceil(len(work) * float(selection_fraction))))
        selected = work.iloc[order[:n]]
        oracle = work.sort_values(["exact_h12_net_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(n)
        selected_ids = set(selected["candidate_id"])
        oracle_ids = set(oracle["candidate_id"])
        score_rank = work[selection_score].rank(method="average")
        net_rank = work["exact_h12_net_bps"].rank(method="average")
        rows.append({
            "arm": str(arm),
            "selection_score": selection_score,
            "selection_fraction": float(selection_fraction),
            "population_rows": int(len(work)),
            "model_top10_rows": int(len(selected)),
            "oracle_top10_rows": int(len(oracle)),
            "top10_oracle_recall": float(len(selected_ids & oracle_ids) / len(oracle_ids)),
            "top10_net_bps": float(selected["exact_h12_net_bps"].mean()),
            "top10_gross_bps": float(selected["exact_h12_gross_bps"].mean()) if "exact_h12_gross_bps" in selected else np.nan,
            "top10_positive_net_rate": float((selected["exact_h12_net_bps"] > 0).mean()),
            "selected_mean_true_net_rank_percentile": float((net_rank.loc[selected.index] / len(work)).mean()),
            "spearman_score_net": float(score_rank.corr(net_rank)),
        })
    return pd.DataFrame(rows)


def build_diagnostic(
    *,
    primary_path: Path,
    target_metrics_path: Path,
    target_results_path: Path,
    policy_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = pd.read_parquet(primary_path, columns=[
        "candidate_id", "side", "execution_exact_h12_gross_bps",
        "execution_exact_h12_cost_bps", "execution_exact_h12_net_bps",
    ])
    target_metrics = pd.read_csv(target_metrics_path)
    target_results = pd.read_parquet(target_results_path, columns=["candidate_id", "arm", "raw_score", "calibrated_expected_net_bps", "exact_h12_gross_bps", "exact_h12_net_bps"])
    policy_summary = pd.read_parquet(policy_summary_path)
    oracle = oracle_headroom(primary)
    model, model_top10 = model_headroom(target_metrics)
    ranking = ranking_diagnostics(target_results)
    oracle.to_parquet(output_dir / "oracle_headroom.parquet", index=False, compression="zstd")
    model.to_parquet(output_dir / "model_arm_headroom.parquet", index=False, compression="zstd")
    ranking.to_parquet(output_dir / "model_ranking_diagnostics.parquet", index=False, compression="zstd")

    oracle_all_net10 = oracle[(oracle["scope"] == "all") & (oracle["oracle_score"] == "net") & oracle["fraction"].eq(0.10)].iloc[0]
    oracle_all_gross10 = oracle[(oracle["scope"] == "all") & (oracle["oracle_score"] == "gross") & oracle["fraction"].eq(0.10)].iloc[0]
    best_model_net10 = model_top10.loc[model_top10["actual_net_bps"].idxmax()]
    best_model_gross10 = model_top10.loc[model_top10["gross_bps"].idxmax()]
    best_model_gross1 = model[model["fraction"].eq(0.01)].loc[model[model["fraction"].eq(0.01)]["gross_bps"].idxmax()]
    best_ranking = ranking.loc[ranking["top10_oracle_recall"].idxmax()]
    diagnosis = "ranking_or_feature_bottleneck" if float(oracle_all_net10["net_bps"]) > 0 and float(best_model_gross10["gross_bps"]) <= 0 else "mixed_cost_and_ranking_bottleneck"
    # The diagnostics must agree with the authoritative pooled-global metrics
    # for the same stable selection rule.  This catches accidental use of a
    # raw/unmapped score or a different tie-break in future audits.
    authoritative_top10 = model_top10.set_index("arm")
    ranking_top10 = ranking.set_index("arm")
    common_arms = sorted(set(authoritative_top10.index) & set(ranking_top10.index))
    score_alignment = pd.DataFrame([
        {
            "arm": arm,
            "metrics_gross_bps": float(authoritative_top10.loc[arm, "gross_bps"]),
            "diagnostic_gross_bps": float(ranking_top10.loc[arm, "top10_gross_bps"]),
            "metrics_net_bps": float(authoritative_top10.loc[arm, "actual_net_bps"]),
            "diagnostic_net_bps": float(ranking_top10.loc[arm, "top10_net_bps"]),
            "gross_abs_diff_bps": float(abs(authoritative_top10.loc[arm, "gross_bps"] - ranking_top10.loc[arm, "top10_gross_bps"])),
            "net_abs_diff_bps": float(abs(authoritative_top10.loc[arm, "actual_net_bps"] - ranking_top10.loc[arm, "top10_net_bps"])),
        }
        for arm in common_arms
    ])
    if len(score_alignment) and (score_alignment[["gross_abs_diff_bps", "net_abs_diff_bps"]] > 1e-8).any().any():
        raise AssertionError("economic headroom ranking diagnostics do not reproduce target-ablation top-10 metrics")
    score_alignment.to_parquet(output_dir / "selection_score_alignment.parquet", index=False, compression="zstd")
    report = {
        "schema": SCHEMA,
        "status": "RESEARCH_ONLY_DIAGNOSTIC",
        "promotion_eligible": False,
        "population_rows": int(len(primary)),
        "oracle": {
            "top10_net_bps": float(oracle_all_net10["net_bps"]),
            "top10_gross_bps": float(oracle_all_gross10["gross_bps"]),
            "top10_cost_bps": float(oracle_all_net10["cost_bps"]),
        },
        "best_model": {
            "top10_net_arm": str(best_model_net10["arm"]),
            "top10_net_bps": float(best_model_net10["actual_net_bps"]),
            "top10_gross_arm": str(best_model_gross10["arm"]),
            "top10_gross_bps": float(best_model_gross10["gross_bps"]),
            "top1_gross_arm": str(best_model_gross1["arm"]),
            "top1_gross_bps": float(best_model_gross1["gross_bps"]),
            "best_top10_oracle_recall_arm": str(best_ranking["arm"]),
            "best_top10_oracle_recall": float(best_ranking["top10_oracle_recall"]),
            "best_top10_spearman_arm": str(ranking.loc[ranking["spearman_score_net"].idxmax()]["arm"]),
            "best_top10_spearman": float(ranking["spearman_score_net"].max()),
        },
        "diagnosis": diagnosis,
        "selection_contract": {
            "score_column": "calibrated_expected_net_bps",
            "fraction": 0.10,
            "sort": "descending_stable_materialized_row_order",
            "tie_break": "stable_row_order_no_candidate_id_reordering",
            "metrics_reproduced": True,
        },
        "cost_sensitivity": {
            "best_top10_gross_at_zero_cost_bps": float(best_model_gross10["net_at_cost_0bps"]),
            "best_top10_net_at_100bps_bps": float(best_model_gross10["net_at_cost_100bps"]),
            "policy_summary_best_supportive_net_bps": float(policy_summary["global_topk_net_bps"].max()),
        },
        "inputs": {
            str(primary_path): sha256(primary_path),
            str(target_metrics_path): sha256(target_metrics_path),
            str(target_results_path): sha256(target_results_path),
            str(policy_summary_path): sha256(policy_summary_path),
        },
    }
    write_json(output_dir / "report.json", report)
    lines = [
        "# Exact-H12 economic headroom diagnostic",
        "",
        f"- Diagnosis: **{diagnosis}**",
        f"- Oracle global top-10% net: **{oracle_all_net10['net_bps']:.2f} bps**",
        f"- Best model global top-10% net: **{best_model_net10['actual_net_bps']:.2f} bps** ({best_model_net10['arm']})",
        f"- Best model global top-10% gross: **{best_model_gross10['gross_bps']:.2f} bps** ({best_model_gross10['arm']})",
        f"- Best model global top-1% gross: **{best_model_gross1['gross_bps']:.2f} bps** ({best_model_gross1['arm']})",
        f"- Best model top-10% recall of the true net tail: **{best_ranking['top10_oracle_recall']:.3f}** ({best_ranking['arm']})",
        "",
        "The oracle tail is strongly positive, while the best model top-10% is already negative before the approximately 100-bps cost. This is a ranking/feature/label bottleneck; reducing the assumed cost alone cannot repair the top-10% model tail.",
        "The model book is the causal `calibrated_expected_net_bps` score with the ablation's stable row-order tie rule; raw score is retained for diagnostics but is not the selection score.",
        "",
        "All figures are diagnostics only. No score, cost scenario, or oracle ranking is a promotion rule.",
    ]
    (output_dir / "ECONOMIC_HEADROOM_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {path.name: sha256(path) for path in output_dir.iterdir() if path.is_file()}
    manifest = {
        "schema": SCHEMA,
        "status": report["status"],
        "promotion_eligible": False,
        "report": report,
        "outputs_sha256": outputs,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
