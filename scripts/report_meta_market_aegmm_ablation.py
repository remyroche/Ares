#!/usr/bin/env python3
"""Report exact-row OOS deltas for the market-state meta AE/GMM ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name"]
TOP_K = (0.10, 0.20, 0.30)
SCORE = "score_base_ev_residual_expert_hier_mapped"


def _metrics(frame: pd.DataFrame, score_column: str, fraction: float) -> dict[str, float]:
    work = frame.loc[np.isfinite(frame[score_column])].copy()
    count = int(np.ceil(len(work) * fraction))
    selected = work.nlargest(count, score_column, keep="all")
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
    return {
        "rows": int(len(work)),
        "selected_rows": int(len(selected)),
        "top_fraction": float(fraction),
        "mean_ev_after_1pct": float(ev.mean()),
        "sum_ev_after_1pct": float(ev.sum()),
        "positive_ev_rate": float((ev > 0.0).mean()),
        "clean_exec_precision": float(pd.to_numeric(selected["clean_exec"], errors="coerce").mean()),
        "dirty_positive_rate": float(pd.to_numeric(selected["dirty_positive"], errors="coerce").mean()),
        "full_path_bad_mae_rate": float(pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()),
        "timeout_rate": float(pd.to_numeric(selected["timeout"], errors="coerce").mean()),
        "score_ic_spearman": float(selected[[score_column, "ev_after_1pct"]].corr(method="spearman").iloc[0, 1]),
    }


def _summaries(joined: pd.DataFrame, group_columns: list[str], scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [((), joined)] if not group_columns else joined.groupby(group_columns, observed=True, dropna=False)
    for group_key, group in groups:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        labels = dict(zip(group_columns, group_key))
        for fraction in TOP_K:
            baseline = _metrics(group, "baseline_score", fraction)
            challenger = _metrics(group, "challenger_score", fraction)
            row: dict[str, object] = {"scope": scope, **labels, "top_fraction": fraction}
            for key, value in baseline.items():
                if key != "top_fraction":
                    row[f"baseline_{key}"] = value
            for key, value in challenger.items():
                if key != "top_fraction":
                    row[f"challenger_{key}"] = value
            for metric in (
                "mean_ev_after_1pct",
                "sum_ev_after_1pct",
                "positive_ev_rate",
                "clean_exec_precision",
                "dirty_positive_rate",
                "full_path_bad_mae_rate",
                "timeout_rate",
                "score_ic_spearman",
            ):
                row[f"delta_{metric}"] = float(challenger[metric] - baseline[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def _state_profile(state_path: Path, inputs_path: Path, oos: pd.DataFrame) -> pd.DataFrame:
    state = pd.read_parquet(state_path)
    inputs = pd.read_parquet(inputs_path)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    inputs["__ts__"] = pd.to_datetime(inputs["__ts__"], utc=True)
    oos_timestamps = pd.DataFrame({"__ts__": pd.to_datetime(oos["__ts__"], utc=True).drop_duplicates()})
    frame = state.merge(inputs, on="__ts__", how="left", validate="one_to_one")
    frame = frame.merge(oos_timestamps.assign(oos_seen=1), on="__ts__", how="left")
    cluster = "meta_market_aegmm_gmm_cluster_id"
    semantic = [
        "market_input__mkt_systemic_deleveraging_score",
        "market_input__mkt_flush_exhaustion_score",
        "market_input__mkt_leverage_rebuild_score",
        "market_input__market_breadth_1h",
        "market_input__market_dispersion_4h",
        "market_input__market_pc1_variance_share_12h",
        "market_input__mkt_ret_4h",
        "market_input__mkt_median_oi_chg_4h_rz",
        "market_input__mkt_funding_mean_z_30d",
        "market_base_completed_hit_n10",
        "market_base_completed_ev_n10",
        "market_base_completed_ic_n10",
        "market_base_gmm_ood_score_mean",
    ]
    semantic = [column for column in semantic if column in frame]
    means = frame[semantic].mean(numeric_only=True)
    stds = frame[semantic].std(numeric_only=True).replace(0.0, np.nan)
    profile = frame.groupby(cluster, observed=True).agg(
        timestamps=("__ts__", "size"),
        oos_timestamps=("oos_seen", "sum"),
        posterior_max=("meta_market_aegmm_gmm_posterior_max", "mean"),
        entropy=("meta_market_aegmm_gmm_entropy", "mean"),
        ood=("meta_market_aegmm_gmm_ood_score", "mean"),
        unknown_probability=("meta_market_aegmm_gmm_unknown_probability", "mean"),
        reconstruction_error=("meta_market_aegmm_AE_reconstruction_error", "mean"),
    )
    raw_means = frame.groupby(cluster, observed=True)[semantic].mean()
    z = raw_means.sub(means, axis="columns").div(stds, axis="columns")
    z = z.add_prefix("z_")
    return profile.join(z).reset_index().sort_values(cluster)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    columns = [*KEYS, "calendar_month", "week_start", "archetype_policy_key", "ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r", "timeout", SCORE]
    baseline = pd.read_parquet(args.baseline, columns=columns).rename(columns={SCORE: "baseline_score"})
    challenger = pd.read_parquet(args.challenger, columns=columns).rename(columns={SCORE: "challenger_score"})
    shared = [column for column in columns if column not in {SCORE, "calendar_month", "week_start", "archetype_policy_key"}]
    joined = baseline.merge(
        challenger[[*KEYS, "challenger_score"]], on=KEYS, how="inner", validate="one_to_one"
    )
    if len(joined) != len(baseline) or len(joined) != len(challenger):
        raise RuntimeError(f"Exact-row comparison failed: baseline={len(baseline)} challenger={len(challenger)} overlap={len(joined)}")
    joined["__ts__"] = pd.to_datetime(joined["__ts__"], utc=True)
    joined["calendar_month"] = joined["__ts__"].dt.strftime("%Y-%m")
    joined["week_start"] = (joined["__ts__"] - pd.to_timedelta(joined["__ts__"].dt.dayofweek, unit="D")).dt.floor("D")
    tables = [
        _summaries(joined, [], "overall"),
        _summaries(joined, ["calendar_month"], "month"),
        _summaries(joined, ["side_name"], "side"),
        _summaries(joined, ["archetype_policy_key"], "archetype"),
        _summaries(joined, ["calendar_month", "side_name", "archetype_policy_key"], "month_side_archetype"),
    ]
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(args.out_dir / "market_aegmm_exact_row_ablation_metrics.csv", index=False)
    profile = _state_profile(args.state, args.inputs, joined)
    profile.to_csv(args.out_dir / "market_aegmm_state_semantic_profile.csv", index=False)
    report = {
        "comparison_contract": "same OOS rows; same base candidate stream; frozen prior long/short HPO parameters; challenger reruns staged MDA once with market state eligible",
        "rows": int(len(joined)),
        "score": SCORE,
        "top_fractions": list(TOP_K),
        "baseline": str(args.baseline),
        "challenger": str(args.challenger),
    }
    (args.out_dir / "market_aegmm_exact_row_ablation_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(metrics.loc[metrics["scope"].eq("overall")].to_string(index=False))


if __name__ == "__main__":
    main()
