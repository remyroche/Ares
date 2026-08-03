#!/usr/bin/env python3
"""Summarize the canonical raw-feature direct-utility multi-task ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_20260729_v1"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
DEFAULT_RESIDUAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet"
DEFAULT_PATH_LABELS = ROOT / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_summary_20260729_v1"
SCHEMA = "canonical_raw_feature_direct_utility_multitask_summary_v1"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def rank_ic(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 3:
        return float("nan")
    return float(
        pd.Series(actual[valid]).rank(method="average").corr(
            pd.Series(predicted[valid]).rank(method="average")
        )
    )


def binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid].astype(float)
    p = np.clip(predicted[valid].astype(float), 1e-6, 1.0 - 1e-6)
    result = {"rows": int(valid.sum()), "brier": float(brier_score_loss(y, p))}
    if np.unique(y).size > 1:
        result["auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    else:
        result["auc"] = result["average_precision"] = float("nan")
    return result


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    valid = mask & np.isfinite(actual) & np.isfinite(predicted)
    return {
        "rows": int(valid.sum()),
        "mae": float(np.mean(np.abs(actual[valid] - predicted[valid]))),
        "rank_ic": rank_ic(actual[valid], predicted[valid]),
    }


def load_targets(panel_path: Path, label_root: Path, candidate_ids: set[str]) -> pd.DataFrame:
    panel_columns = [
        "candidate_id", "execution_net_ev_12h", "execution_gross_ev_12h",
        "execution_mfe_return_12h", "exit_is_timeout",
    ]
    panel = pq.read_table(panel_path, columns=panel_columns).to_pandas()
    panel = panel.loc[panel["candidate_id"].astype(str).isin(candidate_ids)].copy()
    label_columns = [
        "candidate_id", "__meaningful_mfe_reached_12h__",
        "__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_1_5atr_mfe__", "__mae_until_horizon_if_no_1_5atr__",
        "__bars_to_confirmed_adverse_trough__", "__future_slope_atr_per_hour_12h__",
        "__path_auxiliary_target_valid__", "__time_to_first_meaningful_mfe_target_valid__",
    ]
    labels = []
    for side in ("long", "short"):
        table = pq.read_table(label_root / f"train_global_{side}_3.parquet", columns=label_columns).to_pandas()
        labels.append(table.loc[table["candidate_id"].astype(str).isin(candidate_ids)])
    path = pd.concat(labels, ignore_index=True)
    return panel.merge(path, on="candidate_id", how="left", validate="one_to_one")


def head_metrics_for_group(
    frame: pd.DataFrame, *, split: str, feature_arm: str, task_arm: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    net = frame["execution_net_ev_12h"].to_numpy(float)
    gross = frame["execution_gross_ev_12h"].to_numpy(float)
    mfe = frame["execution_mfe_return_12h"].to_numpy(float)
    base = {"split": split, "feature_arm": feature_arm, "task_arm": task_arm}
    rows.append({
        **base, "head": "direct_net", "kind": "regression",
        **regression_metrics(net, frame["direct_net_score"].to_numpy(float), np.ones(len(frame), dtype=bool)),
    })
    definitions = {
        "opportunity": ("binary", (net > 0.0).astype(float), np.ones(len(frame), dtype=bool)),
        "favorable_magnitude": ("regression", net, net > 0.0),
        "adverse_magnitude": ("regression", -net, net < 0.0),
        "exit_conversion_loss": ("regression", np.maximum(mfe - gross, 0.0), np.ones(len(frame), dtype=bool)),
        "timeout": ("binary", frame["exit_is_timeout"].to_numpy(float), np.ones(len(frame), dtype=bool)),
    }
    path_valid = frame["__path_auxiliary_target_valid__"].fillna(False).astype(bool).to_numpy()
    timing_valid = frame["__time_to_first_meaningful_mfe_target_valid__"].fillna(False).astype(bool).to_numpy()
    hit = frame["__meaningful_mfe_reached_12h__"].fillna(False).astype(bool).to_numpy()
    time = pd.to_numeric(frame["__time_to_first_meaningful_mfe_hours_12h__"], errors="coerce").to_numpy(float)
    definitions.update({
        "path_meaningful_hit": ("binary", hit.astype(float), path_valid),
        "path_peak_mfe": ("regression", pd.to_numeric(frame["__peak_mfe_atr_12h__"], errors="coerce").to_numpy(float), path_valid & hit),
        "path_fast_hit_2h": ("binary", (hit & (time <= 2.0)).astype(float), timing_valid),
        "path_mae_if_hit": ("regression", pd.to_numeric(frame["__mae_before_1_5atr_mfe__"], errors="coerce").to_numpy(float), path_valid & hit),
        "path_mae_if_no_hit": ("regression", pd.to_numeric(frame["__mae_until_horizon_if_no_1_5atr__"], errors="coerce").to_numpy(float), path_valid & ~hit),
        "path_confirmed_adverse_trough": ("regression", pd.to_numeric(frame["__bars_to_confirmed_adverse_trough__"], errors="coerce").to_numpy(float), path_valid),
        "path_future_slope": ("regression", pd.to_numeric(frame["__future_slope_atr_per_hour_12h__"], errors="coerce").to_numpy(float), path_valid),
    })
    for head, (kind, actual, mask) in definitions.items():
        prediction = f"diagnostic__{head}"
        if prediction not in frame or not frame[prediction].notna().any():
            continue
        predicted = frame[prediction].to_numpy(float)
        metrics = (
            binary_metrics(actual[mask], predicted[mask])
            if kind == "binary"
            else regression_metrics(actual, predicted, mask)
        )
        rows.append({**base, "head": head, "kind": kind, **metrics})
    return rows


def stable_tail(
    frame: pd.DataFrame, score: str, fraction: float,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        [score, "candidate_id"], ascending=[False, True], kind="stable",
    )
    return ordered.head(max(1, int(math.ceil(len(frame) * fraction))))


def control_tails(frame: pd.DataFrame, score_columns: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for score_name, column in score_columns.items():
        for fraction in FRACTIONS:
            selected = stable_tail(frame, column, fraction)
            net = selected["execution_net_ev_12h"].to_numpy(float)
            rows.append({
                "score_name": score_name, "fraction": fraction,
                "selected_rows": len(selected),
                "mean_net_bps": float(net.mean() * 10_000.0),
                "positive_precision": float((net > 0.0).mean()),
                "long_share": float(selected["side_name"].eq("long").mean()),
            })
    return pd.DataFrame(rows)


def weekly_metrics(frame: pd.DataFrame, score_columns: Mapping[str, str]) -> pd.DataFrame:
    work = frame.copy()
    timestamp = pd.to_datetime(work["__ts__"], utc=True)
    work["week"] = ((timestamp.dt.day - 1) // 7 + 1).astype(int)
    rows = []
    for score_name, column in score_columns.items():
        global_mask = work.index.isin(stable_tail(work, column, 0.10).index)
        for week, group in work.groupby("week", sort=True):
            global_selected = group.loc[global_mask[group.index]]
            local_selected = stable_tail(group, column, 0.10)
            for mode, selected in (
                ("global_april_top10_attribution", global_selected),
                ("weekly_local_top10_diagnostic", local_selected),
            ):
                rows.append({
                    "score_name": score_name, "week": int(week), "mode": mode,
                    "selected_rows": len(selected),
                    "mean_net_bps": float(selected["execution_net_ev_12h"].mean() * 10_000.0) if len(selected) else np.nan,
                    "long_share": float(selected["side_name"].eq("long").mean()) if len(selected) else np.nan,
                })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    winner = source_manifest["selection"]["winner"]
    march = pd.read_parquet(args.source / "march_selection_predictions.parquet")
    april = pd.read_parquet(args.source / "april_reused_diagnostic_predictions.parquet")
    candidate_ids = set(march["candidate_id"].astype(str)) | set(april["candidate_id"].astype(str))
    targets = load_targets(args.panel, args.path_labels, candidate_ids)
    march = march.merge(targets, on="candidate_id", how="left", validate="many_to_one", suffixes=("", "__target"))
    march = march.loc[pd.to_datetime(march["execution_label_end_utc"], utc=True).lt(pd.Timestamp("2025-04-01T00:00:00Z"))]
    april = april.merge(targets, on="candidate_id", how="left", validate="one_to_one", suffixes=("", "__target"))

    head_rows = []
    for (feature_arm, task_arm), group in march.groupby(["feature_arm", "task_arm"], sort=False):
        head_rows.extend(head_metrics_for_group(group, split="march_selection", feature_arm=feature_arm, task_arm=task_arm))
    head_rows.extend(head_metrics_for_group(
        april, split="april_reused_diagnostic",
        feature_arm=winner["feature_arm"], task_arm=winner["task_arm"],
    ))
    head_table = pd.DataFrame(head_rows)

    residual = pq.read_table(
        args.residual,
        columns=[
            "candidate_id", "__ts__", "residual_is_oof", "base_oof_score",
            "base_expected_ev", "residual_expected_ev",
        ],
    ).to_pandas()
    residual["__ts__"] = pd.to_datetime(residual["__ts__"], utc=True)
    residual = residual.loc[
        residual["residual_is_oof"].astype(bool)
        & residual["__ts__"].dt.strftime("%Y-%m").eq("2025-04")
    ]
    controls = april.merge(
        residual.drop(columns="__ts__"), on="candidate_id", how="inner",
        validate="one_to_one", suffixes=("", "__residual"),
    )
    if len(controls) != len(april):
        raise ValueError("residual control does not cover the exact April population")
    score_columns = {
        "frozen_base_oof_score": "base_oof_score__residual",
        "frozen_base_expected_ev": "base_expected_ev",
        "frozen_residual_expected_ev": "residual_expected_ev",
        "joint_winner_raw_direct": "direct_net_score",
        "joint_winner_causal_recent": "causal_recent_side_isotonic_ev",
    }
    control_table = control_tails(controls, score_columns)
    weekly_table = weekly_metrics(controls, score_columns)

    ranking = pd.read_parquet(args.source / "march_candidate_ranking.parquet")
    direct_score = float(ranking.loc[ranking["task_arm"].eq("direct_only"), "selection_score"].iloc[0])
    core_score = float(ranking.loc[
        ranking["feature_arm"].eq("base") & ranking["task_arm"].eq("economic_multitask"),
        "selection_score",
    ].iloc[0])
    ranking["delta_vs_direct_only_bps"] = ranking["selection_score"] - direct_score
    ranking["delta_vs_base_economic_multitask_bps"] = ranking["selection_score"] - core_score

    top10 = control_table.loc[control_table["fraction"].eq(0.10)].set_index("score_name")
    latest = weekly_table.loc[
        weekly_table["mode"].eq("global_april_top10_attribution")
        & weekly_table["week"].eq(weekly_table["week"].max())
    ].set_index("score_name")
    winner_top10 = float(top10.loc["joint_winner_causal_recent", "mean_net_bps"])
    residual_top10 = float(top10.loc["frozen_residual_expected_ev", "mean_net_bps"])
    promotion = {
        "april_mapped_top10_positive": winner_top10 > 0.0,
        "beats_frozen_residual_april_top10": winner_top10 > residual_top10,
        "latest_april_week_positive": float(latest.loc["joint_winner_causal_recent", "mean_net_bps"]) > 0.0,
    }
    promotion["passes_all"] = all(promotion.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}."))
    try:
        head_table.to_parquet(temporary / "head_metrics.parquet", index=False)
        ranking.to_parquet(temporary / "ablation_effects.parquet", index=False)
        control_table.to_parquet(temporary / "april_control_tail_metrics.parquet", index=False)
        weekly_table.to_parquet(temporary / "april_weekly_metrics.parquet", index=False)
        summary = {
            "winner": winner,
            "march_selection_score_bps": float(ranking.iloc[0]["selection_score"]),
            "april_top10_bps": {
                name: float(top10.loc[name, "mean_net_bps"]) for name in top10.index
            },
            "promotion_gates": promotion,
            "decision": "DO_NOT_PROMOTE" if not promotion["passes_all"] else "PORTFOLIO_REPLAY_ELIGIBLE",
        }
        (temporary / "summary.json").write_text(json.dumps(safe(summary), indent=2, sort_keys=True) + "\n")
        manifest = {
            "schema": SCHEMA, "status": "COMPLETED",
            "source_manifest_sha256": sha256_file(args.source / "manifest.json"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "contracts": {
                "population": "identical 69,258-row April path-context population",
                "ranking": "one pooled global top-k with candidate_id tie break",
                "primary": "direct exact-policy net output only",
                "april": "reused diagnostic not promotion evidence",
            },
        }
        manifest["outputs_sha256"] = {
            path.name: sha256_file(path) for path in temporary.iterdir() if path.is_file()
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(safe(manifest), indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n")
        os.replace(temporary, args.output)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--path-labels", type=Path, default=DEFAULT_PATH_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
