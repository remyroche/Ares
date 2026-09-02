#!/usr/bin/env python3
"""Frozen mapping/allocation audit for the direct exact-net q25 challenger.

This is deliberately an audit, not a re-fit.  It compares the unmodified q25
ranker with the frozen side-shrunk map, a pooled causal map, and pooled maps
whose isotonic plateaus are resolved by a predeclared continuous q25-derived
secondary key.  Every choice, including the secondary key, is made on the
historical OOF prediction file.  July 20--23 labels are read only once after
that choice has been persisted in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_cross_era_tail_payoff_mapping_flip import (
    MAP_DAYS,
    causal_map,
)


SCHEMA = "cross_era_direct_q25_mapping_allocation_audit_v1"
CURRENT_START = pd.Timestamp("2026-07-20T00:00:00Z")
SCORE = "q25_net_bps"
MAPPED = "mapped_q25_bps"
SIDES = ("long", "short")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _check_source(source_dir: Path) -> dict[str, Any]:
    report_path = source_dir / "report.json"
    manifest_path = source_dir / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256(report_path) != manifest["report"]["sha256"]:
        raise ValueError("direct challenger report hash mismatch")
    # Do not touch the post-freeze exact-current outcome file here.  The
    # historical OOF and current *prediction* files are enough to freeze the
    # mapping/allocation choice.  Exact July labels are hash-checked and read
    # only after frozen_before_current_evaluation.json is written.
    for name in ("historical_oof_winner", "current_predictions_before_outcomes"):
        record = report["outputs"][name]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"direct challenger output hash mismatch: {name}")
    winner = report.get("winner", {})
    if winner.get("score_column") != SCORE or winner.get("mapped_column") != MAPPED:
        raise ValueError("this audit is defined only for the frozen q25 winner")
    return report


def prepare(frame: pd.DataFrame, *, current: bool) -> pd.DataFrame:
    """Make a generic q25 frame compatible with the causal-map audit helper."""
    required = {"candidate_id", "__ts__", "side_name", SCORE}
    if not current:
        required.add("label_resolution_utc")
        required.add("execution_net_ev_12h")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"q25 mapping source misses columns: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if current:
        # The pre-label current prediction file must be sufficient to map and
        # freeze allocation.  A placeholder is replaced by exact outcomes only
        # after the freeze state is hashed to disk.
        result["execution_net_ev_12h"] = np.nan
    else:
        result["execution_net_ev_12h"] = pd.to_numeric(result["execution_net_ev_12h"], errors="raise")
    result[SCORE] = pd.to_numeric(result[SCORE], errors="raise")
    if current:
        # Target resolution is never consulted: only the historical source is
        # passed to causal_map for the frozen current evaluation.  Giving it a
        # concrete value makes that non-use explicit and testable.
        result["label_resolution_utc"] = result["__ts__"]
    else:
        result["label_resolution_utc"] = pd.to_datetime(result["label_resolution_utc"], utc=True, errors="raise")
    if not result["side_name"].astype(str).isin(SIDES).all():
        raise ValueError("unexpected side in q25 mapping audit")
    if not np.isfinite(result[SCORE]).all():
        raise ValueError("q25 mapping audit requires finite q25 scores")
    return result


def _set_primary(frame: pd.DataFrame, values: pd.Series | np.ndarray, secondary: str) -> pd.DataFrame:
    result = frame.copy()
    result["mapped_bps"] = np.asarray(values, dtype=float)
    if not np.isfinite(result["mapped_bps"]).all():
        raise ValueError("mapping produced non-finite scores")
    raw = result[SCORE].to_numpy(float)
    if secondary == "candidate_id":
        result["secondary_order"] = 0.0
    elif secondary == "raw_percentile":
        # A monotone, continuous re-expression of raw q25.  It therefore
        # cannot alter the primary mapping except within an exact plateau.
        result["secondary_order"] = pd.Series(raw, index=result.index).rank(pct=True, method="average")
    elif secondary == "raw_robust_z":
        values = np.empty(len(result), dtype=float)
        for _, index in result.groupby("side_name", sort=False).groups.items():
            positions = result.index.get_indexer(index)
            sample = raw[positions]
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median)))
            values[positions] = 0.0 if mad == 0.0 else (sample - median) / (1.4826 * mad)
        result["secondary_order"] = values
    else:
        raise ValueError(f"unknown secondary order: {secondary}")
    return result


def select_top(frame: pd.DataFrame) -> pd.DataFrame:
    take = max(1, int(math.ceil(.10 * len(frame))))
    return frame.sort_values(
        ["mapped_bps", "secondary_order", "candidate_id"],
        ascending=[False, False, True], kind="stable",
    ).iloc[:take].copy()


def _tail_metrics(selected: pd.DataFrame) -> tuple[float, float, float]:
    net = selected["execution_net_ev_12h"].to_numpy(float) * 1e4
    count = max(1, int(math.ceil(.05 * len(net))))
    return float(net.mean()), float((net > 0).mean()), float(np.sort(net)[:count].mean())


def economics(frame: pd.DataFrame, *, arm: str, split: str) -> list[dict[str, Any]]:
    """Global and independently reselected monthly top-10 economics."""
    selected = select_top(frame)
    aggregate, precision, cvar = _tail_metrics(selected)
    cutoff = float(selected["mapped_bps"].iloc[-1])
    tie = frame.loc[np.isclose(frame["mapped_bps"], cutoff, rtol=0.0, atol=1e-10)]
    rows = [{
        "arm": arm, "split": split, "level": "aggregate", "period": "all",
        "rows": int(len(selected)), "net_ev_bps": aggregate,
        "positive_precision": precision, "cvar05_bps": cvar,
        "long_rows": int(selected["side_name"].eq("long").sum()),
        "short_rows": int(selected["side_name"].eq("short").sum()),
        "global_top10_cutoff_mapped_bps": cutoff,
        "cutoff_tie_rows": int(len(tie)),
        "selected_cutoff_tie_rows": int(np.isclose(selected["mapped_bps"], cutoff, rtol=0.0, atol=1e-10).sum()),
        "mean_raw_q25_bps": float(selected[SCORE].mean()),
        "mean_mapped_q25_bps": float(selected["mapped_bps"].mean()),
    }]
    for month, population in frame.groupby(frame["__ts__"].dt.strftime("%Y-%m"), sort=True):
        # This intentionally re-ranks each month's full candidate population;
        # grouping the aggregate selected book would not test latest-month
        # admission performance.
        local = select_top(population)
        net, positive, local_cvar = _tail_metrics(local)
        rows.append({
            "arm": arm, "split": split, "level": "month", "period": str(month),
            "rows": int(len(local)), "net_ev_bps": net, "positive_precision": positive,
            "cvar05_bps": local_cvar, "long_rows": int(local["side_name"].eq("long").sum()),
            "short_rows": int(local["side_name"].eq("short").sum()),
            "global_top10_cutoff_mapped_bps": cutoff, "cutoff_tie_rows": int(len(tie)),
            "selected_cutoff_tie_rows": int(np.isclose(selected["mapped_bps"], cutoff, rtol=0.0, atol=1e-10).sum()),
            "mean_raw_q25_bps": float(local[SCORE].mean()), "mean_mapped_q25_bps": float(local["mapped_bps"].mean()),
        })
    for side, local in selected.groupby("side_name", sort=True):
        net, positive, local_cvar = _tail_metrics(local)
        rows.append({
            "arm": arm, "split": split, "level": "side", "period": str(side),
            "rows": int(len(local)), "net_ev_bps": net, "positive_precision": positive,
            "cvar05_bps": local_cvar, "long_rows": int(local["side_name"].eq("long").sum()),
            "short_rows": int(local["side_name"].eq("short").sum()),
            "global_top10_cutoff_mapped_bps": cutoff, "cutoff_tie_rows": int(len(tie)),
            "selected_cutoff_tie_rows": int(np.isclose(selected["mapped_bps"], cutoff, rtol=0.0, atol=1e-10).sum()),
            "mean_raw_q25_bps": float(local[SCORE].mean()), "mean_mapped_q25_bps": float(local["mapped_bps"].mean()),
        })
    return rows


def plateau_diagnostics(frame: pd.DataFrame, *, arm: str, split: str) -> dict[str, Any]:
    selected = select_top(frame)
    cutoff = float(selected["mapped_bps"].iloc[-1])
    plateau = frame.loc[np.isclose(frame["mapped_bps"], cutoff, rtol=0.0, atol=1e-10)]
    selected_plateau = selected.loc[np.isclose(selected["mapped_bps"], cutoff, rtol=0.0, atol=1e-10)]
    return {
        "arm": arm, "split": split, "rows": int(len(frame)), "selected_rows": int(len(selected)),
        "unique_primary_scores": int(frame["mapped_bps"].nunique()), "cutoff_mapped_bps": cutoff,
        "cutoff_plateau_rows": int(len(plateau)), "selected_from_cutoff_plateau": int(len(selected_plateau)),
        "cutoff_plateau_selection_share": float(len(selected_plateau) / len(plateau)),
        "cutoff_plateau_span_raw_q25_bps": float(plateau[SCORE].max() - plateau[SCORE].min()),
        "selected_secondary_min": float(selected_plateau["secondary_order"].min()),
        "selected_secondary_max": float(selected_plateau["secondary_order"].max()),
        "unselected_secondary_max": (
            float(plateau.loc[~plateau.index.isin(selected_plateau.index), "secondary_order"].max())
            if len(plateau) > len(selected_plateau) else None
        ),
    }


def clipping_summary(diagnostics: pd.DataFrame, *, arm: str, split: str) -> list[dict[str, Any]]:
    if diagnostics.empty:
        return [{"arm": arm, "split": split, "side_name": "all", "days": 0, "target_rows": 0,
                 "pooled_above_support_rows": 0, "pooled_below_support_rows": 0,
                 "side_above_support_rows": 0, "side_below_support_rows": 0,
                 "mean_side_shrink_weight": 0.0, "mean_abs_side_shrink_contribution_bps": 0.0}]
    rows = []
    for side, local in diagnostics.groupby("side_name", sort=True):
        rows.append({
            "arm": arm, "split": split, "side_name": str(side), "days": int(len(local)),
            "target_rows": int(local["target_rows"].sum()),
            "pooled_above_support_rows": int(local["pooled_above_support_rows"].sum()),
            "pooled_below_support_rows": int(local["pooled_below_support_rows"].sum()),
            "side_above_support_rows": int(local["side_above_support_rows"].sum()),
            "side_below_support_rows": int(local["side_below_support_rows"].sum()),
            "mean_side_shrink_weight": float(local["side_shrink_weight"].mean()),
            "mean_abs_side_shrink_contribution_bps": float(local["mean_abs_side_shrink_contribution_bps"].mean()),
        })
    return rows


def historical_ledger(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregate = metrics.loc[(metrics["split"] == "historical_oof") & (metrics["level"] == "aggregate")].set_index("arm")
    monthly = metrics.loc[(metrics["split"] == "historical_oof") & (metrics["level"] == "month")]
    expected_months = sorted(monthly["period"].unique())
    rows = []
    for arm, record in aggregate.iterrows():
        local = monthly.loc[monthly["arm"].eq(arm)].sort_values("period")
        rows.append({
            "arm": arm, "month_coverage": sorted(local["period"].unique()) == expected_months and bool((local["rows"] > 0).all()),
            "months": ",".join(expected_months), "aggregate_net_ev_bps": float(record["net_ev_bps"]),
            "worst_month_net_ev_bps": float(local["net_ev_bps"].min()),
            "latest_month_net_ev_bps": float(local.iloc[-1]["net_ev_bps"]),
            "aggregate_cvar05_bps": float(record["cvar05_bps"]),
            "global_top10_long_rows": int(record["long_rows"]), "global_top10_short_rows": int(record["short_rows"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["month_coverage", "aggregate_net_ev_bps", "worst_month_net_ev_bps", "latest_month_net_ev_bps", "aggregate_cvar05_bps", "arm"],
        ascending=[False, False, False, False, False, True], kind="stable",
    ).reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = _check_source(args.source_dir)
    history_path = Path(source["outputs"]["historical_oof_winner"]["path"])
    current_prediction_path = Path(source["outputs"]["current_predictions_before_outcomes"]["path"])
    current_label_path = Path(source["outputs"]["current_scored_exact"]["path"])
    source_frozen_path = args.source_dir / "frozen_before_current_evaluation.json"
    history = prepare(pd.read_parquet(history_path), current=False)
    # This file has no exact current outcomes.  It is the only current input
    # allowed before selection/state freeze below.
    current = prepare(pd.read_parquet(current_prediction_path), current=True)
    if not bool(history["label_resolution_utc"].lt(CURRENT_START).all()):
        raise ValueError("historical OOF leaks labels past the current freeze")
    if not bool(current["__ts__"].ge(CURRENT_START).all()):
        raise ValueError("current evaluation is not wholly post-freeze")

    # Recompute maps independently for support/clipping diagnostics and assert
    # the frozen side-shrunk artifact itself is exactly reproducible.
    hist_side, hist_side_diag = causal_map(history, history, variant="side_shrunk", score=SCORE, shrink_rows=2_000)
    current_side, current_side_diag = causal_map(history, current, variant="side_shrunk", score=SCORE, shrink_rows=2_000)
    parity = {
        "historical_max_abs_difference_bps": float(np.max(np.abs(hist_side["mapped_bps"] - history[MAPPED]))),
        "current_max_abs_difference_bps": float(np.max(np.abs(current_side["mapped_bps"] - current[MAPPED]))),
    }
    if parity["historical_max_abs_difference_bps"] > 1e-9 or parity["current_max_abs_difference_bps"] > 1e-9:
        raise ValueError(f"stored frozen side-shrunk mapping did not reproduce: {parity}")
    hist_pooled, hist_pooled_diag = causal_map(history, history, variant="pooled", score=SCORE)
    current_pooled, current_pooled_diag = causal_map(history, current, variant="pooled", score=SCORE)

    # These are specified before reading current labels.  They preserve the
    # isotonic primary score and differ only inside a plateau.
    definitions = (
        ("raw_q25", history[SCORE], current[SCORE], "candidate_id", None, None),
        ("existing_side_shrunk", history[MAPPED], current[MAPPED], "candidate_id", hist_side_diag, current_side_diag),
        ("pooled_causal", hist_pooled["mapped_bps"], current_pooled["mapped_bps"], "candidate_id", hist_pooled_diag, current_pooled_diag),
        ("pooled_q25_percentile_plateau", hist_pooled["mapped_bps"], current_pooled["mapped_bps"], "raw_percentile", hist_pooled_diag, current_pooled_diag),
        ("pooled_side_robust_z_plateau", hist_pooled["mapped_bps"], current_pooled["mapped_bps"], "raw_robust_z", hist_pooled_diag, current_pooled_diag),
    )
    historical_economics_rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    clipping_rows: list[dict[str, Any]] = []
    prelabel_current_parts: list[pd.DataFrame] = []
    for arm, hist_values, current_values, secondary, hist_diag, current_diag in definitions:
        hist = _set_primary(history, hist_values, secondary)
        frozen_current = _set_primary(current, current_values, secondary)
        historical_economics_rows.extend(economics(hist, arm=arm, split="historical_oof"))
        plateau_rows.extend([plateau_diagnostics(hist, arm=arm, split="historical_oof"), plateau_diagnostics(frozen_current, arm=arm, split="frozen_current")])
        clipping_rows.extend(clipping_summary(hist_diag if hist_diag is not None else pd.DataFrame(), arm=arm, split="historical_oof"))
        clipping_rows.extend(clipping_summary(current_diag if current_diag is not None else pd.DataFrame(), arm=arm, split="frozen_current"))
        prelabel_current_parts.append(frozen_current.loc[:, ["candidate_id", "__ts__", "side_name", SCORE, MAPPED, "mapped_bps", "secondary_order"]].assign(arm=arm))

    historical_metrics = pd.DataFrame(historical_economics_rows)
    ledger = historical_ledger(historical_metrics)
    winner = str(ledger.iloc[0]["arm"])
    selected_plateau_arm = str(
        ledger.loc[ledger["arm"].isin(["pooled_q25_percentile_plateau", "pooled_side_robust_z_plateau"])].iloc[0]["arm"]
    )
    selection = {
        "selected_arm": winner,
        "historically_selected_continuous_plateau_arm": selected_plateau_arm,
        "selection_rule": "historical OOF only: complete month coverage, aggregate global top10 exact net EV, worst month, latest month, CVaR05",
        "current_outcomes_used_for_selection": False,
        "current_outcomes_used_in_mapping": False,
        "predeclared_arms": [item[0] for item in definitions],
        "predeclared_continuous_secondary_keys": ["raw_percentile", "raw_robust_z"],
        "mapping_window_days": MAP_DAYS,
        "current_evaluation": "one frozen July 20--23 exact-label evaluation after historical-only selection",
    }
    args.output_dir.mkdir(parents=True)
    prelabel_freeze = {
        "schema": SCHEMA,
        "selection": selection,
        "current_outcomes_loaded": False,
        "source": {
            "historical_oof": {"path": str(history_path), "sha256": sha256(history_path), "rows": int(len(history))},
            "current_predictions_before_outcomes": {"path": str(current_prediction_path), "sha256": sha256(current_prediction_path), "rows": int(len(current))},
            # Binding is metadata-only at this stage; its contents are not
            # loaded or used until after this freeze file exists.
            "planned_exact_current_label_source": {"path": str(current_label_path), "sha256": source["outputs"]["current_scored_exact"]["sha256"]},
        },
        "frozen_mapping_parity": parity,
    }
    frozen_evaluation_path = args.output_dir / "frozen_before_current_evaluation.json"
    write_json(frozen_evaluation_path, prelabel_freeze)
    frozen_evaluation_sha = sha256(frozen_evaluation_path)

    # Now, and only now, load the exact outcome column.  The frozen choice is
    # already immutable and bound to the pre-label prediction data above.
    if sha256(current_label_path) != source["outputs"]["current_scored_exact"]["sha256"]:
        raise ValueError("exact current label source hash mismatch")
    labels = pd.read_parquet(current_label_path, columns=["candidate_id", "__ts__", "side_name", "execution_net_ev_12h"])
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    label_keys = ["candidate_id", "__ts__", "side_name"]
    if labels.duplicated(label_keys).any() or len(labels) != len(current):
        raise ValueError("exact current labels are not one-to-one with the frozen prediction file")
    metrics_rows = list(historical_economics_rows)
    candidate_parts: list[pd.DataFrame] = []
    for prelabel in prelabel_current_parts:
        evaluated = prelabel.merge(labels, on=label_keys, how="inner", validate="one_to_one")
        if len(evaluated) != len(current):
            raise ValueError("current exact labels do not completely cover frozen predictions")
        metrics_rows.extend(economics(evaluated, arm=str(evaluated["arm"].iloc[0]), split="frozen_current"))
        candidate_parts.append(evaluated)
    metrics = pd.DataFrame(metrics_rows)
    outputs: dict[str, Path] = {
        "arm_economics": args.output_dir / "arm_economics.csv",
        "historical_selection_ledger": args.output_dir / "historical_selection_ledger.csv",
        "plateau_diagnostics": args.output_dir / "plateau_diagnostics.csv",
        "support_clipping_diagnostics": args.output_dir / "support_clipping_diagnostics.csv",
        "current_candidate_mapping_audit": args.output_dir / "current_candidate_mapping_audit.parquet",
        "frozen_before_current_evaluation": frozen_evaluation_path,
        "selection": args.output_dir / "selection.json",
    }
    metrics.to_csv(outputs["arm_economics"], index=False)
    ledger.to_csv(outputs["historical_selection_ledger"], index=False)
    pd.DataFrame(plateau_rows).to_csv(outputs["plateau_diagnostics"], index=False)
    pd.DataFrame(clipping_rows).to_csv(outputs["support_clipping_diagnostics"], index=False)
    pd.concat(candidate_parts, ignore_index=True).to_parquet(outputs["current_candidate_mapping_audit"], index=False)
    write_json(outputs["selection"], selection)
    report = {
        "schema": SCHEMA, "status": "completed_research_only_no_promotion", "promotion_eligible": False,
        "portfolio_replay_authorized": False,
        "source": {
            "direct_challenger_report": {"path": str(args.source_dir / "report.json"), "sha256": sha256(args.source_dir / "report.json")},
            "historical_oof": {"path": str(history_path), "sha256": sha256(history_path), "rows": int(len(history))},
            "current_predictions_before_outcomes": {"path": str(current_prediction_path), "sha256": sha256(current_prediction_path), "rows": int(len(current))},
            "current_scored_exact_labels": {"path": str(current_label_path), "sha256": sha256(current_label_path), "rows": int(len(labels))},
            "source_frozen_state": {"path": str(source_frozen_path), "sha256": sha256(source_frozen_path)},
        },
        "frozen_mapping_parity": parity,
        "prelabel_freeze": {"path": str(frozen_evaluation_path), "sha256": frozen_evaluation_sha},
        "selection": selection,
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
    }
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "manifest.json", {
        "schema": SCHEMA, "status": report["status"], "promotion_eligible": False,
        "report": {"path": str(args.output_dir / "report.json"), "sha256": sha256(args.output_dir / "report.json")},
        "outputs": report["outputs"],
    })
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", type=Path, default=Path("data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"))
    result.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_direct_q25_mapping_allocation_audit_20260730_v1"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
