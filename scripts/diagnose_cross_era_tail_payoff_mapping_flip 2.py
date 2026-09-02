#!/usr/bin/env python3
"""Research-only audit of the cross-era tail-payoff causal mapping flip.

It never retrains or changes the challenger.  Mapping variants are selected
from the frozen historical OOF file only; July 20--23 is evaluated once after
that selection.  The map is deliberately causal: a day's reference set only
contains scores whose entry and 12h label had resolved before that day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


SCHEMA = "cross_era_tail_payoff_mapping_flip_audit_v1"
SIDES = ("long", "short")
CURRENT_START = pd.Timestamp("2026-07-20T00:00:00Z")
MAP_DAYS = 21


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
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare(frame: pd.DataFrame, *, current: bool) -> pd.DataFrame:
    """Normalise the small immutable prediction/evaluation inputs."""
    required = {"candidate_id", "__ts__", "side_name", "tail_ev_bps"}
    if not current:
        required |= {"label_resolution_utc", "execution_net_ev_12h"}
    else:
        required |= {"execution_label_available_at", "execution_net_ev_12h"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"mapping-audit source misses columns: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if current:
        result["label_resolution_utc"] = pd.to_datetime(
            result["execution_label_available_at"], utc=True, errors="raise"
        )
    else:
        result["label_resolution_utc"] = pd.to_datetime(result["label_resolution_utc"], utc=True, errors="raise")
    result["tail_ev_bps"] = pd.to_numeric(result["tail_ev_bps"], errors="raise")
    result["execution_net_ev_12h"] = pd.to_numeric(result["execution_net_ev_12h"], errors="raise")
    if not result["side_name"].astype(str).isin(SIDES).all():
        raise ValueError("unexpected side in mapping audit")
    return result


def _fit_isotonic(reference: pd.DataFrame, score: str) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(reference[score].to_numpy(float), reference["execution_net_ev_12h"].to_numpy(float) * 1e4)
    return model


def _model_summary(model: IsotonicRegression | None, prefix: str) -> dict[str, float | int | None]:
    if model is None:
        return {f"{prefix}_{name}": None for name in ("x_min", "x_max", "y_min", "y_max", "knots", "plateaus")}
    xs = np.asarray(model.X_thresholds_, dtype=float)
    ys = np.asarray(model.y_thresholds_, dtype=float)
    return {
        f"{prefix}_x_min": float(xs.min()), f"{prefix}_x_max": float(xs.max()),
        f"{prefix}_y_min": float(ys.min()), f"{prefix}_y_max": float(ys.max()),
        f"{prefix}_knots": int(len(xs)), f"{prefix}_plateaus": int(len(np.unique(ys))),
    }


def causal_map(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    variant: str,
    score: str = "tail_ev_bps",
    shrink_rows: int = 2_000,
    min_pooled_rows: int = 200,
    min_side_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a 21d causal isotonic map and expose its allocation mechanics."""
    if variant not in {"raw", "pooled", "side_local", "side_shrunk"}:
        raise ValueError(f"unknown mapping variant {variant}")
    out = target.copy()
    out["mapped_bps"] = np.nan
    out["pooled_mapped_bps"] = np.nan
    out["side_mapped_bps"] = np.nan
    out["side_shrink_weight"] = 0.0
    out["side_shrink_contribution_bps"] = 0.0
    diagnostics: list[dict[str, Any]] = []
    if variant == "raw":
        out["mapped_bps"] = out[score].to_numpy(float)
        out["pooled_mapped_bps"] = out[score].to_numpy(float)
        out["side_mapped_bps"] = out[score].to_numpy(float)
        return out, pd.DataFrame(columns=["variant", "day", "side_name"])
    for day, local in out.groupby(out["__ts__"].dt.floor("D"), sort=True):
        lower = day - pd.Timedelta(days=MAP_DAYS)
        available = source.loc[
            source["__ts__"].lt(day) & source["__ts__"].ge(lower)
            & source["label_resolution_utc"].lt(day) & source[score].notna()
        ]
        pooled: IsotonicRegression | None = None
        if len(available) >= min_pooled_rows:
            pooled = _fit_isotonic(available, score)
        for side in SIDES:
            index = local.index[local["side_name"].astype(str).eq(side)]
            if len(index) == 0:
                continue
            side_available = available.loc[available["side_name"].astype(str).eq(side)]
            side_model = _fit_isotonic(side_available, score) if len(side_available) >= min_side_rows else None
            raw = out.loc[index, score].to_numpy(float)
            pooled_values = raw if pooled is None else pooled.predict(raw)
            side_values = raw if side_model is None else side_model.predict(raw)
            if variant == "pooled":
                values, weight = pooled_values, 0.0
            elif variant == "side_local":
                values, weight = (side_values if side_model is not None else pooled_values), float(side_model is not None)
            else:
                weight = len(side_available) / (len(side_available) + shrink_rows) if side_model is not None else 0.0
                values = weight * side_values + (1.0 - weight) * pooled_values
            out.loc[index, "mapped_bps"] = values
            out.loc[index, "pooled_mapped_bps"] = pooled_values
            out.loc[index, "side_mapped_bps"] = side_values
            out.loc[index, "side_shrink_weight"] = weight
            out.loc[index, "side_shrink_contribution_bps"] = values - pooled_values
            record: dict[str, Any] = {
                "variant": variant, "day": day, "side_name": side,
                "target_rows": int(len(index)), "reference_rows": int(len(available)),
                "side_reference_rows": int(len(side_available)), "mapped": bool(pooled is not None),
                "side_model_available": bool(side_model is not None), "side_shrink_weight": float(weight),
                "mean_side_shrink_contribution_bps": float(np.mean(values - pooled_values)),
                "mean_abs_side_shrink_contribution_bps": float(np.mean(np.abs(values - pooled_values))),
                "target_raw_min": float(raw.min()), "target_raw_max": float(raw.max()),
                "target_mapped_min": float(np.min(values)), "target_mapped_max": float(np.max(values)),
            }
            record.update(_model_summary(pooled, "pooled"))
            record.update(_model_summary(side_model, "side"))
            for model, prefix in ((pooled, "pooled"), (side_model, "side")):
                if model is None:
                    record[f"{prefix}_above_support_rows"] = 0
                    record[f"{prefix}_below_support_rows"] = 0
                else:
                    xs = np.asarray(model.X_thresholds_, dtype=float)
                    record[f"{prefix}_above_support_rows"] = int((raw > xs.max()).sum())
                    record[f"{prefix}_below_support_rows"] = int((raw < xs.min()).sum())
            diagnostics.append(record)
    if out["mapped_bps"].isna().any():
        raise AssertionError("mapping did not score every target row")
    return out, pd.DataFrame(diagnostics)


def add_secondary_order(frame: pd.DataFrame, secondary: str) -> pd.DataFrame:
    """Tie-break only: primary mapped values remain unchanged."""
    result = frame.copy()
    raw = result["tail_ev_bps"].to_numpy(float)
    if secondary == "candidate_id":
        result["secondary_order"] = 0.0
    elif secondary == "raw_percentile":
        result["secondary_order"] = pd.Series(raw, index=result.index).rank(pct=True, method="average")
    elif secondary == "raw_robust_z":
        values = np.empty(len(result), dtype=float)
        for side, index in result.groupby("side_name", sort=False).groups.items():
            sample = raw[result.index.get_indexer(index)]
            median = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median)))
            values[result.index.get_indexer(index)] = 0.0 if mad == 0 else (sample - median) / (1.4826 * mad)
        result["secondary_order"] = values
    else:
        raise ValueError(f"unknown secondary order {secondary}")
    return result


def select_top(frame: pd.DataFrame) -> pd.DataFrame:
    finite = frame.loc[np.isfinite(frame["mapped_bps"])].copy()
    take = max(1, int(math.ceil(.10 * len(finite))))
    # secondary_order is intentionally consulted only after the isotonic map.
    return finite.sort_values(
        ["mapped_bps", "secondary_order", "candidate_id"], ascending=[False, False, True], kind="stable"
    ).iloc[:take].copy()


def metric_rows(frame: pd.DataFrame, *, arm: str, split: str) -> list[dict[str, Any]]:
    selected = select_top(frame)
    cutoff = float(selected["mapped_bps"].iloc[-1])
    cutoff_tie_rows = int((frame["mapped_bps"] == cutoff).sum())
    selected_cutoff_tie_rows = int((selected["mapped_bps"] == cutoff).sum())
    rows: list[dict[str, Any]] = []
    for level, key in (("aggregate", None), ("month", frame["__ts__"].dt.strftime("%Y-%m")), ("day", frame["__ts__"].dt.strftime("%Y-%m-%d"))):
        groups: Iterable[tuple[str, pd.DataFrame]]
        if key is None:
            groups = [("all", selected)]
        else:
            groups = selected.groupby(key.loc[selected.index], sort=True)
        for name, local in groups:
            net = local["execution_net_ev_12h"].to_numpy(float) * 1e4
            row: dict[str, Any] = {
                "arm": arm, "split": split, "level": level, "period": str(name), "rows": int(len(local)),
                "net_ev_bps": float(np.mean(net)), "positive_precision": float(np.mean(net > 0)),
                "cvar05_bps": float(np.mean(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))])),
                "mean_raw_tail_ev_bps": float(local["tail_ev_bps"].mean()),
                "mean_mapped_ev_bps": float(local["mapped_bps"].mean()),
                "long_rows": int(local["side_name"].eq("long").sum()),
                "short_rows": int(local["side_name"].eq("short").sum()),
                "global_top10_cutoff_mapped_bps": cutoff, "cutoff_tie_rows": cutoff_tie_rows,
                "selected_cutoff_tie_rows": selected_cutoff_tie_rows,
            }
            for component in ("p_positive", "p_adverse_negative", "p_timeout_negative", "p_other_negative", "q25_positive_bps", "q50_positive_bps", "q50_adverse_bps", "q85_adverse_bps", "q75_other_bps", "q75_timeout_bps"):
                if component in local:
                    row[f"mean_{component}"] = float(pd.to_numeric(local[component], errors="coerce").mean())
            rows.append(row)
    for side, local in selected.groupby("side_name", sort=True):
        net = local["execution_net_ev_12h"].to_numpy(float) * 1e4
        row = {"arm": arm, "split": split, "level": "side", "period": str(side), "rows": int(len(local)),
                     "net_ev_bps": float(net.mean()), "positive_precision": float((net > 0).mean()),
                     "cvar05_bps": float(np.mean(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))])),
                     "mean_raw_tail_ev_bps": float(local["tail_ev_bps"].mean()), "mean_mapped_ev_bps": float(local["mapped_bps"].mean()),
                     "long_rows": int(local["side_name"].eq("long").sum()), "short_rows": int(local["side_name"].eq("short").sum()),
                     "global_top10_cutoff_mapped_bps": cutoff, "cutoff_tie_rows": cutoff_tie_rows,
                     "selected_cutoff_tie_rows": selected_cutoff_tie_rows}
        for component in ("p_positive", "p_adverse_negative", "p_timeout_negative", "p_other_negative", "q25_positive_bps", "q50_positive_bps", "q50_adverse_bps", "q85_adverse_bps", "q75_other_bps", "q75_timeout_bps"):
            if component in local:
                row[f"mean_{component}"] = float(pd.to_numeric(local[component], errors="coerce").mean())
        rows.append(row)
    return rows


def arm_definition() -> list[tuple[str, str, int, str]]:
    return [
        ("raw_tail", "raw", 0, "candidate_id"),
        ("pooled", "pooled", 0, "candidate_id"),
        ("pooled_raw_percentile_tie", "pooled", 0, "raw_percentile"),
        ("pooled_raw_robust_z_tie", "pooled", 0, "raw_robust_z"),
        ("side_local", "side_local", 0, "candidate_id"),
        ("side_shrunk_2000", "side_shrunk", 2_000, "candidate_id"),
        ("side_shrunk_6000", "side_shrunk", 6_000, "candidate_id"),
        ("side_shrunk_2000_raw_percentile_tie", "side_shrunk", 2_000, "raw_percentile"),
        ("side_shrunk_2000_raw_robust_z_tie", "side_shrunk", 2_000, "raw_robust_z"),
    ]


def choose_historical(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    aggregate = metrics.loc[(metrics["split"] == "historical") & (metrics["level"] == "aggregate")].set_index("arm")
    monthly = metrics.loc[(metrics["split"] == "historical") & (metrics["level"] == "month")]
    required_months = sorted(monthly["period"].unique())
    rows: list[dict[str, Any]] = []
    for arm, record in aggregate.iterrows():
        local = monthly.loc[monthly["arm"].eq(arm)]
        covered = sorted(local["period"].unique()) == required_months and bool((local["rows"] > 0).all())
        rows.append({"arm": arm, "aggregate_net_ev_bps": float(record["net_ev_bps"]),
                     "worst_month_net_ev_bps": float(local["net_ev_bps"].min()),
                     "latest_month_net_ev_bps": float(local.sort_values("period").iloc[-1]["net_ev_bps"]),
                     "aggregate_cvar05_bps": float(record["cvar05_bps"]), "month_coverage": covered,
                     "months": ",".join(required_months)})
    ledger = pd.DataFrame(rows).sort_values(
        ["month_coverage", "aggregate_net_ev_bps", "worst_month_net_ev_bps", "latest_month_net_ev_bps", "aggregate_cvar05_bps", "arm"],
        ascending=[False, False, False, False, False, True], kind="stable"
    ).reset_index(drop=True)
    return str(ledger.iloc[0]["arm"]), ledger


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    history_path = args.source_dir / "historical_oof_winner.parquet"
    current_path = args.source_dir / "current_scored_exact.parquet"
    frozen_path = args.source_dir / "frozen_before_current_evaluation.json"
    history = prepare(pd.read_parquet(history_path), current=False)
    current = prepare(pd.read_parquet(current_path), current=True)
    if not bool(history["label_resolution_utc"].lt(CURRENT_START).all()):
        raise ValueError("historical OOF contains labels unresolved at frozen current cutoff")
    if not bool(current["__ts__"].ge(CURRENT_START).all()):
        raise ValueError("current evaluation is not wholly post-freeze")
    all_metrics: list[dict[str, Any]] = []
    all_diagnostics: list[pd.DataFrame] = []
    scored_current: dict[str, pd.DataFrame] = {}
    for arm, variant, shrink, secondary in arm_definition():
        historical_mapped, historical_diag = causal_map(history, history, variant=variant, shrink_rows=shrink)
        historical_mapped = add_secondary_order(historical_mapped, secondary)
        all_metrics.extend(metric_rows(historical_mapped, arm=arm, split="historical"))
        # The only source passed to this map is pre-current OOF data.  The
        # current labels are not fed back into mapping or arm selection.
        current_mapped, current_diag = causal_map(history, current, variant=variant, shrink_rows=shrink)
        current_mapped = add_secondary_order(current_mapped, secondary)
        all_metrics.extend(metric_rows(current_mapped, arm=arm, split="frozen_current"))
        scored_current[arm] = current_mapped
        all_diagnostics.extend([historical_diag.assign(arm=arm, split="historical"), current_diag.assign(arm=arm, split="frozen_current")])
    metrics = pd.DataFrame(all_metrics)
    winner, ledger = choose_historical(metrics)
    # Save a compact current candidate audit for every arm, including raw/model
    # components, so the side allocation flip can be reproduced row by row.
    component_cols = [column for column in current.columns if column.startswith(("p_", "q"))]
    audit_columns = ["candidate_id", "__ts__", "side_name", "tail_ev_bps", "execution_net_ev_12h", *component_cols]
    candidate_audit = pd.concat([
        frame.loc[:, audit_columns + ["mapped_bps", "pooled_mapped_bps", "side_mapped_bps", "side_shrink_weight", "side_shrink_contribution_bps", "secondary_order"]].assign(arm=arm)
        for arm, frame in scored_current.items()
    ], ignore_index=True)
    diagnostics = pd.concat([item for item in all_diagnostics if not item.empty], ignore_index=True)
    metrics.to_csv(args.output_dir / "arm_economics.csv", index=False)
    ledger.to_csv(args.output_dir / "historical_selection_ledger.csv", index=False)
    diagnostics.to_csv(args.output_dir / "mapping_diagnostics.csv", index=False)
    candidate_audit.to_parquet(args.output_dir / "current_candidate_mapping_audit.parquet", index=False)
    selection = {
        "selected_arm": winner,
        "selection_rule": "historical OOF only: complete month coverage, aggregate global top10 net EV, worst month, latest month, CVaR05",
        "current_outcomes_used_for_selection": False,
        "current_outcomes_used_in_mapping": False,
        "current_evaluation": "one frozen July 20--23 evaluation, mapped exclusively against pre-July20 historical OOF labels",
    }
    write_json(args.output_dir / "selection.json", selection)
    report = {
        "schema": SCHEMA, "status": "completed_research_only_no_promotion", "promotion_eligible": False,
        "source": {"dir": str(args.source_dir), "historical_oof": {"path": str(history_path), "sha256": sha256(history_path), "rows": len(history)},
                   "current_scored_exact": {"path": str(current_path), "sha256": sha256(current_path), "rows": len(current)},
                   "frozen_state": {"path": str(frozen_path), "sha256": sha256(frozen_path)}},
        "selection": selection,
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "arm_economics": args.output_dir / "arm_economics.csv", "historical_selection_ledger": args.output_dir / "historical_selection_ledger.csv",
            "mapping_diagnostics": args.output_dir / "mapping_diagnostics.csv", "current_candidate_mapping_audit": args.output_dir / "current_candidate_mapping_audit.parquet",
            "selection": args.output_dir / "selection.json"}.items()},
    }
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "report": {"path": str(args.output_dir / "report.json"), "sha256": sha256(args.output_dir / "report.json")}})
    return report


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_challenger_20260730_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_mapping_flip_audit_20260730_v4"))
    return parser


if __name__ == "__main__":
    run(parser().parse_args())
