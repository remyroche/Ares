#!/usr/bin/env python3
"""Report the matched strict-R3 → Cell-day → R5 → portfolio waterfall.

Global top-k rows are retrospective ranking diagnostics.  Executable rows are
reported separately after causal Cell-day/R5 admission and after the portfolio
auction.  Policy and exact TP6/SL4 economics are evaluated on identical
candidate selections; outcomes never participate in selection construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_r5_canonical_waterfall_v1"
TAILS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
LAYERS = (
    ("base", "base_rank42"),
    ("conditional_consensus", "conditional_consensus_rank"),
    ("base_consensus_75_25", "upstream"),
    ("correctness_final", "final_score"),
    ("cell_day_expected_net", "causal_21d_side_expected_net_bps"),
    ("r5_posterior_expected_net", "trust_posterior_expected_bps"),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(
    *,
    scored_ledger: Path,
    prequential_ledger: Path,
    cell_day_provenance: Path,
    r5_predictions: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    score = pd.read_parquet(scored_ledger)
    base = pd.read_parquet(prequential_ledger, columns=[
        "candidate_id", "__decision_ts__", "h12_label_valid",
        "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps",
    ])
    mapped = pd.read_parquet(cell_day_provenance)
    trust = pd.read_parquet(r5_predictions)
    for frame, name in ((score, "score"), (base, "TP6"), (mapped, "Cell-day"), (trust, "R5")):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} ledger contains duplicate candidate IDs")
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True)
    score = score.loc[
        score["__decision_ts__"].ge(start) & score["__decision_ts__"].lt(end)
    ].copy()
    if not score["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("waterfall contains non-prequential score rows")
    geometry = score["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(geometry) != 1:
        raise ValueError("waterfall requires one frozen geometry bundle")
    lineage = (
        "__decision_ts__", "conversion_bundle_sha256", "geometry_bundle_sha256",
        "upstream_bundle_sha256", "ev_score_family_id", "stack_is_prequential",
    )
    map_required = {
        "candidate_id", *lineage, "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "causal_21d_side_mapping_status",
    }
    missing = sorted(map_required.difference(mapped.columns))
    if missing:
        raise ValueError(f"Cell-day provenance lacks: {missing}")
    map_columns = [
        "candidate_id", *lineage, "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps", "causal_21d_side_mapping_status",
        *( ["cell_day_fixed_score_cell"] if "cell_day_fixed_score_cell" in mapped else []),
        *( ["cell_day_retained_day_support"] if "cell_day_retained_day_support" in mapped else []),
    ]
    mapped = mapped.loc[:, map_columns].copy()
    joined = score.merge(
        mapped, on="candidate_id", how="left", validate="one_to_one",
        suffixes=("", "__map"),
    )
    for field in lineage:
        other = f"{field}__map"
        if not joined[field].astype(str).eq(joined[other].astype(str)).all():
            raise ValueError(f"Cell-day provenance changed lineage field {field}")
        joined = joined.drop(columns=other)
    trust_required = {
        "candidate_id", "__decision_ts__", "trust_posterior_expected_bps",
        "trust_posterior_admitted_ge_50bps", "trust_posterior_available",
        "r5_bundle_cutoff", "geometry_bundle_sha256",
    }
    missing = sorted(trust_required.difference(trust.columns))
    if missing:
        raise ValueError(f"R5 predictions lack: {missing}")
    trust = trust.loc[:, list(trust_required)].rename(columns={
        "__decision_ts__": "__trust_decision_ts__",
        "geometry_bundle_sha256": "__trust_geometry__",
    })
    joined = joined.merge(trust, on="candidate_id", how="left", validate="one_to_one")
    if not joined["__decision_ts__"].eq(pd.to_datetime(joined["__trust_decision_ts__"], utc=True)).all():
        raise ValueError("R5 prediction timestamp mismatch")
    if not joined["geometry_bundle_sha256"].astype(str).eq(
        joined["__trust_geometry__"].astype(str)
    ).all():
        raise ValueError("R5 prediction geometry mismatch")
    joined = joined.drop(columns=["__trust_decision_ts__", "__trust_geometry__"])
    base = base.rename(columns={"__decision_ts__": "__tp6_decision_ts__"})
    joined = joined.merge(base, on="candidate_id", how="left", validate="one_to_one")
    overlap = joined["__tp6_decision_ts__"].notna()
    if not joined.loc[overlap, "__decision_ts__"].eq(
        pd.to_datetime(joined.loc[overlap, "__tp6_decision_ts__"], utc=True)
    ).all():
        raise ValueError("TP6 outcome identity/timestamp mismatch")
    return joined.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _select_tail(frame: pd.DataFrame, score: str, tail: float) -> pd.DataFrame:
    finite = frame.loc[np.isfinite(pd.to_numeric(frame[score], errors="coerce"))]
    count = max(1, int(math.ceil(tail * len(finite)))) if len(finite) else 0
    return finite.sort_values(
        [score, "final_score", "candidate_id"],
        ascending=[False, False, True], kind="stable",
    ).head(count)


def _economic_rows(
    selected: pd.DataFrame,
    *,
    stage: str,
    selection: str,
    tail: float | None,
    period_scope: str,
    period: str,
    population_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contracts = (
        ("optimized_trailing_policy", "policy_path_valid", "policy_gross_bps", "policy_net_bps"),
        ("exact_h12_tp6_sl4", "h12_label_valid", "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps"),
    )
    for outcome, valid_field, gross_field, net_field in contracts:
        valid = (
            selected[valid_field].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(selected[gross_field], errors="coerce"))
            & np.isfinite(pd.to_numeric(selected[net_field], errors="coerce"))
        )
        block = selected.loc[valid]
        net = pd.to_numeric(block[net_field], errors="coerce")
        gross = pd.to_numeric(block[gross_field], errors="coerce")
        rows.append({
            "stage": stage, "selection": selection, "tail": tail,
            "period_scope": period_scope, "period": period, "outcome": outcome,
            "population_rows": int(population_rows),
            "selected_rows": int(len(selected)), "valid_outcomes": int(len(block)),
            "outcome_coverage": float(len(block) / max(len(selected), 1)),
            "gross_bps_per_trade": float(gross.mean()) if len(block) else np.nan,
            "net_bps_per_trade": float(net.mean()) if len(block) else np.nan,
            "net_sum_bps": float(net.sum()) if len(block) else 0.0,
            "positive_rate": float(net.gt(0.0).mean()) if len(block) else np.nan,
        })
    return rows


def _periods(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    work = frame.copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    work["week"] = work["__decision_ts__"].dt.strftime("%G-W%V")
    output = [("all", "all", work)]
    output.extend(("month", str(key), value) for key, value in work.groupby("month", sort=True))
    output.extend(("week", str(key), value) for key, value in work.groupby("week", sort=True))
    return output


def _waterfall(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, period, block in _periods(frame):
        for stage, score in LAYERS:
            for tail in TAILS:
                rows.extend(_economic_rows(
                    _select_tail(block, score, tail), stage=stage,
                    selection="retrospective_global_tail_diagnostic", tail=tail,
                    period_scope=scope, period=period, population_rows=len(block),
                ))
        for stage, admitted_field, score in (
            (
                "cell_day_expected_net", "causal_21d_side_admitted_ge_50bps",
                "causal_21d_side_expected_net_bps",
            ),
            (
                "r5_posterior_expected_net", "trust_posterior_admitted_ge_50bps",
                "trust_posterior_expected_bps",
            ),
        ):
            admitted = block.loc[block[admitted_field].fillna(False).astype(bool)].copy()
            rows.extend(_economic_rows(
                admitted, stage=stage, selection="executable_all_admitted", tail=None,
                period_scope=scope, period=period, population_rows=len(block),
            ))
            for tail in TAILS:
                rows.extend(_economic_rows(
                    _select_tail(admitted, score, tail), stage=stage,
                    selection="diagnostic_tail_within_executable_admission", tail=tail,
                    period_scope=scope, period=period, population_rows=len(block),
                ))
    return pd.DataFrame(rows)


def _portfolio_risk(portfolio_dir: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    decisions = pd.read_parquet(portfolio_dir / "portfolio_decisions.parquet")
    equity = pd.read_parquet(portfolio_dir / "portfolio_equity.parquet")
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    equity = equity.loc[equity["timestamp"].ge(start) & equity["timestamp"].lt(end)].copy()
    value_field = "mtm_equity" if "mtm_equity" in equity else "wallet"
    daily = equity.set_index("timestamp")[value_field].resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    peak = daily.cummax()
    drawdown = daily / peak - 1.0
    downside = returns.loc[returns.lt(0.0)]
    initial = float(daily.iloc[0]) if len(daily) else np.nan
    final = float(daily.iloc[-1]) if len(daily) else np.nan
    total_return = final / initial - 1.0 if np.isfinite(initial) and initial else np.nan
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    years = max((end - start).total_seconds() / (365.25 * 86_400.0), 1.0 / 365.25)
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0 if np.isfinite(total_return) and total_return > -1 else np.nan
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(365.25)) if returns.std(ddof=0) > 0 else np.nan
    sortino = float(returns.mean() / downside.std(ddof=0) * np.sqrt(365.25)) if downside.std(ddof=0) > 0 else np.nan
    ulcer = float(np.sqrt(np.mean(np.square(np.minimum(drawdown, 0.0))))) if len(drawdown) else np.nan
    pain = float(np.mean(np.abs(np.minimum(drawdown, 0.0)))) if len(drawdown) else np.nan
    gains = returns.loc[returns.gt(0.0)].sum()
    losses = -returns.loc[returns.lt(0.0)].sum()
    wallet_days = max((end - start).total_seconds() / 86_400.0, 1.0)
    result = {
        "evaluation_start": start, "evaluation_end_exclusive": end,
        "accepted_trades": int(len(accepted)),
        "trades_per_calendar_day": float(len(accepted) / wallet_days),
        "wallet_start": initial, "wallet_end": final, "total_return": total_return,
        "annualized_return": annualized, "max_drawdown": max_dd,
        "sharpe_daily": sharpe, "sortino_daily": sortino,
        "calmar": annualized / abs(max_dd) if np.isfinite(annualized) and max_dd < 0 else np.nan,
        "ulcer_index": ulcer, "pain_index": pain,
        "pain_ratio": total_return / pain if np.isfinite(total_return) and pain > 0 else np.nan,
        "omega_zero_daily": float(gains / losses) if losses > 0 else np.nan,
        "positive_days": int(returns.gt(0.0).sum()), "negative_days": int(returns.lt(0.0).sum()),
        "mean_wallet_margin_utilisation": float(pd.to_numeric(
            equity.get("committed_wallet_cap_utilization", np.nan), errors="coerce",
        ).mean()),
        "maximum_wallet_margin_utilisation": float(pd.to_numeric(
            equity.get("committed_wallet_cap_utilization", np.nan), errors="coerce",
        ).max()),
    }
    return pd.DataFrame([result])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--cell-day-provenance", type=Path, required=True)
    parser.add_argument("--r5-predictions", type=Path, required=True)
    parser.add_argument("--portfolio-dir", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable waterfall output exists: {args.out_dir}")
    start, end = pd.to_datetime(args.evaluation_start, utc=True), pd.to_datetime(args.evaluation_end, utc=True)
    frame = _load(
        scored_ledger=args.scored_ledger, prequential_ledger=args.prequential_ledger,
        cell_day_provenance=args.cell_day_provenance, r5_predictions=args.r5_predictions,
        start=start, end=end,
    )
    metrics = _waterfall(frame)
    risk = _portfolio_risk(args.portfolio_dir, start=start, end=end)
    args.out_dir.mkdir(parents=True)
    metrics.to_parquet(args.out_dir / "layer_waterfall_metrics.parquet", index=False)
    risk.to_parquet(args.out_dir / "portfolio_risk_metrics.parquet", index=False)
    global_metrics = metrics.loc[metrics["period_scope"].eq("all")].copy()
    global_metrics.to_csv(args.out_dir / "layer_waterfall_global.csv", index=False)
    metrics.loc[metrics["period_scope"].eq("month")].to_csv(
        args.out_dir / "layer_waterfall_monthly.csv", index=False,
    )
    metrics.loc[metrics["period_scope"].eq("week")].to_csv(
        args.out_dir / "layer_waterfall_weekly.csv", index=False,
    )
    manifest = {
        "schema": SCHEMA, "side": "long", "rows": int(len(frame)),
        "evaluation_start": start.isoformat(), "evaluation_end_exclusive": end.isoformat(),
        "tails": list(TAILS), "layers": [stage for stage, _ in LAYERS],
        "tail_interpretation": "retrospective global ranking diagnostic only",
        "executable_selection": "28d Cell-day trim15 then R5 9m posterior >=50 bps then portfolio",
        "outcomes_joined_after_selection": True,
        "scored_ledger_sha256": _sha(args.scored_ledger),
        "prequential_ledger_sha256": _sha(args.prequential_ledger),
        "cell_day_provenance_sha256": _sha(args.cell_day_provenance),
        "r5_predictions_sha256": _sha(args.r5_predictions),
        "portfolio_manifest_sha256": _sha(args.portfolio_dir / "run_manifest.json"),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
