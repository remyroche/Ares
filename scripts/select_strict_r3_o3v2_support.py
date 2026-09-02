#!/usr/bin/env python3
"""Development-only selector for O3-v2 training-support contracts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from select_strict_r3_o3v2_targets import CONTROL, PRIMARY_SCORE, TAIL_WEIGHTS

SCHEMA = "strict_r3_o3v2_support_selection_v1"


def _months(raw: str) -> tuple[str, ...]:
    result = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not result:
        raise ValueError("at least one month is required")
    return result


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _collapse_repeated_contract_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated control receipts while rejecting divergent contracts.

    The staged funnel deliberately evaluates ``S0_uniform`` in several
    bounded screens.  For a given target/support/month/tail it is the same
    deterministic contract, not several independent observations.  Keep one
    copy only after proving all score metrics agree exactly to numerical
    precision.  A divergent duplicate would make the final selector
    ambiguous, so fail closed instead of silently averaging it.
    """
    keys = ("target_arm", "support_arm", "month", "tail")
    numeric = (
        "trades", "net_ev_bps_per_trade", "net_sum_bps", "policy_rank_ic",
        "conditional_policy_rank_ic", "base_rank_correlation",
        "control_ev", "control_ic", "delta_bps",
    )
    kept: list[pd.DataFrame] = []
    for values, local in frame.groupby(list(keys), sort=False, dropna=False):
        if len(local) > 1:
            for column in numeric:
                if column not in local:
                    continue
                series = pd.to_numeric(local[column], errors="coerce").to_numpy(float)
                reference = series[0]
                if not np.allclose(series, reference, rtol=0.0, atol=1e-9, equal_nan=True):
                    raise AssertionError(
                        "non-identical repeated support contract rows for "
                        f"{dict(zip(keys, values, strict=True))}: {column}"
                    )
        kept.append(local.iloc[[0]])
    return pd.concat(kept, ignore_index=True)


def run(
    *, target_metrics: Path, support_metrics: tuple[Path, ...], out: Path,
    months: tuple[str, ...], uniform_from_support: bool = False,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    target = pd.read_parquet(target_metrics)
    baseline = target.loc[
        target["arm"].eq(CONTROL) & target["score"].eq(PRIMARY_SCORE)
        & target["month"].isin(months) & target["tail"].isin(TAIL_WEIGHTS),
        ["month", "tail", "net_ev_bps_per_trade", "policy_rank_ic"],
    ].rename(columns={"net_ev_bps_per_trade": "control_ev", "policy_rank_ic": "control_ic"})
    if baseline.duplicated(["month", "tail"]).any():
        raise AssertionError("non-unique primary control metric")
    parts = [pd.read_parquet(path) for path in support_metrics]
    support = pd.concat(parts, ignore_index=True).loc[lambda x: x["score"].eq(PRIMARY_SCORE)].copy()
    split = support["arm"].str.split("__", n=1, expand=True)
    support["target_arm"], support["support_arm"] = split[0], split[1]
    active_targets = set(support["target_arm"].astype(str))
    if not uniform_from_support:
        # A weighting screen is only interpretable against the exact same
        # target trained uniformly.  Legacy target funnels already emitted
        # those receipts, so materialise them as S0 rather than spending a
        # duplicate fit.  The selected-physical-slot successor instead emits
        # its own S0 receipt: the older five-slot median is not comparable.
        uniform = target.loc[
            target["arm"].isin(active_targets)
            & target["score"].eq(PRIMARY_SCORE)
            & target["month"].isin(months)
            & target["tail"].isin(TAIL_WEIGHTS),
        ].copy()
        uniform["target_arm"] = uniform["arm"].astype(str)
        uniform["support_arm"] = "S0_uniform"
        support = pd.concat([support, uniform], ignore_index=True, sort=False)
    support = support.loc[support["month"].isin(months) & support["tail"].isin(TAIL_WEIGHTS)].merge(
        baseline, on=["month", "tail"], how="inner", validate="many_to_one",
    )
    support["delta_bps"] = support["net_ev_bps_per_trade"] - support["control_ev"]
    support = _collapse_repeated_contract_rows(support)
    rows = []
    for (target_arm, support_arm), local in support.groupby(["target_arm", "support_arm"], sort=True):
        weighted = []
        for _month, part in local.groupby("month", sort=True):
            score = part.set_index("tail")["delta_bps"]
            if set(TAIL_WEIGHTS).issubset(score.index):
                weighted.append(float(sum(TAIL_WEIGHTS[tail] * score.loc[tail] for tail in TAIL_WEIGHTS)))
        if not weighted:
            continue
        values = np.asarray(weighted, dtype=float)
        top5 = local.loc[np.isclose(local["tail"], .05), "delta_bps"].to_numpy(float)
        rows.append({
            "target_arm": target_arm, "support_arm": support_arm, "selection_score_bps": float(values.mean() - .25 * values.std() - max(0., -values.min())),
            "weighted_delta_mean_bps": float(values.mean()), "weighted_delta_worst_bps": float(values.min()),
            "top5_positive_months": int(np.sum(top5 > 0)), "top5_delta_mean_bps": float(top5.mean()),
            "mean_rank_ic_delta": float((local["policy_rank_ic"] - local["control_ic"]).mean()),
        })
    table = pd.DataFrame(rows).sort_values(["target_arm", "selection_score_bps", "support_arm"], ascending=[True, False, True], kind="stable")
    winners = table.groupby("target_arm", as_index=False, sort=False).first()
    out.mkdir(parents=True)
    support.to_parquet(out / "support_development_monthly_delta.parquet", index=False, compression="zstd")
    table.to_parquet(out / "support_development_selection.parquet", index=False, compression="zstd")
    _exclusive_json(out / "selected_support_contracts.json", {
        "schema": SCHEMA, "development_months": list(months), "primary_score": PRIMARY_SCORE,
        "selection": "uniform target control plus weighted top1/top2/top5 development delta, stability-penalised; one winner per already-selected target",
        "uniform_control": (
            "S0_uniform is the fitted selected-physical-slot receipt"
            if uniform_from_support else "S0_uniform is the matching unweighted target-funnel receipt, not a duplicated fit"
        ),
        "selected": winners.loc[:, ["target_arm", "support_arm"]].to_dict("records"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-metrics", type=Path, required=True)
    parser.add_argument("--support-metrics", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", default="2025-10,2025-11,2025-12")
    parser.add_argument("--uniform-from-support", action="store_true", help="require an S0 receipt from the supplied support source; used by selected-physical-slot successors")
    args = parser.parse_args()
    run(target_metrics=args.target_metrics, support_metrics=tuple(args.support_metrics), out=args.out, months=_months(args.months), uniform_from_support=args.uniform_from_support)


if __name__ == "__main__":
    main()
