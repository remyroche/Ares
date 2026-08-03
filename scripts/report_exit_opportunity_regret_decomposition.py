#!/usr/bin/env python3
"""Decompose path opportunity, executable capture, costs, and policy regret."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _daily_ci(
    frame: pd.DataFrame,
    column: str,
    *,
    seed: int,
    draws: int = 2_000,
) -> tuple[float, float]:
    daily = (
        frame.assign(day=pd.to_datetime(frame["__ts__"], utc=True).dt.floor("D"))
        .groupby("day", sort=True)[column]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    if len(daily) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = daily[rng.integers(0, len(daily), size=(draws, len(daily)))]
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def decompose(frame: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    if not arms or arms[0] != "parent":
        raise ValueError("arms must begin with parent")
    required = [
        *IDENTITY,
        "mfe__parent",
        *[f"net__{arm}" for arm in arms],
        *[f"gross__{arm}" for arm in arms],
        *[f"cost__{arm}" for arm in arms],
    ]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError("candidate replay is missing columns: " + ", ".join(missing))
    output = frame.copy()
    net = output[[f"net__{arm}" for arm in arms]].to_numpy(dtype=np.float64)
    gross = output[[f"gross__{arm}" for arm in arms]].to_numpy(dtype=np.float64)
    cost = output[[f"cost__{arm}" for arm in arms]].to_numpy(dtype=np.float64)
    if not np.isfinite(net).all() or not np.isfinite(gross).all():
        raise ValueError("candidate replay contains non-finite economics")
    if not np.allclose(gross - cost, net, rtol=0.0, atol=1e-10):
        raise ValueError("candidate replay violates gross - cost = net")
    best = np.argmax(net, axis=1)
    row = np.arange(len(output))
    output["family_oracle_arm"] = np.asarray(arms, dtype=object)[best]
    output["parent_net_return"] = net[:, 0]
    output["parent_gross_return"] = gross[:, 0]
    output["parent_cost_return"] = cost[:, 0]
    output["family_oracle_net_return"] = net[row, best]
    output["family_oracle_gross_return"] = gross[row, best]
    output["family_oracle_cost_return"] = cost[row, best]
    output["small_family_policy_regret"] = (
        output["family_oracle_net_return"] - output["parent_net_return"]
    )
    output["path_mfe_return"] = pd.to_numeric(
        output["mfe__parent"], errors="raise"
    )
    output["path_opportunity_net_of_parent_cost"] = (
        output["path_mfe_return"] - output["parent_cost_return"]
    )
    output["path_to_family_gross_gap"] = (
        output["path_mfe_return"] - output["family_oracle_gross_return"]
    )
    if (output["small_family_policy_regret"] < -1e-12).any():
        raise ValueError("family oracle is worse than parent on at least one row")
    return output


def _metrics(
    frame: pd.DataFrame,
    *,
    scope: str,
    fold: int | str,
    seed: int,
) -> dict[str, Any]:
    ci_low, ci_high = _daily_ci(
        frame, "family_oracle_net_return", seed=seed
    )
    return {
        "fold": fold,
        "scope": scope,
        "rows": int(len(frame)),
        "path_mfe_bps": float(frame["path_mfe_return"].mean() * 10_000.0),
        "path_net_of_cost_bps": float(
            frame["path_opportunity_net_of_parent_cost"].mean() * 10_000.0
        ),
        "parent_gross_bps": float(frame["parent_gross_return"].mean() * 10_000.0),
        "parent_net_bps": float(frame["parent_net_return"].mean() * 10_000.0),
        "family_oracle_gross_bps": float(
            frame["family_oracle_gross_return"].mean() * 10_000.0
        ),
        "family_oracle_net_bps": float(
            frame["family_oracle_net_return"].mean() * 10_000.0
        ),
        "small_family_policy_regret_bps": float(
            frame["small_family_policy_regret"].mean() * 10_000.0
        ),
        "path_to_family_gross_gap_bps": float(
            frame["path_to_family_gross_gap"].mean() * 10_000.0
        ),
        "parent_positive_rate": float((frame["parent_net_return"] > 0.0).mean()),
        "family_oracle_positive_rate": float(
            (frame["family_oracle_net_return"] > 0.0).mean()
        ),
        "family_oracle_daily_ci_low_bps": float(ci_low * 10_000.0),
        "family_oracle_daily_ci_high_bps": float(ci_high * 10_000.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source = pd.read_parquet(args.candidate_replay)
    if source.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("candidate replay contains duplicate identities")
    frame = decompose(source, args.arms)
    fold_values = pd.to_numeric(frame[args.fold_col], errors="raise").astype(int)
    frame[args.fold_col] = fold_values
    records: list[dict[str, Any]] = []
    for fold, fold_frame in frame.groupby(args.fold_col, sort=True):
        records.append(
            _metrics(
                fold_frame,
                scope="global",
                fold=int(fold),
                seed=args.seed + int(fold),
            )
        )
        for side, side_frame in fold_frame.groupby("side_name", sort=True):
            records.append(
                _metrics(
                    side_frame,
                    scope=str(side),
                    fold=int(fold),
                    seed=args.seed + int(fold),
                )
            )
    later = frame.loc[fold_values > fold_values.min()].copy()
    records.append(
        _metrics(later, scope="global", fold="later_folds", seed=args.seed)
    )
    metrics = pd.DataFrame(records)
    metrics.to_csv(args.output_dir / "opportunity_regret_metrics.csv", index=False)
    frame.to_parquet(
        args.output_dir / "candidate_opportunity_regret.parquet",
        index=False,
        compression="zstd",
    )
    later_global = metrics.loc[
        metrics["fold"].astype(str).eq("later_folds")
        & metrics["scope"].eq("global")
    ].iloc[0]
    latest_fold = int(fold_values.max())
    latest_global = metrics.loc[
        metrics["fold"].astype(str).eq(str(latest_fold))
        & metrics["scope"].eq("global")
    ].iloc[0]
    family_still_negative = (
        float(later_global["family_oracle_net_bps"]) < 0.0
        and float(latest_global["family_oracle_net_bps"]) < 0.0
    )
    path_opportunity_positive = (
        float(later_global["path_net_of_cost_bps"]) > 0.0
        and float(latest_global["path_net_of_cost_bps"]) > 0.0
    )
    if family_still_negative and path_opportunity_positive:
        diagnosis = (
            "positive_path_opportunity_but_small_exit_family_cannot_capture_cost"
        )
    elif family_still_negative:
        diagnosis = "selected_candidates_lack_small_family_net_opportunity"
    else:
        diagnosis = "small_family_contains_positive_hindsight_opportunity"
    summary = {
        "schema": "exit_opportunity_regret_decomposition_v1",
        "status": "diagnostic_not_promotion_evidence",
        "diagnosis": diagnosis,
        "contract": {
            "family_oracle": (
                "per-row hindsight maximum within the frozen executable arm list; "
                "diagnostic only and not a causal policy"
            ),
            "path_mfe": (
                "full-horizon path excursion from executable entry; diagnostic "
                "ceiling, not guaranteed executable PnL"
            ),
            "accounting": (
                "path MFE -> family gross capture -> row cost -> family net; "
                "gross - cost = net asserted for every arm"
            ),
            "admission": (
                "identical fold-local pooled global-top-k identities from the "
                "source replay; no reranking"
            ),
        },
        "arms": list(args.arms),
        "rows": int(len(frame)),
        "later_folds": later_global.to_dict(),
        "latest_fold": latest_global.to_dict(),
        "oracle_arm_share": {
            str(key): float(value)
            for key, value in frame["family_oracle_arm"]
            .value_counts(normalize=True)
            .items()
        },
        "source": {
            "path": str(args.candidate_replay),
            "sha256": _sha256(args.candidate_replay),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fold-col", default="execution_ev_model_ablation_oof_fold"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "parent",
            "stop_0p90",
            "stop_1p10",
            "activation_0p90",
            "activation_1p10",
            "giveback_0p90",
            "giveback_1p10",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260727)
    return parser


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary), indent=2))


if __name__ == "__main__":
    main()
