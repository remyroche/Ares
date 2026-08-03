#!/usr/bin/env python3
"""Create strict chronological TP6/SL4 B10 base OOF predictions for meta work.

Each row is scored by a side-local base model trained only on resolved labels
strictly before its monthly block.  The model contract is the frozen B10 base
winner (B5 .25 + B7 .75, BW4 h=100/tau=100), and the output is a *ledger*, not
an evaluation claim.  Future-path fields are used only to build training
targets; they are never admitted to the feature matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extreme_price_movements.tp6_sl4_target_weights import (
    TP6SL4Columns, TargetParameters, WeightParameters, build_target, build_weight,
)

PARAMS = dict(objective="huber", alpha=.90, n_estimators=80, learning_rate=.05,
              num_leaves=24, min_child_samples=400, colsample_bytree=.8,
              subsample=.8, reg_lambda=10., random_state=20260811, n_jobs=1, verbosity=-1)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    parser.add_argument("--sidecar", type=Path, default=ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1")
    parser.add_argument("--base-contract", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oof-start", default="2024-05-01")
    parser.add_argument("--oof-end", default="2024-12-01")
    parser.add_argument("--block-days", type=int, default=31)
    parser.add_argument(
        "--train-lookback-days",
        type=int,
        default=0,
        help="Use only this many resolved calendar days before each OOF block; 0 keeps all available history.",
    )
    parser.add_argument("--resume", action="store_true", help="reuse immutable completed monthly OOF parts")
    return parser.parse_args()


def _features(root: Path) -> dict[str, list[str]]:
    result = {}
    for side in ("long", "short"):
        manifest = json.loads((root / side / "target_family_manifest.json").read_text())
        fields = manifest.get("feature_contract", {}).get(f"T2_soft_barrier|tp3_sl2|{side}")
        if not isinstance(fields, list) or not 30 <= len(fields) <= 40 or len(fields) != len(set(fields)):
            raise ValueError(f"invalid frozen {side} base feature contract")
        result[side] = list(fields)
    return result


def _read(panel: Path, sidecar: Path, features: dict[str, list[str]]) -> pd.DataFrame:
    pparts, sparts = sorted((panel / "parts").glob("*.parquet")), sorted((sidecar / "parts").glob("*.parquet"))
    if not pparts or len(pparts) != len(sparts):
        raise FileNotFoundError("panel/sidecar partitions are incomplete")
    base_cols = list(dict.fromkeys(["candidate_id", "__ts__", "side_name", "t2_path_mae_atr", *features["long"], *features["short"]]))
    label_cols = ["candidate_id", "__label_available_at__", "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t4_tp6_sl4_terminal_pnl_atr"]
    bases = [pd.read_parquet(part, columns=base_cols) for part in pparts]
    labels = [pd.read_parquet(part, columns=label_cols) for part in sparts]
    base, label = pd.concat(bases, ignore_index=True), pd.concat(labels, ignore_index=True)
    data = base.merge(label, on="candidate_id", how="inner", validate="one_to_one")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    data["__label_available_at__"] = pd.to_datetime(data["__label_available_at__"], utc=True)
    if data.candidate_id.duplicated().any() or not data.__label_available_at__.gt(data.__ts__).all():
        raise ValueError("invalid TP6 label lineage")
    return data


def _matrix(frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
    values = frame.loc[:, fields].replace([np.inf, -np.inf], np.nan)
    coverage = 1. - values.isna().mean()
    if (coverage < .90).any():
        raise ValueError(f"base feature coverage <90%: {coverage[coverage < .90].to_dict()}")
    return values.fillna(0.).to_numpy(np.float32)


def _target_weight(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = replace(TP6SL4Columns(), terminal_atr="t4_tp6_sl4_terminal_pnl_atr")
    parameters = TargetParameters(terminal_beta=.50, terminal_clip_atr=2., adverse_beta=.10, adverse_clip_atr=2.)
    target = .25 * build_target(train, "B5", columns=columns, parameters=parameters) + .75 * build_target(train, "B7", columns=columns, parameters=parameters)
    weight = build_weight(train, "BW4", columns=columns, target=target, target_parameters=replace(parameters, economic_hurdle_bps=100., gross_tau_bps=100.), parameters=WeightParameters())
    return target, weight


def _score(train: pd.DataFrame, evaluation: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    target, weight = _target_weight(train)
    xtr, xev = _matrix(train, fields), _matrix(evaluation, fields)
    raw = np.column_stack([np.maximum(lgb.LGBMRegressor(**PARAMS).fit(xtr, target[:, j], sample_weight=weight).predict(xev), 0.) for j in range(3)])
    prob = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    net = train.t4_tp6_sl4_net_bps.to_numpy(float)
    means = (target * net[:, None] * weight[:, None]).sum(axis=0) / np.maximum((target * weight[:, None]).sum(axis=0), 1e-12)
    return prob, prob @ means


def main() -> None:
    args = _args()
    start, end = pd.Timestamp(args.oof_start, tz="UTC"), pd.Timestamp(args.oof_end, tz="UTC")
    if not start < end or args.block_days < 1:
        raise ValueError("invalid OOF calendar")
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    parts_dir = args.out / "parts"; parts_dir.mkdir(parents=True, exist_ok=True)
    features = _features(args.base_contract); data = _read(args.panel, args.sidecar, features)
    rows, blocks = [], []
    for block_start in pd.date_range(start, end, freq=f"{args.block_days}D", tz="UTC", inclusive="left"):
        block_end = min(block_start + pd.Timedelta(days=args.block_days), end)
        destination = parts_dir / f"{block_start:%Y%m%d}.parquet"
        if destination.exists():
            part = pd.read_parquet(destination)
            rows.append(part); blocks.append({"start": str(block_start), "end": str(block_end), "rows": len(part), "status": "reused"})
            continue
        evaluation = data[data.__ts__.ge(block_start) & data.__ts__.lt(block_end)]
        output = []
        for side in ("long", "short"):
            train_mask = data.side_name.eq(side) & data.__label_available_at__.lt(block_start)
            if args.train_lookback_days:
                train_mask &= data.__label_available_at__.ge(
                    block_start - pd.Timedelta(days=int(args.train_lookback_days))
                )
            train = data[train_mask]
            ev = evaluation[evaluation.side_name.eq(side)]
            if min(len(train), len(ev)) < 1000:
                raise ValueError(f"insufficient {side} rows for {block_start}")
            p, score = _score(train, ev, features[side])
            x = ev[["candidate_id", "__ts__", "side_name", "t4_tp6_sl4_net_bps"]].copy()
            x["base_expected_net_bps"] = score; x[["base_p_upper", "base_p_lower", "base_p_timeout"]] = p
            x["base_fit_resolved_before"] = block_start
            output.append(x)
        part = pd.concat(output, ignore_index=True); part.to_parquet(destination, index=False, compression="zstd"); rows.append(part)
        blocks.append({"start": str(block_start), "end": str(block_end), "rows": len(part), "status": "materialised"})
        print(json.dumps({"block": blocks[-1]}), flush=True)
    ledger = pd.concat(rows, ignore_index=True)
    ledger.to_parquet(args.out / "base_oof_ledger.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({"schema": "tp6_sl4_b10_bw4_base_oof_ledger_v1", "base_target": "B10=.25*B5+.75*B7", "base_weight": "BW4 h=100 tau=100", "geometry": "TP6/SL4 H12", "strict_oof": "every score uses a model fit only on labels resolved before block start", "train_lookback_days": int(args.train_lookback_days), "features": features, "blocks": blocks}, indent=2) + "\n")


if __name__ == "__main__":
    main()
