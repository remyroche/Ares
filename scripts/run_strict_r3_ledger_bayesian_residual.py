#!/usr/bin/env python3
"""Strict-prequential empirical-Bayes residual challenger for the long ledger.

This is deliberately a small, causal recovery experiment.  It learns only
``policy_net_bps - prequential_base_anchor_bps`` on rows whose policy outcome
was available before a held month.  Candidate inputs are restricted to the
frozen 120-field base contract; no future/path, raw K9, held-period percentile,
or admission outcome is used as an input.  The output is an *alternative rank*
diagnostic, not a live admission or sizing change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.trust_sizing_ablation import QuantileBins, residual_classes


SCHEMA = "strict_r3_ledger_bayesian_residual_v1"
SEED = 20260812


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fields(path: Path, side: str) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    values = payload.get("base_fields_by_side", {}).get(side, payload.get("base_fields", []))
    fields = tuple(map(str, values))
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("requires the frozen 120-field strict-R3 base contract")
    return fields


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = frame["__decision_ts__"].dt.strftime("%Y-%m")
    rng = np.random.default_rng(seed)
    quota = int(math.ceil(cap / month.nunique()))
    picked: list[np.ndarray] = []
    for token in sorted(month.unique()):
        positions = np.flatnonzero(month.eq(token).to_numpy())
        if len(positions) > quota:
            positions = np.sort(rng.choice(positions, quota, replace=False))
        picked.append(positions)
    selected = np.concatenate(picked)
    if len(selected) > cap:
        selected = np.sort(rng.choice(selected, cap, replace=False))
    return frame.iloc[selected].sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()


def _select_fields(train: pd.DataFrame, fields: tuple[str, ...], maximum: int) -> tuple[tuple[str, ...], QuantileBins]:
    bins = QuantileBins.fit(train, fields)
    codes = bins.transform(train)
    target = residual_classes(train["policy_net_bps"], train["prequential_base_anchor_bps"])
    rank = pd.to_numeric(train["prequential_base_rank42"], errors="coerce").to_numpy(float)
    conditional = np.digitize(rank, np.unique(np.quantile(rank, [.2, .4, .6, .8])), right=True)
    scored: list[tuple[float, str]] = []
    for col, name in enumerate(fields):
        value = 0.0
        for state in range(5):
            mask = conditional == state
            if int(mask.sum()) < 100:
                continue
            # 5 finite quantile bins plus a missing bin, and five residual bands.
            joint = np.bincount(codes[mask, col].astype(np.int64) * 5 + target[mask].astype(np.int64), minlength=30).reshape(6, 5)
            joint = joint.astype(float) + 0.5
            joint /= joint.sum()
            px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
            value += float(mask.mean()) * float(np.sum(joint * np.log(joint / (px * py))))
        scored.append((value, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = tuple(name for _, name in scored[:maximum])
    return chosen, QuantileBins.fit(train, chosen)


def _fit_predict(train: pd.DataFrame, held: pd.DataFrame, fields: tuple[str, ...], prior_strength: float) -> tuple[pd.DataFrame, dict[str, object]]:
    chosen, bins = _select_fields(train, fields, minimum(12, len(fields)))
    train_codes, held_codes = bins.transform(train), bins.transform(held)
    residual = (
        pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
        - pd.to_numeric(train["prequential_base_anchor_bps"], errors="coerce").to_numpy(float)
    )
    global_mean = float(np.mean(residual))
    effects: list[np.ndarray] = []
    support_rows: list[np.ndarray] = []
    for idx in range(len(chosen)):
        code_train, code_held = train_codes[:, idx], held_codes[:, idx]
        count = int(max(code_train.max(initial=0), code_held.max(initial=0))) + 1
        support = np.bincount(code_train, minlength=count).astype(float)
        total = np.bincount(code_train, weights=residual, minlength=count)
        posterior = (total + prior_strength * global_mean) / (support + prior_strength)
        lookup = np.clip(code_held, 0, count - 1)
        effects.append(posterior[lookup])
        support_rows.append(support[lookup])
    adjustment = np.mean(np.vstack(effects), axis=0)
    expected = pd.to_numeric(held["prequential_base_anchor_bps"], errors="coerce").to_numpy(float) + adjustment
    # Fit rank domain only on the preceding resolved training population.
    train_expected = pd.to_numeric(train["prequential_base_anchor_bps"], errors="coerce").to_numpy(float) + global_mean
    reference = np.sort(train_expected[np.isfinite(train_expected)], kind="stable")
    ranks = np.searchsorted(reference, expected, side="right") / max(len(reference), 1)
    return pd.DataFrame({
        "bayes_residual_bps": adjustment.astype(np.float32),
        "bayes_expected_bps": expected.astype(np.float32),
        "bayes_rank_traincdf": np.clip(ranks, 0.0, 1.0).astype(np.float32),
        "bayes_effective_support": np.median(np.vstack(support_rows), axis=0).astype(np.float32),
    }), {"selected_fields": list(chosen), "global_residual_mean_bps": global_mean}


def _metric_rows(frame: pd.DataFrame, score: str, label: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    frame = frame.loc[np.isfinite(pd.to_numeric(frame[score], errors="coerce")) & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))].copy()
    for period, block in [("all", frame), *[(str(year), group) for year, group in frame.groupby(frame["__decision_ts__"].dt.year, sort=True)]]:
        rho = float(spearmanr(block[score], block["policy_net_bps"]).statistic) if len(block) > 1 else float("nan")
        for tail in (.005, .01, .02, .05):
            n = max(1, int(math.ceil(tail * len(block))))
            selected = block.nlargest(n, score)
            output.append({"arm": label, "period": period, "tail": tail, "rows": len(block), "selected": n, "net_bps_per_trade": float(selected["policy_net_bps"].mean()), "rank_ic": rho})
    return output


def minimum(left: int, right: int) -> int:
    # Named helper keeps the selection expression readable and prevents an
    # accidental accidental use of a floating top-k threshold.
    return min(left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--top-fraction", type=float, default=.30)
    parser.add_argument("--prior-strength", type=float, default=300.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if not (0.0 < args.top_fraction <= 1.0) or args.train_months < 1 or args.train_cap < 1 or args.prior_strength <= 0:
        raise ValueError("invalid training specification")
    fields = _fields(args.base_contract, args.side)
    columns = ["candidate_id", "__decision_ts__", "side_name", "held_month", "policy_path_valid", "policy_label_available_ts", "policy_net_bps", "prequential_base_anchor_bps", "prequential_base_rank42", "prequential_consensus_rank", "prequential_upstream", *fields]
    frame = pd.read_parquet(args.ledger, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq(args.side).all():
        raise ValueError("ledger identity or side contract failed")
    months = sorted(pd.to_datetime(frame["held_month"] + "-01", utc=True).unique())
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for index, start in enumerate(months):
        start = pd.Timestamp(start)
        end = start + pd.offsets.MonthBegin(1)
        held = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
        prior_start = start - pd.DateOffset(months=args.train_months)
        train_all = frame.loc[
            frame["__decision_ts__"].ge(prior_start) & frame["__decision_ts__"].lt(start)
            & frame["policy_label_available_ts"].lt(start)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce"))
            & np.isfinite(pd.to_numeric(frame["prequential_upstream"], errors="coerce")),
        ].copy()
        result = held.loc[:, [
            "candidate_id", "__decision_ts__", "side_name", "held_month",
            "policy_label_available_ts", "prequential_base_rank42",
            "prequential_consensus_rank", "prequential_upstream", "policy_net_bps",
            "policy_path_valid",
        ]].copy()
        if len(train_all) < 1_000:
            result["bayes_available"] = False
            result["bayes_unavailable_reason"] = "insufficient_prior_resolved_support"
            result["bayes_rank_traincdf"] = result["prequential_base_rank42"]
            result["bayes_expected_bps"] = np.nan
            result["bayes_residual_bps"] = np.nan
            result["bayes_effective_support"] = np.nan
            audits.append({"held_month": start.strftime("%Y-%m"), "status": "warmup", "held_rows": len(held), "train_rows_before_top": len(train_all)})
        else:
            floor = float(np.quantile(train_all["prequential_upstream"], 1.0 - args.top_fraction, method="higher"))
            train = _equal_month_sample(train_all.loc[train_all["prequential_upstream"].ge(floor)].copy(), args.train_cap, seed=SEED + index)
            prediction, audit = _fit_predict(train, held, fields, args.prior_strength)
            result = pd.concat([result.reset_index(drop=True), prediction], axis=1)
            result["bayes_available"] = True
            result["bayes_unavailable_reason"] = None
            audits.append({"held_month": start.strftime("%Y-%m"), "status": "complete", "held_rows": len(held), "train_rows_before_top": len(train_all), "train_rows": len(train), "train_score_floor": floor, **audit})
        for alpha in (0.25, 0.50, 0.75, 1.00):
            result[f"bayes_blend_{alpha:.2f}"] = ((1.0 - alpha) * result["prequential_base_rank42"] + alpha * result["bayes_rank_traincdf"])
        parts.append(result)
        print(json.dumps({"event": "month_complete", **audits[-1]}, default=str), flush=True)
    out = pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(out) != len(frame) or out["candidate_id"].duplicated().any():
        raise AssertionError("candidate identity changed")
    metrics: list[dict[str, object]] = []
    for score in ["prequential_upstream", "prequential_base_rank42", *(f"bayes_blend_{alpha:.2f}" for alpha in (.25, .50, .75, 1.00))]:
        metrics.extend(_metric_rows(out.loc[out["bayes_available"].astype(bool)].copy(), score, score))
    args.out_dir.mkdir(parents=True)
    out.to_parquet(args.out_dir / "bayesian_residual_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "bayesian_residual_fold_audit.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(args.out_dir / "bayesian_residual_metrics.parquet", index=False)
    manifest = {"schema": SCHEMA, "ledger": str(args.ledger), "ledger_sha256": _sha(args.ledger), "base_contract": str(args.base_contract), "base_contract_sha256": _sha(args.base_contract), "side": args.side, "target": "policy_net_bps - prequential_base_anchor_bps", "training": {"months": args.train_months, "cap": args.train_cap, "top_fraction": args.top_fraction, "prior_strength": args.prior_strength, "max_fields": 12}, "score_integration": "diagnostic train-CDF blend only; no admission, sizing, or execution change", "strict_prequential": True, "raw_k9_memberships_used": False, "rows": len(out)}
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
