#!/usr/bin/env python3
"""Ablate causal recent score-to-EV mappings for global execution-EV ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ID_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument(
        "--forward",
        type=Path,
        help=(
            "Optional strictly forward-OOS table to append after the historical "
            "OOF stream. Rows must declare is_oof=false and "
            "promotion_eligible=false."
        ),
    )
    parser.add_argument(
        "--forward-score-col",
        help="Score column in --forward; defaults to --score-col.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=21)
    parser.add_argument("--min-reference-rows", type=int, default=500)
    parser.add_argument("--side-support-target", type=float, default=500.0)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    return parser


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference[np.isfinite(reference)])
    if not len(ordered):
        return np.full(len(values), np.nan)
    return np.searchsorted(ordered, values, side="right") / float(len(ordered))


def _robust_z(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = reference[np.isfinite(reference)]
    if not len(valid):
        return np.full(len(values), np.nan)
    median = float(np.median(valid))
    q25, q75 = np.quantile(valid, [0.25, 0.75])
    scale = max(float(q75 - q25) / 1.349, float(np.std(valid)), 1e-9)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _fit_isotonic(
    reference_score: np.ndarray,
    reference_target: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(reference_score) & np.isfinite(reference_target)
    score = reference_score[valid]
    target = reference_target[valid]
    if len(score) < 2 or np.unique(score).size < 2:
        return np.full(len(values), np.nan)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(score, target)
    return model.predict(values)


def causal_mappings(
    frame: pd.DataFrame,
    *,
    score_col: str,
    window_days: int,
    min_reference_rows: int,
    side_support_target: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = frame.copy()
    out["causal_recent_percentile"] = np.nan
    out["causal_recent_robust_z"] = np.nan
    out["causal_recent_isotonic_ev"] = np.nan
    out["causal_recent_side_isotonic_ev"] = np.nan
    decision = pd.to_datetime(out["execution_decision_utc"], utc=True)
    resolved = pd.to_datetime(out["execution_label_end_utc"], utc=True)
    day = decision.dt.floor("D")
    score = pd.to_numeric(out[score_col], errors="raise").to_numpy(dtype=float)
    target = pd.to_numeric(out["execution_net_ev_12h"], errors="raise").to_numpy(
        dtype=float
    )
    sides = out["side_name"].astype(str).str.lower().to_numpy()
    audit: list[dict[str, object]] = []
    for snapshot, batch_idx in out.groupby(day, sort=True).groups.items():
        snapshot = pd.Timestamp(snapshot)
        batch_pos = out.index.get_indexer(batch_idx)
        eligible = (
            resolved.lt(snapshot)
            & resolved.ge(snapshot - pd.Timedelta(days=int(window_days)))
            & np.isfinite(score)
            & np.isfinite(target)
        ).to_numpy()
        ref_pos = np.flatnonzero(eligible)
        if len(ref_pos) < int(min_reference_rows):
            continue
        ref_score = score[ref_pos]
        ref_target = target[ref_pos]
        current_score = score[batch_pos]
        percentile = _empirical_percentile(ref_score, current_score)
        robust_z = _robust_z(ref_score, current_score)
        global_iso = _fit_isotonic(ref_score, ref_target, current_score)
        side_iso = global_iso.copy()
        side_support: dict[str, int] = {}
        for side in ("long", "short"):
            ref_side = sides[ref_pos] == side
            cur_side = sides[batch_pos] == side
            n_side = int(ref_side.sum())
            side_support[side] = n_side
            if not cur_side.any() or n_side < 2:
                continue
            local = _fit_isotonic(
                ref_score[ref_side],
                ref_target[ref_side],
                current_score[cur_side],
            )
            weight = n_side / (n_side + max(float(side_support_target), 0.0))
            side_iso[cur_side] = (
                weight * local + (1.0 - weight) * global_iso[cur_side]
            )
        out.iloc[
            batch_pos, out.columns.get_loc("causal_recent_percentile")
        ] = percentile
        out.iloc[batch_pos, out.columns.get_loc("causal_recent_robust_z")] = robust_z
        out.iloc[
            batch_pos, out.columns.get_loc("causal_recent_isotonic_ev")
        ] = global_iso
        out.iloc[
            batch_pos, out.columns.get_loc("causal_recent_side_isotonic_ev")
        ] = side_iso
        audit.append(
            {
                "snapshot": snapshot.isoformat(),
                "reference_rows": int(len(ref_pos)),
                "current_rows": int(len(batch_pos)),
                "long_reference_rows": side_support["long"],
                "short_reference_rows": side_support["short"],
            }
        )
    return out, audit


def evaluate(
    frame: pd.DataFrame,
    *,
    score_columns: list[str],
    top_k_fraction: float,
    fold_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for score_col in score_columns:
        eligible = frame[np.isfinite(pd.to_numeric(frame[score_col], errors="coerce"))]
        scopes = [("pooled", eligible)]
        scopes.extend(
            (f"fold_{int(fold)}", group)
            for fold, group in eligible.groupby(fold_col, sort=True)
        )
        for scope, group in scopes:
            n = max(1, int(np.ceil(float(top_k_fraction) * len(group))))
            selected = group.nlargest(n, score_col)
            net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
            rows.append(
                {
                    "mapping": score_col,
                    "scope": scope,
                    "eligible_rows": int(len(group)),
                    "selected_rows": int(len(selected)),
                    "mean_net_ev": float(net.mean()),
                    "mean_net_ev_bps": float(10_000.0 * net.mean()),
                    "positive_rate": float((net > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    oof = pd.read_parquet(args.oof)
    flag_col = f"{args.score_col}__is_oof"
    if flag_col not in oof:
        raise ValueError(f"Missing OOF flag {flag_col!r}")
    oof = oof.loc[oof[flag_col].astype(bool)].copy().reset_index(drop=True)
    handoff = pd.read_parquet(
        args.handoff,
        columns=[
            *ID_COLUMNS,
            "execution_label_end_utc",
            "execution_net_ev_12h",
        ],
    )
    supplement = handoff.drop(columns="execution_net_ev_12h")
    frame = oof.merge(
        supplement, on=list(ID_COLUMNS), how="inner", validate="one_to_one"
    )
    if len(frame) != len(oof):
        raise ValueError("Handoff does not cover every OOF score")
    frame["evaluation_origin"] = "historical_outer_oof"
    frame["promotion_eligible"] = True
    if args.forward is not None:
        forward = pd.read_parquet(args.forward)
        forward_score_col = args.forward_score_col or args.score_col
        required = {
            *ID_COLUMNS,
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_net_ev_12h",
            "is_oof",
            "promotion_eligible",
            forward_score_col,
        }
        missing = sorted(required - set(forward.columns))
        if missing:
            raise ValueError(
                "Forward table is missing required columns: " + ", ".join(missing)
            )
        if forward["is_oof"].astype(bool).any():
            raise ValueError("Forward extension must not contain OOF rows")
        if forward["promotion_eligible"].astype(bool).any():
            raise ValueError("Forward extension must remain non-promotable")
        overlap = frame.loc[:, ID_COLUMNS].merge(
            forward.loc[:, ID_COLUMNS],
            on=list(ID_COLUMNS),
            how="inner",
        )
        if len(overlap):
            raise ValueError(
                f"Forward extension overlaps {len(overlap)} historical OOF identities"
            )
        forward = forward.copy()
        forward[args.score_col] = pd.to_numeric(
            forward[forward_score_col], errors="raise"
        )
        max_fold = int(
            pd.to_numeric(
                frame["execution_ev_model_ablation_oof_fold"], errors="raise"
            ).max()
        )
        forward["execution_ev_model_ablation_oof_fold"] = max_fold + 1
        forward["evaluation_origin"] = "frozen_final_fit_forward_oos"
        keep = sorted(set(frame.columns) & set(forward.columns))
        required_after = {
            *ID_COLUMNS,
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_net_ev_12h",
            "execution_ev_model_ablation_oof_fold",
            "evaluation_origin",
            "promotion_eligible",
            args.score_col,
        }
        missing_after = sorted(required_after - set(keep))
        if missing_after:
            raise ValueError(
                "Historical/forward streams lack common mapping columns: "
                + ", ".join(missing_after)
            )
        frame = pd.concat(
            [frame.loc[:, keep], forward.loc[:, keep]],
            ignore_index=True,
            sort=False,
        )
        frame = frame.sort_values(
            ["execution_decision_utc", "__symbol__", "side_name", "candidate_id"],
            kind="stable",
        ).reset_index(drop=True)
    mapped, audit = causal_mappings(
        frame,
        score_col=args.score_col,
        window_days=args.window_days,
        min_reference_rows=args.min_reference_rows,
        side_support_target=args.side_support_target,
    )
    mapping_columns = [
        args.score_col,
        "causal_recent_percentile",
        "causal_recent_robust_z",
        "causal_recent_isotonic_ev",
        "causal_recent_side_isotonic_ev",
    ]
    leaderboard = evaluate(
        mapped,
        score_columns=mapping_columns,
        top_k_fraction=args.top_k_fraction,
        fold_col="execution_ev_model_ablation_oof_fold",
    )
    is_forward = mapped["evaluation_origin"].eq("frozen_final_fit_forward_oos")
    for column in mapping_columns[1:]:
        available = mapped[column].notna()
        mapped[f"{column}__is_oof"] = available & ~is_forward
        mapped[f"{column}__is_forward_oos"] = available & is_forward
    mapped.to_parquet(args.output_dir / "mapped_oof.parquet", index=False)
    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    report = {
        "schema": "execution_ev_recent_mapping_ablation_v1",
        "contract": {
            "ranking_scope": "global pooled across timestamps and sides",
            "per_timestamp_quota": False,
            "reference": "resolved before each UTC-day snapshot",
            "window_days": int(args.window_days),
            "min_reference_rows": int(args.min_reference_rows),
            "forward_extension": (
                "frozen final-fit forward OOS, explicitly non-promotable"
                if args.forward is not None
                else None
            ),
        },
        "rows_by_origin": {
            str(origin): int(len(group))
            for origin, group in mapped.groupby("evaluation_origin", sort=True)
        },
        "daily_audit": audit,
        "leaderboard": leaderboard.to_dict("records"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return args.output_dir / "leaderboard.csv", args.output_dir / "mapped_oof.parquet"


def main() -> None:
    leaderboard, mapped = run(_parser().parse_args())
    print(f"leaderboard: {leaderboard}")
    print(f"mapped_oof: {mapped}")


if __name__ == "__main__":
    main()
