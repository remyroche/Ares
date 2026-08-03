#!/usr/bin/env python3
"""Run a matched four-arm 2024 regime/transition economic ablation.

All arms use the same reconstructed candidate identities and the same one
pooled global top-k after a causal, recent-EV mapping.  Regime and transition
are soft, independently OOF contexts; they are never collapsed into one state.
Action/timing heads are intentionally outside this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
DEFAULT_STATES = ROOT / "data_perp/artifacts/reconstructed_2024_candidate_oof_regime_transition_20260730_v1/candidate_oof_regime_transition.parquet"
DEFAULT_LABELS = ROOT / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/full2024_matched_regime_transition_economic_ablation_20260730_v1"
SCHEMA = "full2024_matched_regime_transition_economic_ablation_v1"
ARMS = ("baseline", "regime_only", "transition_only", "regime_plus_transition")
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
ACTION_TOKENS = ("timing", "wait", "target_price", "mae_head", "action", "entry_price")
TOP_FRACTION = 0.10
WARMUP_DAYS = 90
RIDGE_ALPHA = 30.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def arm_features(columns: Iterable[str]) -> dict[str, list[str]]:
    names = list(columns)
    regime = [name for name in names if name.startswith("regime_state_p__")] + [
        name for name in ("regime_state_entropy", "regime_state_margin", "regime_state_uncertainty", "regime_state_ood_score") if name in names
    ]
    transition = [name for name in names if name.startswith("transition_state_p__")] + [
        name for name in ("transition_active_probability", "transition_state_entropy", "transition_state_margin", "transition_state_uncertainty", "transition_state_ood_score") if name in names
    ]
    bad = [name for name in [*regime, *transition] if any(token in name.lower() for token in ACTION_TOKENS)]
    if bad:
        raise ValueError(f"action-head fields are forbidden from economic ablation: {bad}")
    common = ["score_residual_expected_ev", "side_is_long"]
    return {
        "baseline": common,
        "regime_only": [*common, *regime],
        "transition_only": [*common, *transition],
        "regime_plus_transition": [*common, *regime, *transition],
    }


def load_matched_panel(*, scores_path: Path, states_path: Path, labels_path: Path) -> pd.DataFrame:
    score_columns = [*IDENTITY, "__reconstructed_soft_alpha_12h__", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "score_residual_expected_ev", "residual_is_oof"]
    scores = pd.read_parquet(scores_path, columns=score_columns).copy()
    scores["__ts__"] = pd.to_datetime(scores["__ts__"], utc=True, errors="coerce")
    scores = scores.loc[scores["__ts__"].dt.year.eq(2024)].copy()
    states = pd.read_parquet(states_path).copy()
    labels = pd.read_parquet(labels_path, columns=[*IDENTITY, "__opportunity_occurred_12h__"]).copy()
    for frame in (states, labels):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    for name, frame in (("scores", scores), ("states", states), ("labels", labels)):
        if frame.duplicated(list(IDENTITY)).any() or frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} does not have unique candidate identity")
    panel = scores.merge(states, on=list(IDENTITY), how="inner", validate="one_to_one").merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(panel) != len(scores) or len(panel) != len(states) or len(panel) != len(labels):
        raise ValueError("score, state, and economic label identities must match exactly")
    if not panel["residual_is_oof"].astype(bool).all():
        raise ValueError("matched score source contains non-OOF residual rows")
    if (panel["regime_available_utc"] > panel["__ts__"]).any() or (panel["transition_available_utc"] > panel["__ts__"]).any():
        raise ValueError("soft state availability is after the candidate decision context")
    if (panel["regime_train_end_utc"] >= panel["__ts__"]).any() or (panel["transition_train_end_utc"] >= panel["__ts__"]).any():
        raise ValueError("soft state fold uses candidate-period future context")
    panel["side_is_long"] = panel["side_name"].astype(str).eq("long").astype(float)
    return panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def causal_recent_ev_mapping(panel: pd.DataFrame, *, lookback_days: int = WARMUP_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each arm through only prior resolved 12h EV labels, month by month."""

    output = panel.copy()
    features = arm_features(output.columns)
    fold_rows: list[dict[str, Any]] = []
    month_starts = pd.date_range("2024-01-01", "2025-01-01", freq="MS", tz="UTC")
    for start, end in zip(month_starts[:-1], month_starts[1:]):
        evaluation = output["__ts__"].ge(start) & output["__ts__"].lt(end)
        train = (
            output["__ts__"].lt(start - pd.Timedelta(hours=12))
            & output["__ts__"].ge(start - pd.Timedelta(days=int(lookback_days)))
        )
        if not evaluation.any():
            continue
        # January has no in-2024 resolved economic history.  Every arm is
        # deliberately equal to the frozen reconstructed residual score there,
        # rather than allowing one arm an external/non-matched prior.
        for arm, columns in features.items():
            score_column = f"mapped_score__{arm}"
            if int(train.sum()) < 100:
                output.loc[evaluation, score_column] = output.loc[evaluation, "score_residual_expected_ev"]
                fold_rows.append({"arm": arm, "fold_month": start.strftime("%Y-%m"), "mode": "cold_start_raw_residual_score", "train_rows": int(train.sum()), "eval_rows": int(evaluation.sum()), "train_end_utc": None})
                continue
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA)),
            ])
            model.fit(output.loc[train, columns], output.loc[train, "execution_net_ev_12h"])
            output.loc[evaluation, score_column] = model.predict(output.loc[evaluation, columns])
            fold_rows.append({"arm": arm, "fold_month": start.strftime("%Y-%m"), "mode": "causal_trailing_90d_ridge", "train_rows": int(train.sum()), "eval_rows": int(evaluation.sum()), "train_end_utc": output.loc[train, "__ts__"].max()})
    score_columns = [f"mapped_score__{arm}" for arm in ARMS]
    if output[score_columns].isna().any().any():
        raise ValueError("causal mapping left unmatched candidate scores")
    return output, pd.DataFrame.from_records(fold_rows)


def pooled_global_top_mask(frame: pd.DataFrame, score: pd.Series, *, fraction: float = TOP_FRACTION) -> pd.Series:
    count = int(np.ceil(len(frame) * float(fraction)))
    if count < 1:
        raise ValueError("global top-k has no selected rows")
    order = pd.DataFrame({"score": pd.to_numeric(score, errors="coerce"), "candidate_id": frame["candidate_id"].astype(str)}, index=frame.index)
    if order["score"].isna().any():
        raise ValueError("cannot rank non-finite mapped scores")
    selected_index = order.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(count).index
    return frame.index.isin(selected_index)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    a, b = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
    valid = a.notna() & b.notna()
    return float(a.loc[valid].rank().corr(b.loc[valid].rank(), method="pearson")) if valid.sum() >= 3 else float("nan")


def _selected_economics(frame: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    local = frame.loc[selected]
    if local.empty:
        return {"selected_rows": 0}
    result = {
        "selected_rows": int(len(local)),
        "gross_ev_bps": float(local["execution_gross_ev_12h"].mean() * 10_000),
        "net_ev_bps": float(local["execution_net_ev_12h"].mean() * 10_000),
        "cost_bps": float(local["execution_cost_return"].mean() * 10_000),
        "positive_net_fraction": float(local["execution_net_ev_12h"].gt(0).mean()),
        "opportunity_hit_rate": float(pd.to_numeric(local["__opportunity_occurred_12h__"], errors="coerce").mean()),
        "selected_support_hours": int(local["__ts__"].nunique()),
    }
    return result


def summarize_ablation(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    for arm in ARMS:
        score = mapped[f"mapped_score__{arm}"]
        selected = pooled_global_top_mask(mapped, score)
        economic = _selected_economics(mapped, selected)
        aggregate.append({
            "arm": arm, "rows": int(len(mapped)), "global_top_fraction": TOP_FRACTION,
            "alpha_rank_ic": _spearman(score, mapped["__reconstructed_soft_alpha_12h__"]),
            "execution_rank_ic": _spearman(score, mapped["execution_net_ev_12h"]), **economic,
        })
        week_start = mapped["__ts__"].dt.floor("D") - pd.to_timedelta(mapped["__ts__"].dt.dayofweek, unit="D")
        for frequency, keys in (("week", week_start), ("month", mapped["__ts__"].dt.strftime("%Y-%m"))):
            local = mapped.loc[selected].copy()
            local["period"] = keys.loc[selected].astype(str).to_numpy()
            for period, group in local.groupby("period", sort=True):
                periods.append({"arm": arm, "frequency": frequency, "period": period, **_selected_economics(group, pd.Series(True, index=group.index))})
        for side, group in mapped.loc[selected].groupby("side_name", sort=True):
            sides.append({"arm": arm, "side_name": side, **_selected_economics(group, pd.Series(True, index=group.index))})
    aggregate_frame = pd.DataFrame.from_records(aggregate)
    period_frame = pd.DataFrame.from_records(periods)
    distribution = []
    for (arm, frequency), group in period_frame.groupby(["arm", "frequency"], sort=True):
        ev = group["net_ev_bps"]
        distribution.append({"arm": arm, "frequency": frequency, "periods_with_global_selections": int(len(group)), "net_ev_bps_q10": float(ev.quantile(0.10)), "net_ev_bps_q50": float(ev.quantile(0.50)), "positive_period_fraction": float(ev.gt(0).mean()), "latest_period": str(group["period"].max()), "latest_period_net_ev_bps": float(group.loc[group["period"].eq(group["period"].max()), "net_ev_bps"].iloc[0])})
    worst = period_frame.sort_values(["arm", "frequency", "net_ev_bps", "period"], kind="stable").groupby(["arm", "frequency"], as_index=False, group_keys=False).head(5)
    return aggregate_frame, period_frame, pd.DataFrame.from_records(distribution), pd.DataFrame.from_records(sides), worst


def gate_winner(aggregate: pd.DataFrame, distribution: pd.DataFrame) -> pd.DataFrame:
    baseline = aggregate.set_index("arm").loc["baseline"]
    baseline_week_q10 = distribution.loc[(distribution.arm == "baseline") & (distribution.frequency == "week"), "net_ev_bps_q10"].iloc[0]
    records = []
    for row in aggregate.itertuples(index=False):
        if row.arm == "baseline":
            continue
        q10 = distribution.loc[(distribution.arm == row.arm) & (distribution.frequency == "week"), "net_ev_bps_q10"].iloc[0]
        passed = bool(
            row.net_ev_bps >= baseline.net_ev_bps + 5.0
            and row.execution_rank_ic > baseline.execution_rank_ic
            and q10 >= baseline_week_q10
        )
        records.append({"arm": row.arm, "net_ev_uplift_bps_vs_baseline": row.net_ev_bps - baseline.net_ev_bps, "execution_ic_uplift": row.execution_rank_ic - baseline.execution_rank_ic, "week_q10_delta_bps": q10 - baseline_week_q10, "passes_economic_gate": passed})
    return pd.DataFrame.from_records(records)


def materialize_ablation(*, scores_path: Path = DEFAULT_SCORES, states_path: Path = DEFAULT_STATES, labels_path: Path = DEFAULT_LABELS, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    scores_path, states_path, labels_path, output_dir = map(Path, (scores_path, states_path, labels_path, output_dir))
    panel = load_matched_panel(scores_path=scores_path, states_path=states_path, labels_path=labels_path)
    mapped, folds = causal_recent_ev_mapping(panel)
    aggregate, period, distribution, side, selected_or_worst = summarize_ablation(mapped)
    # `selected_or_worst` is the intentionally compact worst-period table.
    worst = selected_or_worst
    gates = gate_winner(aggregate, distribution)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapped_path = output_dir / "mapped_global_topk_scores.parquet"
    fold_path = output_dir / "causal_mapping_folds.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    period_path = output_dir / "global_selection_period_metrics.csv"
    distribution_path = output_dir / "period_distribution.csv"
    side_path = output_dir / "global_selection_side_metrics.csv"
    worst_path = output_dir / "worst_global_selection_periods.csv"
    gates_path = output_dir / "winner_gates.csv"
    mapped.to_parquet(mapped_path, index=False)
    folds.to_csv(fold_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    period.to_csv(period_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    side.to_csv(side_path, index=False)
    worst.to_csv(worst_path, index=False)
    gates.to_csv(gates_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "research_only": True,
        "promotion_eligible": False,
        "signed_by": "codex_root_matched_economic_ablation",
        "signature_type": "detached_sha256_manifest",
        "arms": list(ARMS),
        "contract": {
            "candidate_composition": "exact matched 2024 identity across every arm",
            "state_inputs": "completed candidate OOF soft regime and transition states, independently preserved",
            "learned_mapping": "trailing 90d causal Ridge; training signal timestamp < evaluation month start - 12h; January is identical raw-score cold start",
            "selection": "one pooled global top10 across all matched candidates after each causal map; never timestamp/side/state local",
            "action_heads": "forbidden; timing, MAE, target-price and wait actions remain outside this ablation",
            "portfolio_replay": "not run; only permitted for a challenger passing every recorded economic gate",
        },
        "sources": {str(path): _sha256(path) for path in (scores_path, states_path, labels_path)},
        "counts": {"matched_rows": int(len(mapped)), "matched_months": int(mapped["__ts__"].dt.strftime("%Y-%m").nunique()), "portfolio_replay_ran": False, "gate_winners": int(gates["passes_economic_gate"].sum())},
        "outputs_sha256": {},
    }
    for path in (mapped_path, fold_path, aggregate_path, period_path, distribution_path, side_path, worst_path, gates_path):
        manifest["outputs_sha256"][path.name] = _sha256(path)
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    (output_dir / "manifest.sha256").write_text(_sha256(manifest_path) + "  manifest.json\n", encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(_safe(materialize_ablation(scores_path=args.scores, states_path=args.states, labels_path=args.labels, output_dir=args.output_dir)), indent=2, sort_keys=True))
