#!/usr/bin/env python3
"""Strict-OOF HPO for the trailing-activation H4 controller.

This is deliberately a narrow, offline successor to the actuator screen.  It
reuses immutable, exact one-minute labels created by the prior rich-parent
policy.  HPO selection is entirely inside 2025: first a label-only strict
monthly-OOF funnel, then exact constrained portfolio replays for the three
best label candidates.  Only the selected 2025 contender is then fitted on
resolved 2025 labels and evaluated once on the later 2026 confirmation block.

It neither imports nor changes live, admission, portfolio, C1 S/R, geometry,
or MC1 artifacts.  Training may use an unauctioned, paired-MC1 >=40-bps label
route; every reported economic metric is paired-MC1 >=50 bps with the normal
global chronological constrained auction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:  # module execution from the repository root
    from scripts import run_causal_sr_h4_actuator_counterfactual_ablation as base
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    import run_causal_sr_h4_actuator_counterfactual_ablation as base


ROOT = Path("data_perp/artifacts/causal_sr_h4_actuator_counterfactual_2025oof_2026confirm_20260901_v1")
DEFAULT_OUT = Path("data_perp/artifacts/causal_sr_h4_activation_hpo_2025oof_2026confirm_20260901_v1")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    target: str  # direct_080_advantage_bps | tight_advantage_bps
    objective: str
    max_depth: int
    num_leaves: int
    min_child_fraction: float
    reg_lambda: float
    learning_rate: float
    n_estimators: int


CONFIGS = (
    ModelConfig("p0_tight_l2_d3_l7", "tight_advantage_bps", "regression_l2", 3, 7, .05, 40., .035, 280),
    ModelConfig("p1_direct080_l2_d3_l7", "direct_080_advantage_bps", "regression_l2", 3, 7, .05, 40., .035, 280),
    ModelConfig("p2_direct080_l2_d2_l4", "direct_080_advantage_bps", "regression_l2", 2, 4, .10, 60., .030, 320),
    ModelConfig("p3_direct080_huber_d2_l4", "direct_080_advantage_bps", "huber", 2, 4, .10, 80., .025, 320),
    ModelConfig("p4_direct080_huber_d3_l7", "direct_080_advantage_bps", "huber", 3, 7, .05, 40., .035, 280),
    ModelConfig("p5_direct080_l2_d4_l15", "direct_080_advantage_bps", "regression_l2", 4, 15, .05, 80., .025, 320),
)

# A bounded authority grid.  All mappings are tightening-only; Stage 1 found
# essentially no useful widening authority and the rich parent policy already
# owns every other actuator.
MAPPINGS = (
    ("q15_m080", 15.0, .80),
    ("q25_m080", 25.0, .80),
    ("q35_m080", 35.0, .80),
    ("q25_m070", 25.0, .70),
    ("q25_m090", 25.0, .90),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _activation_label_states(states: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join only sampled, resolved activation labels to target-free state rows."""
    labels = labels.loc[labels["actuator"].eq("activation")].copy()
    key = ["candidate_id", "state_decision_ts"]
    pivot = labels.pivot(index=key, columns="multiplier", values="advantage_bps").reset_index()
    wanted = {float(v) for v in base.MULTIPLIERS}
    got = {float(v) for v in pivot.columns if isinstance(v, float)}
    if got != wanted:
        raise AssertionError("activation counterfactual multiplier receipt is incomplete")
    pivot["direct_080_advantage_bps"] = pd.to_numeric(pivot[.8], errors="raise")
    pivot["tight_advantage_bps"] = pivot[[.65, .8]].max(axis=1)
    availability = labels.loc[:, [*key, "policy_label_available_ts"]].drop_duplicates(key)
    if availability.duplicated(key).any():
        raise AssertionError("label availability is not unique")
    result = states.merge(pivot.loc[:, [*key, "direct_080_advantage_bps", "tight_advantage_bps"]], on=key, how="inner", validate="one_to_one")
    result = result.merge(availability, on=key, how="inner", validate="one_to_one")
    if result.empty or result["candidate_id"].nunique() < 250:
        raise RuntimeError("insufficient activation label/state overlap")
    return result


def _fit(frame: pd.DataFrame, fields: tuple[str, ...], config: ModelConfig) -> lgb.LGBMRegressor:
    child = max(64, int(np.ceil(len(frame) * float(config.min_child_fraction))))
    kwargs: dict[str, object] = {
        "objective": config.objective,
        "n_estimators": int(config.n_estimators),
        "learning_rate": float(config.learning_rate),
        "max_depth": int(config.max_depth),
        "num_leaves": int(config.num_leaves),
        "min_child_samples": child,
        "subsample": .80,
        "colsample_bytree": .80,
        "reg_lambda": float(config.reg_lambda),
        "random_state": 1729,
        "n_jobs": 2,
        "verbosity": -1,
    }
    # Huber's alpha is retained at LightGBM's documented robust default.
    if config.objective == "huber":
        kwargs["alpha"] = .90
    model = lgb.LGBMRegressor(**kwargs)
    weights = 1.0 / frame.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(frame.loc[:, fields], frame[config.target].to_numpy(float), sample_weight=weights)
    return model


def _strict_schedule(
    *, labels: pd.DataFrame, target_states: pd.DataFrame, fields: tuple[str, ...], config: ModelConfig,
    train_start: pd.Timestamp, held_start: pd.Timestamp, held_end: pd.Timestamp,
) -> pd.DataFrame:
    """Predict a held month using only earlier fully-resolved label rows."""
    test = target_states.loc[
        target_states["entry_decision_ts"].ge(held_start) & target_states["entry_decision_ts"].lt(held_end)
    ].copy()
    result = test.loc[:, ["candidate_id", "state_decision_ts"]].copy()
    result["prediction_bps"] = 0.0
    if test.empty:
        return result
    train = labels.loc[
        labels["entry_decision_ts"].ge(train_start) & labels["entry_decision_ts"].lt(held_start)
        & labels["policy_label_available_ts"].lt(held_start)
    ].copy()
    if train["candidate_id"].nunique() < 250:
        return result
    model = _fit(train, fields, config)
    result["prediction_bps"] = model.predict(test.loc[:, fields])
    return result


def _oof_predictions(
    *, labels: pd.DataFrame, target_states: pd.DataFrame, fields: tuple[str, ...], config: ModelConfig,
    start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for period in pd.period_range(start, end - pd.offsets.MonthBegin(1), freq="M"):
        held = pd.Timestamp(period.start_time, tz="UTC")
        one = _strict_schedule(
            labels=labels, target_states=target_states, fields=fields, config=config,
            train_start=start, held_start=held, held_end=held + pd.offsets.MonthBegin(1),
        )
        one["held_month"] = held.strftime("%Y-%m")
        pieces.append(one)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _label_score(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float | int]:
    key = ["candidate_id", "state_decision_ts"]
    frame = predictions.merge(labels.loc[:, [*key, "direct_080_advantage_bps"]], on=key, how="inner", validate="one_to_one")
    if frame.empty:
        return {"rows": 0, "spearman": float("nan"), "top_decile_advantage_bps": float("nan"), "top_quintile_advantage_bps": float("nan")}
    count10 = max(1, int(np.ceil(len(frame) * .10)))
    count20 = max(1, int(np.ceil(len(frame) * .20)))
    ordered = frame.sort_values("prediction_bps", ascending=False, kind="stable")
    return {
        "rows": int(len(frame)),
        "spearman": float(frame["prediction_bps"].corr(frame["direct_080_advantage_bps"], method="spearman")),
        "top_decile_advantage_bps": float(ordered.head(count10)["direct_080_advantage_bps"].mean()),
        "top_quintile_advantage_bps": float(ordered.head(count20)["direct_080_advantage_bps"].mean()),
    }


def _schedule_mapping(predictions: pd.DataFrame, *, threshold: float, multiplier: float) -> pd.DataFrame:
    frame = predictions.loc[:, ["candidate_id", "state_decision_ts"]].copy()
    frame["actuator"] = "activation"
    frame["multiplier"] = np.where(predictions["prediction_bps"].to_numpy(float) >= float(threshold), float(multiplier), 1.0)
    return frame


def _evaluate_arm(
    *, rows: pd.DataFrame, arrays: dict[str, np.ndarray], route: pd.DataFrame, params: object, median: float,
    schedule: pd.DataFrame | None, arm: str, output: Path,
) -> dict[str, float | int | str]:
    outcome = base._replay(rows, arrays, route, params, median, schedule, "activation" if schedule is not None else None)
    candidates, decisions, accepted, equity, metrics = base._portfolio(outcome, arm)
    if not accepted.empty:
        accepted = accepted.copy()
        indices = pd.to_numeric(accepted["candidate_index"], errors="raise").astype(int).to_numpy()
        accepted["holding_bars"] = candidates.iloc[indices]["holding_bars"].to_numpy()
    extra = base._extra_metrics(accepted)
    outcome.to_parquet(output / f"{arm}_exact1m_outcomes.parquet", index=False, compression="zstd")
    accepted.to_parquet(output / f"{arm}_portfolio_accepted.parquet", index=False, compression="zstd")
    decisions.to_parquet(output / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(output / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    del outcome, candidates, decisions, accepted, equity
    gc.collect()
    return {"arm": arm, **metrics, **extra}


def _monthly(output: Path, arms: list[str], name: str) -> None:
    frames: list[pd.DataFrame] = []
    for arm in arms:
        path = output / f"{arm}_portfolio_accepted.parquet"
        if not path.exists():
            continue
        data = pd.read_parquet(path)
        if data.empty:
            continue
        data["month"] = pd.to_datetime(data["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
        one = data.groupby("month", as_index=False).agg(
            trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum"),
        )
        one.insert(0, "arm", arm)
        frames.append(one)
    (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_parquet(output / name, index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, default=ROOT)
    parser.add_argument("--parent-root", type=Path, default=base.DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATES)
    parser.add_argument("--policy", type=Path, default=base.DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    label_root, parent_root, state_root, policy_path = (args.label_root.resolve(), args.parent_root.resolve(), args.state_root.resolve(), args.policy.resolve())
    route, rows, _, arrays, states = base._load_parent(parent_root, state_root)
    params, median, _ = base._load_policy(policy_path)
    fields = base._state_fields(states)
    raw_labels = pd.read_parquet(label_root / "exact_counterfactual_labels_training_only.parquet")
    raw_labels["candidate_id"] = raw_labels["candidate_id"].astype(str)
    raw_labels["state_decision_ts"] = pd.to_datetime(raw_labels["state_decision_ts"], utc=True, errors="raise")
    raw_labels["policy_label_available_ts"] = pd.to_datetime(raw_labels["policy_label_available_ts"], utc=True, errors="raise")
    label_states = _activation_label_states(states, raw_labels)
    train_start, train_end = _month("2025-06"), _month("2026-01")
    validate_start, validate_end = _month("2026-06"), _month("2026-09")
    normal = route.loc[(route["bcf_mc1_expected_bps"] >= 50.0) & (route["current_mc1_expected_bps"] >= 50.0)].copy()
    normal_2025 = normal.loc[normal["timestamp"].ge(train_start) & normal["timestamp"].lt(train_end)].copy()
    normal_2026 = normal.loc[normal["timestamp"].ge(validate_start) & normal["timestamp"].lt(validate_end)].copy()
    exact_ids = set(rows["candidate_id"].astype(str))
    if not set(normal_2025["candidate_id"]).issubset(exact_ids) or not set(normal_2026["candidate_id"]).issubset(exact_ids):
        raise AssertionError("normal evaluation route contains an incomplete exact path")
    normal_state_2025 = states.loc[states["candidate_id"].isin(set(normal_2025["candidate_id"]))].copy()
    normal_state_2026 = states.loc[states["candidate_id"].isin(set(normal_2026["candidate_id"]))].copy()
    if normal_state_2025.empty or normal_state_2026.empty:
        raise RuntimeError("normal route has no target-free H4 state coverage")
    # Parent equivalence: controller with no schedule is the stored exact rich
    # parent policy, before any model fit or selection takes place.
    probe = normal_2025.head(64)
    replay = base._replay(rows, arrays, probe, params, median, None, None).set_index("candidate_id")["exact_net_bps"]
    archived = pd.read_parquet(parent_root / "exact_1m_rich_parent_outcomes.parquet").set_index("candidate_id").loc[replay.index, "exact_net_bps"]
    if not np.allclose(replay.to_numpy(float), archived.to_numpy(float), atol=1e-8, rtol=0.0):
        raise AssertionError("HPO parent replay is not exact-policy equivalent")
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"position": range(len(fields)), "feature": fields}).to_parquet(out / "existing_target_free_feature_contract.parquet", index=False, compression="zstd")
    label_summary: list[dict[str, object]] = []
    for config in CONFIGS:
        pred = _oof_predictions(labels=label_states, target_states=label_states, fields=fields, config=config, start=train_start, end=train_end)
        score = _label_score(pred, label_states)
        label_summary.append({"config": config.name, **asdict(config), **score})
        pred.to_parquet(out / f"label_oof_{config.name}.parquet", index=False, compression="zstd")
    label_table = pd.DataFrame(label_summary).sort_values(["top_decile_advantage_bps", "spearman", "top_quintile_advantage_bps"], ascending=False, kind="stable")
    label_table.to_parquet(out / "2025_label_oof_hpo_summary.parquet", index=False, compression="zstd")
    finalists = tuple(label_table.head(3)["config"].astype(str))
    lookup = {config.name: config for config in CONFIGS}
    # Recreate only the finalists' normal-route schedules.  This prevents
    # label screening from quietly using the portfolio outcome as a filter.
    schedule_records: list[dict[str, object]] = []
    for name in finalists:
        config = lookup[name]
        pred = _oof_predictions(labels=label_states, target_states=normal_state_2025, fields=fields, config=config, start=train_start, end=train_end)
        pred.to_parquet(out / f"2025_oof_normal_state_predictions_{name}.parquet", index=False, compression="zstd")
        for mapping, threshold, multiplier in MAPPINGS:
            schedule = _schedule_mapping(pred, threshold=threshold, multiplier=multiplier)
            schedule_path = out / f"2025_oof_schedule_{name}__{mapping}.parquet"
            schedule.to_parquet(schedule_path, index=False, compression="zstd")
            schedule_records.append({"config": name, "mapping": mapping, "threshold_bps": threshold, "multiplier": multiplier, "schedule_path": schedule_path})
            del schedule
        del pred
        gc.collect()
    parent_record = _evaluate_arm(rows=rows, arrays=arrays, route=normal_2025, params=params, median=median, schedule=None, arm="parent", output=out)
    exact_records: list[dict[str, object]] = [parent_record]
    for record in schedule_records:
        arm = f"{record['config']}__{record['mapping']}"
        schedule = pd.read_parquet(record["schedule_path"])
        summary = _evaluate_arm(rows=rows, arrays=arrays, route=normal_2025, params=params, median=median, schedule=schedule, arm=arm, output=out)
        changed = int(schedule["multiplier"].ne(1.0).sum())
        del schedule
        gc.collect()
        exact_records.append({**summary, "config": record["config"], "mapping": record["mapping"], "threshold_bps": record["threshold_bps"], "multiplier": record["multiplier"], "scheduled_state_actions": changed})
    exact = pd.DataFrame(exact_records)
    ref = exact.loc[exact["arm"].eq("parent")].iloc[0]
    for field in ("net_bps_per_trade", "total_net_bps", "sortino", "max_drawdown", "worst_week", "cvar10_bps", "worst_month_bps"):
        exact[f"delta_vs_parent_{field}"] = exact[field] - ref[field]
    exact["total_bps_per_abs_drawdown"] = exact["total_net_bps"] / exact["max_drawdown"].abs().clip(lower=1e-9)
    exact.to_parquet(out / "2025_exact_portfolio_hpo_summary.parquet", index=False, compression="zstd")
    _monthly(out, exact["arm"].astype(str).tolist(), "2025_exact_portfolio_monthly_metrics.parquet")
    contenders = exact.loc[exact["arm"].ne("parent")].sort_values(
        ["total_bps_per_abs_drawdown", "net_bps_per_trade", "worst_week"], ascending=False, kind="stable",
    )
    winner = contenders.iloc[0]
    winner_config = lookup[str(winner["config"])]
    # Freeze the selected 2025 configuration and authority; fit only labels
    # whose 12-hour outcome had resolved before the 2026 validation origin.
    frozen_states = normal_state_2026.copy()
    frozen_pred = _strict_schedule(
        labels=label_states, target_states=frozen_states, fields=fields, config=winner_config,
        train_start=train_start, held_start=train_end, held_end=validate_end,
    )
    frozen_schedule = _schedule_mapping(frozen_pred, threshold=float(winner["threshold_bps"]), multiplier=float(winner["multiplier"]))
    frozen_schedule.to_parquet(out / "2026_frozen_winner_schedule.parquet", index=False, compression="zstd")
    validation_parent = _evaluate_arm(rows=rows, arrays=arrays, route=normal_2026, params=params, median=median, schedule=None, arm="2026_parent", output=out)
    validation_winner = _evaluate_arm(rows=rows, arrays=arrays, route=normal_2026, params=params, median=median, schedule=frozen_schedule, arm="2026_frozen_winner", output=out)
    validation = pd.DataFrame([validation_parent, validation_winner])
    ref = validation.loc[validation["arm"].eq("2026_parent")].iloc[0]
    for field in ("net_bps_per_trade", "total_net_bps", "sortino", "max_drawdown", "worst_week", "cvar10_bps", "worst_month_bps"):
        validation[f"delta_vs_parent_{field}"] = validation[field] - ref[field]
    validation.to_parquet(out / "2026_frozen_portfolio_confirmation_summary.parquet", index=False, compression="zstd")
    _monthly(out, validation["arm"].astype(str).tolist(), "2026_frozen_portfolio_monthly_metrics.parquet")
    manifest = {
        "schema": "causal-sr-h4-activation-hpo-v1",
        "scope": "offline research only; no live, exchange, admission, portfolio, or canonical-policy mutation",
        "parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"),
        "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json"),
        "label_root": str(label_root), "label_receipt_sha256": _sha256(label_root / "exact_counterfactual_labels_training_only.parquet"),
        "policy": str(policy_path), "policy_sha256": _sha256(policy_path),
        "source_contract": "causal S/R plus paired BCF/current-v5 MC1 long-only exact rich parent",
        "training_population": "paired MC1 >=40 bps, unauctioned label-only sampling, first/middle/last target-free states, fully resolved H12 outcomes only",
        "assessment_population": "paired MC1 >=50 bps, normal global chronological constrained portfolio auction",
        "selection": "2025-06 through 2025-12 monthly strict-prior OOF; label-only funnel precedes exact portfolio selection",
        "confirmation": "one frozen June-August 2026 confirmation; selection model and authority fitted from resolved 2025 only",
        "features": "all existing numeric target-free H4 state fields; no later feature selection receipt",
        "controller": "trailing activation only; tightening-only mappings; action applies next completed-15m interval",
        "model_configs": [asdict(config) for config in CONFIGS],
        "authority_grid": [{"mapping": name, "threshold_bps": threshold, "multiplier": multiplier} for name, threshold, multiplier in MAPPINGS],
        "label_finalists": list(finalists), "selected_2025_winner": winner.to_dict(),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
