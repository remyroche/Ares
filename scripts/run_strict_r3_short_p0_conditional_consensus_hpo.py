#!/usr/bin/env python3
"""Sequential, ensemble-only HPO for the short P0/F90 consensus heads.

This is deliberately not a standalone-specialist search.  CMI field
membership is frozen on October--December 2024 by
``build_strict_r3_short_p0_consensus_contract.py``.  This runner then uses
only the predeclared 2025-Q1 chronological OOF development months to choose
each head's residual ordinalisation, LambdaRank query geometry, weighting and
parameters by its incremental contribution to the *base + current ensemble*
score.  April 2025 onward is untouched by this selection process.

The result is an immutable v2 HPO contract.  It is not a promoted consensus
until ``run_strict_r3_short_p0_consensus_oof.py`` has refit and assessed it on
the subsequent strict OOS window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_lambdarank_hpo import (  # noqa: E402
    conditional_downstream_summary,
    materialize_lambdarank_params,
    stop_after_no_improvement,
    suggest_broad_lambdarank_params,
)
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    ConsensusHeadSpec,
    _fit_consensus_head,
    _query,
)
from extreme_price_movements.strict_r3_canonical_v2 import load_geometry_bundle  # noqa: E402


SIDE = "short"
TARGET_EDGE_OPTIONS: dict[str, tuple[float, ...]] = {
    "narrow_100_25": (-100.0, -25.0, 25.0, 100.0),
    "canonical_150_50": (-150.0, -50.0, 50.0, 150.0),
    "wide_200_50": (-200.0, -50.0, 50.0, 200.0),
}
QUERY_OPTIONS = ("exact_timestamp_side", "cycle_4h_side")
WEIGHT_OPTIONS = ("ordinary", "equal_month")
NO_IMPROVEMENT_PATIENCE = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        digest.update(str(value.relative_to(path) if path.is_dir() else value.name).encode())
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left"))


def _month_part(root: Path, month: pd.Timestamp) -> Path:
    return root / "ledger" / f"month={month:%Y-%m}" / "prequential_base_ledger.parquet"


def _load_monthly_ledgers(
    roots: list[Path], *, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[Path] = []
    for month in _month_starts(start, end):
        matches = [path for root in roots if (path := _month_part(root, month)).exists()]
        if len(matches) > 1:
            raise ValueError(f"overlapping short ledger sources for {month:%Y-%m}: {matches}")
        if matches:
            parts.append(matches[0])
    if not parts:
        raise FileNotFoundError("no short ledger parts in requested HPO range")
    out = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    if out["candidate_id"].duplicated().any():
        raise ValueError("short HPO ledger has duplicate candidate identities")
    for column in ("__decision_ts__", "policy_label_available_at"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="raise")
    observed = out["side_name"].astype(str).str.lower()
    if not observed.eq(SIDE).all():
        raise ValueError("short consensus HPO received non-short rows")
    return out


def _valid_base(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["base_feature_eligible"].fillna(False).astype(bool)
        & frame["stack_is_prequential"].fillna(False).astype(bool)
        & pd.to_numeric(frame["prequential_base_rank42"], errors="coerce").notna()
        & pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce").notna()
    )


def _valid_policy(frame: pd.DataFrame, *, cutoff: pd.Timestamp | None = None) -> pd.Series:
    valid = (
        _valid_base(frame)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["p0_canonical_net_bps"], errors="coerce").notna()
    )
    if cutoff is not None:
        valid &= frame["policy_label_available_at"].lt(cutoff)
    return valid


def _grade(values: pd.Series, edges: tuple[float, ...]) -> np.ndarray:
    if len(edges) != 4:
        raise ValueError("residual ordinal target must have four edges")
    value = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.select([value <= edge for edge in edges], [0, 1, 2, 3], default=4).astype(np.int8)


def _with_frozen_geometry(
    frame: pd.DataFrame, *, geometry: Any, required: set[str],
) -> pd.DataFrame:
    """Add only stable target-free Geometry/K9 state fields once per ledger."""
    absent_from_raw = required.difference(frame.columns)
    if absent_from_raw:
        state = geometry.transform(frame).reset_index(drop=True)
        # Cluster IDs are intentionally not included by the field contract.
        state = state.loc[:, [c for c in state if not c.startswith("k09__cluster_")]]
        required_state = absent_from_raw.intersection(state.columns)
        missing = absent_from_raw.difference(required_state)
        if missing:
            raise KeyError(f"frozen short Geometry/K9 lacks contract fields: {sorted(missing)}")
        return pd.concat([frame.reset_index(drop=True), state], axis=1)
    return frame


def _sample_complete_queries(
    frame: pd.DataFrame, spec: ConsensusHeadSpec, *, cap: int, seed: int,
) -> pd.DataFrame:
    """Subsample whole queries for HPO; never split a LambdaRank query."""
    if cap < 1_000:
        raise ValueError("HPO query cap must be at least 1,000 rows")
    # Query selection runs once per fold and Optuna trial.  The frozen ledger
    # is wide, while sampling needs only identity, timestamp and side.  Avoiding a
    # complete feature-frame copy prevents memory amplification without
    # changing the sampled query identities or chronology.
    sample_columns = ["candidate_id", "__decision_ts__", "side_name"]
    if "__month__" in frame:
        sample_columns.append("__month__")
    work = frame.loc[:, sample_columns].copy()
    work["__query__"] = _query(work, spec.query).to_numpy()
    # ``run`` materialises this once across the fixed ledger.  Retain a
    # self-contained fallback for direct unit callers, never per-trial full
    # datetime formatting in the production HPO loop.
    if "__month__" not in work:
        work["__month__"] = work["__decision_ts__"].dt.to_period("M").astype(str)
    counts = work["__query__"].value_counts()
    work = work.loc[work["__query__"].map(counts).ge(2)].copy()
    if len(work) <= cap:
        selected_index = work.sort_values(
            ["__query__", "__decision_ts__", "candidate_id"], kind="stable",
        ).index
        return frame.loc[selected_index].reset_index(drop=True)
    meta = (
        work.groupby("__query__", sort=False)
        .agg(rows=("candidate_id", "size"), month=("__month__", "first"), first=("__decision_ts__", "min"))
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    if spec.weight_mode == "equal_month":
        allowance = max(2, cap // max(1, meta["month"].nunique()))
        for month, group in meta.groupby("month", sort=True):
            del month
            used = 0
            group = group.assign(_rand=rng.random(len(group))).sort_values(["_rand", "first", "__query__"], kind="stable")
            for row in group.to_dict("records"):
                if used + int(row["rows"]) <= allowance:
                    keep.append(str(row["__query__"]))
                    used += int(row["rows"])
    else:
        meta = meta.assign(_rand=rng.random(len(meta))).sort_values(["_rand", "first", "__query__"], kind="stable")
        used = 0
        for row in meta.to_dict("records"):
            if used + int(row["rows"]) <= cap:
                keep.append(str(row["__query__"]))
                used += int(row["rows"])
    if not keep:
        raise ValueError(f"{spec.name} HPO sampling retained no complete queries")
    selected_index = work.loc[work["__query__"].isin(keep)].sort_values(
        ["__query__", "__decision_ts__", "candidate_id"], kind="stable",
    ).index
    return frame.loc[selected_index].reset_index(drop=True)


def _candidate_spec(trial: Any, template: ConsensusHeadSpec, *, training_rows: int) -> ConsensusHeadSpec:
    edge_name = trial.suggest_categorical("target_edges", sorted(TARGET_EDGE_OPTIONS))
    query = trial.suggest_categorical("query", QUERY_OPTIONS)
    weight_mode = trial.suggest_categorical("weight_mode", WEIGHT_OPTIONS)
    suggested = suggest_broad_lambdarank_params(
        trial, retained_fraction=.05, median_candidates_per_query=170.0,
    )
    # HPO is an explicitly subsampled proxy: keep a bounded ceiling while
    # retaining every other requested broad parameter dimension.  The winner
    # is re-fitted on the canonical complete-query cap in the later OOS run.
    params = materialize_lambdarank_params(
        suggested, training_rows=max(2, int(training_rows)), max_estimators=300,
    )
    params["verbosity"] = -1
    return ConsensusHeadSpec(
        name=template.name,
        cap=template.cap,
        weight_mode=weight_mode,
        query=query,
        fields=template.fields,
        target_edges_bps=TARGET_EDGE_OPTIONS[edge_name],
        params=params,
    )


def _fit_predict(
    train: pd.DataFrame, held: pd.DataFrame, spec: ConsensusHeadSpec, *, cap: int, seed: int,
) -> np.ndarray:
    sampled = _sample_complete_queries(train, spec, cap=cap, seed=seed)
    grade = _grade(sampled["policy_residual_bps"], spec.target_edges_bps)
    fitted = _fit_consensus_head(sampled, grade, spec, seed=seed)
    _, rank = fitted.predict_rank(held)
    return rank.astype(np.float32)


def _folds(
    ledger: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    output: list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]] = []
    for held_start in _month_starts(start, end):
        held_end = held_start + pd.offsets.MonthBegin(1)
        train = ledger.loc[
            ledger["__decision_ts__"].lt(held_start) & _valid_policy(ledger, cutoff=held_start),
        ].copy()
        held = ledger.loc[
            ledger["__decision_ts__"].ge(held_start)
            & ledger["__decision_ts__"].lt(held_end)
            & _valid_policy(ledger),
        ].copy()
        if len(train) < 2_000 or len(held) < 1_000:
            raise ValueError(f"insufficient strict OOF support for HPO fold {held_start:%Y-%m}")
        output.append((held_start, train, held))
    return output


def _summary(
    held: list[pd.DataFrame], *, candidate: list[np.ndarray], incumbent: list[np.ndarray],
) -> dict[str, float]:
    pieces: list[pd.DataFrame] = []
    for fold, proposed, baseline in zip(held, candidate, incumbent, strict=True):
        work = fold.loc[:, [
            "candidate_id", "__decision_ts__", "__month__", "p0_canonical_net_bps",
        ]].copy()
        work["net_bps"] = work.pop("p0_canonical_net_bps")
        # P0 canonical net already applies its fixed policy cost once.  Gross
        # is deliberately identical here: the head-selection utility ranks
        # exactly the chosen downstream policy target, not an invented gross.
        work["gross_bps"] = work["net_bps"]
        work["candidate"] = np.asarray(proposed, dtype=float)
        work["incumbent"] = np.asarray(baseline, dtype=float)
        pieces.append(work)
    return conditional_downstream_summary(
        pd.concat(pieces, ignore_index=True),
        candidate_score_column="candidate", incumbent_score_column="incumbent",
        net_column="net_bps", gross_column="gross_bps", timestamp_column="__decision_ts__",
        candidate_id_column="candidate_id", month_column="__month__",
    )


def _strict_head_promotion(summary: dict[str, float]) -> bool:
    """Promotion gate for a member of an ensemble, not a standalone head."""
    required = (
        "delta_top1_net_bps", "delta_top2_net_bps", "delta_top5_net_bps",
        "delta_top5_month_worst_net_bps", "conditional_utility_uplift_bps",
    )
    return all(float(summary[name]) > 0.0 for name in required)


def _read_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_short_p0_cmi_consensus_v2" or payload.get("side") != SIDE:
        raise ValueError("short consensus HPO requires the v2 short CMI contract")
    if len(payload.get("heads", [])) != 10:
        raise ValueError("short consensus HPO requires the exact ten frozen head slots")
    return payload


def _head_specs(contract: dict[str, Any]) -> list[ConsensusHeadSpec]:
    params = dict(contract["ranker_params"])
    edges = tuple(float(x) for x in contract["target"]["edges_bps"])
    return [
        ConsensusHeadSpec(
            name=str(raw["name"]), cap=int(raw["cap"]), weight_mode=str(raw["weight_mode"]),
            query=str(raw["query"]), fields=tuple(str(x) for x in raw["fields"]),
            target_edges_bps=edges, params=dict(params),
        )
        for raw in contract["heads"]
    ]


def run(
    *, ledger_roots: list[Path], contract_path: Path, geometry_dir: Path, out: Path,
    development_start: pd.Timestamp, development_end: pd.Timestamp,
    hpo_trials: int, hpo_query_cap: int, max_heads: int, seed: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if hpo_trials < 1 or max_heads < 1:
        raise ValueError("HPO trials and max heads must be positive")
    if development_end <= development_start:
        raise ValueError("development window is empty")
    contract = _read_contract(contract_path)
    templates = _head_specs(contract)[:min(10, max_heads)]
    geometry = load_geometry_bundle(geometry_dir)
    if geometry.bundle_sha256 != contract["geometry"]["bundle_sha256"]:
        raise ValueError("short HPO geometry hash differs from frozen CMI contract")
    # Need every history month that precedes the predeclared OOF development
    # months.  April 2024 is the earliest compatible P0 ledger state.
    history_start = pd.Timestamp("2024-04-01T00:00:00Z")
    ledger = _load_monthly_ledgers(ledger_roots, start=history_start, end=development_end)
    # The equal-month alternatives use this key for every trial/fold.  Compute
    # it once; recomputing ``dt.strftime`` over the full wide ledger dominated
    # the original bounded HPO's runtime without affecting any model input or
    # sampled query identity.
    ledger["__month__"] = ledger["__decision_ts__"].dt.to_period("M").astype(str)
    field_union = {field for head in templates for field in head.fields}
    ledger = _with_frozen_geometry(ledger, geometry=geometry, required=field_union)
    ledger["policy_residual_bps"] = (
        pd.to_numeric(ledger["p0_canonical_net_bps"], errors="coerce")
        - pd.to_numeric(ledger["prequential_base_anchor_bps"], errors="coerce")
    )
    fold_records = _folds(ledger, start=development_start, end=development_end)
    held = [record[2] for record in fold_records]
    base_ranks = [
        pd.to_numeric(fold["prequential_base_rank42"], errors="raise").to_numpy(np.float32)
        for fold in held
    ]
    # Each accepted head is retained as its fold-local *rank* output.  The
    # objective always evaluates the actual median ensemble and 75/25 base
    # blend; it never scores a head's NDCG or raw IC as its promotion metric.
    accepted: list[dict[str, Any]] = []
    accepted_ranks: list[list[np.ndarray]] = []
    all_trials: list[dict[str, Any]] = []
    head_results: list[dict[str, Any]] = []
    import optuna

    for head_index, template in enumerate(templates):
        def incumbent_scores() -> list[np.ndarray]:
            if not accepted_ranks:
                return [value.copy() for value in base_ranks]
            return [
                (.75 * base + .25 * np.nanmedian(np.column_stack([head[fold] for head in accepted_ranks]), axis=1)).astype(np.float32)
                for fold, base in enumerate(base_ranks)
            ]

        baseline = incumbent_scores()

        def objective(trial: Any) -> float:
            # Initial support only chooses the scale for min-leaf conversion;
            # it cannot change target-free identities or the chronological
            # folds used to judge this candidate.
            spec = _candidate_spec(trial, template, training_rows=hpo_query_cap)
            prediction: list[np.ndarray] = []
            for fold_index, (held_start, train, fold_held) in enumerate(fold_records):
                del held_start
                rank = _fit_predict(
                    train, fold_held, spec, cap=hpo_query_cap,
                    seed=seed + head_index * 100_000 + trial.number * 100 + fold_index,
                )
                prediction.append(rank)
                partial_candidate = [
                    (
                        .75 * base
                        + .25 * np.nanmedian(
                            np.column_stack(
                                [rank_list[index] for rank_list in accepted_ranks]
                                + [prediction[index]],
                            ),
                            axis=1,
                        )
                    ).astype(np.float32)
                    if accepted_ranks else (.75 * base + .25 * prediction[index]).astype(np.float32)
                    for index, base in enumerate(base_ranks[:len(prediction)])
                ]
                partial = _summary(
                    held[:len(prediction)], candidate=partial_candidate,
                    incumbent=baseline[:len(prediction)],
                )
                trial.report(float(partial["conditional_utility_uplift_bps"]), step=fold_index)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            candidate = [
                (
                    .75 * base
                    + .25 * np.nanmedian(
                        np.column_stack(
                            [rank_list[fold] for rank_list in accepted_ranks]
                            + [prediction[fold]],
                        ),
                        axis=1,
                    )
                ).astype(np.float32)
                if accepted_ranks else (.75 * base + .25 * prediction[fold]).astype(np.float32)
                for fold, base in enumerate(base_ranks)
            ]
            metrics = _summary(held, candidate=candidate, incumbent=baseline)
            trial.set_user_attr("ensemble_metrics", metrics)
            return float(metrics["conditional_utility_uplift_bps"])

        sampler = optuna.samplers.TPESampler(seed=seed + head_index, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1, interval_steps=1)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            objective, n_trials=hpo_trials, show_progress_bar=False,
            callbacks=[stop_after_no_improvement(patience=NO_IMPROVEMENT_PATIENCE)],
            gc_after_trial=True,
        )
        complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise ValueError(f"{template.name} HPO produced no completed trials")
        best = study.best_trial
        metrics = dict(best.user_attrs["ensemble_metrics"])
        spec = _candidate_spec(
            type("FrozenTrial", (), {
                "suggest_categorical": lambda _, name, values: best.params[name],
                "suggest_int": lambda _, name, low, high: best.params[name],
                "suggest_float": lambda _, name, low, high, **kwargs: best.params[name],
            })(),
            template, training_rows=hpo_query_cap,
        )
        promoted = _strict_head_promotion(metrics)
        # Refit the best candidate once to preserve the corresponding scores
        # for the next head's ensemble-only objective.  A non-promoted head is
        # not added, but its full trial evidence is retained.
        best_ranks: list[np.ndarray] = []
        if promoted:
            for fold_index, (_, train, fold_held) in enumerate(fold_records):
                best_ranks.append(_fit_predict(
                    train, fold_held, spec, cap=hpo_query_cap,
                    seed=seed + 900_000 + head_index * 100 + fold_index,
                ))
            accepted.append({
                "slot": head_index, "name": template.name,
                "target_edges_bps": list(spec.target_edges_bps), "query": spec.query,
                "weight_mode": spec.weight_mode, "ranker_params": spec.params,
                "metrics": metrics,
            })
            accepted_ranks.append(best_ranks)
        for trial in study.trials:
            row = {
                "head": template.name, "slot": head_index, "trial": trial.number,
                "state": trial.state.name, "value": trial.value,
                "promoted_head": promoted,
                "stop_reason": study.user_attrs.get("stop_reason", "trial_budget"),
                "no_improvement_patience": NO_IMPROVEMENT_PATIENCE,
                **trial.params,
            }
            for key, value in (trial.user_attrs.get("ensemble_metrics") or {}).items():
                row[f"metric_{key}"] = value
            all_trials.append(row)
        head_results.append({
            "head": template.name, "slot": head_index, "winner_trial": best.number,
            "winner_uplift_bps": best.value, "promoted": promoted,
            "completed_trials": len(complete), "total_trials": len(study.trials),
            "stop_reason": study.user_attrs.get("stop_reason", "trial_budget"),
            "winner": accepted[-1] if promoted else {
                "target_edges_bps": list(spec.target_edges_bps), "query": spec.query,
                "weight_mode": spec.weight_mode, "ranker_params": spec.params, "metrics": metrics,
            },
        })

    out.mkdir(parents=True)
    pd.DataFrame(all_trials).to_parquet(out / "ensemble_hpo_trials.parquet", index=False, compression="zstd")
    pd.DataFrame([
        {**row, "winner_json": json.dumps(row["winner"], sort_keys=True)}
        for row in head_results
    ]).to_parquet(
        out / "head_hpo_winners.parquet", index=False, compression="zstd",
    )
    # Create the exact frozen contract consumed by the strict OOS runner.  No
    # rejected head survives into the runtime candidate architecture.
    payload = json.loads(json.dumps(contract))
    payload["schema"] = "strict_r3_short_p0_cmi_consensus_v2"
    payload["hpo"] = {
        "status": "complete", "selection_window": {
            "start": development_start.isoformat(), "end_exclusive": development_end.isoformat(),
        },
        "selection_rule": "sequential base-plus-median-ensemble conditional utility; strict all-tail/worst-month promotion",
        "no_improvement_patience": NO_IMPROVEMENT_PATIENCE,
        "hpo_query_cap": int(hpo_query_cap), "requested_trials_per_head": int(hpo_trials),
        "accepted_head_count": int(len(accepted)),
        "rejected_slots_excluded_from_runtime": True,
    }
    winners = {row["name"]: row for row in accepted}
    payload["heads"] = [
        {
            **raw,
            "target_edges_bps": winners[raw["name"]]["target_edges_bps"],
            "query": winners[raw["name"]]["query"],
            "weight_mode": winners[raw["name"]]["weight_mode"],
            "ranker_params": winners[raw["name"]]["ranker_params"],
        }
        for raw in payload["heads"] if raw["name"] in winners
    ]
    if not payload["heads"]:
        payload["runtime_status"] = "base_only_no_consensus_head_passed_development_gate"
    else:
        payload["runtime_status"] = "research_candidate_pending_untouched_oos"
    (out / "short_consensus_hpo_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "schema": "strict_r3_short_p0_conditional_consensus_hpo_v1", "status": "complete", "side": SIDE,
        "development_window": {"start": development_start.isoformat(), "end_exclusive": development_end.isoformat()},
        "untouched_from": development_end.isoformat(),
        "geometry": {"bundle_sha256": geometry.bundle_sha256, "monthly_refit": False, "raw_k9_memberships": False},
        "strict_prequential_training": "each held development month uses only policy outcomes resolved before its first decision timestamp",
        "selection_level": "incremental base-plus-ensemble economics, never standalone-head performance",
        "source_hashes": {
            "template_contract": _sha256(contract_path), "geometry": _sha256(geometry_dir / "run_manifest.json"),
            "ledgers": {str(root): _sha256(root / "run_manifest.json") for root in ledger_roots},
        },
        "accepted_heads": accepted,
        "head_results": head_results,
        "no_improvement_patience": NO_IMPROVEMENT_PATIENCE,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--development-end-exclusive", default="2025-04-01T00:00:00Z")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--hpo-query-cap", type=int, default=50_000)
    parser.add_argument("--max-heads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    print(run(
        ledger_roots=args.ledger_root, contract_path=args.contract, geometry_dir=args.geometry_dir,
        out=args.out, development_start=_utc(args.development_start),
        development_end=_utc(args.development_end_exclusive), hpo_trials=args.trials,
        hpo_query_cap=args.hpo_query_cap, max_heads=args.max_heads, seed=args.seed,
    ))


if __name__ == "__main__":
    main()
