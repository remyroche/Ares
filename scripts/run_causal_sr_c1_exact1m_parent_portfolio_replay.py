#!/usr/bin/env python3
"""Exact-1m, portfolio-constrained C0/C1 parent-policy reconciliation.

This research-only preparatory replay turns the existing target-free C0/C1
MC1 panels into a *common*, source-valid exact-one-minute outcome panel.  It
does not fit or alter any model and it does not import live execution.

The C0 and C1 selections are made before reading one-minute bars.  Complete
paths are then attached only for realised outcome/portfolio evaluation, so a
missing future path can neither select a candidate nor consume a slot.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_rich_policy_contract import (
    Exact1mRichExecutionContract,
    RichExitExtensions,
    replay_exact_1m_rich_policy,
)
from extreme_price_movements.portfolio_policy_replay import (
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _params as portfolio_params,
)
from scripts.run_strict_r3_p8u_exact_1m_simple_policy_optimiser import (
    DEFAULT_MINUTE_ROOT,
    ExactPaths,
    _load_policy,
    _materialize_exact_paths,
)


DEFAULT_JUNJUL = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v5"
DEFAULT_AUG = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_august_20260831_v3"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_c1_exact1m_parent_portfolio_20260831_v1"
ARMS = ("C0_refit_core_postfeb", "C1_refit_core_plus_causal_sr")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arm(root: Path, arm: str, admission_threshold_bps: float = 50.0) -> pd.DataFrame:
    path = root / f"{arm}_target_free_admission.parquet"
    values = pd.read_parquet(path).copy()
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "dual_admitted", "auction_priority_bps",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise AssertionError(f"{path}: missing target-free score fields {missing}")
    values["candidate_id"] = values["candidate_id"].astype(str)
    values["timestamp"] = pd.to_datetime(values.pop("__decision_ts__"), utc=True, errors="raise")
    values["symbol"] = values.pop("__symbol__").astype(str)
    values["side_name"] = values["side_name"].astype(str).str.lower().str.strip()
    values = values.loc[values["side_name"].eq("long")].copy()
    values["bcf_mc1_expected_bps"] = pd.to_numeric(values["bcf_mc1_expected_bps"], errors="raise")
    values["current_mc1_expected_bps"] = pd.to_numeric(values["current_mc1_expected_bps"], errors="raise")
    values["auction_priority_bps"] = pd.to_numeric(values["auction_priority_bps"], errors="raise")
    stored_dual_admitted = values["dual_admitted"].fillna(False).astype(bool)
    if values["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate identity")
    if not values.loc[stored_dual_admitted, ["bcf_mc1_expected_bps", "current_mc1_expected_bps"]].ge(50.0).all(axis=None):
        raise AssertionError(f"{path}: dual-admitted candidate below +50 bps")
    # Recompute this field from target-free MC1 predictions.  The stored flag
    # remains a +50-bps integrity receipt; a research route may use a different
    # explicitly named dual threshold without consulting any outcome.
    values["dual_admitted"] = (
        values["bcf_mc1_expected_bps"].ge(float(admission_threshold_bps))
        & values["current_mc1_expected_bps"].ge(float(admission_threshold_bps))
    )
    return values.loc[:, [
        "candidate_id", "timestamp", "symbol", "side_name", "bcf_final_score",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_admitted",
        "auction_priority_bps",
    ]]


def _common_target_free_routes(
    junjul: Path,
    august: Path,
    *,
    admission_threshold_bps: float = 50.0,
    extra_roots: tuple[Path, ...] = (),
    arms: tuple[str, ...] = ARMS,
) -> dict[str, pd.DataFrame]:
    routes: dict[str, list[pd.DataFrame]] = {arm: [] for arm in arms}
    for root in (junjul, august, *extra_roots):
        for arm in arms:
            panel = _load_arm(root, arm, admission_threshold_bps)
            panel = panel.loc[panel["dual_admitted"]].copy()
            panel["entry_ts"] = panel["timestamp"] + pd.Timedelta(minutes=5)
            routes[arm].append(panel)
    result: dict[str, pd.DataFrame] = {}
    for arm, parts in routes.items():
        table = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "candidate_id"], kind="stable")
        if table["candidate_id"].duplicated().any():
            raise AssertionError(f"{arm}: duplicated identities across source panels")
        result[arm] = table.reset_index(drop=True)
    return result


def _direct_dual_target_free_route(
    path: Path,
    *,
    admission_threshold_bps: float,
) -> pd.DataFrame:
    """Load one immutable, already-paired target-free dual-MC1 panel.

    This supports an exact-path validation of a newly materialised MC1 package
    without copying its values into the legacy C0/C1 archive layout.  It is
    deliberately score-only: no outcome/path field is permitted here.
    """
    values = pd.read_parquet(path).copy()
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "auction_priority_bps",
    }
    missing = sorted(required.difference(values.columns))
    if missing:
        raise AssertionError(f"{path}: missing target-free score fields {missing}")
    forbidden = {
        "policy_path_valid", "policy_net_bps", "policy_gross_bps", "outcome",
        "label", "exit", "exact_net_bps", "exact_gross_bps",
    }
    present = sorted(forbidden.intersection(values.columns))
    if present:
        raise AssertionError(f"{path}: forbidden outcome/path columns {present}")
    values["candidate_id"] = values["candidate_id"].astype(str)
    if values["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate identity")
    values["timestamp"] = pd.to_datetime(values.pop("__decision_ts__"), utc=True, errors="raise")
    values["symbol"] = values.pop("__symbol__").astype(str)
    values["side_name"] = values["side_name"].astype(str).str.lower().str.strip()
    values = values.loc[values["side_name"].eq("long")].copy()
    for field in ("bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"):
        values[field] = pd.to_numeric(values[field], errors="raise")
    values["dual_admitted"] = (
        values["bcf_mc1_expected_bps"].ge(float(admission_threshold_bps))
        & values["current_mc1_expected_bps"].ge(float(admission_threshold_bps))
    )
    values = values.loc[values["dual_admitted"]].copy()
    values["entry_ts"] = values["timestamp"] + pd.Timedelta(minutes=5)
    return values.loc[:, [
        "candidate_id", "timestamp", "entry_ts", "symbol", "side_name",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_admitted",
        "auction_priority_bps",
    ]].reset_index(drop=True)


def _path_population(routes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    values = pd.concat([table for table in routes.values()], ignore_index=True)
    values = values.sort_values(["timestamp", "candidate_id"], kind="stable").drop_duplicates("candidate_id", keep="first")
    if values.empty:
        raise RuntimeError("no target-free dual-MC1 candidates")
    return values.loc[:, [
        "candidate_id", "timestamp", "entry_ts", "symbol", "side_name",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps",
    ]].reset_index(drop=True)


def _outcomes(paths, policy: Path) -> pd.DataFrame:
    params, median, receipt = _load_policy(policy)
    replay = replay_exact_1m_rich_policy(
        positions=pd.DataFrame({
            "entry_price": paths.entry,
            "atr": paths.atr,
            "entry_ts": paths.rows["entry_ts"],
        }),
        highs=paths.high,
        lows=paths.low,
        closes=paths.close,
        params=params,
        median_atr_fraction=median,
        contract=Exact1mRichExecutionContract(entry_delay_minutes=5),
        extensions=RichExitExtensions(),
    )
    if not np.asarray(replay["path_valid"], dtype=bool).all():
        raise AssertionError("complete paths must replay under the rich exact-1m contract")
    net = np.asarray(replay["net_bps"], dtype=float)
    gross = np.asarray(replay["gross_bps"], dtype=float)
    if not np.allclose(gross - net, 100.0, rtol=0.0, atol=1e-8):
        raise AssertionError("rich exact-1m cost must be applied exactly once")
    result = paths.rows.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol"]].copy()
    result["exact_entry_price"] = np.asarray(paths.entry, dtype=float)
    result["exact_signal_atr"] = np.asarray(paths.atr, dtype=float)
    result["exact_gross_bps"] = gross
    result["exact_net_bps"] = net
    result["exact_exit_price"] = np.asarray(replay["exit_price"], dtype=float)
    result["exact_exit_ts"] = pd.to_datetime(replay["exit_timestamp"], utc=True, errors="raise")
    result["exact_exit_minute"] = np.asarray(replay["exit_minute"], dtype=int)
    result["exact_exit_reason"] = np.asarray(replay["exit_reason"], dtype=object)
    result.attrs["policy_receipt"] = receipt
    return result


def _materialize_paths_row_local(
    population: pd.DataFrame,
    *,
    minute_root: Path,
    workers: int,
    prefer_covering_part: bool = False,
) -> tuple[ExactPaths, pd.DataFrame, pd.DataFrame]:
    """Materialise exact paths without letting one corrupt symbol abort peers.

    The standard HPO materialiser correctly treats unreadable named source
    parts as a hard data-integrity error.  For a post-admission replay, the
    proper equivalent is narrower: retain that integrity failure in the audit,
    exclude every affected outcome *after* target-free routing, and continue
    the independent symbols.  Nothing about this function changes a score,
    admission, or auction priority.
    """
    groups = [(str(symbol), group.reset_index(drop=True)) for symbol, group in population.groupby("symbol", sort=True)]

    def one(symbol: str, group: pd.DataFrame):
        try:
            paths, coverage, invalid = _materialize_exact_paths(
                group,
                minute_root=minute_root,
                prefer_covering_part=prefer_covering_part,
            )
        except Exception as exc:  # exact source is unavailable for this symbol
            bad = group.copy()
            bad["outcome_invalid_reason"] = f"exact_1m_source_error:{type(exc).__name__}"
            return symbol, None, pd.DataFrame([{
                "symbol": str(symbol), "candidate_rows": int(len(group)),
                "valid_rows": 0, "reason": f"exact_1m_source_error:{type(exc).__name__}",
                "source_error": str(exc),
            }]), bad
        return symbol, paths, coverage, invalid

    completed: dict[str, tuple[ExactPaths | None, pd.DataFrame, pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="exact1m") as executor:
        futures = {executor.submit(one, symbol, group): symbol for symbol, group in groups}
        for ordinal, future in enumerate(as_completed(futures), start=1):
            symbol, paths, coverage, invalid = future.result()
            completed[symbol] = (paths, coverage, invalid)
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(futures):
                print(f"exact-1m source {ordinal}/{len(futures)} symbols completed", flush=True)
    path_parts: list[ExactPaths] = []
    coverage_parts: list[pd.DataFrame] = []
    invalid_parts: list[pd.DataFrame] = []
    for symbol, _ in groups:
        paths, coverage, invalid = completed[symbol]
        if paths is not None:
            path_parts.append(paths)
        coverage_parts.append(coverage)
        if not invalid.empty:
            invalid_parts.append(invalid)
    if not path_parts:
        raise RuntimeError("no valid exact-one-minute paths after row-local source validation")
    rows = pd.concat([part.rows for part in path_parts], ignore_index=True)
    paths = ExactPaths(
        rows=rows,
        entry=np.concatenate([part.entry for part in path_parts]),
        atr=np.concatenate([part.atr for part in path_parts]),
        high=np.concatenate([part.high for part in path_parts]),
        low=np.concatenate([part.low for part in path_parts]),
        close=np.concatenate([part.close for part in path_parts]),
    )
    if paths.rows["candidate_id"].duplicated().any():
        raise AssertionError("row-local source materialisation duplicated candidate identity")
    coverage = pd.concat(coverage_parts, ignore_index=True)
    invalid = pd.concat(invalid_parts, ignore_index=True) if invalid_parts else pd.DataFrame(
        columns=[*population.columns, "outcome_invalid_reason"]
    )
    if set(paths.rows["candidate_id"]).intersection(set(invalid["candidate_id"])):
        raise AssertionError("an exact path was both valid and invalid")
    if set(paths.rows["candidate_id"]).union(set(invalid["candidate_id"])) != set(population["candidate_id"]):
        raise AssertionError("row-local source materialisation lost target-free identity")
    return paths, coverage, invalid


def _portfolio(route: pd.DataFrame, outcomes: pd.DataFrame, arm: str):
    table = route.merge(outcomes, on=["candidate_id", "timestamp", "entry_ts", "symbol"], how="inner", validate="one_to_one")
    candidates = pd.DataFrame({
        "timestamp": table["entry_ts"],
        "decision_timestamp": table["timestamp"],
        "candidate_id": table["candidate_id"],
        "symbol": table["symbol"],
        "side": "long",
        "strategy_id": arm,
        "policy_archetype": "exact1m_rich_parent",
        "normalized_rank_score": 1.0,
        "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0,
        "portfolio_priority_adjustment": table["auction_priority_bps"],
        "entry_price": table["exact_entry_price"],
        "exit_timestamp": table["exact_exit_ts"],
        "exit_price": table["exact_exit_price"],
        "net_return": table["exact_net_bps"] / 10_000.0,
        "gross_return": table["exact_gross_bps"] / 10_000.0,
        "holding_bars": np.maximum(table["exact_exit_minute"].to_numpy(int) + 1, 1),
        "simple_policy_exit_reason": table["exact_exit_reason"],
        "fees_bps": 100.0,
        "expected_friction_bps": 0.0,
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
        "policy_outcome_available": True,
    })
    candidates = normalise_candidate_table(candidates)
    decisions, equity, _ = replay_candidates(
        candidates, portfolio_params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp",
    )
    indices = pd.to_numeric(decisions["candidate_index"], errors="raise").astype(int).to_numpy()
    decisions = decisions.copy()
    decisions["candidate_id"] = candidates.iloc[indices]["candidate_id"].to_numpy()
    decisions["decision_timestamp"] = candidates.iloc[indices]["decision_timestamp"].to_numpy()
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="raise") * 10_000.0
    metrics = compute_replay_metrics(candidates, decisions, equity, params=portfolio_params())
    metrics.update({
        "target_free_dual_admitted": int(len(route)),
        "exact_path_valid": int(len(candidates)),
        "portfolio_accepted": int(len(accepted)),
        "net_bps_per_trade": float(accepted["net_bps"].mean()) if len(accepted) else float("nan"),
        "total_net_bps": float(accepted["net_bps"].sum()) if len(accepted) else 0.0,
    })
    return candidates, decisions, accepted, equity, metrics


def _monthly(accepted: pd.DataFrame) -> pd.DataFrame:
    result = accepted.copy()
    result["month"] = pd.to_datetime(result["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    return result.groupby("month", as_index=False, sort=True).agg(
        trades=("candidate_id", "size"),
        net_bps_per_trade=("net_bps", "mean"),
        total_net_bps=("net_bps", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junjul-root", type=Path, default=DEFAULT_JUNJUL)
    parser.add_argument("--august-root", type=Path, default=DEFAULT_AUG)
    parser.add_argument(
        "--extra-root", type=Path, action="append", default=[],
        help="Additional immutable target-free C0/C1 score-panel root(s).",
    )
    parser.add_argument(
        "--arm", action="append", default=[], choices=ARMS,
        help="Replay only named arm(s); defaults to both C0 and C1.",
    )
    parser.add_argument(
        "--direct-dual-panel", action="append", nargs=2, metavar=("ARM", "PATH"), default=[],
        help=(
            "Immutable paired target-free MC1 panel, supplied as ARM PATH. "
            "May be repeated; direct panels replace legacy C0/C1 archive inputs."
        ),
    )
    parser.add_argument("--admission-threshold-bps", type=float, default=50.0)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument("--source-workers", type=int, default=8)
    parser.add_argument(
        "--research-fast-covering-part", action="store_true",
        help=(
            "Research-only speed screen: use a single normal immutable part "
            "when its declared bounds cover the requested path range. "
            "A full conflict scan remains required for promotion evidence."
        ),
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if args.admission_threshold_bps <= 0:
        raise ValueError("admission threshold must be positive")
    direct_panels = [(str(arm), Path(path).resolve()) for arm, path in args.direct_dual_panel]
    if direct_panels:
        if args.arm or args.extra_root:
            raise ValueError("--direct-dual-panel cannot be combined with --arm or --extra-root")
        names = [name for name, _ in direct_panels]
        if len(set(names)) != len(names):
            raise ValueError("direct panel arm names must be unique")
        if any(not path.is_file() for _, path in direct_panels):
            raise FileNotFoundError("a --direct-dual-panel path does not exist")
        routes = {
            name: _direct_dual_target_free_route(
                path, admission_threshold_bps=float(args.admission_threshold_bps),
            )
            for name, path in direct_panels
        }
        source_roots: tuple[Path, ...] = ()
        selected_arms = tuple(routes)
    else:
        source_roots = (
            args.junjul_root.resolve(), args.august_root.resolve(),
            *(path.resolve() for path in args.extra_root),
        )
        if len(set(source_roots)) != len(source_roots):
            raise ValueError("source roots must be unique")
        selected_arms = tuple(args.arm) if args.arm else ARMS
        routes = _common_target_free_routes(
            args.junjul_root.resolve(), args.august_root.resolve(),
            admission_threshold_bps=float(args.admission_threshold_bps),
            extra_roots=tuple(path.resolve() for path in args.extra_root),
            arms=selected_arms,
        )
    population = _path_population(routes)
    paths, coverage, invalid = _materialize_paths_row_local(
        population,
        minute_root=args.minute_root.resolve(),
        workers=args.source_workers,
        prefer_covering_part=bool(args.research_fast_covering_part),
    )
    outcome = _outcomes(paths, args.policy.resolve())
    valid_ids = set(outcome["candidate_id"].astype(str))
    out.mkdir(parents=True, exist_ok=False)
    population.to_parquet(out / "union_target_free_dual_admitted.parquet", index=False, compression="zstd")
    coverage.to_parquet(out / "exact_1m_source_coverage.parquet", index=False, compression="zstd")
    invalid.to_parquet(out / "invalid_outcomes_after_target_free_route.parquet", index=False, compression="zstd")
    paths.rows.to_parquet(out / "valid_exact_paths_rows.parquet", index=False, compression="zstd")
    np.savez_compressed(out / "exact_paths.npz", candidate_id=paths.rows["candidate_id"].astype(str).to_numpy(), entry=paths.entry, atr=paths.atr, high=paths.high, low=paths.low, close=paths.close)
    outcome.to_parquet(out / "exact_1m_rich_parent_outcomes.parquet", index=False, compression="zstd")
    results: list[dict[str, object]] = []
    for arm, route in routes.items():
        # Keep the pre-path route count separate from the source-valid replay
        # population.  The source join happens only after target-free routing;
        # reporting the latter as ``target_free_dual_admitted`` would conceal
        # outcome-source coverage and make an incomplete historical archive
        # look like a smaller routed universe.
        target_free_dual_admitted = int(len(route))
        valid_route = route.loc[route["candidate_id"].isin(valid_ids)].copy()
        candidates, decisions, accepted, equity, metrics = _portfolio(valid_route, outcome, arm)
        metrics["target_free_dual_admitted"] = target_free_dual_admitted
        metrics["exact_path_valid"] = int(len(valid_route))
        candidates.to_parquet(out / f"{arm}_portfolio_candidates.parquet", index=False, compression="zstd")
        decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
        accepted.to_parquet(out / f"{arm}_portfolio_accepted.parquet", index=False, compression="zstd")
        equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
        month = _monthly(accepted)
        month.insert(0, "arm", arm)
        month.to_parquet(out / f"{arm}_monthly_portfolio_metrics.parquet", index=False, compression="zstd")
        results.append({"arm": arm, **metrics})
    summary = pd.DataFrame(results)
    if "C0_refit_core_postfeb" in set(summary["arm"]):
        reference = summary.loc[summary["arm"].eq("C0_refit_core_postfeb")].iloc[0]
        for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
            summary[f"delta_vs_C0_{field}"] = summary[field] - reference[field]
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "causal-sr-c1-exact1m-parent-portfolio-v1",
        "scope": "offline research only; exact one-minute rich-parent path, target-free C0/C1 dual-MC1 routes, normal global portfolio auction; no E2/H4 authority in this prerequisite receipt",
        "source_panels": (
            {str(path): _sha256(path) for _, path in direct_panels}
            if direct_panels else {
                str(root / f"{arm}_target_free_admission.parquet"):
                _sha256(root / f"{arm}_target_free_admission.parquet")
                for root in source_roots for arm in selected_arms
            }
        ),
        "policy": str(args.policy.resolve()),
        "policy_sha256": _sha256(args.policy.resolve()),
        "candidate_selection": (
            f"dual BCF/current MC1 >= +{args.admission_threshold_bps:g} bps "
            "before any one-minute path access; BCF mapped EV priority"
        ),
        "arms": list(selected_arms),
        "portfolio": "global chronological controlled 7x/10%-margin slot, two new entries per timestamp, eight concurrent, 80% wallet budget",
        "outcome_handling": "source-invalid exact paths excluded after target-free routing; never used for score/admission/portfolio capacity",
        "union_target_free_rows": int(len(population)),
        "valid_exact_paths": int(len(outcome)),
        "invalid_exact_paths": int(len(invalid)),
        "one_minute_exit": "Exact1mRichExecutionContract entry delay five minutes; rich policy cost 100 bps once",
        "source_workers": int(args.source_workers),
        "source_read_mode": (
            "research_fast_covering_immutable_part"
            if args.research_fast_covering_part else "full_overlapping_part_conflict_scan"
        ),
        "e2_h4": "deliberately absent from this parent-policy prerequisite; they require a same-route transfer/refit and cannot be spliced from the legacy BCF-top-two route",
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
