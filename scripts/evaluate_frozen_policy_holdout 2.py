#!/usr/bin/env python3
"""Evaluate a frozen simple-policy package on a post-policy holdout.

This script intentionally does not optimise thresholds or portfolio parameters.
It scores label-backed rows with the train-meta-frozen model state, maps scores
through the saved policy-window rank references, builds executable candidates
with the saved per-strategy simple-policy params, and replays them with the
saved portfolio policy config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.inference.parity import (  # noqa: E402
    calibrated_score_and_threshold,
)
from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    PolicyRankReferenceStore,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    run_portfolio_policy_replay,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    PORTFOLIO_CANDIDATE_EXPORT_NORMALIZED_RANK_FLOOR,
    _apply_delayed_entry_execution_model,
    _assert_policy_path_coverage,
    _build_simple_policy_candidate_rows,
    _expand_strategy_id_allowlist,
    _fetch_policy_paths,
    _filter_policy_quote_rows,
    _generate_policy_predictions_from_models,
    _json_safe,
    _load_policy_stage_view,
    _load_slice_plan_source_validation,
    _policy_market_data_root,
    _policy_params_from_deployment_strategy,
    _strategy_id_matches_allowlist,
    _write_delay_rejection_reports,
    _write_rank_threshold_band_reports,
    _write_simple_policy_candidate_metadata,
)
from extreme_price_movements.simple_position_sizer import (  # noqa: E402
    load_calibration_curves,
)


def _load_deployment_payload(run_root: Path, market_mode: str) -> tuple[dict[str, Any], Path]:
    candidates = [
        run_root / "policy_params" / f"strategy_for_inference_{market_mode}.json",
        run_root / "policy_params" / "strategy_for_inference.json",
        run_root / "policy_params" / f"best_policy_params_{market_mode}.json",
        run_root / "policy_params" / "best_policy_params.json",
        run_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload, path
    raise FileNotFoundError(f"No deployment payload found under {run_root}")


def _selected_strategies(
    payload: dict[str, Any],
    allowlist: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for strategy in payload.get("strategies") or []:
        if not isinstance(strategy, dict):
            continue
        sid = str(strategy.get("strategy_id") or "")
        if not sid or not bool(strategy.get("selected", True)):
            continue
        if allowlist and not _strategy_id_matches_allowlist(sid, allowlist):
            continue
        out.append(strategy)
    return out


def _build_holdout_stage_view(
    base_stage_view: dict[str, Any],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    symbols = base_stage_view.get("symbols") or base_stage_view.get("allowed_symbols") or []
    return {
        "stage_name": "post_policy_frozen_holdout",
        "source_roles": ["post_policy_frozen_holdout"],
        "symbols": sorted({str(sym) for sym in symbols}),
        "allowed_symbols": sorted({str(sym) for sym in symbols}),
        "allowed_periods": [{"start_ts": start.isoformat(), "end_ts": end.isoformat()}],
        "allowed_start_ts": start.isoformat(),
        "allowed_end_ts": end.isoformat(),
        "n_plans": 1,
        "policy_source": "manual_post_policy_holdout",
    }


def _coerce_utc(value: str, *, name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid {name}: {value}")
    return pd.Timestamp(ts)


def _apply_saved_rank_references(
    frame: pd.DataFrame,
    *,
    strategy_id: str,
    side: str,
    rank_store: PolicyRankReferenceStore,
) -> pd.DataFrame:
    out = frame.copy()
    scores = pd.to_numeric(out["calibrated_score"], errors="coerce")
    out["rank_pct"] = [
        rank_store.lookup(
            strategy_id=strategy_id,
            side=side,
            calibrated_score=float(score),
        ).policy_rank_pct
        if pd.notna(score)
        else np.nan
        for score in scores
    ]
    return out


def _finalise_holdout_candidates(
    frames: list[pd.DataFrame],
    *,
    rank_store: PolicyRankReferenceStore,
    rank_floor: float,
) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    out = pd.concat(usable, ignore_index=True)
    out["calibrated_score"] = pd.to_numeric(out["calibrated_score"], errors="coerce")
    out["strategy_rank_pct"] = pd.to_numeric(out["strategy_rank_pct"], errors="coerce")
    out["auction_rank_score"] = [
        rank_store.lookup_auction(calibrated_score=float(score)).policy_rank_pct
        if pd.notna(score)
        else np.nan
        for score in out["calibrated_score"]
    ]
    out["normalized_rank_score"] = pd.to_numeric(
        out["auction_rank_score"],
        errors="coerce",
    )
    floor = float(np.clip(rank_floor, 0.0, 1.0))
    before = int(len(out))
    out = out.loc[out["normalized_rank_score"] >= floor].copy()
    out["base_strategy_threshold"] = floor
    out = out.sort_values(
        ["timestamp", "normalized_rank_score", "calibrated_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    out.attrs["holdout_rows_before_auction_floor"] = before
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--market-mode", choices=("spot", "perps"), default="perps")
    parser.add_argument("--artifact-source-run-id", default=os.environ.get("EPM_ARTIFACT_SOURCE_RUN_ID", ""))
    parser.add_argument("--predict-start", default=None)
    parser.add_argument("--predict-end", default=None)
    parser.add_argument("--strategy-id", action="append", default=[])
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-portfolio-replay", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    run_root = data_root / "artifacts" / args.run_id
    slice_plan_path = run_root / "slices" / "slice_plan.json"
    meta_state_path = run_root / "models" / "model_state_meta.pkl"
    if not meta_state_path.exists():
        raise SystemExit(f"Missing train-meta model state: {meta_state_path}")
    if args.artifact_source_run_id:
        os.environ["EPM_ARTIFACT_SOURCE_RUN_ID"] = str(args.artifact_source_run_id)

    source_validation = _load_slice_plan_source_validation(slice_plan_path)
    base_stage_view, stage_name = _load_policy_stage_view(slice_plan_path)
    if stage_name != "policy_optimiser" or not base_stage_view:
        raise SystemExit(f"Missing policy_optimiser stage view: {slice_plan_path}")
    policy_predict_end = source_validation.get("policy_optimiser_predict_end")
    if not args.predict_start and not policy_predict_end:
        raise SystemExit("Missing --predict-start and slice plan policy predict end")
    predict_start = _coerce_utc(args.predict_start or policy_predict_end, name="predict-start")
    predict_end = _coerce_utc(
        args.predict_end or pd.Timestamp.utcnow().isoformat(),
        name="predict-end",
    )
    if predict_end <= predict_start:
        raise SystemExit(f"predict-end must be after predict-start: {predict_end} <= {predict_start}")
    policy_end = _coerce_utc(policy_predict_end, name="policy_optimiser_predict_end")
    if predict_start < policy_end:
        raise SystemExit(
            "Holdout must start at or after policy optimisation predict end: "
            f"{predict_start} < {policy_end}"
        )
    holdout_stage_view = _build_holdout_stage_view(
        base_stage_view,
        start=predict_start,
        end=predict_end,
    )

    payload, deployment_path = _load_deployment_payload(run_root, args.market_mode)
    selection_rules = payload.get("selection_rules") or {}
    allowlist = set(_expand_strategy_id_allowlist(args.strategy_id or []))
    strategies = _selected_strategies(payload, allowlist)
    if args.max_strategies is not None:
        strategies = strategies[: int(args.max_strategies)]
    if not strategies:
        raise SystemExit(f"No selected strategies found in {deployment_path}")
    strategy_ids = [str(strategy["strategy_id"]) for strategy in strategies]
    strategy_by_id = {str(strategy["strategy_id"]): strategy for strategy in strategies}

    full_state = joblib.load(meta_state_path)
    frames, sources = _generate_policy_predictions_from_models(
        data_root=str(data_root),
        run_id=args.run_id,
        stage_view=holdout_stage_view,
        max_strategies=None,
        strategy_ids_allowlist=set(strategy_ids),
        market_mode=args.market_mode,
        full_state_override=full_state,
        source_tag="generated_from_train_meta_state_post_policy_holdout",
    )
    if not frames:
        raise SystemExit("No holdout prediction frames generated.")

    calibration_data = load_calibration_curves(str(data_root), args.run_id)
    rank_store = PolicyRankReferenceStore(data_root=data_root, run_id=args.run_id)
    ds = PartitionedOHLCVStore(
        _policy_market_data_root(str(data_root), args.market_mode),
        timeframe="15m",
    )

    candidate_frames: list[pd.DataFrame] = []
    strategy_reports: dict[str, Any] = {}
    for strategy_id in strategy_ids:
        df = frames.get(strategy_id)
        if df is None or df.empty:
            strategy_reports[strategy_id] = {"prediction_rows": 0, "candidate_rows": 0}
            continue
        if "clf" not in df.columns and "oof_pred" in df.columns:
            df = df.copy()
            df["clf"] = df["oof_pred"]
        if "clf" not in df.columns:
            strategy_reports[strategy_id] = {
                "prediction_rows": int(len(df)),
                "candidate_rows": 0,
                "reason": "missing_clf_score",
            }
            continue
        strategy = strategy_by_id[strategy_id]
        side = "short" if strategy_id.startswith("short") else "long"
        params, size_power, threshold = _policy_params_from_deployment_strategy(
            strategy,
            selection_rules,
        )
        work = _filter_policy_quote_rows(df.copy(), args.market_mode)
        work["raw_meta_prediction"] = pd.to_numeric(work["clf"], errors="coerce")
        work["calibrated_score"] = work["raw_meta_prediction"].map(
            lambda raw_score: (
                calibrated_score_and_threshold(
                    raw_score=float(raw_score),
                    strategy_id=strategy_id,
                    calibration_data=calibration_data,
                    default_threshold=1.0,
                )[0]
                if pd.notna(raw_score)
                else np.nan
            )
        )
        work["strategy_id"] = strategy_id
        if "side" not in work.columns:
            work["side"] = -1 if side == "short" else 1
        work = _apply_saved_rank_references(
            work,
            strategy_id=strategy_id,
            side=side,
            rank_store=rank_store,
        )
        work = work.dropna(
            subset=["timestamp", "symbol", "rank_pct", "calibrated_score"]
        ).copy()
        work = work.sort_values("timestamp").reset_index(drop=True)
        local = work.loc[
            pd.to_numeric(work["rank_pct"], errors="coerce") >= float(threshold)
        ].copy().reset_index(drop=True)
        if local.empty:
            strategy_reports[strategy_id] = {
                "prediction_rows": int(len(work)),
                "local_threshold": float(threshold),
                "local_candidate_rows": 0,
                "candidate_rows": 0,
                "source": sources.get(strategy_id),
            }
            continue
        paths = _fetch_policy_paths(local, ds)
        local, paths = _apply_delayed_entry_execution_model(
            local,
            paths,
            data_root=str(data_root),
            market_mode=args.market_mode,
        )
        finite_rows, total_rows, path_coverage = _assert_policy_path_coverage(
            strategy_id=strategy_id,
            paths=paths,
        )
        candidates = _build_simple_policy_candidate_rows(
            strategy_id=strategy_id,
            df_top=local,
            paths=paths,
            cost_pct=DEFAULT_POLICY_PER_SIDE_COST_PCT,
            best_params=params,
            best_size_power=size_power,
            base_strategy_threshold=float(threshold),
            market_mode=args.market_mode,
        )
        candidate_frames.append(candidates)
        net = pd.to_numeric(candidates.get("net_return"), errors="coerce")
        strategy_reports[strategy_id] = {
            "source": sources.get(strategy_id),
            "prediction_rows": int(len(work)),
            "local_threshold": float(threshold),
            "local_candidate_rows": int(len(local)),
            "finite_path_rows": int(finite_rows),
            "path_rows": int(total_rows),
            "path_coverage": float(path_coverage),
            "candidate_rows_before_auction_floor": int(len(candidates)),
            "candidate_net_hit_rate": float((net > 0.0).mean()) if len(net) else None,
            "candidate_mean_net_return": float(net.mean()) if len(net) else None,
        }

    candidate_table = _finalise_holdout_candidates(
        candidate_frames,
        rank_store=rank_store,
        rank_floor=PORTFOLIO_CANDIDATE_EXPORT_NORMALIZED_RANK_FLOOR,
    )
    output_dir = args.output_dir or (
        run_root / "policy_holdout_frozen_replay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "simple_policy_holdout_candidates.parquet"
    candidate_table.to_parquet(candidate_path, index=False)
    _write_simple_policy_candidate_metadata(
        candidate_table,
        output_path=output_dir / "simple_policy_holdout_candidates_metadata.json",
    )
    _write_delay_rejection_reports(candidate_table, output_dir=output_dir)
    _write_rank_threshold_band_reports(candidate_table, output_dir=output_dir)

    replay_report: dict[str, Any] | None = None
    if not args.skip_portfolio_replay and not candidate_table.empty:
        replay_report = run_portfolio_policy_replay(
            data_root=str(data_root),
            run_id=args.run_id,
            market_mode=args.market_mode,
            candidate_path=candidate_path,
            output_dir=output_dir / "portfolio_policy_replay",
            fixed_policy_config_path=run_root
            / "policy_params"
            / "optimized_portfolio_policy_config.json",
            ev_curve_candidate_path=run_root
            / "simple_policy_optimiser"
            / "simple_policy_candidates.parquet",
        )

    summary = {
        "generated_by": "scripts/evaluate_frozen_policy_holdout.py",
        "run_id": args.run_id,
        "market_mode": args.market_mode,
        "artifact_source_run_id": args.artifact_source_run_id or None,
        "deployment_path": str(deployment_path),
        "model_state_path": str(meta_state_path),
        "prediction_start": predict_start.isoformat(),
        "prediction_end": predict_end.isoformat(),
        "policy_optimiser_predict_end": policy_end.isoformat(),
        "rank_reference_source": str(rank_store.manifest_path),
        "candidate_path": str(candidate_path),
        "candidate_rows": int(len(candidate_table)),
        "candidate_rows_before_auction_floor": int(
            candidate_table.attrs.get("holdout_rows_before_auction_floor", len(candidate_table))
        ),
        "candidate_strategy_count": int(
            candidate_table["strategy_id"].nunique() if not candidate_table.empty else 0
        ),
        "strategy_reports": strategy_reports,
        "portfolio_replay_report_path": (
            str(output_dir / "portfolio_policy_replay" / "portfolio_policy_replay_report.json")
            if replay_report is not None
            else None
        ),
        "portfolio_replay_metrics": (
            replay_report.get("global_auction_metrics") if replay_report else None
        ),
        "source_validation": source_validation,
        "holdout_stage_view": holdout_stage_view,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
