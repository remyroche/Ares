#!/usr/bin/env python3
"""Generate policy-OOS prediction handoff from train-meta-frozen artifacts.

This intentionally does not score the policy window with the full inference
final-fit bundle.  The output parquets are accepted by simple_policy_optimiser
only when their manifest proves the source model fit cutoff precedes the policy
prediction window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _expand_strategy_id_allowlist,
    _filter_policy_quote_rows,
    _filter_rows_to_stage_view,
    _generate_policy_predictions_from_models,
    _load_policy_stage_view,
    _load_slice_plan_source_validation,
)
from extreme_price_movements.inference.config import load_trained_symbol_universe  # noqa: E402
from extreme_price_movements.model_loader import load_model_bundle  # noqa: E402
from extreme_price_movements.policy_oos_provenance import (  # noqa: E402
    validate_policy_oos_source_artifacts,
    write_policy_oos_preflight_report,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strategy_id_from_meta_key(key: str) -> str:
    out = str(key)
    if out.endswith("_clf"):
        out = out[: -len("_clf")]
    if out.endswith("_tbm"):
        out = out[: -len("_tbm")]
    if out.endswith("_correctness"):
        out = out[: -len("_correctness")]
    return out


def _timestamp_bounds(df: pd.DataFrame) -> dict[str, Any]:
    if "timestamp" not in df.columns or df.empty:
        return {"min_timestamp": None, "max_timestamp": None}
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"min_timestamp": None, "max_timestamp": None}
    return {
        "min_timestamp": ts.min().isoformat(),
        "max_timestamp": ts.max().isoformat(),
    }


def _load_deployable_trained_universe(data_root: Path, run_id: str) -> set[str]:
    try:
        universe = {
            str(sym)
            for sym in load_trained_symbol_universe(str(data_root), str(run_id))
            if str(sym)
        }
    except Exception as exc:
        raise SystemExit(
            "Unable to load trained/inference universe for policy-OOS generation: "
            f"{exc}"
        ) from exc
    if not universe:
        raise SystemExit(
            "Refusing policy-OOS generation because the trained/inference universe is empty."
        )
    return universe


def _filter_policy_oos_to_trained_universe(
    df: pd.DataFrame,
    *,
    trained_universe: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "symbol" not in df.columns:
        raise SystemExit("Policy-OOS frame is missing required symbol column.")
    symbols = df["symbol"].astype(str)
    keep = symbols.isin(trained_universe)
    outside = sorted(set(symbols.loc[~keep]) - set(trained_universe))
    report = {
        "input_rows": int(len(df)),
        "input_symbols": int(symbols.nunique(dropna=True)),
        "trained_universe_symbols": int(len(trained_universe)),
        "kept_rows": int(keep.sum()),
        "dropped_rows": int((~keep).sum()),
        "dropped_symbols": int(len(outside)),
        "dropped_symbol_sample": outside[:30],
    }
    return df.loc[keep].copy(), report


def _stage_view_for_trained_universe(
    stage_view: dict[str, Any],
    *,
    trained_universe: set[str],
) -> dict[str, Any]:
    out = dict(stage_view)
    for key in ("symbols", "allowed_symbols"):
        values = out.get(key)
        if not values:
            continue
        out[key] = [str(sym) for sym in values if str(sym) in trained_universe]
    if not out.get("symbols") and not out.get("allowed_symbols"):
        out["allowed_symbols"] = sorted(trained_universe)
    return out


def _attach_policy_oos_contract_columns(
    df: pd.DataFrame,
    *,
    market_mode: str,
    source_model_fit_end: Any,
    generation_source: str,
) -> pd.DataFrame:
    out = df.copy()
    out["market_mode"] = str(market_mode)
    out["policy_oos_source_model_fit_end"] = str(source_model_fit_end)
    out["policy_oos_generation_source"] = str(generation_source)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--market-mode", choices=("spot", "perps"), default="perps")
    parser.add_argument(
        "--strategy-id",
        action="append",
        default=[],
        help="Optional deployed strategy id to generate. May be repeated.",
    )
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only validate artifact provenance; do not load models or score rows.",
    )
    args = parser.parse_args()

    started = time.monotonic()
    data_root = Path(args.data_root)
    run_root = data_root / "artifacts" / args.run_id
    slice_plan_path = run_root / "slices" / "slice_plan.json"
    meta_state_path = run_root / "models" / "model_state_meta.pkl"
    scoring_state_path = run_root / "models" / "trained_state.pkl"
    if not scoring_state_path.exists():
        scoring_state_path = meta_state_path
    out_dir = args.output_dir or (run_root / "policy_oos_predictions")
    if not meta_state_path.exists():
        raise SystemExit(f"Missing train-meta model state: {meta_state_path}")
    if not scoring_state_path.exists():
        raise SystemExit(f"Missing scoring model state: {scoring_state_path}")

    source_validation = _load_slice_plan_source_validation(slice_plan_path)
    exchange_context = source_validation.get("exchange_context") or {}
    if args.market_mode == "perps" and not os.environ.get("EPM_EXCHANGE"):
        exchange_id = str(exchange_context.get("exchange") or "").strip()
        if exchange_id:
            os.environ["EPM_EXCHANGE"] = exchange_id
    if not bool(source_validation.get("oos_policy_slice_verified", False)):
        raise SystemExit(
            "Slice plan does not verify a policy-OOS slice: "
            + json.dumps(source_validation, sort_keys=True, default=str)
        )
    model_fit_end = source_validation.get("policy_optimiser_fit_end")
    policy_start = source_validation.get("policy_optimiser_predict_start")
    if not model_fit_end or not policy_start:
        raise SystemExit("Slice plan is missing policy fit/predict timestamps.")
    temporal_oos = bool(source_validation.get("policy_holdout_temporal_disjoint", False))
    row_disjoint_oos = bool(
        source_validation.get("policy_holdout_fit_predict_disjoint", False)
    )
    if not temporal_oos and not row_disjoint_oos:
        raise SystemExit(
            "Slice plan does not prove temporal or row-disjoint policy-OOS safety: "
            + json.dumps(source_validation, sort_keys=True, default=str)
        )

    preflight = validate_policy_oos_source_artifacts(
        run_root=run_root,
        slice_plan_path=slice_plan_path,
        source_validation=source_validation,
    )
    preflight_path = out_dir / "preflight_report.json"
    write_policy_oos_preflight_report(preflight, preflight_path)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True, default=str))
        return 0 if bool(preflight.get("valid")) else 2
    if not bool(preflight.get("valid")):
        raise SystemExit(
            "Refusing to generate policy-OOS predictions because source artifact "
            f"provenance is not policy-OOS safe. See {preflight_path}: "
            + json.dumps(preflight.get("errors", []), sort_keys=True)
        )
    source_model_fit_end = preflight.get("source_model_fit_end") or model_fit_end

    stage_view, stage_name = _load_policy_stage_view(slice_plan_path)
    if stage_name != "policy_optimiser" or not stage_view:
        raise SystemExit(f"Missing policy_optimiser stage view in {slice_plan_path}")
    trained_universe = _load_deployable_trained_universe(data_root, args.run_id)
    scoring_stage_view = _stage_view_for_trained_universe(
        stage_view,
        trained_universe=trained_universe,
    )

    bundle = load_model_bundle(args.run_id, str(data_root))
    if not isinstance(bundle, dict):
        raise SystemExit(f"Unexpected scoring bundle format for run_id={args.run_id}")
    full_state = {"bundle": bundle}
    meta_models = bundle.get("meta_models", {}) if isinstance(bundle, dict) else {}
    if not meta_models:
        raise SystemExit(f"No meta models in {meta_state_path}")

    allowlist = set(args.strategy_id or [])
    if not allowlist:
        allowlist = {_strategy_id_from_meta_key(key) for key in meta_models}
    allowlist = _expand_strategy_id_allowlist(sorted(allowlist))

    # The policy handoff rows are timestamp/symbol candidates; executable labels
    # and paths are built by simple_policy_optimiser from OHLCV/1m data.  We
    # enable this only inside this generator and record it in the manifest.
    old_feature_only = os.environ.get("EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY")
    os.environ["EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY"] = "1"
    try:
        frames, sources = _generate_policy_predictions_from_models(
            data_root=str(data_root),
            run_id=args.run_id,
            stage_view=scoring_stage_view,
            max_strategies=args.max_strategies,
            strategy_ids_allowlist=allowlist,
            market_mode=args.market_mode,
            full_state_override=full_state,
            source_tag="generated_from_train_meta_state",
        )
    finally:
        if old_feature_only is None:
            os.environ.pop("EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY", None)
        else:
            os.environ["EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY"] = old_feature_only

    out_dir.mkdir(parents=True, exist_ok=True)
    model_state_hash = _sha256(scoring_state_path)
    meta_state_hash = _sha256(meta_state_path)
    written: list[dict[str, Any]] = []
    for strategy_id, frame in sorted(frames.items()):
        df = _filter_rows_to_stage_view(frame, stage_view)
        df = _filter_policy_quote_rows(df, args.market_mode)
        if df.empty:
            continue
        df, universe_report = _filter_policy_oos_to_trained_universe(
            df,
            trained_universe=trained_universe,
        )
        if df.empty:
            raise SystemExit(
                "Policy-OOS generation produced no deployable rows for "
                f"{strategy_id} after trained-universe filtering: "
                + json.dumps(universe_report, sort_keys=True, default=str)
            )
        if "clf" not in df.columns and "oof_pred" in df.columns:
            df["clf"] = df["oof_pred"]
        df = _attach_policy_oos_contract_columns(
            df,
            market_mode=args.market_mode,
            source_model_fit_end=source_model_fit_end,
            generation_source=sources.get(strategy_id, "unknown"),
        )
        path = out_dir / f"policy_oos_{strategy_id}_clf.parquet"
        df.to_parquet(path, index=False)
        bounds = _timestamp_bounds(df)
        manifest = {
            "generated_by": "scripts/generate_policy_oos_predictions.py",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "run_id": args.run_id,
            "strategy_id": strategy_id,
            "market_mode": args.market_mode,
            "rows": int(len(df)),
            **bounds,
            "model_provenance": "train_meta_frozen_model_state",
            "generated_from_final_fit_bundle": False,
            "source_model_state_path": str(meta_state_path),
            "source_model_state_sha256": meta_state_hash,
            "scoring_model_state_path": str(scoring_state_path),
            "scoring_model_state_sha256": model_state_hash,
            "source_model_fit_end": str(source_model_fit_end),
            "policy_predict_start": str(policy_start),
            "policy_predict_end": str(
                source_validation.get("policy_optimiser_predict_end")
            ),
            "prediction_source": str(sources.get(strategy_id, "unknown")),
            "candidate_rows_source": "policy_slice_feature_events",
            "executable_path_source": "simple_policy_optimiser_recomputes_from_ohlcv_and_execution_1m",
            "rank_normalization": "simple_policy_optimiser recalculates calibrated_score/rank_pct from clf",
            "slice_contract": source_validation,
            "source_artifact_preflight": preflight,
            "trained_universe_filter": universe_report,
        }
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        written.append(
            {
                "strategy_id": strategy_id,
                "path": str(path),
                **bounds,
                "rows": int(len(df)),
                "trained_universe_filter": universe_report,
            }
        )

    summary = {
        "generated_by": "scripts/generate_policy_oos_predictions.py",
        "run_id": args.run_id,
        "market_mode": args.market_mode,
        "elapsed_seconds": time.monotonic() - started,
        "requested_strategy_ids": sorted(allowlist),
        "written": written,
        "source_model_fit_end": str(source_model_fit_end),
        "policy_predict_start": str(policy_start),
        "source_model_state_path": str(meta_state_path),
        "source_model_state_sha256": meta_state_hash,
        "scoring_model_state_path": str(scoring_state_path),
        "scoring_model_state_sha256": model_state_hash,
        "source_artifact_preflight_path": str(preflight_path),
        "trained_universe_symbols": len(trained_universe),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if not written:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
