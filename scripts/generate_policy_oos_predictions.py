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

import joblib
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
    out_dir = args.output_dir or (run_root / "policy_oos_predictions")
    if not meta_state_path.exists():
        raise SystemExit(f"Missing train-meta model state: {meta_state_path}")

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
    if not (pd.Timestamp(model_fit_end) < pd.Timestamp(policy_start)):
        raise SystemExit(
            f"Model fit cutoff is not before policy window: {model_fit_end} >= {policy_start}"
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

    full_state = joblib.load(meta_state_path)
    if not isinstance(full_state, dict) or not isinstance(full_state.get("bundle"), dict):
        raise SystemExit(f"Unexpected model_state_meta format: {meta_state_path}")
    bundle = full_state.get("bundle", {})
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
            stage_view=stage_view,
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
    model_state_hash = _sha256(meta_state_path)
    written: list[dict[str, Any]] = []
    for strategy_id, frame in sorted(frames.items()):
        df = _filter_rows_to_stage_view(frame, stage_view)
        df = _filter_policy_quote_rows(df, args.market_mode)
        if df.empty:
            continue
        df = df.copy()
        if "clf" not in df.columns and "oof_pred" in df.columns:
            df["clf"] = df["oof_pred"]
        df["policy_oos_source_model_fit_end"] = str(source_model_fit_end)
        df["policy_oos_generation_source"] = str(sources.get(strategy_id, "unknown"))
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
            "source_model_state_sha256": model_state_hash,
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
        }
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        written.append({"strategy_id": strategy_id, "path": str(path), **bounds, "rows": int(len(df))})

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
        "source_model_state_sha256": model_state_hash,
        "source_artifact_preflight_path": str(preflight_path),
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
