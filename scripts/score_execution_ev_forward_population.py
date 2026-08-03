#!/usr/bin/env python3
"""Score a future execution-EV population under the frozen source contract.

The scorer loads the side-local final direct/capture heads, applies the frozen
base-margin interaction, performs a causal 21-day side-local isotonic mapping,
and ranks one pooled global top 10% across all supplied timestamps and sides.
It reports fixed 0/25/50-bps admission floors and permits zero trades.

Optional resolved updates may extend the frozen calibrator seed.  At decision
time ``t`` only updates with ``execution_label_end_utc < t`` are visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_execution_ev_forward_calibrator_seed import (  # noqa: E402
    BASELINE,
    DECISION,
    IDENTITY,
    RESOLUTION,
    SIDES,
    TARGET,
    _load_final_head_models,
    interaction_score,
)


SCHEMA = "execution_ev_forward_scored_population_v1"
DEFAULT_HEAD_ROOT = Path(
    "data_perp/artifacts/execution_ev_forward_final_heads_20260728_v1"
)
DEFAULT_STATE = Path(
    "data_perp/artifacts/execution_ev_forward_calibrator_seed_20260728_v1/"
    "causal_recent_ev_state.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/execution_ev_forward_preoutcome_20260728_v1"
)
AVAILABILITY_COLUMNS = (
    "feature_available_at",
    "base_available_at",
    "residual_available_at",
    "peak_mfe_available_at",
    "path_catboost_available_at",
    "clean_probability_available_at",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _strict_boolean(values: pd.Series, *, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{field} contains null booleans")
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{field} must contain only true/false or 1/0")
    return parsed.astype(bool)


def validate_resolved_updates(
    updates: pd.DataFrame,
    candidates: pd.DataFrame,
    seed_history: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the causal update ledger before it can affect mapping."""

    required = {
        *IDENTITY,
        DECISION,
        RESOLUTION,
        TARGET,
    }
    missing = sorted(required.difference(updates.columns))
    if missing:
        raise ValueError(f"resolved update columns missing: {missing}")
    work = updates.copy()
    if work.duplicated(list(IDENTITY)).any() or work["candidate_id"].duplicated().any():
        raise ValueError("resolved updates contain duplicate identities")
    for column in (DECISION, RESOLUTION):
        if not isinstance(work[column].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"resolved update {column} must be timezone-aware")
        work[column] = work[column].dt.tz_convert("UTC")
    if (work[RESOLUTION] <= work[DECISION]).any():
        raise ValueError("resolved update label end must be after its decision")
    numeric = work[[TARGET]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("resolved update economics must be finite")
    candidate_keys = set(
        candidates.loc[:, list(IDENTITY)].itertuples(index=False, name=None)
    )
    update_keys = set(
        work.loc[:, list(IDENTITY)].itertuples(index=False, name=None)
    )
    if not update_keys.issubset(candidate_keys):
        raise ValueError("resolved updates are not a subset of future candidates")
    score_reference = candidates.set_index(list(IDENTITY))[
        "frozen_margin_capture_interaction_raw"
    ]
    update_index = pd.MultiIndex.from_frame(work.loc[:, list(IDENTITY)])
    expected_scores = score_reference.loc[update_index].to_numpy(dtype=float)
    if "frozen_margin_capture_interaction_raw" in work:
        supplied_scores = pd.to_numeric(
            work["frozen_margin_capture_interaction_raw"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(supplied_scores).all():
            raise ValueError("resolved update scores must be finite")
        if not np.allclose(
            supplied_scores,
            expected_scores,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "resolved update score does not match frozen candidate score"
            )
    else:
        work["frozen_margin_capture_interaction_raw"] = expected_scores
    seed_ids = set(seed_history["candidate_id"].astype(str))
    if seed_ids.intersection(work["candidate_id"].astype(str)):
        raise ValueError("resolved updates overlap the frozen calibrator seed")
    return work


def causal_recent_isotonic_mapping(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    *,
    lookback_days: int,
    minimum_side_rows: int = 100,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Map each decision using only already-resolved trailing observations."""

    decision = pd.to_datetime(candidates[DECISION], utc=True, errors="raise")
    history_resolution = pd.to_datetime(history[RESOLUTION], utc=True, errors="raise")
    output = np.full(len(candidates), np.nan, dtype=float)
    reports: list[dict[str, Any]] = []
    for timestamp in pd.Index(decision.unique()).sort_values():
        timestamp = pd.Timestamp(timestamp)
        decision_positions = np.flatnonzero(decision.eq(timestamp).to_numpy())
        lower = timestamp - pd.Timedelta(days=int(lookback_days))
        causal = history.loc[
            history_resolution.lt(timestamp) & history_resolution.ge(lower)
        ]
        for side in SIDES:
            local_positions = decision_positions[
                candidates.iloc[decision_positions]["side_name"]
                .astype(str)
                .eq(side)
                .to_numpy()
            ]
            if not len(local_positions):
                continue
            train = causal.loc[causal["side_name"].astype(str).eq(side)]
            if len(train) < minimum_side_rows:
                raise ValueError(
                    f"{side} has only {len(train)} causal mapping rows at {timestamp}"
                )
            mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mapper.fit(
                train["frozen_margin_capture_interaction_raw"].to_numpy(dtype=float),
                train[TARGET].to_numpy(dtype=float),
            )
            output[local_positions] = mapper.predict(
                candidates.iloc[local_positions][
                    "frozen_margin_capture_interaction_raw"
                ].to_numpy(dtype=float)
            )
            reports.append(
                {
                    "decision_utc": timestamp,
                    "side": side,
                    "history_rows": int(len(train)),
                    "history_resolution_max_utc": train[RESOLUTION].max(),
                }
            )
    if not np.isfinite(output).all():
        raise ValueError("causal mapping left non-finite forward scores")
    return output, reports


def apply_global_admission(
    frame: pd.DataFrame,
    *,
    score_column: str = "mapped_execution_ev",
    top_k_fraction: float = 0.10,
) -> pd.DataFrame:
    """Rank exactly one pooled global book and apply fixed EV floors."""

    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    output = frame.copy()
    score = pd.to_numeric(output[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError("global admission score is not finite")
    count = max(1, int(math.ceil(top_k_fraction * len(output))))
    if "candidate_id" not in output:
        raise ValueError("global admission requires candidate_id for tie-breaking")
    candidate_id = output["candidate_id"].astype(str).to_numpy()
    if pd.Series(candidate_id).duplicated().any():
        raise ValueError("global admission candidate_id must be unique")
    order = np.lexsort((candidate_id, -score))
    capacity = np.zeros(len(output), dtype=bool)
    capacity[order[:count]] = True
    output["global_top10_capacity_member"] = capacity
    for floor_bps in (0, 25, 50):
        output[f"globally_admitted_floor_{floor_bps}bps"] = capacity & (
            score > floor_bps / 10_000.0
        )
    output["globally_admitted"] = output["globally_admitted_floor_0bps"]
    output["global_rank"] = np.empty(len(output), dtype=np.int64)
    output.loc[output.index[order], "global_rank"] = np.arange(1, len(output) + 1)
    return output


def _score_raw_heads(
    frame: pd.DataFrame,
    *,
    head_root: Path,
    head_manifest: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    state: Mapping[str, Any],
) -> pd.DataFrame:
    models = _load_final_head_models(head_root, head_manifest)
    parts: list[pd.DataFrame] = []
    for side in SIDES:
        local = frame.loc[frame["side_name"].astype(str).eq(side)].copy()
        if local.empty:
            raise ValueError(f"future population does not contain {side}")
        features = list(feature_contract["feature_columns_by_side"][side])
        for column in features:
            prefix = "catboost_archetype__"
            if column.startswith(prefix) and column not in local:
                level = column[len(prefix) :]
                local[column] = (
                    local["catboost_archetype"].astype(str).eq(level).astype("float32")
                )
        missing = sorted(set(features).difference(local.columns))
        if missing:
            raise ValueError(f"{side} future-head features missing: {missing}")
        x = local.loc[:, features].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(x.to_numpy(dtype=np.float32)).all():
            raise ValueError(f"{side} future-head features are not finite")
        local["final_direct_net_raw"] = local[BASELINE].to_numpy(dtype=float) + np.asarray(
            models[side]["direct"].predict(x), dtype=float
        )
        local["final_capture_probability"] = np.asarray(
            models[side]["capture"].predict_proba(x)[:, 1], dtype=float
        )
        standardization = state["sides"][side]["standardization"]
        local["frozen_margin_capture_interaction_raw"], _ = interaction_score(
            local["final_direct_net_raw"].to_numpy(dtype=float),
            local["final_capture_probability"].to_numpy(dtype=float),
            local["base_margin_to_cutoff_z"].to_numpy(dtype=float),
            contract=state["interaction"],
            **standardization,
        )
        parts.append(local)
    return pd.concat(parts).sort_index()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    state = json.loads(args.calibrator_state.read_text(encoding="utf-8"))
    if state.get("schema") != "execution_ev_forward_calibrator_seed_v1":
        raise ValueError("unexpected calibrator-state schema")
    history_path = Path(state["history"]["path"])
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    if _sha256(history_path) != state["history"]["sha256"]:
        raise ValueError("calibrator seed-history hash mismatch")
    head_manifest_path = args.head_root / "manifest.json"
    feature_contract_path = args.head_root / "feature_contract.json"
    head_manifest = json.loads(head_manifest_path.read_text(encoding="utf-8"))
    feature_contract = json.loads(feature_contract_path.read_text(encoding="utf-8"))
    if _sha256(feature_contract_path) != head_manifest["feature_contract"]["sha256"]:
        raise ValueError("final-head feature contract hash mismatch")
    frame = pd.read_parquet(args.preentry)
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("future pre-entry population contains duplicate identities")
    decision = frame[DECISION]
    if not isinstance(decision.dtype, pd.DatetimeTZDtype):
        raise ValueError("future decision timestamps must be stored timezone-aware")
    frame[DECISION] = decision.dt.tz_convert("UTC")
    cutoff = pd.Timestamp(state["first_decision_exclusive_utc"])
    if frame[DECISION].min() <= cutoff:
        raise ValueError("future population is not strictly after the frozen cutoff")
    for column in AVAILABILITY_COLUMNS:
        if column not in frame:
            raise ValueError(f"future availability column missing: {column}")
        available = frame[column]
        if not isinstance(available.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{column} must be stored timezone-aware")
        frame[column] = available.dt.tz_convert("UTC")
        if (frame[column] > frame[DECISION]).any():
            raise ValueError(f"{column} occurs after the decision")
    scored = _score_raw_heads(
        frame,
        head_root=args.head_root,
        head_manifest=head_manifest,
        feature_contract=feature_contract,
        state=state,
    )
    seed_history = pd.read_parquet(history_path)
    if args.resolved_updates is not None:
        updates = pd.read_parquet(args.resolved_updates)
        updates = validate_resolved_updates(updates, scored, seed_history)
        history = pd.concat([seed_history, updates], ignore_index=True)
    else:
        history = seed_history
    scored["mapped_execution_ev"], mapping_report = causal_recent_isotonic_mapping(
        scored,
        history,
        lookback_days=int(state["lookback_days"]),
    )
    scored = apply_global_admission(scored)
    scored["direct_ev_available_at"] = scored[DECISION]
    scored["capture_probability_available_at"] = scored[DECISION]
    scored["mapping_available_at"] = scored[DECISION]
    scored["score_contract"] = "frozen_final_heads_margin_interaction_causal_21d"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "scored_population.parquet"
    scored.to_parquet(output_path, index=False, compression="zstd")
    daily = (
        scored.assign(utc_date=scored[DECISION].dt.floor("D"))
        .groupby("utc_date", as_index=False)
        .agg(
            rows=("candidate_id", "size"),
            both_sides=("side_name", lambda values: len(set(values)) == 2),
        )
    )
    # A complete day is asserted only when supplied coverage explicitly marks
    # it complete.  Row presence alone must not upgrade a partial day.
    if args.complete_days is not None:
        supplied = pd.read_csv(args.complete_days)
        supplied["utc_date"] = pd.to_datetime(
            supplied["utc_date"], utc=True, errors="raise"
        ).dt.floor("D")
        daily = daily.merge(
            supplied[["utc_date", "complete"]],
            on="utc_date",
            how="left",
            validate="one_to_one",
        )
        daily["complete"] = _strict_boolean(
            daily["complete"].fillna(False), field="complete"
        )
    else:
        daily["complete"] = False
    daily_path = args.output_dir / "daily_coverage.csv"
    daily.to_csv(daily_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "scored_preoutcome_population_not_performance_evidence",
        "contract": {
            "mapping": "causal_recent_side_isotonic_ev_21d",
            "ranking": "one pooled global top10 across timestamps and sides",
            "admission_floors_bps": [0, 25, 50],
            "allow_zero_trades": True,
            "no_timestamp_side_asset_quota": True,
        },
        "rows": int(len(scored)),
        "globally_admitted_rows": int(scored["globally_admitted"].sum()),
        "decision_min_utc": scored[DECISION].min(),
        "decision_max_utc": scored[DECISION].max(),
        "mapping_updates": mapping_report,
        "inputs": {
            "preentry": {"path": args.preentry, "sha256": _sha256(args.preentry)},
            "calibrator_state": {
                "path": args.calibrator_state,
                "sha256": _sha256(args.calibrator_state),
            },
            "head_manifest": {
                "path": head_manifest_path,
                "sha256": _sha256(head_manifest_path),
            },
            "resolved_updates": (
                {
                    "path": args.resolved_updates,
                    "sha256": _sha256(args.resolved_updates),
                }
                if args.resolved_updates is not None
                else None
            ),
        },
        "outputs": {
            "scored_population": {
                "path": output_path,
                "sha256": _sha256(output_path),
            },
            "daily_coverage": {"path": daily_path, "sha256": _sha256(daily_path)},
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preentry", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    parser.add_argument("--calibrator-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--resolved-updates", type=Path)
    parser.add_argument(
        "--complete-days",
        type=Path,
        help="CSV with utc_date,complete; absence leaves every day incomplete",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(_parser())
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "globally_admitted_rows": result["globally_admitted_rows"],
            },
            indent=2,
        )
    )
