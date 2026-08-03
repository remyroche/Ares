#!/usr/bin/env python3
"""Materialise the feasible Mar--Jun 2025 blocked-OOF context gap.

This is not a four-arm scorer.  It joins accepted residual OOF candidates to
their exact 12-hour economics, then reuses the established blocked historical
regime/transition adapter.  July--December deliberately remain absent because
there is no compatible frozen base/residual OOF candidate stream to score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_candidate_oof_regime_transition_adapter import materialize_adapter

ID = ("candidate_id", "__ts__", "__symbol__", "side_name")
SCHEMA = "blocked_oof_context_gap_2025_marjun_v1"
OUT = ROOT / "data_perp/artifacts/blocked_oof_context_gap_2025_marjun_20260730_v1"
HISTORICAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet"
CONTINUATION = ROOT / "data_perp/artifacts/mayjun2025_canonical_residual_continuation_20260730_v1/oof_predictions.parquet"
FEBAPR_LABELS = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet"
MAYJUN_LABELS = ROOT / "data_perp/artifacts/mayjul2025_execution_ev_common30_labels_20260727_v2/labels.parquet"
HOURLY_REGIME = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1/hourly_state_calendar.parquet"
HOURLY_TRANSITION = ROOT / "data_perp/artifacts/regime_transition_research_20260726_v3/hourly_transition_dataset.parquet"


class GapContextError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_scores(path: Path, *, months: set[int], source: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    required = {*ID, "residual_expected_ev", "residual_is_oof", "__first_touch_target_soft__"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise GapContextError(f"{source} misses required accepted residual fields: {missing}")
    output = frame.loc[frame["__ts__"].dt.month.isin(months) & frame["residual_is_oof"].astype(bool)].copy()
    output = output.loc[output["residual_expected_ev"].notna()].copy()
    if output.empty or output.duplicated(list(ID)).any():
        raise GapContextError(f"{source} has no unique residual OOF candidate rows")
    output["score_source"] = source
    output["baseline_context_free_raw_score"] = pd.to_numeric(output["residual_expected_ev"], errors="raise")
    return output


def _attach_exact_labels(scores: pd.DataFrame, labels_path: Path, *, expected_months: set[int]) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    labels = labels.loc[labels["__ts__"].dt.month.isin(expected_months)].copy()
    required = {*ID, "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "execution_label_available_at"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise GapContextError(f"exact label source misses {missing}")
    labels = labels.loc[:, [*ID, "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "execution_label_available_at"]]
    if labels.duplicated("candidate_id").any():
        raise GapContextError("exact label identities are duplicated")
    # The accepted Feb--Apr residual ledger retains feature-store symbol
    # spelling (``BTC_USD:USD``), while the exact execution replay retains the
    # simulator spelling (``BTC/USD:USD``).  Candidate ID is the invariant;
    # prove side/time separately and retain the score ledger's identity.
    check = labels.loc[:, ["candidate_id", "__ts__", "side_name"]].rename(columns={"__ts__": "__label_ts__", "side_name": "__label_side__"})
    values = labels.drop(columns=["__ts__", "__symbol__", "side_name"])
    result = scores.merge(values, on="candidate_id", how="left", validate="one_to_one", suffixes=("__score", ""))
    result = result.merge(check, on="candidate_id", how="left", validate="one_to_one")
    if not (result["__ts__"].eq(result.pop("__label_ts__")) & result["side_name"].eq(result.pop("__label_side__"))).all():
        raise GapContextError("candidate IDs disagree on score/label timestamp or side")
    if len(result) != len(scores) or result[["execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"]].isna().any().any():
        raise GapContextError("residual OOF score rows lack exact gross/cost/net labels")
    result["execution_label_available_at"] = pd.to_datetime(result["execution_label_available_at"], utc=True, errors="raise")
    return result


def _assemble(historical: pd.DataFrame, continuation: pd.DataFrame, febapr_labels: Path, mayjun_labels: Path) -> pd.DataFrame:
    march_april = _attach_exact_labels(historical, febapr_labels, expected_months={3, 4})
    may_june = _attach_exact_labels(continuation, mayjun_labels, expected_months={5, 6})
    output = pd.concat([march_april, may_june], ignore_index=True, sort=False).sort_values(["__ts__", "candidate_id"], kind="stable")
    if output.duplicated(list(ID)).any():
        raise GapContextError("assembled 2025 residual OOF identities overlap")
    if not output["residual_is_oof"].astype(bool).all():
        raise GapContextError("non-OOF residual row reached the 2025 gap materialization")
    if output["execution_label_available_at"].lt(output["__ts__"]).any():
        raise GapContextError("exact label availability predates score timestamp")
    return output


def run(*, output_dir: Path = OUT, historical: Path = HISTORICAL, continuation: Path = CONTINUATION,
        febapr_labels: Path = FEBAPR_LABELS, mayjun_labels: Path = MAYJUN_LABELS,
        hourly_regime: Path = HOURLY_REGIME, hourly_transition: Path = HOURLY_TRANSITION) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    scores = _assemble(_load_scores(Path(historical), months={3, 4}, source="accepted_febapr_residual_oof"),
                       _load_scores(Path(continuation), months={5, 6}, source="accepted_mayjun_residual_continuation_oof"),
                       Path(febapr_labels), Path(mayjun_labels))
    stage = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-"))
    try:
        score_path = stage / "candidate_scores_with_exact_labels.parquet"
        scores.to_parquet(score_path, index=False, compression="zstd")
        context_dir = stage / "candidate_oof_regime_transition"
        materialize_adapter(candidates_path=score_path, hourly_regime_path=Path(hourly_regime), hourly_transition_path=Path(hourly_transition),
                            output_dir=context_dir, evaluation_start="2025-03-01T00:00:00Z", evaluation_end="2025-07-01T00:00:00Z",
                            # Historical candidate OOF context uses the
                            # three-state regime geometry.  Preserve that
                            # dimensional contract so state summaries remain
                            # comparable in the later frozen challenger.
                            frequency="month", purge_hours=12, n_components=3, max_features=32, max_lag_hours=2, seed=52)
        context = pd.read_parquet(context_dir / "candidate_oof_regime_transition.parquet")
        if len(context) != len(scores) or context.duplicated(list(ID)).any():
            raise GapContextError("blocked OOF context does not retain exact candidate support")
        coverage = scores.assign(month=scores["__ts__"].dt.strftime("%Y-%m")).groupby(["month", "score_source"], as_index=False).agg(
            candidate_rows=("candidate_id", "size"), label_available_max=("execution_label_available_at", "max"),
            residual_oof=("residual_is_oof", "all"))
        coverage.to_csv(stage / "coverage_by_month.csv", index=False)
        blocker = pd.DataFrame([
            {"period": "2022-01..2023-03", "status": "BLOCKED", "reason": "No candidate-keyed blocked OOF regime+transition sidecar; available base/residual reconstruction begins before context support."},
            {"period": "2025-01..2025-02", "status": "BLOCKED", "reason": "No accepted residual OOF stream: February is declared base-passthrough warm-up and January base OOF is absent."},
            {"period": "2025-07..2025-12", "status": "BLOCKED", "reason": "No compatible frozen base/residual OOF candidate score stream exists; exact labels alone cannot manufacture a scorer."},
            {"period": "2025-03..2025-06", "status": "MATERIALIZED", "reason": "Accepted residual OOF plus exact labels and newly blocked OOF regime/transition context."},
        ])
        blocker.to_csv(stage / "coverage_blocker_ledger.csv", index=False)
        outputs = [score_path, context_dir / "candidate_oof_regime_transition.parquet", context_dir / "hourly_regime_oof.parquet", context_dir / "hourly_transition_oof.parquet", stage / "coverage_by_month.csv", stage / "coverage_blocker_ledger.csv"]
        manifest = {"schema": SCHEMA, "status": "PARTIAL_2025_BLOCKED_OOF_CONTEXT_MATERIALIZED", "scope": "March--June 2025 only; not a four-arm score source and not promotion evidence",
                    "score_contract": "accepted residual_is_oof=true only; no February base-passthrough or in-sample score", "context_contract": "same candidate OOF regime/transition adapter as historical reconstruction; regime pre-block only; transition labels resolved before each fold; layers remain separate",
                    "exact_economics": "candidate-local exact 12h net/gross/cost labels are attached only for later blocked-OOF arm training", "coverage": coverage.to_dict("records"),
                    "inputs": {str(path): _sha(Path(path)) for path in (historical, continuation, febapr_labels, mayjun_labels, hourly_regime, hourly_transition)},
                    "outputs": {str(path.relative_to(stage)): _sha(path) for path in outputs}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, destination)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
