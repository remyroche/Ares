#!/usr/bin/env python3
"""Freeze provenance-safe candidate-context score arms before 2026.

The runner deliberately trains only on a common panel for which both the
residual score and the current-regime / transition context are blocked OOF.
It is a *fixed* regularised linear overlay, not an HPO exercise.  The output
models can subsequently score an authoritative 2026 sidecar, but this runner
will never read (or learn from) 2026 data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
ARMS = ("baseline_context_free", "regime_only", "transition_only", "combined")
SCHEMA = "frozen_contextual_score_arms_v1"
FREEZE = pd.Timestamp("2025-07-01T00:00:00Z")
OUT = ROOT / "data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1"
HIST_SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
HIST_CONTEXT = ROOT / "data_perp/artifacts/reconstructed_2023apr_2024_candidate_oof_regime_transition_20260730_v1/candidate_oof_regime_transition.parquet"
GAP = ROOT / "data_perp/artifacts/blocked_oof_context_gap_2025_marjun_20260730_v1"

# State IDs and raw regime posterior coordinates have no stable semantic
# ordering across independently blocked OOF geometry fits.  We retain only
# cardinality-independent summaries.  Transition fields are named semantic
# probabilities and therefore have an explicit compatible contract.
REGIME_FEATURES = (
    "regime_state_entropy", "regime_state_margin", "regime_state_uncertainty",
    "regime_state_ood_score",
)
TRANSITION_FEATURES = (
    "transition_active_probability", "transition_state_entropy",
    "transition_state_margin", "transition_state_uncertainty", "transition_state_ood_score",
    "transition_state_p__stable", "transition_state_p__approach",
    "transition_state_p__immediate_lead", "transition_state_p__transition",
    "transition_state_p__acceleration", "transition_state_p__early_destination",
    "transition_state_p__settled_destination",
)
BASE_FEATURES = ("baseline_context_free_raw_score", "side_is_long")


class FrozenContextError(RuntimeError):
    """Raised when the frozen training contract cannot be proven."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def _canonical(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise FrozenContextError(f"{role} lacks candidate identity fields: {missing}")
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower().str.strip()
    out["candidate_id"] = out["candidate_id"].astype(str)
    if out.duplicated(list(IDENTITY)).any() or not out.side_name.isin(("long", "short")).all():
        raise FrozenContextError(f"{role} has invalid or duplicate exact candidate identities")
    return out


def _exact_join(scores: pd.DataFrame, context: pd.DataFrame, *, source: str) -> pd.DataFrame:
    scores, context = _canonical(scores, role=f"{source} scores"), _canonical(context, role=f"{source} context")
    if len(scores) != len(context):
        raise FrozenContextError(f"{source} score/context support differs ({len(scores)} != {len(context)})")
    required = [*REGIME_FEATURES, *TRANSITION_FEATURES, "regime_available_utc", "transition_available_utc"]
    missing = sorted(set(required).difference(context.columns))
    if missing:
        raise FrozenContextError(f"{source} context misses blocked-OOF fields: {missing}")
    context = context.loc[:, [*IDENTITY, *required]].copy()
    for field in ("regime_available_utc", "transition_available_utc"):
        context[field] = pd.to_datetime(context[field], utc=True, errors="raise")
        if context[field].gt(context["__ts__"]).any():
            raise FrozenContextError(f"{source} {field} is unavailable at candidate time")
    joined = scores.merge(context, on=list(IDENTITY), how="inner", validate="one_to_one", sort=False)
    if len(joined) != len(scores):
        raise FrozenContextError(f"{source} exact context join lost candidates")
    return joined


def _historical_panel(scores_path: Path, context_path: Path) -> pd.DataFrame:
    score = pd.read_parquet(scores_path)
    required = {"score_residual_expected_ev", "residual_is_oof", "execution_net_ev_12h"}
    missing = sorted(required.difference(score.columns))
    if missing:
        raise FrozenContextError(f"historical score source misses {missing}")
    score = score.loc[score["residual_is_oof"].astype(bool)].copy()
    # Context support begins in April 2023.  Earlier reconstructed rows are a
    # separate research population and cannot be silently matched by a looser
    # timestamp/asof rule.
    score["__ts__"] = pd.to_datetime(score["__ts__"], utc=True, errors="raise")
    score = score.loc[score["__ts__"].ge(pd.Timestamp("2023-04-01T00:00:00Z"))].copy()
    score["baseline_context_free_raw_score"] = pd.to_numeric(score["score_residual_expected_ev"], errors="raise")
    # The historical stack manifest fixes a signal+1h decision and [decision,
    # decision+12h) economics horizon.  Deriving this field retains the exact
    # resolved-before-freeze requirement even though old output did not carry
    # the availability timestamp explicitly.
    score["execution_label_available_at"] = pd.to_datetime(score["__ts__"], utc=True) + pd.Timedelta(hours=13)
    return _exact_join(score, pd.read_parquet(context_path), source="historical")


def _gap_panel(gap_dir: Path) -> pd.DataFrame:
    score_path = gap_dir / "candidate_scores_with_exact_labels.parquet"
    context_path = gap_dir / "candidate_oof_regime_transition/candidate_oof_regime_transition.parquet"
    score = pd.read_parquet(score_path)
    required = {"baseline_context_free_raw_score", "residual_is_oof", "execution_net_ev_12h", "execution_label_available_at"}
    missing = sorted(required.difference(score.columns))
    if missing:
        raise FrozenContextError(f"2025 gap score source misses {missing}")
    if not score["residual_is_oof"].astype(bool).all():
        raise FrozenContextError("2025 gap contains a non-OOF residual score")
    score["execution_label_available_at"] = pd.to_datetime(score["execution_label_available_at"], utc=True, errors="raise")
    return _exact_join(score, pd.read_parquet(context_path), source="2025 gap")


def build_training_panel(historical: pd.DataFrame, gap: pd.DataFrame, *, freeze: pd.Timestamp = FREEZE) -> pd.DataFrame:
    """Combine the exact common panels and reject labels resolved after freeze."""
    panel = pd.concat([historical, gap], ignore_index=True, sort=False)
    panel = _canonical(panel, role="combined blocked-OOF panel")
    if panel.duplicated(list(IDENTITY)).any():
        raise FrozenContextError("historical and 2025 gap panels overlap")
    panel["execution_label_available_at"] = pd.to_datetime(panel["execution_label_available_at"], utc=True, errors="raise")
    if panel["execution_label_available_at"].lt(panel["__ts__"]).any():
        raise FrozenContextError("outcome availability predates its candidate")
    panel = panel.loc[panel["execution_label_available_at"].lt(freeze)].copy()
    if panel.empty or panel["__ts__"].min() < pd.Timestamp("2023-04-01T00:00:00Z"):
        raise FrozenContextError("training panel has unexpected temporal support")
    fields = ["baseline_context_free_raw_score", "execution_net_ev_12h", *REGIME_FEATURES, *TRANSITION_FEATURES]
    panel.loc[:, fields] = panel.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    if np.isinf(panel.loc[:, fields].to_numpy(float)).any():
        raise FrozenContextError("training panel has infinite numeric values")
    panel["side_is_long"] = panel["side_name"].eq("long").astype(float)
    return panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def feature_sets() -> dict[str, list[str]]:
    return {
        "baseline_context_free": list(BASE_FEATURES),
        "regime_only": [*BASE_FEATURES, *REGIME_FEATURES],
        "transition_only": [*BASE_FEATURES, *TRANSITION_FEATURES],
        "combined": [*BASE_FEATURES, *REGIME_FEATURES, *TRANSITION_FEATURES],
    }


def fit_frozen_arms(panel: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit exactly one fixed Ridge overlay per declared arm; no HPO or selection."""
    target = pd.to_numeric(panel["execution_net_ev_12h"], errors="raise").to_numpy(float)
    models: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []
    for arm, features in feature_sets().items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=2.0)),
        ])
        model.fit(panel.loc[:, features], target)
        prediction = np.asarray(model.predict(panel.loc[:, features]), dtype=float)
        models[arm] = model
        rows.append({"arm": arm, "rows": int(len(panel)), "features": len(features),
                     "train_correlation": float(pd.Series(prediction).corr(pd.Series(target), method="spearman")),
                     "train_net_ev_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
                     "fixed_ridge_alpha": 2.0})
    return models, pd.DataFrame(rows)


def run(*, output_dir: Path = OUT, historical_scores: Path = HIST_SCORES,
        historical_context: Path = HIST_CONTEXT, gap_dir: Path = GAP) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    historical = _historical_panel(Path(historical_scores), Path(historical_context))
    gap = _gap_panel(Path(gap_dir))
    panel = build_training_panel(historical, gap)
    models, metrics = fit_frozen_arms(panel)
    stage = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-"))
    try:
        panel_path = stage / "blocked_oof_training_panel.parquet"
        panel.to_parquet(panel_path, index=False, compression="zstd")
        metrics.to_csv(stage / "frozen_fit_diagnostics.csv", index=False)
        model_paths = []
        for arm, model in models.items():
            path = stage / f"{arm}.joblib"
            joblib.dump(model, path)
            model_paths.append(path)
        by_month = panel.assign(month=panel["__ts__"].dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
            candidate_rows=("candidate_id", "size"), label_available_max=("execution_label_available_at", "max"),
            mean_net_ev=("execution_net_ev_12h", "mean"),
        )
        by_month.to_csv(stage / "training_coverage_by_month.csv", index=False)
        contract = {
            "status": "FROZEN_2022_2025_CANDIDATE_CONTEXT", "training_start_utc": "2023-04-01T00:00:00Z",
            "training_end_exclusive_utc": FREEZE.isoformat(), "arms": list(ARMS),
            "target": "execution_net_ev_12h", "fit": "fixed Ridge(alpha=2.0), median imputation, standard scaling; no HPO or feature selection",
            "blocked_oof_requirement": "all base/residual scores and context inputs originate from their candidate's held blocked OOF fold; labels resolve before freeze",
            "regime_inputs": list(REGIME_FEATURES), "transition_inputs": list(TRANSITION_FEATURES),
            "excluded_regime_inputs": ["regime_state_id", "regime_state_p__*"],
            "reason_excluded": "independently fitted blocked geometries have no stable raw component identity; semantic summaries remain comparable",
            "coverage_limitations": [
                "No candidate-keyed blocked OOF regime/transition context before 2023-04-01.",
                "2025-01 is absent and 2025-02 is residual warm-up/non-OOF.",
                "2025-07 through 2025-12 lack a compatible frozen base/residual OOF candidate stream; they are not represented.",
            ],
        }
        _write_json(stage / "score_contract.json", contract)
        outputs = [panel_path, stage / "frozen_fit_diagnostics.csv", stage / "training_coverage_by_month.csv", stage / "score_contract.json", *model_paths]
        manifest = {
            "schema": SCHEMA, "status": "FROZEN_PRE_2026_CONTEXTUAL_SCORE_ARMS_READY",
            "scope": "Four pre-2026 frozen raw score arms; requires authoritative v2 2026 context to apply.",
            "frozen_contextual_coefficients": contract,
            "inputs": {str(path): _sha(path) for path in (historical_scores, historical_context,
                Path(gap_dir) / "candidate_scores_with_exact_labels.parquet",
                Path(gap_dir) / "candidate_oof_regime_transition/candidate_oof_regime_transition.parquet")},
            "outputs": {path.name: _sha(path) for path in outputs},
            "training_rows": int(len(panel),), "training_coverage": by_month.to_dict("records"),
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, destination)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--historical-scores", type=Path, default=HIST_SCORES)
    parser.add_argument("--historical-context", type=Path, default=HIST_CONTEXT)
    parser.add_argument("--gap-dir", type=Path, default=GAP)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir, historical_scores=args.historical_scores,
                         historical_context=args.historical_context, gap_dir=args.gap_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
