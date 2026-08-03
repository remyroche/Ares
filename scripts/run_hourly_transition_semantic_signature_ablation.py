#!/usr/bin/env python3
"""Outcome-free hourly transition-signature support/transfer diagnostic.

This is deliberately an alternative to morphology cluster IDs.  It expands
event support to labelled *hourly onset horizons*, but keeps the underlying
event ID on every positive row and never treats overlapping hours as
independent transition events.  Semantic groups are fixed causal feature
families; no held-2026 data is used to fit, name, rotate or select them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
CATALOGUE = ROOT / "data_perp/artifacts/transition_pattern_catalogue_20260730_v6"
OUT = ROOT / "data_perp/artifacts/hourly_transition_semantic_signature_ablation_20260730_v1"
SCHEMA = "hourly_transition_semantic_signature_ablation_v1"
HORIZON_HOURS = 3

# These are causal 3h deltas on an hourly row.  The groups are fixed by their
# observable mechanism names, not selected against 2026 labels or outcomes.
GROUPS = {
    "breadth_dislocation": (
        "transition_new__breadth_dispersion__delta_3h",
        "transition_new__downside_breadth_intensity__delta_3h",
        "transition_new__btc_resilience_alt_weakness__delta_3h",
    ),
    "washout_reversal": (
        "transition_new__broad_washout_recovery__delta_3h",
        "transition_new__deleveraged_range_climax_reversal__delta_3h",
        "transition_new__deleveraging_without_followthrough__delta_3h",
        "transition_new__short_breakout_exhaustion__delta_3h",
    ),
    "funding_positioning": (
        "transition_new__funding_deleveraging_divergence__delta_3h",
        "transition_new__funding_confirmed_long_flush__delta_3h",
        "transition_new__funding_confirmed_short_covering__delta_3h",
    ),
}
ARMS = {"breadth_dislocation": GROUPS["breadth_dislocation"], "washout_reversal": GROUPS["washout_reversal"], "funding_positioning": GROUPS["funding_positioning"], "combined_fixed_semantics": tuple(field for group in GROUPS.values() for field in group)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temp.write_text(json.dumps(value, default=str, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _hourly(frame: pd.DataFrame, timestamp: str, name: str) -> pd.DataFrame:
    out = frame.copy()
    out[timestamp] = pd.to_datetime(out[timestamp], utc=True, errors="raise")
    if out[timestamp].duplicated().any() or (out[timestamp].astype("int64") % pd.Timedelta(hours=1).value != 0).any():
        raise ValueError(f"{name} must be unique 1h rows")
    return out


def labelled_hourly(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    panel = _hourly(panel, "source_utc", "hourly panel")
    events = events.copy()
    events["anchor_source_utc"] = pd.to_datetime(events["anchor_source_utc"], utc=True, errors="raise")
    if events.event_id.duplicated().any() or (events.anchor_source_utc.astype("int64") % pd.Timedelta(hours=1).value != 0).any():
        raise ValueError("event identity/cadence invalid")
    # A label at hour t asks whether one known event begins in the *next* three
    # full decision hours.  It is not a phase label and features at t cannot
    # see the future event.  Any overlapping positive windows fail closed.
    anchor_to_id = dict(zip(events.anchor_source_utc, events.event_id))
    labels: list[int] = []
    ids: list[str | None] = []
    for time in panel.source_utc:
        matches = [anchor_to_id.get(time + pd.Timedelta(hours=offset)) for offset in range(1, HORIZON_HOURS + 1)]
        active = [value for value in matches if value is not None]
        if len(active) > 1:
            raise ValueError("overlapping event onset horizons would break event identity")
        labels.append(int(bool(active))); ids.append(active[0] if active else None)
    panel["target_onset_next_3h"] = labels
    panel["next_event_id"] = ids
    panel["label_available_utc"] = panel.source_utc + pd.Timedelta(hours=HORIZON_HOURS)
    panel["era"] = panel.source_utc.dt.year.astype(int)
    return panel.loc[panel.label_available_utc <= panel.source_utc.max()].copy()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.10, class_weight="balanced", max_iter=3000, random_state=719)),
    ])
    x = train.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    z = test.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    y = train.target_onset_next_3h.to_numpy(int)
    model.fit(x, y)
    return model.predict_proba(z)[:, 1], model.named_steps["logit"].coef_.ravel()


def metrics(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    y, p = frame.target_onset_next_3h.to_numpy(int), frame[score].to_numpy(float)
    return {"rows": int(len(frame)), "positive_hour_rows": int(y.sum()), "positive_unique_events": int(frame.loc[frame.target_onset_next_3h.eq(1), "next_event_id"].nunique()), "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan, "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan, "brier": float(brier_score_loss(y, p)), "mean_probability": float(p.mean())}


def weekly_block_bootstrap(frame: pd.DataFrame, score: str, draws: int = 500) -> dict[str, float | int]:
    """Uncertainty is resampled by UTC week, not by overlapping positive hours."""
    work = frame.copy(); work["week"] = work.source_utc.dt.strftime("%G-W%V")
    blocks = [group for _, group in work.groupby("week", sort=True)]
    generator = np.random.default_rng(711)
    aucs=[]; aps=[]
    for _ in range(draws):
        sample = pd.concat([blocks[index] for index in generator.integers(0, len(blocks), len(blocks))], ignore_index=True)
        y, p = sample.target_onset_next_3h.to_numpy(int), sample[score].to_numpy(float)
        if np.unique(y).size == 2:
            aucs.append(roc_auc_score(y,p)); aps.append(average_precision_score(y,p))
    return {"bootstrap_unit": "UTC_week", "draws": draws, "valid_draws": len(aucs), "roc_auc_ci_low": float(np.quantile(aucs,.025)) if aucs else np.nan, "roc_auc_ci_high": float(np.quantile(aucs,.975)) if aucs else np.nan, "average_precision_ci_low": float(np.quantile(aps,.025)) if aps else np.nan, "average_precision_ci_high": float(np.quantile(aps,.975)) if aps else np.nan}


def coefficient_stability(train: pd.DataFrame, fields: Sequence[str]) -> dict[str, Any]:
    vectors=[]; eras=[]
    for held in sorted(train.era.unique()):
        fit = train.loc[train.era.ne(held)]
        _, coeff = fit_predict(fit, train.loc[train.era.eq(held)], fields)
        vectors.append(coeff); eras.append(int(held))
    corr=[]
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            corr.append(float(np.corrcoef(vectors[left], vectors[right])[0,1]))
    return {"coefficient_leave_era_out_folds": eras, "coefficient_pairwise_correlation_mean": float(np.mean(corr)), "coefficient_pairwise_correlation_min": float(np.min(corr)), "coefficient_stable": bool(corr and min(corr) >= .70)}


def run(*, ledger: Path = LEDGER, catalogue: Path = CATALOGUE, output: Path = OUT) -> Path:
    ledger, catalogue, output = map(Path, (ledger, catalogue, output))
    if output.exists():
        raise FileExistsError(output)
    panel_path, events_path = ledger / "hourly_state_calendar.parquet", catalogue / "event_preonset_sequences.parquet"
    panel, events = pd.read_parquet(panel_path), pd.read_parquet(events_path, columns=["event_id", "anchor_source_utc"])
    missing = [field for fields in ARMS.values() for field in fields if field not in panel]
    if missing:
        raise KeyError(f"fixed semantic features missing: {sorted(set(missing))}")
    data = labelled_hourly(panel[["source_utc", *ARMS["combined_fixed_semantics"]]], events)
    # The three-hour label must also resolve inside its partition.  This avoids
    # even a boundary label crossing into 2026 during the 2022--2025 fit.
    split = pd.Timestamp("2026-01-01", tz="UTC")
    train = data.loc[data.source_utc.lt(split) & data.label_available_utc.lt(split)].copy()
    assess = data.loc[data.source_utc.ge(split)].copy()
    if train.label_available_utc.max() >= pd.Timestamp("2026-01-01", tz="UTC") or assess.empty:
        raise ValueError("2022-2025 train / 2026 assessment split invalid")
    output_rows=[]; coefficient_rows=[]; scored=[]
    for arm, fields in ARMS.items():
        probability, coeff = fit_predict(train, assess, fields)
        score = f"score__{arm}"; assess[score] = probability
        output_rows.append({"arm": arm, "partition": "train_2022_2025", **metrics(train.assign(**{score: fit_predict(train, train, fields)[0]}), score)})
        output_rows.append({"arm": arm, "partition": "assessment_2026", **metrics(assess, score), **weekly_block_bootstrap(assess, score)})
        stability = coefficient_stability(train, fields)
        coefficient_rows.extend({"arm":arm,"feature":field,"full_train_standardized_coefficient":float(value),**stability} for field,value in zip(fields, coeff))
        scored.append(assess.loc[:, ["source_utc","label_available_utc","target_onset_next_3h","next_event_id",score]])
    support = pd.concat([train.assign(partition="train_2022_2025"), assess.assign(partition="assessment_2026")]).groupby(["partition","era"],as_index=False).agg(hourly_rows=("source_utc","size"),positive_hour_rows=("target_onset_next_3h","sum"),unique_transition_events=("next_event_id","nunique"))
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.DataFrame(output_rows).to_csv(stage / "hourly_transfer_metrics.csv", index=False)
        pd.DataFrame(coefficient_rows).to_csv(stage / "semantic_group_coefficient_stability.csv", index=False)
        support.to_csv(stage / "event_identity_support.csv", index=False)
        merged = scored[0]
        for frame in scored[1:]: merged = merged.merge(frame, on=["source_utc","label_available_utc","target_onset_next_3h","next_event_id"], how="inner", validate="one_to_one")
        merged.to_parquet(stage / "assessment_2026_hourly_scored_labels.parquet", index=False)
        cadence = pd.DataFrame([{"table":"hourly_panel_and_assessment","rows":len(data),"non_hourly_rows":0,"cadence":"1h"},{"table":"event_anchors","rows":len(events),"non_hourly_rows":0,"cadence":"1h"}]); cadence.to_csv(stage / "cadence_audit.csv", index=False)
        contract = {"cadence":"all panel, train and assessment observations are 1h", "label":"onset in next 1-3 hours; label is available after the 3h horizon", "support":"hourly positive windows retain next_event_id; 3 hours for one event are not counted as 3 independent events", "split":"fit/semantic naming only 2022-2025; final assessment is 2026", "outcomes":"no execution/trading outcome, policy score or held-era outcome enters", "semantics":"fixed named causal feature groups; no GMM/HDBSCAN/component ID, forced cluster alignment or hidden state", "uncertainty":"UTC-week block bootstrap on 2026, not row bootstrap"}; write_json(stage / "contract.json", contract)
        files=[path for path in stage.iterdir() if path.is_file()]
        manifest={"schema":SCHEMA,"status":"SEALED_HOURLY_SEMANTIC_SIGNATURE_DIAGNOSTIC_NON_PROMOTION","promotion_eligible":False,"inputs_sha256":{str(panel_path.resolve()):sha(panel_path),str(events_path.resolve()):sha(events_path)},"contract":contract,"outputs_sha256":{path.name:sha(path) for path in files}}; write_json(stage / "manifest.json",manifest); (stage / "manifest.sha256").write_text(f"{sha(stage/'manifest.json')}  manifest.json\n"); os.replace(stage,output); return output
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--ledger",type=Path,default=LEDGER);parser.add_argument("--catalogue",type=Path,default=CATALOGUE);parser.add_argument("--output",type=Path,default=OUT);return parser.parse_args(argv)
if __name__ == "__main__": print(run(**vars(parse_args())))
