#!/usr/bin/env python3
"""Build the fail-closed causal hourly regime and transition sidecars.

Only two sources are allowed: the frozen strict short-horizon LGBM challenger
and the sealed online-BOCPD challenger.  Diagonal/sticky/DAE state identities
and morphology IDs are deliberately not read, even when they are present in
the artifact store.  Historical rows are blocked-OOF; 2026 rows are one
untouched forward pass.  This is a context artifact, never a trading gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "data_perp/artifacts"
LGBM_ROOT = ART / "strict_forward_transition_challenger_20260730_v2"
BOCPD_ROOT = ART / "strict_bocpd_regime_transition_challenger_20260730_v2"
BOCPD_CHECKPOINTS = ART / "strict_bocpd_regime_transition_challenger_20260730_v2_checkpoints"
OUTPUT = ART / "authoritative_soft_regime_transition_sidecars_20260730_v1"
SCHEMA = "authoritative_soft_regime_transition_sidecars_v1"
CADENCE = "1h"
TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
LGBM_SCHEMA = "strict_forward_transition_challenger_v2"
BOCPD_SCHEMA = "strict_bocpd_regime_transition_challenger_v2"
BOCPD_STATUS = "SEALED_STRICT_RESUMABLE_BOCPD"
EXCLUDED = (
    "diagonal_gmm_state_identity",
    "sticky_gmm_state_identity",
    "dae_gmm_state_identity",
    "morphology_component_id",
    "morphology_archetype_id",
)
REGIME_CONTEXT = (
    "bocpd__change_probability_mean",
    "bocpd__change_probability_max",
    "bocpd__run_length_mean",
    "bocpd__run_length_q05",
    "bocpd__run_length_entropy",
    "bocpd__signal_count",
    "bocpd__state_age_hours",
    "bocpd__is_persistent_24h",
    "bocpd__is_persistent_72h",
)


class SidecarError(RuntimeError):
    """A required causal/provenance contract is absent or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path | pd.Timestamp):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sealed_manifest(root: Path, *, schema: str, status: str | None = None) -> dict[str, Any]:
    manifest_path, sidecar = root / "manifest.json", root / "manifest.sha256"
    if not root.is_dir() or not manifest_path.is_file() or not sidecar.is_file():
        raise SidecarError(f"required sealed artifact is absent: {root}")
    if sidecar.read_text(encoding="utf-8").split()[0:1] != [sha256(manifest_path)]:
        raise SidecarError(f"manifest checksum does not verify: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise SidecarError(f"unexpected artifact schema: {root}")
    if status is not None and manifest.get("status") != status:
        raise SidecarError(f"artifact is not sealed in the required status: {root}")
    for name, digest in manifest.get("outputs_sha256", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise SidecarError(f"artifact output checksum does not verify: {path}")
    return manifest


def _hourly(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    result = frame.copy()
    result["source_utc"] = pd.to_datetime(result["source_utc"], utc=True, errors="raise")
    if result["source_utc"].isna().any() or result.duplicated("source_utc").any():
        raise SidecarError(f"{name} requires one non-null row per hourly timestamp")
    nanos = result["source_utc"].astype("int64").to_numpy()
    if (nanos % pd.Timedelta(hours=1).value).any():
        raise SidecarError(f"{name} contains non-hourly timestamps")
    return result.sort_values("source_utc", kind="stable").reset_index(drop=True)


def _entropy_margin(probability: pd.Series) -> tuple[pd.Series, pd.Series]:
    value = pd.to_numeric(probability, errors="coerce").clip(0.0, 1.0)
    entropy = -(
        value.clip(1e-12, 1.0) * np.log(value.clip(1e-12, 1.0))
        + (1.0 - value).clip(1e-12, 1.0) * np.log((1.0 - value).clip(1e-12, 1.0))
    ) / math.log(2.0)
    return entropy.where(value.notna()), (2.0 * (value - 0.5).abs()).where(value.notna())


def _validate_no_excluded_columns(columns: Iterable[str]) -> None:
    observed = [column for column in columns if any(token in column.lower() for token in ("gmm", "morphology"))]
    if observed:
        raise SidecarError("rejected state/morphology output entered sidecar: " + ", ".join(sorted(observed)))


def _lgbm_historical_oof(catalogue: Path, winner: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the frozen winner's blocked predictions without refitting HPO."""
    from scripts.run_strict_forward_transition_challenger_v2 import FOLDS, family_features, model, platt
    from scripts.run_strict_forward_transition_evaluation import label_available

    frame = pd.read_parquet(catalogue).copy()
    frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    resolved = label_available(frame)
    train = frame.loc[
        frame["source_utc"].lt(TRAIN_END)
        & resolved.lt(TRAIN_END)
        & frame["target__transition_active"].notna()
    ].copy()
    train["label_resolution_utc"] = resolved.loc[train.index].to_numpy()
    fold_raw: list[pd.DataFrame] = []
    for number, start in enumerate(FOLDS):
        stop = start + pd.DateOffset(months=6)
        fit = train.loc[
            train["source_utc"].lt(start)
            & train["label_resolution_utc"].lt(start)
        ].copy()
        score = train.loc[train["source_utc"].ge(start) & train["source_utc"].lt(stop)].copy()
        if score.empty or fit.empty or fit["target__transition_active"].nunique() != 2:
            continue
        features = family_features(train, fit, str(winner["family"]))
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        x, z = imputer.fit_transform(fit[features]), imputer.transform(score[features])
        y = fit["target__transition_active"].astype(int).to_numpy()
        weight = float(winner["positive_weight"])
        weights = np.where(y == 1, weight, 1.0)
        fitted = model(str(winner["model"]), multiclass=False, seed=20260730 + number).fit(x, y, sample_weight=weights)
        raw = fitted.predict_proba(z)[:, list(fitted.classes_).index(1)]
        fold_raw.append(pd.DataFrame({
            "source_utc": score["source_utc"].to_numpy(), "fold": number,
            "raw": raw, "y": score["target__transition_active"].astype(int).to_numpy(),
            "label_resolution_utc": score["label_resolution_utc"].to_numpy(),
            "train_end_exclusive_utc": start,
            "fit_label_resolution_max_utc": fit["label_resolution_utc"].max(),
        }))
    if not fold_raw:
        raise SidecarError("frozen LGBM winner produced no blocked-OOF rows")
    raw = pd.concat(fold_raw, ignore_index=True)
    calibrated: list[pd.DataFrame] = []
    for fold, local in raw.groupby("fold", sort=True):
        result = local.copy()
        # Preserve the source runner's exact calibration contract: each fold
        # sees only earlier blocked-fold raw predictions and labels.
        prior = raw.loc[raw["fold"].lt(fold)].copy()
        if len(prior) >= 20:
            if prior["y"].nunique() == 2:
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression(C=1.0, max_iter=200, random_state=20260730).fit(prior[["raw"]], prior["y"].astype(int))
                result["probability"] = calibrator.predict_proba(result[["raw"]])[:, 1]
            else:
                result["probability"] = result["raw"]
        else:
            result["probability"] = result["raw"]
        calibrated.append(result)
    return pd.concat(calibrated, ignore_index=True), train.loc[:, ["source_utc", "label_resolution_utc"]]


def _lgbm_rows(*, catalogue: Path, root: Path, manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical, audit = _lgbm_historical_oof(catalogue, manifest["winner"]["active"])
    forward = _hourly(pd.read_parquet(root / "v2_forward_predictions.parquet"), name="LGBM forward")
    if "transition_probability" not in forward:
        raise SidecarError("strict LGBM forward artifact lacks transition probability")
    historical = _hourly(historical, name="LGBM blocked OOF")
    historical["provenance_partition"] = "blocked_oof_2022_2025"
    # Preserve every 2022--2025 hourly row.  A block without sufficient
    # preceding resolved labels is unavailable warm-up, not silently omitted.
    historical = audit.loc[:, ["source_utc"]].merge(
        historical.loc[:, ["source_utc", "probability", "train_end_exclusive_utc", "fit_label_resolution_max_utc", "label_resolution_utc", "provenance_partition"]],
        on="source_utc", how="left", validate="one_to_one",
    )
    historical["provenance_partition"] = historical["provenance_partition"].fillna("blocked_oof_warmup_unavailable")
    forward = forward.loc[:, ["source_utc", "transition_probability"]].copy()
    forward["train_end_exclusive_utc"] = TRAIN_END
    forward["fit_label_resolution_max_utc"] = pd.NaT
    forward["label_resolution_utc"] = pd.NaT
    forward["provenance_partition"] = "untouched_2026_forward"
    output = pd.concat([
        historical.loc[:, ["source_utc", "probability", "train_end_exclusive_utc", "fit_label_resolution_max_utc", "label_resolution_utc", "provenance_partition"]].rename(columns={"probability": "lgbm_transition_probability"}),
        forward.rename(columns={"transition_probability": "lgbm_transition_probability"}),
    ], ignore_index=True)
    output = _hourly(output, name="combined LGBM")
    output["lgbm_transition_available"] = output["lgbm_transition_probability"].notna()
    output["lgbm_entropy"], output["lgbm_margin"] = _entropy_margin(output["lgbm_transition_probability"])
    output["lgbm_ood_available"] = False
    output["lgbm_ood_score"] = np.nan
    return output, audit


def _bocpd_historical_oof(*, catalogue: Path, current: Path, checkpoints: Path, winners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.run_strict_bocpd_regime_transition_challenger import (
        FOLDS, HEADS, _combined_checkpoint, _features, _fit, _head_context, _load,
    )
    _, train, _ = _load(catalogue, current)
    targets = dict(HEADS)
    audit = train.loc[:, ["source_utc"]].copy()
    from scripts.run_strict_bocpd_regime_transition_challenger import _label_available
    audit["label_resolution_utc"] = _label_available(train).to_numpy()
    pieces: list[pd.DataFrame] = []
    cache_root = checkpoints / "authoritative_sidecar_oof_v1"
    for head, target in HEADS:
        winner = winners.loc[winners["head"].eq(head)]
        if len(winner) != 1:
            raise SidecarError(f"sealed BOCPD winner is ambiguous for {head}")
        row = winner.iloc[0]
        head_pieces: list[pd.DataFrame] = []
        for fold, start in enumerate(FOLDS):
            cache_dir = cache_root / head
            cache_path, cache_manifest = cache_dir / f"fold_{fold:02d}.parquet", cache_dir / f"fold_{fold:02d}.json"
            expected = {
                "schema": "authoritative_bocpd_oof_fold_cache_v1",
                "head": head,
                "fold": fold,
                "train_end_exclusive_utc": str(start),
                "expected_run_hours": int(row["expected_run_hours"]),
                "logistic_c": float(row["logistic_c"]),
                "catalogue_sha256": sha256(catalogue),
                "current_sha256": sha256(current),
                "fit_label_contract": "strictly before fold train end",
            }
            if cache_path.is_file() and cache_manifest.is_file() and json.loads(cache_manifest.read_text(encoding="utf-8")) == safe(expected):
                cached = pd.read_parquet(cache_path)
                # Cache values are immutable model outputs; recompute this
                # audit summary from the authoritative availability vector so
                # an old cache can never claim an unresolved fit label.
                cached["fit_label_resolution_max_utc"] = audit.loc[
                    audit["source_utc"].lt(start) & audit["label_resolution_utc"].lt(start),
                    "label_resolution_utc",
                ].max()
                head_pieces.append(cached)
                continue
            context, _ = _features(
                _combined_checkpoint(checkpoints, horizon=int(row["expected_run_hours"]), split=f"fold_{fold:02d}"),
                len(train.loc[train["source_utc"].lt(start)]),
            )
            score_scope = train.loc[train["source_utc"].lt(start + pd.DateOffset(months=6))]
            fit, score = _head_context(score_scope, context, start=start)
            # The score block must remain intact.  Only the fitting history is
            # restricted to labels resolved before the fold boundary.
            fit = fit.merge(audit, on="source_utc", how="left", validate="one_to_one")
            fit = fit.loc[fit["label_resolution_utc"].lt(start)].drop(columns="label_resolution_utc")
            if score.empty or fit[target].nunique() != 2:
                continue
            probability = _fit(fit, score, target=target, c=float(row["logistic_c"]))
            output = score.loc[:, ["source_utc", *[field for field in REGIME_CONTEXT if field in score]]].copy()
            output["head"] = head
            output["probability"] = probability
            output["train_end_exclusive_utc"] = start
            output["fit_label_resolution_max_utc"] = audit.loc[
                audit["source_utc"].lt(start) & audit["label_resolution_utc"].lt(start),
                "label_resolution_utc",
            ].max()
            cache_dir.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(dir=cache_dir, prefix=f".fold_{fold:02d}."))
            try:
                output.to_parquet(stage / cache_path.name, index=False, compression="zstd")
                (stage / cache_manifest.name).write_text(json.dumps(safe(expected), indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(stage / cache_path.name, cache_path)
                os.replace(stage / cache_manifest.name, cache_manifest)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            head_pieces.append(output)
        if not head_pieces:
            raise SidecarError(f"sealed BOCPD head has no blocked-OOF rows: {head}")
        pieces.append(pd.concat(head_pieces, ignore_index=True))
    return pd.concat(pieces, ignore_index=True), audit


def _bocpd_rows(*, catalogue: Path, current: Path, root: Path, checkpoints: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    winners = pd.read_csv(root / "frozen_bocpd_winners.csv")
    historical, audit = _bocpd_historical_oof(catalogue=catalogue, current=current, checkpoints=checkpoints, winners=winners)
    forward = pd.read_parquet(root / "forward_bocpd_regime_transition.parquet")
    forward["source_utc"] = pd.to_datetime(forward["source_utc"], utc=True, errors="raise")
    heads = sorted(set(winners["head"]))
    def pivot(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
        probs = frame.pivot(index="source_utc", columns="head", values="probability").rename(columns=lambda x: f"bocpd_{x}_probability")
        result = probs.reset_index()
        stable = frame.loc[frame["head"].eq("stable_vs_transition")].copy()
        for field in REGIME_CONTEXT:
            if field in stable:
                result = result.merge(stable.loc[:, ["source_utc", field]], on="source_utc", how="left", validate="one_to_one")
        provenance = frame.groupby("source_utc", as_index=False).agg(train_end_exclusive_utc=("train_end_exclusive_utc", "min"), fit_label_resolution_max_utc=("fit_label_resolution_max_utc", "max")) if "train_end_exclusive_utc" in frame else pd.DataFrame({"source_utc": result["source_utc"], "train_end_exclusive_utc": TRAIN_END, "fit_label_resolution_max_utc": pd.NaT})
        result = result.merge(provenance, on="source_utc", how="left", validate="one_to_one")
        result["provenance_partition"] = partition
        return result
    historical = pivot(historical, "blocked_oof_2022_2025")
    historical = audit.loc[:, ["source_utc"]].merge(
        historical, on="source_utc", how="left", validate="one_to_one",
    )
    historical["provenance_partition"] = historical["provenance_partition"].fillna("blocked_oof_warmup_unavailable")
    forward = pivot(forward, "untouched_2026_forward")
    output = _hourly(pd.concat([historical, forward], ignore_index=True), name="combined BOCPD")
    for head in heads:
        field = f"bocpd_{head}_probability"
        output[f"bocpd_{head}_available"] = output.get(field, pd.Series(index=output.index, dtype=float)).notna()
        output[f"bocpd_{head}_entropy"], output[f"bocpd_{head}_margin"] = _entropy_margin(output.get(field, pd.Series(index=output.index, dtype=float)))
    output["bocpd_regime_available"] = output[[field for field in REGIME_CONTEXT if field in output]].notna().all(axis=1)
    output["bocpd_ood_available"] = False
    output["bocpd_ood_score"] = np.nan
    return output, audit


def assemble_sidecars(lgbm: pd.DataFrame, bocpd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make separate causal regime and transition tables on the hourly union."""
    lgbm, bocpd = _hourly(lgbm, name="LGBM input"), _hourly(bocpd, name="BOCPD input")
    combined = lgbm.merge(bocpd, on="source_utc", how="outer", suffixes=("_lgbm", "_bocpd"), validate="one_to_one")
    combined = _hourly(combined, name="authoritative hourly union")
    _validate_no_excluded_columns(combined.columns)
    validate_label_resolution_audit(combined)
    regime_fields = ["source_utc", *[field for field in REGIME_CONTEXT if field in combined], "bocpd_regime_available", "bocpd_ood_available", "bocpd_ood_score", "provenance_partition_bocpd", "train_end_exclusive_utc_bocpd", "fit_label_resolution_max_utc_bocpd"]
    transition_fields = ["source_utc", "lgbm_transition_probability", "lgbm_entropy", "lgbm_margin", "lgbm_transition_available", "lgbm_ood_available", "lgbm_ood_score", "provenance_partition_lgbm", "train_end_exclusive_utc_lgbm", "fit_label_resolution_max_utc_lgbm", *[field for field in combined if field.startswith("bocpd_") and (field.endswith("_probability") or field.endswith("_entropy") or field.endswith("_margin") or field.endswith("_available"))], "provenance_partition_bocpd", "train_end_exclusive_utc_bocpd", "fit_label_resolution_max_utc_bocpd"]
    regime = combined.loc[:, [field for field in regime_fields if field in combined]].copy()
    transition = combined.loc[:, [field for field in transition_fields if field in combined]].copy()
    return _hourly(regime, name="regime sidecar"), _hourly(transition, name="transition sidecar")


def _coverage(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    result = frame.assign(month=frame["source_utc"].dt.strftime("%Y-%m"))
    availability = [field for field in result if field.endswith("_available")]
    rows = []
    for month, local in result.groupby("month", sort=True):
        for field in availability:
            rows.append({"sidecar": kind, "month": month, "field": field, "hourly_rows": len(local), "available_rows": int(local[field].fillna(False).sum())})
    return pd.DataFrame(rows)


def cadence_audit(*, regime: pd.DataFrame, transition: pd.DataFrame) -> pd.DataFrame:
    """Evidence that feature lookbacks do not turn into sub-hourly model rows."""
    rows = []
    for name, frame in (("regime", regime), ("transition", transition)):
        times = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
        rows.append(
            {
                "sidecar": name,
                "model_sample_cadence": "1h",
                "assessment_sample_cadence": "1h",
                "exact_replay_bar_cadence": "1m_labels_only",
                "native_15m_contract": "causal lookback sampled onto the 1h decision row; never a 15m model example",
                "rows": len(frame),
                "non_hourly_timestamp_rows": int((times.astype("int64") % pd.Timedelta(hours=1).value != 0).sum()),
                "duplicate_timestamp_rows": int(times.duplicated().sum()),
            }
        )
    audit = pd.DataFrame(rows)
    if audit[["non_hourly_timestamp_rows", "duplicate_timestamp_rows"]].to_numpy().any():
        raise SidecarError("cadence audit found non-hourly or duplicate model rows")
    return audit


def validate_label_resolution_audit(frame: pd.DataFrame) -> None:
    """OOF fit labels must resolve before their own fold train end."""
    for suffix in ("lgbm", "bocpd"):
        partition = f"provenance_partition_{suffix}"
        train_end = f"train_end_exclusive_utc_{suffix}"
        resolved = f"fit_label_resolution_max_utc_{suffix}"
        if not {partition, train_end, resolved}.issubset(frame.columns):
            continue
        historical = frame[partition].eq("blocked_oof_2022_2025")
        if not historical.any():
            continue
        end = pd.to_datetime(frame.loc[historical, train_end], utc=True, errors="raise")
        maximum = pd.to_datetime(frame.loc[historical, resolved], utc=True, errors="raise")
        if maximum.isna().any() or end.isna().any() or not maximum.lt(end).all():
            raise SidecarError(f"{suffix} OOF fit-label resolution is not strictly before train end")


def run(*, catalogue: Path, current: Path, lgbm_root: Path, bocpd_root: Path, bocpd_checkpoints: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    lgbm_manifest = _sealed_manifest(lgbm_root, schema=LGBM_SCHEMA)
    bocpd_manifest = _sealed_manifest(bocpd_root, schema=BOCPD_SCHEMA, status=BOCPD_STATUS)
    lgbm, lgbm_audit = _lgbm_rows(catalogue=catalogue, root=lgbm_root, manifest=lgbm_manifest)
    bocpd, bocpd_audit = _bocpd_rows(catalogue=catalogue, current=current, root=bocpd_root, checkpoints=bocpd_checkpoints)
    regime, transition = assemble_sidecars(lgbm, bocpd)
    audit = pd.concat([lgbm_audit.assign(source="lgbm"), bocpd_audit.assign(source="bocpd")], ignore_index=True)
    audit["label_resolution_utc"] = pd.to_datetime(audit["label_resolution_utc"], utc=True, errors="raise")
    coverage = pd.concat([_coverage(regime, kind="regime"), _coverage(transition, kind="transition")], ignore_index=True)
    cadence = cadence_audit(regime=regime, transition=transition)
    reliability = pd.read_csv(bocpd_root / "untouched_2026_discrimination_calibration.csv")
    reliability["source"] = "sealed_strict_bocpd_v2"
    reliability["reliability_status"] = "CONVERGENCE_AND_CALIBRATION_LIMITED_DIAGNOSTIC_ONLY"
    reliability["promotion_eligible"] = False
    reliability["use_constraint"] = "provenance/context only; never a gate, quota, or standalone trading score"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        regime.to_parquet(stage / "soft_regime_hourly.parquet", index=False, compression="zstd")
        transition.to_parquet(stage / "soft_transition_hourly.parquet", index=False, compression="zstd")
        audit.to_csv(stage / "label_resolution_audit.csv", index=False)
        coverage.to_csv(stage / "coverage_by_month.csv", index=False)
        cadence.to_csv(stage / "cadence_audit.csv", index=False)
        reliability.to_csv(stage / "bocpd_reliability.csv", index=False)
        files = {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_CAUSAL_SOFT_REGIME_TRANSITION_SIDECARS",
            "research_only": True,
            "promotion_eligible": False,
            "model_sample_cadence": CADENCE,
            "assessment_sample_cadence": CADENCE,
            "exact_replay_bar_cadence": "1m_labels_only",
            "cadence_contract": "all model and assessment examples are 1h decision rows; multi-timeframe (including native 15m) values are causal lookbacks sampled onto those rows; 1m exists only within nested replay/label paths",
            "historical_contract": "blocked-OOF hourly predictions using only fit labels resolved before each fold train end; unavailable warm-up rows remain unavailable",
            "forward_contract": "untouched 2026 hourly forward predictions from frozen 2022-2025 selections",
            "separation_contract": "regime is BOCPD causal change/run-length context; transition is LGBM and BOCPD short-horizon probabilities; neither is a trading gate",
            "reliability_contract": "BOCPD logistic heads carry the sealed source's convergence warnings and poor calibration into bocpd_reliability.csv; they are diagnostic-only provenance/context and cannot be promoted, gated, quotaed, or used as a standalone trading score",
            "excluded_outputs": list(EXCLUDED),
            "inputs_sha256": {"catalogue": sha256(catalogue), "current": sha256(current), "lgbm_manifest": sha256(lgbm_root / "manifest.json"), "bocpd_manifest": sha256(bocpd_root / "manifest.json")},
            "outputs_sha256": files,
            "counts": {"regime_hourly_rows": len(regime), "transition_hourly_rows": len(transition), "label_audit_rows": len(audit)},
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=ART / "transition_pattern_catalogue_20260730_v6/adaptive_phase_labels.parquet")
    parser.add_argument("--current", type=Path, default=ART / "current_exact_policy_global_book_mapping_source_20260730_v3/causal_mapped_candidates.parquet")
    parser.add_argument("--lgbm-root", type=Path, default=LGBM_ROOT)
    parser.add_argument("--bocpd-root", type=Path, default=BOCPD_ROOT)
    parser.add_argument("--bocpd-checkpoints", type=Path, default=BOCPD_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(safe(run(**vars(args))), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
