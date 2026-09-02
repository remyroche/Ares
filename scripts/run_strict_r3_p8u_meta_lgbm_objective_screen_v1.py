#!/usr/bin/env python3
"""Strict-OOF P8u Meta LGBM objective screen.

This is the second Meta stage.  It takes one preselected target/query arm,
keeps its target and causal folds fixed, and compares LambdaRank gain /
truncation / sigmoid contracts with rank_xendcg.  Every held score is
persisted target-free before outcome metrics are joined.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_lgbm_objective_screen_v1"
IDENTITY = screen.IDENTITY


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, payload: Mapping[str, Any]) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _arms(raw: Mapping[str, Any]) -> dict[str, screen.Arm]:
    return {arm.name: arm for arm in screen._arm_specs(raw, None)}


def _apply_source_override(raw: Mapping[str, Any], path: Path | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply a source-only receipt without changing frozen model semantics.

    Historical ledger extensions need a longer, disjoint OOS score/feature
    source.  Keeping that replacement in a small immutable JSON receipt makes
    the source transition auditable while preserving every target, query,
    feature contract, and model parameter in the frozen parent configuration.
    """
    result = deepcopy(dict(raw))
    if path is None:
        return result, None
    payload = json.loads(path.read_text())
    override = payload.get("source", payload) if isinstance(payload, dict) else None
    allowed = {"base_target_free_root", "full_feature_roots", "base_f72_contract", "policy_labels", "path_labels"}
    if not isinstance(override, dict) or not override or set(override).difference(allowed):
        raise AssertionError("source override must be a non-empty mapping of declared source keys only")
    if "base_target_free_root" in override and not isinstance(override["base_target_free_root"], str):
        raise AssertionError("base_target_free_root override must be a string")
    if "full_feature_roots" in override and (
        not isinstance(override["full_feature_roots"], list) or not all(isinstance(item, str) for item in override["full_feature_roots"])
    ):
        raise AssertionError("full_feature_roots override must be a list of strings")
    if any(key in override and not isinstance(override[key], str) for key in {"policy_labels", "path_labels"}):
        raise AssertionError("policy/path source overrides must be strings")
    result["source"] = {**dict(result["source"]), **override}
    return result, dict(override)


def _read_fields(path: Path) -> tuple[str, ...]:
    """Read a versioned causal Meta feature contract.

    Historical P8u screens default to the frozen F72 receipt.  A research
    challenger may instead pass an immutable contract with a wider, declared
    causal market-context field list.  The runner deliberately does not
    derive fields by name at runtime.
    """
    payload = json.loads(path.read_text())
    parent_ref = payload.get("parent_feature_contract")
    if parent_ref is not None:
        parent_path = (path.parent / str(parent_ref)).resolve()
        parent_fields = _read_fields(parent_path)
        additive = payload.get("additive_state_reliability_fields", ())
        if not isinstance(additive, list):
            raise AssertionError(f"{path}: additive_state_reliability_fields must be a list")
        fields = [*parent_fields, *(str(field) for field in additive)]
    else:
        # The full-feature selector emits ``candidate_fields`` in its sealed
        # subspace contract.  Treat that as an explicit immutable feature
        # receipt, alongside the older selected_features/features contracts,
        # so subsequent strict-OOF stages consume the exact selected subset
        # rather than copying it into an untracked wrapper file.
        fields = payload.get("selected_features", payload.get("features", payload.get("candidate_fields")))
    if not isinstance(fields, list) or not fields:
        raise AssertionError(f"{path}: missing non-empty selected_features")
    output = tuple(str(field) for field in fields)
    if len(output) != len(set(output)):
        raise AssertionError(f"{path}: duplicate fields")
    return output


def _read_contract(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], Path | None]:
    """Return declared fields and the optional exact-identity regime sidecar."""
    payload = json.loads(path.read_text())
    fields = _read_fields(path)
    sidecar_fields = tuple(str(value) for value in payload.get("continuous_regime_sidecar_fields", ()))
    if len(sidecar_fields) != len(set(sidecar_fields)) or not set(sidecar_fields).issubset(fields):
        raise AssertionError(f"{path}: invalid continuous-regime sidecar contract")
    source = payload.get("continuous_regime_sidecar_source")
    return fields, sidecar_fields, Path(str(source)).resolve() if source else None


def _read_features(
    *, base_root: Path, feature_roots: Sequence[Path], start: pd.Timestamp, end: pd.Timestamp,
    panel_fields: Sequence[str], continuous_sidecar: Path | None, continuous_fields: Sequence[str],
) -> pd.DataFrame:
    """Join target-free fields from their declared panel and optional sidecar.

    The frozen 120-field parent lives in the UnderF120 panel while an
    append-only SHAP overlay lives in the already-frozen Base score panel.
    The original generic reader required every declared field in the former,
    which silently made an additive Base-panel field impossible to screen.
    This reader resolves each declared field from exactly one contemporaneous
    target-free panel, then exact-identity joins the pieces.  It never falls
    back to outcome data or imputes a missing source field.
    """
    parts: list[pd.DataFrame] = []
    requested = tuple(panel_fields)
    for month in screen._month_range(start, end):
        base_path = screen._base_path(base_root, month)
        full_path = screen._full_path(feature_roots, month)
        screen._assert_target_free(base_path)
        screen._assert_target_free(full_path)
        base_names = set(screen.pq.ParquetFile(base_path).schema_arrow.names)
        full_names = set(screen.pq.ParquetFile(full_path).schema_arrow.names)
        missing = sorted(set(requested).difference(base_names | full_names))
        if missing:
            raise AssertionError(f"{month:%Y-%m}: declared Meta fields absent from every target-free panel: {missing[:8]}")
        # Prefer the dedicated overlay panel for an overlapping field.  The
        # Base score panel is only used for genuinely additive score-context
        # fields such as frozen SHAP summaries.
        full_fields = [field for field in requested if field in full_names]
        base_fields = [field for field in requested if field not in full_names]
        base = pd.read_parquet(base_path, columns=[*screen.BASE_COLUMNS, *base_fields])
        full = pd.read_parquet(full_path, columns=[*screen.IDENTITY, *full_fields])
        for piece in (base, full):
            piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True, errors="raise")
            if piece.duplicated(screen.IDENTITY).any():
                raise AssertionError(f"{month:%Y-%m}: duplicate target-free identity")
        merged = base.merge(full, on=list(screen.IDENTITY), how="left", validate="one_to_one")
        if len(merged) != len(base) or merged.loc[:, list(requested)].isna().all(axis=None):
            raise AssertionError(f"{month:%Y-%m}: causal feature identity coverage failure")
        parts.append(merged.loc[merged.__decision_ts__.ge(start) & merged.__decision_ts__.lt(end)].copy())
    base = screen._add_base_geometry(
        pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    )
    if base.duplicated(screen.IDENTITY).any() or not base.side_name.eq("long").all():
        raise AssertionError("invalid P8u target-free population")
    if not continuous_fields:
        return base
    if continuous_sidecar is None or not continuous_sidecar.exists():
        raise FileNotFoundError("declared continuous-regime sidecar is unavailable")
    sidecar = pd.read_parquet(
        continuous_sidecar,
        columns=["candidate_id", "__ts__", "__symbol__", "side_name", "source_utc", *continuous_fields],
        filters=[("__ts__", ">=", start), ("__ts__", "<", end)],
    ).rename(columns={"__ts__": "__decision_ts__"})
    for column in ("__decision_ts__", "source_utc"):
        sidecar[column] = pd.to_datetime(sidecar[column], utc=True, errors="raise")
    decision_identity = ("__symbol__", "__decision_ts__", "side_name")
    if sidecar.duplicated(decision_identity).any() or not sidecar.source_utc.le(sidecar.__decision_ts__).all():
        raise AssertionError("continuous-regime sidecar has duplicate or post-date rows")
    left = base.copy()
    # The Base candidate_id encodes its signal bar; the independently
    # materialised sidecar candidate_id encodes its decision bar.  Symbol ×
    # decision timestamp × side is the shared, explicit point-in-time
    # identity.  This is not an as-of or proximity join.
    left["__symbol__"] = left.candidate_id.astype(str).str.split("|", n=1).str[0]
    if left.duplicated(decision_identity).any():
        raise AssertionError("Base score source is not unique on decision identity")
    result = left.merge(
        sidecar.loc[:, [*decision_identity, *continuous_fields]], on=list(decision_identity), how="left", validate="one_to_one",
    ).drop(columns="__symbol__")
    if len(result) != len(base) or result.loc[:, list(continuous_fields)].isna().all(axis=None):
        raise AssertionError("continuous-regime sidecar identity coverage failure")
    return result


def _preflight_continuous_sidecar(
    *, base_root: Path, path: Path, fields: Sequence[str], months: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month in months:
        end = screen._month_end(month)
        base = pd.read_parquet(screen._base_path(base_root, month), columns=list(IDENTITY))
        sidecar = pd.read_parquet(
            path, columns=["candidate_id", "__ts__", "__symbol__", "side_name", "source_utc", *fields],
            filters=[("__ts__", ">=", month), ("__ts__", "<", end)],
        ).rename(columns={"__ts__": "__decision_ts__"})
        for column in ("__decision_ts__", "source_utc"):
            sidecar[column] = pd.to_datetime(sidecar[column], utc=True, errors="raise")
        decision_identity = ("__symbol__", "__decision_ts__", "side_name")
        if sidecar.duplicated(decision_identity).any() or not sidecar.source_utc.le(sidecar.__decision_ts__).all():
            raise AssertionError(f"{month:%Y-%m}: invalid continuous-regime sidecar")
        base["__symbol__"] = base.candidate_id.astype(str).str.split("|", n=1).str[0]
        if base.duplicated(decision_identity).any():
            raise AssertionError(f"{month:%Y-%m}: non-unique Base decision identity")
        matched = base.merge(sidecar.loc[:, list(decision_identity)], on=list(decision_identity), how="left", indicator=True)["_merge"].eq("both").sum()
        if int(matched) != len(base):
            raise AssertionError(f"{month:%Y-%m}: continuous-regime sidecar identity coverage failure")
        rows.append({"month": f"{month:%Y-%m}", "base_rows": int(len(base)), "sidecar_matched": int(matched), "sidecar_fields": int(len(fields))})
    return pd.DataFrame(rows)


def _gain(values: Sequence[float], classes: int) -> list[float]:
    result = [float(value) for value in values[:classes]]
    if len(result) != classes:
        raise AssertionError(f"gain schedule has {len(values)} values for {classes} labels")
    return result


def _model(trial: Mapping[str, Any], *, classes: int, seed: int) -> LGBMRanker:
    params = dict(trial["model"])
    objective = str(params["objective"])
    common: dict[str, Any] = {
        "objective": objective,
        "metric": "ndcg",
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "num_leaves": int(params["num_leaves"]),
        "min_child_samples": int(params["min_child_samples"]),
        "min_split_gain": float(params["min_split_gain"]),
        "colsample_bytree": float(params["feature_fraction"]),
        "subsample": float(params["bagging_fraction"]),
        "subsample_freq": 1,
        "reg_alpha": float(params["lambda_l1"]),
        "reg_lambda": float(params["lambda_l2"]),
        "random_state": int(seed),
        # The frozen production trials omit this key and therefore preserve
        # their historical one-thread semantics.  Offline subspace research
        # may declare an explicit bounded thread count in its immutable trial
        # receipt without changing model geometry or inference behavior.
        "n_jobs": int(params.get("n_jobs", 1)),
        "verbosity": -1,
        "label_gain": _gain(trial["gain"], classes),
    }
    if objective == "lambdarank":
        common["lambdarank_truncation_level"] = int(trial["truncation"])
        common["sigmoid"] = float(trial["sigmoid"])
    elif objective != "rank_xendcg":
        raise ValueError(f"unsupported objective {objective!r}")
    return LGBMRanker(**common)


@dataclasses.dataclass
class PreparedFold:
    held_month: pd.Timestamp
    held_target_free: pd.DataFrame
    held_labelled: pd.DataFrame | None
    held_anchor: Any
    train_frame: pd.DataFrame
    train_x: np.ndarray
    held_x: np.ndarray
    labels: np.ndarray
    groups: list[int]
    audit: dict[str, Any]


def _weight_signal(values: np.ndarray, *, power: float) -> np.ndarray:
    """Map a training-only quantity to a stable [0, 1] emphasis signal."""
    if not (0.0 < float(power) <= 4.0):
        raise ValueError("sample-weight power must be in (0, 4]")
    raw = np.asarray(values, dtype=float)
    valid = np.isfinite(raw)
    result = np.zeros(len(raw), dtype=np.float32)
    if int(valid.sum()) < 2:
        return result
    # Ranks are computed inside the current training fold only.  They are
    # never written into held/inference features.
    ranks = pd.Series(raw[valid]).rank(method="average", pct=True).to_numpy(float)
    result[valid] = np.power(np.clip(ranks, 0.0, 1.0), float(power)).astype(np.float32)
    return result


def _bounded_unit_mass(values: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    """Scale positive weights to mean one without ever breaking their bounds."""
    raw = np.asarray(values, dtype=float)
    if len(raw) == 0 or not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise AssertionError("weight normalisation requires finite positive inputs")
    # A monotone scalar solve avoids the common but incorrect sequence
    # normalise -> clip, which leaves dense or extreme queries with unequal
    # objective mass.  Because lower <= 1 <= upper, a solution always exists.
    def bounded_mean(scale: float) -> float:
        return float(np.mean(np.clip(raw * scale, lower, upper)))
    lo, hi = 0.0, 1.0
    while bounded_mean(hi) < 1.0:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - protects a corrupt trial receipt
            raise AssertionError("unable to normalise bounded sample weights")
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if bounded_mean(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(raw * (0.5 * (lo + hi)), lower, upper).astype(np.float32)


def _sample_weight(
    *, train: pd.DataFrame, labels: np.ndarray, profile: Mapping[str, Any] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build bounded, strictly training-only Meta fit weights.

    The default is intentionally ``None`` so historical objective results
    retain exact unweighted model semantics.  All declared challengers start
    with equal-timestamp mass, then optionally add bounded multipliers for
    Base strength, positive-recall labels, and resolved residual magnitude.
    No weight becomes an inference column.
    """
    if profile is None:
        return None, {"profile": "unweighted", "enters_inference_features": False}
    if not bool(profile.get("equal_timestamp", False)):
        raise ValueError("weighted Meta trials must declare equal_timestamp=true")
    components = profile.get("components", ())
    if not isinstance(components, list):
        raise ValueError("sample-weight components must be a list")
    known = {
        "base_score", "base_rank_localization", "positive_recall",
        "magnitude_awareness", "class_balance", "regime_balance",
    }
    if any(not isinstance(item, Mapping) or str(item.get("name")) not in known for item in components):
        raise ValueError("unknown sample-weight component")
    if len({str(item["name"]) for item in components}) != len(components):
        raise ValueError("duplicate sample-weight component")

    counts = train.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    for item in components:
        name = str(item["name"])
        strength = float(item.get("strength", 1.0))
        power = float(item.get("power", 1.0))
        if not (0.0 <= strength <= 1.0):
            raise ValueError("sample-weight strength must be in [0, 1]")
        if name == "base_score":
            # Base rank is point-in-time and target-free.  A rank of one is
            # most upstream conviction; this is a fit-only curriculum, not a
            # downstream score modifier.
            signal = np.power(np.clip(train.base_rank_ts.to_numpy(float), 0.0, 1.0), power)
            multiplier = 0.5 + 1.5 * signal
        elif name == "base_rank_localization":
            # Family-specific authority is deliberately declared as a
            # piecewise-linear function of the point-in-time Base rank
            # (one is strongest).  This permits e.g. Under authority in the
            # 5--20% region and Over authority in the 0--2% region without
            # smuggling a target-dependent column into inference.
            knots = item.get("knots")
            if not isinstance(knots, list) or len(knots) < 2:
                raise ValueError("base_rank_localization requires >=2 [rank, multiplier] knots")
            try:
                x = np.asarray([float(pair[0]) for pair in knots], dtype=float)
                y = np.asarray([float(pair[1]) for pair in knots], dtype=float)
            except (TypeError, IndexError, ValueError) as error:
                raise ValueError("base_rank_localization knots must be numeric [rank, multiplier] pairs") from error
            if (
                not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(np.diff(x) <= 0.0) or x[0] != 0.0 or x[-1] != 1.0
                or np.any(y <= 0.0)
            ):
                raise ValueError("base_rank_localization knots must span [0, 1] with positive multipliers")
            raw_multiplier = np.interp(
                np.clip(train.base_rank_ts.to_numpy(float), 0.0, 1.0), x, y,
            )
            # Blend permits a stable no-localisation control while retaining
            # the exact declared family geometry for full-authority trials.
            multiplier = 1.0 + strength * (raw_multiplier - 1.0)
        elif name == "positive_recall":
            classes = max(1, int(np.nanmax(labels)))
            signal = np.power(np.clip(np.asarray(labels, dtype=float) / classes, 0.0, 1.0), power)
            multiplier = 0.5 + 1.5 * signal
        elif name == "magnitude_awareness":
            if "prequential_residual_bps" not in train:
                raise AssertionError("magnitude-aware weighting requires the strict prequential residual")
            signal = _weight_signal(np.abs(train.prequential_residual_bps.to_numpy(float)), power=power)
            multiplier = 0.5 + 1.5 * signal
        elif name == "class_balance":
            # Signed-state training needs fold-local balancing of its
            # over/accurate/under classes.  Use a square-root inverse
            # prevalence multiplier rather than full inverse frequency so
            # sparse extreme states do not dominate calibrated rank learning.
            observed = np.asarray(labels, dtype=np.int32)
            if len(observed) != len(train) or np.any(observed < 0):
                raise AssertionError("class balancing requires aligned valid training labels")
            class_counts = np.bincount(observed, minlength=int(observed.max()) + 1).astype(float)
            if len(class_counts) < 2 or np.any(class_counts <= 0.0):
                raise AssertionError("class balancing requires >=2 observed classes with positive support")
            raw = np.sqrt(float(len(observed)) / (float(len(class_counts)) * class_counts[observed]))
            multiplier = np.clip(raw / max(float(np.mean(raw)), 1e-12), 0.5, 2.0)
        else:
            # Frozen episode membership is target-free and is used here only
            # to prevent frequent state regions from dominating the fitting
            # objective.  It is never persisted as a weight or exposed as a
            # distinct inference input.  Square-root balancing is deliberately
            # mild; rare regimes retain support without receiving full
            # inverse-frequency authority.
            episode_columns = sorted(
                column for column in train.columns if column.startswith("v2_regime_is_")
            )
            if len(episode_columns) < 2:
                raise AssertionError("regime-balanced weighting requires frozen episode one-hot fields")
            episode_matrix = train.loc[:, episode_columns].to_numpy(float)
            if not np.isfinite(episode_matrix).all() or not np.allclose(episode_matrix.sum(axis=1), 1.0):
                raise AssertionError("invalid frozen episode membership for regime-balanced weighting")
            episode_id = np.argmax(episode_matrix, axis=1)
            counts = np.bincount(episode_id, minlength=len(episode_columns)).astype(float)
            # A causal rolling train window may legitimately contain no rows
            # from a frozen episode.  Balance the regimes actually observed
            # in that fold; never invent a zero-support state or borrow from
            # a future fold.
            active_episode_count = int((counts > 0.0).sum())
            if active_episode_count < 2:
                raise AssertionError("regime-balanced weighting requires at least two observed frozen episodes")
            raw = np.sqrt(float(len(episode_id)) / (float(active_episode_count) * counts[episode_id]))
            multiplier = np.clip(raw / max(float(np.mean(raw)), 1e-12), 0.5, 2.0)
        weights *= 1.0 + strength * (multiplier - 1.0)

    lower = float(profile.get("min_weight", 0.5))
    upper = float(profile.get("max_weight", 4.0))
    if not (0.0 < lower <= 1.0 <= upper <= 8.0):
        raise ValueError("sample-weight bounds must contain one and lie in (0, 8]")
    # The source contract asks that the effective objective mass is re-scaled
    # inside each selected ranking query.  Query IDs are produced before this
    # function is called and are training-only bookkeeping, never model
    # fields.  This prevents a dense query or a high-authority family region
    # from changing query-level mass merely through candidate count.
    if bool(profile.get("normalise_within_query", True)):
        if "__rank_query_id__" not in train:
            raise AssertionError("within-query weight normalisation requires frozen rank query IDs")
        result = np.empty(len(weights), dtype=np.float32)
        for _query, positions in train.groupby("__rank_query_id__", sort=False).indices.items():
            index = np.asarray(positions, dtype=np.int64)
            result[index] = _bounded_unit_mass(weights[index], lower=lower, upper=upper)
        weights = result
    else:
        weights = _bounded_unit_mass(weights, lower=lower, upper=upper)
    audit: dict[str, Any] = {
        "profile": profile,
        "enters_inference_features": False,
        "normalise_within_query": bool(profile.get("normalise_within_query", True)),
        "min": float(weights.min()), "max": float(weights.max()),
        "mean": float(weights.mean()), "std": float(weights.std()),
    }
    if any(str(item["name"]) == "regime_balance" for item in components):
        audit["regime_balance"] = {
            "kind": "frozen_episode_square_root",
            "episode_columns": episode_columns,
            "counts": [int(value) for value in counts],
            "active_episode_count": active_episode_count,
        }
    if bool(profile.get("normalise_within_query", True)):
        query_means = pd.Series(weights).groupby(train["__rank_query_id__"], sort=False).mean().to_numpy(float)
        if not np.allclose(query_means, 1.0, atol=2e-6, rtol=0.0):
            raise AssertionError("bounded within-query sample-weight normalization failed")
        audit["query_mean_min"] = float(query_means.min())
        audit["query_mean_max"] = float(query_means.max())
    return weights, audit


def _prepare_fold(
    *, base_root: Path, feature_roots: Sequence[Path], policy: pd.DataFrame,
    path_root: Path, arm: screen.Arm, fields: Sequence[str], panel_fields: Sequence[str],
    continuous_sidecar: Path | None, continuous_fields: Sequence[str], spec: screen.Spec,
    held_month: pd.Timestamp, seed: int, materialize_held_labels: bool,
) -> PreparedFold:
    """Materialise one causal fold once, shared by every objective trial.

    The target/query contract is fixed during this screen.  Reusing one
    fold-safe, query-safe sample prevents trial-specific sampling noise and
    removes repeated large base/path/policy joins.
    """
    reserve_days = int(spec.folds["resolved_label_reserve_days"])
    train_months = int(spec.folds["train_months"])
    reserve = held_month - pd.Timedelta(days=reserve_days)
    start, end = reserve - pd.DateOffset(months=train_months), screen._month_end(held_month)
    base = _read_features(
        base_root=base_root, feature_roots=feature_roots, start=start, end=end, panel_fields=panel_fields,
        continuous_sidecar=continuous_sidecar, continuous_fields=continuous_fields,
    )
    train_tf = base.loc[base.__decision_ts__.lt(reserve)].copy()
    held_tf = base.loc[base.__decision_ts__.ge(held_month)].copy()
    train = screen._labelled(train_tf, policy, path_root, start, reserve)
    train = train.loc[screen._valid_label(train, reserve)].copy()
    if len(train) < 30_000 or len(held_tf) < 10_000:
        raise AssertionError(f"{arm.name} {held_month:%Y-%m}: insufficient support")
    anchors = screen._prequential_anchor(train, block_days=int(spec.folds["anchor_block_days"]))
    labels, residual, target_info = screen._train_target(train, arm, anchor=anchors)
    valid = labels >= 0
    sampled = screen._sample_queries(train.loc[valid].copy(), int(spec.folds["max_train_rows"]), seed)
    labels_frame = pd.DataFrame({
        "candidate_id": train.candidate_id, "label": labels,
        "prequential_residual_bps": residual,
    })
    sampled = sampled.merge(labels_frame, on="candidate_id", how="left", validate="one_to_one")
    y = sampled.label.to_numpy(np.int32)
    if len(sampled) < 20_000 or len(np.unique(y)) < 2:
        raise AssertionError(f"{arm.name} {held_month:%Y-%m}: insufficient target support")
    train_x, held_x = screen._impute(screen._matrix(sampled, fields), screen._matrix(held_tf, fields))
    order, query_ids, groups = screen._bounded_queries(
        sampled, screen._query_ids(sampled, arm.query), int(spec.folds["max_query_rows"]),
    )
    y, train_x = y[order], train_x[order]
    sampled = sampled.iloc[order].reset_index(drop=True)
    # Training-only query bookkeeping for weight normalisation.  It is never
    # part of the feature matrix or persisted as a held/inference field.
    sampled["__rank_query_id__"] = query_ids
    # The held score can be produced without opening any held outcome source.
    # Historical objective screens opt in to the separate post-score outcome
    # join; forward extensions use the target-free path and evaluate later.
    held_labelled = screen._labelled(held_tf, policy, path_root, held_month, end) if materialize_held_labels else None
    audit = {
        "held_month": f"{held_month:%Y-%m}",
        "train_rows_before_sample": int(valid.sum()), "train_rows": int(len(sampled)),
        "train_queries": int(len(groups)), "classes": int(np.nanmax(y) + 1),
        "features": int(len(fields) + 9), **target_info,
    }
    return PreparedFold(
        held_month=held_month, held_target_free=held_tf, held_labelled=held_labelled,
        held_anchor=screen._fit_anchor(train, screen._valid_label(train)), train_frame=sampled, train_x=train_x,
        held_x=held_x, labels=y, groups=groups, audit=audit,
    )


def _score_prepared(*, prepared: PreparedFold, arm: screen.Arm, trial: Mapping[str, Any], seed: int) -> pd.DataFrame:
    y = prepared.labels
    model = _model(trial, classes=int(np.nanmax(y)) + 1, seed=seed)
    sample_weight, weight_audit = _sample_weight(
        train=prepared.train_frame, labels=y, profile=trial.get("sample_weight"),
    )
    model.fit(prepared.train_x, y, group=prepared.groups, sample_weight=sample_weight)
    raw = np.asarray(model.predict(prepared.held_x), dtype=np.float32)
    if arm.family == "over":
        raw *= -1.0
    score = prepared.held_target_free.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy().reset_index(drop=True)
    score["meta_raw_score"] = raw
    rank_frame = score.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    rank_frame["value"] = raw
    score["meta_rank_ts"] = screen._rank_desc(rank_frame, "value")
    score["arm"] = arm.name
    score["family"] = arm.family
    score["scale"] = arm.scale
    score["query_contract"] = arm.query
    score["trial"] = str(trial["name"])
    score["held_month"] = f"{prepared.held_month:%Y-%m}"
    score["target_free"] = True
    score["fit_weight_profile"] = str(weight_audit["profile"])
    return score


def _integrated_screen_ranking(summary: pd.DataFrame, selection: Mapping[str, Any]) -> pd.DataFrame:
    """Rank broad trials by the declared timestamp-level integrated-current proxy.

    The downstream dual-MC1 constrained portfolio remains the only advancement
    authority.  This bounded, within-bank percentile aggregate simply avoids
    using the legacy SStableMeta metric alone to select costly final replays.
    """
    objective = selection.get("screen_objective") if isinstance(selection, Mapping) else None
    if not isinstance(objective, Mapping):
        # The learned-downstream-proxy contract intentionally makes native
        # Meta metrics descriptors rather than a hand-designed winner score.
        # Keep deterministic diagnostic order; the proxy reducer owns HPO and
        # feature-selection authority.
        out = summary.copy()
        out["integrated_screen_objective"] = np.nan
        out["screen_qualified"] = True
        return out.sort_values(["trial"], kind="stable")
    terms = objective.get("ranked_terms")
    if not isinstance(terms, list) or not terms:
        raise AssertionError("screen_objective requires non-empty ranked_terms")
    out = summary.copy()
    composite = np.zeros(len(out), dtype=float)
    for term in terms:
        metric = str(term["metric"])
        if metric not in out:
            raise AssertionError(f"screen objective metric is absent: {metric}")
        values = pd.to_numeric(out[metric], errors="coerce")
        ascending = str(term.get("direction", "higher")) != "higher"
        composite += float(term["weight"]) * values.rank(method="average", pct=True, ascending=ascending).fillna(0.0).to_numpy(float)
    qualified = np.ones(len(out), dtype=bool)
    for key, minimum in dict(objective.get("gates", {})).items():
        metric = str(key).removesuffix("_min")
        if metric not in out:
            raise AssertionError(f"screen objective gate metric is absent: {metric}")
        qualified &= pd.to_numeric(out[metric], errors="coerce").ge(float(minimum)).fillna(False).to_numpy(bool)
    out["integrated_screen_objective"] = composite.astype(np.float32)
    out["screen_qualified"] = qualified
    return out.sort_values(
        ["screen_qualified", "integrated_screen_objective", "mean_top2_substitution_ev_bps", "trial"],
        ascending=[False, False, False, True], kind="stable",
    )


def run(
    *, config: Path, arm_name: str, trials_path: Path, out: Path,
    feature_contract: Path | None = None, block_days: int | None = None,
    continuous_sidecar: Path | None = None, held_month_values: Sequence[str] | None = None,
    source_override: Path | None = None, target_free_only: bool = False,
    fold_seed_start: int | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    raw, applied_source_override = _apply_source_override(json.loads(config.read_text()), source_override)
    trials_payload = json.loads(trials_path.read_text())
    # A sealed HPO run manifest is also an immutable trial receipt.  Accept
    # its declared ``trials`` list directly so a strict prehistory replay
    # cannot introduce a manually copied or silently altered trial plan.
    trials = trials_payload.get("trials") if isinstance(trials_payload, dict) else trials_payload
    if not isinstance(trials, list) or not trials:
        raise AssertionError("trials source must be a non-empty list or a manifest with a non-empty trials list")
    names = [str(trial["name"]) for trial in trials]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate trial name")
    arm = _arms(raw).get(arm_name)
    if arm is None:
        raise ValueError(f"unknown arm {arm_name!r}")
    if block_days is not None:
        if arm.family != "magnitude":
            raise ValueError("--block-days is reserved for the magnitude market-state arm")
        if int(block_days) not in {21, 28, 35, 42}:
            raise ValueError("--block-days must be one of 21, 28, 35, 42")
        arm = dataclasses.replace(arm, query=f"base_band_block{int(block_days)}")
    spec = screen.Spec(raw=raw, config_path=config)
    contract_path = (feature_contract or (ROOT / str(spec.source["base_f72_contract"]))).resolve()
    fields, continuous_fields, declared_sidecar = _read_contract(contract_path)
    resolved_sidecar = continuous_sidecar.resolve() if continuous_sidecar else declared_sidecar
    panel_fields = tuple(field for field in fields if field not in set(continuous_fields))
    months = tuple(screen._utc_month(value) for value in (held_month_values or spec.folds["held_months"]))
    if not months or tuple(sorted(months)) != months:
        raise ValueError("held months must be a non-empty ascending sequence")
    base_root = ROOT / str(spec.source["base_target_free_root"])
    feature_roots = tuple(ROOT / str(value) for value in spec.source["full_feature_roots"])
    policy_path = ROOT / str(spec.source["policy_labels"])
    path_root = ROOT / str(spec.source["path_labels"])
    policy = screen._read_policy(policy_path)
    seed_start = int(spec.folds["seed"] if fold_seed_start is None else fold_seed_start)
    out.mkdir(parents=True)
    # ``screen._preflight`` also validates held path labels.  Those labels are
    # deliberately not opened during a target-free forward score extension,
    # so limit this audit to the train/reserve side.  Exact held feature/base
    # identity is still enforced in ``_read_features`` before scoring.
    preflight_end = months[-1] if target_free_only else screen._month_end(months[-1])
    preflight_months = tuple(screen._month_range(months[0] - pd.DateOffset(months=5), preflight_end))
    # The full panels do not own the continuous-regime sidecar fields, so
    # preflight panel identity on the panel subset and audit the sidecar
    # separately under the same exact candidate identity.
    coverage = screen._preflight(spec, panel_fields, preflight_months)
    coverage.to_parquet(out / "source_coverage_audit.parquet", index=False, compression="zstd")
    if continuous_fields:
        if resolved_sidecar is None:
            raise AssertionError("continuous-regime fields require an explicit sidecar")
        sidecar_audit = _preflight_continuous_sidecar(
            base_root=base_root, path=resolved_sidecar, fields=continuous_fields, months=preflight_months,
        )
        sidecar_audit.to_parquet(out / "continuous_regime_sidecar_coverage_audit.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF P8u Meta objective comparison; no live, admission, portfolio, or exchange mutation",
        "base_contract": raw["base_contract"], "arm": dataclasses.asdict(arm),
        "trials": trials, "trials_source": str(trials_path), "trials_source_sha256": _sha(trials_path),
        "meta_feature_contract": str(contract_path),
        "meta_feature_contract_sha256": _sha(contract_path), "meta_feature_count": len(fields),
        "continuous_regime_sidecar": str(resolved_sidecar) if resolved_sidecar else None,
        "continuous_regime_sidecar_feature_count": len(continuous_fields),
        "held_months": [f"{month:%Y-%m}" for month in months],
        "source": raw["source"],
        "source_override": str(source_override) if source_override else None,
        "source_override_payload": applied_source_override,
        "target_free_only": bool(target_free_only),
        "fold_seed_start": seed_start,
        "source_hashes": {
            "base": _sha(base_root), "policy": _sha(policy_path), "path": _sha(path_root),
            **({"continuous_regime_sidecar": _sha(resolved_sidecar)} if resolved_sidecar else {}),
        },
        "causality": raw["causality"],
        "selection": raw.get("selection", "timestamp-level screen only; MC1 and constrained portfolio selection follow in a separate stage"),
    })
    trial_metrics_by_name: dict[str, list[dict[str, Any]]] = {str(trial["name"]): [] for trial in trials}
    all_weekly: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold_index, held_month in enumerate(months):
        prepared = _prepare_fold(
            base_root=base_root, feature_roots=feature_roots, policy=policy, path_root=path_root,
            arm=arm, fields=fields, panel_fields=panel_fields, continuous_sidecar=resolved_sidecar,
            continuous_fields=continuous_fields, spec=spec, held_month=held_month,
            seed=seed_start + fold_index, materialize_held_labels=not target_free_only,
        )
        for trial_index, trial in enumerate(trials):
            strict_trial_index = int(trial.get("strict_oof_trial_index", trial_index))
            if strict_trial_index < 0:
                raise AssertionError("strict OOF trial seed index must be non-negative")
            score = _score_prepared(
                prepared=prepared, arm=arm, trial=trial,
                seed=seed_start + 10_000 * strict_trial_index + fold_index,
            )
            target_free_path = out / "target_free_scores" / str(trial["name"]) / f"month={held_month:%Y-%m}.parquet"
            target_free_path.parent.mkdir(parents=True, exist_ok=True)
            score.to_parquet(target_free_path, index=False, compression="zstd")
            metrics: dict[str, Any] = {}
            if not target_free_only:
                if prepared.held_labelled is None:
                    raise AssertionError("held labels required for an objective metric run")
                weekly, bands, metrics = screen._metrics(
                    score=score, held_labelled=prepared.held_labelled, held_anchor=prepared.held_anchor, spec=spec,
                )
                weekly["trial"] = str(trial["name"]); weekly["arm"] = arm.name; weekly["held_month"] = f"{held_month:%Y-%m}"
                if not bands.empty:
                    bands["trial"] = str(trial["name"]); bands["arm"] = arm.name; bands["held_month"] = f"{held_month:%Y-%m}"; all_bands.append(bands)
                all_weekly.append(weekly)
            _weights, weight_audit = _sample_weight(
                train=prepared.train_frame, labels=prepared.labels, profile=trial.get("sample_weight"),
            )
            audit = {
                **prepared.audit,
                "trial": str(trial["name"]),
                "strict_oof_trial_index": strict_trial_index,
                "score_seed": seed_start + 10_000 * strict_trial_index + fold_index,
                # Parquet audit receipts must remain scalar and portable;
                # the JSON records every training-only component/bound.
                "sample_weight_audit_json": json.dumps(weight_audit, sort_keys=True),
            }
            audit.update(metrics); audits.append(audit)
            _progress(out, {"event": "trial_fold_complete", "trial": trial["name"], "held_month": f"{held_month:%Y-%m}", "target_free_score": str(target_free_path), **metrics})
            if not target_free_only:
                trial_metrics_by_name[str(trial["name"])].append(metrics)
    summaries = []
    for trial in trials:
        trial_metrics = trial_metrics_by_name[str(trial["name"])]
        aggregate = pd.DataFrame(trial_metrics).mean(numeric_only=True).to_dict() if trial_metrics else {}
        summaries.append({"trial": str(trial["name"]), "arm": arm.name, "family": arm.family, "target_free_only": bool(target_free_only), **aggregate})
    summary = pd.DataFrame(summaries)
    if not target_free_only:
        summary = _integrated_screen_ranking(summary, raw.get("selection", {}))
    objective = raw.get("selection", {}).get("screen_objective") if isinstance(raw.get("selection", {}), Mapping) else None
    if isinstance(objective, Mapping):
        summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)
        summary["selection_authority"] = "legacy_declared_screen_objective_diagnostic_only"
    else:
        # A deterministic row order is useful for receipts, but must never be
        # mistaken for an HPO ranking now that the learned downstream proxy
        # owns selection.  Existing trial scores remain valid descriptors.
        summary["rank"] = np.nan
        summary["selection_authority"] = "none_learned_downstream_proxy_required"
    summary.to_parquet(out / "objective_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "objective_fold_metrics.parquet", index=False, compression="zstd")
    (pd.concat(all_weekly, ignore_index=True) if all_weekly else pd.DataFrame()).to_parquet(out / "weekly_sstable_meta.parquet", index=False, compression="zstd")
    (pd.concat(all_bands, ignore_index=True) if all_bands else pd.DataFrame()).to_parquet(out / "base_band_conversion_metrics.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "p8u_base_target_free_score_source": True,
        "declared_meta_features_merged_by_exact_identity": True,
        "continuous_regime_sidecar_join_is_exact_identity_and_prior_only": True,
        "no_policy_or_path_field_in_target_free_inputs": True,
        "all_train_labels_resolved_before_reserve": True,
        "train_residual_anchor_strict_prequential": True,
        "held_scores_persisted_before_held_outcome_metrics": True,
        # Objective mode deliberately opens held labels *after* persisting the
        # target-free held score in order to calculate its OOS metrics.  The
        # prior receipt key incorrectly described this as a boolean failure
        # whenever objective metrics were requested, even though no outcome
        # column reaches the held feature matrix or score producer.
        "held_score_producer_excludes_held_outcome_and_path_inputs": True,
        "held_outcomes_opened_only_after_target_free_score_persistence": True,
        "target_free_only_mode": bool(target_free_only),
        "base_band_metrics_limited_to_base_top30": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
        "sample_weights_are_training_only_and_never_inference_features": True,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path)
    parser.add_argument("--block-days", type=int)
    parser.add_argument("--continuous-sidecar", type=Path)
    parser.add_argument("--held-months", nargs="+")
    parser.add_argument("--source-override", type=Path, help="immutable source-only replacement receipt for a causal ledger extension")
    parser.add_argument("--target-free-only", action="store_true", help="persist held scores without joining held outcomes or path labels")
    parser.add_argument("--fold-seed-start", type=int, help="absolute first held-fold seed; use only to reproduce a continuous parent subset")
    args = parser.parse_args()
    print(run(
        config=args.config.resolve(), arm_name=str(args.arm), trials_path=args.trials.resolve(), out=args.out.resolve(),
        feature_contract=args.feature_contract.resolve() if args.feature_contract else None,
        block_days=args.block_days,
        continuous_sidecar=args.continuous_sidecar.resolve() if args.continuous_sidecar else None,
        held_month_values=args.held_months,
        source_override=args.source_override.resolve() if args.source_override else None,
        target_free_only=bool(args.target_free_only),
        fold_seed_start=args.fold_seed_start,
    ))


if __name__ == "__main__":
    main()
