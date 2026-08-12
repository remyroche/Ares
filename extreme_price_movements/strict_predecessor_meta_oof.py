"""Materialise a compact, strictly-nested predecessor-meta OOF hand-off.

``S2`` is allowed to consume a predecessor-meta representation only when that
representation could have existed for the very candidate being scored.  This
module makes that requirement concrete.  For every transport, side and
chronological refit block it fits a small ridge predecessor only on
``inner_oof`` rows whose labels were resolved *strictly before* the first
candidate in the scored block.  It never trains on an outer row, and it never
uses a row to form its own predecessor feature.

The predecessor has exactly six declared input semantics and emits exactly six
coefficient contributions.  Emitting the contributions rather than a seventh
opaque score gives S2 a small, auditable representation while retaining the
meaning of the individual portability/support/risk signals.  The intercept is
deliberately not emitted.

The writer creates a new immutable ledger rather than mutating the source
base-to-meta ledger.  Both the narrow feature table and the joined ledger use
the full candidate identity; an outer join must be exact in both directions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


SCHEMA = "strict_predecessor_meta_oof_v1"
STATUS = "STRICT_NESTED_PREDECESSOR_META_OOF_MATERIALIZED"
SOURCE_LEDGER_STATUS = "STRICT_BASE_TO_META_LEDGER_ASSEMBLED"
IDENTITY: tuple[str, ...] = (
    "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
)
PREDECESSOR_SEMANTICS: tuple[str, ...] = (
    "upgrade_portability",
    "downgrade_portability",
    "unstable_upgrade_share",
    "covariance_break_share",
    "support_score",
    "reasoning_entropy",
)
PREDECESSOR_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"predecessor_meta__{name}" for name in PREDECESSOR_SEMANTICS
)
LINEAGE_COLUMNS: tuple[str, ...] = (
    "predecessor_oof_fit_end_ts",
    "predecessor_oof_generated_ts",
    "predecessor_oof_available_ts",
    "predecessor_same_side_strict_oof",
)
RAW_LEAF_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")


class StrictPredecessorMetaOOFError(ValueError):
    """Raised when a predecessor feature cannot prove strict nesting."""


@dataclass(frozen=True)
class StrictPredecessorMetaOOFConfig:
    """Fixed, compact predecessor-meta training contract.

    ``source_feature_map`` maps each mandated semantic to one already-causal,
    scalar field in the immutable ledger.  It is intentionally explicit:
    inferring a source from a broad name pattern would make this a hidden
    feature-selection step and can silently choose a different field later.
    """

    source_feature_map: Mapping[str, str] = field(default_factory=dict)
    min_train_rows: int = 32
    ridge_alpha: float = 10.0
    refit_interval_hours: int = 24
    robust_scale_floor: float = 1e-6

    def validate(self) -> None:
        supplied = {str(key): str(value) for key, value in dict(self.source_feature_map).items()}
        if set(supplied) != set(PREDECESSOR_SEMANTICS):
            missing = sorted(set(PREDECESSOR_SEMANTICS).difference(supplied))
            extra = sorted(set(supplied).difference(PREDECESSOR_SEMANTICS))
            raise StrictPredecessorMetaOOFError(
                "source_feature_map must declare exactly the six mandated semantics "
                f"(missing={missing}, extra={extra})"
            )
        values = tuple(supplied[name] for name in PREDECESSOR_SEMANTICS)
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise StrictPredecessorMetaOOFError(
                "the six predecessor semantics require six distinct non-empty source columns"
            )
        if any(_raw_leaf_name(value) for value in values):
            raise StrictPredecessorMetaOOFError("predecessor source features may not be raw leaf identifiers")
        if int(self.min_train_rows) < 2:
            raise StrictPredecessorMetaOOFError("min_train_rows must be at least two")
        if int(self.refit_interval_hours) < 1:
            raise StrictPredecessorMetaOOFError("refit_interval_hours must be positive")
        if not np.isfinite(self.ridge_alpha) or float(self.ridge_alpha) < 0.0:
            raise StrictPredecessorMetaOOFError("ridge_alpha must be finite and non-negative")
        if not np.isfinite(self.robust_scale_floor) or float(self.robust_scale_floor) <= 0.0:
            raise StrictPredecessorMetaOOFError("robust_scale_floor must be finite and positive")

    @property
    def source_columns(self) -> tuple[str, ...]:
        self.validate()
        values = {str(key): str(value) for key, value in dict(self.source_feature_map).items()}
        return tuple(values[name] for name in PREDECESSOR_SEMANTICS)


@dataclass(frozen=True)
class StrictPredecessorMetaOOFResult:
    """The narrow feature hand-off plus its exact-identity joined ledger."""

    features: pd.DataFrame
    ledger: pd.DataFrame
    fit_audit: pd.DataFrame
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class StrictPredecessorMetaOOFArtifact:
    """A verified immutable predecessor root usable by the S2 runner."""

    root: Path
    features: pd.DataFrame
    ledger: pd.DataFrame
    feature_columns: tuple[str, ...]
    manifest: Mapping[str, Any]
    ledger_path: Path


def _raw_leaf_name(value: object) -> bool:
    name = str(value).lower()
    return not name.startswith("base_reasoning__g1_leaf_assignment_count") and any(
        token in name for token in RAW_LEAF_TOKENS
    )


def _forbid_raw_leaf(columns: Iterable[object], *, source: str) -> None:
    bad = sorted(str(column) for column in columns if _raw_leaf_name(column))
    if bad:
        raise StrictPredecessorMetaOOFError(
            f"{source} contains raw fold-local leaf identifiers: {bad}"
        )


def _utc(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise StrictPredecessorMetaOOFError(f"{column} must contain valid UTC timestamps")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_digest(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["decision_ts"] = pd.to_datetime(values["decision_ts"], utc=True).astype("int64")
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    payload = values.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """Return the complete strict identity used by every predecessor guard.

    ``candidate_id`` identifies a candidate only within a transport run.  The
    immutable meta ledger deliberately permits the same identifier to appear
    in independent transports, so any global candidate-id check or join would
    either reject a valid ledger or, worse, compare the wrong rows.
    """

    return pd.MultiIndex.from_frame(frame.loc[:, list(IDENTITY)])


def _shares_exact_identity(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return bool(_identity_index(left).isin(_identity_index(right)).any())


def _normalise_ledger(
    ledger: pd.DataFrame, *, config: StrictPredecessorMetaOOFConfig,
) -> pd.DataFrame:
    config.validate()
    required = {
        *IDENTITY,
        "label_available_ts",
        "base_expected_bps",
        "realized_net_bps",
        "base_same_side_strict_oof",
        *config.source_columns,
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise StrictPredecessorMetaOOFError(f"source immutable ledger lacks required fields: {missing}")
    _forbid_raw_leaf(config.source_columns, source="predecessor source feature contract")
    work = ledger.copy()
    work["decision_ts"] = _utc(work, "decision_ts")
    work["label_available_ts"] = _utc(work, "label_available_ts")
    for column in ("candidate_id", "side_name", "fold_id", "transport", "meta_partition"):
        if work[column].isna().any() or work[column].astype(str).str.strip().eq("").any():
            raise StrictPredecessorMetaOOFError(f"{column} must be non-null and non-empty")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise StrictPredecessorMetaOOFError("predecessor source ledger must retain long/short separation")
    if not work["meta_partition"].astype(str).isin(("inner_oof", "outer_test")).all():
        raise StrictPredecessorMetaOOFError("predecessor source ledger has an unknown meta partition")
    if work.duplicated(list(IDENTITY)).any():
        raise StrictPredecessorMetaOOFError("source immutable ledger duplicates exact candidate identity")
    if not work["base_same_side_strict_oof"].fillna(False).astype(bool).all():
        raise StrictPredecessorMetaOOFError("all predecessor source rows must prove same-side strict base OOF")
    if not work["label_available_ts"].gt(work["decision_ts"]).all():
        raise StrictPredecessorMetaOOFError("every predecessor source label must resolve strictly after decision_ts")
    for column in ("base_expected_bps", "realized_net_bps", *config.source_columns):
        numeric = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise StrictPredecessorMetaOOFError(
                f"predecessor source {column} must be finite on every candidate"
            )
        work[column] = numeric
    return work.sort_values(
        ["transport", "side_name", "decision_ts", "candidate_id", "fold_id", "meta_partition"],
        kind="stable",
    ).reset_index(drop=True)


def _refit_anchor(values: pd.Series, *, interval_hours: int) -> pd.Series:
    width = int(pd.Timedelta(hours=int(interval_hours)).value)
    return pd.to_datetime((values.astype("int64") // width) * width, utc=True)


def _robust_standardise(
    train: np.ndarray, test: np.ndarray, *, floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.median(train, axis=0)
    scale = 1.4826 * np.median(np.abs(train - centre), axis=0)
    scale = np.maximum(scale, float(floor))
    return (train - centre) / scale, (test - centre) / scale


def _build_features(
    work: pd.DataFrame, *, config: StrictPredecessorMetaOOFConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit only preceding inner labels, one causal model per bounded block."""

    source_columns = config.source_columns
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for (transport, side), cell in work.groupby(["transport", "side_name"], sort=True, observed=True):
        local = cell.copy()
        local["__refit_anchor__"] = _refit_anchor(
            local["decision_ts"], interval_hours=config.refit_interval_hours,
        )
        for anchor, scored in local.groupby("__refit_anchor__", sort=True, observed=True):
            scored = scored.sort_values(
                ["decision_ts", "candidate_id", "fold_id", "meta_partition"], kind="stable"
            ).copy()
            first_decision = pd.Timestamp(scored["decision_ts"].min())
            # The inner restriction is the key nested boundary.  An outer
            # outcome must never train the predecessor, even for later outer
            # candidates.  Strict '< first_decision' additionally makes one
            # fitted block safe for every row that it scores.
            train = local.loc[
                local["meta_partition"].astype(str).eq("inner_oof")
                & local["label_available_ts"].lt(first_decision)
            ].sort_values(
                ["label_available_ts", "decision_ts", "candidate_id", "fold_id", "meta_partition"],
                kind="stable",
            )
            if not train.empty:
                if not train["side_name"].eq(side).all() or not train["transport"].eq(transport).all():
                    raise AssertionError("predecessor train scope crossed side or transport")
                if not train["meta_partition"].eq("inner_oof").all():
                    raise AssertionError("outer row entered predecessor training")
                if not train["label_available_ts"].lt(first_decision).all():
                    raise AssertionError("unresolved label entered predecessor training")
                # Since source labels resolve after their decisions, this is
                # also a direct same-row/in-sample guard rather than relying
                # on the partition name alone.
                if _shares_exact_identity(train, scored):
                    raise StrictPredecessorMetaOOFError(
                        "same exact candidate identity entered its own predecessor-meta training set"
                    )
                if not train["decision_ts"].lt(first_decision).all():
                    raise StrictPredecessorMetaOOFError(
                        "predecessor training contains an in-sample or future decision row"
                    )

            fitted = len(train) >= int(config.min_train_rows)
            if fitted:
                train_x = train.loc[:, list(source_columns)].to_numpy(dtype=float)
                scored_x = scored.loc[:, list(source_columns)].to_numpy(dtype=float)
                normal_train, normal_scored = _robust_standardise(
                    train_x, scored_x, floor=float(config.robust_scale_floor),
                )
                target = (
                    train["realized_net_bps"].to_numpy(dtype=float)
                    - train["base_expected_bps"].to_numpy(dtype=float)
                )
                model = Ridge(alpha=float(config.ridge_alpha), fit_intercept=True)
                model.fit(normal_train, target)
                contributions = normal_scored * np.asarray(model.coef_, dtype=float)[None, :]
                mode = "side_local_ridge_component_oof"
            else:
                contributions = np.zeros((len(scored), len(PREDECESSOR_FEATURE_COLUMNS)), dtype=float)
                mode = "strict_zero_predecessor_fallback_insufficient_prior_inner_rows"

            if not np.isfinite(contributions).all():
                raise StrictPredecessorMetaOOFError("predecessor component features are non-finite")
            max_label = train["label_available_ts"].max() if not train.empty else pd.NaT
            # With no eligible row there is no material fitted endpoint.  The
            # zero fallback is nevertheless causal, and its declared endpoint
            # is an explicit decision-exclusive sentinel; the audit preserves
            # ``NaT`` so it cannot be mistaken for a trained model.
            fit_end = max_label if not pd.isna(max_label) else first_decision - pd.Timedelta(nanoseconds=1)
            generated = first_decision
            if not pd.Timestamp(fit_end) < first_decision or generated > first_decision:
                raise AssertionError("predecessor OOF timestamp construction is unsafe")
            result = scored.loc[:, list(IDENTITY)].copy()
            for index, name in enumerate(PREDECESSOR_FEATURE_COLUMNS):
                result[name] = contributions[:, index].astype(np.float32)
            result["predecessor_oof_fit_end_ts"] = pd.Timestamp(fit_end)
            result["predecessor_oof_generated_ts"] = generated
            result["predecessor_oof_available_ts"] = generated
            result["predecessor_same_side_strict_oof"] = True
            rows.append(result)
            audits.append({
                "transport": str(transport),
                "side_name": str(side),
                "refit_anchor_ts": generated,
                "first_scored_decision_ts": first_decision,
                "last_scored_decision_ts": scored["decision_ts"].max(),
                "scored_rows": int(len(scored)),
                "train_rows": int(len(train)),
                "max_label_available_used": max_label,
                "strict_prior_resolved": bool(train.empty or train["label_available_ts"].lt(first_decision).all()),
                "inner_only_training": bool(train.empty or train["meta_partition"].eq("inner_oof").all()),
                "same_row_or_in_sample_rejected": bool(
                    train.empty or not _shares_exact_identity(train, scored)
                ),
                "fit_mode": mode,
                "source_feature_columns_json": json.dumps(list(source_columns)),
                "emitted_feature_columns_json": json.dumps(list(PREDECESSOR_FEATURE_COLUMNS)),
            })
    if not rows:
        raise StrictPredecessorMetaOOFError("source immutable ledger contains no candidate rows")
    features = pd.concat(rows, ignore_index=True)
    if features.duplicated(list(IDENTITY)).any():
        raise StrictPredecessorMetaOOFError("predecessor output duplicates exact candidate identity")
    if len(features) != len(work):
        raise StrictPredecessorMetaOOFError("predecessor OOF materialization dropped candidate rows")
    features = features.sort_values(
        ["transport", "meta_partition", "decision_ts", "candidate_id", "side_name", "fold_id"],
        kind="stable",
    ).reset_index(drop=True)
    return features, pd.DataFrame(audits)


def join_strict_predecessor_oof_features(
    ledger: pd.DataFrame, features: pd.DataFrame,
) -> pd.DataFrame:
    """Exact-identity join of the compact predecessor hand-off into a new ledger."""

    required = {*IDENTITY, *PREDECESSOR_FEATURE_COLUMNS, *LINEAGE_COLUMNS}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise StrictPredecessorMetaOOFError(f"predecessor features lack required columns: {missing}")
    if features.duplicated(list(IDENTITY)).any() or ledger.duplicated(list(IDENTITY)).any():
        raise StrictPredecessorMetaOOFError("exact candidate identity must be unique before predecessor join")
    collisions = sorted(set(ledger.columns).intersection(set(features.columns)).difference(IDENTITY))
    if collisions:
        raise StrictPredecessorMetaOOFError(
            f"source ledger already contains predecessor output columns: {collisions}"
        )
    left = ledger.copy()
    right = features.copy()
    left["decision_ts"] = _utc(left, "decision_ts")
    right["decision_ts"] = _utc(right, "decision_ts")
    merged = left.merge(right, on=list(IDENTITY), how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        missing_predecessor = int(merged["_merge"].eq("left_only").sum())
        extra_predecessor = int(merged["_merge"].eq("right_only").sum())
        raise StrictPredecessorMetaOOFError(
            "predecessor features and source ledger identities differ "
            f"(missing_predecessor={missing_predecessor}, extra_predecessor={extra_predecessor})"
        )
    merged = merged.drop(columns="_merge")
    decision = _utc(merged, "decision_ts")
    fit_end = _utc(merged, "predecessor_oof_fit_end_ts")
    generated = _utc(merged, "predecessor_oof_generated_ts")
    available = _utc(merged, "predecessor_oof_available_ts")
    if not merged["predecessor_same_side_strict_oof"].fillna(False).astype(bool).all():
        raise StrictPredecessorMetaOOFError("predecessor rows must prove same-side strict OOF")
    if not fit_end.lt(decision).all() or not generated.le(decision).all() or not available.le(decision).all():
        raise StrictPredecessorMetaOOFError("predecessor timestamp lineage is not decision-time causal")
    if not fit_end.le(generated).all() or not generated.le(available).all():
        raise StrictPredecessorMetaOOFError("predecessor timestamp order is inconsistent")
    values = merged.loc[:, list(PREDECESSOR_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise StrictPredecessorMetaOOFError("predecessor emitted features must be finite")
    return merged.sort_values(
        ["transport", "meta_partition", "decision_ts", "candidate_id", "side_name", "fold_id"],
        kind="stable",
    ).reset_index(drop=True)


def materialize_strict_predecessor_meta_oof(
    ledger: pd.DataFrame, *, config: StrictPredecessorMetaOOFConfig,
    source_ledger_sha256: str | None = None,
) -> StrictPredecessorMetaOOFResult:
    """Build six nested predecessor-meta component OOF fields and a new ledger."""

    work = _normalise_ledger(ledger, config=config)
    features, audit = _build_features(work, config=config)
    joined = join_strict_predecessor_oof_features(work, features)
    if _identity_digest(work) != _identity_digest(joined):
        raise AssertionError("predecessor exact-identity join changed candidate population")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "feature_columns": list(PREDECESSOR_FEATURE_COLUMNS),
        "source_feature_map": {
            name: str(dict(config.source_feature_map)[name]) for name in PREDECESSOR_SEMANTICS
        },
        "contract": {
            "predecessor_training": "same transport and side; inner_oof rows only; label_available_ts < first scored decision in refit block",
            "same_row_in_sample": "rejected by exact candidate identity and strict label availability",
            "output": "exactly six predecessor ridge component contributions; intercept is not emitted",
            "lineage": "fit_end < decision_ts; generated/available <= decision_ts",
            "raw_leaf_ids": "forbidden in the predecessor source feature contract",
            "joined_ledger": "exact full candidate identity outer join; source is never mutated",
        },
        "config": asdict(config),
        "candidate_identity_sha256": _identity_digest(work),
        "source_ledger_sha256": source_ledger_sha256,
        "row_count": int(len(work)),
    }
    return StrictPredecessorMetaOOFResult(features, joined, audit, manifest)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictPredecessorMetaOOFError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise StrictPredecessorMetaOOFError(f"JSON artifact must be an object: {path}")
    return value


def load_immutable_meta_ledger_for_predecessor(
    root: str | os.PathLike[str],
) -> tuple[pd.DataFrame, Mapping[str, Any], str]:
    """Load only a completed base-to-meta immutable ledger for the predecessor."""

    directory = Path(root)
    manifest_path = directory / "meta_ledger_manifest.json"
    ledger_path = directory / "base_to_meta_reasoning_ledger.parquet"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != SOURCE_LEDGER_STATUS:
        raise StrictPredecessorMetaOOFError("source root is not a completed immutable base-to-meta ledger")
    if not ledger_path.is_file():
        raise StrictPredecessorMetaOOFError("source immutable ledger table is missing")
    recorded = manifest.get("sha256", {}).get(ledger_path.name) if isinstance(manifest.get("sha256"), Mapping) else None
    actual = _sha256(ledger_path)
    if not isinstance(recorded, str) or recorded != actual:
        raise StrictPredecessorMetaOOFError("source immutable ledger hash does not match its manifest")
    return pd.read_parquet(ledger_path), manifest, actual


def write_immutable_strict_predecessor_meta_oof(
    result: StrictPredecessorMetaOOFResult, output_dir: str | os.PathLike[str],
) -> Path:
    """Atomically write a non-overwritable predecessor hand-off root."""

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite strict predecessor OOF root: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        narrow_columns = [*IDENTITY, *PREDECESSOR_FEATURE_COLUMNS, *LINEAGE_COLUMNS]
        narrow = result.features.loc[:, narrow_columns]
        feature_path = temporary / "predecessor_oof_features.parquet"
        ledger_path = temporary / "base_to_meta_reasoning_ledger_predecessor_oof.parquet"
        audit_path = temporary / "predecessor_oof_fit_audit.parquet"
        feature_list_path = temporary / "predecessor_oof_feature_columns.json"
        narrow.to_parquet(feature_path, index=False, compression="zstd")
        result.ledger.to_parquet(ledger_path, index=False, compression="zstd")
        result.fit_audit.to_parquet(audit_path, index=False, compression="zstd")
        feature_list_path.write_text(
            json.dumps(list(PREDECESSOR_FEATURE_COLUMNS), indent=2) + "\n", encoding="utf-8"
        )
        manifest = dict(result.manifest)
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["immutable_output"] = True
        manifest["files"] = {
            "features": feature_path.name,
            "joined_ledger": ledger_path.name,
            "fit_audit": audit_path.name,
            "feature_columns": feature_list_path.name,
        }
        manifest["sha256"] = {
            path.name: _sha256(path)
            for path in (feature_path, ledger_path, audit_path, feature_list_path)
        }
        (temporary / "predecessor_oof_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_immutable_strict_predecessor_meta_oof(
    root: str | os.PathLike[str],
) -> StrictPredecessorMetaOOFArtifact:
    """Verify and load the only predecessor artifact accepted by the S2 CLI."""

    directory = Path(root)
    manifest = _read_json(directory / "predecessor_oof_manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise StrictPredecessorMetaOOFError("predecessor root is not a completed strict nested OOF artifact")
    if list(manifest.get("feature_columns", ())) != list(PREDECESSOR_FEATURE_COLUMNS):
        raise StrictPredecessorMetaOOFError("predecessor root does not expose exactly the mandated six features")
    files = manifest.get("files")
    hashes = manifest.get("sha256")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise StrictPredecessorMetaOOFError("predecessor root lacks immutable file hashes")
    required_files = ("features", "joined_ledger", "fit_audit", "feature_columns")
    if any(not isinstance(files.get(name), str) for name in required_files):
        raise StrictPredecessorMetaOOFError("predecessor root has an incomplete file contract")
    paths = {name: directory / str(files[name]) for name in required_files}
    for name, path in paths.items():
        if not path.is_file() or hashes.get(path.name) != _sha256(path):
            raise StrictPredecessorMetaOOFError(f"predecessor immutable file hash mismatch: {name}")
    names = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
    if names != list(PREDECESSOR_FEATURE_COLUMNS):
        raise StrictPredecessorMetaOOFError("predecessor feature-list artifact differs from the mandated surface")
    features = pd.read_parquet(paths["features"])
    ledger = pd.read_parquet(paths["joined_ledger"])
    # Re-running the exact join validates both candidate identity and every
    # timestamp lineage before S2 is allowed to touch this hand-off.
    source_columns = [name for name in ledger.columns if name not in set(features.columns).difference(IDENTITY)]
    source = ledger.loc[:, source_columns]
    verified = join_strict_predecessor_oof_features(source, features)
    if _identity_digest(verified) != str(manifest.get("candidate_identity_sha256", "")):
        raise StrictPredecessorMetaOOFError("predecessor joined ledger candidate identity hash mismatch")
    return StrictPredecessorMetaOOFArtifact(
        root=directory,
        features=features,
        ledger=verified,
        feature_columns=PREDECESSOR_FEATURE_COLUMNS,
        manifest=manifest,
        ledger_path=paths["joined_ledger"],
    )


__all__ = [
    "IDENTITY", "LINEAGE_COLUMNS", "PREDECESSOR_FEATURE_COLUMNS", "PREDECESSOR_SEMANTICS",
    "SCHEMA", "STATUS", "StrictPredecessorMetaOOFArtifact", "StrictPredecessorMetaOOFConfig",
    "StrictPredecessorMetaOOFError", "StrictPredecessorMetaOOFResult",
    "join_strict_predecessor_oof_features", "load_immutable_meta_ledger_for_predecessor",
    "load_immutable_strict_predecessor_meta_oof", "materialize_strict_predecessor_meta_oof",
    "write_immutable_strict_predecessor_meta_oof",
]
