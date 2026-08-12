"""Strict prerequisites for the causal H1--H5 family-health materialiser.

There are two deliberately separate, immutable prerequisite artifacts:

``strict_fold_causal_context``
    A full, outcome-free causal regime/context sidecar for the fixed July 2023
    through November 2024 research interval.  It uses the existing OOF market
    regime materialiser unchanged, including its train-only frozen state fits,
    backward candidate join, continuous context and relationship-break output.

``strict_predecessor_family_selection``
    A compact selection of token-free rule families.  The selector observes
    only completed *inner-OOF* predecessor rows whose labels were available
    strictly before the declared cutoff.  It never reads the labels from the
    scored/evaluation partition and deliberately ranks families by stable
    support and contribution mass rather than realised economics.  The latter
    remains available only to H1--H3's strictly prequential state updates.

The separation matters.  A rich context sidecar can safely be made available
through the untouched November 2024 OOS boundary, whereas a family selection
has a specific predecessor cutoff and may only be applied to later rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .causal_leaf_health import CausalLeafHealthError
from .causal_leaf_health_artifacts import (
    StrictOOFFamilyInputs,
)
from .leaf_family_contributions import (
    LeafFamilyContributionConfig,
    materialize_leaf_family_contributions,
)
from .regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity
from .strict_event_store import (
    StrictEventStore,
    iter_predecessor_selection_pairs,
    load_strict_event_store,
)

try:  # Used only for the disk-spilling predecessor selector.
    import duckdb
except ImportError:  # pragma: no cover - production and tests install DuckDB
    duckdb = None


STRICT_CONTEXT_SCHEMA = "strict_fold_causal_context_v1"
STRICT_CONTEXT_STATUS = "MATERIALIZED_STRICT_FOLD_CAUSAL_CONTEXT"
FAMILY_SELECTION_SCHEMA = "strict_predecessor_family_selection_v1"
FAMILY_SELECTION_STATUS = "FROZEN_STRICT_PREDECESSOR_FAMILY_SELECTION"
FAMILY_SELECTION_ROOT_STATUS = "MATERIALIZED_STRICT_PREDECESSOR_FAMILY_SELECTIONS"

STRICT_CONTEXT_START_UTC = pd.Timestamp("2023-07-01T00:00:00Z")
# November is included.  This is an exclusive end boundary so the context
# needed for a later untouched November replay is already materialised.
STRICT_CONTEXT_END_EXCLUSIVE_UTC = pd.Timestamp("2024-12-01T00:00:00Z")

FAMILY_KEY_COLUMNS: tuple[str, ...] = (
    "feature_contract_sha256",
    "side_name",
    "head_name",
    "rule_signature",
    "contribution_direction",
)
FAMILY_SELECTION_KINDS: tuple[str, ...] = ("context", "covariance", "relationship")

# A fixed compact surface for H3/H4/H5.  These are all decision-time fields
# produced by the existing causal state/context materialiser.  It is not a
# supervised selection and its contract deliberately includes one causal
# relationship-break coordinate for H5.
DEFAULT_HEALTH_CONTEXT_COLUMNS: tuple[str, ...] = (
    "regime_entropy",
    "regime_top2_margin",
    "state_age_hours",
    "state_switch_probability",
    "market_regime__ood_distance_percentile",
    "transition_state_p__active",
    "continuous_regime__trend_quality__z_90d",
    "continuous_regime__volatility__z_90d",
    "continuous_regime__breadth__z_90d",
    "continuous_regime__relationship_break__volatility_liquidity__residual_abs_30d",
)


class CausalLeafHealthPrerequisiteError(CausalLeafHealthError):
    """Raised when a context or predecessor-selection boundary is unsafe."""


@dataclass(frozen=True)
class PredecessorFamilySelectionConfig:
    """Support-only deterministic limits for H3/H4/H5 family selection.

    The selector intentionally does not rank on realised PnL, accuracy, or a
    target.  The label availability boundary still matters: it proves the
    selected family population was known before the evaluation segment, while
    keeping selection from consuming any evaluation outcome.
    """

    min_rows: int = 24
    min_independent_timestamps: int = 12
    min_trading_days: int = 3
    min_symbols: int = 3
    max_context_families_per_scope: int = 12
    max_covariance_families_per_scope: int = 8
    max_relationship_families_per_scope: int = 12
    allowed_meta_partition: str = "inner_oof"

    def validate(self) -> None:
        if any(
            int(value) <= 0
            for value in (
                self.min_rows,
                self.min_independent_timestamps,
                self.min_trading_days,
                self.min_symbols,
                self.max_context_families_per_scope,
                self.max_covariance_families_per_scope,
                self.max_relationship_families_per_scope,
            )
        ):
            raise CausalLeafHealthPrerequisiteError("family-selection supports and maxima must be positive")
        if str(self.allowed_meta_partition) != "inner_oof":
            raise CausalLeafHealthPrerequisiteError(
                "family selection is deliberately restricted to predecessor inner_oof rows"
            )

    def max_for_kind(self, kind: str) -> int:
        if kind == "context":
            return int(self.max_context_families_per_scope)
        if kind == "covariance":
            return int(self.max_covariance_families_per_scope)
        if kind == "relationship":
            return int(self.max_relationship_families_per_scope)
        raise CausalLeafHealthPrerequisiteError(f"unknown family selection kind: {kind}")


@dataclass(frozen=True)
class FrozenFamilySelection:
    """One verified selection manifest suitable for a later H1--H5 run."""

    kind: str
    cutoff_utc: pd.Timestamp
    selected_families: frozenset[tuple[str, str, str, str, str]]
    payload: dict[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class _StreamingFamilySelectionSource:
    """The tiny provenance surface needed after a streaming selection pass.

    Unlike :class:`StrictOOFFamilyInputs`, this deliberately has no candidate
    or contribution dataframes.  Keeping those tables out of the selector is
    what prevents a 7M-row strict root from becoming a many-GB pandas merge.
    """

    strict_roots: tuple[str, ...]
    strict_root_manifest_sha256: dict[str, str]
    event_store_root: str | None = None
    event_store_manifest_sha256: str | None = None


_STRICT_SHARD_REQUIRED = {
    "candidate_id", "decision_ts", "label_available_ts", "side_name", "fold_id",
    "feature_generation_ts", "feature_contract_sha256", "base_expected_bps", "asset",
    "r3_class",
}
_HEAD_PREDICTION_REQUIRED = {
    "candidate_id", "__ts__", "side_name", "head_name", "fold_id", "base_prediction",
}
_HEAD_LABEL_REQUIRED = {
    "candidate_id", "__ts__", "side_name", "head_name", "fold_id", "label__r3_class",
    "label__net_bps", "label__label_available_ts",
}
_HEAD_JOIN_KEYS = ["candidate_id", "__ts__", "side_name", "head_name", "fold_id"]
_SHARD_JOIN_KEYS = ["candidate_id", "__ts__", "side_name", "fold_id"]


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise CausalLeafHealthPrerequisiteError(f"{name} must be a finite UTC timestamp")
    return pd.Timestamp(parsed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CausalLeafHealthPrerequisiteError(f"invalid prerequisite JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CausalLeafHealthPrerequisiteError(f"prerequisite JSON must be an object: {path}")
    return value


def strict_candidate_population(
    strict_roots: Sequence[str | Path],
    *,
    start_utc: str | pd.Timestamp = STRICT_CONTEXT_START_UTC,
    end_exclusive_utc: str | pd.Timestamp = STRICT_CONTEXT_END_EXCLUSIVE_UTC,
) -> pd.DataFrame:
    """Build the exact candidate identity needed by a strict context sidecar.

    This reader uses strict prediction shards only.  It intentionally drops
    labels, base scores, economics and head identities before calling the
    reusable outcome-free regime materialiser.
    """

    start = _utc(start_utc, name="context start")
    end = _utc(end_exclusive_utc, name="context end")
    if start >= end:
        raise CausalLeafHealthPrerequisiteError("context start must be before context end")
    # Import locally so this prerequisite module remains free of script import
    # side effects when only predecessor selections are materialised.
    from .causal_leaf_health_artifacts import _strict_candidate_shards  # noqa: PLC0415

    roots = [Path(item) for item in strict_roots]
    if not roots:
        raise CausalLeafHealthPrerequisiteError("at least one strict root is required for causal context")
    pieces: list[pd.DataFrame] = []
    for root in roots:
        shards = _strict_candidate_shards(root)
        required = {"candidate_id", "decision_ts", "asset", "side_name"}
        missing = sorted(required.difference(shards.columns))
        if missing:
            raise CausalLeafHealthPrerequisiteError(
                f"strict prediction shards lack candidate context identity: {missing}"
            )
        local = shards.loc[:, ["candidate_id", "decision_ts", "asset", "side_name"]].copy()
        local = local.rename(columns={"decision_ts": "__ts__", "asset": "__symbol__"})
        local["__ts__"] = pd.to_datetime(local["__ts__"], utc=True, errors="coerce")
        if local["__ts__"].isna().any():
            raise CausalLeafHealthPrerequisiteError("strict prediction shard candidate times are invalid")
        pieces.append(local)
    result = pd.concat(pieces, ignore_index=True)
    result = result.loc[result["__ts__"].ge(start) & result["__ts__"].lt(end)].copy()
    if result.empty:
        raise CausalLeafHealthPrerequisiteError("strict roots have no candidates inside the required causal-context window")
    # Base semantic heads may share the same candidate source.  Collapse only
    # exact repeats; any disagreement in its immutable identity is a fail-close
    # condition rather than an arbitrary candidate join.
    result["candidate_id"] = result["candidate_id"].astype("string")
    result["__symbol__"] = result["__symbol__"].astype("string")
    result["side_name"] = result["side_name"].astype("string").str.lower()
    identity = list(IDENTITY_COLUMNS)
    result = result.drop_duplicates(identity, keep="first")
    duplicate_id = result["candidate_id"].duplicated(keep=False)
    if duplicate_id.any():
        sample = result.loc[duplicate_id, identity].head(4).to_dict("records")
        raise CausalLeafHealthPrerequisiteError(
            "strict candidate population reuses a candidate_id across identities; "
            f"cannot perform an exact context join: {sample}"
        )
    result = validate_candidate_identity(result.loc[:, identity]).sort_values("__ts__", kind="stable").reset_index(drop=True)
    return result


def _validate_full_context_sidecar(
    sidecar: Path,
    expected_candidates: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    health_context_columns: Sequence[str],
) -> dict[str, Any]:
    timeline_path = sidecar / "hourly_oof_market_regimes.parquet"
    candidate_path = sidecar / "candidate_oof_market_regimes.parquet"
    if not timeline_path.is_file() or not candidate_path.is_file():
        raise CausalLeafHealthPrerequisiteError("causal regime materialiser did not emit both hourly and candidate sidecars")
    timeline = pd.read_parquet(timeline_path, columns=["source_utc"])
    timeline["source_utc"] = pd.to_datetime(timeline["source_utc"], utc=True, errors="coerce")
    if timeline["source_utc"].isna().any() or timeline["source_utc"].duplicated().any():
        raise CausalLeafHealthPrerequisiteError("full causal context hourly timeline has invalid/duplicate timestamps")
    if timeline["source_utc"].lt(start).any() or timeline["source_utc"].ge(end).any():
        raise CausalLeafHealthPrerequisiteError("full causal context hourly timeline escaped its declared window")
    required_months = pd.date_range(
        start=start.normalize(), end=(end - pd.Timedelta(days=1)).normalize(), freq="MS", tz="UTC",
    ).strftime("%Y-%m").tolist()
    observed_months = set(timeline["source_utc"].dt.strftime("%Y-%m"))
    missing_months = [item for item in required_months if item not in observed_months]
    if missing_months:
        raise CausalLeafHealthPrerequisiteError(
            f"full causal context does not cover every requested month: {missing_months}"
        )
    candidate = pd.read_parquet(candidate_path)
    candidate = validate_candidate_identity(candidate)
    expected = validate_candidate_identity(expected_candidates)
    merge = expected.merge(
        candidate.loc[:, list(IDENTITY_COLUMNS)], on=list(IDENTITY_COLUMNS), how="outer", indicator=True,
    )
    if not merge["_merge"].eq("both").all():
        raise CausalLeafHealthPrerequisiteError("causal context candidate sidecar is not an exact strict candidate population")
    required = {"regime_available_utc", *health_context_columns}
    missing = sorted(required.difference(candidate.columns))
    if missing:
        raise CausalLeafHealthPrerequisiteError(
            f"causal context lacks the fixed H3/H4/H5 context contract: {missing}"
        )
    available = pd.to_datetime(candidate["regime_available_utc"], utc=True, errors="coerce")
    if available.isna().any() or available.gt(candidate["__ts__"]).any():
        raise CausalLeafHealthPrerequisiteError("causal context candidate availability is missing or looks ahead")
    values = candidate.loc[:, list(health_context_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise CausalLeafHealthPrerequisiteError("fixed H3/H4/H5 causal context fields must be finite for every strict candidate")
    return {
        "timeline_rows": int(len(timeline)),
        "candidate_rows": int(len(candidate)),
        "month_coverage": required_months,
        "health_context_columns": list(health_context_columns),
    }


def materialize_strict_fold_causal_context(
    strict_roots: Sequence[str | Path],
    output_dir: str | Path,
    *,
    panel_path: str | Path,
    start_utc: str | pd.Timestamp = STRICT_CONTEXT_START_UTC,
    end_exclusive_utc: str | pd.Timestamp = STRICT_CONTEXT_END_EXCLUSIVE_UTC,
    frequency: str = "quarter",
    purge_hours: int = 12,
    max_features_per_view: int = 20,
    max_lag_hours: int = 2,
    seed: int = 20260803,
    health_context_columns: Sequence[str] = DEFAULT_HEALTH_CONTEXT_COLUMNS,
) -> Path:
    """Materialise one immutable July-2023--Nov-2024 causal context sidecar.

    The wrapped materialiser owns all state discovery and continuous-context
    computation.  This boundary only derives an outcome-free strict candidate
    identity, enforces complete month/candidate coverage, and records the
    fixed context fields used by the H1--H5 health contract.
    """

    start = _utc(start_utc, name="context start")
    end = _utc(end_exclusive_utc, name="context end")
    if start != STRICT_CONTEXT_START_UTC or end != STRICT_CONTEXT_END_EXCLUSIVE_UTC:
        raise CausalLeafHealthPrerequisiteError(
            "strict H1--H5 causal context must cover exactly 2023-07-01 through 2024-12-01"
        )
    if len(tuple(health_context_columns)) == 0 or len(tuple(health_context_columns)) > 10:
        raise CausalLeafHealthPrerequisiteError("the fixed H3/H4/H5 context contract must contain 1--10 fields")
    target = Path(output_dir)
    panel = Path(panel_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite strict causal context sidecar: {target}")
    if not panel.is_file():
        raise FileNotFoundError(panel)
    candidates = strict_candidate_population(strict_roots, start_utc=start, end_exclusive_utc=end)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        candidate_path = temporary / "strict_candidates.parquet"
        candidates.to_parquet(candidate_path, index=False, compression="zstd")
        # Reuse the existing causal regime/context generator rather than
        # shadowing any feature calculations here.
        from scripts.materialize_oof_market_regime_systems import materialize as materialize_oof_regimes  # noqa: PLC0415

        sidecar = temporary / "sidecar"
        materialize_oof_regimes(
            panel_path=panel,
            output_dir=sidecar,
            evaluation_start=start.isoformat(),
            evaluation_end=end.isoformat(),
            candidate_path=candidate_path,
            frequency=frequency,
            purge_hours=int(purge_hours),
            max_features_per_view=int(max_features_per_view),
            max_lag_hours=int(max_lag_hours),
            seed=int(seed),
            primary_state_count=5,
            primary_merge_low_support_state=False,
            systems=None,
        )
        coverage = _validate_full_context_sidecar(
            sidecar, candidates, start=start, end=end,
            health_context_columns=tuple(health_context_columns),
        )
        roots = [Path(item) for item in strict_roots]
        base_manifest = sidecar / "manifest.json"
        context_manifest = {
            "schema": STRICT_CONTEXT_SCHEMA,
            "status": STRICT_CONTEXT_STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "window": {
                "start_utc": start,
                "end_exclusive_utc": end,
                "includes_untouched_november_2024_context": True,
            },
            "inputs": {
                "hourly_multiview_panel": {"path": str(panel.resolve()), "sha256": _sha256(panel)},
                "strict_roots": {
                    str(root): _sha256(root / "strict_oof_reasoning_manifest.json")
                    for root in roots
                },
                "wrapped_oof_regime_manifest_sha256": _sha256(base_manifest),
            },
            "contract": {
                "source": "existing materialize_oof_market_regime_systems; no duplicate state/context feature logic",
                "outcomes": "strict candidate identity only; labels, base scores and economics are excluded before causal context materialisation",
                "fit": "existing train-only chronological frozen state fits; continuous context and relationship breaks are prequential",
                "candidate_join": "existing backward as-of only join",
                "health_default_context_columns": list(health_context_columns),
                "selection": "no family, feature, model or policy selection uses November 2024 outcomes",
            },
            "coverage": coverage,
        }
        manifest_path = sidecar / "strict_fold_context_manifest.json"
        manifest_path.write_text(
            json.dumps(_safe(context_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (sidecar / "strict_fold_context_manifest.sha256").write_text(
            _sha256(manifest_path) + "  strict_fold_context_manifest.json\n", encoding="utf-8"
        )
        os.replace(sidecar, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _require_duckdb() -> None:
    if duckdb is None:
        raise CausalLeafHealthPrerequisiteError(
            "duckdb is required for bounded strict predecessor family selection"
        )


def _utc_series(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise CausalLeafHealthPrerequisiteError(f"{name} must contain finite UTC timestamps")
    return parsed


def _normalise_strict_shard(
    path: Path,
    *,
    side: str,
    transport: str,
    partition: str,
) -> pd.DataFrame:
    """Read one strict shard only; never concatenate a whole root in memory."""

    if not path.is_file():
        raise CausalLeafHealthPrerequisiteError(f"strict health source is missing prediction shard: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(_STRICT_SHARD_REQUIRED.difference(frame.columns))
    if missing:
        raise CausalLeafHealthPrerequisiteError(f"strict prediction shard lacks H1 lineage: {missing}")
    frame = frame.copy()
    frame["decision_ts"] = _utc_series(frame["decision_ts"], name="strict shard decision_ts")
    frame["label_available_ts"] = _utc_series(frame["label_available_ts"], name="strict shard label_available_ts")
    frame["feature_generation_ts"] = _utc_series(frame["feature_generation_ts"], name="strict shard feature_generation_ts")
    if not frame["side_name"].astype(str).str.lower().eq(side).all():
        raise CausalLeafHealthPrerequisiteError("strict prediction shards cross their side directory")
    if not frame["feature_generation_ts"].le(frame["decision_ts"]).all():
        raise CausalLeafHealthPrerequisiteError("strict health shard feature time is after decision time")
    if not frame["label_available_ts"].ge(frame["decision_ts"]).all():
        raise CausalLeafHealthPrerequisiteError("strict health shard label resolves before decision time")
    frame["transport"] = transport
    frame["meta_partition"] = partition
    return frame


def _strict_root_layout(root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest = _json(root / "strict_oof_reasoning_manifest.json")
    if manifest.get("status") != "STRICT_OOF_BASE_REASONING_MATERIALIZED":
        raise CausalLeafHealthPrerequisiteError(f"strict health root is not complete: {root}")
    transports = tuple(sorted({str(value) for value in manifest.get("transports", [])}))
    if not transports:
        raise CausalLeafHealthPrerequisiteError("strict health root has no transports")
    return manifest, transports


def _validate_root_shard_identity_bounded(root: Path, transports: Sequence[str], state_db: Any) -> None:
    """Validate all root shards with an on-disk uniqueness table.

    The old collector concatenated every shard just to prove this invariant.
    This uses a transient DuckDB table, so duplicate detection remains exact
    while peak Python memory is one shard.
    """

    state_db.execute(
        """
        CREATE TEMP TABLE root_shard_identity (
            candidate_id VARCHAR, decision_ts TIMESTAMPTZ, side_name VARCHAR,
            fold_id VARCHAR, transport VARCHAR, meta_partition VARCHAR,
            PRIMARY KEY (candidate_id, decision_ts, side_name, fold_id, transport, meta_partition)
        )
        """
    )
    try:
        for transport in transports:
            for side in ("long", "short"):
                directory = root / "base_prediction_shards" / transport / side
                for filename, partition in (
                    ("strict_oof_predictions.parquet", "inner_oof"),
                    ("outer_predictions.parquet", "outer_test"),
                ):
                    shard = _normalise_strict_shard(
                        directory / filename, side=side, transport=transport, partition=partition,
                    )
                    identity = shard.loc[:, [
                        "candidate_id", "decision_ts", "side_name", "fold_id", "transport", "meta_partition",
                    ]]
                    try:
                        state_db.register("__root_shard_batch", identity)
                        state_db.execute("INSERT INTO root_shard_identity SELECT * FROM __root_shard_batch")
                    except Exception as exc:
                        raise CausalLeafHealthPrerequisiteError(
                            "strict prediction roots duplicate candidate identities"
                        ) from exc
                    finally:
                        try:
                            state_db.unregister("__root_shard_batch")
                        except Exception:
                            pass
    finally:
        state_db.execute("DROP TABLE root_shard_identity")


def _artifact_scope(root: Path, artifact: Path, transports: Sequence[str]) -> tuple[str, str, str, str, str, int]:
    """Read and cross-check immutable per-head scope against its directory."""

    manifest = _json(artifact / "base_reasoning_manifest.json")
    if manifest.get("status") != "MATERIALIZED_STRICT_OOF":
        raise CausalLeafHealthPrerequisiteError(f"per-head strict artifact is incomplete: {artifact}")
    provenance = manifest.get("provenance", {})
    head = str(manifest.get("head_name", ""))
    side = str(manifest.get("side_name", "")).lower()
    fold = str(manifest.get("fold_id", ""))
    contract = str(provenance.get("feature_contract_sha256", ""))
    class_index = provenance.get("class_index")
    if not head or side not in {"long", "short"} or not fold or not contract or class_index is None:
        raise CausalLeafHealthPrerequisiteError(f"per-head strict manifest lacks scope lineage: {artifact}")
    try:
        relative = artifact.relative_to(root / "strict_oof_base_reasoning").parts
    except ValueError as exc:
        raise CausalLeafHealthPrerequisiteError(f"per-head artifact escaped strict root: {artifact}") from exc
    if len(relative) != 5 or relative[1] != "folds":
        raise CausalLeafHealthPrerequisiteError(f"per-head artifact has invalid strict layout: {artifact}")
    transport, _, directory_side, directory_fold, directory_head = relative
    # Production fold IDs include their transport prefix to be globally unique
    # (``transport_a_..._inner_00``), while the immutable directory stores the
    # compact fold suffix.  Tests and older roots may use the suffix directly.
    directory_fold_matches = fold == directory_fold or fold == f"{transport}_{directory_fold}"
    if (
        transport not in set(transports)
        or directory_side.lower() != side
        or not directory_fold_matches
        or directory_head != head
    ):
        raise CausalLeafHealthPrerequisiteError("per-head strict artifact directory disagrees with immutable manifest scope")
    try:
        return transport, side, head, fold, contract, int(class_index)
    except (TypeError, ValueError) as exc:
        raise CausalLeafHealthPrerequisiteError("per-head strict manifest class_index is not integer-like") from exc


def _artifact_candidate_provenance_bounded(
    root: Path,
    artifact: Path,
    *,
    transport: str,
    side: str,
    head: str,
    fold: str,
    contract: str,
    class_index: int,
) -> pd.DataFrame:
    """Reproduce the old per-head lineage join, scoped to one artifact."""

    prediction_path = artifact / "base_reasoning_predictions.parquet"
    label_path = artifact / "base_reasoning_labels.parquet"
    if not prediction_path.is_file() or not label_path.is_file():
        raise CausalLeafHealthPrerequisiteError(f"per-head strict artifact lacks predictions or labels: {artifact}")
    prediction = pd.read_parquet(prediction_path)
    labels = pd.read_parquet(label_path)
    if missing := sorted(_HEAD_PREDICTION_REQUIRED.difference(prediction.columns)):
        raise CausalLeafHealthPrerequisiteError(f"per-head prediction table lacks {missing}")
    if missing := sorted(_HEAD_LABEL_REQUIRED.difference(labels.columns)):
        raise CausalLeafHealthPrerequisiteError(f"per-head label table lacks {missing}")
    merged = prediction.merge(labels, on=_HEAD_JOIN_KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(prediction) or len(merged) != len(labels):
        raise CausalLeafHealthPrerequisiteError("per-head strict predictions and labels do not have identical identities")
    merged["__ts__"] = _utc_series(merged["__ts__"], name="per-head decision timestamp")
    merged["label__label_available_ts"] = _utc_series(
        merged["label__label_available_ts"], name="per-head label availability",
    )
    if (
        not merged["head_name"].astype(str).eq(head).all()
        or not merged["side_name"].astype(str).str.lower().eq(side).all()
        or not merged["fold_id"].astype(str).eq(fold).all()
    ):
        raise CausalLeafHealthPrerequisiteError("per-head strict rows cross manifest scope")
    directory = root / "base_prediction_shards" / transport / side
    shard_rows = pd.concat([
        _normalise_strict_shard(
            directory / "strict_oof_predictions.parquet", side=side, transport=transport, partition="inner_oof",
        ),
        _normalise_strict_shard(
            directory / "outer_predictions.parquet", side=side, transport=transport, partition="outer_test",
        ),
    ], ignore_index=True)
    candidate = merged.merge(
        shard_rows,
        left_on=["candidate_id", "__ts__", "side_name", "fold_id"],
        right_on=["candidate_id", "decision_ts", "side_name", "fold_id"],
        how="left", validate="one_to_one", indicator=True, suffixes=("_head", ""),
    )
    if not candidate["_merge"].eq("both").all():
        raise CausalLeafHealthPrerequisiteError("per-head artifact cannot prove a matching strict prediction shard")
    if not candidate["feature_contract_sha256"].astype(str).eq(contract).all():
        raise CausalLeafHealthPrerequisiteError("per-head feature contract differs from strict prediction shard")
    if not candidate["label__label_available_ts"].astype("int64").eq(candidate["label_available_ts"].astype("int64")).all():
        raise CausalLeafHealthPrerequisiteError("per-head label availability differs from strict prediction shard")
    # These are intentionally parsed exactly as in the full health collector.
    # They are discarded below: realised outcomes never enter family selection.
    candidate["semantic_label"] = candidate["label__r3_class"].astype(int).eq(class_index).astype(float)
    candidate["head_prediction"] = pd.to_numeric(candidate["base_prediction"], errors="coerce")
    candidate["net_bps"] = pd.to_numeric(candidate["label__net_bps"], errors="coerce")
    return candidate.loc[:, [
        "candidate_id", "decision_ts", "feature_generation_ts", "label_available_ts",
        "side_name", "head_name", "fold_id", "transport", "meta_partition",
        "feature_contract_sha256", "asset",
    ]].rename(columns={"decision_ts": "__ts__"}).copy()


def _create_streaming_selection_tables(state_db: Any) -> None:
    key = ", ".join(FAMILY_KEY_COLUMNS)
    state_db.execute(
        f"""
        CREATE TABLE family_aggregate (
            feature_contract_sha256 VARCHAR, side_name VARCHAR, head_name VARCHAR,
            rule_signature VARCHAR, contribution_direction VARCHAR,
            predecessor_rows BIGINT, contribution_abs_mass DOUBLE,
            latest_label_available_utc TIMESTAMPTZ,
            PRIMARY KEY ({key})
        )
        """
    )
    for suffix, column_type in (("timestamps", "TIMESTAMPTZ"), ("days", "VARCHAR"), ("symbols", "VARCHAR")):
        state_db.execute(
            f"""
            CREATE TABLE family_{suffix} (
                feature_contract_sha256 VARCHAR, side_name VARCHAR, head_name VARCHAR,
                rule_signature VARCHAR, contribution_direction VARCHAR, observed_value {column_type},
                PRIMARY KEY ({key}, observed_value)
            )
            """
        )


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _aggregate_one_artifact_bounded(
    state_db: Any,
    *,
    contribution_path: Path,
    provenance_path: Path,
    contract: str,
    cutoff: pd.Timestamp,
) -> tuple[int, int]:
    """Spill one artifact's token-free rows into compact global aggregates."""

    contribution = _sql_literal(contribution_path)
    provenance = _sql_literal(provenance_path)
    contract_sql = _sql_literal(contract)
    cutoff_sql = _sql_literal(cutoff.isoformat())
    join = " AND ".join(f"c.{column} = p.{column}" for column in _HEAD_JOIN_KEYS)
    # Family output has a row only for non-zero local contributions.  Every
    # such row must be traceable to the same per-head candidate artifact,
    # independently of whether it is predecessor eligible.
    unmatched = state_db.execute(
        f"""
        SELECT count(*) FROM read_parquet({contribution}) c
        LEFT JOIN read_parquet({provenance}) p ON {join}
        WHERE p.candidate_id IS NULL
        """
    ).fetchone()[0]
    if int(unmatched):
        raise CausalLeafHealthPrerequisiteError(
            "strict family contribution cannot prove candidate/head provenance"
        )
    predecessor_count = state_db.execute(
        f"""
        SELECT count(*) FROM read_parquet({provenance})
        WHERE meta_partition = 'inner_oof'
          AND label_available_ts < CAST({cutoff_sql} AS TIMESTAMPTZ)
        """
    ).fetchone()[0]
    condition = (
        "p.meta_partition = 'inner_oof' "
        f"AND p.label_available_ts < CAST({cutoff_sql} AS TIMESTAMPTZ)"
    )
    group = "c.side_name, c.head_name, c.rule_signature, c.contribution_direction"
    source = (
        f"FROM read_parquet({contribution}) c JOIN read_parquet({provenance}) p ON {join} "
        f"WHERE {condition}"
    )
    state_db.execute(
        f"""
        INSERT INTO family_aggregate
        SELECT {contract_sql}, c.side_name, c.head_name, c.rule_signature, c.contribution_direction,
               count(*), sum(abs(c.family_ensemble_tree_contribution)), max(p.label_available_ts)
        {source}
        GROUP BY {group}
        ON CONFLICT ({', '.join(FAMILY_KEY_COLUMNS)}) DO UPDATE SET
            predecessor_rows = family_aggregate.predecessor_rows + excluded.predecessor_rows,
            contribution_abs_mass = family_aggregate.contribution_abs_mass + excluded.contribution_abs_mass,
            latest_label_available_utc = greatest(
                family_aggregate.latest_label_available_utc, excluded.latest_label_available_utc
            )
        """
    )
    for suffix, value in (
        ("timestamps", "c.__ts__"),
        ("days", "strftime(c.__ts__, '%Y-%m-%d')"),
        ("symbols", "p.asset"),
    ):
        non_null = "" if suffix != "symbols" else " AND p.asset IS NOT NULL"
        state_db.execute(
            f"""
            INSERT INTO family_{suffix}
            SELECT DISTINCT {contract_sql}, c.side_name, c.head_name, c.rule_signature,
                            c.contribution_direction, {value}
            {source}{non_null}
            ON CONFLICT ({', '.join(FAMILY_KEY_COLUMNS)}, observed_value) DO NOTHING
            """
        )
    return int(predecessor_count), int(unmatched)


def _streaming_selection_audit(
    strict_roots: Sequence[str | Path],
    *,
    cutoff: pd.Timestamp,
    config: PredecessorFamilySelectionConfig,
    spill_directory: Path,
) -> tuple[pd.DataFrame, _StreamingFamilySelectionSource]:
    """Produce the exact support-only audit without a root-wide pandas merge.

    Each strict head is validated and materialised independently.  DuckDB keeps
    only compact family aggregates and distinct support sets on disk.  No raw
    leaf token is persisted outside ``materialize_leaf_family_contributions``'s
    same-artifact temporary scope, and realised economics are never selected.
    """

    _require_duckdb()
    roots = [Path(item) for item in strict_roots]
    if not roots:
        raise CausalLeafHealthPrerequisiteError("at least one completed strict root is required")
    if len({str(item.resolve()) for item in roots}) != len(roots):
        raise CausalLeafHealthPrerequisiteError("strict family input roots must be distinct")
    spill_directory.mkdir(parents=True, exist_ok=False)
    db_path = spill_directory / "family_selection.duckdb"
    state_db = duckdb.connect(str(db_path))
    state_db.execute("PRAGMA memory_limit='1024MB'")
    state_db.execute(f"PRAGMA temp_directory={_sql_literal(spill_directory / 'duckdb_tmp')}")
    state_db.execute("PRAGMA threads=2")
    _create_streaming_selection_tables(state_db)
    manifest_hashes: dict[str, str] = {}
    predecessor_rows = 0
    try:
        for root_index, root in enumerate(roots):
            _, transports = _strict_root_layout(root)
            _validate_root_shard_identity_bounded(root, transports, state_db)
            manifest_path = root / "strict_oof_reasoning_manifest.json"
            manifest_hashes[str(root)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            artifact_paths = sorted(root.rglob("base_reasoning_manifest.json"))
            if not artifact_paths:
                raise CausalLeafHealthPrerequisiteError(f"strict health root has no per-head artifacts: {root}")
            for artifact_index, manifest_path in enumerate(artifact_paths):
                artifact = manifest_path.parent
                transport, side, head, fold, contract, class_index = _artifact_scope(root, artifact, transports)
                provenance = _artifact_candidate_provenance_bounded(
                    root, artifact, transport=transport, side=side, head=head, fold=fold,
                    contract=contract, class_index=class_index,
                )
                prefix = f"{root_index:02d}_{artifact_index:03d}"
                provenance_path = spill_directory / f"{prefix}_provenance.parquet"
                contribution_path = spill_directory / f"{prefix}_token_free_contributions.parquet"
                provenance.to_parquet(provenance_path, index=False, compression="zstd")
                try:
                    result = materialize_leaf_family_contributions(
                        artifact, contribution_path,
                        config=LeafFamilyContributionConfig(
                            assignment_batch_rows=25_000,
                            max_rows_per_output_bucket=125_000,
                        ),
                    )
                    if result.contribution_row_count <= 0:
                        raise CausalLeafHealthPrerequisiteError("strict artifact emitted no non-zero family contributions")
                    count, _ = _aggregate_one_artifact_bounded(
                        state_db, contribution_path=contribution_path, provenance_path=provenance_path,
                        contract=contract, cutoff=cutoff,
                    )
                    predecessor_rows += count
                finally:
                    contribution_path.unlink(missing_ok=True)
                    provenance_path.unlink(missing_ok=True)
                    del provenance
        if predecessor_rows <= 0:
            raise CausalLeafHealthPrerequisiteError(
                "no resolved predecessor inner_oof rows exist before the family-selection cutoff"
            )
        key = ", ".join(FAMILY_KEY_COLUMNS)
        audit = state_db.execute(
            f"""
            SELECT a.*, t.predecessor_timestamps, d.predecessor_days, s.predecessor_symbols
            FROM family_aggregate a
            JOIN (SELECT {key}, count(*) AS predecessor_timestamps FROM family_timestamps GROUP BY {key}) t USING ({key})
            JOIN (SELECT {key}, count(*) AS predecessor_days FROM family_days GROUP BY {key}) d USING ({key})
            LEFT JOIN (SELECT {key}, count(*) AS predecessor_symbols FROM family_symbols GROUP BY {key}) s USING ({key})
            ORDER BY feature_contract_sha256, side_name, head_name, contribution_direction, rule_signature
            """
        ).fetchdf()
    finally:
        state_db.close()
    if audit.empty:
        raise CausalLeafHealthPrerequisiteError("resolved predecessor rows have no token-free family contributions")
    audit["predecessor_symbols"] = audit["predecessor_symbols"].fillna(0).astype("int64")
    audit["contribution_abs_mean"] = (
        audit["contribution_abs_mass"].to_numpy(dtype=float) / audit["predecessor_rows"].to_numpy(dtype=float)
    )
    audit["eligible_support"] = (
        audit["predecessor_rows"].ge(int(config.min_rows))
        & audit["predecessor_timestamps"].ge(int(config.min_independent_timestamps))
        & audit["predecessor_days"].ge(int(config.min_trading_days))
        & audit["predecessor_symbols"].ge(int(config.min_symbols))
    )
    audit["selection_support_score"] = (
        np.log1p(audit["predecessor_rows"].to_numpy(dtype=float))
        * audit["contribution_abs_mass"].to_numpy(dtype=float)
    )
    audit["latest_label_available_utc"] = pd.to_datetime(
        audit["latest_label_available_utc"], utc=True, errors="coerce",
    )
    if audit["latest_label_available_utc"].isna().any() or not audit["latest_label_available_utc"].lt(cutoff).all():
        raise CausalLeafHealthPrerequisiteError("family-selection audit included an unresolved/evaluation label")
    ordered = audit.loc[:, [
        *FAMILY_KEY_COLUMNS, "predecessor_rows", "predecessor_timestamps", "predecessor_days",
        "predecessor_symbols", "contribution_abs_mass", "contribution_abs_mean",
        "latest_label_available_utc", "eligible_support", "selection_support_score",
    ]].sort_values(
        ["feature_contract_sha256", "side_name", "head_name", "contribution_direction", "selection_support_score", "rule_signature"],
        ascending=[True, True, True, True, False, True], kind="stable",
    ).reset_index(drop=True)
    source = _StreamingFamilySelectionSource(
        strict_roots=tuple(str(root) for root in roots),
        strict_root_manifest_sha256=manifest_hashes,
    )
    return ordered, source


def _finalise_support_only_selection_audit(
    state_db: Any,
    *,
    cutoff: pd.Timestamp,
    config: PredecessorFamilySelectionConfig,
) -> pd.DataFrame:
    """Read bounded aggregate state into the canonical outcome-free audit."""

    key = ", ".join(FAMILY_KEY_COLUMNS)
    audit = state_db.execute(
        f"""
        SELECT a.*, t.predecessor_timestamps, d.predecessor_days, s.predecessor_symbols
        FROM family_aggregate a
        JOIN (SELECT {key}, count(*) AS predecessor_timestamps FROM family_timestamps GROUP BY {key}) t USING ({key})
        JOIN (SELECT {key}, count(*) AS predecessor_days FROM family_days GROUP BY {key}) d USING ({key})
        LEFT JOIN (SELECT {key}, count(*) AS predecessor_symbols FROM family_symbols GROUP BY {key}) s USING ({key})
        ORDER BY feature_contract_sha256, side_name, head_name, contribution_direction, rule_signature
        """
    ).fetchdf()
    if audit.empty:
        raise CausalLeafHealthPrerequisiteError("resolved predecessor rows have no token-free family contributions")
    audit["predecessor_symbols"] = audit["predecessor_symbols"].fillna(0).astype("int64")
    audit["contribution_abs_mean"] = (
        audit["contribution_abs_mass"].to_numpy(dtype=float) / audit["predecessor_rows"].to_numpy(dtype=float)
    )
    audit["eligible_support"] = (
        audit["predecessor_rows"].ge(int(config.min_rows))
        & audit["predecessor_timestamps"].ge(int(config.min_independent_timestamps))
        & audit["predecessor_days"].ge(int(config.min_trading_days))
        & audit["predecessor_symbols"].ge(int(config.min_symbols))
    )
    audit["selection_support_score"] = (
        np.log1p(audit["predecessor_rows"].to_numpy(dtype=float))
        * audit["contribution_abs_mass"].to_numpy(dtype=float)
    )
    audit["latest_label_available_utc"] = pd.to_datetime(
        audit["latest_label_available_utc"], utc=True, errors="coerce",
    )
    if audit["latest_label_available_utc"].isna().any() or not audit["latest_label_available_utc"].lt(cutoff).all():
        raise CausalLeafHealthPrerequisiteError("family-selection audit included an unresolved/evaluation label")
    return audit.loc[:, [
        *FAMILY_KEY_COLUMNS, "predecessor_rows", "predecessor_timestamps", "predecessor_days",
        "predecessor_symbols", "contribution_abs_mass", "contribution_abs_mean",
        "latest_label_available_utc", "eligible_support", "selection_support_score",
    ]].sort_values(
        ["feature_contract_sha256", "side_name", "head_name", "contribution_direction", "selection_support_score", "rule_signature"],
        ascending=[True, True, True, True, False, True], kind="stable",
    ).reset_index(drop=True)


def _aggregate_event_store_selection_chunk(
    state_db: Any,
    *,
    candidates: pd.DataFrame,
    contributions: pd.DataFrame,
) -> int:
    """Aggregate one already cutoff-filtered, token-free event-store pair."""

    candidate_columns = [
        "candidate_id", "decision_ts", "side_name", "head_name", "fold_id", "transport",
        "meta_partition", "asset", "label_available_ts",
    ]
    missing = sorted(set(candidate_columns).difference(candidates.columns))
    if missing:
        raise CausalLeafHealthPrerequisiteError(f"event-store candidate selection part lacks {missing}")
    forbidden = {"semantic_label", "head_prediction", "net_bps", "base_expected_bps"}.intersection(contributions.columns)
    if forbidden:
        raise CausalLeafHealthPrerequisiteError("event-store predecessor contribution contains outcome fields")
    key = ["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition"]
    provenance = candidates.loc[:, candidate_columns].rename(columns={"decision_ts": "__ts__"})
    work = contributions.merge(provenance, on=key, how="inner", validate="many_to_one")
    if work.empty:
        raise CausalLeafHealthPrerequisiteError("event-store eligible predecessor candidate has no exact family contribution")
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce")
    work["label_available_ts"] = pd.to_datetime(work["label_available_ts"], utc=True, errors="coerce")
    work["family_ensemble_tree_contribution"] = pd.to_numeric(
        work["family_ensemble_tree_contribution"], errors="coerce"
    )
    if work[["__ts__", "label_available_ts", "family_ensemble_tree_contribution"]].isna().any().any():
        raise CausalLeafHealthPrerequisiteError("event-store predecessor selection has invalid timing/contribution data")
    state_db.register("event_store_selection_chunk", work)
    try:
        key_sql = ", ".join(FAMILY_KEY_COLUMNS)
        group = "feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction"
        state_db.execute(
            f"""
            INSERT INTO family_aggregate
            SELECT feature_contract_sha256, side_name, head_name, rule_signature, contribution_direction,
                   count(*), sum(abs(family_ensemble_tree_contribution)), max(label_available_ts)
            FROM event_store_selection_chunk
            GROUP BY {group}
            ON CONFLICT ({key_sql}) DO UPDATE SET
                predecessor_rows = family_aggregate.predecessor_rows + excluded.predecessor_rows,
                contribution_abs_mass = family_aggregate.contribution_abs_mass + excluded.contribution_abs_mass,
                latest_label_available_utc = greatest(
                    family_aggregate.latest_label_available_utc, excluded.latest_label_available_utc
                )
            """
        )
        for suffix, value in (
            ("timestamps", "__ts__"),
            ("days", "strftime(__ts__, '%Y-%m-%d')"),
            ("symbols", "asset"),
        ):
            non_null = "" if suffix != "symbols" else " WHERE asset IS NOT NULL"
            state_db.execute(
                f"""
                INSERT INTO family_{suffix}
                SELECT DISTINCT feature_contract_sha256, side_name, head_name, rule_signature,
                                contribution_direction, {value}
                FROM event_store_selection_chunk{non_null}
                ON CONFLICT ({key_sql}, observed_value) DO NOTHING
                """
            )
    finally:
        state_db.unregister("event_store_selection_chunk")
    return int(len(candidates))


def _event_store_selection_audit(
    event_store: str | Path | StrictEventStore,
    *,
    cutoff: pd.Timestamp,
    config: PredecessorFamilySelectionConfig,
    spill_directory: Path,
) -> tuple[pd.DataFrame, _StreamingFamilySelectionSource]:
    """Run the same support-only selector using a sealed reusable event store.

    The event store has already reconciled strict roots and source spool
    hashes.  This pass validates that sealed lineage but opens a family part
    only after its matching candidate part has an eligible inner-OOF row.
    Realised economics never enter either the scanner or the aggregate.
    """

    _require_duckdb()
    store = event_store if isinstance(event_store, StrictEventStore) else load_strict_event_store(
        event_store, verify_parts=False, verify_source=True,
    )
    spill_directory.mkdir(parents=True, exist_ok=False)
    state_db = duckdb.connect(str(spill_directory / "family_selection.duckdb"))
    state_db.execute("PRAGMA memory_limit='1024MB'")
    state_db.execute(f"PRAGMA temp_directory={_sql_literal(spill_directory / 'duckdb_tmp')}")
    state_db.execute("PRAGMA threads=2")
    _create_streaming_selection_tables(state_db)
    predecessor_rows = 0
    try:
        for _, candidates, contributions in iter_predecessor_selection_pairs(store, cutoff):
            predecessor_rows += _aggregate_event_store_selection_chunk(
                state_db, candidates=candidates, contributions=contributions,
            )
        if predecessor_rows <= 0:
            raise CausalLeafHealthPrerequisiteError(
                "no resolved predecessor inner_oof rows exist before the family-selection cutoff"
            )
        audit = _finalise_support_only_selection_audit(state_db, cutoff=cutoff, config=config)
    finally:
        state_db.close()
    source_manifest_hash = hashlib.sha256(store.manifest_path.read_bytes()).hexdigest()
    source_data = store.manifest.get("source", {})
    roots = tuple(map(str, source_data.get("strict_roots", []))) if isinstance(source_data, dict) else ()
    hashes = dict(source_data.get("strict_root_manifest_sha256", {})) if isinstance(source_data, dict) else {}
    if not roots or not hashes:
        raise CausalLeafHealthPrerequisiteError("sealed event store lacks strict-root lineage")
    return audit, _StreamingFamilySelectionSource(
        strict_roots=roots,
        strict_root_manifest_sha256=hashes,
        event_store_root=str(store.root),
        event_store_manifest_sha256=source_manifest_hash,
    )


def _selection_audit(
    collected: StrictOOFFamilyInputs,
    *,
    cutoff: pd.Timestamp,
    config: PredecessorFamilySelectionConfig,
) -> pd.DataFrame:
    candidates = collected.candidates.copy()
    candidates["label_available_ts"] = pd.to_datetime(candidates["label_available_ts"], utc=True, errors="coerce")
    if candidates["label_available_ts"].isna().any():
        raise CausalLeafHealthPrerequisiteError("strict family inputs have invalid label availability")
    predecessors = candidates.loc[
        candidates["meta_partition"].astype(str).eq(config.allowed_meta_partition)
        & candidates["label_available_ts"].lt(cutoff)
    ].copy()
    if predecessors.empty:
        raise CausalLeafHealthPrerequisiteError(
            "no resolved predecessor inner_oof rows exist before the family-selection cutoff"
        )
    keys = [
        "candidate_id", "__ts__", "side_name", "head_name", "fold_id",
        "transport", "meta_partition", "feature_contract_sha256",
    ]
    predecessor_identity = predecessors.rename(columns={"decision_ts": "__ts__"})
    contribution = collected.contributions.merge(
        predecessor_identity.loc[:, keys + ["label_available_ts", "asset"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if contribution.empty:
        raise CausalLeafHealthPrerequisiteError("resolved predecessor rows have no token-free family contributions")
    contribution["__ts__"] = pd.to_datetime(contribution["__ts__"], utc=True, errors="coerce")
    if contribution["__ts__"].isna().any():
        raise CausalLeafHealthPrerequisiteError("strict predecessor decisions have invalid timestamps")
    contribution["__day__"] = contribution["__ts__"].dt.strftime("%Y-%m-%d")
    contribution["__abs_contribution__"] = np.abs(
        pd.to_numeric(contribution["family_ensemble_tree_contribution"], errors="coerce")
    )
    if not np.isfinite(contribution["__abs_contribution__"].to_numpy(dtype=float)).all():
        raise CausalLeafHealthPrerequisiteError("strict predecessor family contributions are not finite")
    group = contribution.groupby(list(FAMILY_KEY_COLUMNS), observed=True, sort=True)
    audit = group.agg(
        predecessor_rows=("candidate_id", "size"),
        predecessor_timestamps=("__ts__", "nunique"),
        predecessor_days=("__day__", "nunique"),
        predecessor_symbols=("asset", "nunique"),
        contribution_abs_mass=("__abs_contribution__", "sum"),
        contribution_abs_mean=("__abs_contribution__", "mean"),
        latest_label_available_utc=("label_available_ts", "max"),
    ).reset_index()
    audit["eligible_support"] = (
        audit["predecessor_rows"].ge(int(config.min_rows))
        & audit["predecessor_timestamps"].ge(int(config.min_independent_timestamps))
        & audit["predecessor_days"].ge(int(config.min_trading_days))
        & audit["predecessor_symbols"].ge(int(config.min_symbols))
    )
    # This is deliberately outcome-free.  It selects families which are both
    # active and repeatedly expressed by the frozen base ensemble, while the
    # strict resolution cutoff proves every source row predates evaluation.
    audit["selection_support_score"] = (
        np.log1p(audit["predecessor_rows"].to_numpy(dtype=float))
        * audit["contribution_abs_mass"].to_numpy(dtype=float)
    )
    if not audit["latest_label_available_utc"].lt(cutoff).all():
        raise CausalLeafHealthPrerequisiteError("family-selection audit included an unresolved/evaluation label")
    return audit.sort_values(
        ["feature_contract_sha256", "side_name", "head_name", "contribution_direction", "selection_support_score", "rule_signature"],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _selected_payload(
    audit: pd.DataFrame,
    *,
    kind: str,
    cutoff: pd.Timestamp,
    config: PredecessorFamilySelectionConfig,
    collected: StrictOOFFamilyInputs | _StreamingFamilySelectionSource,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if kind not in FAMILY_SELECTION_KINDS:
        raise CausalLeafHealthPrerequisiteError(f"unknown family selection kind: {kind}")
    scope = ["feature_contract_sha256", "side_name", "head_name", "contribution_direction"]
    selected = audit.loc[audit["eligible_support"]].copy()
    selected["selection_rank"] = (
        selected.groupby(scope, observed=True, sort=False).cumcount() + 1
    )
    selected["selected"] = selected["selection_rank"].le(config.max_for_kind(kind))
    selected = selected.loc[selected["selected"]].copy()
    fields = list(FAMILY_KEY_COLUMNS)
    families = selected.loc[:, fields].to_dict("records")
    payload = {
        "schema": FAMILY_SELECTION_SCHEMA,
        "status": FAMILY_SELECTION_STATUS,
        "selection_kind": kind,
        "selection_cutoff_utc": cutoff,
        "selected_families": families,
        "source": {
            "strict_roots": list(collected.strict_roots),
            "strict_root_manifest_sha256": collected.strict_root_manifest_sha256,
            "event_store_root": collected.event_store_root,
            "event_store_manifest_sha256": collected.event_store_manifest_sha256,
            "allowed_meta_partition": config.allowed_meta_partition,
            "label_availability_boundary": "label_available_ts < selection_cutoff_utc",
            "evaluation_labels_used": False,
            "selection_metric": "outcome-free log1p(predecessor_rows) * absolute family contribution mass",
            "selection_columns": list(FAMILY_KEY_COLUMNS),
        },
        "config": {
            "min_rows": int(config.min_rows),
            "min_independent_timestamps": int(config.min_independent_timestamps),
            "min_trading_days": int(config.min_trading_days),
            "min_symbols": int(config.min_symbols),
            "max_families_per_scope": int(config.max_for_kind(kind)),
        },
        "row_counts": {
            "audit_families": int(len(audit)),
            "eligible_families": int(audit["eligible_support"].sum()),
            "selected_families": int(len(selected)),
        },
    }
    return payload, selected


def materialize_strict_predecessor_family_selections(
    strict_roots: Sequence[str | Path] | None,
    output_dir: str | Path,
    *,
    selection_cutoff_utc: str | pd.Timestamp,
    config: PredecessorFamilySelectionConfig = PredecessorFamilySelectionConfig(),
    event_store: str | Path | StrictEventStore | None = None,
) -> Path:
    """Freeze H3/H4/H5 family manifests from prior-resolved inner-OOF rows.

    The generated manifests cannot be applied to any candidate whose feature
    time precedes their declared cutoff; :func:`load_frozen_family_selection`
    and the health CLI enforce that application boundary.
    """

    config.validate()
    cutoff = _utc(selection_cutoff_utc, name="family selection cutoff")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite family-selection artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        if event_store is not None:
            if strict_roots:
                raise CausalLeafHealthPrerequisiteError(
                    "event-store selection is mutually exclusive with legacy strict roots"
                )
            audit, collected = _event_store_selection_audit(
                event_store, cutoff=cutoff, config=config,
                spill_directory=temporary / "bounded_selection_spill",
            )
        else:
            if not strict_roots:
                raise CausalLeafHealthPrerequisiteError("strict roots are required without --event-store")
            audit, collected = _streaming_selection_audit(
                strict_roots, cutoff=cutoff, config=config,
                spill_directory=temporary / "bounded_selection_spill",
            )
        audit_paths: list[pd.DataFrame] = []
        outputs: dict[str, str] = {}
        for kind in FAMILY_SELECTION_KINDS:
            payload, selected = _selected_payload(
                audit, kind=kind, cutoff=cutoff, config=config, collected=collected,
            )
            name = f"h{ {'context': '3', 'covariance': '4', 'relationship': '5'}[kind] }_{kind}_family_selection.json"
            path = temporary / name
            path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            outputs[name] = _sha256(path)
            tagged = audit.copy()
            selected_keys = set(tuple(row) for row in selected.loc[:, list(FAMILY_KEY_COLUMNS)].itertuples(index=False, name=None))
            tagged["selection_kind"] = kind
            tagged["selected"] = [
                tuple(row) in selected_keys
                for row in tagged.loc[:, list(FAMILY_KEY_COLUMNS)].itertuples(index=False, name=None)
            ]
            audit_paths.append(tagged)
        audit_frame = pd.concat(audit_paths, ignore_index=True)
        audit_path = temporary / "predecessor_family_selection_audit.parquet"
        audit_frame.to_parquet(audit_path, index=False, compression="zstd")
        outputs[audit_path.name] = _sha256(audit_path)
        manifest = {
            "schema": FAMILY_SELECTION_SCHEMA,
            "status": FAMILY_SELECTION_ROOT_STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_cutoff_utc": cutoff,
            "contract": {
                "selection": "strict predecessor-resolved inner_oof rows only",
                "label_availability": "label_available_ts < selection_cutoff_utc",
                "evaluation_labels_used": False,
                "raw_leaf_ids": "never read; source is same-artifact token-free family contributions",
                "application": "health candidates must have feature_generation_ts >= selection_cutoff_utc",
            },
            "config": _safe(config.__dict__),
            "strict_roots": list(collected.strict_roots),
            "strict_root_manifest_sha256": collected.strict_root_manifest_sha256,
            "event_store_root": collected.event_store_root,
            "event_store_manifest_sha256": collected.event_store_manifest_sha256,
            "outputs": outputs,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "manifest.sha256").write_text(
            _sha256(manifest_path) + "  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_frozen_family_selection(
    manifest_path: str | Path,
    *,
    expected_kind: str | None = None,
) -> FrozenFamilySelection:
    """Read and validate one immutable predecessor-only selection manifest."""

    path = Path(manifest_path)
    payload = _json(path)
    if payload.get("schema") != FAMILY_SELECTION_SCHEMA or payload.get("status") != FAMILY_SELECTION_STATUS:
        raise CausalLeafHealthPrerequisiteError(f"family selection manifest is not a strict frozen predecessor artifact: {path}")
    kind = str(payload.get("selection_kind", ""))
    if kind not in FAMILY_SELECTION_KINDS:
        raise CausalLeafHealthPrerequisiteError("family selection manifest has an unknown kind")
    if expected_kind is not None and kind != expected_kind:
        raise CausalLeafHealthPrerequisiteError(
            f"family selection kind mismatch: expected {expected_kind}, got {kind}"
        )
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("allowed_meta_partition") != "inner_oof":
        raise CausalLeafHealthPrerequisiteError("family selection manifest did not use predecessor inner_oof rows")
    if source.get("label_availability_boundary") != "label_available_ts < selection_cutoff_utc":
        raise CausalLeafHealthPrerequisiteError("family selection manifest lacks a strict label-availability boundary")
    if bool(source.get("evaluation_labels_used", True)):
        raise CausalLeafHealthPrerequisiteError("family selection manifest consumed evaluation labels")
    cutoff = _utc(payload.get("selection_cutoff_utc"), name="family selection cutoff")
    entries = payload.get("selected_families")
    if not isinstance(entries, list):
        raise CausalLeafHealthPrerequisiteError("family selection manifest selected_families must be a list")
    values: set[tuple[str, str, str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(FAMILY_KEY_COLUMNS).difference(entry):
            raise CausalLeafHealthPrerequisiteError("family selection entry lacks the full five-field identity")
        values.add(tuple(str(entry[name]) for name in FAMILY_KEY_COLUMNS))
    if len(values) != len(entries):
        raise CausalLeafHealthPrerequisiteError("family selection manifest duplicates a family identity")
    return FrozenFamilySelection(kind, cutoff, frozenset(values), payload, path)


def validate_selection_application(
    selections: Iterable[FrozenFamilySelection],
    *,
    strict_roots: Sequence[str | Path],
) -> pd.Timestamp | None:
    """Validate a selection plan and return its causal activation cutoff.

    The health builder retains prior observations as state history, but emits
    H3/H4/H5 only at or after the returned cutoff.  Thus an earlier candidate
    in the same chronological root cannot receive a family feature chosen at
    a later predecessor boundary.
    """

    items = tuple(selections)
    cutoffs = {item.cutoff_utc for item in items}
    if len(cutoffs) > 1:
        raise CausalLeafHealthPrerequisiteError(
            "H3/H4/H5 family selections must share one activation cutoff"
        )
    cutoff = next(iter(cutoffs), None)
    if cutoff is None:
        return None
    # Reuse the strict candidate reader to ensure the requested root has a
    # genuine post-cutoff segment.  Earlier rows are valid state history but
    # their H3/H4/H5 values are explicitly zeroed by the activation boundary.
    from .causal_leaf_health_artifacts import _strict_candidate_shards  # noqa: PLC0415

    times: list[pd.Series] = []
    for item in strict_roots:
        frame = _strict_candidate_shards(Path(item))
        times.append(pd.to_datetime(frame["feature_generation_ts"], utc=True, errors="coerce"))
    values = pd.concat(times, ignore_index=True)
    if values.isna().any():
        raise CausalLeafHealthPrerequisiteError("strict health candidates have invalid feature-generation timestamps")
    if not values.ge(cutoff).any():
        raise CausalLeafHealthPrerequisiteError(
            "strict health roots contain no post-cutoff candidates for the frozen family selection: "
            f"{cutoff.isoformat()}"
        )
    return cutoff


def load_strict_fold_context(
    root: str | Path,
    *,
    context_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Load the shared hourly timeline from a verified strict context root."""

    directory = Path(root)
    manifest = _json(directory / "strict_fold_context_manifest.json")
    if manifest.get("schema") != STRICT_CONTEXT_SCHEMA or manifest.get("status") != STRICT_CONTEXT_STATUS:
        raise CausalLeafHealthPrerequisiteError("context root is not a strict full causal context sidecar")
    window = manifest.get("window", {})
    if (
        _utc(window.get("start_utc"), name="context manifest start") != STRICT_CONTEXT_START_UTC
        or _utc(window.get("end_exclusive_utc"), name="context manifest end") != STRICT_CONTEXT_END_EXCLUSIVE_UTC
    ):
        raise CausalLeafHealthPrerequisiteError("context root does not cover the required July-2023--Nov-2024 interval")
    columns = tuple(context_columns or manifest.get("contract", {}).get("health_default_context_columns", ()))
    if not columns:
        raise CausalLeafHealthPrerequisiteError("strict context root has no declared H3/H4/H5 fields")
    if len(columns) > 10:
        raise CausalLeafHealthPrerequisiteError("strict context root declares too many H3/H4/H5 fields")
    timeline_path = directory / "hourly_oof_market_regimes.parquet"
    if not timeline_path.is_file():
        raise CausalLeafHealthPrerequisiteError("strict context root lacks its hourly causal timeline")
    context = pd.read_parquet(timeline_path, columns=["source_utc", *columns])
    context = context.rename(columns={"source_utc": "regime_available_utc"})
    context["regime_available_utc"] = pd.to_datetime(context["regime_available_utc"], utc=True, errors="coerce")
    if context["regime_available_utc"].isna().any() or context["regime_available_utc"].duplicated().any():
        raise CausalLeafHealthPrerequisiteError("strict context timeline has invalid/duplicate availability timestamps")
    if context["regime_available_utc"].lt(STRICT_CONTEXT_START_UTC).any() or context["regime_available_utc"].ge(STRICT_CONTEXT_END_EXCLUSIVE_UTC).any():
        raise CausalLeafHealthPrerequisiteError("strict context timeline escaped its declared time boundary")
    values = context.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise CausalLeafHealthPrerequisiteError("strict context fields are not finite")
    context.loc[:, list(columns)] = values.astype(np.float32)
    return context.sort_values("regime_available_utc", kind="stable").reset_index(drop=True), columns, manifest


__all__ = [
    "STRICT_CONTEXT_SCHEMA",
    "STRICT_CONTEXT_STATUS",
    "FAMILY_SELECTION_SCHEMA",
    "FAMILY_SELECTION_STATUS",
    "FAMILY_SELECTION_ROOT_STATUS",
    "STRICT_CONTEXT_START_UTC",
    "STRICT_CONTEXT_END_EXCLUSIVE_UTC",
    "FAMILY_KEY_COLUMNS",
    "FAMILY_SELECTION_KINDS",
    "DEFAULT_HEALTH_CONTEXT_COLUMNS",
    "CausalLeafHealthPrerequisiteError",
    "PredecessorFamilySelectionConfig",
    "FrozenFamilySelection",
    "strict_candidate_population",
    "materialize_strict_fold_causal_context",
    "materialize_strict_predecessor_family_selections",
    "load_frozen_family_selection",
    "validate_selection_application",
    "load_strict_fold_context",
]
