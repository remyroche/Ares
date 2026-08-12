"""Fail-closed materialisation of the Stage-II path/context ledger.

This is intentionally a data bridge, not a label generator or a model runner.
It joins the frozen Stage-I identity population to predeclared canonical path
substrates and to the frozen selected-panel context.  Every source choice,
descriptor alias, timing convention and checkpoint is content-bound.
"""

from __future__ import annotations

from hashlib import sha256
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .path_auxiliary_targets import build_path_auxiliary_targets
from .stage_ii_execution import StageIIExecutionError, file_sha256


SOURCE_MAP_SCHEMA = "stage_ii_enriched_path_source_map_v1"
OUTPUT_SCHEMA = "stage_ii_enriched_path_context_ledger_v1"
IDENTITY = (
    "candidate_id", "symbol", "side_name", "signal_close_ts", "decision_ts",
    "label_available_ts",
)
# The only realised labels that may cross this bridge.  A candidate spec must
# explicitly name a subset; arbitrary path/outcome columns cannot slip in.
PATH_DESCRIPTOR_WHITELIST = frozenset({
    "path_arch_peak_mfe_atr", "path_arch_mae_before_meaningful_mfe_r",
    "path_arch_time_to_first_meaningful_mfe_h", "path_arch_future_slope_atr_per_hour_12h",
    "path_arch_final_return_net_1pct", "path_arch_peak_retention_ratio",
    "path_arch_mfe_mae_efficiency", "path_arch_time_to_peak_mfe_h",
    "path_arch_adverse_trough_atr", "path_arch_adverse_trough_recovery_fraction",
    "path_arch_mae_before_mfe", "path_arch_time_to_90pct_peak_mfe_h",
    "path_arch_raw_peak_mfe_r", "path_arch_peak_mfe_r",
    "path_arch_peak_mfe_minus_cost_atr", "path_arch_peak_mfe_div_cost",
    "path_arch_future_slope_atr_per_hour_4h",
    "__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__",
    "__mae_before_meaningful_mfe_atr_12h__", "__bars_before_price_stops_decreasing_12h__",
    "__future_slope_atr_per_hour_12h__", "__log1p_peak_mfe_atr_12h__",
    "__log1p_time_to_first_meaningful_mfe_hours_12h__",
    "__log1p_mae_before_meaningful_mfe_atr_12h__",
    "__log1p_bars_before_price_stops_decreasing_12h__",
    "__log1p_future_slope_atr_per_hour_12h__",
})


class StageIIEnrichedMaterializationError(StageIIExecutionError):
    pass


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()


def _sha_text(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _utc(value: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if result.isna().any():
        raise StageIIEnrichedMaterializationError(f"{name} contains invalid timestamps")
    return result


def _identity_hash(frame: pd.DataFrame) -> str:
    missing = set(IDENTITY).difference(frame.columns)
    if missing:
        raise StageIIEnrichedMaterializationError(f"identity hash lacks {sorted(missing)}")
    work = frame.loc[:, list(IDENTITY)].copy()
    for name in ("signal_close_ts", "decision_ts", "label_available_ts"):
        work[name] = _utc(work[name], name).astype("int64")
    work["side_name"] = work.side_name.astype(str).str.lower()
    work["symbol"] = work.symbol.astype(str)
    work["candidate_id"] = work.candidate_id.astype(str)
    if work.duplicated().any():
        raise StageIIEnrichedMaterializationError("duplicate immutable enriched identity")
    return _sha_text(work.sort_values(list(IDENTITY), kind="stable").to_dict("records"))


def _files(values: Any, *, base: Path) -> list[Path]:
    raw = [values] if isinstance(values, str) else list(values or ())
    found: list[Path] = []
    for item in raw:
        text = str(item)
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = base / candidate
        matches = [Path(value) for value in glob.glob(str(candidate), recursive=True)]
        if not matches and candidate.is_file():
            matches = [candidate]
        found.extend(path.resolve() for path in matches if path.is_file() and path.suffix == ".parquet")
    unique = sorted(set(found))
    if not unique:
        raise StageIIEnrichedMaterializationError("path source has no declared parquet files")
    return unique


def _source_fingerprint(paths: Sequence[Path]) -> str:
    return _sha_text([{"path": str(path), "sha256": file_sha256(path)} for path in paths])


def load_source_map(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SOURCE_MAP_SCHEMA:
        raise StageIIEnrichedMaterializationError("unsupported Stage-II path-source-map schema")
    sources = raw.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise StageIIEnrichedMaterializationError("source map must declare exactly the 2024, January-2025 and 2025+ substrates")
    expected = {"historical_2024", "native_january_2025", "path_archetype_2025_plus"}
    names = {str(item.get("source_id", "")) for item in sources if isinstance(item, Mapping)}
    if names != expected:
        raise StageIIEnrichedMaterializationError("source map must use the three canonical source ids")
    previous_end: pd.Timestamp | None = None
    for item in sorted(sources, key=lambda row: str(row["start_utc"])):
        start, end = _utc(pd.Series([item.get("start_utc")]), "source start").iloc[0], _utc(pd.Series([item.get("end_utc")]), "source end").iloc[0]
        if not start < end or (previous_end is not None and start != previous_end):
            raise StageIIEnrichedMaterializationError("source routing must be contiguous, non-overlapping UTC intervals")
        previous_end = end
        if str(item.get("kind")) not in {"canonical_parquet", "native_path_descriptors"}:
            raise StageIIEnrichedMaterializationError("source kind must be canonical_parquet or native_path_descriptors")
        mapping = item.get("descriptor_mapping")
        if not isinstance(mapping, Mapping) or not mapping:
            raise StageIIEnrichedMaterializationError("each source requires explicit descriptor_mapping")
        unknown = set(map(str, mapping)).difference(PATH_DESCRIPTOR_WHITELIST)
        if unknown:
            raise StageIIEnrichedMaterializationError(f"descriptor mapping contains non-whitelisted fields: {sorted(unknown)}")
    return raw


def parse_candidate_spec(path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise StageIIEnrichedMaterializationError("candidate spec must be a JSON object")
    meta = tuple(dict.fromkeys(map(str, raw.get("meta_feature_cols", ()))))
    causal: list[str] = []
    descriptors: list[str] = []
    entries = raw.get("candidates")
    if not meta or not isinstance(entries, list) or not 1 <= len(entries) <= 8:
        raise StageIIEnrichedMaterializationError("candidate spec must contain bounded meta fields and one to eight candidates")
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("config"), Mapping):
            raise StageIIEnrichedMaterializationError("candidate spec has malformed candidate")
        causal.extend(map(str, entry.get("causal_feature_cols", ())))
        descriptors.extend(map(str, entry["config"].get("path_descriptor_cols", ())))
    required_path = tuple(dict.fromkeys(descriptors))
    unknown = set(required_path).difference(PATH_DESCRIPTOR_WHITELIST)
    if unknown:
        raise StageIIEnrichedMaterializationError(f"candidate spec requests non-whitelisted path descriptors: {sorted(unknown)}")
    if not required_path:
        raise StageIIEnrichedMaterializationError("Stage-II candidate spec requires at least one realised path descriptor")
    return tuple(dict.fromkeys((*meta, *causal))), required_path


def _stage_i_population(
    stage_i_dir: Path,
    *,
    signal_start: pd.Timestamp | None = None,
    signal_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    manifest = json.loads((stage_i_dir / "manifest.json").read_text(encoding="utf-8"))
    schema = str(manifest.get("schema", ""))
    if schema not in {
        "stage_i_production_winner_oos_v1",
        "stage_i_target_specific_direct_fq3_oos_v1",
    } or manifest.get("status") != "complete":
        raise StageIIEnrichedMaterializationError("Stage-I input must be a completed frozen production OOS artifact")
    path = stage_i_dir / (
        "full_history_raw_oof_predictions.parquet"
        if schema == "stage_i_production_winner_oos_v1"
        else "full_history_strict_oof_predictions.parquet"
    )
    if not path.is_file():
        raise StageIIEnrichedMaterializationError("Stage-I artifact lacks full_history_raw_oof_predictions.parquet")
    raw = pd.read_parquet(path)
    symbol_column = "symbol" if "symbol" in raw.columns else "__symbol__"
    required = {"candidate_id", "side_name", symbol_column, "decision_ts", "label_available_ts", "base_strict_oof_available"}
    missing = required.difference(raw.columns)
    if missing:
        raise StageIIEnrichedMaterializationError(f"Stage-I population lacks {sorted(missing)}")
    flag = pd.to_numeric(raw.base_strict_oof_available, errors="coerce").eq(1.0)
    out = raw.loc[flag, ["candidate_id", "side_name", symbol_column, "decision_ts", "label_available_ts"]].copy()
    out = out.rename(columns={symbol_column: "symbol"})
    out["decision_ts"] = _utc(out.decision_ts, "Stage-I decision_ts")
    out["label_available_ts"] = _utc(out.label_available_ts, "Stage-I label_available_ts")
    out["signal_close_ts"] = out.decision_ts - pd.Timedelta(hours=1)
    if signal_start is not None or signal_end is not None:
        # Stage II is deliberately bounded by its declared path substrates.
        # A full-history Stage-I artifact may retain earlier rows for audit,
        # but no un-routed history may silently become an uncovered Stage-II
        # observation (or trigger a request to backfill a different contract).
        mask = pd.Series(True, index=out.index)
        if signal_start is not None:
            mask &= out.signal_close_ts.ge(signal_start)
        if signal_end is not None:
            mask &= out.signal_close_ts.lt(signal_end)
        out = out.loc[mask].copy()
        if out.empty:
            raise StageIIEnrichedMaterializationError(
                "declared Stage-II source interval contains no Stage-I base OOF identities"
            )
    if not out.label_available_ts.eq(out.decision_ts + pd.Timedelta(hours=12)).all():
        raise StageIIEnrichedMaterializationError("Stage-I rows do not obey exact decision+12h label availability")
    out["side_name"] = out.side_name.astype(str).str.lower()
    if not out.side_name.isin(("long", "short")).all() or out.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIIEnrichedMaterializationError("Stage-I immutable population is noncanonical or duplicated")
    return out.loc[:, list(IDENTITY)].sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)


def _load_context(selected_panel: Path, *, fields: Sequence[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    # The target-specific Stage-I materializer deliberately writes a compact
    # feature matrix beside a separate immutable identity/target contract.  It
    # is not safe to recover side or timestamp from a filename or row order;
    # join the two only on their shared candidate identity and prove all other
    # aliases agree.
    if selected_panel.is_dir():
        for side_dir in sorted(path for path in selected_panel.iterdir() if path.is_dir()):
            feature_path, contract_path = side_dir / "features.parquet", side_dir / "contract.parquet"
            if not feature_path.is_file() or not contract_path.is_file():
                continue
            feature_schema = set(pq.ParquetFile(feature_path).schema_arrow.names)
            if not set(fields).issubset(feature_schema):
                continue
            feature_identity = {"candidate_id", "__ts__", "__symbol__"}
            if not feature_identity.issubset(feature_schema):
                continue
            contract = pd.read_parquet(contract_path)
            contract_required = {"candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts"}
            if not contract_required.issubset(contract.columns):
                raise StageIIEnrichedMaterializationError(
                    f"{side_dir}: Stage-I contract lacks immutable feature identities"
                )
            features = pd.read_parquet(
                feature_path,
                columns=list(dict.fromkeys(("candidate_id", "__ts__", "__symbol__", *fields))),
            )
            if features.duplicated(["candidate_id", "__ts__", "__symbol__"]).any():
                raise StageIIEnrichedMaterializationError(
                    f"{side_dir}: Stage-I feature matrix has duplicate identities"
                )
            contract = contract.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts"]].copy()
            if contract.duplicated(["candidate_id", "__ts__", "__symbol__"]).any():
                raise StageIIEnrichedMaterializationError(
                    f"{side_dir}: Stage-I contract has duplicate feature identities"
                )
            joined = contract.merge(
                features, on=["candidate_id", "__ts__", "__symbol__"], how="inner",
                validate="one_to_one", sort=False,
            )
            if len(joined) != len(contract) or len(joined) != len(features):
                raise StageIIEnrichedMaterializationError(
                    f"{side_dir}: Stage-I feature/contract identities do not match"
                )
            joined = joined.rename(columns={"__symbol__": "symbol"})
            parts.append(joined.loc[:, ["candidate_id", "side_name", "symbol", "decision_ts", *fields]].copy())

    files = _files([str(selected_panel / "**" / "*.parquet")] if selected_panel.is_dir() else [str(selected_panel)], base=Path.cwd())
    for file in files:
        probe = pd.read_parquet(file)
        available = set(probe.columns)
        identity_alias = {"symbol": "symbol", "side_name": "side_name", "candidate_id": "candidate_id", "decision_ts": "decision_ts"}
        if not set(identity_alias.values()).issubset(available):
            continue
        missing = set(fields).difference(available)
        if missing:
            # Not every panel shard necessarily owns both side surfaces; the
            # complete union is checked after concatenation.
            continue
        parts.append(probe.loc[:, ["candidate_id", "side_name", "symbol", "decision_ts", *fields]].copy())
    if not parts:
        raise StageIIEnrichedMaterializationError("selected-panel context has no identity-aligned shard with all requested fields")
    context = pd.concat(parts, ignore_index=True)
    context["decision_ts"] = _utc(context.decision_ts, "context decision_ts")
    context["side_name"] = context.side_name.astype(str).str.lower()
    key = ["candidate_id", "side_name", "symbol", "decision_ts"]
    if context.duplicated(key).any():
        raise StageIIEnrichedMaterializationError("selected-panel context has duplicate identities")
    if not np.isfinite(context.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        raise StageIIEnrichedMaterializationError("selected-panel context has non-finite declared causal fields")
    return context


def _normalise_source(frame: pd.DataFrame, source: Mapping[str, Any], *, descriptors: Sequence[str]) -> pd.DataFrame:
    columns = dict(source.get("columns", {}))
    def field(name: str, default: str) -> str:
        return str(columns.get(name, default))
    raw_required = {field("candidate_id", "candidate_id"), field("side_name", "side_name"), field("symbol", "__symbol__"), field("signal_close_ts", "__ts__"), field("decision_ts", "__decision_ts__")}
    mapping = {str(key): str(value) for key, value in dict(source["descriptor_mapping"]).items() if str(key) in descriptors}
    if set(descriptors).difference(mapping):
        raise StageIIEnrichedMaterializationError(f"{source['source_id']} does not map every requested descriptor")
    needed = raw_required | set(mapping.values())
    label_col = columns.get("label_available_ts")
    if label_col is not None:
        needed.add(str(label_col))
    missing = needed.difference(frame.columns)
    if missing:
        raise StageIIEnrichedMaterializationError(f"{source['source_id']} source lacks {sorted(missing)}")
    out = pd.DataFrame({
        "candidate_id": frame[field("candidate_id", "candidate_id")].astype(str),
        "side_name": frame[field("side_name", "side_name")].astype(str).str.lower(),
        "symbol": frame[field("symbol", "__symbol__")].astype(str),
        "signal_close_ts": _utc(frame[field("signal_close_ts", "__ts__")], "path signal timestamp"),
        "decision_ts": _utc(frame[field("decision_ts", "__decision_ts__")], "path decision timestamp"),
    })
    if label_col is not None:
        out["label_available_ts"] = _utc(frame[str(label_col)], "path label availability")
    elif int(source.get("label_available_offset_hours", -1)) == 12:
        out["label_available_ts"] = out.decision_ts + pd.Timedelta(hours=12)
    else:
        raise StageIIEnrichedMaterializationError(f"{source['source_id']} must explicitly declare label availability or +12h derivation")
    for output, raw in mapping.items():
        out[output] = pd.to_numeric(frame[raw], errors="coerce")
    if not out.decision_ts.eq(out.signal_close_ts + pd.Timedelta(hours=1)).all() or not out.label_available_ts.eq(out.decision_ts + pd.Timedelta(hours=12)).all():
        raise StageIIEnrichedMaterializationError(f"{source['source_id']} violates signal+1h/decision+12h timing")
    if out.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIIEnrichedMaterializationError(f"{source['source_id']} has duplicate path identities")
    if not np.isfinite(out.loc[:, list(descriptors)].to_numpy(float)).all():
        raise StageIIEnrichedMaterializationError(f"{source['source_id']} has non-finite declared descriptors")
    return out


def _materialize_native_path_descriptors(
    frame: pd.DataFrame, source: Mapping[str, Any]
) -> pd.DataFrame:
    """Derive only predeclared 12h auxiliary descriptors from exact 1m paths.

    The ATR field is mandatory and named by the source map: there is no hidden
    barrier/ATR proxy.  This is the January bridge's only path computation and
    is deterministic from the archived native 1m path at decision time.
    """

    columns = dict(source.get("columns", {}))
    path_col = str(columns.get("native_path", "native_future_ohlc_path"))
    entry_col = str(columns.get("entry_price", "decision_price"))
    atr_col = str(columns.get("atr_fraction", ""))
    required = {path_col, entry_col, atr_col, str(columns.get("side_name", "side_name"))}
    if not atr_col or not required.issubset(frame.columns):
        raise StageIIEnrichedMaterializationError(
            f"{source['source_id']}: raw native paths require explicitly declared native_path, entry_price and atr_fraction columns"
        )
    parsed = []
    for value in frame[path_col]:
        try:
            payload = json.loads(value)
            timestamp = np.asarray(payload["timestamp"], dtype=np.int64)
            high = np.asarray(payload["high"], dtype=np.float64)
            low = np.asarray(payload["low"], dtype=np.float64)
        except (TypeError, ValueError, KeyError) as exc:
            raise StageIIEnrichedMaterializationError(
                f"{source['source_id']}: malformed native OHLC path"
            ) from exc
        if len(timestamp) != 720 or len(high) != 720 or len(low) != 720 or np.any(np.diff(timestamp) != 60_000_000_000):
            raise StageIIEnrichedMaterializationError(
                f"{source['source_id']}: native path must be contiguous 720x1m bars"
            )
        parsed.append((timestamp, high, low))
    decision_col = str(columns.get("decision_ts", "__decision_ts__"))
    decision_ns = _utc(frame[decision_col], "native decision timestamp").astype("int64").to_numpy()
    if any(item[0][0] != decision_ns[index] for index, item in enumerate(parsed)):
        raise StageIIEnrichedMaterializationError(f"{source['source_id']}: native path does not begin at decision_ts")
    high = np.asarray([item[1] for item in parsed], dtype=np.float64)
    low = np.asarray([item[2] for item in parsed], dtype=np.float64)
    side = np.where(frame[str(columns.get("side_name", "side_name"))].astype(str).str.lower().eq("short"), -1.0, 1.0)
    targets = build_path_auxiliary_targets(
        entry_price=pd.to_numeric(frame[entry_col], errors="coerce").to_numpy(float),
        future_high=high, future_low=low,
        atr_fraction=pd.to_numeric(frame[atr_col], errors="coerce").to_numpy(float),
        side_sign=side, bar_minutes=1, horizon_hours=12,
    ).as_columns()
    out = frame.copy()
    for name, values in targets.items():
        out[name] = values
    return out


def materialize_stage_ii_enriched_ledger(
    *, stage_i_oos_dir: str | Path, selected_panel: str | Path, candidate_spec: str | Path,
    source_map: str | Path, output_dir: str | Path, resume: bool = False,
) -> Path:
    """Materialise a restart-safe immutable ledger; never backfill missing rows."""
    stage_i_dir, context_dir, output = Path(stage_i_oos_dir).resolve(), Path(selected_panel).resolve(), Path(output_dir).resolve()
    sources_path = Path(source_map).resolve()
    source_config = load_source_map(sources_path)
    causal_fields, path_fields = parse_candidate_spec(candidate_spec)
    request = {
        "stage_i_manifest_sha256": file_sha256(stage_i_dir / "manifest.json"),
        "selected_panel": str(context_dir), "candidate_spec_sha256": file_sha256(candidate_spec),
        "source_map_sha256": file_sha256(sources_path), "causal_fields": causal_fields, "path_fields": path_fields,
    }
    state_path = output / "materialization_state.json"
    if output.exists() and not resume:
        raise StageIIEnrichedMaterializationError("output already exists; use a new path or --resume")
    output.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("request_sha256") != _sha_text(request):
            raise StageIIEnrichedMaterializationError("resume request differs from existing materialization state")
    else:
        state_path.write_text(json.dumps({"schema": OUTPUT_SCHEMA, "status": "running", "request_sha256": _sha_text(request)}, indent=2) + "\n")
    source_start = min(
        _utc(pd.Series([source["start_utc"]]), "source start").iloc[0]
        for source in source_config["sources"]
    )
    source_end = max(
        _utc(pd.Series([source["end_utc"]]), "source end").iloc[0]
        for source in source_config["sources"]
    )
    population = _stage_i_population(
        stage_i_dir, signal_start=source_start, signal_end=source_end,
    )
    path_parts: list[pd.DataFrame] = []
    source_lineage: list[dict[str, Any]] = []
    for source in source_config["sources"]:
        source_id = str(source["source_id"])
        checkpoint = output / f"checkpoint_{source_id}.parquet"
        files = _files(source.get("paths"), base=sources_path.parent)
        fingerprint = _source_fingerprint(files)
        if checkpoint.is_file():
            part = pd.read_parquet(checkpoint)
        else:
            if str(source["kind"]) not in {"canonical_parquet", "native_path_descriptors"}:
                raise AssertionError("validated source kind drift")
            raw = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
            if str(source["kind"]) == "native_path_descriptors" and str(
                dict(source.get("columns", {})).get("native_path", "native_future_ohlc_path")
            ) in raw.columns:
                raw = _materialize_native_path_descriptors(raw, source)
            part = _normalise_source(raw, source, descriptors=path_fields)
            start = _utc(pd.Series([source["start_utc"]]), "source start").iloc[0]
            end = _utc(pd.Series([source["end_utc"]]), "source end").iloc[0]
            part = part.loc[part.signal_close_ts.ge(start) & part.signal_close_ts.lt(end)].copy()
            part.to_parquet(checkpoint, index=False, compression="zstd")
        path_parts.append(part)
        source_lineage.append({"source_id": source_id, "paths": [str(file) for file in files], "source_sha256": fingerprint, "identity_sha256": _identity_hash(part)})
    paths = pd.concat(path_parts, ignore_index=True)
    if paths.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIIEnrichedMaterializationError("source-date routing produced duplicate path identities")
    expected_keys = ["candidate_id", "side_name", "symbol", "decision_ts", "signal_close_ts", "label_available_ts"]
    joined = population.merge(paths, on=expected_keys, how="left", validate="one_to_one", indicator=True, sort=False)
    if not joined["_merge"].eq("both").all():
        raise StageIIEnrichedMaterializationError("canonical path sources do not cover every direct Stage-I base OOF identity")
    context = _load_context(context_dir, fields=causal_fields)
    joined = joined.merge(context, on=["candidate_id", "side_name", "symbol", "decision_ts"], how="left", validate="one_to_one", indicator="__context_merge__", sort=False)
    if not joined["__context_merge__"].eq("both").all():
        raise StageIIEnrichedMaterializationError("frozen selected-panel context does not cover every Stage-I base OOF identity")
    final = joined.loc[:, [*IDENTITY, *causal_fields, *path_fields]].copy()
    if final.duplicated(list(IDENTITY)).any() or not np.isfinite(final.loc[:, [*causal_fields, *path_fields]].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        raise StageIIEnrichedMaterializationError("enriched output has duplicate identity or non-finite declared input")
    ledger = output / "stage_ii_enriched_ledger.parquet"
    final.sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable").to_parquet(ledger, index=False, compression="zstd")
    manifest = {
        "schema": OUTPUT_SCHEMA, "status": "complete", "ledger_sha256": file_sha256(ledger),
        "identity_columns": list(IDENTITY), "causal_columns": list(causal_fields), "path_descriptor_columns": list(path_fields),
        "label_lineage": {"artifact_path": str(sources_path), "artifact_sha256": file_sha256(sources_path), "identity_sha256": _identity_hash(paths)},
        "context_lineage": {"artifact_path": str(context_dir), "artifact_sha256": _sha_text({"stage_i_manifest": request["stage_i_manifest_sha256"], "selected_panel": str(context_dir)}), "identity_sha256": _identity_hash(context.merge(population.loc[:, ["candidate_id", "side_name", "symbol", "decision_ts", "signal_close_ts", "label_available_ts"]], on=["candidate_id", "side_name", "symbol", "decision_ts"], how="inner", validate="one_to_one"))},
        "source_map": {"path": str(sources_path), "sha256": file_sha256(sources_path), "sources": source_lineage},
        "stage_i_oos": {"path": str(stage_i_dir), "manifest_sha256": request["stage_i_manifest_sha256"], "identity_sha256": _identity_hash(population), "source_interval": {"start_utc": str(source_start), "end_utc": str(source_end)}},
        "rows": int(len(final)), "restart_safe_checkpoints": [f"checkpoint_{item['source_id']}.parquet" for item in source_lineage],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    state_path.write_text(json.dumps({"schema": OUTPUT_SCHEMA, "status": "complete", "request_sha256": _sha_text(request), "manifest_sha256": file_sha256(output / "manifest.json")}, indent=2) + "\n")
    return output


__all__ = ["IDENTITY", "OUTPUT_SCHEMA", "PATH_DESCRIPTOR_WHITELIST", "SOURCE_MAP_SCHEMA", "StageIIEnrichedMaterializationError", "load_source_map", "materialize_stage_ii_enriched_ledger", "parse_candidate_spec"]
