#!/usr/bin/env python3
"""Resumable side-local base/meta AE/GMM representation search.

This is research infrastructure, not a production trainer.  It consumes one
already materialised, point-in-time feature parquet per layer, fits state
blocks only on rows strictly before ``--train-end``, and writes optional
sidecars.  No existing model or live feature schema is changed by this script.

Example (prepare only)::

  python scripts/run_side_local_ae_gmm_representation_search.py prepare \
    --layer meta --input-parquet ... --train-end 2026-06-26 --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.side_local_ae_gmm_search import (  # noqa: E402
    SCHEMA_VERSION,
    SIDES,
    SearchConfig,
    SideLocalState,
    available_layer_features,
    config_feature_universe,
    correlation_cluster_representatives,
    evaluate_density_candidate,
    feature_schema_hash,
    fit_encoder,
    fit_gmm,
    nested_feature_prefixes,
    refine_top_diagonal_candidates,
    score_candidates,
    stable_feature_filter,
    univariate_relief_mda_ranking,
)


STAGES = ("plan", "prepare", "proxy", "full", "refine", "final-refit", "feature-ablation", "materialize", "all")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _timestamp(frame: pd.DataFrame) -> pd.Series:
    for column in ("timestamp", "ts", "event_ts", "__ts__"):
        if column in frame:
            return pd.to_datetime(frame[column], utc=True, errors="coerce")
    raise KeyError("input requires timestamp, ts, or event_ts")


def _side(frame: pd.DataFrame) -> pd.Series:
    for column in ("side_name", "side"):
        if column in frame:
            return frame[column].astype(str).str.lower()
    raise KeyError("input requires side_name or side")


def _target(frame: pd.DataFrame, supplied: str | None) -> str:
    candidates = [supplied] if supplied else []
    candidates += ["target_soft", "first_touch_target_soft", "__first_touch_target_soft__", "target"]
    for name in candidates:
        if name and name in frame:
            return name
    raise KeyError("input requires a soft target column; pass --target-column")


def _net_ev(frame: pd.DataFrame, supplied: str | None) -> str | None:
    candidates = [supplied] if supplied else []
    candidates += ["ret_net", "first_touch_capture_net", "__first_touch_capture_net__", "u_policy_net", "__u_policy_net__", "exec_margin_net", "ev_after_1pct", "net_ev"]
    return next((name for name in candidates if name and name in frame), None)


def _archetype(frame: pd.DataFrame) -> str | None:
    return next((name for name in ("archetype_label_family", "__archetype_label_family__", "policy_archetype", "local_side_archetype", "source_archetype") if name in frame), None)


def _load_label_features(
    args: argparse.Namespace,
    *,
    requested: list[str],
    max_rows_per_side: int,
    scratch_name: str,
) -> pd.DataFrame:
    """Load a deterministic label reference and only the requested PTI features.

    The final refit must not fetch the full config universe again.  Selection is
    already frozen, so retrieving only the selected side-local feature union is
    both faster and the exact inference schema of the serialized package.
    """
    source = _load_label_reference_columns(Path(args.labels_path), shard_cap_per_side=int(args.label_shard_cap_per_side))
    print(f"[state_reference] streamed label rows={len(source)}", flush=True)
    ts_all = _timestamp(source)
    end = pd.Timestamp(args.train_end, tz="UTC")
    source = source.loc[ts_all.notna() & (ts_all < end)].copy()
    source["timestamp"] = ts_all.loc[source.index]
    source = _cap_side_bme(source, max_rows_per_side=max_rows_per_side)
    print(f"[state_reference] capped B/M/E rows={len(source)}", flush=True)
    scratch = Path(args.out) / "_reference_feature_batches" / scratch_name
    scratch.mkdir(parents=True, exist_ok=True)
    reference_path = scratch / "reference_identity.parquet"
    identity_columns = [name for name in ("__ts__", "__symbol__", "side_name", "side", "candidate_id") if name in source]
    reuse_batches = False
    if args.reuse_feature_batches and reference_path.exists() and identity_columns:
        try:
            prior_reference = pd.read_parquet(reference_path, columns=identity_columns)
            current_reference = source[identity_columns].reset_index(drop=True)
            reuse_batches = len(prior_reference) == len(current_reference) and prior_reference.equals(current_reference)
        except Exception:
            reuse_batches = False
    source.to_parquet(reference_path, index=False)
    batches = []
    for start in range(0, len(requested), int(args.feature_load_batch)):
        feature_batch = requested[start : start + int(args.feature_load_batch)]
        batch_id = f"{start:04d}_{start + len(feature_batch):04d}"
        names_path = scratch / f"features_{batch_id}.json"
        output_path = scratch / f"features_{batch_id}.parquet"
        reusable = reuse_batches and output_path.exists()
        if not reusable:
            names_path.write_text(json.dumps(feature_batch) + "\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/side_local_feature_batch_worker.py"), "--reference-parquet", str(reference_path), "--feature-dir", str(args.feature_dir), "--features-json", str(names_path), "--out", str(output_path)]
            completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"Feature batch {batch_id} failed: {completed.stderr[-2000:]}")
        additions = pd.read_parquet(output_path)
        if len(additions) != len(source):
            raise RuntimeError(f"Feature batch {batch_id} row count mismatch ({len(additions)} != {len(source)})")
        if not additions.empty:
            batches.append(additions.reset_index(drop=True).astype(np.float32, copy=False))
        mode = "reused" if reusable else "loaded"
        print(f"[state_reference] feature batch {min(start + int(args.feature_load_batch), len(requested))}/{len(requested)} {mode}", flush=True)
    if not batches:
        raise RuntimeError("Point-in-time feature-store lookup returned no columns for label reference")
    return pd.concat([source.reset_index(drop=True), *batches], axis=1, copy=False)


def _load_train(args: argparse.Namespace) -> pd.DataFrame:
    if args.labels_path:
        source = _load_label_features(
            args,
            requested=config_feature_universe(args.layer),
            max_rows_per_side=int(args.reference_cap_per_side),
            scratch_name=args.layer,
        )
    else:
        source = _read(Path(args.input_parquet))
    if args.outcome_parquet:
        outcomes = _read(Path(args.outcome_parquet))
        keys = [name for name in ("__ts__", "__symbol__", "side_name", "candidate_id") if name in source and name in outcomes]
        if len(keys) < 3:
            raise ValueError("--outcome-parquet requires at least timestamp, symbol, and side identity keys")
        outcome_columns = [name for name in outcomes.columns if name not in source.columns or any(token in str(name).lower() for token in ("target", "first_touch", "u_policy", "ret_net", "ev_after", "mfe", "mae", "timeout", "stop", "clean_exec", "dirty_positive"))]
        merged = source.merge(outcomes[keys + list(dict.fromkeys(outcome_columns))], on=keys, how="left", validate="one_to_one", suffixes=("", "__outcome"))
        if len(merged) != len(source):
            raise RuntimeError("Outcome join changed source row cardinality")
        source = merged
    if args.feature_dir:
        # Fetch only config fields absent from the materialised handoff.  This
        # is a point-in-time lookup keyed by __ts__/__symbol__, never a merge
        # on future rows.  It lets a compact candidate ledger expose the full
        # layer universe without serialising a multi-gigabyte duplicate first.
        requested = [name for name in config_feature_universe(args.layer) if name not in source]
        if requested:
            from scripts.run_materialized_trailing_label_topk_lgbm_hpo import _load_feature_store_columns
            additions, _ = _load_feature_store_columns(source, feature_dir=Path(args.feature_dir), selected_features=requested)
            if not additions.empty:
                source = pd.concat([source.reset_index(drop=True), additions.reset_index(drop=True)], axis=1, copy=False)
    ts = _timestamp(source)
    end = pd.Timestamp(args.train_end, tz="UTC")
    result = source.loc[ts.notna() & (ts < end)].copy()
    result["timestamp"] = ts.loc[result.index]
    if result.empty:
        raise ValueError("No rows before --train-end")
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _cap_side_bme(frame: pd.DataFrame, *, max_rows_per_side: int) -> pd.DataFrame:
    """Deterministic beginning/middle/end cap before fetching raw features."""
    parts = []
    for _, local in frame.groupby(_side(frame), observed=True, sort=True):
        local = local.sort_values("timestamp" if "timestamp" in local else "__ts__", kind="stable")
        if len(local) <= max_rows_per_side:
            parts.append(local); continue
        bounds = np.array_split(np.arange(len(local)), 3)
        counts = [max_rows_per_side // 3 + (index < max_rows_per_side % 3) for index in range(3)]
        take = np.concatenate([band[np.linspace(0, len(band) - 1, count, dtype=np.int32)] for band, count in zip(bounds, counts)])
        parts.append(local.iloc[np.sort(take)])
    return pd.concat(parts, ignore_index=True, copy=False)


def _load_label_reference_columns(path: Path, *, shard_cap_per_side: int) -> pd.DataFrame:
    """Read only identity, labels, and evaluation fields from monthly shards.

    Avoiding the historical wide label frame is essential here: feature inputs
    are fetched only after the deterministic B/M/E cap has been applied.
    """
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No label parquets found in {path}")
    wanted = {
        "__ts__", "__symbol__", "side", "side_name", "candidate_id", "timeframe",
        "__first_touch_target_soft__", "__first_touch_capture_net__", "__u_policy_net__",
        "__archetype_label_family__", "__archetype_policy_key__", "__archetype_policy_role__",
        "__mfe__", "__mae__", "__tp__", "__sl__", "__is_timeout__", "__bars_to_mfe__",
        "__mae_ret__", "__mfe_ret__", "__first_touch_stop__", "__first_touch_timeout__",
        "__first_touch_hit__", "__first_touch_eligible__", "__first_touch_valid_path__",
    }
    frames = []
    for file in files:
        available = set(pd.read_parquet(file, columns=[]).columns) if False else None
        # pandas/pyarrow can inspect schema without decoding all columns.
        import pyarrow.parquet as pq
        columns = [name for name in pq.read_schema(file).names if name in wanted]
        part = pd.read_parquet(file, columns=columns)
        # Monthly label shards are much wider in row count than the reference
        # needs.  Deterministic time-spread rows preserve early/mid/late state
        # coverage without ever holding the full historical label frame.
        if len(part) > shard_cap_per_side:
            groups = []
            side_col = "side_name" if "side_name" in part else "side"
            for _, local in part.groupby(side_col, observed=True, sort=True):
                take = np.linspace(0, len(local) - 1, min(len(local), shard_cap_per_side), dtype=np.int32)
                groups.append(local.iloc[take])
            part = pd.concat(groups, ignore_index=True, copy=False)
        frames.append(part)
        print(f"[state_reference] sampled label shard={file.name} rows={len(part)}", flush=True)
    frame = pd.concat(frames, ignore_index=True, copy=False)
    if "__ts__" not in frame or "__symbol__" not in frame:
        raise ValueError("Label reference requires __ts__ and __symbol__")
    return frame


def _side_dir(out: Path, layer: str, side: str) -> Path:
    return out / layer / side


def run_plan(args: argparse.Namespace) -> None:
    payload = {
        "schema": SCHEMA_VERSION,
        "stage": "plan",
        "layer": args.layer,
        "train_end": args.train_end,
        "input_parquet": str(args.input_parquet),
        "base_config_feature_count": len(config_feature_universe("base")),
        "meta_config_feature_count": len(config_feature_universe("meta")),
        "contract": {
            "reference": "immutable beginning/middle/end rows per side x layer",
            "proxy": "same 50k rows per encoder/prefix candidate",
            "gmm_proxy": "diagonal K=3,5 reg=.003,.01",
            "promotion": "two encoder/prefix combinations per side x layer",
            "full": "150k rows; K=3,4,5,6; diagonal/tied; reg=.001/.003/.01/.03",
            "bhattacharyya": "top three distinct-K diagonal candidates, inner chronological validation only",
            "outer_oos": "not used for candidate/lambda selection",
        },
    }
    _write_json(Path(args.out) / "plan.json", payload)
    print(json.dumps(payload, indent=2))


def run_prepare(args: argparse.Namespace) -> None:
    train = _load_train(args)
    layer_features = available_layer_features(train, args.layer)
    target = _target(train, args.target_column)
    net_ev = _net_ev(train, args.net_ev_column)
    if not layer_features:
        raise ValueError("No available observable config features in input parquet")
    for side in SIDES:
        local = train.loc[_side(train).eq(side)].reset_index(drop=True)
        if len(local) < 2_000:
            raise ValueError(f"{args.layer}/{side}: insufficient train rows ({len(local)})")
        destination = _side_dir(Path(args.out), args.layer, side)
        destination.mkdir(parents=True, exist_ok=True)
        stats = stable_feature_filter(local, layer_features)
        # Cheap relevance first so a correlation group retains a useful feature.
        x = local[layer_features].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(local[target], errors="coerce").fillna(0.0)
        relevance = x.corrwith(y, method="spearman").abs().fillna(0.0).to_dict()
        representatives, corr = correlation_cluster_representatives(local, stats, relevance)
        ranking = univariate_relief_mda_ranking(local, representatives, target_column=target, net_ev_column=net_ev, random_state=args.seed)
        prefixes = nested_feature_prefixes(ranking)
        stats.to_csv(destination / "feature_stability.csv", index=False)
        corr.to_csv(destination / "feature_correlation_groups.csv", index=False)
        ranking.to_csv(destination / "feature_ranking.csv", index=False)
        _write_json(destination / "feature_prefixes.json", {str(key): value for key, value in prefixes.items()})
        # Keep only required columns in the reference cache, not a duplicate full input parquet.
        # Preserve the complete screened ranking in the compact reference so
        # 75/150 representation prefixes remain possible even when only a
        # smaller subset has positive standalone MDA importance.
        useful = ranking["feature"].astype(str).tolist()
        # Path fields are evaluation-only.  They are never passed to the
        # encoder or probe feature set, but retaining them in the compact
        # reference is required to score causal path-behaviour separation.
        path_evaluation = [
            name for name in local.columns
            if any(token in str(name).lower() for token in (
                "mfe", "mae", "first_touch_hit", "first_touch_stop",
                "first_touch_timeout", "bars_to_mfe", "bars_to_mae", "__tp__", "__sl__",
            ))
        ]
        columns = list(dict.fromkeys(["timestamp", *[name for name in ("side_name", "side") if name in local], target, *([net_ev] if net_ev else []), *([_archetype(local)] if _archetype(local) else []), *path_evaluation, *useful]))
        local[columns].to_parquet(destination / "train_reference.parquet", index=False)
        _write_json(destination / "prepare_manifest.json", {
            "schema": SCHEMA_VERSION, "layer": args.layer, "side_name": side,
            "train_end": args.train_end, "train_rows": len(local), "target_column": target,
            "net_ev_column": net_ev, "archetype_column": _archetype(local),
            "config_feature_count": len(config_feature_universe(args.layer)),
            "available_observable_feature_count": len(layer_features),
            "available_observable_features": list(map(str, layer_features)),
            "unavailable_config_feature_count": len(set(config_feature_universe(args.layer)).difference(map(str, layer_features))),
            "stable_feature_count": int(stats["stable"].sum()),
            "correlation_representative_count": len(representatives), "feature_prefixes": {str(key): len(value) for key, value in prefixes.items()},
            "feature_selection_contract": "availability/stability -> corr(.90) representative -> top200 univariate + top100 Relief rescue -> chronological MDA stability",
        })
        print(f"prepared {args.layer}/{side}: rows={len(local)} configured={len(layer_features)} stable={int(stats['stable'].sum())} representatives={len(representatives)}")


def _candidate_space(prefix_size: int) -> Iterable[tuple[str, int]]:
    dimensions = (12, 16) if prefix_size <= 75 else (16, 24)
    # Legacy sklearn DAE has only bottlenecks 8/16; avoid falsely labeling a 12d model DAE.
    for dim in dimensions:
        if dim == 16:
            yield "dae", dim
        yield "masked", dim


def _fit_panel(local: pd.DataFrame, prefix: list[str], config: SearchConfig, *, rows: int, epochs: int, specs: Iterable[tuple[int, str, float]], target: str, net_ev: str | None, archetype: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    objects: dict[str, Any] = {}
    for kind, latent_dim in _candidate_space(len(prefix)):
        encoder, latent, indices = fit_encoder(local, prefix, config, rows=rows, kind=kind, latent_dim=latent_dim, epochs=epochs)
        scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(latent)
        scaled = scaler.transform(latent).astype(np.float32)
        for components, covariance, reg in specs:
            candidate_id = f"{kind}_p{len(prefix)}_z{latent_dim}_k{components}_{covariance}_r{reg:g}"
            gmm = fit_gmm(scaled, components=components, covariance_type=covariance, reg_covar=reg, random_state=config.random_state)
            metrics = evaluate_density_candidate(local.iloc[indices].reset_index(drop=True), encoder=encoder, scaler=scaler, gmm=gmm, features=prefix, target_column=target, net_ev_column=net_ev, archetype_column=archetype, random_state=config.random_state)
            record = {"candidate_id": candidate_id, "encoder_kind": kind, "latent_dim": latent_dim, "prefix_size": len(prefix), "components": components, "covariance_type": covariance, "reg_covar": reg, **metrics}
            records.append(record)
            objects[candidate_id] = {"encoder": encoder, "scaler": scaler, "gmm": gmm, "indices": indices}
    return records, objects


def run_proxy(args: argparse.Namespace) -> None:
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        local = _read(destination / "train_reference.parquet")
        manifest = json.loads((destination / "prepare_manifest.json").read_text())
        prefixes = {int(key): value for key, value in json.loads((destination / "feature_prefixes.json").read_text()).items()}
        config = SearchConfig(layer=args.layer, side_name=side, random_state=args.seed)
        records, objects = [], {}
        seen_prefixes: set[tuple[str, ...]] = set()
        for _, prefix in prefixes.items():
            if len(prefix) < 8:
                continue
            key = tuple(prefix)
            if key in seen_prefixes:
                continue
            seen_prefixes.add(key)
            result, fitted = _fit_panel(local, prefix, config, rows=min(config.proxy_rows, len(local)), epochs=config.proxy_epochs, specs=config.proxy_gmm_specs, target=manifest["target_column"], net_ev=manifest.get("net_ev_column"), archetype=manifest.get("archetype_column"))
            records += result; objects.update(fitted)
        ranked = score_candidates(pd.DataFrame(records))
        ranked.to_csv(destination / "proxy_candidates.csv", index=False)
        joblib.dump(objects, destination / "proxy_objects.joblib", compress=3)
        promoted = ranked.drop_duplicates(["encoder_kind", "prefix_size"], keep="first").head(2)
        promoted.to_csv(destination / "promoted_proxy_candidates.csv", index=False)
        print(f"proxy {args.layer}/{side}: candidates={len(ranked)} promoted={len(promoted)}")


def run_full(args: argparse.Namespace) -> None:
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        local = _read(destination / "train_reference.parquet")
        manifest = json.loads((destination / "prepare_manifest.json").read_text())
        prefixes = {int(key): value for key, value in json.loads((destination / "feature_prefixes.json").read_text()).items()}
        promoted = pd.read_csv(destination / "promoted_proxy_candidates.csv")
        config = SearchConfig(layer=args.layer, side_name=side, random_state=args.seed)
        records, objects = [], {}
        for row in promoted.itertuples(index=False):
            prefix = prefixes[int(row.prefix_size)]
            kind, latent_dim = str(row.encoder_kind), int(row.latent_dim)
            encoder, latent, indices = fit_encoder(local, prefix, config, rows=min(config.final_rows, len(local)), kind=kind, latent_dim=latent_dim, epochs=config.final_epochs)
            scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(latent)
            scaled = scaler.transform(latent).astype(np.float32)
            for components in config.final_components:
                for covariance in config.final_covariance_types:
                    for reg in config.final_reg_covars:
                        candidate_id = f"{kind}_p{len(prefix)}_z{latent_dim}_k{components}_{covariance}_r{reg:g}"
                        gmm = fit_gmm(scaled, components=components, covariance_type=covariance, reg_covar=reg, random_state=config.random_state)
                        metrics = evaluate_density_candidate(local.iloc[indices].reset_index(drop=True), encoder=encoder, scaler=scaler, gmm=gmm, features=prefix, target_column=manifest["target_column"], net_ev_column=manifest.get("net_ev_column"), archetype_column=manifest.get("archetype_column"), random_state=config.random_state)
                        records.append({"candidate_id": candidate_id, "encoder_kind": kind, "latent_dim": latent_dim, "prefix_size": len(prefix), "components": components, "covariance_type": covariance, "reg_covar": reg, **metrics})
                        objects[candidate_id] = {"encoder": encoder, "scaler": scaler, "gmm": gmm, "features": prefix, "indices": indices}
        ranked = score_candidates(pd.DataFrame(records))
        ranked.to_csv(destination / "full_candidates.csv", index=False)
        joblib.dump(objects, destination / "full_objects.joblib", compress=3)
        print(f"full {args.layer}/{side}: candidates={len(ranked)}")


def run_refine(args: argparse.Namespace) -> None:
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        ranked = pd.read_csv(destination / "full_candidates.csv")
        objects = joblib.load(destination / "full_objects.joblib")
        local = _read(destination / "train_reference.parquet")
        manifest = json.loads((destination / "prepare_manifest.json").read_text())
        config = SearchConfig(layer=args.layer, side_name=side, random_state=args.seed)
        candidates = []
        for row in ranked.itertuples(index=False):
            obj = objects[str(row.candidate_id)]
            candidates.append({"candidate_id": row.candidate_id, "components": row.components, "covariance_type": row.covariance_type, "gmm": obj["gmm"]})
        # The lambda evaluator has no outer OOS access. It evaluates the last inner quarter only.
        def evaluate(refined_state: dict[str, Any], inner: pd.DataFrame) -> float:
            chosen = next(item for item in candidates if item["candidate_id"] == current["candidate_id"])
            obj = objects[str(chosen["candidate_id"])]
            encoder, scaler, features = obj["encoder"], obj["scaler"], obj["features"]
            values = inner[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            native = encoder.transform_native(values, sides=np.repeat(side, len(values)))
            latent = scaler.transform(native.latent)
            # This score is deliberately local: OOS is not read during lambda selection.
            stats = __import__("extreme_price_movements.side_local_ae_gmm_search", fromlist=["_gmm_statistics"])._gmm_statistics(refined_state, latent)
            target = pd.to_numeric(inner[manifest["target_column"]], errors="coerce").fillna(0.0).to_numpy()
            signal = pd.to_numeric(inner[manifest["net_ev_column"]], errors="coerce").fillna(0.0).to_numpy() if manifest.get("net_ev_column") in inner else target
            rank = stats["posterior"].max(axis=1) - stats["entropy"]
            take = max(1, int(np.ceil(.10 * len(rank))))
            return float(np.mean(signal[np.argsort(rank)[-take:]]))
        inner = local.iloc[max(0, int(.75 * len(local))):].reset_index(drop=True)
        all_records = []
        selected_for_refinement = []
        seen_k: set[int] = set()
        for candidate in candidates:
            if str(candidate["covariance_type"]) != "diag" or int(candidate["components"]) in seen_k:
                continue
            selected_for_refinement.append(candidate)
            seen_k.add(int(candidate["components"]))
            if len(selected_for_refinement) == 3:
                break
        # Bind candidate in a small closure; each ladder uses its own raw latent sample.
        for current in selected_for_refinement:
            obj = objects[str(current["candidate_id"])]
            values = local.iloc[obj["indices"]][obj["features"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            latent = obj["scaler"].transform(obj["encoder"].transform_native(values, sides=np.repeat(side, len(values))).latent)
            all_records += refine_top_diagonal_candidates(latent, [current], inner, evaluate=evaluate, config=config)
        pd.DataFrame(all_records).to_csv(destination / "bhattacharyya_inner_validation.csv", index=False)
        print(f"refine {args.layer}/{side}: rows={len(all_records)}")


def _winner_row(destination: Path) -> pd.Series:
    ranked = pd.read_csv(destination / "full_candidates.csv")
    if ranked.empty:
        raise ValueError(f"No full density candidates available under {destination}")
    return ranked.sort_values("candidate_score", ascending=False, kind="stable").iloc[0]


def _identity_hash(frame: pd.DataFrame) -> str:
    columns = [name for name in ("__ts__", "__symbol__", "side_name", "side", "candidate_id") if name in frame]
    if not columns:
        return ""
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _inference_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return the minimum outcome-free feature frame used to validate a package."""
    side_columns = [name for name in ("side_name", "side") if name in frame]
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise ValueError(f"Final refit source lacks selected features: {missing[:8]}")
    return frame[[*side_columns, *feature_names]].copy()


def run_final_refit(args: argparse.Namespace) -> None:
    """Select promoted geometries and serialize packages on 150k B/M/E rows.

    Proxy promotion is frozen from the immutable 50k reference.  This stage is
    the only place where promoted encoder/prefix pairs are refit and their full
    GMM grids compared.  It therefore cannot accidentally select a geometry on
    proxy-scale density fits and package it at a different scale.
    """
    if not args.labels_path:
        raise ValueError("final-refit currently requires --labels-path for causal PTI base features")
    if args.layer != "base":
        raise ValueError("final-refit is currently implemented for the base layer only")
    promoted_by_side: dict[str, pd.DataFrame] = {}
    prefix_by_side: dict[str, dict[int, list[str]]] = {}
    selected_features: dict[str, list[str]] = {}
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        prefixes = {int(key): value for key, value in json.loads((destination / "feature_prefixes.json").read_text()).items()}
        promoted = pd.read_csv(destination / "promoted_proxy_candidates.csv")
        if promoted.empty:
            raise ValueError(f"{args.layer}/{side}: no promoted proxy candidates")
        for prefix_size in promoted["prefix_size"].astype(int).unique():
            if prefix_size not in prefixes:
                raise KeyError(f"{args.layer}/{side}: promoted proxy requests missing prefix {prefix_size}")
        promoted_by_side[side] = promoted
        prefix_by_side[side] = prefixes
        selected_features[side] = list(dict.fromkeys(
            name for prefix_size in promoted["prefix_size"].astype(int)
            for name in prefixes[prefix_size]
        ))
    requested = list(dict.fromkeys([name for side in SIDES for name in selected_features[side]]))
    train = _load_label_features(
        args,
        requested=requested,
        max_rows_per_side=int(args.final_refit_rows),
        scratch_name=f"{args.layer}_final_refit",
    )
    from extreme_price_movements.side_local_ae_gmm_search import _density_quality, _gmm_statistics
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        local = train.loc[_side(train).eq(side)].sort_values("timestamp", kind="stable").reset_index(drop=True)
        if len(local) < int(args.final_refit_rows):
            raise ValueError(
                f"{args.layer}/{side}: final refit requires {args.final_refit_rows} B/M/E rows, got {len(local)}. "
                "Increase --label-shard-cap-per-side without changing the frozen winner."
            )
        config = SearchConfig(layer=args.layer, side_name=side, random_state=args.seed)
        from sklearn.preprocessing import StandardScaler
        selection_manifest = json.loads((destination / "prepare_manifest.json").read_text())
        records: list[dict[str, Any]] = []
        objects: dict[str, dict[str, Any]] = {}
        for promoted in promoted_by_side[side].itertuples(index=False):
            features = list(map(str, prefix_by_side[side][int(promoted.prefix_size)]))
            encoder, latent, indices = fit_encoder(
                local,
                features,
                config,
                rows=int(args.final_refit_rows),
                kind=str(promoted.encoder_kind),
                latent_dim=int(promoted.latent_dim),
                epochs=config.final_epochs,
            )
            scaler = StandardScaler().fit(latent)
            scaled = scaler.transform(latent).astype(np.float32)
            for components in config.final_components:
                for covariance in config.final_covariance_types:
                    for reg in config.final_reg_covars:
                        candidate_id = f"{promoted.encoder_kind}_p{len(features)}_z{int(promoted.latent_dim)}_k{components}_{covariance}_r{reg:g}"
                        gmm = fit_gmm(scaled, components=components, covariance_type=covariance, reg_covar=reg, random_state=config.random_state)
                        metrics = evaluate_density_candidate(
                            local.iloc[indices].reset_index(drop=True), encoder=encoder, scaler=scaler, gmm=gmm,
                            features=features, target_column=selection_manifest["target_column"],
                            net_ev_column=selection_manifest.get("net_ev_column"),
                            archetype_column=selection_manifest.get("archetype_column"), random_state=config.random_state,
                        )
                        records.append({
                            "candidate_id": candidate_id, "encoder_kind": str(promoted.encoder_kind),
                            "latent_dim": int(promoted.latent_dim), "prefix_size": len(features),
                            "components": components, "covariance_type": covariance, "reg_covar": reg, **metrics,
                        })
                        objects[candidate_id] = {
                            "encoder": encoder, "scaler": scaler, "gmm": gmm, "features": features,
                            "indices": indices, "scaled": scaled,
                        }
        ranked = score_candidates(pd.DataFrame(records))
        ranked.to_csv(destination / "final_refit_candidates.csv", index=False)
        # Refinement is deliberately selected only on the last inner quarter
        # of the same 150k reference.  It is diagnostic unless it clears a
        # later explicit promotion test; the serialized package below remains
        # the unpenalized density winner.
        inner = local.iloc[max(0, int(.75 * len(local))):].reset_index(drop=True)
        refinement_records: list[dict[str, Any]] = []
        seen_components: set[int] = set()
        for candidate in ranked.itertuples(index=False):
            if str(candidate.covariance_type) != "diag" or int(candidate.components) in seen_components:
                continue
            seen_components.add(int(candidate.components))
            candidate_id = str(candidate.candidate_id)
            candidate_obj = objects[candidate_id]

            def _inner_score(refined_state: dict[str, Any], frame: pd.DataFrame, *, _obj: dict[str, Any] = candidate_obj) -> float:
                values = frame[_obj["features"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                latent_inner = _obj["scaler"].transform(
                    _obj["encoder"].transform_native(values, sides=np.repeat(side, len(values))).latent
                )
                stats_inner = _gmm_statistics(refined_state, latent_inner)
                rank = stats_inner["posterior"].max(axis=1) - stats_inner["entropy"]
                signal_column = selection_manifest.get("net_ev_column")
                signal = (
                    pd.to_numeric(frame[signal_column], errors="coerce").fillna(0.0).to_numpy()
                    if signal_column in frame else
                    pd.to_numeric(frame[selection_manifest["target_column"]], errors="coerce").fillna(0.0).to_numpy()
                )
                take = max(1, int(np.ceil(.10 * len(rank))))
                return float(np.mean(signal[np.argsort(rank)[-take:]]))

            refinement_records.extend(refine_top_diagonal_candidates(
                candidate_obj["scaled"],
                [{"candidate_id": candidate_id, "components": int(candidate.components), "covariance_type": "diag", "gmm": candidate_obj["gmm"]}],
                inner,
                evaluate=_inner_score,
                config=config,
            ))
            if len(seen_components) == 3:
                break
        pd.DataFrame(refinement_records).to_csv(destination / "final_refit_bhattacharyya_inner_validation.csv", index=False)
        winner = ranked.iloc[0]
        selected_id = str(winner.candidate_id)
        obj = objects[selected_id]
        features = list(obj["features"])
        encoder, scaler, gmm, indices, scaled = obj["encoder"], obj["scaler"], obj["gmm"], obj["indices"], obj["scaled"]
        train_values = local.iloc[indices][features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        native = encoder.transform_native(train_values, sides=np.repeat(side, len(train_values)))
        stats = _gmm_statistics(gmm, scaled)
        reconstruction = native.reconstruction_error
        if reconstruction is None:
            reconstruction = np.zeros(len(scaled), dtype=np.float32)
        novelty_reference = stats["min_mahalanobis"] + np.asarray(reconstruction, dtype=np.float32)
        state = SideLocalState(
            config=config,
            feature_names=features,
            encoder_state=encoder.to_state(),
            latent_scaler=scaler,
            gmm=gmm,
            novelty_reference=novelty_reference,
            component_count=int(winner.components),
            selected_candidate_id=selected_id,
            feature_schema_hash=feature_schema_hash(features),
            metadata={
                "package_status": "final_refit_selected_geometry",
                "selection_candidate_id": selected_id,
                "selection_source": "final_refit_candidates.csv",
                "selection_train_rows": int(len(local)),
                "final_refit_rows": int(len(indices)),
                "reference_identity_hash": _identity_hash(local.iloc[indices]),
                "reference_start": str(local.iloc[indices]["timestamp"].min()),
                "reference_end": str(local.iloc[indices]["timestamp"].max()),
                "bhattacharyya_refinement": "evaluated_inner_only_not_promoted",
                "outcome_columns_used_for_fit": False,
                "inference_contract": "selected observable features + side only; outcome/path fields rejected",
            },
        )
        # A transform against an outcome-free frame is the final parity check.
        validation = state.transform(_inference_frame(local.iloc[: min(512, len(local))], features))
        if int(validation[f"{state.prefix}_valid"].sum()) == 0:
            raise RuntimeError(f"{args.layer}/{side}: final state failed outcome-free transform validation")
        package_path = destination / "final_selected_state.joblib"
        joblib.dump(state, package_path, compress=3)
        occupancy = stats["posterior"].mean(axis=0)
        final_manifest = {
            "schema": SCHEMA_VERSION,
            "package_status": "final_refit_selected_geometry",
            "layer": args.layer,
            "side_name": side,
            "selected_candidate": winner.to_dict(),
            "selected_feature_count": len(features),
            "selected_features": features,
            "feature_schema_hash": state.feature_schema_hash,
            "reference_rows": int(len(indices)),
            "reference_identity_hash": state.metadata["reference_identity_hash"],
            "reference_start": state.metadata["reference_start"],
            "reference_end": state.metadata["reference_end"],
            "gmm_component_occupancy": occupancy.tolist(),
            "density_quality": float(_density_quality(gmm, scaled)),
            "transform_validation_rows": int(len(validation)),
            "transform_validation_valid_rows": int(validation[f"{state.prefix}_valid"].sum()),
            "bhattacharyya_refinement": "unpenalized incumbent retained; run separate inner-only refinement before promoting a penalized state",
            "training_contract": "time-stratified B/M/E final refit; no outcomes supplied to encoder/scaler/GMM fit",
            "inference_excluded_columns": "target/outcome/realized/path columns rejected before transform",
            "package_path": str(package_path),
        }
        _write_json(destination / "final_refit_manifest.json", final_manifest)
        print(
            f"final-refit {args.layer}/{side}: rows={len(indices)} features={len(features)} "
            f"components={int(winner.components)} package={package_path}",
            flush=True,
        )


def _load_cached_final_reference(args: argparse.Namespace) -> pd.DataFrame:
    cache = Path(args.out) / "_reference_feature_batches" / f"{args.layer}_final_refit"
    identity_path = cache / "reference_identity.parquet"
    batches = sorted(cache.glob("features_*.parquet"))
    if not identity_path.exists() or not batches:
        raise FileNotFoundError("feature-ablation requires the completed final-refit cached reference batches")
    identity = pd.read_parquet(identity_path)
    additions = [pd.read_parquet(path) for path in batches]
    frame = pd.concat([identity.reset_index(drop=True), *[part.reset_index(drop=True) for part in additions]], axis=1, copy=False)
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def run_feature_ablation(args: argparse.Namespace) -> None:
    """Evaluate fixed-geometry 100/125 MDA prefixes on the final reference.

    Only the number of already-ranked observable features changes.  Encoder
    kind, latent size, GMM geometry, regularization, rows, seed, labels, and
    inner chronological evaluation remain identical to the selected package.
    """
    if args.layer != "base":
        raise ValueError("feature-ablation is currently implemented for the base layer only")
    sizes = tuple(sorted({int(value) for value in str(args.feature_ablation_prefixes).split(",") if value.strip()}))
    if not sizes or min(sizes) < 8:
        raise ValueError("--feature-ablation-prefixes requires valid positive feature counts")
    train = _load_cached_final_reference(args)
    from extreme_price_movements.side_local_ae_gmm_search import _density_quality, _gmm_statistics
    from sklearn.preprocessing import StandardScaler
    records: list[dict[str, Any]] = []
    output = Path(args.out) / args.layer / "fixed_geometry_feature_count_ablation.csv"
    checkpoint = output.with_name("fixed_geometry_feature_count_ablation_checkpoint.csv")
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        manifest = json.loads((destination / "final_refit_manifest.json").read_text())
        baseline = manifest["selected_candidate"]
        ranking = pd.read_csv(destination / "feature_ranking.csv")
        order = "selection_score" if "selection_score" in ranking else "mda_stable_importance"
        ranked_features = ranking.sort_values(order, ascending=False, kind="stable")["feature"].astype(str).tolist()
        local = train.loc[_side(train).eq(side)].reset_index(drop=True)
        if len(local) < int(args.final_refit_rows):
            raise ValueError(f"{side}: cached final reference has {len(local)} rows, needs {args.final_refit_rows}")
        selection_manifest = json.loads((destination / "prepare_manifest.json").read_text())
        config = SearchConfig(layer=args.layer, side_name=side, random_state=args.seed)
        baseline_table = pd.read_csv(destination / "final_refit_candidates.csv")
        baseline_metrics = baseline_table.loc[baseline_table["candidate_id"].eq(str(baseline["candidate_id"]))].iloc[0]
        for size in sizes:
            features = ranked_features[:size]
            if len(features) != size:
                raise ValueError(f"{side}: ranking exposes only {len(features)} features, cannot run prefix {size}")
            missing = [name for name in features if name not in local]
            if missing:
                raise KeyError(f"{side}/p{size}: cached reference missing ranked features: {missing[:8]}")
            encoder, latent, indices = fit_encoder(
                local, features, config, rows=int(args.final_refit_rows),
                kind=str(baseline["encoder_kind"]), latent_dim=int(baseline["latent_dim"]), epochs=config.final_epochs,
            )
            scaler = StandardScaler().fit(latent)
            scaled = scaler.transform(latent).astype(np.float32)
            gmm = fit_gmm(
                scaled, components=int(baseline["components"]), covariance_type=str(baseline["covariance_type"]),
                reg_covar=float(baseline["reg_covar"]), random_state=config.random_state,
            )
            metrics = evaluate_density_candidate(
                local.iloc[indices].reset_index(drop=True), encoder=encoder, scaler=scaler, gmm=gmm, features=features,
                target_column=selection_manifest["target_column"], net_ev_column=selection_manifest.get("net_ev_column"),
                archetype_column=selection_manifest.get("archetype_column"), random_state=config.random_state,
            )
            train_values = local.iloc[indices][features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            native = encoder.transform_native(train_values, sides=np.repeat(side, len(train_values)))
            reconstruction = native.reconstruction_error
            if reconstruction is None:
                reconstruction = np.zeros(len(scaled), dtype=np.float32)
            state_stats = _gmm_statistics(gmm, scaled)
            state = SideLocalState(
                config=config, feature_names=features, encoder_state=encoder.to_state(), latent_scaler=scaler, gmm=gmm,
                novelty_reference=state_stats["min_mahalanobis"] + np.asarray(reconstruction, dtype=np.float32),
                component_count=int(baseline["components"]), selected_candidate_id=f"{baseline['candidate_id']}__p{size}",
                feature_schema_hash=feature_schema_hash(features),
                metadata={"package_status": "feature_count_ablation", "parent_candidate_id": baseline["candidate_id"], "only_changed_parameter": "mda_feature_prefix_size", "feature_prefix_size": size},
            )
            validation = state.transform(_inference_frame(local.iloc[:512], features))
            if int(validation[f"{state.prefix}_valid"].sum()) != len(validation):
                raise RuntimeError(f"{side}/p{size}: outcome-free transform validation failed")
            package = destination / f"feature_ablation_p{size}_state.joblib"
            joblib.dump(state, package, compress=3)
            row = {
                "side_name": side, "feature_prefix_size": size, "parent_candidate_id": baseline["candidate_id"],
                "encoder_kind": baseline["encoder_kind"], "latent_dim": int(baseline["latent_dim"]),
                "components": int(baseline["components"]), "covariance_type": baseline["covariance_type"], "reg_covar": float(baseline["reg_covar"]),
                "density_quality": float(_density_quality(gmm, scaled)), "package_path": str(package), **metrics,
            }
            for field in ("incremental_top10_ev", "incremental_top20_ev", "archetype_rank_gain", "path_separation", "stability", "raw_top10_ev", "state_top10_ev", "raw_top20_ev", "state_top20_ev"):
                if field in baseline_metrics and field in row:
                    row[f"delta_vs_selected_{field}"] = float(row[field]) - float(baseline_metrics[field])
            records.append(row)
            # Persist each arm independently.  Long DAE/AE fits can take
            # minutes; a later interruption must not discard completed arms.
            persisted = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
            persisted = pd.concat([persisted, pd.DataFrame([row])], ignore_index=True, copy=False)
            persisted = persisted.drop_duplicates(["side_name", "feature_prefix_size"], keep="last")
            persisted.to_csv(checkpoint, index=False)
            print(f"feature-ablation {side}/p{size}: top10={row['state_top10_ev']:.6f} top20={row['state_top20_ev']:.6f}", flush=True)
    # Include any persisted arms from an interrupted prior process.
    persisted = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    pd.concat([persisted, pd.DataFrame(records)], ignore_index=True, copy=False).drop_duplicates(
        ["side_name", "feature_prefix_size"], keep="last"
    ).to_csv(output, index=False)
    _write_json(output.with_suffix(".manifest.json"), {
        "schema": SCHEMA_VERSION, "layer": args.layer, "feature_prefix_sizes": list(sizes),
        "contract": "same frozen 150k reference, selected encoder/GMM geometry, seed, and inner evaluation; only MDA feature prefix changes",
    })
    print(f"feature-ablation complete: {output}")


def run_materialize(args: argparse.Namespace) -> None:
    if not args.materialize_parquet:
        raise ValueError("materialize requires --materialize-parquet")
    source = _read(Path(args.materialize_parquet))
    blocks = []
    manifests = []
    for side in SIDES:
        destination = _side_dir(Path(args.out), args.layer, side)
        final_package = destination / "final_selected_state.joblib"
        values = source.drop(columns=[column for column in source.columns if any(token in str(column).lower() for token in ("target", "outcome", "realized", "mfe", "mae", "timeout", "stop", "pnl", "profit"))], errors="ignore")
        if final_package.exists():
            # Once a 150k selected-geometry package exists it is the canonical
            # state source.  Never fall back silently to a 50k search object.
            state = joblib.load(final_package)
            blocks.append(state.transform(values))
            manifests.append(state.manifest())
            continue
        ranked = pd.read_csv(destination / "full_candidates.csv")
        objects = joblib.load(destination / "full_objects.joblib")
        # Prefer inner-selected lambda only if the winning lambda is positive and is reproducibly recorded.
        selected_id = str(ranked.iloc[0].candidate_id)
        obj = objects[selected_id]
        config = SearchConfig(layer=args.layer, side_name=side, random_state=args.seed)
        # Novelty reference is fit-only and never recomputed from materialisation rows.
        train_reference = _read(destination / "train_reference.parquet")
        train_values = train_reference[obj["features"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        train_native = obj["encoder"].transform_native(train_values, sides=np.repeat(side, len(train_values)))
        train_latent = obj["scaler"].transform(train_native.latent)
        from extreme_price_movements.side_local_ae_gmm_search import _gmm_statistics  # local/private by design
        novelty_reference = _gmm_statistics(obj["gmm"], train_latent)["min_mahalanobis"]
        # Construct through the public transformer; source is stripped of outcomes first.
        state = SideLocalState(config=config, feature_names=list(obj["features"]), encoder_state=obj["encoder"].to_state(), latent_scaler=obj["scaler"], gmm=obj["gmm"], novelty_reference=novelty_reference, component_count=int(ranked.iloc[0].components), selected_candidate_id=selected_id, feature_schema_hash=feature_schema_hash(obj["features"]), metadata={"research_only": True, "side_local_component_ids": True, "novelty_reference_source": "train_reference_only"})
        blocks.append(state.transform(values))
        manifests.append(state.manifest())
        joblib.dump(state, destination / "selected_state.joblib", compress=3)
    sidecar = pd.concat(blocks, axis=1)
    output = Path(args.out) / args.layer / "materialized_side_local_state_features.parquet"
    sidecar.to_parquet(output, index=False)
    _write_json(output.with_suffix(".manifest.json"), {"schema": SCHEMA_VERSION, "research_only": True, "source": str(args.materialize_parquet), "states": manifests, "inference_excluded_columns": "target/outcome/realized/path columns rejected before transform"})
    print(f"materialized {output}: rows={len(sidecar)} cols={len(sidecar.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--layer", choices=("base", "meta"), required=True)
    parser.add_argument("--input-parquet", help="Point-in-time feature parquet for the selected layer")
    parser.add_argument("--labels-path", help="Base-only causal labels directory/file; uses a B/M/E reference then point-in-time feature lookup")
    parser.add_argument("--train-end", required=True, help="Exclusive UTC train boundary; outer OOS must be later")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-column")
    parser.add_argument("--net-ev-column")
    parser.add_argument("--materialize-parquet")
    parser.add_argument("--feature-dir", help="Point-in-time feature store; required with --labels-path, optional with --input-parquet")
    parser.add_argument("--outcome-parquet", help="Optional aligned scored ledger supplying targets/path outcomes to a feature-only meta handoff")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--reference-cap-per-side", type=int, default=160000)
    parser.add_argument("--final-refit-rows", type=int, default=150000, help="Frozen selected-geometry B/M/E refit rows per side; selection is never rerun")
    parser.add_argument("--feature-ablation-prefixes", default="100,125", help="Frozen MDA prefix counts for the fixed-geometry feature ablation")
    parser.add_argument("--label-shard-cap-per-side", type=int, default=8000)
    parser.add_argument("--feature-load-batch", type=int, default=8)
    parser.add_argument("--reuse-feature-batches", action=argparse.BooleanOptionalAction, default=True, help="Reuse identity-verified point-in-time feature batches during a deterministic preparation rebuild")
    args = parser.parse_args()
    if not args.input_parquet and not args.labels_path:
        parser.error("one of --input-parquet or --labels-path is required")
    if args.labels_path and not args.feature_dir:
        parser.error("--labels-path requires --feature-dir")
    if args.stage == "plan": run_plan(args); return
    stages = ("prepare", "proxy", "full", "refine") if args.stage == "all" else (args.stage,)
    for stage in stages:
        {
            "prepare": run_prepare,
            "proxy": run_proxy,
            "full": run_full,
            "refine": run_refine,
            "final-refit": run_final_refit,
            "feature-ablation": run_feature_ablation,
            "materialize": run_materialize,
        }[stage](args)


if __name__ == "__main__":
    main()
