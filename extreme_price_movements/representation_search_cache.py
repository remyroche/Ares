"""Immutable, memory-mapped inputs for alternative representation searches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features_gmm_ae import (
    ae_gmm_cycle_reference_indices,
    ae_gmm_cycle_sample_identity_hash,
)

KEY_COLUMNS = ("__ts__", "__symbol__", "side")
FEATURE_GROUPS = (
    "price_trend",
    "volatility",
    "volume",
    "oi_funding",
    "residuals",
    "market_context",
    "regime_source",
    "side",
)


@dataclass(frozen=True)
class ReferenceCacheManifest:
    schema: str
    source_rows: int
    reference_rows: int
    feature_count: int
    feature_names: tuple[str, ...]
    feature_groups: dict[str, tuple[int, ...]]
    row_identity_hash: str
    feature_value_hash: str
    sample_indices: dict[str, str]
    scaler_fit_sample: str
    clip_bounds: tuple[float, float]
    raw_path: str
    scaled_path: str
    keys_path: str
    center_path: str
    scale_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_feature_groups(feature_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Assign every observable input to one stable corruption group."""

    groups: dict[str, list[int]] = {name: [] for name in FEATURE_GROUPS}
    for index, feature in enumerate(map(str, feature_names)):
        low = feature.lower()
        if low in {"side", "side_name", "__side__"}:
            group = "side"
        elif "regime_source" in low or low.startswith("__regime_"):
            group = "regime_source"
        elif any(token in low for token in ("funding", "open_interest", "oi_", "_oi", "leverage")):
            group = "oi_funding"
        elif any(token in low for token in ("volume", "volu", "quote_volume", "flow", "trade_size")):
            group = "volume"
        elif any(token in low for token in ("volatility", "atr", "_rv", "rvol", "vol_of_vol", "range_")):
            group = "volatility"
        elif any(token in low for token in ("resid", "peer", "idiosyn", "innovation")):
            group = "residuals"
        elif any(token in low for token in ("market", "btc", "eth", "xasset", "breadth", "dispersion", "corr_")):
            group = "market_context"
        else:
            group = "price_trend"
        groups[group].append(index)
    return {name: tuple(values) for name, values in groups.items()}


def _hash_feature_values(values: np.ndarray, feature_names: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(map(str, feature_names)).encode("utf-8"))
    matrix = np.asarray(values, dtype=np.float32, order="C")
    row_hashes = pd.util.hash_pandas_object(
        pd.DataFrame(matrix, columns=list(map(str, feature_names))), index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _time_spread_local_indices(n_rows: int, max_rows: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    bands = np.array_split(np.arange(n_rows, dtype=np.int64), 3)
    base, remainder = divmod(int(max_rows), 3)
    selected: list[np.ndarray] = []
    for band_index, band in enumerate(bands):
        take = min(len(band), base + (1 if band_index < remainder else 0))
        if take:
            selected.append(band[np.linspace(0, len(band) - 1, take, dtype=np.int64)])
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def prepare_reference_cache(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    output_dir: Path,
    reference_rows: int = 300_000,
    scaler_rows: int = 100_000,
    sample_sizes: Sequence[int] = (15_000, 100_000, 150_000, 250_000, 300_000),
    clip_bounds: tuple[float, float] = (-8.0, 8.0),
) -> ReferenceCacheManifest:
    """Transform observable inputs once and persist exact reusable row contracts."""

    required = [*KEY_COLUMNS, *map(str, feature_names)]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Reference cache is missing columns: {missing[:20]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_indices = ae_gmm_cycle_reference_indices(
        frame["__ts__"],
        symbols=frame["__symbol__"],
        sides=frame["side"],
        max_rows=int(reference_rows),
    )
    reference = frame.iloc[reference_indices]
    keys = reference.loc[:, KEY_COLUMNS].copy()
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="coerce")
    keys["__symbol__"] = keys["__symbol__"].astype(str)
    keys["side"] = pd.to_numeric(keys["side"], errors="coerce").astype(np.int8)
    if bool(keys.isna().any(axis=None)) or bool(keys.duplicated(list(KEY_COLUMNS)).any()):
        raise ValueError("Reference cache keys must be finite and unique")
    raw_frame = reference.loc[:, list(map(str, feature_names))].apply(
        pd.to_numeric, errors="coerce"
    )
    scaler_idx = _time_spread_local_indices(len(reference), int(scaler_rows))
    scaler_frame = raw_frame.iloc[scaler_idx]
    center = scaler_frame.median(axis=0, skipna=True).fillna(0.0).to_numpy(np.float32)
    q75 = scaler_frame.quantile(0.75).to_numpy(np.float32)
    q25 = scaler_frame.quantile(0.25).to_numpy(np.float32)
    scale = np.maximum(q75 - q25, 1e-6).astype(np.float32, copy=False)
    raw = raw_frame.to_numpy(dtype=np.float32, copy=False)
    invalid = ~np.isfinite(raw)
    if bool(invalid.any()):
        raw = raw.copy()
        raw[invalid] = np.broadcast_to(center, raw.shape)[invalid]
    scaled = np.clip(
        (raw - center.reshape(1, -1)) / scale.reshape(1, -1),
        float(clip_bounds[0]),
        float(clip_bounds[1]),
    ).astype(np.float32, copy=False)
    raw_path = output_dir / "reference_raw.npy"
    scaled_path = output_dir / "reference_scaled.npy"
    center_path = output_dir / "robust_center.npy"
    scale_path = output_dir / "robust_scale.npy"
    keys_path = output_dir / "reference_keys.parquet"
    np.save(raw_path, raw, allow_pickle=False)
    np.save(scaled_path, scaled, allow_pickle=False)
    np.save(center_path, center, allow_pickle=False)
    np.save(scale_path, scale, allow_pickle=False)
    keys.reset_index(drop=True).to_parquet(keys_path, index=False)
    samples: dict[str, str] = {}
    for size in sorted(set(map(int, sample_sizes))):
        local = _time_spread_local_indices(len(reference), size)
        path = output_dir / f"sample_indices_{size}.npy"
        np.save(path, local, allow_pickle=False)
        samples[str(size)] = str(path)
    manifest = ReferenceCacheManifest(
        schema="alternative_representation_reference_cache_v1",
        source_rows=int(len(frame)),
        reference_rows=int(len(reference)),
        feature_count=int(len(feature_names)),
        feature_names=tuple(map(str, feature_names)),
        feature_groups=infer_feature_groups(feature_names),
        row_identity_hash=ae_gmm_cycle_sample_identity_hash(
            frame["__ts__"],
            symbols=frame["__symbol__"],
            sides=frame["side"],
            indices=reference_indices,
        ),
        feature_value_hash=_hash_feature_values(raw, feature_names),
        sample_indices=samples,
        scaler_fit_sample=str(output_dir / f"sample_indices_{int(scaler_rows)}.npy"),
        clip_bounds=(float(clip_bounds[0]), float(clip_bounds[1])),
        raw_path=str(raw_path),
        scaled_path=str(scaled_path),
        keys_path=str(keys_path),
        center_path=str(center_path),
        scale_path=str(scale_path),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_reference_cache(path: Path) -> tuple[ReferenceCacheManifest, np.ndarray, pd.DataFrame]:
    root = Path(path)
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload["feature_names"] = tuple(payload["feature_names"])
    payload["feature_groups"] = {
        str(name): tuple(map(int, values))
        for name, values in payload["feature_groups"].items()
    }
    payload["clip_bounds"] = tuple(map(float, payload["clip_bounds"]))
    manifest = ReferenceCacheManifest(**payload)
    scaled = np.load(manifest.scaled_path, mmap_mode="r")
    keys = pd.read_parquet(manifest.keys_path)
    return manifest, scaled, keys


def cached_side_conditioned_donor_map(
    sides: Sequence[Any],
    *,
    seed: int,
    output_path: Path | None = None,
) -> np.ndarray:
    """Create a deterministic donor permutation that never crosses side."""

    side_values = np.asarray(sides).reshape(-1)
    rng = np.random.default_rng(int(seed))
    donors = np.arange(len(side_values), dtype=np.int64)
    for side in np.unique(side_values):
        positions = np.flatnonzero(side_values == side)
        if len(positions) > 1:
            donors[positions] = rng.permutation(positions)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, donors, allow_pickle=False)
    return donors


def apply_group_corruption(
    values: np.ndarray,
    *,
    group_indices: Sequence[int],
    donor_indices: np.ndarray,
) -> np.ndarray:
    """Jointly replace one feature group while preserving within-group geometry."""

    source = np.asarray(values, dtype=np.float32)
    donors = np.asarray(donor_indices, dtype=np.int64)
    columns = np.asarray(group_indices, dtype=np.int64)
    if source.ndim != 2 or len(donors) != len(source):
        raise ValueError("Corruption values and donor map must be row-aligned")
    if np.any((donors < 0) | (donors >= len(source))):
        raise ValueError("Donor indices are out of range")
    output = source.copy()
    if len(columns):
        output[:, columns] = source[donors[:, None], columns[None, :]]
    return output


__all__ = [
    "FEATURE_GROUPS",
    "KEY_COLUMNS",
    "ReferenceCacheManifest",
    "apply_group_corruption",
    "cached_side_conditioned_donor_map",
    "infer_feature_groups",
    "load_reference_cache",
    "prepare_reference_cache",
]
