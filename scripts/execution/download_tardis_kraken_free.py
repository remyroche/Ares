#!/usr/bin/env python3
"""Resumably download only manifest-approved Tardis free sample days.

Raw files are immutable inputs.  A failed retry cannot remove or overwrite a
validated `.csv.gz`; downloaded bytes first land in a private temporary
directory and are atomically moved into the research data tree only after
gzip/integrity checks pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_gzip(path: Path) -> tuple[int, str]:
    with gzip.open(path, "rb") as handle:
        # Force the gzip CRC/trailer to be read, without retaining the payload.
        while handle.read(1024 * 1024):
            pass
    return int(path.stat().st_size), _sha256(path)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "__").replace(":", "_")


def _target_path(root: Path, *, exchange: str, data_type: str, symbol: str, sample_date: pd.Timestamp) -> Path:
    return root / "raw" / exchange / data_type / _safe_symbol(symbol) / f"{sample_date.date()}.csv.gz"


def _is_month_start(stamp: pd.Timestamp) -> bool:
    return bool(stamp.day == 1 and stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0)


def _download_one_tardis(
    *,
    exchange: str,
    data_type: str,
    symbol: str,
    sample_date: pd.Timestamp,
    temp_dir: Path,
    api_key: str | None,
    client_timeout_seconds: int,
) -> Path:
    try:
        # tardis-dev >=4 exposes the supported public function at package
        # root.  Keep the import local so manifest/dry-run use remains free of
        # this optional network dependency.
        from tardis_dev import download_datasets  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "tardis-dev is required for downloads. Install it explicitly with "
            "`python -m pip install tardis-dev`; no download was attempted."
        ) from exc
    end = sample_date + pd.Timedelta(days=1)
    download_datasets(
        exchange=exchange,
        data_types=[data_type],
        from_date=sample_date.date().isoformat(),
        to_date=end.date().isoformat(),
        symbols=[symbol],
        api_key=api_key,
        download_dir=str(temp_dir),
        timeout=int(client_timeout_seconds),
        concurrency=1,
    )
    candidates = sorted(temp_dir.rglob("*.csv.gz"), key=lambda item: item.stat().st_size, reverse=True)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one downloaded {data_type} file for {symbol}/{sample_date.date()}, found {len(candidates)}")
    return candidates[0]


def _tardis_dataset_url(*, exchange: str, data_type: str, symbol: str, sample_date: pd.Timestamp) -> str:
    """Return the documented public Tardis dataset URL for one exact file."""
    normalized = str(symbol).replace(":", "-").replace("/", "-").upper()
    return (
        f"https://datasets.tardis.dev/v1/{exchange}/{data_type}/"
        f"{sample_date.strftime('%Y/%m/%d')}/{normalized}.csv.gz"
    )


def _download_one_direct(
    *,
    exchange: str,
    data_type: str,
    symbol: str,
    sample_date: pd.Timestamp,
    temp_dir: Path,
    api_key: str | None,
    client_timeout_seconds: int,
) -> Path:
    """Bounded direct fallback for environments where aiohttp cannot connect.

    Bytes land only in the caller's private temporary directory.  The shared
    atomic promotion path validates the completed gzip before it becomes an
    immutable raw input.
    """
    destination = temp_dir / "tardis_download.csv.gz"
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--max-time", str(int(client_timeout_seconds)),
        "--header", "User-Agent: Ares-execution-research/1",
        "--header", "Accept-Encoding: gzip",
        "--output", str(destination),
        _tardis_dataset_url(exchange=exchange, data_type=data_type, symbol=symbol, sample_date=sample_date),
    ]
    if api_key:
        command[8:8] = ["--header", f"Authorization: Bearer {api_key}"]
    subprocess.run(command, check=True, timeout=int(client_timeout_seconds) + 10)
    if not destination.exists() or not destination.stat().st_size:
        raise RuntimeError("direct Tardis download returned no bytes")
    return destination


def _atomic_promote(source: Path, target: Path) -> tuple[int, str]:
    size, checksum = _validate_gzip(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(source, stage)
    _validate_gzip(stage)
    os.replace(stage, target)
    return size, checksum


def download_manifest(
    manifest: pd.DataFrame,
    *,
    data_root: Path,
    retries: int,
    api_key: str | None,
    dry_run: bool,
    client_timeout_seconds: int,
    transport: str,
) -> pd.DataFrame:
    required = {"exchange", "dataset_symbol", "sample_date", "data_type", "status"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks fields: {sorted(missing)}")
    out = manifest.copy()
    out["sample_date"] = pd.to_datetime(out["sample_date"], utc=True, errors="coerce")
    for index, row in out.iterrows():
        if str(row["status"]) not in {"pending", "download_failed", "unavailable"}:
            continue
        sample_date = pd.Timestamp(row["sample_date"])
        if pd.isna(sample_date) or not _is_month_start(sample_date):
            raise ValueError(f"refusing non-free-sample date {sample_date!r}")
        if str(row["status"]) == "unavailable":
            continue
        target = _target_path(
            data_root, exchange=str(row["exchange"]), data_type=str(row["data_type"]),
            symbol=str(row["dataset_symbol"]), sample_date=sample_date,
        )
        out.at[index, "download_target"] = str(target)
        if target.exists():
            try:
                size, checksum = _validate_gzip(target)
            except (OSError, EOFError) as exc:
                out.at[index, "status"] = "corrupt_existing"
                out.at[index, "error"] = str(exc)
                continue
            out.at[index, "status"] = "downloaded"
            out.at[index, "file_size"] = size
            out.at[index, "checksum_sha256"] = checksum
            out.at[index, "error"] = ""
            continue
        if dry_run:
            out.at[index, "status"] = "planned"
            continue
        last_error = ""
        for attempt in range(int(retries) + 1):
            try:
                with tempfile.TemporaryDirectory(prefix="ares_tardis_") as temporary:
                    kwargs = {
                        "exchange": str(row["exchange"]), "data_type": str(row["data_type"]),
                        "symbol": str(row["dataset_symbol"]), "sample_date": sample_date,
                        "temp_dir": Path(temporary), "api_key": api_key,
                        "client_timeout_seconds": int(client_timeout_seconds),
                    }
                    downloaded = _download_one_direct(**kwargs) if transport == "direct" else _download_one_tardis(**kwargs)
                    size, checksum = _atomic_promote(downloaded, target)
                out.at[index, "status"] = "downloaded"
                out.at[index, "file_size"] = size
                out.at[index, "checksum_sha256"] = checksum
                out.at[index, "error"] = ""
                break
            except Exception as exc:  # downloader must retain a durable failure receipt
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < int(retries):
                    time.sleep(min(2 ** attempt, 8))
        else:
            text = last_error.lower()
            out.at[index, "status"] = "unavailable" if "not available" in text or "404" in text else "download_failed"
            out.at[index, "error"] = last_error
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/execution/tardis")
    parser.add_argument("--out-manifest", type=Path, help="Defaults to a sibling download-status parquet")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--api-key", default=os.environ.get("TARDIS_API_KEY"))
    parser.add_argument(
        "--client-timeout-seconds", type=int, default=120,
        help="Bound one client request; unavailable free dates must fail with a durable receipt rather than retry for 30 minutes.",
    )
    parser.add_argument("--transport", choices=("direct", "client"), default="direct")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dataset-symbols", nargs="+",
        help="Optional exact manifest dataset symbols for a bounded pilot; no fuzzy matching.",
    )
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    if args.dataset_symbols:
        requested = set(str(value) for value in args.dataset_symbols)
        available = set(manifest["dataset_symbol"].astype(str))
        unknown = sorted(requested.difference(available))
        if unknown:
            raise ValueError(f"requested dataset symbols are absent from the manifest: {unknown}")
        manifest = manifest.loc[manifest["dataset_symbol"].astype(str).isin(requested)].copy()
    result = download_manifest(
        manifest, data_root=args.data_root, retries=args.retries,
        api_key=args.api_key, dry_run=bool(args.dry_run),
        client_timeout_seconds=int(args.client_timeout_seconds),
        transport=str(args.transport),
    )
    out = args.out_manifest or args.manifest.with_name(args.manifest.stem + "_download_status.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    receipt: dict[str, Any] = {
        "schema": "ares.tardis_free_download.v1",
        "manifest": str(args.manifest),
        "output_manifest": str(out),
        "data_root": str(args.data_root),
        "free_sample_rule_enforced": True,
        "dry_run": bool(args.dry_run),
        "client_timeout_seconds": int(args.client_timeout_seconds),
        "transport": str(args.transport),
        "dataset_symbols": sorted(set(args.dataset_symbols or [])),
        "status_counts": {str(key): int(value) for key, value in result["status"].value_counts(dropna=False).items()},
    }
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
