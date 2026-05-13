"""Offline feature-parity job for live decision time T vs OOS features at T."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from extreme_price_movements.inference.live_feature_parity import (
    build_feature_parity_report,
    summarize_feature_parity,
)


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table format: {p}")


def _normalise_symbol(symbol: object) -> str:
    return str(symbol or "").upper().strip().replace(":USDT", "").replace("/", "_").replace("-", "_")


def _decision_symbols(decisions: pd.DataFrame) -> list[str]:
    if decisions.empty or "symbol" not in decisions.columns:
        return []
    return sorted({_normalise_symbol(v) for v in decisions["symbol"].dropna() if str(v)})


def _wide_from_long(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    required = {"timestamp", "symbol", "feature", "value"}
    if not required.issubset(df.columns):
        return {}
    out: Dict[str, pd.DataFrame] = {}
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp", "symbol", "feature"])
    for feature, grp in work.groupby("feature", sort=False):
        wide = (
            grp.pivot_table(
                index="timestamp",
                columns="symbol",
                values="value",
                aggfunc="last",
            )
            .sort_index()
            .rename_axis(index=None, columns=None)
        )
        out[str(feature)] = wide
    return out


def _is_symbol_partitioned_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("symbol=*.parquet"))


def _symbol_from_partition(path: Path) -> str:
    stem = path.stem
    if stem.startswith("symbol="):
        stem = stem[len("symbol=") :]
    return _normalise_symbol(stem)


def _read_symbol_feature_file(path: Path, feature_keys: Optional[set[str]]) -> pd.DataFrame:
    if feature_keys:
        try:
            return pd.read_parquet(path, columns=sorted(feature_keys))
        except Exception:
            df = pd.read_parquet(path)
            cols = [c for c in df.columns if str(c) in feature_keys]
            return df[cols]
    return pd.read_parquet(path)


def _load_symbol_partitioned_feature_frames(
    path: Path,
    *,
    feature_keys: Optional[Iterable[str]] = None,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    wanted_features = {str(f) for f in feature_keys or [] if str(f)}
    wanted_symbols = {_normalise_symbol(s) for s in symbols or [] if str(s)}
    files = sorted(path.glob("symbol=*.parquet"))
    if wanted_symbols:
        files = [f for f in files if _symbol_from_partition(f) in wanted_symbols]

    by_feature: Dict[str, list[pd.Series]] = {}
    for file in files:
        symbol = _symbol_from_partition(file)
        df = _read_symbol_feature_file(file, wanted_features or None)
        if df.empty:
            continue
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.loc[pd.notna(df.index)]
        for feature in df.columns:
            if wanted_features and str(feature) not in wanted_features:
                continue
            by_feature.setdefault(str(feature), []).append(df[feature].rename(symbol))

    frames: Dict[str, pd.DataFrame] = {}
    for feature, series_list in by_feature.items():
        frames[feature] = pd.concat(series_list, axis=1).sort_index()
    return frames


def load_feature_frames(
    path: str | Path,
    *,
    feature_keys: Optional[Iterable[str]] = None,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load feature frames for offline parity.

    Accepted formats:
      - Directory of parquet/csv files, one feature per file, with timestamp index
        or a timestamp column and symbols as columns.
      - Single long-form parquet/csv with columns: timestamp, symbol, feature, value.
    """
    p = Path(path)
    if not p.exists():
        return {}
    if _is_symbol_partitioned_dir(p):
        return _load_symbol_partitioned_feature_frames(
            p,
            feature_keys=feature_keys,
            symbols=symbols,
        )
    files = sorted(p.glob("*.parquet")) + sorted(p.glob("*.csv")) if p.is_dir() else [p]
    frames: Dict[str, pd.DataFrame] = {}
    for file in files:
        df = _read_table(file)
        long_frames = _wide_from_long(df)
        if long_frames:
            frames.update(long_frames)
            continue
        work = df.copy()
        if "timestamp" in work.columns:
            work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
            work = work.dropna(subset=["timestamp"]).set_index("timestamp")
        else:
            work.index = pd.to_datetime(work.index, utc=True, errors="coerce")
            work = work.loc[pd.notna(work.index)]
        frames[file.stem] = work.sort_index()
    return frames


def run_offline_feature_parity_job(
    *,
    decisions_path: str | Path,
    live_features_path: str | Path,
    oos_features_path: str | Path,
    output_dir: str | Path,
    feature_keys: Optional[Iterable[str]] = None,
    allow_asof: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = _read_table(decisions_path)
    feature_key_list = [str(f) for f in feature_keys or [] if str(f)] or None
    symbols = _decision_symbols(decisions)
    live_features = load_feature_frames(
        live_features_path,
        feature_keys=feature_key_list,
        symbols=symbols,
    )
    oos_features = load_feature_frames(
        oos_features_path,
        feature_keys=feature_key_list,
        symbols=symbols,
    )
    report = build_feature_parity_report(
        live_features,
        oos_features,
        decisions=decisions,
        feature_keys=feature_keys,
        include_extra_features=feature_keys is None,
        allow_asof=allow_asof,
    )
    summary = summarize_feature_parity(report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "feature_parity_report.csv", index=False)
    summary.to_csv(out / "feature_parity_summary.csv", index=False)
    return report, summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--live-features", required=True)
    parser.add_argument("--oos-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--allow-asof", action="store_true")
    args = parser.parse_args(argv)
    report, summary = run_offline_feature_parity_job(
        decisions_path=args.decisions,
        live_features_path=args.live_features,
        oos_features_path=args.oos_features,
        output_dir=args.output_dir,
        feature_keys=args.features,
        allow_asof=bool(args.allow_asof),
    )
    print(
        "offline feature parity complete: "
        f"rows={len(report)} features={summary['feature'].nunique() if not summary.empty else 0} "
        f"output_dir={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
