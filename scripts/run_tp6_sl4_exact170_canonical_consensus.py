#!/usr/bin/env python3
"""Exact-170 replay of the canonical TP6/SL4/H12 ten-head consensus stack.

The script deliberately keeps feature materialisation and model replay in one
versioned utility.  The feature materialiser uses the causal hourly feature
engine with a full cross-sectional panel, retains exactly the frozen 120-field
contract, and records field/row coverage.  The replay is monthly prequential:
strict R3 base -> train-only isotonic net map -> ten LambdaRank residual heads
-> 75/25 monthly side rank blend -> one pooled global tail ranking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, LGBMRanker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONTRACT_SOURCE = ROOT / 'config/strict_r3_canonical_v2_feature_contract.json'
RAW_15M_QUARANTINE_SOURCE = ROOT / 'config/strict_r3_raw_15m_source_quarantine_v1.json'
SIDECAR_QUARANTINE_SOURCE = ROOT / 'config/strict_r3_panel_sidecar_quarantine_v1.json'
LABEL_ROOT = ROOT / 'data_perp/artifacts/tp6_sl4_exact170_labels_20260808_v1'
MINUTE_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/execution_1m/ohlcv'
CONSOLIDATED_MINUTE_ROOT = ROOT / 'data_perp/artifacts/exact170_minute_consolidated_20260808_v1'
OHLCV_ROOT = ROOT / 'data_perp/ohlcv'
OB_ROOT = ROOT / 'data_perp/orderbook_hourly'
AUTHORITATIVE_OB_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/orderbook_hourly_native'
FROZEN_INPUT_BACKFILL_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly'
# Forward materialisation may use a newer, immutable primitive-cache snapshot
# without changing the historical schema-v1 replay.  The selected cache is an
# input to the feature contract and is therefore recorded in every downstream
# manifest.  This is deliberately an environment override rather than a
# moving "latest" pointer: inference must name the exact snapshot it uses.
CANONICAL_INPUT_CACHE_ROOT = Path(os.environ.get(
    'STRICT_R3_CANONICAL_INPUT_CACHE_ROOT',
    str(ROOT / 'data_perp/artifacts/canonical_hourly_primitive_cache_v1'),
))
# ``download_kraken_15m_hf.py`` writes to this shared HF store.  Keep the
# exchange-local directory only as a legacy fallback: it may contain an older
# regularised tail while the shared store has the freshly downloaded bars.
HF_15M_ROOT = ROOT / '15m_ohlcv_perp'
RAW_15M_ROOT = ROOT / 'data_perp/exchanges/krakenfutures/raw/ohlcv_15m'


def _raw_15m_quarantine_receipt() -> dict[str, object]:
    """Return the versioned raw-15m quarantine contract.

    Some historical parquet shards can become unreadable by Arrow's filtered
    scan even though the filesystem can still read their bytes.  This is a
    source-availability failure, never a reason to retry indefinitely or to
    substitute a different bar interval.  A versioned, explicit quarantine
    lets the already-approved shared 15-minute mirror fill that source while
    retaining the decision-time 15-minute semantics and a reproducible
    lineage receipt.
    """
    if not RAW_15M_QUARANTINE_SOURCE.is_file():
        return {"schema": "strict_r3_raw_15m_source_quarantine_v1", "entries": {}}
    payload = json.loads(RAW_15M_QUARANTINE_SOURCE.read_text())
    if payload.get("schema") != "strict_r3_raw_15m_source_quarantine_v1":
        raise ValueError("unsupported raw-15m quarantine schema")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("raw-15m quarantine entries must be a mapping")
    return payload


def _raw_15m_is_quarantined(path: Path) -> bool:
    if path.parent != RAW_15M_ROOT:
        return False
    return path.name in _raw_15m_quarantine_receipt()["entries"]


def _panel_sidecar_quarantine_receipt() -> dict[str, object]:
    """Return explicitly quarantined causal panel sidecars by field family."""
    if not SIDECAR_QUARANTINE_SOURCE.is_file():
        return {"schema": "strict_r3_panel_sidecar_quarantine_v1", "entries": {}}
    payload = json.loads(SIDECAR_QUARANTINE_SOURCE.read_text())
    if payload.get("schema") != "strict_r3_panel_sidecar_quarantine_v1":
        raise ValueError("unsupported panel-sidecar quarantine schema")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("panel-sidecar quarantine entries must be a mapping")
    return payload


def _panel_sidecar_is_quarantined(field: str, path: Path) -> bool:
    entries = _panel_sidecar_quarantine_receipt()["entries"]
    field_entries = entries.get(field, {})
    if not isinstance(field_entries, dict):
        raise ValueError(f"panel-sidecar quarantine field={field} must be a mapping")
    return path.name in field_entries

BASE_PARAMS = dict(
    objective='multiclass', num_class=3, n_estimators=220, learning_rate=0.035,
    max_depth=5, num_leaves=24, min_child_samples=2400, colsample_bytree=0.85,
    reg_lambda=20.0, subsample=1.0, random_state=17, n_jobs=4, verbosity=-1,
)
RANK_PARAMS = dict(
    objective='lambdarank', n_estimators=120, learning_rate=0.035, max_depth=5,
    num_leaves=31, min_child_samples=300, colsample_bytree=0.82,
    subsample=0.82, subsample_freq=1, reg_alpha=0.02, reg_lambda=2.0,
    max_bin=127, label_gain=[0, 0.25, 1, 3, 7], lambdarank_truncation_level=10,
    random_state=17, n_jobs=4, verbosity=-1,
)
CAPS = (40, 60, 80, 100, 120)
WEIGHT_MODES = ('ordinary', 'equal_month')
BASE_TRAIN_CAP = 240_000
# The engine only materialises a few cross-asset order-book composites when
# their primitive z-score frames are requested too.  They are deterministic
# decision-time dependencies, not additional model inputs; the parquet output
# below still retains only the frozen side contracts.
FROZEN_GENERATION_DEPENDENCIES = (
    # The band-analytics adapter produces this legacy alias directly.  The
    # frozen contract uses the bps-named equivalent below, which is copied
    # only after the feature engine has completed its causal transform.
    'ob_spread_z_24h',
    'ob_spread_bps_z_24h',
    'ob_spread_bps',
    'ob_depth_l20_to_qv_z_7d',
    # Cross-sectional composites are requested by their final names.  Their
    # raw parents must also be present when the complete-universe reduction is
    # performed; otherwise the generic engine correctly emits an all-zero
    # placeholder for an unavailable parent.
    'volume_z_12',
    'volume_z_24',
    'oi_to_volume_7d_z_180d',
    'xasset_mkt_spread_bps',
    # ``post_liquidation_rebound_score`` is a selected long feature, but its
    # five-input market composite is assembled only when these causal parents
    # are in the requested workset.  Two parents are already selected long
    # fields; these three are generation-only dependencies and never become
    # additional model inputs.
    'mkt_recovery_from_24h_low_atr',
    'mkt_price_up_oi_down_1h',
    'funding_mean_reversion_after_oi_flush',
)


def _load_contract() -> dict[str, list[str]]:
    obj = json.loads(CONTRACT_SOURCE.read_text())
    if obj.get('schema') == 'strict_r3_canonical_feature_contract_v2':
        return {
            'long': list(obj['base_fields_by_side']['long']),
            'short': list(obj['base_fields_by_side']['short']),
        }
    return {'long': list(obj['sets']['long']['120']), 'short': list(obj['sets']['short']['120'])}


def _label_parts() -> list[Path]:
    return sorted(LABEL_ROOT.glob('parts/month=*/side=*.parquet'))


def _load_labels() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in _label_parts()]
    out = pd.concat(frames, ignore_index=True)
    for c in ('__ts__', '__decision_ts__', '__label_available_at__'):
        out[c] = pd.to_datetime(out[c], utc=True)
    out['month'] = out['__ts__'].dt.to_period('M').astype(str)
    out['valid'] = out['label_valid'].astype(bool) & (~out['target_invalid'].astype(bool))
    event = pd.to_numeric(out['t2_tp6_sl4_event'], errors='coerce')
    robust = pd.to_numeric(out['robust_clear_event_b25'], errors='coerce')
    # R3: lower-first is adverse; robust clear is class 2; all other valid
    # paths (timeout and marginal upper) are the weak/unresolved class.
    out['r3_class'] = np.nan
    out.loc[out['valid'] & event.eq(1), 'r3_class'] = 0
    out.loc[out['valid'] & event.ne(1) & robust.eq(1), 'r3_class'] = 2
    out.loc[out['valid'] & event.ne(1) & ~robust.eq(1), 'r3_class'] = 1
    out['net_bps'] = pd.to_numeric(out['t4_tp6_sl4_net_bps'], errors='coerce')
    out['gross_bps'] = pd.to_numeric(out['t4_tp6_sl4_gross_bps'], errors='coerce')
    return out


def _source_map(symbols: Iterable[str]) -> dict[str, str | None]:
    dirs = {p.name.removeprefix('symbol='): p for p in OHLCV_ROOT.glob('symbol=*') if p.is_dir()}
    result: dict[str, str | None] = {}
    for sym in symbols:
        base = sym.split('/')[0]
        candidates = [f'{base}_USDT', f'{base}_USDC', f'{base}_USD', f'{base}_USD:USD']
        found = next((c for c in candidates if c in dirs), None)
        result[sym] = found
    return result


def _read_hourly_source(source: str | None, start: pd.Timestamp, end: pd.Timestamp):
    if source is not None:
        symbol_root = OHLCV_ROOT / f'symbol={source}'
        # The legacy archive is partitioned by year.  Opening every fragment
        # (including years that cannot satisfy the causal half-open query)
        # dominates recent source refreshes because Arrow must still read
        # each Parquet footer.  Restrict discovery to the calendar years that
        # overlap the requested interval.  This is an I/O optimisation only:
        # the per-row timestamp filter below remains the semantic authority.
        final_instant = pd.Timestamp(end) - pd.Timedelta(nanoseconds=1)
        years = range(pd.Timestamp(start).year, final_instant.year + 1)
        year_roots = [symbol_root / f'year={year}' for year in years]
        existing_year_roots = [path for path in year_roots if path.is_dir()]
        if existing_year_roots:
            paths = sorted(
                child
                for root in existing_year_roots
                for child in root.glob('**/*.parquet')
            )
        else:
            # Keep compatibility with a genuinely unpartitioned legacy
            # symbol root, but do not recurse into non-overlapping
            # ``year=YYYY`` directories.  Such files cannot provide a row in
            # the requested window and their Parquet-footer scans can stall a
            # live source refresh for minutes.
            paths = sorted(symbol_root.glob('*.parquet'))
        frames = []
        for p in paths:
            try:
                x = pd.read_parquet(p)
            except Exception:
                continue
            if 'ts' not in x.columns:
                continue
            x['ts'] = pd.to_datetime(x['ts'], utc=True)
            x = x[(x['ts'] >= start) & (x['ts'] < end)]
            if len(x): frames.append(x)
        if frames:
            x = pd.concat(frames, ignore_index=True).drop_duplicates('ts').sort_values('ts').set_index('ts')
            return x
    return None


def _read_canonical_input_cache(symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    """Read the compact, source-audited OHLCV cache before legacy panels."""
    path = CANONICAL_INPUT_CACHE_ROOT / 'hourly' / f"symbol={symbol.replace('/', '_')}" / 'part.parquet'
    if not path.exists():
        return None
    try:
        # ``part.parquet`` spans the full historical contract and lives on a
        # comparatively slow artifact volume on the live host.  Reading every
        # row for every symbol made a one-hour inference checkpoint take many
        # minutes.  The timestamp is an Arrow-visible index field, so push the
        # causal half-open window into the Parquet scan.  This changes neither
        # selected columns nor values and retains the defensive frame filter
        # below for engines that return extra row groups.
        x = pd.read_parquet(
            path,
            filters=[
                ('ts', '>=', pd.Timestamp(start).to_pydatetime()),
                ('ts', '<', pd.Timestamp(end).to_pydatetime()),
            ],
        )
        if not isinstance(x.index, pd.DatetimeIndex):
            if 'ts' not in x:
                return None
            x = x.set_index('ts')
        x.index = pd.to_datetime(x.index, utc=True)
        x = x[(x.index >= start) & (x.index < end)]
        return x if len(x) else None
    except Exception:
        return None


def _read_official_trade_hourly(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Read Kraken's cached public one-hour trade candles.

    This source is a causal coarse-bar fallback for a lagging 15-minute
    mirror.  It never replaces an already available downloaded 15-minute
    value: callers combine it only into missing cells.  Keeping the adapter
    separate from the mark/OI/order-book merger also makes the OHLCV source
    choice explicit in the inference lineage.
    """
    base = symbol.split('/')[0]
    path = FROZEN_INPUT_BACKFILL_ROOT / f'{base}_USD_USD.parquet'
    if not path.exists():
        return None
    fields = ['open', 'high', 'low', 'close', 'volume']
    try:
        raw = pd.read_parquet(
            path,
            columns=fields,
            filters=[
                ('__index_level_0__', '>=', pd.Timestamp(start).to_pydatetime()),
                ('__index_level_0__', '<', pd.Timestamp(end).to_pydatetime()),
            ],
        )
        if not isinstance(raw.index, pd.DatetimeIndex):
            return None
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw.loc[(raw.index >= start) & (raw.index < end), fields]
        raw = raw.apply(pd.to_numeric, errors='coerce').sort_index()
        raw = raw.loc[~raw.index.duplicated(keep='last')]
        raw = raw.dropna(subset=['open', 'high', 'low', 'close'])
    except Exception:
        return None
    return raw if len(raw) else None


def _read_downloaded_15m_hourly(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bar_phase_minutes: int = 0,
):
    """Resample the source-faithful 15-minute cache into complete hourly bars.

    The exchange-local raw download is authoritative.  The shared HF mirror is
    only a fill source because its June-2026 snapshot contains synthetic flat,
    zero-volume rows for otherwise active contracts.  Those rows are not valid
    observations: treating them as real bars destroys range, volatility, wick,
    correlation and Donchian features while concealing the source defect.

    Both sources remain 15-minute, decision-time inputs.  Requiring all four
    constituent bars retains the frozen hourly contract; the code never falls
    back to one-minute data in the canonical path.
    """
    name = f"{symbol.lower().replace('/', '')}_15m.parquet"
    fields = ['open', 'high', 'low', 'close', 'volume']

    def _load(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        # Do this before Arrow opens the file.  A quarantined raw shard has
        # previously blocked three independent phase materialisers inside the
        # parquet metadata read.  The shared 15-minute mirror below is the
        # declared same-interval fallback; no hourly or minute substitute is
        # introduced by this guard.
        if _raw_15m_is_quarantined(path):
            return None
        filters = [
            ('__index_level_0__', '>=', pd.Timestamp(start).to_pydatetime()),
            ('__index_level_0__', '<', pd.Timestamp(end).to_pydatetime()),
        ]
        try:
            # New live-cache rows carry an explicit provenance bit.  Retain a
            # legacy fallback because historical parquet files pre-date this
            # field and must keep their conservative flat-bar treatment.
            raw = pd.read_parquet(
                path,
                columns=[*fields, 'exchange_observed'],
                filters=filters,
            )
        except Exception:
            try:
                raw = pd.read_parquet(path, columns=fields, filters=filters)
            except Exception:
                return None
        if not isinstance(raw.index, pd.DatetimeIndex):
            return None
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw.loc[(raw.index >= start) & (raw.index < end)]
        raw = raw.loc[~raw.index.duplicated(keep='last')].sort_index()
        if raw.empty:
            return None
        observed = (
            raw['exchange_observed'].astype('boolean')
            if 'exchange_observed' in raw.columns
            else pd.Series(pd.NA, index=raw.index, dtype='boolean')
        )
        raw = raw.loc[:, fields].apply(pd.to_numeric, errors='coerce')
        finite_ohlc = raw[['open', 'high', 'low', 'close']].notna().all(axis=1)
        flat_zero = (
            finite_ohlc
            & raw['open'].eq(raw['high'])
            & raw['high'].eq(raw['low'])
            & raw['low'].eq(raw['close'])
            & raw['volume'].fillna(0.0).le(0.0)
        )
        # A currently observed Kraken candle is authoritative even if it is
        # flat with no reported trade volume.  Locally filled rows are marked
        # false.  Unlabelled legacy rows remain fail-closed under the prior
        # heuristic until a fresh observed replacement is available.
        synthetic_flat_zero = flat_zero & ~observed.fillna(False)
        # Do not let synthetic padding count toward a complete hourly bar.
        raw.loc[synthetic_flat_zero, fields] = np.nan
        return raw

    raw_download = _load(RAW_15M_ROOT / name)
    shared_mirror = _load(HF_15M_ROOT / name)
    if raw_download is None:
        raw = shared_mirror
    elif shared_mirror is None:
        raw = raw_download
    else:
        # The raw exchange download wins where it is present; the shared mirror
        # only fills genuine timestamp gaps after its synthetic bars have been
        # invalidated above.
        raw = raw_download.combine_first(shared_mirror)
    if raw is None or raw.empty:
        return None
    phase = int(bar_phase_minutes)
    if phase not in (0, 15, 30, 45):
        raise ValueError('bar_phase_minutes must be one of 0, 15, 30, 45')
    # The model contract remains one observation per hour.  A non-zero phase
    # merely shifts the hour boundary: a row at 23:15 summarises the four
    # fully observed 15-minute bars [23:15, 00:15), and is therefore usable
    # for a 00:15 decision.  It must not be confused with upsampling the
    # feature model itself to a 15-minute row cadence.
    out = raw.resample(
        '1h', label='left', closed='left', origin='epoch',
        offset=f'{phase}min',
    ).agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
        close=('close', 'last'), volume=('volume', 'sum'),
        count=('close', 'count'),
        coarse_trade_size_proxy_15m=('volume', 'median'),
    )
    out = out.loc[out['count'].eq(4)].drop(columns='count')
    return out if len(out) else None


def _read_downloaded_15m_decision_open(
    symbol: str,
    decision_index: pd.DatetimeIndex,
    *,
    return_lineage: bool = False,
) -> pd.Series | pd.DataFrame:
    """Read the exact trade-candle open at each decision timestamp.

    Unlike :func:`_read_downloaded_15m_hourly`, this adapter deliberately does
    not require the other three constituent 15-minute bars. At a live
    decision boundary those bars are in the future. The exchange-local raw
    cache retains precedence and the shared HF cache fills only missing
    timestamps, matching the canonical coarse-source ordering.

    Only ``open`` is consumed. No high, low, close, volume, future-path or
    completed-hour information can affect entry availability.  Kraken's
    public 15-minute chart can lag while its current official one-hour trade
    candle is already available.  In that case the timestamp-identical hourly
    ``open`` fills the missing 15-minute value *only when that hourly candle
    contains positive trade volume*.  A zero-volume OHLC row is an exchange
    placeholder, not evidence of an executable price, and must fail closed.
    The fallback never overwrites a 15-minute open and it is never carried
    from another timestamp.

    ``return_lineage`` exposes source and trade-volume metadata to the
    target-free candidate materialiser.  These fields are execution audit
    metadata only; they are never model inputs.  In particular, a direct
    15-minute row with zero (or unknown) trade volume is retained as lineage,
    rather than silently upgraded to an executable entry.  The materialiser
    validates such a row against the contemporaneous book before admitting it.
    """
    index = pd.DatetimeIndex(pd.to_datetime(decision_index, utc=True)).sort_values()
    index = index[~index.duplicated()]
    if index.empty:
        empty = pd.Series(dtype=np.float64, index=index, name='decision_open')
        if not return_lineage:
            return empty
        return pd.DataFrame({
            'decision_open': empty,
            'decision_open_source': pd.Series(dtype='object', index=index),
            'decision_open_trade_volume': pd.Series(dtype=np.float64, index=index),
            'decision_open_hourly_volume': pd.Series(dtype=np.float64, index=index),
        })
    name = f"{symbol.lower().replace('/', '')}_15m.parquet"

    def _load(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            raw = pd.read_parquet(
                path,
                columns=['open', 'volume'],
                filters=[
                    (
                        '__index_level_0__', '>=',
                        pd.Timestamp(index.min()).to_pydatetime(),
                    ),
                    (
                        '__index_level_0__', '<=',
                        pd.Timestamp(index.max()).to_pydatetime(),
                    ),
                ],
            )
        except Exception:
            # A legacy cache without volume is still useful as a price source,
            # but its entry must later pass the same book-alignment validation
            # as a zero-volume row.  Do not treat unknown volume as a trade.
            try:
                raw = pd.read_parquet(
                    path,
                    columns=['open'],
                    filters=[
                        (
                            '__index_level_0__', '>=',
                            pd.Timestamp(index.min()).to_pydatetime(),
                        ),
                        (
                            '__index_level_0__', '<=',
                            pd.Timestamp(index.max()).to_pydatetime(),
                        ),
                    ],
                )
            except Exception:
                return None
        if not isinstance(raw.index, pd.DatetimeIndex):
            return None
        raw.index = pd.to_datetime(raw.index, utc=True, errors='coerce')
        frame = pd.DataFrame({
            'open': pd.to_numeric(raw['open'], errors='coerce'),
            'volume': pd.to_numeric(raw.get('volume'), errors='coerce'),
        }, index=raw.index)
        frame = frame.loc[~frame.index.isna()]
        frame = frame.loc[~frame.index.duplicated(keep='last')].sort_index()
        frame = frame.reindex(index)
        return frame if frame['open'].notna().any() else None

    raw_download = _load(RAW_15M_ROOT / name)
    shared_mirror = _load(HF_15M_ROOT / name)
    if raw_download is None:
        selected = shared_mirror
        raw_present = pd.Series(False, index=index)
    elif shared_mirror is None:
        selected = raw_download
        raw_present = raw_download['open'].notna()
    else:
        raw_present = raw_download['open'].notna()
        selected = raw_download.combine_first(shared_mirror)
    if selected is None:
        selected = pd.DataFrame({
            'open': pd.Series(np.nan, index=index, dtype=np.float64),
            'volume': pd.Series(np.nan, index=index, dtype=np.float64),
        })
        raw_present = pd.Series(False, index=index)
    values = pd.to_numeric(selected['open'], errors='coerce').reindex(index)
    # The row timestamped exactly at ``decision_index`` is the 15-minute bar
    # which has just opened.  Its open is available for the canonical entry
    # convention, but its final volume is not: a later refresh can otherwise
    # observe trades from minutes after the decision and retroactively turn a
    # zero/unknown-volume entry into a trade-supported one.  Keep the price
    # lineage while forcing decision-time book corroboration for every direct
    # 15m decision open.  This makes a delayed replay identical to the live
    # decision rather than granting it future intra-bar information.
    trade_volume = pd.Series(np.nan, index=index, dtype=np.float64)
    source = pd.Series('unavailable', index=index, dtype='object')
    source.loc[values.notna() & raw_present] = 'raw_15m'
    source.loc[values.notna() & ~raw_present] = 'shared_15m'
    hourly_volume = pd.Series(np.nan, index=index, dtype=np.float64)
    if values.isna().any():
        official = _read_official_trade_hourly(
            symbol,
            pd.Timestamp(index.min()),
            pd.Timestamp(index.max()) + pd.Timedelta(hours=1),
        )
        if official is not None and {"open", "volume"}.issubset(official.columns):
            official_open = pd.to_numeric(
                official["open"], errors="coerce",
            ).reindex(index)
            hourly_volume = pd.to_numeric(
                official["volume"], errors="coerce",
            ).reindex(index)
            valid_hourly_trade = official_open.notna() & hourly_volume.gt(0.0)
            fill = values.isna() & valid_hourly_trade
            values = values.where(~fill, official_open)
            source.loc[fill] = 'official_hourly_trade'
            rejected_hourly = values.isna() & official_open.notna() & ~valid_hourly_trade
            source.loc[rejected_hourly & hourly_volume.le(0.0)] = (
                'official_hourly_zero_volume'
            )
            source.loc[rejected_hourly & ~hourly_volume.le(0.0)] = (
                'official_hourly_invalid'
            )
    values = values.reindex(index).rename('decision_open')
    if not return_lineage:
        return values
    return pd.DataFrame({
        'decision_open': values,
        'decision_open_source': source.reindex(index),
        'decision_open_trade_volume': trade_volume.reindex(index),
        'decision_open_hourly_volume': hourly_volume.reindex(index),
    })


def _read_minute_fallback(symbol: str, start: pd.Timestamp, end: pd.Timestamp, *, floor_start: pd.Timestamp | None = None):
    minute_symbol = f'{symbol.split("/")[0]}_USD:USD'
    # Prefer the complete raw minute history.  The consolidated artifact is a
    # late-period checkpoint and is intentionally only a fallback when a raw
    # symbol directory is absent.
    root = MINUTE_ROOT / f'symbol={minute_symbol}'
    if not root.exists():
        root = CONSOLIDATED_MINUTE_ROOT / f'symbol={minute_symbol}'
    if not root.exists():
        return None
    try:
        dataset = ds.dataset(root, format='parquet', partitioning='hive')
        read_start = max(start, floor_start) if floor_start is not None else start
        x = dataset.to_table(filter=(ds.field('ts') >= read_start) & (ds.field('ts') < end), columns=['ts','open','high','low','close','volume']).to_pandas()
    except Exception:
        return None
    if x.empty: return None
    x['ts'] = pd.to_datetime(x['ts'], utc=True)
    x = x.drop_duplicates('ts').set_index('ts').sort_index()
    return x.resample('1h', label='left', closed='left').agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum')).dropna(subset=['close'])


def _project_last_completed_calendar_hour(
    frame: pd.DataFrame | None,
    *,
    phase_index: pd.DatetimeIndex,
    phase_minutes: int,
) -> pd.DataFrame | None:
    """Project completed calendar-H1 bars onto a shifted decision grid.

    A strict-R3 candidate at ``00:15`` is decided at ``01:15``.  When the
    exact shifted 15-minute source is unavailable, the last *completed*
    calendar hour is therefore the bar labelled ``00:00`` (the interval
    ``[00:00, 01:00)``), not the partly formed ``01:00`` bar.  The projection
    has one common source convention for the entire cross-section: it does
    not mix an exact shifted bar for liquid symbols with a stale bar for the
    rest of the universe.

    This is deliberately an as-of projection, rather than a forward fill:
    each shifted row is an exact lookup of its prior completed calendar-H1
    label.  Consequently it cannot consume a price, volume, mark, or other
    hourly primitive published after the feature timestamp.
    """
    if frame is None or frame.empty:
        return None
    phase = int(phase_minutes)
    if phase not in (15, 30, 45):
        raise ValueError("calendar-H1 phase projection is only defined off :00")
    work = frame.copy()
    work.index = pd.to_datetime(work.index, utc=True)
    work = work.loc[~work.index.duplicated(keep="last")].sort_index()
    # Hourly sources are labelled by their interval start.  At a shifted
    # feature time, the immediately preceding calendar hour has just closed.
    source_index = (
        phase_index.floor("h") - pd.Timedelta(hours=1)
    )
    projected = work.reindex(source_index)
    projected.index = phase_index
    return projected


def _make_panel(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    allow_minute_fallback: bool = False,
    bar_phase_minutes: int = 0,
):
    """Build the causal source panel from hourly and 15-minute primitives.

    The frozen strict-R3 feature contract permits downloaded 15-minute bars
    and hourly sources only.  One-minute execution fragments can be useful
    for a separately named research proxy, but are intentionally prohibited
    here: using them only when another source happens to be absent changes
    source semantics by asset and period.  Missing approved primitives must
    remain missing and be surfaced by the strict feature-availability gate.

    ``allow_minute_fallback`` exists solely for explicitly named legacy
    research reproductions; the canonical materialiser never enables it.
    """
    phase = int(bar_phase_minutes)
    if phase not in (0, 15, 30, 45):
        raise ValueError('bar_phase_minutes must be one of 0, 15, 30, 45')
    if int(pd.Timestamp(start).minute) != phase:
        raise ValueError(
            'phase-shifted source panel start must share bar_phase_minutes; '
            f'start={start} phase={phase}'
        )
    source_map = _source_map(symbols)
    fields = ['open','high','low','close','volume','mark_open','mark_high','mark_low','mark_close','mark_price','index_open','index_high','index_low','index_close','index_price','spot_open','spot_high','spot_low','spot_close','spot_volume','funding_rate','open_interest','coarse_trade_size_proxy_15m']
    by_field: dict[str, dict[str, pd.Series]] = {f:{} for f in fields}
    # A shifted decision owns the complete rolling one-hour interval ending at
    # its phase boundary.  For example, the 00:15 decision consumes the four
    # source-faithful 15-minute bars [23:15, 00:15).  It must never be
    # substituted with the preceding calendar H1 [23:00, 00:00): that would
    # both discard the available latest quarter-hour information and change
    # the meaning of the frozen H1 features across phases.
    hourly_start = pd.Timestamp(start)
    expected = pd.date_range(
        start, end - pd.Timedelta(hours=1), freq='1h', tz='UTC',
    )

    def _load_symbol(sym: str) -> tuple[str, dict[str, pd.Series]]:
        """Read one independent causal source chain without mutating state."""
        # Official hourly candles are a valid native source only for the :00
        # contract.  They cannot represent a shifted rolling H1 interval.
        official_hourly = _read_official_trade_hourly(
            sym, hourly_start, end,
        )
        raw_15m = _read_downloaded_15m_hourly(
            sym, hourly_start, end, bar_phase_minutes=phase,
        )
        if phase:
            # A shifted row is valid only when the complete four-bar rolling
            # H1 source is present.  Calendar-H1/canonical fallbacks have a
            # different window and must not be mixed into this representation.
            x = raw_15m
        else:
            # Source precedence must be *cell-local*, never determined by
            # whether the entire requested horizon happens to be complete.
            # The former all-window shortcut meant that appending a later
            # missing coarse bar could cause the canonical cache to be opened
            # for an earlier interval and therefore change historical mark/
            # index/OI-derived features.  Merge the frozen source chain for
            # every horizon instead: downloaded 15-minute values win; the
            # canonical hourly cache fills genuinely missing primitives; and
            # official hourly trade candles are the final coarse fallback.
            # This is both the documented precedence contract and invariant
            # to appending future source rows.
            x = raw_15m
            canonical_cache = _read_canonical_input_cache(sym, hourly_start, end)
            if canonical_cache is not None:
                x = x.combine_first(canonical_cache) if x is not None else canonical_cache
            if official_hourly is not None:
                x = x.combine_first(official_hourly) if x is not None else official_hourly
        if not phase:
            # The legacy hourly archive remains the final source-faithful
            # fallback.  It must fill only cells that all higher-precedence
            # sources leave unavailable.  Testing ``x is None`` here was
            # still horizon-sensitive: a partial official archive beginning
            # in a later month suppressed older legacy cells merely because
            # it had *some* values in the expanded window.  Merge it
            # cell-locally so an appended later archive cannot erase an
            # earlier executable observation.
            legacy_hourly = _read_hourly_source(source_map[sym], hourly_start, end)
            if legacy_hourly is not None:
                x = x.combine_first(legacy_hourly) if x is not None else legacy_hourly
        if x is None and allow_minute_fallback:
            x = _read_minute_fallback(sym, start, end, floor_start=pd.Timestamp('2026-01-01', tz='UTC'))
        elif (
            x is not None
            and allow_minute_fallback
            and x.index.max() < end - pd.Timedelta(hours=1)
        ):
            # The bundled hourly sidecar currently ends in May 2026.  Append
            # the exact 1-minute source after its last timestamp so June/July
            # candidates do not silently become all-missing feature rows.
            tail = _read_minute_fallback(sym, start, end, floor_start=x.index.max() - pd.Timedelta(hours=2))
            if tail is not None and len(tail):
                x = pd.concat([x, tail]).sort_index()
                x = x[~x.index.duplicated(keep='last')]
        if x is None:
            return sym, {}
        x.index = pd.to_datetime(x.index, utc=True)
        source_values: dict[str, pd.Series] = {}
        for f in fields:
            if f in x.columns:
                source_values[f] = pd.to_numeric(x[f], errors='coerce')
        return sym, source_values

    # Each source chain is symbol-local and read-only.  The historical
    # Parquet estate includes mixed-age shards whose filtered metadata reads
    # can stall when driven by a 16-way pool.  Four workers preserve the
    # frozen source precedence/order while keeping file pressure bounded; the
    # 16-way setting is reserved for the separate network-bound funding/OI
    # refresher, not local Parquet reads.
    workers = min(
        4,
        max(1, int(os.environ.get('STRICT_R3_SOURCE_IO_WORKERS', '4'))),
        max(1, len(symbols)),
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(_load_symbol, symbols))
    for n, (sym, source_values) in enumerate(loaded, 1):
        for field, values in source_values.items():
            by_field[field][sym] = values
        if n % 25 == 0:
            print(json.dumps({'event':'source_loaded','number':n,'symbols':len(symbols)}), flush=True)
    idx = pd.date_range(start, end - pd.Timedelta(hours=1), freq='1h', tz='UTC')
    panel: dict[str, pd.DataFrame] = {}
    for f, values in by_field.items():
        panel[f] = pd.concat(values, axis=1).reindex(idx) if values else pd.DataFrame(index=idx, columns=symbols, dtype=np.float32)
        panel[f] = panel[f].reindex(columns=symbols).astype(np.float32)
    panel['quote_volume'] = (panel['close'] * panel['volume']).astype(np.float32)
    return panel, source_map


def _official_orderbook_analytics(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Adapt official depth-at-distance analytics into engine primitives.

    The frozen model never consumes L10/L20 quantities directly.  Its selected
    fields consume spread, near-touch depth, broad depth and their normalised
    cross-sectional derivatives.  Kraken provides these source-faithfully as
    5/25/50-bp liquidity bands.  The mapping is explicit and versioned here;
    it never claims a 50-bp band is literally twenty price levels.
    """
    base = symbol.split('/')[0]
    path = FROZEN_INPUT_BACKFILL_ROOT / f'{base}_USD_USD.parquet'
    if not path.exists():
        return pd.DataFrame()
    try:
        read_fields = [
            'ob_bid_bestPrice', 'ob_ask_bestPrice',
            'ob_bid_liquidity005', 'ob_ask_liquidity005',
            'ob_bid_liquidity025', 'ob_ask_liquidity025',
            'ob_bid_liquidity05', 'ob_ask_liquidity05',
            'close', 'volume',
        ]
        raw = pd.read_parquet(
            path,
            columns=read_fields,
            filters=[
                ('__index_level_0__', '>=', pd.Timestamp(start).to_pydatetime()),
                ('__index_level_0__', '<', pd.Timestamp(end).to_pydatetime()),
            ],
        )
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw = raw[(raw.index >= start) & (raw.index < end)]
    except Exception:
        return pd.DataFrame()
    required = ('ob_bid_bestPrice', 'ob_ask_bestPrice', 'ob_bid_liquidity005', 'ob_ask_liquidity005', 'ob_bid_liquidity025', 'ob_ask_liquidity025', 'ob_bid_liquidity05', 'ob_ask_liquidity05')
    if not set(required).issubset(raw.columns):
        return pd.DataFrame()
    out = pd.DataFrame(index=raw.index)
    out['best_bid'] = pd.to_numeric(raw['ob_bid_bestPrice'], errors='coerce')
    out['best_ask'] = pd.to_numeric(raw['ob_ask_bestPrice'], errors='coerce')
    out['mid'] = (out['best_bid'] + out['best_ask']) * 0.5
    # 5bp is the closest available official depth band; it replaces the
    # unselected raw top-level quantity only as a near-touch aggregate.
    out['bid_qty_1'] = pd.to_numeric(raw['ob_bid_liquidity005'], errors='coerce')
    out['ask_qty_1'] = pd.to_numeric(raw['ob_ask_liquidity005'], errors='coerce')
    # 25bp and 50bp are the explicit medium/broad-depth primitives used to
    # produce the selected depth-normalisation context, not level counts.
    out['cum_bid_qty_l10'] = pd.to_numeric(raw['ob_bid_liquidity025'], errors='coerce')
    out['cum_ask_qty_l10'] = pd.to_numeric(raw['ob_ask_liquidity025'], errors='coerce')
    out['cum_bid_qty_l20'] = pd.to_numeric(raw['ob_bid_liquidity05'], errors='coerce')
    out['cum_ask_qty_l20'] = pd.to_numeric(raw['ob_ask_liquidity05'], errors='coerce')
    out['snapshot_ts'] = out.index
    out['source'] = 'kraken_futures_orderbook_analytics_bands_v1'
    if {'close', 'volume'}.issubset(raw.columns):
        out['notional_1h'] = (
            pd.to_numeric(raw['close'], errors='coerce')
            * pd.to_numeric(raw['volume'], errors='coerce')
        )
    return out.replace([np.inf, -np.inf], np.nan)


def _asof_reindex(frame: pd.DataFrame, idx: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """Causally carry a slower sidecar onto a shifted H1 grid.

    The sidecars are observed snapshots rather than future-labelled values.
    Reindexing with a forward fill may therefore use only the latest snapshot
    at or before a phase-shifted hour boundary; it cannot draw from a later
    decision.  The phase-0 case is numerically identical to ordinary exact
    reindexing whenever the hourly sidecar is complete.
    """
    work = frame.sort_index().reindex(columns=symbols)
    return work.reindex(idx, method='ffill').astype(np.float32)


def _add_orderbook_panels(panel: dict[str, pd.DataFrame], symbols: list[str], idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    ob_fields = ['best_bid','best_ask','mid','bid_qty_1','ask_qty_1','cum_bid_qty_l10','cum_ask_qty_l10','cum_bid_qty_l20','cum_ask_qty_l20','notional_1h','mean_trade_qty_1h']
    out = {f: {} for f in ob_fields}
    # Sidecars are asynchronous decision-time observations.  A shifted H1
    # boundary (for example :15) must be allowed to use the latest official
    # snapshot at :00; slicing strictly from :15 makes an otherwise causal
    # forward fill look unavailable for the whole first bar.  One prior hour
    # is sufficient because the final as-of projection below never reaches
    # forward.
    source_start = pd.Timestamp(start) - pd.Timedelta(hours=1)
    for sym in symbols:
        base = sym.split('/')[0]
        x = _official_orderbook_analytics(sym, source_start, end)
        if x.empty:
            candidate = next(iter(sorted(AUTHORITATIVE_OB_ROOT.glob(f'{base}_USD*.parquet'))), None)
            if candidate is not None:
                try:
                    x = pd.read_parquet(candidate)
                    x.index = pd.to_datetime(x.index, utc=True)
                    x = x[(x.index >= source_start) & (x.index < end)]
                except Exception:
                    x = pd.DataFrame()
        # The historical sidecars currently identify themselves as
        # ``local_ohlcv_summary``.  That is a deterministic proxy, not an
        # authoritative L2 source.  A frozen contract must never manufacture
        # a depth/spread input from OHLCV, so accept only explicitly native
        # order-book provenance and leave all other cells unavailable.
        if len(x) and 'source' in x and x['source'].astype(str).eq('local_ohlcv_summary').all():
            x = pd.DataFrame()
        if x.empty:
            continue
        # Kraken's official trade candles provide hourly executed notional.
        # This is an exact causal aggregate, unlike the prohibited OHLCV
        # depth/spread proxy.  The archive has no historical trade *count*,
        # so ``mean_trade_qty_1h`` remains unavailable rather than being
        # manufactured from volume.
        if 'notional_1h' not in x and {'mid'}.issubset(x.columns):
            raw_path = FROZEN_INPUT_BACKFILL_ROOT / f'{base}_USD_USD.parquet'
            if raw_path.exists():
                try:
                    raw_trade = pd.read_parquet(raw_path, columns=['close', 'volume'])
                    raw_trade.index = pd.to_datetime(raw_trade.index, utc=True)
                    x['notional_1h'] = (
                        pd.to_numeric(raw_trade['close'], errors='coerce')
                        * pd.to_numeric(raw_trade['volume'], errors='coerce')
                    ).reindex(x.index)
                except Exception:
                    pass
        for f in ob_fields:
            if f in x.columns: out[f][sym] = pd.to_numeric(x[f], errors='coerce')
        # User-approved fallback for the one unrecoverable frozen input.  This
        # is intentionally distinct from native trade size: it is the median
        # executed 15-minute bar quantity from downloaded complete bars.  The
        # feature engine shifts it one bar before deriving the z-score, so it
        # remains decision-time causal.
        proxy = panel.get('coarse_trade_size_proxy_15m')
        if (
            'mean_trade_qty_1h' not in x.columns
            and isinstance(proxy, pd.DataFrame)
            and sym in proxy.columns
        ):
            out['mean_trade_qty_1h'][sym] = pd.to_numeric(proxy[sym], errors='coerce')
    for f, values in out.items():
        panel[f'orderbook_{f}'] = _asof_reindex(pd.concat(values, axis=1), idx, symbols) if values else pd.DataFrame(index=idx, columns=symbols, dtype=np.float32)


def _add_oi_funding_panels(panel: dict[str, pd.DataFrame], symbols: list[str], idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    """Attach persisted causal derivatives sidecars before feature generation."""
    roots = {'open_interest': ROOT/'data_perp/exchanges/krakenfutures/open_interest_hourly', 'funding_rate': ROOT/'data_perp/exchanges/krakenfutures/funding_hourly'}
    # Retain the most recent state strictly before a shifted H1 boundary for
    # causal as-of alignment.  The resulting value is never a future sample.
    source_start = pd.Timestamp(start) - pd.Timedelta(hours=1)
    for field, root in roots.items():
        def _load_symbol_sidecar(sym: str) -> tuple[str, pd.Series | None]:
            base = sym.split('/')[0]
            # Exact contract identity first.  A broad prefix can otherwise
            # select e.g. BTC_USD_BTC (an older inverse/alias sidecar) before
            # BTC_USD_USD, which silently makes current funding unavailable.
            canonical = root / f'{base}_USD_USD.parquet'
            path = canonical if canonical.is_file() else next(
                iter(sorted(root.glob(f'{base}_USD*.parquet'))), None
            )
            if path is None:
                return sym, None
            # Quarantined sidecars have failed a bounded Arrow-read audit.
            # Leave them unavailable (and allow the declared funding archive
            # to fill only where it exists) instead of allowing one legacy
            # file to block every independent phase materialisation.
            if _panel_sidecar_is_quarantined(field, path):
                return sym, None
            try:
                # The sidecars are index-timestamped.  Push the same causal
                # range into Arrow before it opens all historical row groups;
                # the defensive local slice below remains in place for older
                # Parquet writers that do not honour index filters.
                x = pd.read_parquet(
                    path,
                    columns=[field],
                    filters=[
                        ('__index_level_0__', '>=', source_start.to_pydatetime()),
                        ('__index_level_0__', '<', pd.Timestamp(end).to_pydatetime()),
                    ],
                )
            except Exception:
                try:
                    x = pd.read_parquet(path, columns=[field])
                except Exception:
                    return sym, None
            x.index = pd.to_datetime(x.index, utc=True); x=x[(x.index>=source_start)&(x.index<end)]
            return (sym, pd.to_numeric(x[field], errors='coerce')) if field in x else (sym, None)

        # Each sidecar is independent and read-only.  The old sequential
        # implementation paid 160 cold-file seeks per field before any model
        # work began.  A small bounded pool keeps local artifact pressure
        # reasonable while preserving deterministic symbol/result ordering.
        sidecar_workers = min(
            4,
            max(1, int(os.environ.get('STRICT_R3_SIDECAR_IO_WORKERS', '4'))),
            max(1, len(symbols)),
        )
        with ThreadPoolExecutor(max_workers=sidecar_workers) as executor:
            loaded = list(executor.map(_load_symbol_sidecar, symbols))
        values = {sym: series for sym, series in loaded if series is not None}
        # Funding sidecars cover the recent period only.  Backfill the same
        # causal hourly rate from Kraken's historical export, keeping the
        # sidecar value where both exist.
        if field == 'funding_rate':
            missing_symbols = [
                sym for sym in symbols
                if sym not in values
                or not values[sym].reindex(idx).notna().all()
            ]
            archive = ROOT/'data_perp/exchanges/krakenfutures/raw/funding_rates/kraken_historical_funding_rates.zip'
            if archive.exists() and missing_symbols:
                # The archive is optional historical backfill, never an input
                # requirement for a contemporaneous sidecar.  A torn or
                # externally corrupted archive must therefore leave only its
                # affected historical values unavailable rather than blocking
                # the entire 170-symbol causal panel or tempting a future-data
                # repair.  Valid archives retain the exact prior behavior.
                try:
                    with zipfile.ZipFile(archive) as zf:
                        names = set(zf.namelist())
                        for sym in missing_symbols:
                            base=sym.split('/')[0]
                            # Kraken's official export names perpetuals
                            # ``PF_<base>USD`` (and uses XBT for BTC).
                            trade_base = 'XBT' if base == 'BTC' else base
                            matches=[n for n in names if n.endswith(f'PF_{trade_base}USD.csv')]
                            if not matches: continue
                            try:
                                h=pd.read_csv(zf.open(matches[0]), usecols=['timestamp','relative_rate'])
                                h['timestamp']=pd.to_datetime(h['timestamp'],utc=True); h=h[(h.timestamp>=source_start)&(h.timestamp<end)].set_index('timestamp')
                                hist=pd.to_numeric(h['relative_rate'],errors='coerce')
                                values[sym]=hist.combine_first(values[sym]) if sym in values else hist
                            except Exception: continue
                except (zipfile.BadZipFile, OSError):
                    pass
        recovered = _asof_reindex(pd.concat(values,axis=1), idx, symbols) if values else pd.DataFrame(index=idx,columns=symbols,dtype=np.float32)
        existing = panel.get(field)
        panel[field] = existing.combine_first(recovered).astype(np.float32) if isinstance(existing, pd.DataFrame) else recovered


def _add_frozen_input_backfill(panel: dict[str, pd.DataFrame], symbols: list[str], idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    """Merge explicit, cached official mark/OI history after local stores end."""
    if not FROZEN_INPUT_BACKFILL_ROOT.exists():
        return
    collected: dict[str, dict[str, pd.Series]] = {'mark_price': {}, 'open_interest': {}}
    for sym in symbols:
        base = sym.split('/')[0]
        path = FROZEN_INPUT_BACKFILL_ROOT / f'{base}_USD_USD.parquet'
        if not path.exists():
            continue
        try:
            x = pd.read_parquet(
                path,
                columns=['mark_price', 'open_interest'],
                filters=[
                    ('__index_level_0__', '>=', pd.Timestamp(start).to_pydatetime()),
                    ('__index_level_0__', '<', pd.Timestamp(end).to_pydatetime()),
                ],
            )
            x.index = pd.to_datetime(x.index, utc=True)
            x = x[(x.index >= start) & (x.index < end)]
        except Exception:
            continue
        for field in collected:
            if field in x:
                collected[field][sym] = pd.to_numeric(x[field], errors='coerce')
    for field, values in collected.items():
        if not values:
            continue
        recovered = _asof_reindex(pd.concat(values, axis=1), idx, symbols)
        existing = panel.get(field)
        panel[field] = existing.combine_first(recovered).astype(np.float32) if isinstance(existing, pd.DataFrame) else recovered
    # The Kraken mark is an official, decision-time fair-price reference.  If
    # historical spot/index candles are not available, expose it explicitly as
    # the basis reference rather than silently falling back to trade OHLCV.
    # This lets the existing leverage feature use its OI/funding/basis formula
    # causally; provenance is recorded in the primitive audit.
    mark = panel.get('mark_price')
    index = panel.get('index_price')
    if isinstance(mark, pd.DataFrame) and mark.notna().any().any():
        panel['index_price'] = (
            index.combine_first(mark).astype(np.float32)
            if isinstance(index, pd.DataFrame)
            else mark.astype(np.float32)
        )


def _requires_orderbook(contract: dict[str, list[str]]) -> bool:
    names = set(contract['long']) | set(contract['short'])
    return any(any(token in name.lower() for token in ('ob_', 'orderbook', 'depth', 'spread', 'wall', 'liquidity')) for name in names)


def assert_orderbook_source_preflight(contract: dict[str, list[str]], out_dir: Path, symbols: list[str]) -> None:
    """Fail before opening any known proxy sidecar or generating features."""
    if not _requires_orderbook(contract):
        return
    audit = pd.DataFrame([{
        'primitive': 'native_orderbook_l2',
        'required_by_frozen_contract': True,
        'source_root': str(FROZEN_INPUT_BACKFILL_ROOT),
        'available': bool(sum((FROZEN_INPUT_BACKFILL_ROOT / f"{symbol.split('/')[0]}_USD_USD.parquet").exists() for symbol in symbols) > 0),
        'source_semantics': 'official Kraken 5bp/25bp/50bp depth bands mapped to near/medium/broad context primitives',
        'legacy_proxy_root': str(OB_ROOT),
        'legacy_proxy_policy': 'rejected: local_ohlcv_summary is not an authoritative primitive',
    }])
    audit.to_parquet(out_dir / 'primitive_source_preflight.parquet', index=False)
    if not bool(audit.loc[0, 'available']):
        raise RuntimeError(
            'Frozen contract requires native historical order-book depth/spread, but '
            f'no authoritative analytics archive exists at {FROZEN_INPUT_BACKFILL_ROOT}. The legacy '
            'data_perp/orderbook_hourly panels are local_ohlcv_summary proxies and are rejected before materialisation.'
        )


def assert_primitive_source_gate(panel: dict[str, pd.DataFrame], contract: dict[str, list[str]], out_dir: Path) -> None:
    """Reject unavailable required primitives before expensive feature work.

    This is intentionally stricter than a feature-coverage audit.  If the
    frozen contract asks for order-book depth/spread information, a cached
    OHLCV approximation cannot be substituted.  The report distinguishes the
    upstream primitive defect from model or label failure.
    """
    orderbook_required = _requires_orderbook(contract)
    primitive_rows = []
    for primitive in ('close', 'mark_price', 'open_interest', 'funding_rate', 'orderbook_best_bid', 'orderbook_best_ask', 'orderbook_cum_bid_qty_l20'):
        frame = panel.get(primitive)
        fraction = float(frame.notna().to_numpy().mean()) if isinstance(frame, pd.DataFrame) and frame.size else 0.0
        primitive_rows.append({'primitive': primitive, 'finite_fraction': fraction, 'available': bool(fraction > 0.0)})
    audit = pd.DataFrame(primitive_rows)
    audit.to_parquet(out_dir / 'primitive_source_gate.parquet', index=False)
    native_orderbook = audit.loc[audit.primitive.eq('orderbook_best_bid'), 'available'].any()
    if orderbook_required and not native_orderbook:
        raise RuntimeError(
            'Frozen contract requires order-book depth/spread features, but no authoritative historical order-book primitive is available. '
            'The legacy local_ohlcv_summary sidecars are intentionally rejected; provide a native L2/trade-book archive before feature generation.'
        )


def materialize_features(
    out_dir: Path,
    labels: pd.DataFrame,
    contract: dict[str, list[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    bar_phase_minutes: int = 0,
    full_feature_universe: bool = False,
    reference_symbols: Sequence[str] = (),
    context_symbols: Sequence[str] = (),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Candidate membership can legitimately be a routed subset of the market.
    # Cross-sectional fields must nevertheless be computed from the complete
    # point-in-time market context, *before* that candidate filter.  The
    # optional context universe augments raw inputs only: output identities
    # remain exactly the supplied target-free label grid.
    candidate_symbols = sorted({str(value) for value in labels['__symbol__'].astype(str).unique()})
    symbols = sorted({*candidate_symbols, *(str(value) for value in reference_symbols), *(str(value) for value in context_symbols)})
    assert_orderbook_source_preflight(contract, out_dir, symbols)
    # Canonical strict-R3 materialisation must never invoke minute-bar data.
    panel, source_map = _make_panel(
        symbols,
        start,
        end,
        allow_minute_fallback=False,
        bar_phase_minutes=bar_phase_minutes,
    )
    _add_orderbook_panels(panel, symbols, panel['close'].index, start, end)
    _add_oi_funding_panels(panel, symbols, panel['close'].index, start, end)
    _add_frozen_input_backfill(panel, symbols, panel['close'].index, start, end)
    assert_primitive_source_gate(panel, contract, out_dir)
    from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
    from extreme_price_movements.config import CFG
    mkt = compute_market_features(panel, symbols)
    gates = add_regime_gates(mkt, gate_vol_lookback_hours=24*7, gate_trend_thr=0.0)
    requested = list(dict.fromkeys(contract['long'] + contract['short'] + list(FROZEN_GENERATION_DEPENDENCIES)))
    cfg = dict(CFG)
    cfg.update({
        'atr_n':14,
        'use_perps':True,
        'feature_portability_mode':'off',
        'feature_portability_strict':False,
        # We supply official hourly depth-at-distance analytics through the
        # legacy wide-panel adapter above.  Do not invoke the separate raw-L2
        # snapshot reader: with no ``orderbook_hourly`` table it emits empty
        # defaults and overwrites the valid band-derived fields.
        'enable_orderbook_features':False,
        'enable_orderbook_wall_features':False,
        'live_lgbm_mask_feature_fast_path_enabled':False,
    })
    print(json.dumps({
        'event':'feature_generation_start', 'symbols':len(symbols),
        'hours':len(panel['close']), 'requested_fields':(
            'full_config_causal_universe' if full_feature_universe else len(requested)
        ),
    }), flush=True)
    # The default preserves the frozen production contract exactly.  The
    # opt-in branch is research-only: it asks the canonical engine for the
    # complete current causal-config union, then writes only the generated
    # numeric feature panels.  It never expands a live model contract.
    generated, feat_index, feat_cols = compute_features_hourly(
        panel,
        gates,
        cfg,
        requested_feature_keys=None if full_feature_universe else requested,
    )
    if full_feature_universe:
        requested = sorted(
            str(key) for key, value in generated.items()
            if isinstance(value, pd.DataFrame)
        )
        if not requested:
            raise RuntimeError('full causal feature-universe request generated no DataFrame features')
    # The early wide-panel path historically named this value
    # ``ob_spread_z_24h``; the frozen contract stores the mathematically
    # identical explicit-bps alias.  Preserve the value, do not recompute it
    # from any later information.
    if (
        not isinstance(generated.get('ob_spread_bps_z_24h'), pd.DataFrame)
        and isinstance(generated.get('ob_spread_z_24h'), pd.DataFrame)
    ):
        generated['ob_spread_bps_z_24h'] = generated['ob_spread_z_24h'].astype(np.float32)
    # ``xasset_mkt_spread_bps`` is a market-level field.  The historical
    # wide-panel order-book adapter exposes the per-asset spread but not this
    # broadcast aggregate, so materialise it here from the complete
    # point-in-time universe before any candidate filtering.
    spread = generated.get('ob_spread_bps')
    if isinstance(spread, pd.DataFrame) and not spread.empty:
        basket_bases = {
            str(value).split('/', 1)[0].upper()
            for value in cfg.get('market_basket', [])
        }
        basket = [
            symbol for symbol in spread.columns
            if str(symbol).split('/', 1)[0].upper() in basket_bases
        ]
        source = spread[basket] if basket else spread
        market_spread = source.mean(axis=1, skipna=True).astype(np.float32)
        generated['xasset_mkt_spread_bps'] = pd.DataFrame(
            np.broadcast_to(
                market_spread.to_numpy(dtype=np.float32)[:, None],
                (len(market_spread), len(spread.columns)),
            ),
            index=spread.index,
            columns=spread.columns,
        ).astype(np.float32, copy=False)
    # Some parents (notably the explicit spread alias above) become available
    # only after the main feature call.  Re-run only the cheap vectorised
    # complete-universe reductions so every frozen composite is derived from
    # its real causal parent rather than retaining an early zero placeholder.
    from extreme_price_movements.features import _add_regime_panel_composite_features
    _add_regime_panel_composite_features(
        generated,
        set(requested),
        cfg,
        pd.Index(feat_index),
        pd.Index(feat_cols),
    )
    target_keys = labels[['__ts__','__symbol__']].drop_duplicates().sort_values(['__ts__','__symbol__'])
    unique_ts = pd.DatetimeIndex(target_keys['__ts__'].unique()).sort_values()
    target_index = pd.MultiIndex.from_frame(target_keys[['__ts__','__symbol__']])
    # Construct the wide output in one operation.  Repeated ``out[key] =``
    # assignments fragment pandas' block manager for the full causal universe
    # (roughly 1,400 fields), turning a compact research materialisation into
    # a quadratic-time, high-memory operation.
    values_by_key: dict[str, np.ndarray] = {}
    for key in requested:
        frame = generated.get(key)
        if not isinstance(frame, pd.DataFrame):
            values_by_key[key] = np.full(len(target_keys), np.nan, dtype=np.float32)
            continue
        frame = frame.reindex(index=unique_ts, columns=symbols)
        # Pandas 3 rejects ``dropna=`` on its new stack implementation even
        # though the causal materialiser needs the full timestamp × symbol
        # grid.  Prefer the historical call where supported; its successor
        # keeps the same full-grid semantics for the subsequent explicit
        # reindex onto ``target_keys``.
        try:
            stacked = frame.stack(dropna=False).rename(key)
        except ValueError:
            stacked = frame.stack(future_stack=True).rename(key)
        values_by_key[key] = stacked.reindex(target_index).to_numpy(dtype=np.float32)
    out = pd.concat(
        [target_keys.reset_index(drop=True), pd.DataFrame(values_by_key)],
        axis=1,
        copy=False,
    )
    out_path = out_dir / (
        'causal_feature_universe.parquet' if full_feature_universe else 'canonical120_features.parquet'
    )
    out.to_parquet(out_path, index=False, compression='zstd')
    coverage = []
    for key in requested:
        vals = pd.to_numeric(out[key], errors='coerce')
        coverage.append({'feature':key,'rows':int(len(vals)),'finite_rows':int(vals.notna().sum()),'finite_fraction':float(vals.notna().mean()),'n_unique':int(vals.nunique(dropna=True))})
    pd.DataFrame(coverage).to_parquet(out_dir/'feature_coverage.parquet', index=False)
    manifest = {
        'schema':(
            'exact170_causal_feature_universe_panel_v1'
            if full_feature_universe else 'exact170_canonical120_feature_panel_v1'
        ),
        'source_map':source_map,
        'start':str(start),
        'end_exclusive':str(end),
        'bar_phase_minutes': int(bar_phase_minutes),
        'feature_cadence_contract': (
            'one completed H1 observation per row; phase shifts only the '
            '15-minute-derived H1 boundary and preserves all row-based H1 '
            'lookbacks'
        ),
        'symbols':symbols,
        'candidate_symbols': candidate_symbols,
        'reference_symbols': sorted({str(value) for value in reference_symbols}),
        'context_symbols': sorted({str(value) for value in context_symbols}),
        'requested_fields':requested,
        'full_feature_universe': bool(full_feature_universe),
        'generated_fields':sorted(generated),
        'rows':int(len(out)),
        'field_coverage':coverage,
        'approved_proxy_inputs': {
            'ob_trade_size_to_l1_depth_z_24h': {
                'primitive': 'mean_trade_qty_1h',
                'source': 'downloaded_15m_bar_volume_median_v1',
                'definition': 'median executed volume across the four complete 15-minute bars in the prior hourly bar',
                'causal_shift_bars': 1,
                'approval': 'user-authorized 2026-08-09',
            }
        },
        'bar_source_contract': {
            'allowed': [
                'hourly_source', 'downloaded_15m_resampled_hourly',
                'official_kraken_trade_hourly_missing_cell_fallback',
            ],
            'downloaded_15m_precedence': (
                'exchange_local_raw_archive, then shared_hf_mirror for genuine gaps'
            ),
            'official_hourly_precedence': (
                'fills missing coarse OHLCV cells only; never overwrites a complete '
                'downloaded 15-minute-derived value'
            ),
            'synthetic_flat_zero_bar_policy': 'invalidated before hourly resampling',
            'minute_fallback': 'prohibited',
        },
    }
    (out_dir/'feature_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    return out_path


def _impute(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]):
    med = train[fields].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).median().fillna(0.0)
    return train[fields].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0).astype(np.float32), test[fields].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0.0).astype(np.float32)


def assert_raw_contract_coverage(
    labels: pd.DataFrame,
    feature_path: Path,
    contract: dict[str, list[str]],
    months: list[str],
    *,
    minimum: float = 1.0,
) -> pd.DataFrame:
    """Fail closed when the frozen contract is not actually materialised.

    Imputation is a modelling convenience, not evidence that a frozen input was
    present.  Every retained candidate must have its entire side-local contract
    finite.  A field may legitimately be absent for an asset before its listing
    or before an authoritative feed began; those *rows* are discarded rather
    than forcing the whole universe to fail.  The gate fails only when a
    requested side/month has no fully available rows at all.
    """
    feature = pd.read_parquet(feature_path)
    keys = labels.loc[labels.month.isin(months), ['__ts__', '__symbol__', 'side_name']].drop_duplicates()
    audit = keys.merge(feature, on=['__ts__', '__symbol__'], how='left', validate='many_to_one')
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for side, fields in contract.items():
        part = audit.loc[audit.side_name.eq(side), fields]
        if part.empty:
            # Long-only / short-only replays legitimately omit the other
            # side.  There is no candidate population to validate in that
            # case; do not turn a side absent by scope into a coverage error.
            # Any side that is present remains subject to the 100%-complete
            # per-row frozen-contract gate below.
            continue
        finite = part.replace([np.inf, -np.inf], np.nan).notna()
        row_fraction = finite.mean(axis=1)
        for field, fraction in finite.mean(axis=0).items():
            value = float(fraction)
            rows.append({
                'side': side,
                'field': field,
                'raw_finite_fraction': value,
                'rows': int(len(part)),
                'row_median_feature_fraction': float(row_fraction.median()),
                'row_p10_feature_fraction': float(row_fraction.quantile(0.10)),
                'passes_raw_gate': bool(value >= minimum),
                'rows_with_complete_contract': int(row_fraction.eq(1.0).sum()),
                'complete_contract_fraction': float(row_fraction.eq(1.0).mean()),
            })
        dated = audit.loc[audit.side_name.eq(side), ['__ts__']].copy()
        dated['month'] = pd.to_datetime(dated['__ts__'], utc=True).dt.strftime('%Y-%m')
        dated['complete_contract'] = row_fraction.eq(1.0).to_numpy()
        for month, group in dated.groupby('month', sort=True):
            complete = int(group.complete_contract.sum())
            rows.append({
                'side': side, 'field': '__row_contract__', 'month': month,
                'raw_finite_fraction': float(group.complete_contract.mean()), 'rows': int(len(group)),
                'row_median_feature_fraction': float(row_fraction.loc[group.index].median()),
                'row_p10_feature_fraction': float(row_fraction.loc[group.index].quantile(0.10)),
                'passes_raw_gate': bool(complete > 0), 'rows_with_complete_contract': complete,
                'complete_contract_fraction': float(group.complete_contract.mean()),
            })
            if complete == 0:
                failures.append(f'{side}.{month}: zero complete-contract rows')
    result = pd.DataFrame(rows)
    result.to_parquet(feature_path.parent / 'scored_contract_raw_coverage.parquet', index=False)
    if failures:
        preview = ', '.join(failures[:20])
        more = '' if len(failures) <= 20 else f' (+{len(failures)-20} more)'
        raise RuntimeError(
            'Frozen canonical feature contract has no 100%-available rows for: '
            + preview + more
        )
    return result


def _groups(ts: pd.Series) -> np.ndarray:
    return pd.to_datetime(ts, utc=True).dt.floor('4h').astype(str).to_numpy()


def _fit_ranker(X: pd.DataFrame, y: np.ndarray, ts: pd.Series, months: pd.Series, mode: str):
    g = _groups(ts)
    _, counts = np.unique(g, return_counts=True)
    keep = counts >= 2
    valid_groups = set(np.unique(g)[keep])
    mask = np.array([v in valid_groups for v in g])
    if mask.sum() < 20 or np.unique(y[mask]).size < 2:
        return None
    row_idx = np.flatnonzero(mask)
    order = np.argsort(g[mask], kind='stable')
    row_idx = row_idx[order]
    g2 = g[row_idx]
    y2 = y[row_idx]
    m2 = months.iloc[row_idx]
    _, counts2 = np.unique(g2, return_counts=True)
    weights = None
    if mode == 'equal_month':
        freq = m2.value_counts()
        weights = m2.map(lambda m: 1.0 / float(freq.get(m, 1))).to_numpy(float)
        weights *= len(weights) / max(weights.sum(), 1e-12)
    model = LGBMRanker(**RANK_PARAMS)
    model.fit(X.iloc[row_idx], y2, group=counts2, sample_weight=weights)
    return model


def _pct(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    return s.rank(method='average', pct=True).to_numpy(float)


def _tail_metrics(frame: pd.DataFrame, score: str, label: str = 'canonical') -> list[dict]:
    rows=[]; ordered=frame.sort_values(score, ascending=False).reset_index(drop=True); n=len(ordered)
    for pct in (0.01,0.02,0.05,0.10):
        k=max(1,int(np.ceil(pct*n))); x=ordered.iloc[:k]
        rows.append({'label':label,'tail':f'top{int(pct*100)}pct','rows':int(k),'gross_bps_per_trade':float(x.gross_bps.mean()),'net_bps_per_trade':float(x.net_bps.mean()),'net_sum_bps':float(x.net_bps.sum()),'rank_ic':float(frame[[score,'net_bps']].corr(method='spearman').iloc[0,1]) if n>2 else np.nan})
    return rows


def replay(out_dir: Path, labels: pd.DataFrame, feature_path: Path, contract: dict[str, list[str]], months: list[str]):
    assert_raw_contract_coverage(labels, feature_path, contract, months)
    feat = pd.read_parquet(feature_path)
    labels = labels.merge(feat, on=['__ts__','__symbol__'], how='left', validate='many_to_one')
    predictions=[]; fold_meta=[]
    for month in months:
        cutoff = pd.Timestamp(month+'-01', tz='UTC')
        test_all = labels[(labels.month == month) & labels.valid & labels.r3_class.notna()].copy()
        train_all = labels[(labels.__label_available_at__ < cutoff) & labels.valid & labels.r3_class.notna()].copy()
        if test_all.empty or train_all.empty: continue
        print(json.dumps({'event':'fold_start','month':month,'train_rows':len(train_all),'test_rows':len(test_all)}), flush=True)
        for side in ('long','short'):
            tr = train_all[train_all.side_name == side].copy(); te = test_all[test_all.side_name == side].copy()
            if tr.empty or te.empty: continue
            fields = contract[side]
            # Do not turn an asset with an unavailable frozen source contract
            # into a median-imputed trade.  The field-level gate above proves
            # population support; this row-level gate protects the actual
            # train/test populations (for example a single unreadable asset).
            tr_raw_fraction = tr[fields].replace([np.inf, -np.inf], np.nan).notna().mean(axis=1)
            te_raw_fraction = te[fields].replace([np.inf, -np.inf], np.nan).notna().mean(axis=1)
            tr_excluded = int(tr_raw_fraction.lt(1.0).sum())
            te_excluded = int(te_raw_fraction.lt(1.0).sum())
            tr = tr.loc[tr_raw_fraction.eq(1.0)].copy()
            te = te.loc[te_raw_fraction.eq(1.0)].copy()
            if tr.empty or te.empty:
                continue
            Xtr, Xte = _impute(tr, te, fields)
            # The canonical producer recorded a 240k-row fit cap.  Keep the
            # latest matured rows so the cap is deterministic and causal;
            # downstream residual labels still use every matured row scored by
            # this capped base.
            base_fit = tr.sort_values('__label_available_at__').tail(BASE_TRAIN_CAP)
            Xbase, _ = _impute(base_fit, te, fields)
            base = LGBMClassifier(**BASE_PARAMS)
            base.fit(Xbase, base_fit.r3_class.astype(int).to_numpy())
            p_tr = base.predict_proba(Xtr); p_te = base.predict_proba(Xte)
            tr['base_score'] = p_tr[:,2] - 0.5*p_tr[:,0]; te['base_score'] = p_te[:,2] - 0.5*p_te[:,0]
            iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
            iso.fit(tr.base_score.to_numpy(), tr.net_bps.to_numpy())
            tr['base_anchor_bps'] = iso.predict(tr.base_score.to_numpy()); te['base_anchor_bps'] = iso.predict(te.base_score.to_numpy())
            tr['residual_bps'] = tr.net_bps - tr.base_anchor_bps
            tr['resid_grade'] = np.select([tr.residual_bps.le(-150),tr.residual_bps.le(-50),tr.residual_bps.le(50),tr.residual_bps.le(150)],[0,1,2,3],default=4).astype(int)
            head_preds=[]
            for cap in CAPS:
                for mode in WEIGHT_MODES:
                    model = _fit_ranker(Xtr.iloc[:, :cap], tr.resid_grade.to_numpy(), tr.__ts__, tr.month, mode)
                    pred = model.predict(Xte.iloc[:, :cap]) if model is not None else np.zeros(len(te), float)
                    head_preds.append(pred)
                    te[f'head_{cap}_{mode}'] = pred
            raw = np.column_stack(head_preds)
            te['consensus_rank'] = np.nanmedian(np.column_stack([_pct(raw[:,i]) for i in range(raw.shape[1])]), axis=1)
            te['base_rank'] = _pct(te.base_anchor_bps.to_numpy())
            te['final_score'] = .75*te.base_rank + .25*te.consensus_rank
            keep_cols=['candidate_id','__ts__','__decision_ts__','__symbol__','side_name','month','gross_bps','net_bps','base_score','base_anchor_bps','base_rank','consensus_rank','final_score']
            predictions.append(te[keep_cols])
            fold_meta.append({'month':month,'side':side,'train_rows':len(tr),'base_fit_rows':len(base_fit),'test_rows':len(te),'train_rows_excluded_raw_contract':tr_excluded,'test_rows_excluded_raw_contract':te_excluded,'base_feature_finite_fraction':float(np.isfinite(Xtr.to_numpy()).mean()),'test_feature_finite_fraction':float(np.isfinite(Xte.to_numpy()).mean())})
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    pred.to_parquet(out_dir/'predictions.parquet', index=False, compression='zstd')
    metrics=[]
    if len(pred):
        for score,label in [('base_rank','base_only'),('consensus_rank','consensus_only'),('final_score','canonical_ten_head')]: metrics.extend(_tail_metrics(pred,score,label))
        for month, g in pred.groupby('month'):
            for score,label in [('base_rank','base_only'),('consensus_rank','consensus_only'),('final_score','canonical_ten_head')]:
                for row in _tail_metrics(g,score,label): row['month']=month; metrics.append(row)
        for side, g in pred.groupby('side_name'):
            for score,label in [('base_rank','base_only'),('consensus_rank','consensus_only'),('final_score','canonical_ten_head')]:
                for row in _tail_metrics(g,score,label): row['side']=side; metrics.append(row)
    pd.DataFrame(metrics).to_parquet(out_dir/'metrics.parquet', index=False)
    pd.DataFrame(fold_meta).to_parquet(out_dir/'fold_metrics.parquet', index=False)
    return pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--feature-dir',type=Path,required=True); ap.add_argument('--materialize',action='store_true'); ap.add_argument('--replay',action='store_true'); ap.add_argument('--start',default='2024-07-01'); ap.add_argument('--end',default='2026-07-11'); ap.add_argument('--months',default='2026-02,2026-03,2026-04,2026-05,2026-06,2026-07')
    a=ap.parse_args(); labels=_load_labels(); contract=_load_contract(); start=pd.Timestamp(a.start,tz='UTC'); end=pd.Timestamp(a.end,tz='UTC'); a.feature_dir.mkdir(parents=True,exist_ok=True); a.out_dir.mkdir(parents=True,exist_ok=True)
    feature_path=a.feature_dir/'canonical120_features.parquet'
    if a.materialize or not feature_path.exists(): feature_path=materialize_features(a.feature_dir,labels,contract,start,end)
    if a.replay: replay(a.out_dir,labels,feature_path,contract,[m.strip() for m in a.months.split(',') if m.strip()])
    manifest={'schema':'exact170_canonical_ten_head_consensus_replay_v2','feature_contract_source':str(CONTRACT_SOURCE),'feature_path':str(feature_path),'label_root':str(LABEL_ROOT),'canonical_input_cache':str(CANONICAL_INPUT_CACHE_ROOT),'base_params':BASE_PARAMS,'rank_params':RANK_PARAMS,'caps':CAPS,'weight_modes':WEIGHT_MODES,'query':'4h UTC bucket x side','blend':'0.75 base percentile + 0.25 median ten-head percentile','ranking':'pooled global over valid held-month rows','raw_contract_policy':'every frozen-contract field must be finite on every retained train/test row; no median-imputed missing-source rows','primitive_policy':'native/downloaded sources only; reject local_ohlcv_summary order-book proxy','months':[m.strip() for m in a.months.split(',') if m.strip()]}
    (a.out_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))

if __name__=='__main__': main()
