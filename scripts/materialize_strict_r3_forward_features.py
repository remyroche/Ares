#!/usr/bin/env python3
"""Materialise a side-local frozen strict-R3 contract for an unlabeled grid.

The canonical feature builder is deliberately reused verbatim.  This wrapper
only supplies the score-time candidate identities and restricts generation to
the deployed long contract; it never creates outcome labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import (  # noqa: E402
    SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS,
)
from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    _add_frozen_input_backfill, _add_oi_funding_panels, _add_orderbook_panels,
    _load_contract, _make_panel, materialize_features,
)


_IDENTITY_COLUMNS = [
    'candidate_id', '__ts__', '__decision_ts__', '__symbol__', 'side_name',
]


def _assert_complete_market_universe(
    candidate_path: Path,
    candidates: pd.DataFrame,
) -> dict[str, object] | None:
    """Fail closed when a target-free grid was filtered before feature work.

    Cross-sectional features are properties of the point-in-time market, not
    of the subset that later passes spread/executability gates.  Canonical
    target-free candidate artifacts carry their full universe in the sibling
    manifest.  When that contract is present, require every signal timestamp
    to contain exactly that same symbol set before any feature is generated.

    Older research inputs without this schema remain readable, but they do
    not receive a canonical complete-universe attestation.
    """
    manifest_path = candidate_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "strict_r3_canonical_forward_v2_target_free_hourly_grid":
        return None
    source_map = manifest.get("source_map")
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("target-free candidate manifest lacks a non-empty source_map")
    expected = frozenset(str(symbol) for symbol in source_map)
    declared_rows = int(manifest.get("universe_rows", len(expected)))
    if declared_rows != len(expected):
        raise ValueError(
            "target-free manifest universe_rows disagrees with source_map: "
            f"{declared_rows} != {len(expected)}"
        )
    observed = frozenset(candidates["__symbol__"].astype(str).unique())
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))
    if missing or extra:
        raise ValueError(
            "feature generation requires the complete point-in-time market universe "
            "before eligibility filtering; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    counts = candidates.groupby("__ts__", sort=False)["__symbol__"].nunique()
    bad = counts.ne(len(expected))
    if bad.any():
        examples = {
            str(ts): int(count)
            for ts, count in counts.loc[bad].head(5).items()
        }
        raise ValueError(
            "feature generation requires every timestamp to contain the full "
            f"{len(expected)}-symbol universe; bad_timestamp_counts={examples}"
        )
    return {
        "candidate_manifest": str(manifest_path),
        "universe_rows": len(expected),
        "timestamp_rows": int(len(counts)),
        "complete_universe_before_candidate_filtering": True,
    }


def _attach_target_free_identity(feature_path: Path, candidates: pd.DataFrame) -> None:
    """Make the scoring panel directly candidate-keyed.

    The frozen feature generator naturally returns ``(__ts__, __symbol__)``
    rows.  The live contract is instead an immutable target-free candidate
    identity.  Attach it at materialisation time so a later extraction step
    cannot accidentally shift signal and decision timestamps while joining
    the panel to the scorer.
    """
    missing = sorted(set(_IDENTITY_COLUMNS).difference(candidates.columns))
    if missing:
        raise ValueError(f'candidate grid lacks scoring identities: {missing}')
    identities = candidates.loc[:, _IDENTITY_COLUMNS].copy()
    identities['__ts__'] = pd.to_datetime(identities['__ts__'], utc=True)
    identities['__decision_ts__'] = pd.to_datetime(identities['__decision_ts__'], utc=True)
    if identities['candidate_id'].duplicated().any() or identities.duplicated(['__ts__', '__symbol__']).any():
        raise ValueError('target-free candidates are not unique by ID and signal keys')

    feature = pd.read_parquet(feature_path)
    required = {'__ts__', '__symbol__'}
    if not required.issubset(feature.columns):
        raise ValueError(f'feature panel lacks signal keys: {sorted(required.difference(feature.columns))}')
    feature['__ts__'] = pd.to_datetime(feature['__ts__'], utc=True)
    if feature.duplicated(['__ts__', '__symbol__']).any():
        raise ValueError('feature panel has duplicate signal keys')
    value_columns = [
        column for column in feature.columns
        if column not in {'candidate_id', '__decision_ts__', 'side_name'}
    ]
    merged = identities.merge(
        feature.loc[:, value_columns], on=['__ts__', '__symbol__'], how='left',
        validate='one_to_one',
    )
    if len(merged) != len(identities) or merged['candidate_id'].duplicated().any():
        raise AssertionError('feature materialisation changed target-free candidate identity')
    ordered = [*_IDENTITY_COLUMNS, *[
        column for column in merged.columns if column not in _IDENTITY_COLUMNS
    ]]
    merged.loc[:, ordered].to_parquet(feature_path, index=False, compression='zstd')


def _repair_cross_asset_state_fields(
    feature_path: Path,
    *,
    candidates: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    repair_only: set[str] | None = None,
    source_panel: dict[str, pd.DataFrame] | None = None,
    bar_phase_minutes: int = 0,
) -> None:
    """Attach two causal state fields omitted by the generic materializer.

    Both definitions are native to the feature library: ``median_alt_minus_btc``
    is median non-BTC/ETH four-hour log return minus BTC four-hour log return;
    ``cross_asset_corr_1h`` is the mean four-observation rolling correlation
    between per-asset one-hour log return and contemporaneous market return.
    The calculation has no path/outcome inputs and is broadcast only after the
    complete hourly source panel has been assembled.
    """
    out = pd.read_parquet(feature_path)
    needed = {
        'median_alt_minus_btc', 'cross_asset_corr_1h',
        'cross_asset_downside_corr_4h',
        'q_lower_tail__xasset_mkt_spread_bps',
        'q_lower_tail__ob_spread_bps_z_24h',
        'xs_dispersion__ob_spread_bps_z_24h',
        'q_lower_tail__volume_z_24',
        'q_upper_tail__ob_spread_bps_z_24h',
        'q_tail_asym__ob_spread_bps_z_24h',
        'xs_dispersion__oi_to_volume_7d_z_180d',
        'q_tail_width__volume_z_12',
        'q_tail_width__oi_to_volume_7d_z_180d',
    }
    available = needed.intersection(out.columns)
    if repair_only is not None:
        requested = set(repair_only).intersection(available)
        unknown = set(repair_only).difference(available)
        if unknown:
            raise ValueError(f"requested repair fields are absent: {sorted(unknown)}")
    else:
        requested = available
    if not requested:
        return
    symbols = sorted(candidates['__symbol__'].astype(str).unique())
    # Incremental producers have already loaded and causally bounded this
    # complete-universe panel. Reopening 170 archive sources here duplicated
    # identical I/O and dominated hourly latency. The optional handoff is an
    # exact input reuse: callers remain responsible for the same symbol/time
    # contract, which is asserted below. Batch callers retain the historical
    # loader fallback.
    if source_panel is None:
        panel, _ = _make_panel(
            symbols, start, end, bar_phase_minutes=bar_phase_minutes,
        )
    else:
        panel = source_panel
        if not isinstance(panel.get('close'), pd.DataFrame):
            raise ValueError('source_panel lacks close matrix')
        panel_index = pd.DatetimeIndex(panel['close'].index)
        if len(panel_index) == 0 or panel_index.min() > start or panel_index.max() >= end:
            raise ValueError('source_panel does not match declared causal time bounds')
        missing_symbols = sorted(set(symbols).difference(panel['close'].columns))
        if missing_symbols:
            raise ValueError(
                f'source_panel misses repair symbols: {missing_symbols[:8]}'
            )
    close = panel['close'].reindex(columns=symbols).replace([np.inf, -np.inf], np.nan)
    log_close = np.log(close.where(close > 0.0))
    ret_1h = log_close.diff(1)
    market_ret = ret_1h.mean(axis=1, skipna=True)
    corr = ret_1h.rolling(4, min_periods=2).corr(market_ret).mean(axis=1, skipna=True).clip(-1.0, 1.0)
    btc = next((c for c in close.columns if str(c).upper().startswith('BTC/')), None)
    eth = next((c for c in close.columns if str(c).upper().startswith('ETH/')), None)
    alt_cols = [c for c in close.columns if c not in {btc, eth}]
    ret_4h = log_close.diff(4)
    median_alt = ret_4h[alt_cols].median(axis=1, skipna=True) if alt_cols else pd.Series(np.nan, index=close.index)
    btc_ret = ret_4h[btc] if btc is not None else pd.Series(np.nan, index=close.index)
    state = pd.DataFrame({
        '__ts__': close.index,
        'median_alt_minus_btc': (median_alt - btc_ret).astype(np.float32),
        'cross_asset_corr_1h': corr.astype(np.float32),
    })
    # A downside-only correlation is undefined when the complete causal
    # four-hour window has no downside variation.  In that precise state the
    # economically neutral value is zero: there is no observed downside
    # co-movement.  Do not carry another timestamp and do not replace missing
    # values caused by inadequate source coverage.
    downside_field = 'cross_asset_downside_corr_4h'
    if downside_field in out.columns:
        downside = pd.to_numeric(out[downside_field], errors='coerce').replace(
            [np.inf, -np.inf], np.nan,
        )
        close_coverage = close.notna().mean(axis=1)
        neutral_ts = set(close.index[
            close_coverage.ge(0.90)
            & close.index.to_series().sub(close.index.min()).ge(pd.Timedelta(hours=4))
        ])
        neutral_rows = downside.isna() & out['__ts__'].isin(neutral_ts)
        out.loc[neutral_rows, downside_field] = np.float32(0.0)
    # Rebuild the cross-sectional spread composites from the causal per-asset
    # parent already persisted by the canonical generator.  This also repairs
    # older panels created before the parent/composite dependency ordering was
    # made explicit.
    spread_z_name = next(
        (name for name in ('ob_spread_bps_z_24h', 'ob_spread_z_24h') if name in out.columns),
        None,
    )
    if spread_z_name is not None:
        spread_z = (
            out.pivot(index='__ts__', columns='__symbol__', values=spread_z_name)
            .reindex(index=close.index, columns=symbols)
        )
        q10 = spread_z.quantile(0.10, axis=1)
        q25 = spread_z.quantile(0.25, axis=1)
        q50 = spread_z.quantile(0.50, axis=1)
        q75 = spread_z.quantile(0.75, axis=1)
        q90 = spread_z.quantile(0.90, axis=1)
        state['q_lower_tail__ob_spread_bps_z_24h'] = q10.astype(np.float32)
        state['xs_dispersion__ob_spread_bps_z_24h'] = (q75 - q25).astype(np.float32)
        state['q_upper_tail__ob_spread_bps_z_24h'] = q90.astype(np.float32)
        state['q_tail_asym__ob_spread_bps_z_24h'] = (
            (q90 + q10) - (2.0 * q50)
        ).astype(np.float32)
    quote_volume = panel.get('quote_volume')
    if not isinstance(quote_volume, pd.DataFrame):
        volume = panel.get('volume')
        quote_volume = (
            volume.reindex_like(close) * close
            if isinstance(volume, pd.DataFrame) else None
        )
    if isinstance(quote_volume, pd.DataFrame):
        log_qv = np.log1p(
            quote_volume.reindex(index=close.index, columns=close.columns)
            .replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
        )
        mean12 = log_qv.rolling(12, min_periods=12).mean()
        std12 = log_qv.rolling(12, min_periods=12).std(ddof=0).replace(0.0, np.nan)
        mean24 = log_qv.rolling(24, min_periods=24).mean()
        std24 = log_qv.rolling(24, min_periods=24).std(ddof=0).replace(0.0, np.nan)
        volume_z12 = (log_qv - mean12) / std12
        volume_z24 = (log_qv - mean24) / std24
        state['q_tail_width__volume_z_12'] = (
            volume_z12.quantile(0.90, axis=1) - volume_z12.quantile(0.10, axis=1)
        ).astype(np.float32)
        state['q_lower_tail__volume_z_24'] = volume_z24.quantile(0.10, axis=1).astype(np.float32)
    if 'q_lower_tail__xasset_mkt_spread_bps' in out.columns:
        bid_ready = isinstance(panel.get('orderbook_best_bid'), pd.DataFrame)
        ask_ready = isinstance(panel.get('orderbook_best_ask'), pd.DataFrame)
        if not (bid_ready and ask_ready):
            _add_orderbook_panels(panel, symbols, close.index, start, end)
        bid = panel.get('orderbook_best_bid')
        ask = panel.get('orderbook_best_ask')
        if isinstance(bid, pd.DataFrame) and isinstance(ask, pd.DataFrame):
            bid = bid.reindex(index=close.index, columns=close.columns)
            ask = ask.reindex(index=close.index, columns=close.columns)
            mid = 0.5 * (bid + ask)
            spread_bps = (10_000.0 * (ask - bid) / mid.where(mid > 0.0)).replace(
                [np.inf, -np.inf], np.nan,
            )
            # The parent is market-level and broadcast, so its lower-tail
            # reduction equals the complete-universe contemporaneous mean.
            state['q_lower_tail__xasset_mkt_spread_bps'] = spread_bps.mean(
                axis=1, skipna=True,
            ).astype(np.float32)
    oi_fields = {
        'xs_dispersion__oi_to_volume_7d_z_180d',
        'q_tail_width__oi_to_volume_7d_z_180d',
    }
    if oi_fields.intersection(out.columns):
        # The general feature engine deliberately fails the native-relative
        # OI/quote-volume ratio closed when unit provenance is incomplete.
        # The strict contract nevertheless contains two market cross-sections
        # of that scale-invariant ratio.  Reproduce the native formula here
        # from the authoritative hourly OI and quote-volume panels, then
        # reduce across the complete universe before joining candidates.
        # Incremental callers pass a panel that was already assembled from
        # the authoritative current sidecar plus frozen backfill.  Reopening
        # every per-symbol OI/funding archive here duplicates that exact I/O
        # and can dominate the hourly latency.  Batch callers, or incomplete
        # panels, retain the historical loading/merge path.
        oi_frame = panel.get('open_interest')
        # ``_make_panel`` always allocates every primitive key, including an
        # empty all-NaN OI matrix when the coarse source lacks OI.  Treating
        # that allocation as loaded data made the surgical path skip the
        # authoritative OI sidecars entirely.  A batch repair without a
        # preassembled source panel must always merge those sidecars; an
        # incremental caller may reuse its already-complete panel.
        oi_ready = (
            isinstance(oi_frame, pd.DataFrame)
            and bool(oi_frame.notna().to_numpy().any())
        )
        if source_panel is None or not oi_ready:
            _add_oi_funding_panels(panel, symbols, close.index, start, end)
            # Match the main canonical materialiser's source precedence.  The
            # current OI sidecar can lag the authoritative frozen Kraken
            # backfill; combine-first retains the higher-priority sidecar.
            _add_frozen_input_backfill(panel, symbols, close.index, start, end)
        oi = panel.get('open_interest')
        if isinstance(oi, pd.DataFrame) and isinstance(quote_volume, pd.DataFrame):
            from extreme_price_movements.features_oi import (
                rolling_long_iqr_robust_zscore_by_symbol,
            )
            oi = (
                oi.reindex(index=close.index, columns=close.columns)
                .replace([np.inf, -np.inf], np.nan).where(lambda frame: frame > 0.0)
                .ffill(limit=8)
            )
            qv = (
                quote_volume.reindex(index=close.index, columns=close.columns)
                .replace([np.inf, -np.inf], np.nan).where(lambda frame: frame > 0.0)
            )
            volume_7d = qv.rolling(24 * 7, min_periods=1).sum()
            ratio = np.log1p((oi / volume_7d.clip(lower=1e-12)).clip(lower=0.0))
            robust_z = rolling_long_iqr_robust_zscore_by_symbol(
                ratio, 24 * 180,
            ).clip(-10.0, 10.0)
            state['xs_dispersion__oi_to_volume_7d_z_180d'] = (
                robust_z.quantile(0.75, axis=1) - robust_z.quantile(0.25, axis=1)
            ).astype(np.float32)
            state['q_tail_width__oi_to_volume_7d_z_180d'] = (
                robust_z.quantile(0.90, axis=1) - robust_z.quantile(0.10, axis=1)
            ).astype(np.float32)
    out['__ts__'] = pd.to_datetime(out['__ts__'], utc=True)
    replacement_columns = [name for name in state.columns if name in requested]
    out = out.drop(columns=replacement_columns, errors='ignore').merge(
        state.loc[:, ['__ts__', *replacement_columns]],
        on='__ts__', how='left', validate='many_to_one',
    )
    trade_size_proxy = 'ob_trade_size_to_l1_depth_z_24h'
    if trade_size_proxy in out.columns:
        values = pd.to_numeric(out[trade_size_proxy], errors='coerce').replace(
            [np.inf, -np.inf], np.nan,
        )
        # Explicitly authorised coarse fallback: when an asset's 15-minute
        # trade-size/depth proxy is absent but peers are available at the same
        # decision timestamp, use the complete-universe contemporaneous
        # median.  This is causal, neutral in cross-section, and never
        # backfills from a later timestamp.
        timestamp_median = values.groupby(out['__ts__'], sort=False).transform('median')
        out[trade_size_proxy] = values.fillna(timestamp_median).astype(np.float32)
    out.to_parquet(feature_path, index=False, compression='zstd')


def _refresh_feature_coverage(feature_path: Path, coverage_path: Path) -> None:
    """Keep the persisted audit aligned with post-materialisation repairs."""
    frame = pd.read_parquet(feature_path)
    keys = [
        key for key in frame.columns
        if key not in {'candidate_id', '__ts__', '__decision_ts__', '__symbol__', 'side_name'}
    ]
    rows = len(frame)
    records = []
    for key in keys:
        value = pd.to_numeric(frame[key], errors='coerce').replace([np.inf, -np.inf], np.nan)
        finite = int(value.notna().sum())
        records.append({
            'feature': key, 'rows': rows, 'finite_rows': finite,
            'finite_fraction': float(finite / rows) if rows else 0.0,
            'n_unique': int(value.nunique(dropna=True)),
        })
    pd.DataFrame(records).to_parquet(coverage_path, index=False)
    manifest_path = feature_path.parent / 'feature_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest['field_coverage'] = records
        manifest['post_materialisation_repairs'] = {
            'cross_sectional_parent_dependency_order': (
                'raw causal parents are materialised before complete-universe reductions'
            ),
            'oi_to_volume_7d_z_180d': (
                'native-relative OI / trailing 7d quote volume; causal 180d robust z; '
                'complete-universe IQR and p90-p10 reductions'
            ),
            'ob_trade_size_to_l1_depth_z_24h_missing_asset_fallback': (
                'same-timestamp complete-universe median of the authorised 15m coarse proxy; '
                'no later timestamp fill'
            ),
            'cross_asset_downside_corr_4h_zero_variance': (
                'undefined downside-only correlation is neutral zero only after four '
                'causal hours and with complete-universe close coverage at least 90%; '
                'no value from another timestamp is used'
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-start",
        help="optional inclusive UTC decision timestamp used to slice a larger target-free grid",
    )
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument(
        "--bar-phase-minutes", type=int, default=0, choices=(0, 15, 30, 45),
        help=(
            "hour-boundary phase for research sampling. The model still sees "
            "one H1 observation per row; phase only moves the completed-H1 "
            "window and never upsamples feature rows."
        ),
    )
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument(
        "--repair-existing-features", type=Path,
        help=(
            "copy an existing target-free feature panel then recompute only "
            "the declared causal OI cross-sections; avoids a full feature "
            "rebuild when all other frozen fields are already verified"
        ),
    )
    parser.add_argument(
        "--full-feature-universe", action="store_true",
        help=(
            "research only: materialize the canonical engine's complete current "
            "causal-config feature union instead of the frozen inference contract"
        ),
    )
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates)
    required = {"candidate_id", "__ts__", "__symbol__", "side_name"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate grid lacks required columns: {missing}")
    if not candidates.side_name.astype(str).str.lower().eq(args.side).all():
        raise ValueError(f"forward candidates must all match side={args.side}")
    if candidates.candidate_id.duplicated().any():
        raise ValueError("candidate grid has duplicate candidate IDs")
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True)
    base_contract = _load_contract()
    declared = json.loads(
        (ROOT / "config/strict_r3_canonical_v2_feature_contract.json").read_text()
    )
    severe_context = list(declared["severe_context_fields"])
    # ``materialize_features`` generates the union of the two lists.  Include
    # the complete Severe contract on both side-local runs: 44 of these fields
    # are not in the short base list, and omitting them makes short scoring
    # impossible even though base feature generation succeeds.
    # Always materialise the registered session candidate pool.  Frozen model
    # matrices continue to select only their declared fields, while a successor
    # base/conditional-consensus contract can consume the exact same inference
    # columns without requiring a divergent feature path.
    requested = list(
        dict.fromkeys(
            [
                *base_contract[args.side],
                *severe_context,
                *SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS,
            ]
        )
    )
    contract = {
        "long": requested if args.side == "long" else [],
        "short": requested if args.side == "short" else [],
    }
    history_start = pd.to_datetime(args.history_start, utc=True)
    end_exclusive = pd.to_datetime(args.end_exclusive, utc=True)
    if int(history_start.minute) != int(args.bar_phase_minutes):
        raise ValueError("history-start minute must equal --bar-phase-minutes")
    if int(end_exclusive.minute) != int(args.bar_phase_minutes):
        raise ValueError("end-exclusive minute must equal --bar-phase-minutes")
    candidate_start = (
        pd.to_datetime(args.candidate_start, utc=True)
        if args.candidate_start is not None else None
    )
    if candidate_start is not None:
        candidates = candidates.loc[
            candidates["__decision_ts__"].ge(candidate_start)
            & candidates["__decision_ts__"].lt(end_exclusive)
        ].copy()
        if candidates.empty:
            raise ValueError("candidate-start/end-exclusive selected no target-free candidates")
    universe_attestation = _assert_complete_market_universe(args.candidates, candidates)
    if args.repair_existing_features is not None:
        source = args.repair_existing_features
        if not source.exists():
            raise FileNotFoundError(source)
        original = pd.read_parquet(source)
        required_keys = {'__ts__', '__symbol__'}
        if not required_keys.issubset(original.columns):
            raise ValueError(f"existing feature panel lacks {sorted(required_keys)}")
        original['__ts__'] = pd.to_datetime(original['__ts__'], utc=True)
        candidate_keys = candidates.loc[:, ['__ts__', '__symbol__']].drop_duplicates()
        feature_keys = original.loc[:, ['__ts__', '__symbol__']].drop_duplicates()
        if len(candidate_keys) != len(candidates) or len(feature_keys) != len(original):
            raise ValueError('candidate/feature identities must be unique for a surgical repair')
        if len(candidate_keys.merge(feature_keys, on=['__ts__', '__symbol__'], how='inner')) != len(candidate_keys):
            raise ValueError('existing feature panel identities do not match target-free candidates')
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / 'canonical120_features.parquet'
        original.to_parquet(path, index=False, compression='zstd')
        (args.out_dir / 'feature_manifest.json').write_text(json.dumps({
            'schema': 'strict_r3_forward_feature_surgical_repair_v1',
            'repair_from': str(source),
            'repair_fields': [
                'xs_dispersion__oi_to_volume_7d_z_180d',
                'q_tail_width__oi_to_volume_7d_z_180d',
            ],
            'history_start': str(history_start),
            'end_exclusive': str(end_exclusive),
            'candidate_rows': int(len(candidates)),
            'complete_universe_attestation': universe_attestation,
            'causal_source_precedence': (
                'incremental OI sidecar followed by frozen Kraken backfill; '
                'complete point-in-time cross-section before candidate filtering'
            ),
        }, indent=2, default=str))
        _repair_cross_asset_state_fields(
            path,
            candidates=candidates,
            start=history_start,
            end=end_exclusive,
            repair_only={
                'xs_dispersion__oi_to_volume_7d_z_180d',
                'q_tail_width__oi_to_volume_7d_z_180d',
            },
            bar_phase_minutes=args.bar_phase_minutes,
        )
        _attach_target_free_identity(path, candidates)
        _refresh_feature_coverage(path, args.out_dir / 'feature_coverage.parquet')
        print(json.dumps({
            "event": "surgical_oi_repair_complete", "features": str(path),
            "rows": len(candidates), "side": args.side,
        }))
        return
    path = materialize_features(
        args.out_dir,
        candidates,
        contract,
        history_start,
        end_exclusive,
        bar_phase_minutes=args.bar_phase_minutes,
        full_feature_universe=bool(args.full_feature_universe),
    )
    if not args.full_feature_universe:
        _repair_cross_asset_state_fields(
            path,
            candidates=candidates,
            start=history_start,
            end=end_exclusive,
            bar_phase_minutes=args.bar_phase_minutes,
        )
    # The research-only full-universe branch has just calculated the complete
    # causal engine surface from the same pre-filtered point-in-time panel,
    # including the OI/funding and cross-asset parents.  Recomputing the two
    # legacy frozen-contract repair fields would duplicate the most expensive
    # panel load without changing their value.
    _attach_target_free_identity(path, candidates)
    if not args.full_feature_universe:
        _refresh_feature_coverage(path, args.out_dir / 'feature_coverage.parquet')
    manifest_path = args.out_dir / 'feature_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['complete_universe_attestation'] = universe_attestation
    if args.full_feature_universe:
        manifest['postbuild_cross_asset_repair'] = 'not_run: complete causal engine surface already computed directly'
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({
        "event": "complete", "features": str(path), "rows": len(candidates),
        "side": args.side, "base_fields": len(base_contract[args.side]),
        "severe_context_fields": len(severe_context),
        "requested_unique_fields": len(requested),
        "full_feature_universe": bool(args.full_feature_universe),
    }))


if __name__ == "__main__":
    main()
