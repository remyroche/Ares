from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.fast_funcs import (
    numba_adx,
    numba_atr,
    numba_ewma,
    numba_rolling_max,
    numba_rolling_mean,
    numba_rolling_min,
    numba_rolling_std,
    numba_rsi,
)

LOCATION_FILTER_COLUMNS: tuple[str, ...] = (
    "LOC_01_AboveEMA",
    "LOC_02_BelowEMA",
    "LOC_03_BetweenFastMidEMA",
    "LOC_04_BetweenMidSlowEMA",
    "LOC_05_StackedAboveAllEMAs",
    "LOC_06_StackedBelowAllEMAs",
    "LOC_07_TouchFastEMA_Long",
    "LOC_08_TouchFastEMA_Short",
    "LOC_09_TouchMidEMA_Long",
    "LOC_10_TouchMidEMA_Short",
    "LOC_11_DeepPullbackToSlowEMA_Long",
    "LOC_12_DeepPullbackToSlowEMA_Short",
    "LOC_13_EMAValueZone_Long",
    "LOC_14_EMAValueZone_Short",
    "LOC_20_AboveVWAP",
    "LOC_21_BelowVWAP",
    "LOC_22_AtVWAP_Long",
    "LOC_23_AtVWAP_Short",
    "LOC_24_VWAPPlus1Dev",
    "LOC_25_VWAPMinus1Dev",
    "LOC_26_VWAPPlus2Dev",
    "LOC_27_VWAPMinus2Dev",
    "LOC_28_BetweenVWAPAndPlus1Dev",
    "LOC_29_BetweenVWAPAndMinus1Dev",
    "LOC_30_ReclaimVWAPZone_Long",
    "LOC_31_LoseVWAPZone_Short",
    "LOC_40_UpperQuartileOfRange",
    "LOC_41_LowerQuartileOfRange",
    "LOC_42_MidRange",
    "LOC_43_NearRangeHigh",
    "LOC_44_NearRangeLow",
    "LOC_45_AtRangeBreakoutZone_Long",
    "LOC_46_AtRangeBreakdownZone_Short",
    "LOC_50_AbovePriorHigh",
    "LOC_51_BelowPriorLow",
    "LOC_52_InsidePriorRange",
    "LOC_53_NearPriorHigh",
    "LOC_54_NearPriorLow",
    "LOC_55_AboveLastSwingHigh",
    "LOC_56_BelowLastSwingLow",
    "LOC_57_NearLastSwingHigh",
    "LOC_58_NearLastSwingLow",
    "LOC_59_BetweenLastSwingLowHigh",
    "LOC_70_AboveSessionOpen",
    "LOC_71_BelowSessionOpen",
    "LOC_72_AtSessionOpen_Long",
    "LOC_73_AtSessionOpen_Short",
    "LOC_74_AboveInitialBalanceMid",
    "LOC_75_BelowInitialBalanceMid",
    "LOC_76_NearInitialBalanceHigh",
    "LOC_77_NearInitialBalanceLow",
    "LOC_78_AtSessionHighZone",
    "LOC_79_AtSessionLowZone",
    "LOC_80_UpperHalfOfSessionRange",
    "LOC_81_LowerHalfOfSessionRange",
    "LOC_90_AbovePrevDayHigh",
    "LOC_91_BelowPrevDayLow",
    "LOC_92_InsidePrevDayRange",
    "LOC_93_NearPrevDayHigh",
    "LOC_94_NearPrevDayLow",
    "LOC_95_AbovePrevDayMid",
    "LOC_96_BelowPrevDayMid",
    "LOC_97_NearPrevWeekHigh",
    "LOC_98_NearPrevWeekLow",
    "LOC_99_InsidePrevWeekRange",
    "LOC_110_AboveBBMid",
    "LOC_111_BelowBBMid",
    "LOC_112_AtBBUpper",
    "LOC_113_AtBBLower",
    "LOC_114_OutsideBBUpper",
    "LOC_115_OutsideBBLower",
    "LOC_116_AtKCUpper",
    "LOC_117_AtKCLower",
    "LOC_118_BetweenBBMidAndUpper",
    "LOC_119_BetweenBBMidAndLower",
    "LOC_130_ShallowPullback_Long",
    "LOC_131_DeepPullback_Long",
    "LOC_132_ShallowPullback_Short",
    "LOC_133_DeepPullback_Short",
    "LOC_134_Fib382Zone_Long",
    "LOC_135_Fib50Zone_Long",
    "LOC_136_Fib618Zone_Long",
    "LOC_137_Fib382Zone_Short",
    "LOC_138_Fib50Zone_Short",
    "LOC_139_Fib618Zone_Short",
    "LOC_150_AtPivotResistance",
    "LOC_151_AtPivotSupport",
    "LOC_152_BetweenPivotAndR1",
    "LOC_153_BetweenPivotAndS1",
    "LOC_154_AtLiquidityPoolHigh",
    "LOC_155_AtLiquidityPoolLow",
    "LOC_156_AtUntestedBreakoutLevel",
    "LOC_157_AtUntestedBreakdownLevel",
    "LOC_170_NotTooExtendedAboveEMA",
)

INTRADAY_TRIGGER_COLUMNS: tuple[str, ...] = (
    "LONG_01_WideBullBody",
    "LONG_02_3CloseMomentum",
    "LONG_03_RollingHighBreakout",
    "LONG_04_EMATagCloseAbove",
    "SHORT_04_EMATagCloseBelow",
    "LONG_05_SmallBullContinuation",
    "SHORT_05_SmallBearContinuation",
    "LONG_10_2BarMomentum",
    "SHORT_10_2BarMomentum",
    "LONG_11_3BarPriceAcceleration",
    "SHORT_11_3BarPriceAcceleration",
    "LONG_12_HH_HL_Impulse",
    "SHORT_12_LL_LH_Impulse",
    "LONG_13_BullCloseNearHigh",
    "SHORT_13_BearCloseNearLow",
    "LONG_14_MomentumWithRelVol",
    "SHORT_14_MomentumWithRelVol",
    "LONG_15_MomoIgnition",
    "SHORT_15_MomoIgnition",
    "LONG_20_HighBreakClose",
    "SHORT_20_LowBreakClose",
    "LONG_21_DonchianBreak",
    "SHORT_21_DonchianBreak",
    "LONG_22_OpeningRangeBreak",
    "SHORT_22_OpeningRangeBreak",
    "LONG_23_InsideBarBreak",
    "SHORT_23_InsideBarBreak",
    "LONG_24_OutsideBarResolution",
    "SHORT_24_OutsideBarResolution",
    "LONG_25_NRBreakout",
    "SHORT_25_NRBreakout",
    "LONG_26_SqueezeRelease",
    "SHORT_26_SqueezeRelease",
    "LONG_27_PivotBreak",
    "SHORT_27_PivotBreak",
    "LONG_28_LevelBreakRetestHold",
    "SHORT_28_LevelBreakRetestHold",
    "LONG_30_EMA10_PullbackBounce",
    "SHORT_30_EMA10_PullbackReject",
    "LONG_31_EMA20_PullbackBounce",
    "SHORT_31_EMA20_PullbackReject",
    "LONG_32_EMAStackPullback",
    "SHORT_32_EMAStackPullback",
    "LONG_33_VWAPPullbackHold",
    "SHORT_33_VWAPPullbackReject",
    "LONG_34_BreakoutThenInsideContinuation",
    "SHORT_34_BreakdownThenInsideContinuation",
    "LONG_35_MicroPullbackHigherLow",
    "SHORT_35_MicroPullbackLowerHigh",
    "LONG_36_FlagBreak",
    "SHORT_36_FlagBreak",
    "LONG_37_HighTightFlag",
    "SHORT_37_LowTightFlag",
    "LONG_40_HammerReversal",
    "SHORT_40_ShootingStarReversal",
    "LONG_41_BullEngulf",
    "SHORT_41_BearEngulf",
    "LONG_42_FailedBreakdown",
    "SHORT_42_FailedBreakout",
    "LONG_43_Spring",
    "SHORT_43_Upthrust",
    "LONG_44_OutsideReversalUp",
    "SHORT_44_OutsideReversalDown",
    "LONG_45_3BarReversal",
    "SHORT_45_3BarReversal",
    "LONG_46_StopRunReclaim",
    "SHORT_46_StopRunReject",
    "LONG_50_BBLowerSnapback",
    "SHORT_50_BBUpperSnapback",
    "LONG_51_KCExtensionRevert",
    "SHORT_51_KCExtensionRevert",
    "LONG_52_VWAPStretchRevert",
    "SHORT_52_VWAPStretchRevert",
    "LONG_53_RSIRecovery",
    "SHORT_53_RSIReject",
    "LONG_54_StochCrossFromOS",
    "SHORT_54_StochCrossFromOB",
    "LONG_60_CloseCrossEMA",
    "SHORT_60_CloseCrossEMA",
    "LONG_61_FastCrossMid",
    "SHORT_61_FastCrossMid",
    "LONG_62_PriceReclaimsEMAStack",
    "SHORT_62_PriceLosesEMAStack",
    "LONG_63_EMACompressionExpansion",
    "SHORT_63_EMACompressionExpansion",
    "LONG_70_VWAPCrossHold",
    "SHORT_70_VWAPCrossReject",
    "LONG_71_VWAPReclaimAfterUndercut",
    "SHORT_71_VWAPRejectAfterOvershoot",
    "LONG_72_VWAPTrendContinuation",
    "SHORT_72_VWAPTrendContinuation",
    "LONG_80_RangeLowReversal",
    "SHORT_80_RangeHighReversal",
    "LONG_81_RangeEscape",
    "SHORT_81_RangeEscape",
    "LONG_82_IBHBreak",
    "SHORT_82_IBLBreak",
    "LONG_83_PreviousHighBreak",
    "SHORT_83_PreviousLowBreak",
    "LONG_84_PreviousLowSweepReclaim",
    "SHORT_84_PreviousHighSweepReject",
    "LONG_90_RangeExpansion",
    "SHORT_90_RangeExpansion",
    "LONG_91_TRExpansionBreak",
    "SHORT_91_TRExpansionBreak",
    "LONG_92_CompressionThenExpansion",
    "SHORT_92_CompressionThenExpansion",
    "LONG_93_NR7Expansion",
    "SHORT_93_NR7Expansion",
    "LONG_100_BOS_Up",
    "SHORT_100_BOS_Down",
    "LONG_101_CHOCH_Up",
    "SHORT_101_CHOCH_Down",
    "LONG_102_HigherLowContinuation",
    "SHORT_102_LowerHighContinuation",
    "LONG_103_FlipZoneLong",
    "SHORT_103_FlipZoneShort",
    "LONG_110_LongLowerWickAbsorption",
    "SHORT_110_LongUpperWickAbsorption",
    "LONG_111_BearTrapCandle",
    "SHORT_111_BullTrapCandle",
    "LONG_112_DojiResolveUp",
    "SHORT_112_DojiResolveDown",
    "LONG_113_PinBarBreakUp",
    "SHORT_113_PinBarBreakDown",
    "LONG_120_RSITrendPush",
    "SHORT_120_RSITrendPush",
    "LONG_121_ADX_DI_Long",
    "SHORT_121_ADX_DI_Short",
    "LONG_122_RSIMidlineReclaim",
    "SHORT_122_RSIMidlineLose",
    "LONG_130_DislocationUp",
    "SHORT_130_DislocationDown",
    "LONG_131_DislocationFillHold",
    "SHORT_131_DislocationFillReject",
    "LONG_140_ThreeWhiteSoldiersLite",
    "SHORT_140_ThreeBlackCrowsLite",
    "LONG_141_1_2_3_ReversalUp",
    "SHORT_141_1_2_3_ReversalDown",
    "LONG_142_PauseThenGo",
    "SHORT_142_PauseThenGo",
    "LONG_150_BreakoutQuality",
    "SHORT_150_BreakdownQuality",
    "LONG_151_PullbackQuality",
    "SHORT_151_PullbackQuality",
    "LONG_152_ReversalQuality",
    "SHORT_152_ReversalQuality",
    "LONG_153_SqueezeTrendRelease",
    "SHORT_153_SqueezeTrendRelease",
)

LOC_CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "loc_ema_stack_pos_24",
    "loc_ema_stack_pos_48",
    "loc_vwap_dev_z_24",
    "loc_vwap_dev_z_48",
    "loc_range_pos_24",
    "loc_range_pos_48",
    "loc_prior_bar_pos_24",
    "loc_prior_bar_pos_48",
    "loc_swing_range_pos_24",
    "loc_swing_range_pos_48",
    "loc_session_pos_24",
    "loc_session_pos_48",
    "loc_initial_balance_pos_24",
    "loc_initial_balance_pos_48",
    "loc_prev_day_range_pos_24",
    "loc_prev_day_range_pos_48",
    "loc_prev_week_range_pos_24",
    "loc_prev_week_range_pos_48",
    "loc_bb_channel_pos_24",
    "loc_bb_channel_pos_48",
    "loc_pullback_depth_24",
    "loc_pullback_depth_48",
    "loc_pivot_ladder_pos_24",
    "loc_pivot_ladder_pos_48",
)


PERSISTED_INTRADAY_LIBRARY_COLUMNS: tuple[str, ...] = LOC_CONTINUOUS_COLUMNS


def _new_like(
    ref: pd.Series | pd.DataFrame, values: np.ndarray, dtype: str = "float32"
) -> pd.Series | pd.DataFrame:
    if isinstance(ref, pd.DataFrame):
        return pd.DataFrame(values, index=ref.index, columns=ref.columns, dtype=dtype)
    return pd.Series(values, index=ref.index, dtype=dtype)


def _nan_like(ref: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(ref, pd.DataFrame):
        return pd.DataFrame(
            np.nan, index=ref.index, columns=ref.columns, dtype="float32"
        )
    return pd.Series(np.nan, index=ref.index, dtype="float32")


def trigger_family_from_column(name: str) -> str:
    parts = name.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Malformed trigger column name: {name}")
    token = parts[1]
    if not token.isdigit():
        raise ValueError(f"Malformed trigger numeric token: {name}")
    num = int(token)
    if num < 10:
        return "core_trigger"
    if num < 20:
        return "pure_momentum"
    if num < 30:
        return "breakout"
    if num < 40:
        return "pullback_continuation"
    if num < 50:
        return "reversal_rejection"
    if num < 60:
        return "mean_reversion"
    if num < 70:
        return "ema_cross_reclaim"
    if num < 80:
        return "vwap_trigger"
    if num < 90:
        return "range_session"
    if num < 100:
        return "volatility_expansion"
    if num < 110:
        return "market_structure"
    if num < 120:
        return "wick_absorption"
    if num < 130:
        return "oscillator_confirmed"
    if num < 140:
        return "dislocation"
    if num < 150:
        return "multi_bar_pattern"
    return "hybrid_quality"


def _safe_div(
    a: pd.Series, b: pd.Series | np.ndarray | float, eps: float = 1e-8
) -> pd.Series:
    return (a / np.maximum(b, eps)).astype("float32")


def _ema(s: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    alpha = 2.0 / (n + 1.0)
    return numba_ewma(s, alpha=alpha, adjust=False)


def _sma(s: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    return numba_rolling_mean(s, int(n))


def _stdev(s: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    return numba_rolling_std(s, int(n))


def _rolling_high(s: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    return numba_rolling_max(s, int(n))


def _rolling_low(s: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    return numba_rolling_min(s, int(n))


def _true_range(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    prev_close = close.shift(1)
    tr = np.maximum.reduce(
        [
            (high - low).to_numpy(dtype=np.float32, copy=False),
            (high - prev_close).abs().to_numpy(dtype=np.float32, copy=False),
            (low - prev_close).abs().to_numpy(dtype=np.float32, copy=False),
        ]
    )
    return _new_like(close, tr, dtype="float32")


def _atr(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    n: int,
) -> pd.Series | pd.DataFrame:
    tr = _true_range(high, low, close)
    return numba_ewma(tr, alpha=1.0 / int(n), adjust=False)


def _rsi(close: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    return numba_rsi(close, int(n))


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    ll = _rolling_low(low, n)
    hh = _rolling_high(high, n)
    return (100.0 * _safe_div(close - ll, hh - ll)).astype("float32")


def _stoch_d(stoch_k: pd.Series, smooth: int) -> pd.Series:
    return _sma(stoch_k, smooth).astype("float32")


def _adx(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    n: int,
) -> tuple[
    pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame
]:
    return numba_adx(high, low, close, int(n))


def _distance(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).abs().astype("float32")


def _nr(range_: pd.Series, n: int) -> pd.Series:
    return range_ <= range_.rolling(int(n), min_periods=int(n)).min()


def _bb_width(bb_up: pd.Series, bb_dn: pd.Series, bb_mid: pd.Series) -> pd.Series:
    return _safe_div(bb_up - bb_dn, bb_mid.abs())


def _ema_compression(
    ema_fast: pd.Series | pd.DataFrame,
    ema_mid: pd.Series | pd.DataFrame,
    ema_slow: pd.Series | pd.DataFrame,
    atr: pd.Series | pd.DataFrame,
    thr: float,
) -> pd.Series | pd.DataFrame:
    spread = np.maximum.reduce(
        [
            (ema_fast - ema_mid).abs().to_numpy(dtype=np.float32, copy=False),
            (ema_mid - ema_slow).abs().to_numpy(dtype=np.float32, copy=False),
            (ema_fast - ema_slow).abs().to_numpy(dtype=np.float32, copy=False),
        ]
    )
    spread_obj = _new_like(ema_fast, spread, dtype="float32")
    return _safe_div(spread_obj, atr.astype("float32")) <= float(thr)


def _session_vwap(df: pd.DataFrame, session_key: Optional[str]) -> pd.Series:
    if session_key is None or session_key not in df.columns:
        pv = (df["close"] * df["volume"]).cumsum()
        vv = df["volume"].cumsum()
        return _safe_div(pv.astype("float32"), vv.astype("float32"))
    pv = (df["close"] * df["volume"]).groupby(df[session_key], sort=False).cumsum()
    vv = df["volume"].groupby(df[session_key], sort=False).cumsum()
    return _safe_div(pv.astype("float32"), vv.astype("float32"))


def _session_open(series: pd.Series, session_ids: Optional[pd.Series]) -> pd.Series:
    if session_ids is None:
        return pd.Series(
            np.repeat(series.iloc[0], len(series)),
            index=series.index,
            dtype="float32",
        )
    return series.groupby(session_ids, sort=False).transform("first").astype("float32")


def _session_high(series: pd.Series, session_ids: Optional[pd.Series]) -> pd.Series:
    if session_ids is None:
        return series.cummax().astype("float32")
    return series.groupby(session_ids, sort=False).cummax().astype("float32")


def _session_low(series: pd.Series, session_ids: Optional[pd.Series]) -> pd.Series:
    if session_ids is None:
        return series.cummin().astype("float32")
    return series.groupby(session_ids, sort=False).cummin().astype("float32")


def _opening_range(
    high: pd.Series,
    low: pd.Series,
    session_ids: Optional[pd.Series],
    bars: int = 3,
) -> tuple[pd.Series, pd.Series]:
    if session_ids is None:
        mask = np.arange(len(high)) < int(bars)
        orh = pd.Series(np.where(mask, high, np.nan), index=high.index).cummax()
        orl = pd.Series(np.where(mask, low, np.nan), index=low.index).cummin()
        return orh.ffill().astype("float32"), orl.ffill().astype("float32")

    pos = high.groupby(session_ids, sort=False).cumcount()
    # Mask data outside the opening range window
    h_in = high.where(pos < int(bars))
    l_in = low.where(pos < int(bars))

    # Use cummax to be causal. The value at the end of the window (pos == bars-1)
    # will be the true opening range high/low.
    orh = h_in.groupby(session_ids, sort=False).cummax()
    orl = l_in.groupby(session_ids, sort=False).cummin()

    # Forward fill the opening range value to the rest of the session
    orh = orh.groupby(session_ids, sort=False).ffill()
    orl = orl.groupby(session_ids, sort=False).ffill()

    return orh.astype("float32"), orl.astype("float32")


def _pivot_points(
    prev_high: pd.Series, prev_low: pd.Series, prev_close: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pivot = ((prev_high + prev_low + prev_close) / 3.0).astype("float32")
    r1 = (2.0 * pivot - prev_low).astype("float32")
    s1 = (2.0 * pivot - prev_high).astype("float32")
    return pivot, r1, s1


def _fractals(
    high: pd.Series, low: pd.Series, n: int = 2
) -> tuple[pd.Series, pd.Series]:
    n_int = int(n)
    win = 2 * n_int + 1
    # Purely causal rolling max/min (bar t is the LAST of the window)
    sh_roll = high.rolling(win, min_periods=win).max()
    sl_roll = low.rolling(win, min_periods=win).min()

    # A peak at t-n is confirmed at t if high[t-n] is the max of [t-2n, t]
    is_sh = high.shift(n_int) == sh_roll
    is_sl = low.shift(n_int) == sl_roll

    swing_high = high.shift(n_int).where(is_sh)
    swing_low = low.shift(n_int).where(is_sl)
    return swing_high.astype("float32"), swing_low.astype("float32")


def _last_valid(series: pd.Series) -> pd.Series:
    return series.ffill().astype("float32")


def _prev_valid(series: pd.Series) -> pd.Series:
    return series.where(series.notna()).ffill().shift(1).astype("float32")


def _flag_channels(
    high: pd.Series, low: pd.Series, flag_len: int
) -> tuple[pd.Series, pd.Series]:
    upper = (
        high.rolling(flag_len, min_periods=flag_len).max().shift(1).astype("float32")
    )
    lower = low.rolling(flag_len, min_periods=flag_len).min().shift(1).astype("float32")
    return upper, lower


def _impulse_up(
    close: pd.Series, atr: pd.Series, impulse_len: int, impulse_thr: float
) -> pd.Series:
    return _safe_div(close - close.shift(int(impulse_len)), atr) >= float(impulse_thr)


def _impulse_down(
    close: pd.Series, atr: pd.Series, impulse_len: int, impulse_thr: float
) -> pd.Series:
    return _safe_div(close.shift(int(impulse_len)) - close, atr) >= float(impulse_thr)


def _pullback_channel_down(high: pd.Series, flag_len: int) -> pd.Series:
    rh = high.rolling(flag_len, min_periods=flag_len).max()
    return rh.diff().rolling(flag_len, min_periods=flag_len).mean() < 0


def _pullback_channel_up(low: pd.Series, flag_len: int) -> pd.Series:
    rl = low.rolling(flag_len, min_periods=flag_len).min()
    return rl.diff().rolling(flag_len, min_periods=flag_len).mean() > 0


@dataclass(frozen=True)
class _NormalizedInput:
    source_is_dict: bool
    df: pd.DataFrame
    o: pd.Series | pd.DataFrame
    h: pd.Series | pd.DataFrame
    l: pd.Series | pd.DataFrame
    c: pd.Series | pd.DataFrame
    v: pd.Series | pd.DataFrame
    session_key: Optional[str]
    session_ids: Optional[pd.Series]


class _PrimitiveCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], pd.Series] = {}

    def shift(
        self, key: str, s: pd.Series | pd.DataFrame, n: int
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("shift", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = s.shift(int(n)).astype("float32")
        return self._cache[cache_key]

    def ema(
        self, key: str, s: pd.Series | pd.DataFrame, n: int
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("ema", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = _ema(s, int(n))
        return self._cache[cache_key]

    def sma(
        self, key: str, s: pd.Series | pd.DataFrame, n: int
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("sma", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = _sma(s, int(n))
        return self._cache[cache_key]

    def stdev(
        self, key: str, s: pd.Series | pd.DataFrame, n: int
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("stdev", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = _stdev(s, int(n))
        return self._cache[cache_key]

    def rolling_high(
        self, key: str, s: pd.Series | pd.DataFrame, n: int
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("roll_high", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = _rolling_high(s, int(n))
        return self._cache[cache_key]

    def rolling_low(
        self, key: str, s: pd.Series | pd.DataFrame, n: int
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("roll_low", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = _rolling_low(s, int(n))
        return self._cache[cache_key]

    def atr(
        self,
        key: str,
        high: pd.Series | pd.DataFrame,
        low: pd.Series | pd.DataFrame,
        close: pd.Series | pd.DataFrame,
        n: int,
    ) -> pd.Series | pd.DataFrame:
        cache_key = ("atr", key, int(n))
        if cache_key not in self._cache:
            self._cache[cache_key] = _atr(high, low, close, int(n))
        return self._cache[cache_key]


def _coerce_panel(
    x: Any,
    index: Optional[pd.Index] = None,
    columns: Optional[pd.Index] = None,
) -> pd.Series | pd.DataFrame:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        obj = x
    else:
        obj = pd.Series(x)
    if index is not None and not obj.index.equals(index):
        obj = obj.reindex(index)
    if isinstance(obj, pd.DataFrame) and columns is not None:
        obj = obj.reindex(columns=columns)
    return obj.astype("float32")


def _normalize_input(
    data: pd.DataFrame | Mapping[str, Any],
    session_key_name: str,
) -> _NormalizedInput:
    required = ("open", "high", "low", "close", "volume")
    if isinstance(data, pd.DataFrame):
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = data
        session_key = session_key_name if session_key_name in df.columns else None
        session_ids = df[session_key] if session_key is not None else None
        return _NormalizedInput(
            source_is_dict=False,
            df=df,
            o=df["open"].astype("float32"),
            h=df["high"].astype("float32"),
            l=df["low"].astype("float32"),
            c=df["close"].astype("float32"),
            v=df["volume"].astype("float32"),
            session_key=session_key,
            session_ids=session_ids,
        )

    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    close_obj = _coerce_panel(data["close"])
    if isinstance(close_obj, pd.DataFrame):
        base_index = close_obj.index
        base_columns = close_obj.columns
        o = _coerce_panel(data["open"], base_index, base_columns)
        h = _coerce_panel(data["high"], base_index, base_columns)
        l = _coerce_panel(data["low"], base_index, base_columns)
        c = _coerce_panel(data["close"], base_index, base_columns)
        v = _coerce_panel(data["volume"], base_index, base_columns)
        session_key = session_key_name if session_key_name in data else None
        session_ids = (
            _coerce_panel(data[session_key], base_index) if session_key else None
        )
        if isinstance(session_ids, pd.DataFrame):
            session_ids = session_ids.iloc[:, 0]
        df = pd.DataFrame(index=base_index)
        return _NormalizedInput(
            source_is_dict=True,
            df=df,
            o=o,
            h=h,
            l=l,
            c=c,
            v=v,
            session_key=session_key,
            session_ids=session_ids,
        )

    base_index = close_obj.index
    df = pd.DataFrame(
        {k: _coerce_panel(data[k], base_index) for k in required}, index=base_index
    )
    session_key = session_key_name if session_key_name in data else None
    session_ids = _coerce_panel(data[session_key], base_index) if session_key else None
    if isinstance(session_ids, pd.DataFrame):
        session_ids = session_ids.iloc[:, 0]
    if session_ids is not None:
        df[session_key_name] = session_ids
    return _NormalizedInput(
        source_is_dict=True,
        df=df,
        o=df["open"],
        h=df["high"],
        l=df["low"],
        c=df["close"],
        v=df["volume"],
        session_key=session_key,
        session_ids=session_ids,
    )


def build_intraday_crypto_library(
    df: pd.DataFrame | Mapping[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame | Dict[str, pd.Series]:
    p: Dict[str, Any] = {
        "eps": 1e-8,
        "atr_len": 14,
        "ema_fast_len": 10,
        "ema_mid_len": 20,
        "ema_slow_len": 50,
        "bb_len": 20,
        "bb_mult": 2.0,
        "kc_len": 20,
        "kc_mult": 1.5,
        "rsi_len": 14,
        "stoch_len": 14,
        "stoch_smooth": 3,
        "adx_len": 14,
        "vol_len": 20,
        "lookback": 20,
        "dc_len": 20,
        "opening_range_bars": 3,
        "initial_balance_bars": 12,
        "close_pos_thr": 0.2,
        "close_loc_thr": 0.6,
        "range_atr_min": 1.0,
        "range_atr_max": 0.8,
        "range_atr_hi": 1.5,
        "range_atr_break": 1.2,
        "threshold": 1.0,
        "body_ratio_min": 0.5,
        "body_ratio_max": 0.4,
        "body_ratio_hi": 0.7,
        "wick_ratio_min": 0.4,
        "wick_ratio_hi": 0.65,
        "rel_vol_min": 1.2,
        "rel_vol_hi": 2.0,
        "rsi_os": 30.0,
        "rsi_ob": 70.0,
        "rsi_trend_min": 55.0,
        "rsi_trend_max": 45.0,
        "stoch_os": 20.0,
        "stoch_ob": 80.0,
        "tr_exp_thr": 1.5,
        "bb_width_thr": 0.06,
        "reversal_close_frac": 0.6,
        "absorb_close_frac": 0.6,
        "doji_body_thr": 0.15,
        "pause_body_thr": 0.3,
        "vwap_dev_mult": 2.0,
        "vwap_pullback_buffer": 0.15,
        "dislocation_thr": 0.75,
        "stoprun_len": 10,
        "near_level_atr": 0.25,
        "level_buffer_atr": 0.20,
        "comp_thr": 0.35,
        "impulse_len": 8,
        "impulse_thr": 2.0,
        "ema_tag_len": 20,
        "session_key": "session_id",
    }
    if params:
        p.update(params)

    normalized = _normalize_input(df, str(p["session_key"]))

    out: Dict[str, Any] = {}
    o = normalized.o
    h = normalized.h
    l = normalized.l
    c = normalized.c
    v = normalized.v
    base_df = normalized.df
    eps = float(p["eps"])
    session_key = normalized.session_key
    session_ids = normalized.session_ids
    cache = _PrimitiveCache()

    out["range"] = (h - l).astype("float32")
    out["body"] = (c - o).abs().astype("float32")
    out["upper_wick"] = (
        h - np.maximum(o.to_numpy(copy=False), c.to_numpy(copy=False))
    ).astype("float32")
    out["lower_wick"] = (
        np.minimum(o.to_numpy(copy=False), c.to_numpy(copy=False)) - l
    ).astype("float32")
    out["body_ratio"] = _safe_div(out["body"], out["range"], eps)
    out["upper_wick_ratio"] = _safe_div(out["upper_wick"], out["range"], eps)
    out["lower_wick_ratio"] = _safe_div(out["lower_wick"], out["range"], eps)
    out["tr"] = _true_range(h, l, c)
    out["atr"] = _atr(h, l, c, int(p["atr_len"]))
    out["range_atr"] = _safe_div(out["range"], out["atr"], eps)
    out["true_range_atr"] = _safe_div(out["tr"], out["atr"], eps)
    out["bull"] = (c > o).astype("int8")
    out["bear"] = (c < o).astype("int8")
    out["inside_bar"] = (
        (h <= cache.shift("h", h, 1)) & (l >= cache.shift("l", l, 1))
    ).astype("int8")
    out["outside_bar"] = (
        (h >= cache.shift("h", h, 1)) & (l <= cache.shift("l", l, 1))
    ).astype("int8")
    out["close_near_high"] = (
        _safe_div(h - c, out["range"], eps) <= float(p["close_pos_thr"])
    ).astype("int8")
    out["close_near_low"] = (
        _safe_div(c - l, out["range"], eps) <= float(p["close_pos_thr"])
    ).astype("int8")
    out["mid_close_high"] = (
        _safe_div(c - l, out["range"], eps) >= float(p["close_loc_thr"])
    ).astype("int8")
    out["mid_close_low"] = (
        _safe_div(h - c, out["range"], eps) >= float(p["close_loc_thr"])
    ).astype("int8")
    out["hh"] = (h > cache.shift("h", h, 1)).astype("int8")
    out["hl"] = (l > cache.shift("l", l, 1)).astype("int8")
    out["lh"] = (h < cache.shift("h", h, 1)).astype("int8")
    out["ll"] = (l < cache.shift("l", l, 1)).astype("int8")
    out["ema_fast"] = cache.ema("c", c, int(p["ema_fast_len"]))
    out["ema_mid"] = cache.ema("c", c, int(p["ema_mid_len"]))
    out["ema_slow"] = cache.ema("c", c, int(p["ema_slow_len"]))
    out["ema_tag"] = cache.ema("c", c, int(p["ema_tag_len"]))
    if isinstance(c, pd.DataFrame):
        if session_ids is None:
            pv = (c * v).cumsum()
            vv = v.cumsum()
        else:
            pv = (c * v).groupby(session_ids, sort=False).cumsum()
            vv = v.groupby(session_ids, sort=False).cumsum()
        out["vwap_session"] = _safe_div(pv.astype("float32"), vv.astype("float32"), eps)
    else:
        out["vwap_session"] = _session_vwap(base_df, session_key)
    if session_ids is None:
        m = c.expanding().mean()
        m2 = c.pow(2).expanding().mean()
        out["session_stdev"] = np.sqrt((m2 - m.pow(2)).clip(lower=0.0)).astype(
            "float32"
        )
    else:
        # Optimized session_stdev: compute mean and mean of squares using cumsum and cumcount
        grp = c.groupby(session_ids, sort=False)
        cum_sum = grp.cumsum()
        cum_sum_sq = c.pow(2).groupby(session_ids, sort=False).cumsum()
        cum_count = grp.cumcount() + 1
        if isinstance(c, pd.DataFrame):
            m = cum_sum.div(cum_count, axis=0)
            m2 = cum_sum_sq.div(cum_count, axis=0)
        else:
            m = cum_sum / cum_count
            m2 = cum_sum_sq / cum_count
        out["session_stdev"] = np.sqrt((m2 - m.pow(2)).clip(lower=0.0)).astype(
            "float32"
        )
    out["rolling_high_n"] = cache.rolling_high("h", h, int(p["lookback"]))
    out["rolling_low_n"] = cache.rolling_low("l", l, int(p["lookback"]))
    out["donchian_high_n"] = cache.rolling_high("h", h, int(p["dc_len"]))
    out["donchian_low_n"] = cache.rolling_low("l", l, int(p["dc_len"]))
    out["bb_mid"] = cache.sma("c", c, int(p["bb_len"]))
    out["bb_up"] = (
        out["bb_mid"] + float(p["bb_mult"]) * cache.stdev("c", c, int(p["bb_len"]))
    ).astype("float32")
    out["bb_dn"] = (
        out["bb_mid"] - float(p["bb_mult"]) * cache.stdev("c", c, int(p["bb_len"]))
    ).astype("float32")
    out["bb_width"] = _bb_width(out["bb_up"], out["bb_dn"], out["bb_mid"])
    out["kc_mid"] = cache.ema("c", c, int(p["kc_len"]))
    out["kc_up"] = (
        out["kc_mid"]
        + float(p["kc_mult"]) * cache.atr("ohlc", h, l, c, int(p["kc_len"]))
    ).astype("float32")
    out["kc_dn"] = (
        out["kc_mid"]
        - float(p["kc_mult"]) * cache.atr("ohlc", h, l, c, int(p["kc_len"]))
    ).astype("float32")
    out["kc_contained"] = (
        (out["bb_up"] <= out["kc_up"]) & (out["bb_dn"] >= out["kc_dn"])
    ).astype("int8")
    out["rsi"] = _rsi(c, int(p["rsi_len"]))
    out["stoch_k"] = _stoch_k(h, l, c, int(p["stoch_len"]))
    out["stoch_d"] = _stoch_d(out["stoch_k"], int(p["stoch_smooth"]))
    out["adx"], out["plus_di"], out["minus_di"] = _adx(h, l, c, int(p["adx_len"]))
    out["vol_ma"] = cache.sma("v", v, int(p["vol_len"]))
    out["rel_vol"] = _safe_div(v, out["vol_ma"], eps)
    out["trend_up_ema"] = (
        (out["ema_fast"] > out["ema_mid"]) & (out["ema_mid"] > out["ema_slow"])
    ).astype("int8")
    out["trend_dn_ema"] = (
        (out["ema_fast"] < out["ema_mid"]) & (out["ema_mid"] < out["ema_slow"])
    ).astype("int8")
    out["slope_up"] = (
        out["ema_mid"] > cache.shift("ema_mid", out["ema_mid"], 1)
    ).astype("int8")
    out["slope_dn"] = (
        out["ema_mid"] < cache.shift("ema_mid", out["ema_mid"], 1)
    ).astype("int8")
    out["above_vwap"] = (c > out["vwap_session"]).astype("int8")
    out["below_vwap"] = (c < out["vwap_session"]).astype("int8")
    out["expansion_regime"] = (out["range_atr"] >= float(p["range_atr_min"])).astype(
        "int8"
    )
    out["compression_reg"] = (out["range_atr"] <= float(p["range_atr_max"])).astype(
        "int8"
    )
    out["trend_regime"] = (out["adx"] >= 20.0).astype("int8")
    out["mr_regime"] = (out["adx"] <= 18.0).astype("int8")
    out["session_open"] = _session_open(o, session_ids)
    out["session_high"] = _session_high(h, session_ids)
    out["session_low"] = _session_low(l, session_ids)
    out["session_range_high"] = out["session_high"]
    out["session_range_low"] = out["session_low"]
    out["opening_range_high"], out["opening_range_low"] = _opening_range(
        h, l, session_ids, int(p["opening_range_bars"])
    )
    out["initial_balance_high"], out["initial_balance_low"] = _opening_range(
        h, l, session_ids, int(p["initial_balance_bars"])
    )
    out["initial_balance_mid"] = (
        (out["initial_balance_high"] + out["initial_balance_low"]) / 2.0
    ).astype("float32")

    for col in (
        "prev_day_high",
        "prev_day_low",
        "prev_week_high",
        "prev_week_low",
        "support_level",
        "resistance_level",
        "prior_support",
        "prior_resistance",
        "equal_highs_level",
        "equal_lows_level",
        "fresh_breakout_level",
        "fresh_breakdown_level",
        "key_level",
    ):
        if col in base_df.columns:
            out[col] = base_df[col].astype("float32")
        else:
            out[col] = _nan_like(c)

    prev_close = cache.shift("c", c, 1)
    prev_day_high = out["prev_day_high"].fillna(
        cache.shift("roll_high_24", cache.rolling_high("h", h, 24), 1)
    )
    prev_day_low = out["prev_day_low"].fillna(
        cache.shift("roll_low_24", cache.rolling_low("l", l, 24), 1)
    )
    out["prev_day_high"] = prev_day_high.astype("float32")
    out["prev_day_low"] = prev_day_low.astype("float32")
    prev_week_high = out["prev_week_high"].fillna(
        cache.shift("roll_high_168", cache.rolling_high("h", h, 24 * 7), 1)
    )
    prev_week_low = out["prev_week_low"].fillna(
        cache.shift("roll_low_168", cache.rolling_low("l", l, 24 * 7), 1)
    )
    out["prev_week_high"] = prev_week_high.astype("float32")
    out["prev_week_low"] = prev_week_low.astype("float32")
    pivot, pivot_r1, pivot_s1 = _pivot_points(prev_day_high, prev_day_low, prev_close)
    out["pivot"] = pivot
    out["pivot_r1"] = pivot_r1
    out["pivot_s1"] = pivot_s1

    swing_high_raw, swing_low_raw = _fractals(
        h, l, int(p["last_pivot_len"]) if "last_pivot_len" in p else 2
    )
    out["last_swing_high"] = _last_valid(swing_high_raw)
    out["last_swing_low"] = _last_valid(swing_low_raw)
    out["prior_swing_high"] = _prev_valid(swing_high_raw)
    out["prior_swing_low"] = _prev_valid(swing_low_raw)
    out["last_lower_high"] = out["last_swing_high"]
    out["last_higher_low"] = out["last_swing_low"]

    out["close_pos_in_range"] = _safe_div(
        c - out["rolling_low_n"], out["rolling_high_n"] - out["rolling_low_n"], eps
    )
    approx_next_res = _new_like(
        c,
        np.maximum.reduce(
            [
                out["prev_day_high"].to_numpy(copy=False),
                out["last_swing_high"].to_numpy(copy=False),
                out["rolling_high_n"].to_numpy(copy=False),
            ]
        ),
    )
    approx_next_sup = _new_like(
        c,
        np.minimum.reduce(
            [
                out["prev_day_low"].to_numpy(copy=False),
                out["last_swing_low"].to_numpy(copy=False),
                out["rolling_low_n"].to_numpy(copy=False),
            ]
        ),
    )
    out["distance_to_next_resistance"] = (
        (approx_next_res - c).clip(lower=0).astype("float32")
    )
    out["distance_to_next_support"] = (
        (c - approx_next_sup).clip(lower=0).astype("float32")
    )
    out["nr_n"] = _nr(out["range"], int(p["lookback"])).astype("int8")
    out["nr7"] = (out["range"] <= out["range"].rolling(7, min_periods=7).min()).astype(
        "int8"
    )
    out["ema_compression"] = _ema_compression(
        out["ema_fast"],
        out["ema_mid"],
        out["ema_slow"],
        out["atr"],
        float(p["comp_thr"]),
    ).astype("int8")
    out["flag_upper_trendline"], out["flag_lower_trendline"] = _flag_channels(h, l, 8)
    out["impulse_up"] = _impulse_up(
        c, out["atr"], int(p["impulse_len"]), float(p["impulse_thr"])
    ).astype("int8")
    out["impulse_down"] = _impulse_down(
        c, out["atr"], int(p["impulse_len"]), float(p["impulse_thr"])
    ).astype("int8")
    out["pullback_channel_down"] = _pullback_channel_down(h, 8).astype("int8")
    out["pullback_channel_up"] = _pullback_channel_up(l, 8).astype("int8")
    out["vwap_dev_1"] = out["session_stdev"].astype("float32")
    out["vwap_dev_2"] = (2.0 * out["session_stdev"]).astype("float32")

    impulse_origin = cache.shift("c", c, int(p["impulse_len"]))
    out["impulse_origin"] = impulse_origin
    out["impulse_origin_short"] = impulse_origin
    impulse_move = (c - impulse_origin).abs().astype("float32")
    out["pullback_from_last_impulse_pct"] = _safe_div(
        (c - cache.rolling_high("c", c, int(p["impulse_len"]))).abs(),
        np.maximum(impulse_move, eps),
        eps,
    )
    hh_imp = cache.rolling_high("h", h, int(p["impulse_len"]))
    ll_imp = cache.rolling_low("l", l, int(p["impulse_len"]))
    span_imp = (hh_imp - ll_imp).astype("float32")
    out["fib_382_low"] = (hh_imp - 0.45 * span_imp).astype("float32")
    out["fib_382_high"] = (hh_imp - 0.30 * span_imp).astype("float32")
    out["fib_50_low"] = (hh_imp - 0.55 * span_imp).astype("float32")
    out["fib_50_high"] = (hh_imp - 0.45 * span_imp).astype("float32")
    out["fib_618_low"] = (hh_imp - 0.70 * span_imp).astype("float32")
    out["fib_618_high"] = (hh_imp - 0.55 * span_imp).astype("float32")
    out["fib_382_low_short"] = (ll_imp + 0.30 * span_imp).astype("float32")
    out["fib_382_high_short"] = (ll_imp + 0.45 * span_imp).astype("float32")
    out["fib_50_low_short"] = (ll_imp + 0.45 * span_imp).astype("float32")
    out["fib_50_high_short"] = (ll_imp + 0.55 * span_imp).astype("float32")
    out["fib_618_low_short"] = (ll_imp + 0.55 * span_imp).astype("float32")
    out["fib_618_high_short"] = (ll_imp + 0.70 * span_imp).astype("float32")

    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_mid = ((prev_h + prev_l) / 2.0).astype("float32")

    for lb in (24, 48):
        ema_fast_lb = _ema(c, max(4, lb // 3))
        ema_mid_lb = _ema(c, max(8, lb // 2))
        ema_slow_lb = _ema(c, lb)
        denom_ema = np.maximum((ema_fast_lb - ema_slow_lb).abs(), out["atr"])
        out[f"loc_ema_stack_pos_{lb}"] = _safe_div(
            c - ema_mid_lb, denom_ema, eps
        ).astype("float32")

        out[f"loc_vwap_dev_z_{lb}"] = _safe_div(
            c - out["vwap_session"],
            np.maximum(out["session_stdev"], out["atr"] * 0.5),
            eps,
        ).astype("float32")

        rh_lb = _rolling_high(h, lb)
        rl_lb = _rolling_low(l, lb)
        out[f"loc_range_pos_{lb}"] = _safe_div(c - rl_lb, rh_lb - rl_lb, eps).astype(
            "float32"
        )

        out[f"loc_prior_bar_pos_{lb}"] = _safe_div(
            c - prev_mid, prev_h - prev_l, eps
        ).astype("float32")

        out[f"loc_swing_range_pos_{lb}"] = _safe_div(
            c - out["last_swing_low"],
            out["last_swing_high"] - out["last_swing_low"],
            eps,
        ).astype("float32")

        out[f"loc_session_pos_{lb}"] = _safe_div(
            c - out["session_low"],
            out["session_high"] - out["session_low"],
            eps,
        ).astype("float32")

        out[f"loc_initial_balance_pos_{lb}"] = _safe_div(
            c - out["initial_balance_low"],
            out["initial_balance_high"] - out["initial_balance_low"],
            eps,
        ).astype("float32")

        out[f"loc_prev_day_range_pos_{lb}"] = _safe_div(
            c - out["prev_day_low"],
            out["prev_day_high"] - out["prev_day_low"],
            eps,
        ).astype("float32")

        out[f"loc_prev_week_range_pos_{lb}"] = _safe_div(
            c - out["prev_week_low"],
            out["prev_week_high"] - out["prev_week_low"],
            eps,
        ).astype("float32")

        bb_mid_lb = _sma(c, lb)
        bb_std_lb = _stdev(c, lb)
        bb_up_lb = (bb_mid_lb + 2.0 * bb_std_lb).astype("float32")
        bb_dn_lb = (bb_mid_lb - 2.0 * bb_std_lb).astype("float32")
        out[f"loc_bb_channel_pos_{lb}"] = _safe_div(
            c - bb_dn_lb, bb_up_lb - bb_dn_lb, eps
        ).astype("float32")

        hh_lb = _rolling_high(h, lb)
        ll_lb = _rolling_low(l, lb)
        span_lb = (hh_lb - ll_lb).astype("float32")
        pull_from_high = _safe_div(hh_lb - c, span_lb, eps)
        pull_from_low = _safe_div(c - ll_lb, span_lb, eps)
        out[f"loc_pullback_depth_{lb}"] = _new_like(
            c,
            np.where(
                c.to_numpy(copy=False) >= (ll_lb + 0.5 * span_lb).to_numpy(copy=False),
                pull_from_high.to_numpy(copy=False),
                pull_from_low.to_numpy(copy=False),
            ),
            dtype="float32",
        )

        out[f"loc_pivot_ladder_pos_{lb}"] = _safe_div(
            c - out["pivot"], out["pivot_r1"] - out["pivot_s1"], eps
        ).astype("float32")

    if isinstance(c, pd.DataFrame):
        if normalized.source_is_dict:
            return {
                col: out[col].astype("float32")
                for col in PERSISTED_INTRADAY_LIBRARY_COLUMNS
                if col in out
            }

        stacked = {
            name: out[name].stack(dropna=False).astype("float32")
            for name in PERSISTED_INTRADAY_LIBRARY_COLUMNS
            if name in out
        }
        result = pd.DataFrame(stacked, index=next(iter(stacked.values())).index)
        result.index.names = ["timestamp", "symbol"]
        return result

    result = pd.DataFrame(out, index=base_df.index)
    result = result.loc[
        :, [col for col in PERSISTED_INTRADAY_LIBRARY_COLUMNS if col in result.columns]
    ]
    if normalized.source_is_dict:
        return {col: result[col] for col in result.columns}
    return result
