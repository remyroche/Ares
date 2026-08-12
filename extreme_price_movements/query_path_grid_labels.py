"""Vectorised exact-H12 first-touch primitives for query-grade grids.

This is deliberately label-only: all outputs are realised after entry and may
be used only for training/query diagnostics after ``entry + 12h``.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def first_touch_grid_h12(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         starts: np.ndarray, entry: np.ndarray, atr: np.ndarray,
                         side: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return favourable/adverse first-touch minutes and terminal ATR PnL.

    Arrays have shape ``(rows, thresholds)``; -1 means no touch.  A bar that
    hits both thresholds gets the same minute in both arrays.  Consumers apply
    the declared adverse tie break, keeping all contracts identical.
    """
    n=len(starts); k=len(thresholds); favorable=np.full((n,k),-1,np.int16); adverse=np.full((n,k),-1,np.int16); terminal=np.full(n,np.nan,np.float32)
    for row in range(n):
        start=starts[row]; e=entry[row]; a=atr[row]
        if start<0 or start+720>len(close) or not np.isfinite(e) or not np.isfinite(a) or e<=0. or a<=0.: continue
        complete=True
        for offset in range(720):
            pos=start+offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]): complete=False; break
            if side[row]>0.:
                up=(high[pos]-e)/a; down=(e-low[pos])/a
            else:
                up=(e-low[pos])/a; down=(high[pos]-e)/a
            for j in range(k):
                threshold=thresholds[j]
                if favorable[row,j]<0 and up>=threshold: favorable[row,j]=offset+1
                if adverse[row,j]<0 and down>=threshold: adverse[row,j]=offset+1
        if complete: terminal[row]=side[row]*(close[start+719]-e)/a
        else:
            for j in range(k): favorable[row,j]=-1; adverse[row,j]=-1
    return favorable,adverse,terminal


@njit(cache=True)
def first_touch_grid_horizon(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                             starts: np.ndarray, entry: np.ndarray, atr: np.ndarray,
                             side: np.ndarray, thresholds: np.ndarray,
                             horizon_bars: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolution-agnostic first-touch proxy; offset remains 1-indexed bars."""
    n=len(starts); k=len(thresholds); favorable=np.full((n,k),-1,np.int16); adverse=np.full((n,k),-1,np.int16); terminal=np.full(n,np.nan,np.float32)
    for row in range(n):
        start=starts[row]; e=entry[row]; a=atr[row]
        if start<0 or start+horizon_bars>len(close) or not np.isfinite(e) or not np.isfinite(a) or e<=0. or a<=0.: continue
        complete=True
        for offset in range(horizon_bars):
            pos=start+offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]): complete=False; break
            if side[row]>0.: up=(high[pos]-e)/a; down=(e-low[pos])/a
            else: up=(e-low[pos])/a; down=(high[pos]-e)/a
            for j in range(k):
                threshold=thresholds[j]
                if favorable[row,j]<0 and up>=threshold: favorable[row,j]=offset+1
                if adverse[row,j]<0 and down>=threshold: adverse[row,j]=offset+1
        if complete: terminal[row]=side[row]*(close[start+horizon_bars-1]-e)/a
        else:
            for j in range(k): favorable[row,j]=-1; adverse[row,j]=-1
    return favorable,adverse,terminal


def contract_event(favorable_minute: np.ndarray, adverse_minute: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return favourable-first, adverse-first and timeout flags with adverse ties."""
    fav=np.asarray(favorable_minute); adv=np.asarray(adverse_minute)
    favorable=(fav>=0)&((adv<0)|(fav<adv)); adverse=(adv>=0)&((fav<0)|(adv<=fav)); timeout=~favorable&~adverse
    return favorable,adverse,timeout


@njit(cache=True)
def path_extrema_h12(high: np.ndarray, low: np.ndarray, starts: np.ndarray,
                     entry: np.ndarray, atr: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum favourable/adverse H12 excursion in ATR units.

    This exists for the absolute-bps grade family, whose thresholds cannot be
    inferred safely from a finite ATR first-touch grid.
    """
    n=len(starts); mfe=np.full(n,np.nan,np.float32); mae=np.full(n,np.nan,np.float32)
    for row in range(n):
        start=starts[row]; e=entry[row]; a=atr[row]
        if start<0 or start+720>len(high) or not np.isfinite(e) or not np.isfinite(a) or e<=0. or a<=0.: continue
        best=0.; worst=0.; complete=True
        for offset in range(720):
            pos=start+offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]): complete=False; break
            if side[row]>0.:
                up=(high[pos]-e)/a; down=(e-low[pos])/a
            else:
                up=(e-low[pos])/a; down=(high[pos]-e)/a
            if up>best: best=up
            if down>worst: worst=down
        if complete: mfe[row]=best; mae[row]=worst
    return mfe,mae


@njit(cache=True)
def path_extrema_horizon(high: np.ndarray, low: np.ndarray, starts: np.ndarray,
                         entry: np.ndarray, atr: np.ndarray, side: np.ndarray,
                         horizon_bars: int) -> tuple[np.ndarray, np.ndarray]:
    n=len(starts); mfe=np.full(n,np.nan,np.float32); mae=np.full(n,np.nan,np.float32)
    for row in range(n):
        start=starts[row]; e=entry[row]; a=atr[row]
        if start<0 or start+horizon_bars>len(high) or not np.isfinite(e) or not np.isfinite(a) or e<=0. or a<=0.: continue
        best=0.; worst=0.; complete=True
        for offset in range(horizon_bars):
            pos=start+offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]): complete=False; break
            if side[row]>0.: up=(high[pos]-e)/a; down=(e-low[pos])/a
            else: up=(e-low[pos])/a; down=(high[pos]-e)/a
            if up>best: best=up
            if down>worst: worst=down
        if complete: mfe[row]=best; mae[row]=worst
    return mfe,mae


def triple_barrier_outcome_bps(*, favorable_minute: np.ndarray, adverse_minute: np.ndarray,
                               terminal_atr: np.ndarray, atr_bps: np.ndarray,
                               upper_atr: float, lower_atr: float,
                               cost_bps: float = 100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact contract gross/net bps and event code (0=upper,1=lower,2=TO).

    The terminal path return is used only for a genuine H12 timeout; barriers
    settle at their declared ATR threshold.  Cost is subtracted exactly once.
    """
    upper,lower,timeout=contract_event(favorable_minute,adverse_minute)
    atr_bps=np.asarray(atr_bps,dtype=float); terminal=np.asarray(terminal_atr,dtype=float)
    gross=np.where(upper,upper_atr*atr_bps,np.where(lower,-lower_atr*atr_bps,terminal*atr_bps))
    event=np.where(upper,0,np.where(lower,1,2)).astype(np.int8)
    return gross, gross-float(cost_bps), event
