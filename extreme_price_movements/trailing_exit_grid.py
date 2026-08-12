"""Numba-accelerated H12 stop/trailing-profit grid for frozen-score replay."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def simulate_h12_stop_trailing_grid(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                    starts: np.ndarray, entry: np.ndarray, atr: np.ndarray,
                                    side: np.ndarray, stop_atr: np.ndarray,
                                    activation_atr: np.ndarray, giveback_atr: np.ndarray,
                                    horizon_bars: int = 720) -> np.ndarray:
    """Return gross ATR outcomes, shape (rows, stops, activations, givebacks).

    A stop has adverse same-bar precedence.  Trailing exits at peak minus the
    declared giveback after activation; unresolved paths settle at H12 close.
    Incomplete paths remain NaN rather than being converted to a loss.
    """
    n=len(starts); ns=len(stop_atr); na=len(activation_atr); ng=len(giveback_atr)
    out=np.full((n,ns,na,ng),np.nan,np.float32)
    for row in range(n):
        start=starts[row]; e=entry[row]; a=atr[row]
        if start<0 or start+horizon_bars>len(close) or not np.isfinite(e) or not np.isfinite(a) or e<=0. or a<=0.: continue
        complete=True
        for offset in range(horizon_bars):
            pos=start+offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]) or not np.isfinite(close[pos]): complete=False; break
        if not complete: continue
        for si in range(ns):
            for ai in range(na):
                for gi in range(ng):
                    peak=0.; active=False; result=np.nan
                    for offset in range(horizon_bars):
                        pos=start+offset
                        if side[row]>0.:
                            favorable=(high[pos]-e)/a; adverse=(e-low[pos])/a; trailing_value=(low[pos]-e)/a
                        else:
                            favorable=(e-low[pos])/a; adverse=(high[pos]-e)/a; trailing_value=(e-high[pos])/a
                        # Conservative conflict convention: a bar that can
                        # stop also has stop priority over a trailing fill.
                        if adverse>=stop_atr[si]: result=-stop_atr[si]; break
                        if favorable>peak: peak=favorable
                        was_active=active
                        if peak>=activation_atr[ai]: active=True
                        # A newly activated trailing stop cannot claim a
                        # same-bar retracement whose intrabar order is unknown.
                        if was_active and trailing_value<=peak-giveback_atr[gi]: result=peak-giveback_atr[gi]; break
                    if np.isnan(result): result=side[row]*(close[start+horizon_bars-1]-e)/a
                    out[row,si,ai,gi]=result
    return out


def net_bps(gross_atr: np.ndarray, atr_bps: np.ndarray, *, cost_bps: float = 100.) -> np.ndarray:
    """Convert ATR-grid outcomes to net bps, applying cost exactly once."""
    scale=np.asarray(atr_bps,dtype=np.float32).reshape((-1,1,1,1))
    return np.asarray(gross_atr,dtype=np.float32)*scale-float(cost_bps)
