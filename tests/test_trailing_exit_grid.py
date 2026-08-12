import numpy as np

from extreme_price_movements.trailing_exit_grid import net_bps, simulate_h12_stop_trailing_grid


def test_stop_has_same_bar_priority_and_cost_once():
    high=np.full(720,100.,dtype=float); low=np.full(720,100.,dtype=float); close=np.full(720,100.,dtype=float)
    high[0]=103.; low[0]=98.  # reaches +3 and -2 in the same bar
    result=simulate_h12_stop_trailing_grid(high,low,close,np.array([0]),np.array([100.]),np.array([1.]),np.array([1.]),np.array([2.]),np.array([1.]),np.array([.5]))
    assert result.shape==(1,1,1,1)
    assert result[0,0,0,0]==-2.
    assert net_bps(result,np.array([50.]))[0,0,0,0]==-200.


def test_trailing_exit_uses_peak_less_giveback():
    high=np.full(720,100.,dtype=float); low=np.full(720,100.,dtype=float); close=np.full(720,100.,dtype=float)
    high[0]=102.; low[0]=100.; high[1]=104.; low[1]=102.; high[2]=103.; low[2]=100.
    result=simulate_h12_stop_trailing_grid(high,low,close,np.array([0]),np.array([100.]),np.array([1.]),np.array([1.]),np.array([3.]),np.array([2.]),np.array([1.]))
    assert result[0,0,0,0]==3.
