import numpy as np

from extreme_price_movements.query_path_grid_labels import contract_event, first_touch_grid_h12, first_touch_grid_horizon, path_extrema_h12, triple_barrier_outcome_bps


def test_grid_records_both_sides_and_adverse_tie_break_is_consumer_side():
    high=np.full(720,101.,dtype=float); low=np.full(720,99.,dtype=float); close=np.full(720,100.,dtype=float)
    favorable, adverse, terminal=first_touch_grid_h12(high,low,close,np.array([0]),np.array([100.]),np.array([1.]),np.array([1.]),np.array([1.,2.]))
    assert favorable[0,0]==1 and adverse[0,0]==1
    f,a,t=contract_event(favorable[:,0],adverse[:,0])
    assert not f[0] and a[0] and not t[0]
    assert np.isfinite(terminal[0])


def test_triple_barrier_outcome_uses_terminal_only_for_timeout():
    gross,net,event=triple_barrier_outcome_bps(favorable_minute=np.array([1,-1,-1]),adverse_minute=np.array([-1,1,-1]),terminal_atr=np.array([9.,9.,2.]),atr_bps=np.array([50.,50.,50.]),upper_atr=4.,lower_atr=2.)
    assert gross.tolist()==[200.,-100.,100.]
    assert net.tolist()==[100.,-200.,0.]
    assert event.tolist()==[0,1,2]


def test_path_extrema_are_complete_and_side_normalized():
    high=np.full(720,102.,dtype=float); low=np.full(720,97.,dtype=float)
    starts=np.array([0],dtype=np.int64)
    long_mfe,long_mae=path_extrema_h12(high,low,starts,np.array([100.]),np.array([1.]),np.array([1.]))
    short_mfe,short_mae=path_extrema_h12(high,low,starts,np.array([100.]),np.array([1.]),np.array([-1.]))
    assert long_mfe.tolist()==[2.] and long_mae.tolist()==[3.]
    assert short_mfe.tolist()==[3.] and short_mae.tolist()==[2.]


def test_resolution_agnostic_grid_uses_declared_horizon():
    high=np.array([101.,103.,101.,101.]); low=np.array([99.,99.,99.,99.]); close=np.array([100.,102.,100.,100.])
    fav,adv,terminal=first_touch_grid_horizon(high,low,close,np.array([0]),np.array([100.]),np.array([1.]),np.array([1.]),np.array([2.]),4)
    assert fav.tolist()==[[2]] and adv.tolist()==[[-1]] and terminal.tolist()==[0.]
