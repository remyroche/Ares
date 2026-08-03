import numpy as np
import pandas as pd

from scripts.run_stage_d_action_mechanism_ablation import (
    ACTION_THRESHOLD_BPS, action_from_prediction, bootstrap, calibration, day_block_estimate,
    hierarchical_preprocess, train_mask,
)


def test_folds_use_resolved_action_labels_only():
    start=pd.Timestamp('2024-04-01T00:00Z')
    frame=pd.DataFrame({'action_decision_ts':[start-pd.Timedelta(hours=13),start-pd.Timedelta(hours=11),start-pd.Timedelta(hours=13)],'label_available_ts':[start-pd.Timedelta(minutes=1),start-pd.Timedelta(minutes=1),start]})
    assert train_mask(frame,start).tolist()==[True,False,False]


def test_scalers_fit_on_training_data_only():
    train=pd.DataFrame({'a':np.arange(100,dtype=float),'b':np.arange(100,dtype=float)**2})
    test=pd.DataFrame({'a':[1e9,2e9],'b':[1e12,2e12]});groups={'A0_minimal_action_state_control':['a','b']};y=np.sin(np.arange(100))
    _,_,state1=hierarchical_preprocess(train,test,['A0_minimal_action_state_control'],groups,y,7)
    _,_,state2=hierarchical_preprocess(train,test*100,['A0_minimal_action_state_control'],groups,y,7)
    assert state1==state2


def test_feature_selection_uses_training_data_only():
    train=pd.DataFrame({'a':np.arange(100,dtype=float),'b':np.arange(100,dtype=float)[::-1]})
    groups={'A0_minimal_action_state_control':['a','b']};y=np.arange(100,dtype=float)
    _,_,s1=hierarchical_preprocess(train,pd.DataFrame({'a':[0.],'b':[0.]}),list(groups),groups,y,9)
    _,_,s2=hierarchical_preprocess(train,pd.DataFrame({'a':[99999.],'b':[-99999.]}),list(groups),groups,y,9)
    assert s1['selected']==s2['selected'] and s1['groups']==s2['groups']


def test_incremental_group_selector_seed_is_arm_position_invariant():
    train=pd.DataFrame({'a':np.arange(120,dtype=float),'b':np.sin(np.arange(120)),'c':np.cos(np.arange(120))});test=train.head(3);y=np.arange(120,dtype=float)
    groups={'A0_minimal_action_state_control':['a'],'A2_candle_rejection_structure':['b','c']}
    _,_,alone=hierarchical_preprocess(train,test,['A2_candle_rejection_structure'],groups,y,11)
    _,_,cumulative=hierarchical_preprocess(train,test,['A0_minimal_action_state_control','A2_candle_rejection_structure'],groups,y,11)
    assert alone['groups']['A2_candle_rejection_structure']==cumulative['groups']['A2_candle_rejection_structure']


def test_side_outputs_are_mapped_to_incremental_bps():
    raw=np.array([-25.,0.,40.]);mapped,_,state=calibration(pd.DataFrame(),raw,100.)
    np.testing.assert_array_equal(mapped,raw);assert state['source']=='identity_bps_insufficient_prior_oof'


def test_action_threshold_uses_development_data_only():
    assert ACTION_THRESHOLD_BPS==0.0
    assert action_from_prediction(np.array([-1.,0.,1.])).tolist()==['EXIT_NOW','EXIT_NOW','CONTINUE_FROZEN_POLICY']


def test_day_bootstrap_recomputes_per_trade_estimand():
    day=pd.DataFrame({'c_sum':[100.,0.],'e_sum':[50.,0.],'rows':[100,1]})
    value=day_block_estimate(day,np.array([0,1]))
    assert value[0]==100/101 and value[0] != np.mean([1.,0.])


def test_paired_bootstrap_reuses_draws_for_blocked_identical_arm():
    rows=[]
    for arm in ['D2','D3']:
        for i,(day,n,c,e) in enumerate([('2024-04-01',100,1.,.5),('2024-04-02',1,-2.,-1.)]):
            for j in range(n):
                rows.append({'arm':arm,'split':'development_oof','side':'long','candidate_id':f'{i}-{j}',
                             'action_decision_ts':pd.Timestamp(day,tz='UTC'),
                             'incremental_vs_always_continue_bps':c,
                             'incremental_vs_always_exit_bps':e})
    out=bootstrap(pd.DataFrame(rows)).sort_values(['arm','side','baseline']).reset_index(drop=True)
    left=out[out.arm.eq('D2')].drop(columns='arm').reset_index(drop=True)
    right=out[out.arm.eq('D3')].drop(columns='arm').reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right)
