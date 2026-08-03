from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1]/"scripts/run_canonical_execution_reliability_exit_hurdle_ablation.py";S=importlib.util.spec_from_file_location("exit_hurdle",P);assert S and S.loader;M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
def _x():
 t=pd.date_range("2025-03-01",periods=40,freq="h",tz="UTC");return pd.DataFrame({"candidate_id":[str(i) for i in range(40)],"side_name":["long"]*40,"__symbol__":["A"]*40,"__ts__":t,M.TIME:t,M.END:t+pd.Timedelta(hours=12),M.NET:np.linspace(-.03,.03,40),"base_oof_score":np.arange(40.),**{z.split(' x ',1)[1]:np.arange(40.) for z in M.contract()["feature_contract"]["bounded_interactions"]}})
def test_interactions_are_train_only_bounded_and_five():
 x=_x();o=M.interactions(x.iloc[:10],x.iloc[10:],M.contract());assert o.shape==(30,5) and (o.abs()<=1).all().all()
def test_h1_combination_keeps_signed_non_success_payoffs():
 h={"opp":np.array([.5]),"success":np.array([.6]),"gain":np.array([.02]),"pay_opp_failure":np.array([.01]),"pay_noopp":np.array([-.01])};assert np.isclose(M.combine("H1",h)[0],.5*(.6*.02+.4*.01)+.5*(-.01))
def test_h4_inverse_log_severity_is_used():
 h={"severe":np.array([.25]),"nonsevere":np.array([.01]),"severity":np.array([np.log(3)])};assert np.isclose(M.combine("H4",h)[0],.75*.01-.25*.02)
def test_other_adverse_is_forbidden_by_contract():assert M.contract()["unsupported_target_rule"]["target_deployed_other_adverse_exit_attribution_only"].startswith("FORBIDDEN")
def test_global_random_tie_is_pooled():
 x=_x().iloc[:4].copy();x["raw_score"]=[1,1,0,0];x[M.NET]=[.02,-.02,0,0];r=M.parent.random_tie_expected(x,"raw_score",.25);assert r["selected_rows"]==1 and np.isclose(r["random_tie_expected_net_bps"],0)
def _checkpoint_frames():
 x=_x().copy();x["feature_x"]=np.arange(len(x),dtype=float);x["target_x"]=(np.arange(len(x))%2).astype(int)
 return x.iloc[:25].copy(),x.iloc[25:].copy()
def test_checkpoint_resume_reuses_exact_unit_without_duplicate_fit(tmp_path,monkeypatch):
 train,valid=_checkpoint_frames();calls=[]
 class Model:classes_=np.array([0,1])
 def fake_fit(*args):
  calls.append(args);return np.linspace(.1,.9,len(args[1])),{"geometry":"tiny","features":["feature_x"]},Model(),["feature_x"]
 monkeypatch.setattr(M,"fit",fake_fit);stats={"fitted":0,"reused":0};kw=dict(root=tmp_path,resume=True,global_fingerprint="frozen",architecture="H1",variant="v",fold="fold1",side="long",head="opp",train=train,valid=valid,features=["feature_x"],target="target_x",task="binary",geoms=(),seed=7,stats=stats)
 first=M.checkpointed_fit(**kw);second=M.checkpointed_fit(**kw)
 assert len(calls)==1 and stats=={"fitted":1,"reused":1}
 assert np.array_equal(first[0],second[0]) and (tmp_path/"H1/v/fold1/long/opp/metadata.json").is_file()
def test_checkpoint_rejects_changed_fingerprint_and_identity(tmp_path,monkeypatch):
 train,valid=_checkpoint_frames()
 class Model:classes_=np.array([0,1])
 monkeypatch.setattr(M,"fit",lambda *a:(np.zeros(len(a[1])),{"geometry":"tiny","features":["feature_x"]},Model(),["feature_x"]))
 stats={"fitted":0,"reused":0};kw=dict(root=tmp_path,resume=True,global_fingerprint="frozen",architecture="H1",variant="v",fold="fold1",side="long",head="opp",train=train,valid=valid,features=["feature_x"],target="target_x",task="binary",geoms=(),seed=7,stats=stats)
 M.checkpointed_fit(**kw)
 import pytest
 with pytest.raises(M.ContractError,match="fingerprint/identity mismatch"):M.checkpointed_fit(**{**kw,"global_fingerprint":"changed"})
 changed=valid.copy();changed.loc[changed.index[0],"candidate_id"]="changed"
 with pytest.raises(M.ContractError,match="fingerprint/identity mismatch"):M.checkpointed_fit(**{**kw,"valid":changed})
def test_outer_fold_purges_unavailable_labels_chronologically():
 x=_x();fold={"validation_start_utc":"2025-03-02T00:00:00Z","validation_end_utc":"2025-03-02T08:00:00Z"}
 train,valid=M.outer(x,fold)
 assert not ((x.loc[train,M.END]>=pd.Timestamp(fold["validation_start_utc"])).any())
 assert not ((x.loc[valid,M.TIME]<pd.Timestamp(fold["validation_start_utc"])).any())
def test_run_lock_rejects_concurrent_duplicate(tmp_path):
 import pytest
 lock=tmp_path/"run.lock"
 with M.RunLock(lock):
  with pytest.raises(M.ContractError,match="duplicate fit"):M.RunLock(lock).__enter__()
