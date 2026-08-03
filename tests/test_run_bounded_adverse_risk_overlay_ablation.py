from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1]/"scripts/run_bounded_adverse_risk_overlay_ablation.py";S=importlib.util.spec_from_file_location("overlay",P);assert S and S.loader;M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
def test_signed_penalty_direction_and_train_only_bounds(monkeypatch):
 x=pd.DataFrame({"target_deployed_exit_economics_class":["hard_adverse"]*120,"execution_net_ev_12h":np.linspace(-.2,-.01,120),"target_conditional_severe_loss_mask":[True]*120,"target_conditional_severe_loss_log1p_100bps":np.linspace(.1,3,120)})
 monkeypatch.setattr(M,"contract",lambda:{"clip_quantiles":[.01,.99]})
 b=M.bounds(x);assert b["h2_upper"]<=0 and b["h4_upper"]<=0
 assert np.clip(.2*-.1,b["h2_lower"],b["h2_upper"])<=0 and np.clip(-.3*.01*np.expm1(2),b["h4_lower"],b["h4_upper"])<=0
def test_global_selection_is_pooled_not_side_quota():
 x=pd.DataFrame({"candidate_id":["a","b","c","d"],"side_name":["long","long","short","short"],"execution_net_ev_12h":[.02,-.02,0.,0.],"raw_score":[4.,3.,2.,1.]})
 r=M.H.parent.random_tie_expected(x,"raw_score",.25);assert r["selected_rows"]==1 and r["random_tie_expected_net_bps"]==200.
def test_checkpoint_payload_rejects_identity_and_hash_drift(tmp_path,monkeypatch):
 t=pd.to_datetime(["2025-03-01"],utc=True);x=pd.DataFrame({"candidate_id":["a"],"side_name":["long"],"__symbol__":["X"],"__ts__":t,M.TIME:t,M.END:t+pd.Timedelta(hours=12)})
 root=tmp_path;monkeypatch.setattr(M,"CKPT",root);(root/"checkpoint_contract.json").write_text(json.dumps({"fingerprint":"f"}))
 d=root/"H2/primary25/fold/long/class";d.mkdir(parents=True);pred=d/"predictions.npz"
 with pred.open("wb") as f:np.savez_compressed(f,prediction=np.array([[.2,.3,.4,.1]]))
 digest=hashlib.sha256(pred.read_bytes()).hexdigest();expected={"architecture":"H2","variant":"primary25","fold":"fold","side":"long","head":"class","global_fingerprint":"f","train_identity_sha256":M.H.identity_hash(x),"valid_identity_sha256":M.H.identity_hash(x),"train_rows":1,"valid_rows":1}
 (d/"metadata.json").write_text(json.dumps({"expected":expected,"predictions_sha256":digest}))
 assert M.payload("H2","fold","long","class",x,x)[0].shape==(1,4)
 import pytest
 with pytest.raises(M.ContractError,match="identity mismatch"):M.payload("H2","fold","long","class",x,x.assign(candidate_id="changed"))
 meta=json.loads((d/"metadata.json").read_text());meta["predictions_sha256"]="wrong";(d/"metadata.json").write_text(json.dumps(meta))
 with pytest.raises(M.ContractError,match="payload hash mismatch"):M.payload("H2","fold","long","class",x,x)
