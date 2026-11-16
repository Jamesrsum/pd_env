"""Feature allowlist/source integrity, grouped training-only tuning and resume."""
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
import numpy as np
import pandas as pd
import pytest
from src.training.acoustic_xgboost import acoustic_join,grouped_folds,check_device,search


def tables():
    frame=pd.DataFrame({'recording_id':['a','b','c'],'participant_id':['p1','p2','p3'],
        'duplicate_component':['g1','g2','g3'],'split':['train','train','val'],'file_sha256':['sha1','sha2','sha3']})
    features=pd.DataFrame({'recording_id':['c','a','b'],'praat_full__f0_mean_hz':[3.,1.,np.nan],
        'egemaps__hnr':[5.,6.,np.inf],'spectral__mfcc_1_mean':[6.,7.,8.],
        'diagnosis':[1,1,0],'age':[80,30,50],'medication':['x','x','x'],'status':['ok']*3})
    quality=pd.DataFrame({'recording_id':['b','c','a'],'file_sha256':['sha2','sha3','sha1']})
    return frame,features,quality


def test_only_acoustic_prefixes_selected_and_source_order_preserved():
    frame,features,quality=tables();x,names,audit=acoustic_join(frame,features,quality,False)
    assert names==['praat_full__f0_mean_hz','egemaps__hnr','spectral__mfcc_1_mean']
    assert x[0,0]==1 and x[2,0]==3 and np.isnan(x[1,0]) and np.isnan(x[1,1])
    assert audit['source_hashes_matched'] and audit['infinities_mapped_to_missing']==1
    with pytest.raises(ValueError,match='184'):acoustic_join(frame,features,quality)


@pytest.mark.parametrize('failure',['source','missing','duplicate','test'])
def test_source_failures_do_not_silently_replace_initial_visits(failure):
    frame,features,quality=tables()
    if failure=='source':quality.loc[0,'file_sha256']='wrong'
    if failure=='missing':features=features.iloc[:-1]
    if failure=='duplicate':features=pd.concat([features,features.iloc[:1]])
    if failure=='test':frame.loc[0,'split']='test'
    with pytest.raises(ValueError):acoustic_join(frame,features,quality,False)


def training():
    frame=pd.DataFrame({'recording_id':[f'r{i}' for i in range(96)],'participant_id':[f'p{i}' for i in range(96)],
        'duplicate_component':[f'g{i//2}' for i in range(96)],'split':'train','initial_recording':True,
        'label':np.arange(96)//2%2})
    x=np.random.default_rng(4).normal(size=(96,6)).astype('float32');x[:,0]+=frame.label*2;x[::7,1]=np.nan
    return x,frame


def test_participant_duplicate_groups_isolated_and_validation_rejected():
    x,f=training()
    for a,b in grouped_folds(f,3,4):assert not set(f.iloc[a].duplicate_component)&set(f.iloc[b].duplicate_component)
    f.loc[0,'split']='val'
    with pytest.raises(ValueError,match='TRAIN'):grouped_folds(f,3,4)


def test_GPU_fallback_is_rejected():
    class Booster:
        def save_config(self):return json.dumps({'learner':{'generic_param':{'device':'cpu'}}})
    assert check_device(Booster(),'cpu')=='cpu'
    with pytest.raises(RuntimeError,match='GPU hist'):check_device(Booster(),'cuda')


def test_training_only_search_and_exact_sampler_resume(tmp_path,monkeypatch):
    import xgboost as xgb
    x,f=training();calls=[];original=xgb.cv
    def inspected(params,matrix,**kwargs):
        np.testing.assert_array_equal(matrix.get_label(),f.label)
        calls.append(len(matrix.get_label()));return original(params,matrix,**kwargs)
    monkeypatch.setattr(xgb,'cv',inspected)
    cfg={'device':'cpu','threads':1,'seed':4,'cv_folds':3,'trials':2,'max_rounds':12,'early_stopping_rounds':3}
    first=search(x,f,cfg,tmp_path/'resumed');assert first['completed_trials']==2 and 1<=first['boost_rounds']<=12
    assert first['cv_auroc']>.5
    cfg['trials']=3;resumed=search(x,f,cfg,tmp_path/'resumed');continuous=search(x,f,cfg,tmp_path/'continuous')
    assert resumed['completed_trials']==continuous['completed_trials']==3
    a=pd.read_csv(tmp_path/'resumed/trials.csv');b=pd.read_csv(tmp_path/'continuous/trials.csv')
    columns=[c for c in a if c.startswith('params_')]+['value']
    pd.testing.assert_frame_equal(a[columns],b[columns])
    assert all(n==96 for n in calls)
    changed=x.copy();changed[0,0]+=1
    with pytest.raises(ValueError,match='mismatch'):search(changed,f,cfg,tmp_path/'resumed')
