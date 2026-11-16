"""Synthetic invariance, masking, train-only transforms, and operating-point checks."""
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.models.hear import frontend,HearAdapter,verify_assets
from src.training.research_objectives import (diagnosis_supcon,grouped_masked_huber,balanced_people_batches,
    DiagnosisProjector,controlled_objective,acoustic_target_scaling)
from src.training.downstream import screen,initial_cohort,fuse_fixed_lr
from src.training.development import operating_point


class SyntheticHear(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([nn.Linear(1,1) for _ in range(3)]);self.projection=nn.Linear(1,512)
    def forward(self,x):
        z=x.mean((1,2,3))[:,None]
        for block in self.blocks:z=block(z)
        return SimpleNamespace(pooler_output=self.projection(z))


def test_official_frontend_is_pinned_nonmutating_batch_independent():
    torch.set_num_threads(1)
    base=Path(__file__).resolve().parents[1]/'src/models/hear_vendor'
    p=json.loads((base/'provenance.json').read_text())
    for name,info in p['files'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==info['sha256']
    t=torch.arange(32000)/16000
    a=torch.sin(2*torch.pi*180*t)*.2;b=torch.sin(2*torch.pi*230*t)*5+7
    original=a.clone();batched=frontend([a,b])
    assert batched.shape==(2,1,192,128) and torch.isfinite(batched).all()
    torch.testing.assert_close(batched[0:1],frontend([a]),rtol=0,atol=0)
    torch.testing.assert_close(a,original,rtol=0,atol=0)
    for bad in [torch.zeros(31999),torch.zeros(32001),torch.full((32000,),float('nan'))]:
        with pytest.raises(ValueError):frontend([bad])


@pytest.mark.parametrize('policy',['frozen','last2','full'])
def test_record_pooling_gradients_and_frozen_parameters(policy):
    torch.manual_seed(2);model=HearAdapter(SyntheticHear(),'blocks');model.configure(policy);model.train()
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    frozen={n for n,p in model.named_parameters() if not p.requires_grad}
    wave=np.random.default_rng(4).normal(0,.1,64000).astype('float32')
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    logits=model.forward_waves([wave],'cpu');logits.square().sum().backward();opt.step()
    changed={n for n,p in model.named_parameters() if not torch.equal(p,before[n])}
    assert not changed.intersection(frozen)
    assert any(n.startswith('head') for n in changed)
    assert any(n.startswith('backbone') for n in changed)==(policy!='frozen')
    assert model.record_features(wave,'cpu',True).shape==(1024,)
    assert model.record_features(wave[:32000],'cpu',True)[512:].eq(0).all()


def test_hear_missing_assets_fail_before_model_load(tmp_path):
    lock=Path(__file__).resolve().parents[1]/'configs/pretrained_models.lock.json'
    with pytest.raises(FileNotFoundError,match='HeAR asset unavailable'):verify_assets(tmp_path,lock)
    with pytest.raises(ValueError,match='Inspect'):HearAdapter(SyntheticHear()).configure('last2')


def test_supcon_masks_and_valid_gradients():
    z=torch.randn(6,8,requires_grad=True);labels=torch.tensor([0,0,0,1,1,1])
    ids=['a','a','b','c','d','e'];components=['1','1','2','3','4','5']
    loss=diagnosis_supcon(z,labels,ids,components);loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    with pytest.raises(ValueError,match='Every anchor'):diagnosis_supcon(z[:2],labels[:2],ids[:2],components[:2])


def test_missing_auxiliary_targets_cannot_affect_loss_or_gradient():
    pred=torch.randn(3,4,requires_grad=True);target=torch.randn(3,4);mask=torch.tensor([[1,1,0,0],[1,0,0,0],[1,1,0,0]],dtype=torch.bool)
    groups=[[0,1],[2,3]]
    a=grouped_masked_huber(pred,target,mask,groups);ga=torch.autograd.grad(a,pred)[0]
    target[~mask]=float('nan');b=grouped_masked_huber(pred,target,mask,groups);gb=torch.autograd.grad(b,pred)[0]
    torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb)
    assert gb[~mask].eq(0).all()
    empty=grouped_masked_huber(pred,target,torch.zeros_like(mask),groups)
    assert empty.item()==0


def cohort():
    return pd.DataFrame([{'participant_id':f'synthetic-{s}-{i}','duplicate_component':f'{s}-{i}',
        'split':s,'label':i%2,'initial_recording':True} for s,n in [('train',24),('val',12)] for i in range(n)])


def test_initial_visits_not_replaced_after_filtering():
    f=cohort();f.loc[0,'initial_recording']=False
    assert len(initial_cohort(f))==len(f)-1
    f.loc[0,'split']='test'
    with pytest.raises(ValueError,match='Sealed'):initial_cohort(f)


def test_downstream_validation_cannot_change_fitting():
    f=cohort();rng=np.random.default_rng(1);x=rng.normal(size=(len(f),8));x[:,0]+=f.label
    specs=[{'kind':'lr','C':.1,'pca':None},{'kind':'lr','C':.1,'pca':3},{'kind':'svm','C':.1,'pca':None}]
    a=screen(x,f,fold_count=2,specs=specs);x[f.split.eq('val')]+=1000
    b=screen(x,f,fold_count=2,specs=specs)
    assert a['selected']['spec']==b['selected']['spec']
    for ca,cb in zip(a['candidates'],b['candidates']):
        np.testing.assert_array_equal(ca['oof'],cb['oof'])
        np.testing.assert_array_equal(ca['state']['pipeline']['scale'].mean_,cb['state']['pipeline']['scale'].mean_)
        if ca['state']['calibrator'] is not None:
            np.testing.assert_array_equal(ca['state']['calibrator'].coef_,cb['state']['calibrator'].coef_)
    fusion_a=fuse_fixed_lr([a,a]);fusion_b=fuse_fixed_lr([b,b])
    np.testing.assert_array_equal(fusion_a['oof_stack']['state'].coef_,fusion_b['oof_stack']['state'].coef_)


def test_threshold_float32_ties_survive_serialization():
    prob=np.array([.1,.2,.3,.4,.8,.9],dtype='float32');labels=np.array([0,0,0,0,1,1])
    result=operating_point(labels,prob,.5)
    restored=json.loads(json.dumps(result))
    assert restored['specificity']==.75
    assert np.asarray(prob,dtype='float64')[2]<restored['threshold']


def test_contrastive_batch_distinct_people():
    f=cohort();f=f[f.split.eq('train')];f=pd.concat([f,f]).reset_index(drop=True)
    batches=list(balanced_people_batches(f,2,8))
    assert batches
    for ix in batches:
        b=f.loc[ix];assert not b.participant_id.duplicated().any()
        assert b.label.value_counts().nunique()==1


@pytest.mark.parametrize('mode',['bce','supcon','acoustic'])
def test_joint_objective_backpropagates_to_encoder(mode):
    torch.manual_seed(3);encoder=nn.Linear(5,8);classifier=nn.Linear(8,1);projector=DiagnosisProjector(8)
    x=torch.randn(8,5);z=encoder(x);y=torch.tensor([0,0,0,0,1,1,1,1]);ids=list(range(8))
    targets=torch.randn(8,4);targets[0,1]=float('nan')
    loss,parts=controlled_objective(classifier(z).squeeze(-1),y,z,ids,ids,mode,projector,
        nn.Linear(8,4)(z),targets,torch.isfinite(targets),[[0,1],[2,3]],True,.2)
    loss.backward()
    assert torch.isfinite(encoder.weight.grad).all() and encoder.weight.grad.abs().sum()>0
    assert parts['bce']>0


def test_auxiliary_scaling_uses_observed_initial_train_only():
    f=cohort();rng=np.random.default_rng(3);x=rng.normal(size=(len(f),4));x[:,3]=np.nan
    a,mask,state=acoustic_target_scaling(x,f)
    x[f.split.eq('val')]=10000
    _,_,other=acoustic_target_scaling(x,f)
    np.testing.assert_array_equal(state['mean'],other['mean'])
    np.testing.assert_array_equal(state['columns'],[0,1,2])
    assert a.shape==mask.shape==(len(f),3)


def test_hear_submission_graph_is_smoke_gated(tmp_path,monkeypatch):
    from scripts import submit_hear_protocol as submit
    for key,value in {'MPOWER_MODEL_ROOT':str(tmp_path/'models'),'MPOWER_BENCHMARK_ROOT':str(tmp_path),
        'MPOWER_BENCHMARK_PYTHON':sys.executable,'MPOWER_ADAPT_SNAPSHOT':str(tmp_path/'snapshot'),
        'MPOWER_HEAR_BLOCK_PATH':'inspected.blocks'}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(submit.HearAdapter,'from_local',lambda *a:SimpleNamespace(configure=lambda p:{'policy':p}))
    commands=[]
    def fake_run(command,**kwargs):
        commands.append(command);return SimpleNamespace(stdout=str(100+len(commands))+'\n')
    monkeypatch.setattr(submit.subprocess,'run',fake_run)
    submit.main()
    assert len(commands)==6
    assert commands[0][-1]=='smoke'
    assert '--dependency=afterok:101' in commands[1]
    assert '--dependency=afterok:102' in commands[2]
    assert '--dependency=afterok:103' in commands[3]
    assert '--dependency=afterok:103' in commands[4]
    assert '--dependency=afterok:104' in commands[5]
    receipt=next(tmp_path.glob('hear_protocol/*/submission.json'))
    assert json.loads(receipt.read_text())['full_job']=='106'


def test_hear_submission_refuses_missing_assets_without_sbatch(tmp_path,monkeypatch):
    from scripts import submit_hear_protocol as submit
    for key in ['MPOWER_MODEL_ROOT','MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_ADAPT_SNAPSHOT','MPOWER_HEAR_BLOCK_PATH']:
        monkeypatch.setenv(key,str(tmp_path))
    def unexpected(*a,**k):raise AssertionError('Must not submit with missing weights')
    monkeypatch.setattr(submit.subprocess,'run',unexpected)
    with pytest.raises(FileNotFoundError,match='HeAR asset unavailable'):submit.main()
