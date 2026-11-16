"""Matched schedules, phase identity, parameter checks and mid-run resume."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
from torch import nn
from src.training.controlled_adaptation import run_stage,seed_all,learning_rates


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.kind='synthetic';self.backbone=nn.Sequential(nn.Linear(1,4),nn.Tanh(),nn.Linear(4,4));self.head=nn.Linear(4,1)
    def configure(self,policy):
        self.backbone.requires_grad_(policy!='frozen')
        return {'trainable_parameters':sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}
    def train(self,mode=True):
        super().train(mode)
        if all(not p.requires_grad for p in self.backbone.parameters()):self.backbone.eval()
        return self
    def forward_waves(self,waves,device):
        x=torch.tensor([[float(w.mean())] for w in waves],device=device)
        return self.head(self.backbone(x)).reshape(-1)


@pytest.fixture
def experiment(tmp_path):
    rows=[]
    for split in ['train','val']:
        for label in [0,1]:
            for person in range(3):
                for visit in range(2):
                    path=tmp_path/f'{split}-{label}-{person}-{visit}.npy';np.save(path,np.full(32000,.2*label-.1+.01*visit,dtype='float32'))
                    rows.append({'split':split,'label':label,'participant_id':f'synthetic-{split}-{label}-{person}',
                                 'recording_id':path.stem,'waveform_path':str(path)})
    cfg={'seed':4,'warmup_epochs':2,'adaptation_epochs':3,'optimizer':'sgd','weight_decay':.01,
         'warmup_head_lr':.03,'adapt_head_lr':.03,'head_min_lr':.003,'backbone_lr':.01,'backbone_min_lr':.001,
         'effective_batch_size':4,'microbatch_size':2,'max_grad_norm':1.}
    return pd.DataFrame(rows),cfg


def test_controlled_warmup_and_branch_state(experiment,tmp_path):
    f,c=experiment;seed_all(c['seed']);warm=Tiny()
    run_stage(f,warm,'frozen',c,tmp_path/'warm','cpu','warmup')
    outcomes=[]
    for policy in ['frozen','last2','full']:
        seed_all(c['seed']);model=Tiny()
        result=run_stage(f,model,policy,c,tmp_path/policy,'cpu','branch',tmp_path/'warm/best.pt')
        assert result['selected_policy']==policy and result['stage']=='branch'
        assert result['backbone_changed']==(policy!='frozen')
        assert result['head_changed'] and result['frozen_parameters_unchanged']
        outcomes.append(result)
    assert len({o['initial_head_sha256'] for o in outcomes})==1
    assert len({o['initial_backbone_sha256'] for o in outcomes})==1
    for epoch in range(c['adaptation_epochs']):
        assert learning_rates(c,epoch,c['adaptation_epochs'],'branch')[0]>=c['head_min_lr']


def test_interrupted_mid_adaptation_resume_matches(experiment,tmp_path):
    f,c=experiment;seed_all(c['seed']);warm=Tiny();run_stage(f,warm,'frozen',c,tmp_path/'warm','cpu','warmup')
    seed_all(4);continuous=Tiny();run_stage(f,continuous,'full',c,tmp_path/'continuous','cpu','branch',tmp_path/'warm/best.pt')
    seed_all(4);interrupted=Tiny();run_stage(f,interrupted,'full',c,tmp_path/'resumed','cpu','branch',tmp_path/'warm/best.pt',max_epochs=1)
    assert not (tmp_path/'resumed/DONE.json').exists()
    seed_all(999);resumed=Tiny();run_stage(f,resumed,'full',c,tmp_path/'resumed','cpu','branch',tmp_path/'warm/best.pt')
    for a,b in zip(continuous.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert json.loads((tmp_path/'continuous/history.json').read_text())==json.loads((tmp_path/'resumed/history.json').read_text())


def test_controlled_rejects_final_test(experiment,tmp_path):
    f,c=experiment;f.loc[0,'split']='test'
    with pytest.raises(ValueError,match='sealed'):run_stage(f,Tiny(),'frozen',c,tmp_path/'bad','cpu','warmup')


def test_controlled_uses_canonical_initial_visits_independent_of_order(experiment,tmp_path):
    f,c=experiment;f['initial_recording']=f.recording_id.str.endswith('-0')
    expected=set(f.loc[f.split.eq('val') & f.initial_recording,'recording_id'])
    f=f.iloc[::-1]
    run_stage(f,Tiny(),'frozen',c,tmp_path/'initial','cpu','warmup',max_epochs=1)
    actual=pd.read_csv(tmp_path/'initial/validation_predictions.csv')
    assert set(actual.recording_id)==expected
    f['initial_recording']=True
    with pytest.raises(ValueError,match='canonical'):run_stage(f,Tiny(),'frozen',c,tmp_path/'duplicated','cpu','warmup')


def test_ast_native_batch_matches_singleton_gradients(tmp_path,monkeypatch):
    import transformers
    from src.training.controlled_adaptation import ControlledAdapter
    torch.set_num_threads(1)
    c=transformers.ASTConfig(hidden_size=24,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,
                             hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:c)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:transformers.ASTFeatureExtractor())
    monkeypatch.setattr(transformers.ASTModel,'from_pretrained',lambda *a,**k:transformers.ASTModel(c))
    path=tmp_path/'checkpoints/a';path.mkdir(parents=True);(path/'download_provenance.json').write_text(json.dumps({'revision':'p','repo_id':'synthetic'}))
    cfg={'models':{'a':{'checkpoint':'synthetic','revision':'p','local_alias':'a'}}}
    model=ControlledAdapter('a',tmp_path,cfg);model.configure('full');model.train()
    assert isinstance(model.head[0],nn.LayerNorm)
    waves=[np.random.default_rng(i).normal(0,.1,32000).astype('float32') for i in range(2)]
    batched=model.forward_waves(waves,'cpu');batched.sum().backward()
    before={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);singleton=torch.cat([model.forward_waves([w],'cpu') for w in waves]);singleton.sum().backward()
    torch.testing.assert_close(batched,singleton,rtol=1e-4,atol=1e-5)
    for k,p in model.named_parameters():
        if k in before:torch.testing.assert_close(before[k],p.grad,rtol=2e-3,atol=2e-4)
