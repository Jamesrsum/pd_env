"""LoRA identity/base freezing, exact LP transfer and paper gradient semantics."""
import copy
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.parameter_efficient import LoRALinear,inject_lora,StandardizedLinear
from src.training.literature_runner import objective,run
from src.training.controlled_adaptation import seed_all
from src.training.research_objectives import diagnosis_supcon


def test_lora_zero_initialization_updates_adapters_only_and_roundtrip():
    torch.manual_seed(4);base=nn.Linear(8,8);original=copy.deepcopy(base)
    layer=LoRALinear(base,rank=2,alpha=4)
    x=torch.randn(6,8);torch.testing.assert_close(layer(x),original(x),rtol=0,atol=0)
    before={n:p.clone() for n,p in layer.named_parameters()}
    opt=torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.01)
    for _ in range(2):
        opt.zero_grad();layer(x).square().mean().backward();opt.step()
    assert torch.equal(layer.base.weight,before['base.weight'])
    assert torch.equal(layer.base.bias,before['base.bias'])
    assert not torch.equal(layer.lora_A,before['lora_A'])
    assert not torch.equal(layer.lora_B,before['lora_B'])
    restored=LoRALinear(nn.Linear(8,8),2,4);restored.load_state_dict(layer.state_dict())
    torch.testing.assert_close(layer(x),restored(x),rtol=0,atol=0)


def test_lora_injection_rejects_missing_target_before_mutation():
    backbone=nn.Sequential(nn.Linear(4,4),nn.Linear(4,4))
    with pytest.raises(ValueError):inject_lora(backbone,['0','missing'],2)
    assert not any(isinstance(m,LoRALinear) for m in backbone.modules())
    paths=inject_lora(backbone,['0','1'],2)
    assert paths==['0','1']
    assert sum(p.numel() for p in backbone.parameters() if p.requires_grad)==32


def test_train_fitted_logistic_probe_is_preserved_without_new_normalization():
    rng=np.random.default_rng(2);x=rng.normal(size=(100,8))*np.arange(1,9)+np.arange(8)*10;y=np.arange(100)%2
    scaler=StandardScaler().fit(x);head=LogisticRegression(C=.1).fit(scaler.transform(x),y)
    model=StandardizedLinear(scaler.mean_,scaler.scale_,head.coef_,head.intercept_)
    with torch.no_grad():actual=torch.sigmoid(model(torch.tensor(x,dtype=torch.float32))).numpy().ravel()
    np.testing.assert_allclose(actual,head.predict_proba(scaler.transform(x))[:,1],rtol=1e-6,atol=1e-6)
    assert dict(model.named_buffers()).keys()=={'mean','scale'}


class Tiny(nn.Module):
    def __init__(self,family='hubert_supcon',policy='full'):
        super().__init__();self.family=family;self.policy=policy
        self.backbone=nn.Sequential(nn.Linear(1,8),nn.Dropout(.2),nn.Tanh(),nn.Linear(8,8))
        self.head=nn.Linear(8,2 if family=='hubert_supcon' else 1);self.projector=nn.Linear(8,4)
        self.backbone.requires_grad_(policy!='frozen')
    def representations(self,waves,device):
        return self.backbone(torch.tensor([[float(w.mean())] for w in waves],device=device))
    def classify(self,z,detached=False):return self.head(z.detach() if detached else z)


def test_paper_detached_classifier_does_not_update_encoder_through_CE():
    model=Tiny();z=torch.randn(8,8,requires_grad=True);y=torch.tensor([0,0,0,0,1,1,1,1]);ids=np.arange(8)
    ce=F.cross_entropy(model.classify(z,detached=True),y)
    assert torch.autograd.grad(ce,z,allow_unused=True,retain_graph=True)[0] is None
    cfg={'family':'hubert_supcon','objective':'detached_supcon','temperature':.07}
    loss,_=objective(model,z,y,ids,ids,cfg,.5)
    expected=diagnosis_supcon(model.projector(z),y,ids,ids,.07)
    actual_grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected,z,retain_graph=True)[0]
    torch.testing.assert_close(actual_grad,expected_grad)
    assert torch.autograd.grad(loss,model.head.weight,retain_graph=True)[0].abs().sum()>0
    assert torch.autograd.grad(loss,model.projector.weight)[0].abs().sum()>0


@pytest.fixture
def fixture(tmp_path):
    rows=[]
    for split in ['train','val']:
        for person in range(8):
            for visit in range(2):
                path=tmp_path/f'{split}-{person}-{visit}.npy';np.save(path,np.full(32000,(person%2)*.2-.1+visit*.01,dtype='float32'))
                rows.append({'recording_id':path.stem,'participant_id':f'{split}-{person}','duplicate_component':f'{split}-{person}',
                    'label':person%2,'split':split,'initial_recording':visit==0,'waveform_path':str(path)})
    cfg={'method':'synthetic','family':'hubert_supcon','objective':'detached_supcon','policy':'full','epochs':3,
        'seed':4,'batch_size':8,'microbatch_size':1,'temperature':.07,'head_lr':.001,'backbone_lr':.001,
        'weight_decay':.01,'interpretation':'synthetic execution only'}
    return pd.DataFrame(rows),cfg


def test_actual_contrastive_batch_and_training_audits(fixture,tmp_path):
    f,c=fixture;seed_all(4);d=run(f,Tiny(),c,tmp_path/'full','cpu')
    assert d['trainable_backbone_or_adapter_changed'] and d['pretrained_weights_changed']
    history=json.loads((tmp_path/'full/history.json').read_text())
    assert all(r['actual_batch_max']==8 and r['actual_batch_min']==8 for r in history)
    assert d['validation_participants']==8 and not d['test_labels_read']


def test_exact_resume_restores_optimizer_and_all_rngs(fixture,tmp_path):
    f,c=fixture;seed_all(4);continuous=Tiny();run(f,continuous,c,tmp_path/'continuous','cpu')
    seed_all(4);interrupted=Tiny();run(f,interrupted,c,tmp_path/'resume','cpu',max_epochs=1)
    assert not (tmp_path/'resume/DONE.json').exists()
    seed_all(999);resumed=Tiny();run(f,resumed,c,tmp_path/'resume','cpu')
    for a,b in zip(continuous.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert json.loads((tmp_path/'continuous/history.json').read_text())==json.loads((tmp_path/'resume/history.json').read_text())


def test_initial_lp_is_explicit_candidate_and_frozen_control_stays_frozen(fixture,tmp_path):
    f,c=fixture;c={**c,'family':'hear_lpft','objective':'bce','policy':'frozen'}
    seed_all(4);d=run(f,Tiny('hear_lpft','frozen'),c,tmp_path/'frozen','cpu')
    assert not d['pretrained_weights_changed'] and not d['trainable_backbone_or_adapter_changed']
    assert d['frozen_parameters_unchanged']
    assert (tmp_path/'frozen/best.pt').exists()
    assert d['selected_stage'] in ['lp_initialization','adaptation']


def test_test_rows_and_duplicate_split_leakage_rejected(fixture,tmp_path):
    f,c=fixture;f.loc[0,'split']='test'
    with pytest.raises(ValueError,match='Sealed'):run(f,Tiny(),c,tmp_path/'bad','cpu')
    f.loc[0,'split']='train';f.loc[f.split.eq('val'),'duplicate_component']=f.iloc[0].duplicate_component
    with pytest.raises(ValueError,match='Split leakage'):run(f,Tiny(),c,tmp_path/'bad','cpu')


def test_tabicl_uses_only_local_checkpoint_and_training_context(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.training.low_data import tabular_predict
    captured={}
    class Local:
        def __init__(self,**kwargs):captured['settings']=kwargs
        def fit(self,x,y):captured['train']=x.copy();captured['labels']=y.copy();self.classes_=np.array([0,1])
        def predict_proba(self,x):return np.tile([.3,.7],(len(x),1))
    monkeypatch.setitem(sys.modules,'tabicl',SimpleNamespace(TabICLClassifier=Local))
    path=tmp_path/'weights.ckpt';path.write_bytes(b'synthetic')
    x=np.zeros((6,4));xv=np.full((2,4),100);y=np.array([0,1,0,1,0,1])
    p=tabular_predict(x,y,xv,path,'explicit.ckpt',4,'cpu',32)
    assert captured['settings']['allow_auto_download'] is False
    assert captured['settings']['model_path']==str(path) and captured['settings']['checkpoint_version']=='explicit.ckpt'
    np.testing.assert_array_equal(captured['train'],x);np.testing.assert_array_equal(captured['labels'],y)
    np.testing.assert_allclose(p,.7)


def test_literature_submission_dependencies_cap_gpu_concurrency(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import submit_literature_benchmarks as submit
    monkeypatch.setattr(sys,'argv',['submit_literature_benchmarks.py'])
    for key in ['MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_ADAPT_SNAPSHOT','MPOWER_MODEL_ROOT','MPOWER_HEAR_SOURCE','MPOWER_FROZEN_ROOT']:
        monkeypatch.setenv(key,str(tmp_path))
    commands=[]
    def fake_run(command,**kwargs):
        commands.append(command);return SimpleNamespace(stdout=str(101+len(commands)-1)+'\n')
    monkeypatch.setattr(submit.subprocess,'run',fake_run);submit.main()
    assert len(commands)==6
    assert '--dependency=afterok:101:102:103' in commands[3]
    assert '--dependency=afterok:102:101' in commands[4] and '--array=0-1%1' in commands[4]
    assert '--dependency=afterok:101,afterany:104' in commands[5] and '--array=0-2%1' in commands[5]


def test_hubert_native_mask_parameter_is_loaded_before_disabling_augmentation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import transformers
    from src.models.literature_adapter import LiteratureAdapter
    config=SimpleNamespace(hidden_size=768,mask_time_prob=.065,mask_feature_prob=0.,layerdrop=.1)
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.feature_extractor=nn.Linear(1,1);self.masked_spec_embed=nn.Parameter(torch.zeros(768))
            self.config=copy.deepcopy(config)
    def load(*args,**kwargs):
        assert kwargs['config'].mask_time_prob==.065
        return Backbone(),{'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[]}
    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',load)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:config)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:None)
    (tmp_path/'configs').mkdir();(tmp_path/'configs/literature_models.lock.json').write_text(json.dumps({'models':{'hubert_base_ls960':{'revision':'synthetic'}}}))
    assets=tmp_path/'models/checkpoints/hubert_base_ls960';assets.mkdir(parents=True)
    (assets/'download_provenance.json').write_text(json.dumps({'revision':'synthetic'}))
    model=LiteratureAdapter(tmp_path/'models',tmp_path,{'family':'hubert_supcon','policy':'full','gradient_checkpointing':False})
    assert model.backbone.masked_spec_embed.requires_grad is False and model.backbone.config.mask_time_prob==0
    assert config.mask_time_prob==.065
