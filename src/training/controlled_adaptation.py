"""Paired adaptation branches from one best warm-up checkpoint.

Explicit head/backbone schedules, phase-labelled metrics and complete RNG resume.
"""
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.training.adaptation_runner import BinaryAdapter,epoch_indices,validate_predictions


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def fingerprint(tensors):
    sha=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        sha.update(name.encode());sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


class ControlledAdapter(BinaryAdapter):
    def __init__(self,name,model_root,base_cfg,head_kind='layernorm_linear'):
        super().__init__(name,model_root,base_cfg)
        if head_kind=='layernorm_linear':
            width=self.head.in_features;self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        elif head_kind!='linear':raise ValueError('Unknown classifier architecture')
        self.head_kind=head_kind

    def forward_waves(self,waves,device):
        # AST has a fixed native grid, allowing true minibatches. Waveform models
        # retain singleton forwards until padding invariance is separately tested.
        if self.kind=='audio-spectrogram-transformer':
            batch=self.feature(waves,sampling_rate=16000,return_tensors='pt')
            return self({'input_values':batch.input_values.to(device)})
        return torch.cat([self({k:v.to(device) for k,v in self.frontend(wave).items()}) for wave in waves])


def learning_rates(cfg,epoch,total,stage):
    if stage=='warmup':return cfg['warmup_head_lr'],0.
    fraction=epoch/max(1,total-1);factor=.5*(1+math.cos(math.pi*fraction))
    return (cfg['head_min_lr']+(cfg['adapt_head_lr']-cfg['head_min_lr'])*factor,
            cfg['backbone_min_lr']+(cfg['backbone_lr']-cfg['backbone_min_lr'])*factor)


def optimizer_for(model,cfg,head_lr,backbone_lr):
    groups=[{'name':'head','params':list(model.head.parameters()),'lr':head_lr}]
    backbone=[p for p in model.backbone.parameters() if p.requires_grad]
    if backbone:groups.append({'name':'backbone','params':backbone,'lr':backbone_lr})
    if cfg['optimizer']=='sgd':return torch.optim.SGD(groups,momentum=.9,nesterov=True,weight_decay=cfg['weight_decay'])
    if cfg['optimizer']=='adam':return torch.optim.Adam(groups,betas=tuple(cfg.get('adam_betas',[.95,.999])),weight_decay=cfg['weight_decay'])
    if cfg['optimizer']=='adamw':return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    raise ValueError('Unknown optimizer')


def evaluate(model,val,device):
    model.eval();predictions=[]
    with torch.no_grad():
        for row in val.itertuples():
            wave=np.load(row.waveform_path)[:160000].astype('float32')
            predictions.append(float(torch.sigmoid(model.forward_waves([wave],device))[0]))
    return validate_predictions(val.label.to_numpy(),np.array(predictions)),predictions


def atomic_torch(path,payload):
    path=Path(path);temp=path.with_suffix('.tmp.pt');torch.save(payload,temp);temp.replace(path)


def run_stage(frame,model,policy,cfg,out,device,stage,warmup=None,max_epochs=None):
    if stage not in ['warmup','branch']:raise ValueError('Unknown stage')
    if not frame.split.isin(['train','val']).all():raise ValueError('Final test is sealed')
    if stage=='warmup' and policy!='frozen':raise ValueError('Warm-up must freeze the backbone')
    if stage=='branch' and warmup is None:raise ValueError('Branch requires shared best warm-up')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model.to(device)
    signature=hashlib.sha256(json.dumps({'config':cfg,'policy':policy,'stage':stage},sort_keys=True).encode()).hexdigest()
    train=frame[frame.split.eq('train')];val=frame[frame.split.eq('val')]
    val=val.loc[val.initial_recording] if 'initial_recording' in val else val.drop_duplicates('participant_id')
    if val.empty or val.participant_id.duplicated().any():raise ValueError('Missing or repeated canonical initial assessment')
    epochs=cfg['warmup_epochs'] if stage=='warmup' else cfg['adaptation_epochs']
    seed_all(cfg['seed']);history=[];best_key=None;start=0
    if stage=='branch':
        initial=torch.load(warmup,map_location='cpu',weights_only=False)
        if initial['stage']!='warmup' or initial['policy']!='frozen':raise ValueError('Not a warm-up checkpoint')
        model.load_state_dict(initial['model'])
        # Identical branch RNG and visit schedules; no optimizer state carries over
        # asymmetrically between a frozen control and an adapted branch.
        seed_all(cfg['seed']+100000)
    model.configure(policy)
    initial_backbone=fingerprint(model.backbone.state_dict());initial_head=fingerprint(model.head.state_dict())
    expected_frozen={name:p.detach().cpu().clone() for name,p in model.backbone.named_parameters() if not p.requires_grad}
    schedule=model.configure(policy);head_lr,backbone_lr=learning_rates(cfg,0,epochs,stage)
    optimizer=optimizer_for(model,cfg,head_lr,backbone_lr)
    resume=out/'last.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Checkpoint/config mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        history=saved['history'];best_key=tuple(saved['best_key']);start=saved['epoch']+1
        initial_backbone=saved['initial_backbone'];initial_head=saved['initial_head']
    initial_metrics=saved['initial_metrics'] if resume.exists() else evaluate(model,val,device)[0]
    for epoch in range(start,epochs if max_epochs is None else min(epochs,max_epochs)):
        model.train();head_lr,backbone_lr=learning_rates(cfg,epoch,epochs,stage)
        for group in optimizer.param_groups:group['lr']=head_lr if group['name']=='head' else backbone_lr
        indices=epoch_indices(train,cfg['seed']+(0 if stage=='warmup' else 100000)+epoch)
        losses=[];gradient_seen=False
        # Effective batches are normalized by actual people in each final batch.
        for offset in range(0,len(indices),cfg['effective_batch_size']):
            effective=indices[offset:offset+cfg['effective_batch_size']];optimizer.zero_grad(set_to_none=True)
            micro=cfg['microbatch_size'] if model.kind=='audio-spectrogram-transformer' else 1
            for begin in range(0,len(effective),micro):
                selected=effective[begin:begin+micro]
                waves=[np.load(frame.loc[i,'waveform_path'])[:160000].astype('float32') for i in selected]
                labels=torch.tensor([float(frame.loc[i,'label']) for i in selected],device=device)
                logits=model.forward_waves(waves,device)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='sum')/len(effective)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();losses.append(float(loss.detach()))
            for parameter in model.backbone.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():raise ValueError('Nonfinite backbone gradient')
                    gradient_seen=gradient_seen or bool(torch.count_nonzero(parameter.grad))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True);optimizer.step()
        if policy!='frozen' and not gradient_seen:raise AssertionError('No nonzero backbone gradients')
        metrics,prob=evaluate(model,val,device)
        key=(metrics['auroc'],-metrics['log_loss'])
        row={'stage':stage,'policy':policy,'epoch':epoch+1,'head_lr':head_lr,'backbone_lr':backbone_lr if policy!='frozen' else 0.,
             'backbone_gradient_seen':gradient_seen,'loss_sum':sum(losses),**metrics};history.append(row)
        if best_key is None or key>best_key:
            best_key=key
            atomic_torch(out/'best.pt',{'model':model.state_dict(),'stage':stage,'policy':policy,'epoch':epoch+1,'metrics':metrics,'signature':signature})
            prediction=val[['recording_id','participant_id','label']].copy();prediction['probability']=prob;prediction.to_csv(out/'validation_predictions.csv',index=False)
            (out/'validation_metrics.json').write_text(json.dumps({'stage':stage,'selected_policy':policy,'epoch':epoch+1,**metrics},indent=2))
        atomic_torch(resume,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'rng':rng_state(),
            'history':history,'best_key':best_key,'signature':signature,'initial_metrics':initial_metrics,'initial_backbone':initial_backbone,'initial_head':initial_head})
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    if len(history)<epochs:return {'status':'interrupted_for_resume_test','epochs':len(history)}
    for name,p in model.backbone.named_parameters():
        if name in expected_frozen and not torch.equal(expected_frozen[name],p.detach().cpu()):raise AssertionError(f'Frozen parameter changed: {name}')
    backbone_changed=fingerprint(model.backbone.state_dict())!=initial_backbone
    head_changed=fingerprint(model.head.state_dict())!=initial_head
    if policy=='frozen' and backbone_changed:raise AssertionError('Frozen backbone changed')
    if policy!='frozen' and not backbone_changed:raise AssertionError('Requested adaptation did not change backbone')
    if not head_changed:raise AssertionError('Classifier did not train')
    best=json.loads((out/'validation_metrics.json').read_text())
    done={'status':'complete','stage':stage,'requested_policy':policy,'selected_policy':best['selected_policy'],
        'selected_epoch':best['epoch'],'best_phase_auroc':best['auroc'],'initial_auroc':initial_metrics['auroc'],
        'initial_backbone_sha256':initial_backbone,'initial_head_sha256':initial_head,
        'backbone_changed':backbone_changed,'head_changed':head_changed,'frozen_parameters_unchanged':True,
        'participants_train':int(train.participant_id.nunique()),'participants_val':len(val),'test_labels_read':False,
        'schedule':schedule}
    (out/'DONE.json').write_text(json.dumps(done,indent=2));return done
