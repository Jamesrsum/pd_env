"""Audited LP-FT/LoRA and detached-classifier SupCon adaptations.

These are method transfers to mPower; no original-dataset SOTA replication is
claimed. Initial LP is a selectable epoch-zero baseline, never hidden as tuning.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.training.adaptation_runner import epoch_indices,validate_predictions
from src.training.controlled_adaptation import seed_all,rng_state,restore_rng,fingerprint,atomic_torch
from src.training.research_objectives import balanced_people_batches,diagnosis_supcon


def pretrained_state(backbone):
    return {name.replace('.base.','.'):value for name,value in backbone.state_dict().items()
            if not name.endswith(('.lora_A','.lora_B'))}


def objective(model,z,labels,people,components,cfg,prior):
    if cfg['family']=='hear_lpft':
        logits=model.classify(z).reshape(-1)
        return F.binary_cross_entropy_with_logits(logits,labels.float()),{'classification':True,'contrastive':False}
    detached=cfg['objective']=='detached_supcon'
    logits=model.classify(z,detached=detached)
    ce=F.cross_entropy(logits,labels,reduction='none')
    # Same class-balanced participant schedule in BOTH speech arms. Recover the
    # original training-person prevalence for the diagnosis classifier.
    ce=(ce*torch.where(labels.bool(),2*prior,2*(1-prior))).mean()
    if detached:
        contrast=diagnosis_supcon(model.projector(z),labels,people,components,cfg['temperature'])
        return ce+contrast,{'classification':True,'contrastive':True}
    return ce,{'classification':True,'contrastive':False}


def predict(model,frame,device):
    model.eval();values=[]
    with torch.no_grad():
        for row in frame.itertuples():
            z=model.representations([np.load(row.waveform_path)[:160000]],device)
            logits=model.classify(z)
            p=torch.sigmoid(logits.reshape(-1))[0] if model.family=='hear_lpft' else logits.softmax(-1)[0,1]
            values.append(float(p))
    return np.asarray(values)


def run(frame,model,cfg,out,device,max_epochs=None):
    if not frame.split.isin(['train','val']).all():raise ValueError('Sealed test forbidden')
    for identity in ['participant_id','duplicate_component']:
        if frame.groupby(identity).split.nunique().gt(1).any():raise ValueError('Split leakage')
    train=frame.loc[frame.split.eq('train')]
    val=frame.loc[frame.split.eq('val') & frame.initial_recording]
    if val.empty or val.participant_id.duplicated().any():raise ValueError('Canonical initial validation required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model.to(device)
    seed_all(cfg['seed']);prior=float(train.drop_duplicates('participant_id').label.mean())
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    initial_base=fingerprint(pretrained_state(model.backbone));initial_head=fingerprint(model.head.state_dict())
    initial_projector=fingerprint(model.projector.state_dict()) if hasattr(model,'projector') else None
    frozen={n:p.detach().cpu().clone() for n,p in model.backbone.named_parameters() if not p.requires_grad}
    trainables={n:p.detach().cpu().clone() for n,p in model.backbone.named_parameters() if p.requires_grad}
    groups=[{'name':'head','params':list(model.head.parameters()),'lr':cfg['head_lr']}]
    parameters=[p for p in model.backbone.parameters() if p.requires_grad]
    if parameters:groups.append({'name':'backbone','params':parameters,'lr':cfg['backbone_lr']})
    if hasattr(model,'projector') and cfg['objective']=='detached_supcon':
        groups.append({'name':'projector','params':list(model.projector.parameters()),'lr':cfg['backbone_lr']})
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=cfg['weight_decay'])
    def save_best(epoch,p,metric,initial=False):
        stage=('lp_initialization' if cfg['family']=='hear_lpft' else 'pretrained_random_head') if initial else 'adaptation'
        atomic_torch(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'metrics':metric,'config':cfg,
            'selected_stage':stage,'signature':signature})
        d=val[['recording_id','participant_id','label']].copy();d['probability']=p;d.to_csv(out/'validation_predictions.csv',index=False)
        (out/'validation_metrics.json').write_text(json.dumps({'epoch':epoch,'selected_stage':stage,**metric},indent=2))
    start=0;history=[]
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['signature']!=signature:raise ValueError('Resume config mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        history=state['history'];start=state['epoch'];best_key=tuple(state['best_key']);initial_metrics=state['initial_metrics']
        initial_base=state['initial_base'];initial_head=state['initial_head']
        initial_projector=state['initial_projector']
    else:
        p=predict(model,val,device);initial_metrics=validate_predictions(val.label.to_numpy(),p)
        best_key=(initial_metrics['auroc'],-initial_metrics['log_loss'])
        save_best(0,p,initial_metrics,initial=True)
    for epoch in range(start,min(cfg['epochs'],max_epochs or cfg['epochs'])):
        model.train()
        if cfg['family']=='hubert_supcon':batches=list(balanced_people_batches(train,cfg['seed']+epoch,cfg['batch_size']))
        else:
            indices=epoch_indices(train,cfg['seed']+epoch)
            batches=[indices[i:i+cfg['batch_size']] for i in range(0,len(indices),cfg['batch_size'])]
        if not batches:raise ValueError('No valid participant batches')
        losses=[];gradient_seen=False;counts=[]
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            # Contrastive arms hold one actual batch's graph (checkpointed), so
            # all 32 people are present in the denominator. BCE permits micros.
            micros=[indices] if cfg['family']=='hubert_supcon' else [indices[i:i+cfg['microbatch_size']] for i in range(0,len(indices),cfg['microbatch_size'])]
            for micro in micros:
                rows=frame.loc[micro];waves=[np.load(p)[:160000] for p in rows.waveform_path]
                z=model.representations(waves,device);labels=torch.as_tensor(rows.label.to_numpy(),device=device,dtype=torch.long)
                loss,_=objective(model,z,labels,rows.participant_id.to_numpy(),rows.duplicate_component.to_numpy(),cfg,prior)
                loss=loss*len(micro)/len(indices)
                if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
                loss.backward();losses.append(float(loss.detach()))
            for p in model.backbone.parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite encoder gradient')
                    gradient_seen=gradient_seen or bool(p.grad.count_nonzero())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();counts.append(len(indices))
        if cfg['policy']!='frozen' and not gradient_seen:raise AssertionError('No backbone/LoRA gradient')
        p=predict(model,val,device);metric=validate_predictions(val.label.to_numpy(),p)
        key=(metric['auroc'],-metric['log_loss'])
        if key>best_key:best_key=key;save_best(epoch+1,p,metric)
        row={'epoch':epoch+1,'mean_batch_loss':sum(losses)/len(batches),'encoder_gradient_seen':gradient_seen,
             'actual_batch_min':min(counts),'actual_batch_max':max(counts),**metric}
        history.append(row)
        atomic_torch(out/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
            'signature':signature,'epoch':epoch+1,'history':history,'best_key':best_key,'initial_metrics':initial_metrics,
            'initial_base':initial_base,'initial_head':initial_head,'initial_projector':initial_projector})
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    if len(history)<cfg['epochs']:return {'status':'interrupted_for_resume_check'}
    for name,p in model.backbone.named_parameters():
        if name in frozen and not torch.equal(frozen[name],p.detach().cpu()):raise AssertionError(f'Frozen base parameter changed: {name}')
    changed=any(not torch.equal(p.detach().cpu(),trainables[name]) for name,p in model.backbone.named_parameters() if name in trainables)
    if changed!=(cfg['policy']!='frozen'):raise AssertionError('Backbone/adapter update audit failed')
    base_changed=fingerprint(pretrained_state(model.backbone))!=initial_base
    if cfg['policy']=='lora' and base_changed:raise AssertionError('LoRA changed original pretrained weights')
    best=json.loads((out/'validation_metrics.json').read_text())
    done={'status':'complete','method':cfg['method'],'family':cfg['family'],'policy':cfg['policy'],'objective':cfg['objective'],
        'selected_epoch':best['epoch'],'selected_stage':best['selected_stage'],'validation_metrics':best,
        'initial_validation_metrics':initial_metrics,'initial_base_sha256':initial_base,'initial_head_sha256':initial_head,
        'pretrained_weights_changed':base_changed,'trainable_backbone_or_adapter_changed':changed,
        'frozen_parameters_unchanged':True,'head_changed':fingerprint(model.head.state_dict())!=initial_head,
        'projector_changed':fingerprint(model.projector.state_dict())!=initial_projector if initial_projector else False,
        'trainable_backbone_parameters':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
        'test_labels_read':False,'training_pd_prior':prior,'train_participants':int(train.participant_id.nunique()),
        'validation_participants':len(val),'interpretation':cfg['interpretation']}
    (out/'DONE.json').write_text(json.dumps(done,indent=2));return done
