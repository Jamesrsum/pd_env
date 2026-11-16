"""Differentiable binary adaptation using immutable development-only waveforms."""
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from src.models.adaptation import set_trainable_backbone,freeze_batchnorm_statistics
from src.data.frozen_protocol import development_frame


def training_manifest(snapshot,smoke=False,per_cell=4):
    snapshot=Path(snapshot);path=snapshot/'development_manifest.parquet'
    ready=json.loads((snapshot/'READY.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ready['manifest_sha256']:raise ValueError('Snapshot changed')
    frame=pd.read_parquet(path)
    if not frame.split.isin(['train','val']).all():raise ValueError('Refusing a manifest containing sealed-test rows')
    frame=development_frame(frame).sort_values(['participant_id','recorded_at','recording_id']).reset_index(drop=True)
    if frame.groupby('duplicate_component').split.nunique().gt(1).any():raise ValueError('Duplicate component split leakage')
    if smoke:frame=frame.drop_duplicates('participant_id').groupby(['split','label'],group_keys=False).head(per_cell).reset_index(drop=True)
    return frame,ready['manifest_sha256']


class BinaryAdapter(nn.Module):
    def __init__(self,name,model_root,cfg):
        super().__init__();self.name=name;self.policy='frozen';self.kind='resnet50' if name=='resnet50' else None
        if name=='resnet50':
            from torchvision.models import resnet50
            path=Path(model_root)/'resnet50_imagenet1k_v2.pth'
            self.backbone=resnet50(weights=None);self.backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            dim=self.backbone.fc.in_features;self.backbone.fc=nn.Identity();self.feature=None
            self.asset={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            from transformers import AutoConfig,AutoFeatureExtractor,ASTModel,Wav2Vec2Model,WavLMModel
            entry=cfg['models'][name];path=Path(model_root)/'checkpoints'/entry['local_alias']
            prov=json.loads((path/'download_provenance.json').read_text())
            if prov['revision']!=entry['revision'] or prov['repo_id']!=entry['checkpoint']:raise ValueError('Checkpoint provenance mismatch')
            c=AutoConfig.from_pretrained(path,local_files_only=True);self.kind=c.model_type
            # Explicitly disable latent masking and layerdrop in this primary vowel arm.
            for key in ['mask_time_prob','mask_feature_prob','layerdrop']:
                if hasattr(c,key):setattr(c,key,0.0)
            cls={'audio-spectrogram-transformer':ASTModel,'wav2vec2':Wav2Vec2Model,'wavlm':WavLMModel}[self.kind]
            self.backbone=cls.from_pretrained(path,config=c,local_files_only=True)
            self.feature=AutoFeatureExtractor.from_pretrained(path,local_files_only=True);dim=c.hidden_size;self.asset=entry
        self.head=nn.Linear(dim,1)

    def configure(self,policy):
        self.policy=policy
        count=0 if policy=='frozen' else int(policy[4:]) if policy.startswith('last') else 0
        return set_trainable_backbone(self.backbone,self.kind,last_blocks=count,full=policy=='full')

    def train(self,mode=True):
        super().train(mode)
        # Frozen modules must not update dropout/statistics during head warm-up.
        self.backbone.eval()
        if mode and self.policy=='full':self.backbone.train()
        elif mode and self.policy.startswith('last'):
            layers=self.backbone.layer4 if self.kind=='resnet50' else self.backbone.encoder.layer if self.kind=='audio-spectrogram-transformer' else self.backbone.encoder.layers
            for layer in list(layers)[-int(self.policy[4:]):]:layer.train()
        if self.kind=='resnet50':freeze_batchnorm_statistics(self.backbone)
        return self

    def frontend(self,wave):
        if self.kind=='resnet50':
            import librosa
            mel=librosa.feature.melspectrogram(y=wave,sr=16000,n_fft=400,hop_length=160,win_length=400,window='hamming',n_mels=128,power=2)
            image=torch.from_numpy((np.clip(librosa.power_to_db(mel,ref=np.max),-80,0)+80)/80)[None,None]
            image=torch.nn.functional.interpolate(image,size=(224,224),mode='bilinear',align_corners=False).repeat(1,3,1,1)
            mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
            return {'image':(image-mean)/std}
        native=self.feature([wave],sampling_rate=16000,return_tensors='pt',padding=False,return_attention_mask=True)
        result={'input_values':native.input_values}
        if self.kind!='audio-spectrogram-transformer' and getattr(self.backbone.config,'feat_extract_norm','group')!='group':result['attention_mask']=native.attention_mask
        return result

    def forward(self,inputs):
        if self.kind=='resnet50':pooled=self.backbone(inputs['image'])
        else:
            result=self.backbone(**inputs,return_dict=True)
            pooled=result.pooler_output if self.kind=='audio-spectrogram-transformer' else result.last_hidden_state.mean(1)
        return self.head(pooled).reshape(-1)


def epoch_indices(frame,seed):
    rng=np.random.default_rng(seed)
    groups=[g.index.to_numpy() for _,g in frame.groupby('participant_id',sort=True)]
    indices=[int(rng.choice(g)) for g in groups];rng.shuffle(indices);return indices


def validate_predictions(labels,prob):
    return {'auroc':float(roc_auc_score(labels,prob)),'auprc':float(average_precision_score(labels,prob)),
            'log_loss':float(log_loss(labels,prob,labels=[0,1])),'participants':len(labels)}


def run_supervised(frame,model,policy,cfg,out,device,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng_seed=cfg['seed'];torch.manual_seed(rng_seed);np.random.seed(rng_seed);random.seed(rng_seed)
    model.to(device);train=frame[frame.split.eq('train')];val=frame[frame.split.eq('val')].drop_duplicates('participant_id')
    head_epochs=1 if smoke else cfg['head_epochs'];adapt_epochs=1 if smoke else cfg['adaptation_epochs']
    accumulation=4 if smoke else cfg['effective_batch_size'];total=head_epochs+adapt_epochs
    frozen_reference={};optimizer=None;history=[];best_key=None;start_epoch=0
    resume=out/'last.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);start_epoch=saved['epoch']+1
        history=saved['history'];best_key=tuple(saved['best_key']);torch.set_rng_state(saved['rng'])
    model.configure('frozen')
    for epoch in range(start_epoch,total):
        active='frozen' if epoch<head_epochs else policy
        schedule=model.configure(active)
        model.train()
        if active=='frozen':lr=.001;kind='SGD'
        elif model.kind in ['resnet50','audio-spectrogram-transformer']:
            lr=(.0001 if model.kind=='resnet50' else .00001);kind='SGD'
        else:lr=cfg['partial_backbone_lr'];kind='AdamW'
        if epoch>=head_epochs:lr=1e-6+(lr-1e-6)*.5*(1+np.cos(np.pi*(epoch-head_epochs)/max(1,adapt_epochs-1)))
        groups=[{'params':list(model.head.parameters()),'lr':lr if kind=='SGD' else cfg['head_lr']}]
        backbone_parameters=[p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_parameters:groups.append({'params':backbone_parameters,'lr':lr})
        # Persist optimizer state across epochs; reset only on warm-up/adaptation transition.
        if optimizer is None or epoch==head_epochs:
            optimizer=torch.optim.SGD(groups,momentum=.9,nesterov=True,weight_decay=.01) if kind=='SGD' else torch.optim.AdamW(groups,weight_decay=.01)
            if resume.exists() and epoch==start_epoch and saved['phase']==active:optimizer.load_state_dict(saved['optimizer'])
        for group,desired in zip(optimizer.param_groups,groups):group['lr']=desired['lr']
        indices=epoch_indices(train,rng_seed+epoch);optimizer.zero_grad(set_to_none=True);losses=[]
        for step,index in enumerate(indices):
            row=frame.loc[index];wave=np.load(row.waveform_path)[:160000].astype('float32')
            inp={k:v.to(device) for k,v in model.frontend(wave).items()}
            logit=model(inp);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor([float(row.label)],device=device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            group_start=(step//accumulation)*accumulation;group_size=min(accumulation,len(indices)-group_start)
            (loss/group_size).backward();losses.append(float(loss.detach()))
            if (step+1)%accumulation==0 or step+1==len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
        model.eval();prob=[]
        with torch.no_grad():
            for row in val.itertuples():
                inp={k:v.to(device) for k,v in model.frontend(np.load(row.waveform_path)[:160000].astype('float32')).items()}
                prob.append(float(torch.sigmoid(model(inp))[0]))
        metrics=validate_predictions(val.label.to_numpy(),np.array(prob));history.append({'epoch':epoch,'phase':active,'loss':float(np.mean(losses)),**metrics})
        key=(metrics['auroc'],-metrics['log_loss'])
        if best_key is None or key>best_key:
            best_key=key
            torch.save({'model':model.state_dict(),'epoch':epoch,'policy':active},out/'best.pt')
            prediction=val[['recording_id','participant_id','label']].copy();prediction['probability']=prob;prediction.to_csv(out/'validation_predictions.csv',index=False)
            (out/'validation_metrics.json').write_text(json.dumps(metrics,indent=2))
        torch.save({'epoch':epoch,'phase':active,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'history':history,'best_key':best_key},out/'last.tmp.pt')
        (out/'last.tmp.pt').replace(resume)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(history[-1]),flush=True)
    (out/'DONE.json').write_text(json.dumps({'status':'smoke_complete' if smoke else 'development_complete','epochs':total,'test_labels_read':False,'requested_policy':policy,'participants_train':train.participant_id.nunique(),'participants_val':len(val),'best_validation_auroc':best_key[0]},indent=2))
