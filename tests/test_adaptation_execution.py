from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.training.adaptation_runner import training_manifest,epoch_indices,run_supervised


def test_snapshot_rejects_test_rows(tmp_path):
    d=pd.DataFrame({'participant_id':['synthetic-a'],'recording_id':['synthetic-r'],'recorded_at':['2020-01-01'],
                    'split':['test'],'diagnosis':['pd'],'duplicate_component':['synthetic-a']})
    path=tmp_path/'development_manifest.parquet';d.to_parquet(path,index=False)
    (tmp_path/'READY.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError,match='sealed-test'):training_manifest(tmp_path)


def test_participant_sampling_and_real_updates(tmp_path):
    torch.manual_seed(42);rows=[]
    for split in ['train','val']:
        for label in [0,1]:
            for visit in range(2):
                path=tmp_path/f'{split}-{label}-{visit}.npy';np.save(path,np.full(32000,label*.2-.1,dtype='float32'))
                rows.append({'split':split,'label':label,'participant_id':f'synthetic-{split}-{label}',
                             'recording_id':f'synthetic-{split}-{label}-{visit}','waveform_path':str(path)})
    frame=pd.DataFrame(rows);ix=epoch_indices(frame[frame.split.eq('train')],12)
    assert len(ix)==2 and frame.loc[ix].participant_id.nunique()==2
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.kind='synthetic';self.backbone=nn.Linear(1,3);self.head=nn.Linear(3,1)
        def configure(self,policy):
            self.backbone.requires_grad_(policy!='frozen');return {}
        def frontend(self,wave):return {'x':torch.tensor([[float(wave.mean())]])}
        def forward(self,inputs):return self.head(self.backbone(inputs['x'])).reshape(-1)
    model=Tiny();before=model.backbone.weight.detach().clone()
    cfg={'seed':42,'head_epochs':1,'adaptation_epochs':1,'effective_batch_size':4,'head_lr':.01,'partial_backbone_lr':.01,'max_grad_norm':1}
    run_supervised(frame,model,'last1',cfg,tmp_path/'run','cpu',smoke=True)
    assert not torch.equal(before,model.backbone.weight)
    done=json.loads((tmp_path/'run/DONE.json').read_text());assert not done['test_labels_read']
    assert len(pd.read_csv(tmp_path/'run/validation_predictions.csv'))==2
    final=model.backbone.weight.detach().clone()
    run_supervised(frame,model,'last1',cfg,tmp_path/'run','cpu',smoke=True)
    assert torch.equal(final,model.backbone.weight)
