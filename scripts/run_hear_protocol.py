#!/usr/bin/env python3
"""Authorized local HeAR: readiness, frozen features, or controlled adaptation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import torch
import yaml
from src.models.hear import HearAdapter,verify_assets
from src.training.adaptation_runner import training_manifest
from src.training.controlled_adaptation import run_stage,seed_all,fingerprint
from src.data.benchmark import digest


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stage',choices=['readiness','extract','warmup','branch'],required=True)
    p.add_argument('--policy',choices=['frozen','last2','full'],default='frozen')
    p.add_argument('--smoke',action='store_true');p.add_argument('--block-path')
    a=p.parse_args();os.umask(0o077);torch.set_num_threads(4)
    lock=REPO/'configs/pretrained_models.lock.json'
    path,entry=verify_assets(os.environ['MPOWER_MODEL_ROOT'],lock)
    cfg=yaml.safe_load((REPO/'configs/hear_protocol.yaml').read_text())
    if a.block_path:cfg['block_path']=a.block_path
    seed_all(cfg['seed']);model=HearAdapter.from_local(os.environ['MPOWER_MODEL_ROOT'],lock,cfg['block_path'])
    if a.stage=='readiness':
        print(json.dumps({'status':'assets_and_architecture_loaded','revision':entry['revision'],
            'block_sequences':[(n,len(m)) for n,m in model.backbone.named_modules() if isinstance(m,torch.nn.ModuleList)],
            'runtime':{'torch':torch.__version__},'inference_tested':False},indent=2));return
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'],a.smoke,4)
    cfg.update(manifest_sha256=sha,asset_revision=entry['revision'])
    root=Path(os.environ['MPOWER_HEAR_ROOT']).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    root=root/('smoke' if a.smoke else 'runs');root.mkdir(parents=True,exist_ok=True)
    if a.smoke:cfg.update(warmup_epochs=2,adaptation_epochs=2,effective_batch_size=8)
    if a.stage!='extract':
        out=root/('warmup' if a.stage=='warmup' else a.policy)
        warm=root/'warmup/best.pt' if a.stage=='branch' else None
        print(json.dumps(run_stage(frame,model,a.policy,cfg,out,'cuda',a.stage,warm),indent=2));return
    model.configure('frozen');model.to('cuda').eval()
    signature=digest({'revision':entry['revision'],'frontend':json.loads((REPO/'src/models/hear_vendor/provenance.json').read_text()),
        'adapter_sha256':hashlib.sha256((REPO/'src/models/hear.py').read_bytes()).hexdigest(),
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','transformers']},
        'context':'matched_2s','pooling':['mean','mean_population_sd'],'manifest':sha})
    out=root/'frozen';cache=out/signature;cache.mkdir(parents=True,exist_ok=True)
    before=fingerprint(model.backbone.state_dict());vectors=[];features={'mean':[],'mean_sd':[]}
    with torch.no_grad():
        for number,row in enumerate(frame.itertuples()):
            key=digest({'recording_id':row.recording_id,'source':row.file_sha256,'derived':row.mono16k_sha256,'signature':signature})
            dest=cache/f'{key}.npz'
            if not dest.exists():
                z=model.record_features(np.load(row.waveform_path),'cuda',mean_sd=True).cpu().numpy()
                temp=dest.with_suffix('.tmp.npz');np.savez(temp,mean=z[:512],mean_sd=z);temp.replace(dest)
            with np.load(dest,allow_pickle=False) as saved:
                if saved['mean'].shape!=(512,) or saved['mean_sd'].shape!=(1024,) or not np.isfinite(saved['mean_sd']).all():raise ValueError('Invalid frozen cache')
                for name in features:features[name].append(saved[name].copy())
            vectors.append({'recording_id':row.recording_id,'path':str(dest)})
            if (number+1)%100==0:print('HeAR extracted',number+1,'/',len(frame),flush=True)
    if before!=fingerprint(model.backbone.state_dict()):raise AssertionError('Frozen HeAR changed')
    import pandas as pd
    pd.DataFrame(vectors).to_parquet(out/'feature_index.parquet',index=False)
    del model
    torch.cuda.empty_cache()
    import joblib
    from src.training.development import fit_development
    head_cfg=yaml.safe_load((REPO/'configs/frozen_model_testing.yaml').read_text())['head']
    if a.smoke:head_cfg.update(epochs=2,cv_folds=2,C_grid=[.1])
    rows=[]
    for name,values in features.items():
        results=fit_development(np.stack(values),frame,head_cfg,cfg['seed'])
        for kind,result in results.items():
            directory=out/'heads'/name/kind;directory.mkdir(parents=True,exist_ok=True)
            joblib.dump(result['state'],directory/'head.joblib')
            result['initial_validation_predictions'].to_csv(directory/'validation_predictions.csv',index=False)
            rows.append({'representation':name,'head':kind,**result['metrics']})
    pd.DataFrame(rows).to_csv(out/'validation_results.csv',index=False)
    (out/'DONE.json').write_text(json.dumps({'status':'complete','signature':signature,'manifest_sha256':sha,
        'recordings':len(frame),'encoder_unchanged':True,'test_labels_read':False,'config':cfg},indent=2))


if __name__=='__main__':main()
