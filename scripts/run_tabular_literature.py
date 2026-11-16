#!/usr/bin/env python3
"""Small-label TabICL v1/v2 comparisons on cached audio; no remote data inference."""
import argparse
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import pandas as pd
import torch
import joblib
from src.training.adaptation_runner import training_manifest
from src.training.downstream import initial_cohort,metrics
from src.training.low_data import train_pca,logistic,tabular_predict
from src.data.benchmark import digest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.umask(0o077);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu' if a.smoke else 'cuda'
    if not a.smoke and not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT']);frame=initial_cohort(frame).reset_index(drop=True)
    if a.smoke:frame=frame.groupby(['split','label'],group_keys=False).head(16).reset_index(drop=True)
    root=Path(os.environ['MPOWER_LITERATURE_ROOT']).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    out=root/('tabular_smoke' if a.smoke else 'tabular_runs');out.mkdir(parents=True,exist_ok=True)
    frozen=Path(os.environ['MPOWER_FROZEN_ROOT']);hear=Path(os.environ['MPOWER_HEAR_SOURCE'])/'runs/frozen'
    hd=json.loads((hear/'DONE.json').read_text())
    if hd['manifest_sha256']!=sha or not hd['encoder_unchanged']:raise ValueError('HeAR cache mismatch')
    hi=pd.read_parquet(hear/'feature_index.parquet').set_index('recording_id',verify_integrity=True)
    sources={};arrays={};vectors=[]
    for row in frame.itertuples():
        path=Path(hi.loc[row.recording_id,'path'])
        if path.parent!=hear/hd['signature']:raise ValueError('HeAR feature signature mismatch')
        with np.load(path,allow_pickle=False) as f:vectors.append(f['mean'].copy())
    arrays['hear']=np.stack(vectors);sources['hear']=hd['signature']
    if not a.smoke:
        for name,tap in [('whisper_small','block_6/mean_sd/window_mean'),('wavlm_large','block_12/mean_sd/window_mean')]:
            source=json.loads((frozen/'runs'/name/'matched_2s/provenance.json').read_text())
            if source['manifest_sha256']!=sha:raise ValueError('Frozen source manifest mismatch')
            cache=frozen/'pooled_features'/name/'matched_2s'/source['signature'];values=[]
            for row in frame.itertuples():
                key=digest({'recording_id':row.recording_id,'source':row.file_sha256,'derived':row.mono16k_sha256,'signature':source['signature']})
                with np.load(cache/f'{key}.npz',allow_pickle=False) as f:values.append(f[tap].copy())
            arrays[name]=np.stack(values);sources[name]=source['signature']
    train=frame.split.eq('train').to_numpy();val=~train;y=frame.loc[train,'label'].to_numpy();yv=frame.loc[val,'label'].to_numpy()
    lock=json.loads((REPO/'configs/tabicl_models.lock.json').read_text())['models'];results=[]
    rng=np.random.default_rng(20261112)
    for name,x in arrays.items():
        if not np.isfinite(x).all():raise ValueError('Nonfinite cached features')
        budgets=[len(y)] if a.smoke or name!='hear' else [50,100,500,len(y)]
        for budget in budgets:
            if budget==len(y):ix=np.arange(len(y))
            else:
                # Predeclared stratified label-budget simulation, no validation labels.
                pd_count=max(3,round(budget*y.mean()))
                ix=np.r_[rng.choice(np.flatnonzero(y==1),pd_count,replace=False),rng.choice(np.flatnonzero(y==0),budget-pd_count,replace=False)]
                rng.shuffle(ix)
            n_components=8 if a.smoke else min(128,len(ix)//3,x.shape[1])
            xt=x[train][ix];yt=y[ix];xv=x[val];groups=frame.loc[train,'duplicate_component'].to_numpy()[ix]
            transform=train_pca(xt,n_components,20261112);z=transform.transform(xt).astype('float32');zv=transform.transform(xv).astype('float32')
            arm=out/name/str(budget);arm.mkdir(parents=True,exist_ok=True);joblib.dump(transform,arm/'pca.joblib')
            base=frame.loc[val,['recording_id','participant_id','label']].copy()
            for classifier in ['lr','tabicl_v1','tabicl_v2']:
                print('Classifier',name,budget,classifier,flush=True)
                if classifier=='lr':
                    probability,state=logistic(xt,yt,groups,xv,n_components,20261112);joblib.dump(state,arm/'lr.joblib')
                else:
                    asset=lock[classifier];file=asset['files'][0]['name']
                    model_path=Path(os.environ['MPOWER_MODEL_ROOT'])/'checkpoints'/classifier/file
                    import hashlib
                    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=asset['files'][0]['sha256']:raise ValueError('TabICL checksum mismatch')
                    probability=tabular_predict(z,yt,zv,model_path,file,20261112,device,32 if classifier=='tabicl_v1' else 8,a.smoke)
                d=base.copy();d['probability']=probability;d.to_csv(arm/f'{classifier}_predictions.csv',index=False)
                results.append({'encoder':name,'labelled_train_people':len(ix),'train_PD':int(yt.sum()),'classifier':classifier,
                    'pca_dim':n_components,**metrics(yv,probability)})
                pd.DataFrame(results).to_csv(out/'validation_results.csv',index=False)
    (out/'DONE.json').write_text(json.dumps({'status':'smoke_complete' if a.smoke else 'development_complete',
        'test_labels_read':False,'manifest_sha256':sha,'features':sources,'comparisons':len(results),
        'tabicl_version':'2.2.0','local_inference_only':True,'checkpoint_versions':lock,
        'label_budget_seeds':[20261112],'PCA_fit':'labelled_training_context_only; LR CV refits within training folds',
        'interpretation':'adjacent_field_classifier_adaptation_not_original_tabular_benchmark_reproduction'},indent=2))


if __name__=='__main__':main()
