#!/usr/bin/env python3
"""Development-only initial-visit heads/fusion using provenance-keyed frozen caches."""
import argparse
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import joblib
import numpy as np
import pandas as pd
from src.training.adaptation_runner import training_manifest
from src.training.downstream import initial_cohort,screen,fuse_fixed_lr
from src.data.benchmark import digest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.umask(0o077)
    root=Path(os.environ['MPOWER_FROZEN_ROOT']);out=Path(os.environ['MPOWER_DOWNSTREAM_ROOT']).resolve()
    if out==REPO or REPO in out.parents:raise ValueError('Private output required')
    out=out/('smoke' if a.smoke else 'full');out.mkdir(parents=True,exist_ok=True)
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'])
    frame=initial_cohort(frame).reset_index(drop=True)
    if a.smoke:frame=frame.groupby(['split','label'],group_keys=False).head(12).reset_index(drop=True)
    matrices={};sources={}
    # Predeclared representations, rather than searching every cached tap again.
    for model,key in [('whisper_small','block_6/mean_sd/window_mean'),('wavlm_large','block_12/mean_sd/window_mean')]:
        context='matched_2s';source=json.loads((root/'runs'/model/context/'provenance.json').read_text())
        if source['manifest_sha256']!=sha or source['context']!=context or source['test_labels_read']:raise ValueError('Feature source mismatch')
        cache=root/'pooled_features'/model/context/source['signature'];values=[]
        for row in frame.itertuples():
            identifier=digest({'recording_id':row.recording_id,'source':row.file_sha256,'derived':row.mono16k_sha256,'signature':source['signature']})
            with np.load(cache/f'{identifier}.npz',allow_pickle=False) as saved:values.append(saved[key].copy())
        matrices[model]=np.stack(values);sources[model]={'signature':source['signature'],'representation':key}
        if not np.isfinite(matrices[model]).all():raise ValueError('Nonfinite encoder features')
    if os.environ.get('MPOWER_HEAR_ROOT'):
        hear=Path(os.environ['MPOWER_HEAR_ROOT'])/'runs/frozen'
        done=json.loads((hear/'DONE.json').read_text())
        if done['manifest_sha256']!=sha or not done['encoder_unchanged'] or done['test_labels_read']:raise ValueError('HeAR provenance mismatch')
        index=pd.read_parquet(hear/'feature_index.parquet').set_index('recording_id',verify_integrity=True)
        values=[]
        for row in frame.itertuples():
            path=Path(index.loc[row.recording_id,'path'])
            if path.parent!=hear/done['signature']:raise ValueError('HeAR cache signature mismatch')
            with np.load(path,allow_pickle=False) as saved:values.append(saved['mean'].copy())
        matrices['hear']=np.stack(values)
        if matrices['hear'].shape!=(len(frame),512) or not np.isfinite(matrices['hear']).all():raise ValueError('Invalid HeAR vectors')
        sources['hear']={'signature':done['signature'],'representation':'512D window mean'}
    audio_names=list(matrices)
    matrices['early_fusion']=np.column_stack(list(matrices.values()))
    # Fixed age polynomial; categorical vocabulary uses training data only.
    # Imputation/scaling/PCA/classifiers are refitted within each CV fold.
    age=pd.to_numeric(frame.age,errors='coerce').to_numpy(dtype=float)/100
    train=frame.split.eq('train').to_numpy()
    device=frame.phoneInfo.fillna('unknown').astype(str)
    categories=sorted(device[train].unique())
    nuisance=np.column_stack([age,age**2,age**3,*[device.eq(c).to_numpy(dtype=float) for c in categories]])
    matrices['age_device']=nuisance
    matrices['age_device_whisper']=np.column_stack([nuisance,matrices['whisper_small']])
    matrices['age_device_fusion']=np.column_stack([nuisance,matrices['early_fusion']])
    specs=([{'kind':'lr','C':.1,'pca':None},{'kind':'lr','C':.1,'pca':8},{'kind':'svm','C':.1,'pca':None}] if a.smoke else None)
    results={};rows=[];yv=frame.loc[~train,'label'].to_numpy()
    predictions=frame.loc[~train,['recording_id','participant_id','duplicate_component','label','age']].copy()
    for name,x in matrices.items():
        print('Fitting',name,x.shape,flush=True)
        result=screen(x,frame,fold_count=2 if a.smoke else 3,specs=specs);results[name]=result
        chosen=result['selected'];rows.append({'experiment':name,'spec':json.dumps(chosen['spec']),'cv_auroc':chosen['cv_auroc'],**chosen['metrics']})
        predictions[name]=chosen['probability']
        joblib.dump(chosen['state'],out/f'{name}.joblib')
        (out/f'{name}_search.json').write_text(json.dumps([{'spec':c['spec'],'cv_auroc':c['cv_auroc']} for c in result['candidates']],indent=2))
        pd.DataFrame(rows).to_csv(out/'validation_results.csv',index=False)
    for name,r in fuse_fixed_lr([results[n] for n in audio_names]).items():
        rows.append({'experiment':name,**r['metrics']});predictions[name]=r['probability']
        if 'state' in r:joblib.dump(r['state'],out/f'{name}.joblib')
    pd.DataFrame(rows).to_csv(out/'validation_results.csv',index=False);predictions.to_parquet(out/'validation_predictions.parquet',index=False)
    # Paired component-bootstrap incremental AUROC, with the same resampled people.
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(20261112);groups=predictions.duplicate_component.to_numpy();unique=np.unique(groups);delta=[]
    p0=predictions.age_device.to_numpy();p1=predictions.age_device_fusion.to_numpy()
    for _ in range(200 if a.smoke else 1000):
        ix=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(yv[ix]))==2:delta.append(roc_auc_score(yv[ix],p1[ix])-roc_auc_score(yv[ix],p0[ix]))
    interval=np.quantile(delta,[.025,.975]).tolist()
    report={'status':'smoke_complete' if a.smoke else 'development_complete','manifest_sha256':sha,'sources':sources,
        'participants_train':int(train.sum()),'participants_val':int((~train).sum()),'test_labels_read':False,
        'primary_cohort':'canonical initial recording, no replacement after filtering','nuisance_device_categories':categories,
        'nuisance_age_basis':'fixed age/100, square, cube; missing imputed inside training folds',
        'incremental_fusion_auroc':float(roc_auc_score(yv,p1)-roc_auc_score(yv,p0)),
        'incremental_component_bootstrap_95pct':interval,'bootstrap_interpretation':'exploratory; does not correct extensive validation selection'}
    (out/'DONE.json').write_text(json.dumps(report,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
