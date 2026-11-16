#!/usr/bin/env python3
"""Export aggregate acoustic tree results and conditional paired uncertainty."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root/'runs';done=json.loads((root/'DONE.json').read_text())
    if done['status']!='development_complete' or done['test_labels_read'] or not done['GPU_hist_verified']:raise ValueError('Incomplete or invalid GPU run')
    files={'primary':root/'acoustic_primary/validation_predictions.csv',
        'repeated':root/'equal_person_repeated_visits/validation_predictions.csv','logistic':root/'logistic_reference_predictions.csv'}
    predictions={};reference=None
    for name,path in files.items():
        d=pd.read_csv(path,dtype={'recording_id':str,'participant_id':str,'duplicate_component':str}).sort_values('recording_id').reset_index(drop=True)
        if d.participant_id.duplicated().any():raise ValueError('Multiple validation visits')
        if reference is None:reference=d
        elif not reference[['recording_id','participant_id','duplicate_component','label']].equals(d[['recording_id','participant_id','duplicate_component','label']]):raise ValueError('Paired prediction identity mismatch')
        predictions[name]=d.probability.to_numpy(dtype=float)
    y=reference.label.to_numpy();groups=reference.duplicate_component.to_numpy();unique=np.unique(groups)
    members=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(20261116)
    pairs={'primary_minus_logistic':('primary','logistic'),'repeated_minus_primary':('repeated','primary'),'repeated_minus_logistic':('repeated','logistic')};values={k:[] for k in pairs}
    for _ in range(2000):
        ix=np.concatenate([members[i] for i in rng.integers(len(members),size=len(members))])
        if len(np.unique(y[ix]))<2:continue
        auc={k:roc_auc_score(y[ix],v[ix]) for k,v in predictions.items()}
        for name,(left,right) in pairs.items():values[name].append(auc[left]-auc[right])
    comparisons={name:{'auroc_difference':float(roc_auc_score(y,predictions[left])-roc_auc_score(y,predictions[right])),
        'paired_component_bootstrap_95pct':np.quantile(values[name],[.025,.975]).tolist()} for name,(left,right) in pairs.items()}
    a.output.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(root/'validation_results.csv');table.to_csv(a.output/'metrics.csv',index=False)
    importance=pd.read_csv(root/'feature_importance.csv');importance.to_csv(a.output/'feature_importance.csv',index=False)
    trials=pd.read_csv(root/'search/trials.csv');cols=[c for c in trials if c in ['number','value','state'] or c.startswith(('params_','user_attrs_'))];trials[cols].to_csv(a.output/'search_trials.csv',index=False)
    report={'status':'completed_GPU_acoustic_XGBoost','test_labels_read':False,'validation_people':len(y),'validation_PD':int(y.sum()),
        'train_people':done['train_people'],'repeated_train_recordings':done['repeated_training_rows'],
        'feature_count':done['features'],'best_search':done['best_search'],'paired_comparisons':comparisons,
        'GPU_verification':json.loads((root/'search/device_verified.json').read_text()),
        'model_reload_verified':done['model_reload_verified'],'SHAP_sum_verified':done['GPU_SHAP_sum_verified'],
        'uncertainty':'Exploratory validation intervals conditional on selected models; not corrected for model selection or training seeds.',
        'comparison_design':'Primary versus logistic uses initial training visits. Repeated-visit model uses more training recordings with equal-person weights and fixed primary hyperparameters.',
        'feature_span':'Existing full-clip acoustic measurements; not recomputed at encoder 10s cap.'}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='xgb-acoustics-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ax[0].scatter(trials.number,trials.value,s=12,alpha=.6,label='Trial CV AUROC');ax[0].plot(trials.number,trials.value.cummax(),label='Best CV AUROC')
    ax[0].set(xlabel='TPE trial',ylabel='Training grouped-CV AUROC',title='GPU XGBoost: 60 trials');ax[0].legend(fontsize=8)
    top=importance.head(12).iloc[::-1];labels=[s.replace('egemaps__','eGeMAPS: ').replace('spectral__','Spectral: ').replace('praat_full__','Praat: ') for s in top.feature]
    ax[1].barh(labels,top.mean_abs_SHAP_log_odds);ax[1].set(xlabel='Mean |SHAP|, log-odds',title='Primary model: validation attribution');ax[1].tick_params(axis='y',labelsize=7)
    fig.savefig(a.output/'search_and_importance.png',dpi=160);plt.close(fig)
    print(json.dumps({'best_search':report['best_search'],'paired_comparisons':comparisons},indent=2))


if __name__=='__main__':main()
