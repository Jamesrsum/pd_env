#!/usr/bin/env python3
"""Audit completed HeAR development runs and export aggregate results only."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,roc_curve
from src.training.adaptation_runner import training_manifest
from src.training.development import operating_point


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--baseline-panel',type=Path)
    parser.add_argument('--ast-root',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--bootstrap',type=int,default=2000)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    frame,sha=training_manifest(a.snapshot)
    val=frame.loc[frame.split.eq('val') & frame.initial_recording].sort_values('recording_id').reset_index(drop=True)
    if val.participant_id.duplicated().any():raise ValueError('Repeated initial participant')
    y=val.label.to_numpy();prob={};checks={};histories={}
    def predictions(path,column='probability'):
        d=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,dtype={'recording_id':str,'participant_id':str})
        if d.recording_id.duplicated().any() or set(d.recording_id)!=set(val.recording_id):raise ValueError('Prediction cohort mismatch')
        d=d.set_index('recording_id').loc[val.recording_id]
        if not np.array_equal(d.label.to_numpy(),y) or not np.array_equal(d.participant_id.to_numpy(),val.participant_id.to_numpy()):raise ValueError('Prediction identity/label mismatch')
        p=d[column].to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
        return p
    frozen=json.loads((a.root/'runs/frozen/DONE.json').read_text())
    panel=json.loads((a.root/'downstream_panel/full/DONE.json').read_text())
    if frozen['manifest_sha256']!=sha or panel['manifest_sha256']!=sha or not frozen['encoder_unchanged']:raise ValueError('Provenance mismatch')
    for representation in ['mean','mean_sd']:
        for head in ['logistic','mlp']:
            prob[f'frozen_{representation}_{head}']=predictions(a.root/f'runs/frozen/heads/{representation}/{head}/validation_predictions.csv')
    done=[]
    for name,relative in [('warmup','warmup'),('native_frozen','adaptation/frozen'),('last2','adaptation/last2'),('full','adaptation/full')]:
        path=a.root/'runs'/relative;d=json.loads((path/'DONE.json').read_text())
        if d['status']!='complete' or d['test_labels_read'] or not d['head_changed'] or not d['frozen_parameters_unchanged']:raise ValueError('Incomplete or invalid training audit')
        if d['backbone_changed']!=(name in ['last2','full']):raise ValueError('Backbone change audit failed')
        prob[name]=predictions(path/'validation_predictions.csv')
        histories[name]=json.loads((path/'history.json').read_text())
        if name!='warmup':done.append(d)
        checks[name]={k:d[k] for k in ['selected_epoch','initial_auroc','backbone_changed','head_changed','frozen_parameters_unchanged']}
    if len({d['initial_backbone_sha256'] for d in done})!=1 or len({d['initial_head_sha256'] for d in done})!=1:raise ValueError('Unmatched branch initialization')
    for name in pd.read_csv(a.root/'downstream_panel/full/validation_results.csv').experiment:
        prob['panel_'+name]=predictions(a.root/'downstream_panel/full/validation_predictions.parquet',name)
    if a.baseline_panel:prob['two_encoder_early_fusion']=predictions(a.baseline_panel/'validation_predictions.parquet','early_fusion')
    ast_checks={}
    if a.ast_root:
        for policy in ['frozen','last2','full']:
            path=a.ast_root/'runs'/policy;d=json.loads((path/'DONE.json').read_text())
            if d['status']!='complete' or d['test_labels_read'] or not d['frozen_parameters_unchanged']:raise ValueError('Invalid AST completion audit')
            prob['ast_'+policy]=predictions(path/'validation_predictions.csv')
            ast_checks[policy]={k:d[k] for k in ['best_phase_auroc','selected_epoch','backbone_changed','head_changed','frozen_parameters_unchanged']}
    results={}
    for name,p in prob.items():
        results[name]={'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'log_loss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p)),
            'accuracy_at_0_5':float(np.mean((p>=.5)==y)),**operating_point(y,p)}
    pairs={'full_minus_native_frozen':('full','native_frozen'),
        'last2_minus_native_frozen':('last2','native_frozen'),
        'full_minus_frozen_logistic':('full','frozen_mean_logistic'),
        'three_encoder_fusion_minus_best_single_WavLM':('panel_early_fusion','panel_wavlm_large'),
        'age_device_three_encoder_fusion_minus_age_device':('panel_age_device_fusion','panel_age_device'),
        'age_device_whisper_minus_age_device':('panel_age_device_whisper','panel_age_device')}
    if a.baseline_panel:pairs['three_minus_two_encoder_fusion']=('panel_early_fusion','two_encoder_early_fusion')
    if a.ast_root:
        pairs['AST_full_minus_frozen']=('ast_full','ast_frozen')
        pairs['AST_last2_minus_frozen']=('ast_last2','ast_frozen')
    groups=val.duplicate_component.to_numpy();unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(20261116);samples={k:[] for k in pairs}
    for _ in range(a.bootstrap):
        ix=np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])
        if len(np.unique(y[ix]))<2:continue
        auc={n:roc_auc_score(y[ix],prob[n][ix]) for n in set(sum((list(p) for p in pairs.values()),[]))}
        for name,(left,right) in pairs.items():samples[name].append(auc[left]-auc[right])
    comparisons={name:{'auroc_delta':results[left]['auroc']-results[right]['auroc'],
        'paired_component_bootstrap_95pct':np.quantile(samples[name],[.025,.975]).tolist()} for name,(left,right) in pairs.items()}
    report={'status':'completed_development_analysis','manifest_sha256':sha,'recordings_extracted':frozen['recordings'],
        'participants_train':int(frame.loc[frame.split.eq('train'),'participant_id'].nunique()),'validation_participants':len(y),
        'validation_pd':int(y.sum()),'validation_controls':int((y==0).sum()),'validation_duplicate_components':len(unique),
        'test_labels_read':False,'branch_initialization_matched':True,'training_audits':checks,
        'corrected_AST_audits':ast_checks,
        'metrics':results,'paired_comparisons':comparisons,'bootstrap_replicates':a.bootstrap,
        'uncertainty_limit':'Exploratory development intervals; conditional on fitted models/selected checkpoints, not corrected for model selection or training seeds.',
        'training_designs':{'frozen_probe':'all training visits with equal-person weights; train-CV LR selection',
            'native_adaptation':'one visit/person/epoch; identical LayerNorm+linear head and best warm-up for branches',
            'downstream_panel':'canonical initial training visit/person; train-CV head selection; training OOF calibration/stacking'},
        'threshold_selection':'validation only; not a prospective test operating point'}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame([{'experiment':name,**m} for name,m in results.items()]).to_csv(a.output/'metrics.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='hear-results-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,label in [('native_frozen','Frozen backbone'),('last2','Last two blocks'),('full','Full fine-tuning')]:
        h=histories[name];axes[0].plot([r['epoch'] for r in h],[r['auroc'] for r in h],label=label)
    axes[0].axhline(results['frozen_mean_logistic']['auroc'],color='black',ls='--',label='Frozen mean + LR (different head)')
    axes[0].set(xlabel='Branch epoch after shared warm-up',ylabel='Validation AUROC',title='Controlled HeAR adaptation');axes[0].legend(fontsize=8)
    for name,label in [('frozen_mean_logistic','Frozen HeAR + LR'),('full','Full HeAR fine-tuning'),('panel_early_fusion','Three-encoder fusion'),('panel_age_device','Age + device')]:
        fpr,tpr,_=roc_curve(y,prob[name]);axes[1].plot(fpr,tpr,label=f'{label}: {results[name]["auroc"]:.3f}')
    axes[1].plot([0,1],[0,1],color='gray',ls=':');axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title=f'Initial validation visits (n={len(y)})');axes[1].legend(fontsize=8)
    fig.savefig(a.output/'learning_and_roc.png',dpi=180);plt.close(fig)
    print(json.dumps({'metrics':results,'paired_comparisons':comparisons},indent=2),flush=True)


if __name__=='__main__':main()
