#!/usr/bin/env python3
"""Aggregate the completed adjacent-field classifier panel; no individual exports."""
import argparse
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.training.adaptation_runner import training_manifest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--snapshot',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    done=json.loads((a.root/'tabular_runs/DONE.json').read_text())
    frame,sha=training_manifest(a.snapshot)
    if done['status']!='development_complete' or done['manifest_sha256']!=sha or done['test_labels_read']:raise ValueError('Invalid completed panel')
    val=frame.loc[frame.split.eq('val') & frame.initial_recording].sort_values('recording_id').reset_index(drop=True)
    table=pd.read_csv(a.root/'tabular_runs/validation_results.csv');predictions={}
    y=val.label.to_numpy()
    for row in table.itertuples():
        key=(row.encoder,row.labelled_train_people,row.classifier)
        d=pd.read_csv(a.root/'tabular_runs'/row.encoder/str(row.labelled_train_people)/f'{row.classifier}_predictions.csv',dtype={'recording_id':str,'participant_id':str})
        if d.recording_id.duplicated().any() or set(d.recording_id)!=set(val.recording_id):raise ValueError('Cohort mismatch')
        d=d.set_index('recording_id').loc[val.recording_id]
        if not np.array_equal(d.label,y) or not np.array_equal(d.participant_id,val.participant_id):raise ValueError('Identity/label mismatch')
        predictions[key]=d.probability.to_numpy(dtype=float)
        if abs(roc_auc_score(y,predictions[key])-row.auroc)>1e-10:raise ValueError('Saved AUROC not reproducible')
    max_labels=int(table.labelled_train_people.max())
    pairs={}
    for encoder in ['hear','whisper_small','wavlm_large']:
        pairs[f'{encoder}_v2_minus_LR']=((encoder,max_labels,'tabicl_v2'),(encoder,max_labels,'lr'))
    pairs['wavlm_v2_minus_v1']=(('wavlm_large',max_labels,'tabicl_v2'),('wavlm_large',max_labels,'tabicl_v1'))
    for n in [50,100,500]:pairs[f'hear_{n}_labels_v2_minus_LR']=(('hear',n,'tabicl_v2'),('hear',n,'lr'))
    groups=val.duplicate_component.to_numpy();unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique]
    rng=np.random.default_rng(20261116);values={k:[] for k in pairs}
    needed=set(k for pair in pairs.values() for k in pair)
    for _ in range(2000):
        ix=np.concatenate([members[i] for i in rng.integers(len(members),size=len(members))])
        if len(np.unique(y[ix]))<2:continue
        auc={k:roc_auc_score(y[ix],predictions[k][ix]) for k in needed}
        for name,(left,right) in pairs.items():values[name].append(auc[left]-auc[right])
    comparisons={name:{'auroc_delta':float(roc_auc_score(y,predictions[left])-roc_auc_score(y,predictions[right])),
        'paired_component_bootstrap_95pct':np.quantile(values[name],[.025,.975]).tolist()} for name,(left,right) in pairs.items()}
    summary={'status':'completed_development_panel','comparisons':len(table),'validation_participants':len(val),
        'validation_PD':int(y.sum()),'test_labels_read':False,'paired_comparisons':comparisons,
        'bootstrap_replicates':2000,'training_context_seeds':[20261112],
        'limitations':['Exploratory validation; no correction for model selection or training-context seed variability.',
            'Single non-nested stratified context per budget; PCA dimensions vary. Compare classifiers within a budget, not a pure causal label-count effect.',
            'TabICL pretraining uses synthetic tables; small labelled context is not foundation training from scratch.']}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');table.to_csv(a.output/'metrics.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
