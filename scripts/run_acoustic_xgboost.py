#!/usr/bin/env python3
"""GPU XGBoost/TPE on verified acoustic columns, without opening the final test."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.training.adaptation_runner import training_manifest
from src.training.downstream import initial_cohort,metrics
from src.data.frozen_protocol import participant_weights
from src.training.acoustic_xgboost import PREFIXES,acoustic_join,grouped_folds,search,base_parameters,check_device


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.umask(0o077)
    cfg=yaml.safe_load((REPO/'configs/acoustic_xgboost.yaml').read_text())
    if not xgb.build_info().get('USE_CUDA'):raise RuntimeError('A CUDA-enabled XGBoost wheel is required; inherited package is CPU-only')
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'])
    if a.smoke:
        ids=initial_cohort(frame).groupby(['split','label'],group_keys=False).head(16).participant_id
        selected=frame.loc[frame.participant_id.isin(ids)]
        frame=pd.concat([selected.loc[selected.initial_recording],selected.loc[~selected.initial_recording].groupby('participant_id').head(1)]).sort_values('recording_id').reset_index(drop=True)
        cfg.update(trials=2,max_rounds=32,early_stopping_rounds=5,cv_folds=2)
    processed=Path(os.environ['MPOWER_OUTPUT_ROOT'])/'processed'
    feature_path=processed/'full_acoustic_features.parquet';quality_path=processed/'full_audio_quality_metrics.parquet'
    columns=[c for c in pq.read_schema(feature_path).names if c.startswith(PREFIXES)]
    # Predicate pushdown selects development IDs; source features have no labels.
    filters=[('recording_id','in',frame.recording_id.tolist())]
    features=pd.read_parquet(feature_path,columns=['recording_id',*columns],filters=filters)
    quality=pd.read_parquet(quality_path,columns=['recording_id','file_sha256','analysis_samples','analysis_sample_rate'],filters=filters)
    x,columns,audit=acoustic_join(frame,features,quality)
    cfg['manifest_sha256']=sha;cfg['feature_source_sha256']=hashlib.sha256(feature_path.read_bytes()).hexdigest()
    cfg['quality_source_sha256']=hashlib.sha256(quality_path.read_bytes()).hexdigest()
    cfg['feature_columns']=columns
    root=Path(os.environ['MPOWER_XGBOOST_ROOT']).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    out=root/('smoke' if a.smoke else 'runs');out.mkdir(parents=True,exist_ok=True)
    initial=frame.initial_recording.to_numpy();train=frame.split.eq('train').to_numpy();val=~train & initial
    tr=frame.loc[train & initial].reset_index(drop=True);va=frame.loc[val].reset_index(drop=True)
    xt=x[train & initial];xv=x[val];y=tr.label.to_numpy();yv=va.label.to_numpy()
    if len(initial_cohort(frame).query('split == "train"'))!=len(tr):raise ValueError('Canonical training cohort mismatch')
    provenance={'config':cfg,'join_audit':audit,'versions':{p:importlib.metadata.version(p) for p in ['xgboost','optuna','numpy','scikit-learn']},
        'test_labels_read':False,'source_span':'existing full-clip acoustic extraction; not recomputed at the frozen-model 10s cap',
        'excluded_input_fields':'IDs, labels, demographics, medication, device, timestamps, statuses and QC metadata',
        'feature_schema':columns,'initial_training_people':len(tr),'initial_validation_people':len(va)}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    frame[['recording_id','participant_id','duplicate_component','split','initial_recording']].to_parquet(out/'private_input_index.parquet',index=False)
    best=search(xt,tr,cfg,out/'search')
    parameters={**base_parameters(cfg),**best['params']};names=[f'f{i:03d}' for i in range(len(columns))]
    rows=[];predictions={}
    def finish(name,matrix,target,validation,features_used,weights=None):
        local=[names[i] for i in features_used]
        model=xgb.train(parameters,xgb.DMatrix(matrix,label=target,weight=weights,feature_names=local,missing=np.nan),num_boost_round=best['boost_rounds'])
        check_device(model,'cuda')
        directory=out/name;directory.mkdir(exist_ok=True)
        model.save_model(directory/'model.ubj')
        dv=xgb.DMatrix(validation,feature_names=local,missing=np.nan);probability=model.predict(dv)
        restored=xgb.Booster();restored.load_model(directory/'model.ubj');restored.set_param({'device':'cuda'})
        np.testing.assert_allclose(probability,restored.predict(dv),rtol=1e-6,atol=1e-7)
        d=va[['recording_id','participant_id','duplicate_component','label']].copy();d['probability']=probability
        d.to_csv(directory/'validation_predictions.csv',index=False);predictions[name]=probability
        row={'experiment':name,'features':len(features_used),'boost_rounds':best['boost_rounds'],**metrics(yv,probability)}
        rows.append(row);(directory/'validation_metrics.json').write_text(json.dumps(row,indent=2))
        if name=='acoustic_primary':
            contrib=model.predict(dv,pred_contribs=True)
            np.testing.assert_allclose(contrib.sum(1),model.predict(dv,output_margin=True),rtol=1e-3,atol=1e-4)
            importance=pd.DataFrame({'feature':columns,'mean_abs_SHAP_log_odds':np.abs(contrib[:,:-1]).mean(0),
                'gain':[model.get_score(importance_type='gain').get(n,0.) for n in names]})
            importance.sort_values('mean_abs_SHAP_log_odds',ascending=False).to_csv(out/'feature_importance.csv',index=False)
            tree=model.trees_to_dataframe();tree['acoustic_feature']=tree.Feature.map(dict(zip(names,columns))).fillna('Leaf')
            tree.to_csv(out/'tree_structure.csv',index=False)
        print(json.dumps(row),flush=True)
    all_features=list(range(len(columns)))
    finish('acoustic_primary',xt,y,xv,all_features)
    for name,prefixes in [('praat_only',('praat_full__','praat_stable__')),('egemaps_only',('egemaps__',)),('spectral_only',('spectral__',))]:
        ix=[i for i,c in enumerate(columns) if c.startswith(prefixes)]
        finish(name,xt[:,ix],y,xv[:,ix],ix)
    finish('missingness_only',np.isnan(xt).astype('float32'),y,np.isnan(xv).astype('float32'),all_features)
    weights=participant_weights(frame.loc[train,'participant_id'])
    finish('equal_person_repeated_visits',x[train],frame.loc[train,'label'].to_numpy(),xv,all_features,weights)
    # A linear reference, including train-fitted imputation/missingness indicators.
    folds=grouped_folds(tr,cfg['cv_folds'],cfg['seed']);linear=[]
    def linear_head(c):return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,max_iter=2000))
    for c in [.001,.01,.1,1.]:
        auc=[]
        for ia,ib in folds:
            head=linear_head(c).fit(xt[ia],y[ia]);auc.append(roc_auc_score(y[ib],head.predict_proba(xt[ib])[:,1]))
        linear.append({'C':c,'cv_auroc':float(np.mean(auc))})
    chosen=max(linear,key=lambda r:r['cv_auroc'])['C'];lr=linear_head(chosen).fit(xt,y);lp=lr.predict_proba(xv)[:,1]
    joblib.dump(lr,out/'logistic_reference.joblib');d=va[['recording_id','participant_id','duplicate_component','label']].copy();d['probability']=lp;d.to_csv(out/'logistic_reference_predictions.csv',index=False)
    rows.append({'experiment':'logistic_reference','chosen_C':chosen,**metrics(yv,lp)});predictions['logistic_reference']=lp
    pd.DataFrame(rows).to_csv(out/'validation_results.csv',index=False)
    (out/'linear_cv.json').write_text(json.dumps(linear,indent=2))
    # Only aggregate head/feature diagnostics are exported; participant results stay private.
    status={'status':'real_GPU_smoke_passed' if a.smoke else 'development_complete','test_labels_read':False,
        'GPU_hist_verified':True,'GPU_SHAP_sum_verified':True,'model_reload_verified':True,'features':len(columns),
        'manifest_sha256':sha,'train_people':len(tr),'validation_people':len(va),'repeated_training_rows':int(train.sum()),
        'repeated_training_weight_sum':float(weights.sum()),'best_search':best,'comparisons':len(rows),
        'diagnostic_hyperparameters':'feature-block/missingness/repeated-visit arms reuse primary CV hyperparameters, without retuning on validation'}
    (out/'DONE.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)


if __name__=='__main__':main()
