"""Participant-grouped GPU tree search using TRAIN data only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PREFIXES=('praat_full__','praat_stable__','egemaps__','spectral__')


def acoustic_join(frame,features,quality,strict_schema=True):
    if not frame.split.isin(['train','val']).all():raise ValueError('Sealed test rows forbidden')
    if frame.recording_id.duplicated().any():raise ValueError('Duplicate canonical recording IDs')
    for name in ['participant_id','duplicate_component']:
        if frame.groupby(name).split.nunique().gt(1).any():raise ValueError('Participant/component split leakage')
    columns=[c for c in features if c.startswith(PREFIXES)]
    counts={p:sum(c.startswith(p) for c in columns) for p in PREFIXES}
    if strict_schema and list(counts.values())!=[22,22,88,52]:raise ValueError(f'Expected established 184-feature schema: {counts}')
    if not columns:raise ValueError('No acoustic features')
    if features.recording_id.duplicated().any() or quality.recording_id.duplicated().any():raise ValueError('Duplicate feature/source recording IDs')
    table=features.set_index('recording_id');qc=quality.set_index('recording_id')
    if not set(frame.recording_id)<=set(table.index) or not set(frame.recording_id)<=set(qc.index):raise ValueError('Missing canonical acoustic/source rows; do not replace initial visits')
    aligned=table.loc[frame.recording_id,columns]
    q=qc.loc[frame.recording_id]
    if q.file_sha256.isna().any() or frame.file_sha256.isna().any():raise ValueError('Missing acoustic source hashes')
    if not np.array_equal(q.file_sha256.astype(str).to_numpy(),frame.file_sha256.astype(str).to_numpy()):raise ValueError('Acoustic source hash mismatch')
    x=aligned.to_numpy(dtype=np.float32)
    infinite=int(np.isinf(x).sum());x[~np.isfinite(x)]=np.nan
    if np.isnan(x).all(1).any():raise ValueError('Entirely missing acoustic row; extraction must be audited')
    return x,columns,{'feature_counts':counts,'infinities_mapped_to_missing':infinite,
        'source_hashes_matched':True,'rows':len(frame),'feature_count':len(columns)}


def grouped_folds(frame,n_splits,seed):
    if not frame.split.eq('train').all():raise ValueError('Hyperparameter search accepts TRAIN rows only')
    if frame.participant_id.duplicated().any():raise ValueError('Search requires one canonical initial training visit/person')
    if not frame.initial_recording.all():raise ValueError('Noncanonical training endpoint')
    folds=list(StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(
        np.zeros(len(frame)),frame.label.to_numpy(),frame.duplicate_component))
    for a,b in folds:
        if set(frame.iloc[a].duplicate_component)&set(frame.iloc[b].duplicate_component):raise ValueError('CV component leakage')
        if len(np.unique(frame.iloc[a].label))<2 or len(np.unique(frame.iloc[b].label))<2:raise ValueError('CV fold lacks a class')
    return folds


def base_parameters(cfg):
    return {'objective':'binary:logistic','eval_metric':'auc','tree_method':'hist',
            'device':cfg['device'],'nthread':cfg['threads'],'seed':cfg['seed'],'validate_parameters':True}


def check_device(booster,requested):
    config=booster.save_config();device=json.loads(config)['learner']['generic_param']['device']
    if requested.startswith('cuda') and (not device.startswith('cuda') or 'grow_gpu_hist' not in config):
        raise RuntimeError(f'XGBoost did not use GPU hist: requested {requested}, actual {device}')
    return device


def suggest(trial):
    return {'max_depth':trial.suggest_int('max_depth',2,8),
        'min_child_weight':trial.suggest_float('min_child_weight',1.,64.,log=True),
        'eta':trial.suggest_float('eta',.005,.2,log=True),
        'subsample':trial.suggest_float('subsample',.6,1.),
        'colsample_bytree':trial.suggest_float('colsample_bytree',.5,1.),
        'lambda':trial.suggest_float('lambda',.001,100.,log=True),
        'alpha':trial.suggest_float('alpha',1e-8,10.,log=True),
        'gamma':trial.suggest_float('gamma',0.,5.),
        'max_bin':trial.suggest_categorical('max_bin',[128,256,512])}


def search(x,training,cfg,out):
    """No validation features or labels are accepted by this function."""
    import xgboost as xgb
    import optuna
    import joblib
    import fcntl
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    folds=grouped_folds(training,cfg['cv_folds'],cfg['seed'])
    x=np.asarray(x,dtype=np.float32);y=training.label.to_numpy()
    signature=hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k!='trials'},sort_keys=True).encode()
        +x.tobytes()+y.tobytes()+training.recording_id.astype(str).str.cat(sep='|').encode()
        +training.participant_id.astype(str).str.cat(sep='|').encode()
        +training.duplicate_component.astype(str).str.cat(sep='|').encode()).hexdigest()
    metadata=out/'search_identity.json'
    lock=(out/'search.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if metadata.exists() and json.loads(metadata.read_text())['signature']!=signature:raise ValueError('Resume search/data/config mismatch')
        metadata.write_text(json.dumps({'signature':signature,'selection':'training-only grouped CV mean AUROC',
            'validation_used_for_search':False,'train_people':len(training),'folds':len(folds)},indent=2))
        sampler_path=out/'sampler.joblib'
        sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(
            seed=cfg['seed'],n_startup_trials=12,multivariate=True)
        study=optuna.create_study(study_name='acoustic_initial_training',direction='maximize',sampler=sampler,
            storage='sqlite:///'+str((out/'study.sqlite3').resolve()),load_if_exists=True)
        for trial in study.trials:
            if trial.state==optuna.trial.TrialState.RUNNING:study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
        matrix=xgb.DMatrix(x,label=y,missing=np.nan)
        base=base_parameters(cfg)
        proof=xgb.train({**base,'max_depth':2},matrix,num_boost_round=2)
        actual=check_device(proof,cfg['device'])
        (out/'device_verified.json').write_text(json.dumps({'requested':cfg['device'],'actual':actual,
            'CUDA_build':xgb.build_info().get('USE_CUDA',False),'histogram_verified':True},indent=2))
        if not study.trials:
            study.enqueue_trial({'max_depth':3,'min_child_weight':5.,'eta':.05,'subsample':.8,
                'colsample_bytree':.8,'lambda':5.,'alpha':.01,'gamma':0.,'max_bin':256})
        def objective(trial):
            parameters=suggest(trial)
            cv=xgb.cv({**base,**parameters},matrix,folds=folds,num_boost_round=cfg['max_rounds'],
                early_stopping_rounds=cfg['early_stopping_rounds'],maximize=True,shuffle=False,
                verbose_eval=False,seed=cfg['seed'])
            if len(cv)==0 or not np.isfinite(cv['test-auc-mean'].iloc[-1]):raise ValueError('Nonfinite CV score')
            trial.set_user_attr('boost_rounds',len(cv));trial.set_user_attr('cv_auc_std',float(cv['test-auc-std'].iloc[-1]))
            cv.to_csv(out/f'trial_{trial.number:03d}_cv.csv',index=False)
            return float(cv['test-auc-mean'].iloc[-1])
        def checkpoint(study,trial):
            temporary=sampler_path.with_suffix('.tmp.joblib');joblib.dump(study.sampler,temporary);temporary.replace(sampler_path)
            study.trials_dataframe().to_csv(out/'trials.csv',index=False)
            (out/'progress.json').write_text(json.dumps({'trials_finished':sum(t.state.is_finished() for t in study.trials),
                'target_trials':cfg['trials'],'best_training_cv_auroc':study.best_value,'best_trial':study.best_trial.number},indent=2))
        completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective,n_trials=max(0,cfg['trials']-completed),callbacks=[checkpoint],n_jobs=1)
        best=study.best_trial
        result={'params':best.params,'boost_rounds':best.user_attrs['boost_rounds'],'cv_auroc':best.value,
            'cv_auc_std':best.user_attrs['cv_auc_std'],'trial':best.number,
            'completed_trials':sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),'signature':signature}
        (out/'best_parameters.json').write_text(json.dumps(result,indent=2));return result
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
