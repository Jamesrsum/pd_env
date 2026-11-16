"""Initial-visit head comparisons, fold-fitted transforms and OOF-only fusion."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from src.training.development import operating_point


def initial_cohort(frame):
    if not frame.split.isin(['train','val']).all():
        raise ValueError('Sealed test forbidden')
    for identity in ['participant_id','duplicate_component']:
        if frame.groupby(identity).split.nunique().gt(1).any():
            raise ValueError('Participant/component crosses splits')
    initial=frame.loc[frame.initial_recording].copy()
    if initial.participant_id.duplicated().any():
        raise ValueError('Multiple canonical initial visits')
    if initial.label.isna().any() or not initial.label.isin([0,1]).all():
        raise ValueError('Resolved binary labels required')
    return initial


def head(spec, seed):
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]
    if spec.get('pca'):
        steps.append(('pca',PCA(n_components=spec['pca'],svd_solver='randomized',random_state=seed)))
    classifier=(LinearSVC(C=spec['C'],random_state=seed,max_iter=10000,dual='auto') if spec['kind']=='svm'
                else LogisticRegression(C=spec['C'],random_state=seed,max_iter=2000))
    return Pipeline(steps+[('head',classifier)])


def metrics(y,p):
    return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),
            'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),
            'participants':len(y),**operating_point(y,p)}


def screen(matrix,frame,seed=20261112,fold_count=3,specs=None):
    """One canonical initial assessment/person; all selection uses training CV."""
    cohort=initial_cohort(frame)
    if len(cohort)!=len(frame) or not np.array_equal(cohort.index,frame.index):
        raise ValueError('Provide exactly the canonical initial cohort in matrix order')
    x=np.asarray(matrix);train=frame.split.eq('train').to_numpy();xt=x[train];xv=x[~train]
    tr=frame.loc[train];y=tr.label.to_numpy();yv=frame.loc[~train,'label'].to_numpy()
    folds=list(StratifiedGroupKFold(n_splits=fold_count,shuffle=True,random_state=seed).split(xt,y,tr.duplicate_component))
    for a,b in folds:
        if set(tr.iloc[a].participant_id)&set(tr.iloc[b].participant_id):raise AssertionError('Person leakage')
        if len(np.unique(y[b]))<2:raise ValueError('CV fold lacks a class')
    if specs is None:
        specs=[{'kind':'lr','C':c,'pca':p} for p in [None,32,128] for c in [.01,.1,1,10]]
        specs += [{'kind':'svm','C':c,'pca':None} for c in [.01,.1,1]]
    candidates=[]
    for spec in specs:
        if spec.get('pca') and spec['pca']>min(xt.shape[1],min(len(a) for a,b in folds)):
            continue
        oof=np.full(len(y),np.nan);scores=[]
        for a,b in folds:
            fitted=head(spec,seed).fit(xt[a],y[a])
            raw=(fitted.decision_function(xt[b]) if spec['kind']=='svm' else fitted.predict_proba(xt[b])[:,1])
            oof[b]=raw;scores.append(roc_auc_score(y[b],raw))
        if not np.isfinite(oof).all():raise ValueError('Incomplete OOF predictions')
        fitted=head(spec,seed).fit(xt,y)
        if spec['kind']=='svm':
            calibration=LogisticRegression(C=1.,random_state=seed).fit(oof[:,None],y)
            probability=calibration.predict_proba(fitted.decision_function(xv)[:,None])[:,1]
        else:
            calibration=None;probability=fitted.predict_proba(xv)[:,1]
        candidates.append({'spec':spec,'cv_auroc':float(np.mean(scores)), 'oof':oof,
                           'probability':probability,'state':{'pipeline':fitted,'calibrator':calibration}})
    if not candidates:raise ValueError('No feasible classifiers')
    chosen=max(candidates,key=lambda c:c['cv_auroc'])
    return {'selected':{**chosen,'metrics':metrics(yv,chosen['probability'])},'candidates':candidates,
            'folds':folds,'train_labels':y,'validation_labels':yv}


def fuse_fixed_lr(results):
    """Predeclared LR C=.1 OOF predictions, avoiding selected-head stacking bias."""
    fixed=[]
    for result in results:
        fixed.append(next(c for c in result['candidates'] if c['spec']=={'kind':'lr','C':.1,'pca':None}))
    y=results[0]['train_labels'];yv=results[0]['validation_labels']
    for r in results[1:]:
        if not np.array_equal(y,r['train_labels']) or not np.array_equal(yv,r['validation_labels']):
            raise ValueError('Fusion cohort mismatch')
    train=np.column_stack([c['oof'] for c in fixed]);val=np.column_stack([c['probability'] for c in fixed])
    stack=LogisticRegression(C=.1,max_iter=1000).fit(train,y)
    average=val.mean(1);stacked=stack.predict_proba(val)[:,1]
    return {'equal_average':{'probability':average,'metrics':metrics(yv,average)},
            'oof_stack':{'probability':stacked,'metrics':metrics(yv,stacked),'state':stack}}
