"""Train-only PCA and locally executed, fixed-checkpoint tabular foundation heads."""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def train_pca(x,n_components,seed):
    if n_components>=len(x):raise ValueError('PCA dimension must be below training size')
    return make_pipeline(StandardScaler(),PCA(n_components=n_components,svd_solver='randomized',random_state=seed)).fit(x)


def logistic(x,y,groups,xv,n_components,seed):
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(x,y,groups))
    search=[]
    for c in [.001,.01,.1,1.]:
        scores=[]
        for a,b in folds:
            transform=train_pca(x[a],n_components,seed)
            head=LogisticRegression(C=c,max_iter=2000).fit(transform.transform(x[a]),y[a])
            scores.append(roc_auc_score(y[b],head.predict_proba(transform.transform(x[b]))[:,1]))
        search.append({'C':c,'cv_auroc':float(np.mean(scores))})
    selected=max(search,key=lambda r:r['cv_auroc'])['C']
    transform=train_pca(x,n_components,seed)
    head=LogisticRegression(C=selected,max_iter=2000).fit(transform.transform(x),y)
    return head.predict_proba(transform.transform(xv))[:,1],{'transform':transform,'head':head,'search':search}


def tabular_predict(xt,y,xv,model_path,checkpoint_version,seed,device,estimators,smoke=False):
    from tabicl import TabICLClassifier
    path=Path(model_path)
    if not path.is_file():raise FileNotFoundError('Local pinned TabICL checkpoint required')
    model=TabICLClassifier(model_path=str(path),allow_auto_download=False,checkpoint_version=checkpoint_version,
        n_estimators=2 if smoke else estimators,device=device,random_state=seed,n_jobs=4,
        use_amp=False,use_fa3=False,batch_size=2)
    # fit receives training context ONLY. Validation labels never enter the model.
    model.fit(xt,y)
    if list(model.classes_)!=[0,1]:raise ValueError('TabICL binary class ordering mismatch')
    return model.predict_proba(xv)[:,1]
