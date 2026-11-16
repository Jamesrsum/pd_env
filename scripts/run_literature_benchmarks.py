#!/usr/bin/env python3
"""Train bounded, source-audited literature adaptations on sealed-test-safe inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from src.models.literature_adapter import LiteratureAdapter
from src.training.adaptation_runner import training_manifest
from src.training.controlled_adaptation import seed_all
from src.training.literature_runner import run,predict


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--method',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.umask(0o077);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    options=yaml.safe_load((REPO/'configs/literature_benchmarks.yaml').read_text())
    cfg={**options['methods'][a.method],'method':a.method,'seed':options['seed']}
    cfg['declared_deviations']=options['declared_deviations']
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'],a.smoke,16 if cfg['family']=='hubert_supcon' else 4)
    if a.smoke:cfg['epochs']=2
    cfg['manifest_sha256']=sha
    seed_all(cfg['seed']);lp=None
    if cfg['family']=='hear_lpft':
        source=Path(os.environ['MPOWER_HEAR_SOURCE'])/'runs/frozen'
        done=json.loads((source/'DONE.json').read_text())
        if done['manifest_sha256']!=sha or not done['encoder_unchanged']:raise ValueError('LP source provenance mismatch')
        path=source/'heads/mean/logistic/head.joblib';lp=joblib.load(path)
        cfg['lp_head_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();cfg['lp_source_signature']=done['signature']
    model=LiteratureAdapter(os.environ['MPOWER_MODEL_ROOT'],REPO,cfg,lp)
    cfg['model_revision']=model.asset['revision']
    if cfg['family']=='hubert_supcon':
        cfg['actual_encoder_augmentation']=model.runtime_config
        if any(model.runtime_config.values()):raise ValueError('Actual loaded encoder augmentation was not disabled')
    model.to('cuda')
    if cfg['family']=='hear_lpft':
        val=frame.loc[frame.split.eq('val') & frame.initial_recording]
        expected=pd.read_csv(source/'heads/mean/logistic/validation_predictions.csv',dtype={'recording_id':str}).set_index('recording_id')
        actual=predict(model,val,'cuda')
        np.testing.assert_allclose(actual,expected.loc[val.recording_id,'probability'],rtol=2e-4,atol=2e-5)
        cfg['converged_LP_prediction_equivalence_verified']=True
    root=Path(os.environ['MPOWER_LITERATURE_ROOT']).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    out=root/('smoke' if a.smoke else 'runs')/a.method;out.mkdir(parents=True,exist_ok=True)
    (out/'provenance.json').write_text(json.dumps({'config':cfg,'asset':model.asset,'lora_targets':model.lora_targets,
        'torch':torch.__version__,'test_labels_read':False,'claim':'method_adaptation_not_original_paper_SOTA_reproduction'},indent=2))
    print(json.dumps(run(frame,model,cfg,out,'cuda')),flush=True)


if __name__=='__main__':main()
