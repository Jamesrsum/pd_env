#!/usr/bin/env python3
"""Controlled AST warm-up/branches with shared initialization and phase labels."""
import argparse
import json
import os
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import torch,yaml
from src.training.controlled_adaptation import ControlledAdapter,run_stage,seed_all
from src.training.adaptation_runner import training_manifest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--stage',choices=['warmup','branch'],required=True);p.add_argument('--policy',choices=['frozen','last2','full'],default='frozen');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    os.umask(0o077);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
    cfg=yaml.safe_load((REPO/'configs/ast_controlled_finetuning.yaml').read_text());base=yaml.safe_load((REPO/'configs/frozen_benchmark_local.yaml').read_text())
    seed_all(cfg['seed']);frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'],smoke=a.smoke,per_cell=4)
    if a.smoke:cfg.update(warmup_epochs=2,adaptation_epochs=2,microbatch_size=2,effective_batch_size=8)
    model=ControlledAdapter(cfg['model'],os.environ['MPOWER_MODEL_ROOT'],base,cfg['head_kind'])
    cfg['manifest_sha256']=sha;cfg['asset_revision']=model.asset['revision']
    root=Path(os.environ['MPOWER_CONTROLLED_ROOT']).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    out=root/('smoke' if a.smoke else 'runs')/('warmup' if a.stage=='warmup' else a.policy)
    warm=out.parent/'warmup/best.pt' if a.stage=='branch' else None
    out.mkdir(parents=True,exist_ok=True)
    (out/'provenance.json').write_text(json.dumps({'config':cfg,'manifest_sha256':sha,'stage':a.stage,'policy':a.policy,
        'asset':model.asset,'framework':torch.__version__,'head':cfg['head_kind'],'test_labels_read':False},indent=2))
    print(json.dumps(run_stage(frame,model,a.policy,cfg,out,'cuda',a.stage,warm),indent=2))
if __name__=='__main__':main()
