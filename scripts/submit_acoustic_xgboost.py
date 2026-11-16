#!/usr/bin/env python3
"""Snapshot a real GPU smoke job and dependent 60-trial acoustic tree search."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
REPO=Path(__file__).resolve().parents[1]


def main():
    os.umask(0o077)
    for key in ['MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_OUTPUT_ROOT','MPOWER_ADAPT_SNAPSHOT']:
        if not os.environ.get(key):raise ValueError(f'Set {key}')
    root=(Path(os.environ['MPOWER_BENCHMARK_ROOT'])/'acoustic_xgboost'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir();logs=root/'logs';logs.mkdir()
    for name in ['src','scripts','configs']:
        shutil.copytree(REPO/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env={**os.environ,'MPOWER_XGBOOST_CODE':str(code),'MPOWER_XGBOOST_ROOT':str(root)}
    receipt={'root':str(root),'test_labels_read':False,'trials':60,'cv_folds':3,'features':184,'max_gpu_concurrency':1}
    def save():(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def submit(extra,args):
        result=subprocess.run(['sbatch','--parsable',*extra,'--output='+str(logs/'xgboost_%j.out'),
            '--error='+str(logs/'xgboost_%j.err'),str(code/'scripts/acoustic_xgboost.slurm'),*args],env=env,text=True,capture_output=True,check=True)
        return result.stdout.strip().split(';')[0]
    receipt['smoke_job']=submit([],['--smoke']);save()
    receipt['search_job']=submit(['--dependency=afterok:'+receipt['smoke_job'],'--kill-on-invalid-dep=yes'],[]);save()
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
