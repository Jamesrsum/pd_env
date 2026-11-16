#!/usr/bin/env python3
"""Snapshot code and submit a CPU smoke check followed by development-only heads."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1]


def main():
    os.umask(0o077)
    for key in ['MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_ADAPT_SNAPSHOT','MPOWER_FROZEN_ROOT']:
        if not os.environ.get(key):raise ValueError(f'Set {key}')
    root=Path(os.environ['MPOWER_BENCHMARK_ROOT'])/'downstream_panel'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=root.resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir();logs=root/'logs';logs.mkdir()
    for name in ['src','scripts','configs']:
        shutil.copytree(REPO/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env={**os.environ,'MPOWER_DOWNSTREAM_CODE':str(code),'MPOWER_DOWNSTREAM_ROOT':str(root)}
    receipt={'root':str(root),'test_labels_read':False,'cohort':'canonical initial visit','max_cpu_concurrency':1}
    def submit(extra,args):
        r=subprocess.run(['sbatch','--parsable',*extra,'--output='+str(logs/'panel_%j.out'),
            '--error='+str(logs/'panel_%j.err'),str(code/'scripts/downstream_panel.slurm'),*args],env=env,text=True,capture_output=True,check=True)
        return r.stdout.strip().split(';')[0]
    receipt['smoke_job']=submit([],['--smoke'])
    (root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    receipt['full_job']=submit(['--dependency=afterok:'+receipt['smoke_job'],'--kill-on-invalid-dep=yes'],[])
    (root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
