#!/usr/bin/env python3
"""Fail before submission without verified gated assets and an inspected block path."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from src.models.hear import HearAdapter


def main():
    os.umask(0o077)
    for key in ['MPOWER_MODEL_ROOT','MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_ADAPT_SNAPSHOT','MPOWER_HEAR_BLOCK_PATH']:
        if not os.environ.get(key):raise ValueError(f'Set {key}; inspect authorized model modules rather than guessing block paths')
    model=HearAdapter.from_local(os.environ['MPOWER_MODEL_ROOT'],REPO/'configs/pretrained_models.lock.json',os.environ['MPOWER_HEAR_BLOCK_PATH'])
    schedule=model.configure('last2');del model
    root=Path(os.environ['MPOWER_BENCHMARK_ROOT'])/'hear_protocol'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=root.resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private output required')
    root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir();logs=root/'logs';logs.mkdir()
    for name in ['src','scripts','configs']:
        shutil.copytree(REPO/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env={**os.environ,'MPOWER_HEAR_CODE':str(code),'MPOWER_HEAR_ROOT':str(root)}
    receipt={'root':str(root),'test_labels_read':False,'preflight_last2':schedule,'max_gpu_concurrency':2}
    def save():(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def submit(stage,dependency=None,policy=None):
        command=['sbatch','--parsable','--output='+str(logs/(stage+'_%j.out')),'--error='+str(logs/(stage+'_%j.err'))]
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command += [str(code/'scripts/hear_protocol.slurm'),stage]
        if policy:command+=['--policy',policy]
        result=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
        return result.stdout.strip().split(';')[0]
    receipt['smoke_job']=submit('smoke');save()
    receipt['frozen_job']=submit('extract',receipt['smoke_job']);save()
    receipt['warmup_job']=submit('warmup',receipt['frozen_job']);save()
    # Two concurrent GPU branches; full follows the frozen branch.
    receipt['frozen_branch_job']=submit('branch',receipt['warmup_job'],'frozen');save()
    receipt['last2_job']=submit('branch',receipt['warmup_job'],'last2');save()
    receipt['full_job']=submit('branch',receipt['frozen_branch_job'],'full');save()
    if os.environ.get('MPOWER_FROZEN_ROOT'):
        downstream_env={**env,'MPOWER_DOWNSTREAM_CODE':str(code),'MPOWER_DOWNSTREAM_ROOT':str(root/'downstream_panel')}
        command=['sbatch','--parsable','--dependency=afterok:'+receipt['frozen_job'],'--kill-on-invalid-dep=yes',
            '--output='+str(logs/'downstream_%j.out'),'--error='+str(logs/'downstream_%j.err'),
            str(code/'scripts/downstream_panel.slurm')]
        result=subprocess.run(command,env=downstream_env,text=True,capture_output=True,check=True)
        receipt['downstream_job']=result.stdout.strip().split(';')[0];save()
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
