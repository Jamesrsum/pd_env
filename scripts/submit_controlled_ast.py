#!/usr/bin/env python3
"""Snapshot and submit controlled AST validation plus paired one-seed experiment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1]


def main():
    os.umask(0o077)
    for env in ['MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_ADAPT_SNAPSHOT','MPOWER_MODEL_ROOT']:
        if not os.environ.get(env):raise ValueError(f'Set {env}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(os.environ['MPOWER_BENCHMARK_ROOT'])/'controlled_ast'/stamp;root.mkdir(parents=True,exist_ok=False)
    code=root/'code';code.mkdir();logs=root/'logs';logs.mkdir()
    for name in ['src','scripts','configs']:
        shutil.copytree(REPO/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env={**os.environ,'MPOWER_CONTROLLED_CODE':str(code),'MPOWER_CONTROLLED_ROOT':str(root)}
    receipt={'root':str(root),'protocol':'paired AST native-LN head, shared best warmup, independent head/backbone LR',
             'max_gpu_concurrency':2,'policies':['frozen','last2','full'],'test_labels_read':False}
    def save():(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def submit(template,extra=[]):
        r=subprocess.run(['sbatch','--parsable',*extra,'--output='+str(logs/(template+'_%A_%a.out')),
            '--error='+str(logs/(template+'_%A_%a.err')),str(code/'scripts'/f'{template}.slurm')],env=env,text=True,capture_output=True,check=True)
        return r.stdout.strip().split(';')[0]
    receipt['smoke_job']=submit('controlled_ast_smoke');save()
    receipt['warmup_job']=submit('controlled_ast_warmup',['--dependency=afterok:'+receipt['smoke_job'],'--kill-on-invalid-dep=yes']);save()
    receipt['branch_array_job']=submit('controlled_ast_branch',['--dependency=afterok:'+receipt['warmup_job'],'--kill-on-invalid-dep=yes','--array=0-2%2']);save()
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
