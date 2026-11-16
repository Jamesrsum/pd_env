#!/usr/bin/env python3
"""Snapshot bounded literature comparisons, with per-family real smoke gates."""
from datetime import datetime,timezone
import json
import os
import argparse
from pathlib import Path
import shutil
import subprocess
REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--speech-only-root',type=Path,help='Private parent for an isolated corrected speech retry')
    args=parser.parse_args()
    os.umask(0o077)
    for key in ['MPOWER_BENCHMARK_ROOT','MPOWER_BENCHMARK_PYTHON','MPOWER_ADAPT_SNAPSHOT',
                'MPOWER_MODEL_ROOT','MPOWER_HEAR_SOURCE','MPOWER_FROZEN_ROOT']:
        if not os.environ.get(key):raise ValueError(f'Set {key}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=(args.speech_only_root/('speech_retry_'+stamp) if args.speech_only_root else Path(os.environ['MPOWER_BENCHMARK_ROOT'])/'literature_methods'/stamp)
    root=root.resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Private outputs required')
    root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir();logs=root/'logs';logs.mkdir()
    for name in ['src','scripts','configs']:
        shutil.copytree(REPO/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env={**os.environ,'MPOWER_LITERATURE_CODE':str(code),'MPOWER_LITERATURE_ROOT':str(root)}
    receipt={'root':str(root),'max_gpu_concurrency':2,'test_labels_read':False,
        'methods':['hear_lp_frozen','hear_lp_full','hear_lp_lora','hubert_ce','hubert_detached_supcon','TabICLv1','TabICLv2'],
        'claim':'source_audited_method_adaptations_not_original_dataset_SOTA_reproductions'}
    def save():(root/'submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def submit(template,args,extra=[]):
        command=['sbatch','--parsable',*extra,'--output='+str(logs/'job_%A_%a.out'),'--error='+str(logs/'job_%A_%a.err'),str(code/'scripts'/template),*args]
        r=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
        return r.stdout.strip().split(';')[0]
    if args.speech_only_root:
        receipt['methods']=['hubert_ce','hubert_detached_supcon']
        receipt['speech_smoke_job']=submit('literature_gpu.slurm',['smoke','speech']);save()
        receipt['speech_array_job']=submit('literature_gpu.slurm',['array','speech'],[
            '--dependency=afterok:'+receipt['speech_smoke_job'],'--kill-on-invalid-dep=yes','--array=0-1%1']);save()
        print(json.dumps(receipt,indent=2));return
    receipt['hear_smoke_job']=submit('literature_gpu.slurm',['smoke','hear']);save()
    receipt['speech_smoke_job']=submit('literature_gpu.slurm',['smoke','speech']);save()
    receipt['tabular_smoke_job']=submit('literature_tabular_smoke.slurm',[]);save()
    # Tabular inference waits for both GPU smoke tests so it cannot cause a
    # third concurrent GPU allocation alongside the two family smoke jobs.
    deps=':'.join(receipt[k] for k in ['hear_smoke_job','speech_smoke_job','tabular_smoke_job'])
    receipt['tabular_job']=submit('literature_gpu.slurm',['tabular'],['--dependency=afterok:'+deps,'--kill-on-invalid-dep=yes']);save()
    receipt['speech_array_job']=submit('literature_gpu.slurm',['array','speech'],[
        '--dependency=afterok:'+receipt['speech_smoke_job']+':'+receipt['hear_smoke_job'],'--kill-on-invalid-dep=yes','--array=0-1%1']);save()
    receipt['hear_array_job']=submit('literature_gpu.slurm',['array','hear'],[
        '--dependency=afterok:'+receipt['hear_smoke_job']+',afterany:'+receipt['tabular_job'],'--kill-on-invalid-dep=yes','--array=0-2%1']);save()
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
