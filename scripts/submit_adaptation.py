"""Freeze source and submit supervised smoke + dependent training through helper."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[1];root=Path(os.environ['MPOWER_ADAPT_ROOT']).resolve()
if repo==root or repo in root.parents:raise ValueError('Private run root required')
code=root/'code'
if code.exists():raise FileExistsError('Choose a new run root rather than change submitted code')
code.mkdir(parents=True)
for name in ['src','scripts','configs']:
    shutil.copytree(repo/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
env=dict(os.environ,MPOWER_CODE_ROOT=str(code))
def submit(template,**extra):
    result=subprocess.run(['bash',str(repo/'scripts/submit_mpower.sh'),template],env={**env,**{k:str(v) for k,v in extra.items()}},capture_output=True,text=True,check=True)
    value=result.stdout.strip().splitlines()[-1].split(';')[0]
    if not value.isdigit():raise ValueError(f'Invalid Slurm job response: {result.stdout}')
    return value
smoke=submit('adaptation_smoke')
training=submit('adaptation_train',MPOWER_SBATCH_ARRAY='0-13%2',MPOWER_SBATCH_DEPENDENCY=f'afterok:{smoke}')
summary=submit('adaptation_summary',MPOWER_SBATCH_DEPENDENCY=f'afterany:{training}')
result={'smoke_job':smoke,'supervised_array_job':training,'summary_job':summary,'supervised_tasks':14,'max_gpu_concurrency':2,
        'test_labels_read':False,'contrastive_status':'held_pending_research_reassessment','hear_status':'blocked_missing_weights',
        'code_hashes':{str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*.py')}}
(root/'submission.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='code_hashes'},indent=2))
