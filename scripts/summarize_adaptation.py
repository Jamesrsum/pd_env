import json,os
from pathlib import Path
import pandas as pd
root=Path(os.environ['MPOWER_ADAPT_ROOT']);rows=[]
for directory in sorted((root/'runs').glob('*')):
    if (directory/'DONE.json').exists():
        d=json.loads((directory/'DONE.json').read_text());metrics=json.loads((directory/'validation_metrics.json').read_text())
        rows.append({'experiment':directory.name,'status':d['status'],**metrics})
    else:rows.append({'experiment':directory.name,'status':'incomplete'})
p=root/'alignment_runs/validation_leaderboard.csv'
if p.exists():rows.extend(pd.read_csv(p).to_dict('records'))
pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
(root/'SUMMARY.json').write_text(json.dumps({'supervised_complete':len(list((root/'runs').glob('*/DONE.json'))),'expected_supervised':14,
    'contrastive_complete':(root/'alignment_runs/DONE.json').exists(),'test_labels_read':False,'hear':json.loads((root/'assets.json').read_text())['hear']},indent=2))
print((root/'SUMMARY.json').read_text())
