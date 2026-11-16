#!/usr/bin/env python3
"""Require preserved frozen features and matched, genuinely trained smoke arms."""
import json
import os
import argparse
from pathlib import Path


def audit(root):
    root=Path(root)/'smoke'
    gpu=json.loads((root/'GPU_VERIFIED.json').read_text())
    frozen=json.loads((root/'frozen/DONE.json').read_text())
    warm=json.loads((root/'warmup/DONE.json').read_text())
    if not gpu['batch_singleton_invariant'] or not gpu['state_reload_invariant']:raise ValueError('GPU invariance verification missing')
    if not frozen.get('encoder_unchanged') or not frozen.get('signature') or not frozen.get('manifest_sha256'):raise ValueError('Frozen feature provenance lost')
    if warm['backbone_changed'] or not warm['head_changed']:raise ValueError('Warm-up freeze failed')
    arms={p:json.loads((root/'adaptation'/p/'DONE.json').read_text()) for p in ['frozen','last2','full']}
    for policy,d in arms.items():
        if d['status']!='complete' or d['requested_policy']!=policy or d['selected_policy']!=policy:raise ValueError('Incorrect branch completion/selection')
        if d['backbone_changed']!=(policy!='frozen') or not d['head_changed'] or not d['frozen_parameters_unchanged']:raise ValueError('Parameter change audit failed')
        if d['test_labels_read']:raise ValueError('Test access forbidden')
    if len({d['initial_backbone_sha256'] for d in arms.values()})!=1:raise ValueError('Branch backbone initialization differs')
    if len({d['initial_head_sha256'] for d in arms.values()})!=1:raise ValueError('Branch head initialization differs')
    result={'status':'real_pretrained_smoke_passed','test_labels_read':False,'frozen_feature_provenance_preserved':True,
        'branch_initialization_matched':True,'all_parameter_change_audits_passed':True,
        'policies':list(arms),'train_participants':warm['participants_train'],'validation_participants':warm['participants_val']}
    (root/'SMOKE_PASSED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default=os.environ.get('MPOWER_HEAR_ROOT'))
    args=parser.parse_args()
    if not args.root:parser.error('Set MPOWER_HEAR_ROOT or pass --root')
    print(json.dumps(audit(args.root)),flush=True)
