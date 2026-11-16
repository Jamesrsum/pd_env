#!/usr/bin/env python3
"""Require matched starts, real adapter updates, and correct contrastive batch size."""
import argparse
import json
import os
from pathlib import Path


def audit(root,family):
    methods=(['hear_lp_frozen','hear_lp_full','hear_lp_lora'] if family=='hear'
             else ['hubert_ce','hubert_detached_supcon'])
    values=[]
    for method in methods:
        path=Path(root)/'smoke'/method;d=json.loads((path/'DONE.json').read_text())
        if d['status']!='complete' or d['test_labels_read'] or not d['frozen_parameters_unchanged'] or not d['head_changed']:raise ValueError('Invalid completion audit')
        expected=method!='hear_lp_frozen'
        if d['trainable_backbone_or_adapter_changed']!=expected:raise ValueError('Requested adaptation did not occur')
        if method=='hear_lp_lora' and (d['pretrained_weights_changed'] or d['trainable_backbone_parameters']!=786432):raise ValueError('LoRA trainability audit failed')
        if family=='speech':
            provenance=json.loads((path/'provenance.json').read_text())
            runtime=provenance['config']['actual_encoder_augmentation']
            if set(runtime)!={'mask_time_prob','mask_feature_prob','layerdrop'} or any(runtime.values()):raise ValueError('Actual encoder configuration differs from protocol')
            history=json.loads((path/'history.json').read_text())
            if any(row['actual_batch_min']!=32 or row['actual_batch_max']!=32 for row in history):raise ValueError('Not a real 32-person contrastive/control batch')
            if d['projector_changed']!=(method=='hubert_detached_supcon'):raise ValueError('Incorrect projector updates')
        values.append(d)
    if len({d['initial_base_sha256'] for d in values})!=1 or len({d['initial_head_sha256'] for d in values})!=1:
        raise ValueError('Research arms did not share their initial backbone/head')
    result={'status':'real_pretrained_smoke_passed','family':family,'methods':methods,'test_labels_read':False,
            'initialization_matched':True,'frozen_parameters_unchanged':True}
    dest=Path(root)/'smoke'/f'{family}_PASSED.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--family',choices=['hear','speech'],required=True);a=p.parse_args()
    print(json.dumps(audit(os.environ['MPOWER_LITERATURE_ROOT'],a.family)),flush=True)
