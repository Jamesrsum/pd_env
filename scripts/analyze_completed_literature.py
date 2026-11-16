#!/usr/bin/env python3
"""Verify completed production predictions; publish aggregate metrics, never rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from src.training.adaptation_runner import training_manifest
from src.training.development import operating_point


def analyze(root, snapshot, output, bootstrap=2000):
    frame, sha = training_manifest(snapshot)
    val = frame.loc[frame.split.eq('val') & frame.initial_recording].sort_values('recording_id')
    if val.participant_id.duplicated().any() or val.duplicate_component.duplicated().any():
        raise ValueError('Bootstrap requires unique validation people/components')
    y = val.label.to_numpy()
    methods = ['hear_lp_frozen', 'hear_lp_full', 'hear_lp_lora', 'hubert_ce', 'hubert_detached_supcon']
    rows, histories, audits, provenance, probabilities = [], [], {}, {}, {}
    for method in methods:
        run = root / ('runs' if method.startswith('hear') else 'speech_retry_20261116T092846Z/runs') / method
        done = json.loads((run / 'DONE.json').read_text())
        source = json.loads((run / 'provenance.json').read_text())
        if done['status'] != 'complete' or done['test_labels_read'] or source['config']['manifest_sha256'] != sha:
            raise ValueError('Incomplete run or manifest mismatch')
        if not done['frozen_parameters_unchanged']:
            raise ValueError('Frozen parameter audit failed')
        d = pd.read_csv(run / 'validation_predictions.csv', dtype={'recording_id': str, 'participant_id': str})
        if d.recording_id.duplicated().any() or set(d.recording_id) != set(val.recording_id):
            raise ValueError('Prediction cohort mismatch')
        d = d.set_index('recording_id').loc[val.recording_id]
        if not np.array_equal(d.label, y) or not np.array_equal(d.participant_id, val.participant_id):
            raise ValueError('Prediction label or participant mismatch')
        p = d.probability.to_numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid probability')
        auc = float(roc_auc_score(y, p))
        if not np.isclose(auc, done['validation_metrics']['auroc'], atol=1e-12):
            raise ValueError('Saved metric disagrees with predictions')
        probabilities[method] = p
        rows.append({'method': method, 'auroc': auc, 'auprc': float(average_precision_score(y, p)),
                     'log_loss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)),
                     'participants': len(y), 'selected_epoch': done['selected_epoch'],
                     'selected_stage': done['selected_stage'], **operating_point(y, p)})
        history = json.loads((run / 'history.json').read_text())
        if len(history) != source['config']['epochs']:
            raise ValueError('Truncated production history')
        histories += [{'method': method, **row} for row in history]
        audits[method] = done
        # Only a fixed public configuration/asset whitelist; no private paths or IDs.
        provenance[method] = {k: source[k] for k in ['config', 'asset', 'lora_targets', 'torch', 'claim']}
    pairs = [('hear_lp_full', 'hear_lp_frozen'), ('hear_lp_lora', 'hear_lp_frozen'),
             ('hear_lp_full', 'hear_lp_lora'), ('hubert_detached_supcon', 'hubert_ce')]
    rng = np.random.default_rng(20261116)
    samples = [rng.integers(0, len(y), len(y)) for _ in range(bootstrap)]
    comparisons = {}
    for a, b in pairs:
        delta = [roc_auc_score(y[ix], probabilities[a][ix]) - roc_auc_score(y[ix], probabilities[b][ix])
                 for ix in samples if len(np.unique(y[ix])) == 2]
        comparisons[f'{a}_minus_{b}'] = {'auroc_delta': float(roc_auc_score(y, probabilities[a])-roc_auc_score(y, probabilities[b])),
                                         'paired_component_bootstrap_95pct': np.quantile(delta, [.025, .975]).tolist()}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(histories).to_csv(output / 'learning_history.csv', index=False)
    summary = {'status': 'all_five_production_arms_verified_complete', 'manifest_sha256': sha,
               'validation_participants': len(y), 'validation_pd': int(y.sum()), 'test_labels_read': False,
               'bootstrap_replicates': bootstrap, 'bootstrap_seed': 20261116,
               'paired_comparisons': comparisons, 'training_audits': audits,
               'limitations': ['One training seed; validation-selected epochs.',
                              'Intervals condition on fitted models; no model-selection or multiplicity correction.',
                              'Repeated validation reuse; final test sealed; no clinical or exact paper-SOTA claim.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(pd.DataFrame(rows)[['method', 'auroc', 'selected_epoch']].to_string(index=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/results/literature_completed')
    parser.add_argument('--bootstrap', type=int, default=2000)
    a = parser.parse_args()
    analyze(a.root, a.snapshot, a.output, a.bootstrap)
